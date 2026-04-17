"""
多模态自主调控预训练模型 - 第②步
融合多种数据模态（负荷、电压、频率、天气等）进行电网调控决策
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from .neuron_models import (
    LeakyIntegrateFire, StaticSynapse, DynamicSynapse
)


class MultimodalEmbedding(nn.Module):
    """
    多模态嵌入层 - 统一不同来源的数据到共同特征空间
    
    支持的模态：
        - 负荷数据：时间序列特征
        - 电压数据：动态稳定性特征
        - 频率数据：实时调控响应特征
        - 天气数据：预测性特征
        - 时间特征：周期性特征（小时、日、周）
    """
    
    def __init__(
        self,
        modalities: Dict[str, int],  # {modality_name: input_dim}
        hidden_dim: int = 64,
        embedding_dim: int = 32
    ):
        super().__init__()
        self.modalities = modalities
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # 为每个模态创建独立的编码器
        self.encoders = nn.ModuleDict()
        for modality_name, input_dim in modalities.items():
            encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim)
            )
            self.encoders[modality_name] = encoder
        
        # 模态融合门（学习每个模态的重要性）
        self.fusion_gate = nn.Sequential(
            nn.Linear(embedding_dim * len(modalities), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(modalities)),
            nn.Softmax(dim=1)
        )
    
    def forward(self, modalities_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            modalities_data: {modality_name: (batch_size, input_dim)}
            
        Returns:
            fused_embedding: (batch_size, embedding_dim) 融合后的嵌入
            modality_weights: (batch_size, num_modalities) 模态权重
        """
        embeddings = []
        for modality_name in self.modalities.keys():
            data = modalities_data[modality_name]
            emb = self.encoders[modality_name](data)
            embeddings.append(emb)
        
        # (batch_size, embedding_dim * num_modalities)
        concat_emb = torch.cat(embeddings, dim=1)
        
        # 计算模态权重
        modality_weights = self.fusion_gate(concat_emb)  # (batch_size, num_modalities)
        
        # 加权融合
        stacked_emb = torch.stack(embeddings, dim=1)  # (batch_size, num_modalities, embedding_dim)
        weights_expanded = modality_weights.unsqueeze(2)  # (batch_size, num_modalities, 1)
        fused_embedding = (stacked_emb * weights_expanded).sum(dim=1)  # (batch_size, embedding_dim)
        
        return fused_embedding, modality_weights


class SpikeFormerControlBlock(nn.Module):
    """
    脉冲Transformer控制块 - 结合脉冲神经网络和自注意力
    用于时间序列决策和因果关系学习
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 4,
        hidden_dim: int = 128,
        num_lif_neurons: int = 64
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # 自注意力层
        self.multi_head_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(feature_dim)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        self.ff_norm = nn.LayerNorm(feature_dim)
        
        # 脉冲神经元层（用于判决）
        self.lif_neurons = nn.ModuleList([
            LeakyIntegrateFire(feature_dim, tau=2.0)
            for _ in range(num_lif_neurons)
        ])
        
        # LIF输出聚合
        self.lif_aggregate = nn.Linear(num_lif_neurons, feature_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (batch_size, seq_len, feature_dim) 序列特征
            attention_mask: (seq_len, seq_len) 因果掩码
            
        Returns:
            output: (batch_size, seq_len, feature_dim) 输出特征
            spike_rates: 各LIF神经元的发火率
        """
        # 自注意力
        attn_out, _ = self.multi_head_attn(
            x, x, x, 
            attn_mask=attention_mask,
            average_attn_weights=False
        )
        x = self.attn_norm(x + attn_out)
        
        # 前馈网络
        ff_out = self.feed_forward(x)
        x = self.ff_norm(x + ff_out)
        
        # 脉冲神经元决策层
        batch_size, seq_len, feature_dim = x.shape
        spike_rates = []
        
        spike_outputs = []
        for i, lif in enumerate(self.lif_neurons):
            # 为每个LIF神经元处理序列
            neuron_output = []
            for t in range(seq_len):
                spike, _ = lif(x[:, t, :].unsqueeze(1))
                neuron_output.append(spike)
            
            spike_rate = torch.mean(torch.cat(neuron_output, dim=1), dim=1)
            spike_rates.append(spike_rate)
            spike_outputs.append(torch.cat(neuron_output, dim=1))
        
        # (batch_size, seq_len, num_lif_neurons)
        lif_output = torch.stack(spike_outputs, dim=2)
        lif_output = lif_output.reshape(batch_size, seq_len, -1)
        
        # 聚合LIF输出
        lif_output = self.lif_aggregate(lif_output)
        
        # 最终输出：结合注意力和脉冲输出
        output = x + 0.3 * lif_output
        
        return output, spike_rates


class MultimodalControlNetwork(nn.Module):
    """
    多模态自主控制网络 - 电网调控的核心模型
    
    架构：
        多模态嵌入 → 脉冲Transformer块(×N) → 调控决策头
    """
    
    def __init__(
        self,
        modalities: Dict[str, int],
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        num_blocks: int = 4,
        num_control_outputs: int = 5,  # 频率、电压、负荷、应急、预留
        seq_len: int = 720
    ):
        super().__init__()
        self.modalities = modalities
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        
        # 第1步：多模态嵌入
        self.multimodal_embedding = MultimodalEmbedding(
            modalities=modalities,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim
        )
        
        # 位置编码
        self.pos_encoding = self._create_positional_encoding(seq_len, embedding_dim)
        
        # 第2步：脉冲Transformer块
        self.spike_blocks = nn.ModuleList([
            SpikeFormerControlBlock(
                feature_dim=embedding_dim,
                num_heads=4,
                hidden_dim=hidden_dim,
                num_lif_neurons=32
            )
            for _ in range(num_blocks)
        ])
        
        # 第3步：调控决策头
        self.control_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_control_outputs)
        )
        
        # 置信度预测（模型自信程度）
        self.confidence_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def _create_positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """创建正弦位置编码"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, seq_len, d_model)
    
    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """创建因果掩码（只看过去）"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.masked_fill(mask, float('-inf'))
    
    def forward(
        self,
        modalities_data: Dict[str, torch.Tensor],
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            modalities_data: {modality_name: (batch_size, seq_len, input_dim) or (batch_size, input_dim)}
            return_attention: 是否返回注意力权重
            
        Returns:
            {
                'control_actions': (batch_size, num_control_outputs) 调控指令
                'confidence': (batch_size, 1) 模型置信度
                'modality_weights': (batch_size, num_modalities) 模态权重
                'spike_rates': 脉冲发火率列表
            }
        """
        batch_size = next(iter(modalities_data.values())).shape[0]
        
        # 处理输入维度
        processed_data = {}
        for modality_name, data in modalities_data.items():
            if data.dim() == 2:  # (batch_size, input_dim)
                # 扩展为序列：复制seq_len次
                processed_data[modality_name] = data.unsqueeze(1).expand(-1, self.seq_len, -1)
            else:  # (batch_size, seq_len, input_dim)
                processed_data[modality_name] = data
        
        # 多模态嵌入
        all_embeddings = []
        modality_weights_list = []
        
        for i in range(self.seq_len):
            # 提取时刻i的各模态数据
            t_data = {
                name: processed_data[name][:, i, :]
                for name in self.modalities.keys()
            }
            
            # 嵌入和融合
            t_emb, t_weights = self.multimodal_embedding(t_data)
            all_embeddings.append(t_emb)
            if i == 0:
                modality_weights = t_weights
        
        # (batch_size, seq_len, embedding_dim)
        embedded_seq = torch.stack(all_embeddings, dim=1)
        
        # 加位置编码
        pos_enc = self.pos_encoding[:, :self.seq_len, :].to(embedded_seq.device)
        embedded_seq = embedded_seq + pos_enc
        
        # 脉冲Transformer块
        causal_mask = self._create_causal_mask(self.seq_len).to(embedded_seq.device)
        all_spike_rates = []
        
        for block in self.spike_blocks:
            embedded_seq, spike_rates = block(embedded_seq, attention_mask=causal_mask)
            all_spike_rates.extend(spike_rates)
        
        # 取最后时刻的表示
        final_repr = embedded_seq[:, -1, :]  # (batch_size, embedding_dim)
        
        # 调控决策
        control_actions = self.control_head(final_repr)  # (batch_size, num_control_outputs)
        confidence = self.confidence_head(final_repr)  # (batch_size, 1)
        
        return {
            'control_actions': control_actions,
            'confidence': confidence.squeeze(-1),
            'modality_weights': modality_weights,
            'spike_rates': all_spike_rates,
            'final_representation': final_repr
        }


# ==================== 预训练目标 ====================

class ControlPretrainingLoss(nn.Module):
    """
    电网调控预训练的多任务损失函数
    
    包括：
        1. 重建损失：恢复缺失的模态
        2. 对比损失：时间序列的连贯性
        3. 调控一致性损失：确保决策的稳定性
        4. 不确定性正则化：鼓励模型给出有信心的决策
    """
    
    def __init__(
        self,
        weight_reconstruction: float = 0.3,
        weight_contrastive: float = 0.3,
        weight_consistency: float = 0.2,
        weight_uncertainty: float = 0.2
    ):
        super().__init__()
        self.weight_reconstruction = weight_reconstruction
        self.weight_contrastive = weight_contrastive
        self.weight_consistency = weight_consistency
        self.weight_uncertainty = weight_uncertainty
    
    def reconstruction_loss(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """MSE损失：重建缺失的模态数据"""
        return F.mse_loss(original, reconstructed)
    
    def contrastive_loss(
        self,
        embeddings: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        对比损失：相邻时间步的表示应相似
        """
        # (batch_size, seq_len, embedding_dim)
        batch_size, seq_len, _ = embeddings.shape
        
        loss = 0
        for t in range(seq_len - 1):
            # 当前时刻 vs 下一时刻
            curr = embeddings[:, t, :]  # (batch_size, embedding_dim)
            next = embeddings[:, t + 1, :]
            
            # 余弦相似度
            sim = F.cosine_similarity(curr, next, dim=1)  # (batch_size,)
            
            # 最小化两者的距离（让相邻时刻相似）
            loss += torch.mean(1 - sim)
        
        return loss / (seq_len - 1)
    
    def consistency_loss(
        self,
        control_actions: torch.Tensor
    ) -> torch.Tensor:
        """
        调控一致性损失：相邻时刻的调控指令不应频繁变化
        防止控制指令过度振荡
        """
        # control_actions: (batch_size, seq_len, num_control_outputs)
        if control_actions.dim() == 2:
            control_actions = control_actions.unsqueeze(1)
        
        batch_size, seq_len, num_outputs = control_actions.shape
        
        loss = 0
        for t in range(seq_len - 1):
            curr = control_actions[:, t, :]
            next = control_actions[:, t + 1, :]
            
            # L2距离：确保相邻时刻相似
            loss += torch.mean((curr - next) ** 2)
        
        return loss / max(seq_len - 1, 1)
    
    def uncertainty_regularization(
        self,
        confidence: torch.Tensor
    ) -> torch.Tensor:
        """
        不确定性正则化：鼓励模型给出有信心的决策
        """
        # confidence: (batch_size,)，值在[0,1]
        # 目标：maximize log(confidence) 或 minimize -log(confidence)
        return -torch.mean(torch.log(confidence + 1e-6))
    
    def forward(
        self,
        original_data: torch.Tensor,
        reconstructed_data: torch.Tensor,
        embeddings: torch.Tensor,
        control_actions: torch.Tensor,
        confidence: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算总损失
        """
        loss_recon = self.reconstruction_loss(original_data, reconstructed_data)
        loss_contrast = self.contrastive_loss(embeddings)
        loss_consist = self.consistency_loss(control_actions)
        loss_uncert = self.uncertainty_regularization(confidence)
        
        total_loss = (
            self.weight_reconstruction * loss_recon +
            self.weight_contrastive * loss_contrast +
            self.weight_consistency * loss_consist +
            self.weight_uncertainty * loss_uncert
        )
        
        return total_loss, {
            'reconstruction': loss_recon.item(),
            'contrastive': loss_contrast.item(),
            'consistency': loss_consist.item(),
            'uncertainty': loss_uncert.item()
        }
