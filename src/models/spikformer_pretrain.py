"""
Spikformer预训练模型 - 专门用于动态特征表示学习
支持分钟级负荷和可再生能源时间序列的多任务学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class SpikeFunction(torch.autograd.Function):
    """
    脉冲激活函数的前向和后向传播
    使用替代梯度方法进行反向传播
    """
    @staticmethod
    def forward(ctx, x, threshold=0.5):
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        return (x > threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        threshold = ctx.threshold
        # 使用sigmoid作为替代梯度
        grad_input = grad_output * (1.0 / (1.0 + (x / threshold).abs()).pow(2))
        return grad_input, None


class SpikeNeuron(nn.Module):
    """脉冲神经元层"""
    def __init__(self, threshold=0.5, tau=0.5):
        super().__init__()
        self.threshold = threshold
        self.tau = tau  # 膜时间常数
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_dim) 输入张量
        Returns:
            spike: 脉冲输出
            membrane: 膜电势状态
        """
        # 膜电势动力学
        membrane = x
        spike = SpikeFunction.apply(x, self.threshold)
        return spike, membrane


class SpikingSelfAttention(nn.Module):
    """
    基于脉冲的自注意机制
    适配时间序列的长期依赖学习
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # 脉冲相关
        self.spike_neuron = SpikeNeuron()
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: 注意力掩码 (可选)
        """
        batch_size, seq_len, _ = x.shape
        
        # 线性投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑为多头
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力权重
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用脉冲激活
        attn_spike, _ = self.spike_neuron(scores)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_weights * attn_spike  # 通过脉冲门控
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力到值
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        
        # 最后的线性投影
        output = self.out_proj(context)
        
        return output


class SpikformerBlock(nn.Module):
    """
    Spikformer编码块
    结合自注意和前馈网络，集成脉冲神经元
    """
    def __init__(self, hidden_dim, num_heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        
        # 自注意层
        self.self_attn = SpikingSelfAttention(hidden_dim, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
        # 脉冲神经元
        self.spike_neuron = SpikeNeuron()
    
    def forward(self, x):
        # 自注意 + 残差
        attn_out = self.self_attn(self.attn_norm(x))
        x = x + attn_out
        
        # 前馈 + 残差
        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + ffn_out
        
        # 脉冲激活
        spike_out, _ = self.spike_neuron(x)
        x = spike_out + x * (1 - spike_out)  # 混合脉冲和连续
        
        return x


class TimeSeriesEncoder(nn.Module):
    """时间序列编码器 - 将原始序列映射到隐藏空间"""
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # 多层Spikformer块
        self.encoder_blocks = nn.ModuleList([
            SpikformerBlock(hidden_dim, num_heads=8, ffn_dim=2048)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) 原始时间序列
        Returns:
            out: (batch, seq_len, hidden_dim) 编码后的表示
        """
        # 投影到隐藏空间
        x = self.input_proj(x)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # 通过编码块
        for block in self.encoder_blocks:
            x = block(x)
        
        # 归一化
        x = self.norm(x)
        
        return x


class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, hidden_dim, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                            -(math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class RepresentationLearningHead(nn.Module):
    """
    多任务动态特征学习头
    学习序列的潜在动态表示
    """
    def __init__(self, hidden_dim, embedding_dim=256):
        super().__init__()
        
        # 全局平均池化 + 投影
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 动态特征提取（聚类的中心）
        self.dynamic_center = nn.Parameter(torch.randn(embedding_dim))
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_dim) 编码后的表示
        Returns:
            embedding: (batch, embedding_dim) 每个样本的动态特征表示
            proximity: (batch,) 与动态中心的接近度
        """
        # 全局池化到 (batch, hidden_dim)
        pooled = self.global_pool(x.transpose(1, 2)).squeeze(-1)
        
        # 投影到嵌入空间
        embedding = self.projection(pooled)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        # 计算与动态中心的接近度
        dynamic_center_norm = F.normalize(self.dynamic_center, p=2, dim=0)
        proximity = F.cosine_similarity(embedding, dynamic_center_norm.unsqueeze(0))
        
        return embedding, proximity


class SpikformerPretrainModel(nn.Module):
    """
    完整的Spikformer预训练模型
    针对动态特征表示学习的多任务学习
    """
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 embedding_dim: int = 256,
                 num_encoder_layers: int = 4,
                 num_heads: int = 8):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # 编码器
        self.encoder = TimeSeriesEncoder(input_dim, hidden_dim, num_encoder_layers)
        
        # 表示学习头
        self.repr_head = RepresentationLearningHead(hidden_dim, embedding_dim)
        
        # 辅助头（聚类中心初始化）
        self.cluster_centers = nn.Parameter(torch.randn(10, embedding_dim))  # 10个聚类中心
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim) 输入时间序列
            
        Returns:
            包含以下键的字典:
            - 'embedding': (batch, embedding_dim) 动态特征表示
            - 'proximity': (batch,) 与动态中心的接近度
            - 'encoded': (batch, seq_len, hidden_dim) 编码后的完整序列
        """
        # 编码
        encoded = self.encoder(x)
        
        # 表示学习
        embedding, proximity = self.repr_head(encoded)
        
        return {
            'embedding': embedding,
            'proximity': proximity,
            'encoded': encoded,
        }
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """获取输入序列的动态特征嵌入"""
        output = self.forward(x)
        return output['embedding']


if __name__ == "__main__":
    # 测试模型
    batch_size = 4
    seq_len = 1440
    input_dim = 10
    
    model = SpikformerPretrainModel(input_dim=input_dim, hidden_dim=256, embedding_dim=256)
    
    # 模拟输入
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 前向传播
    output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"嵌入形状: {output['embedding'].shape}")
    print(f"接近度形状: {output['proximity'].shape}")
    print(f"编码形状: {output['encoded'].shape}")
