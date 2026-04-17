"""
Spikformer预训练训练脚本
针对分钟级负荷数据的动态特征表示学习
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple
import logging
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.models.spikformer_pretrain import SpikformerPretrainModel
from src.data.load_renewable_dataset import TimeSeriesDataset, LoadRenewableDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RepresentationLearningLoss(nn.Module):
    """
    动态特征表示学习的多任务损失函数
    
    包含：
    1. 簇内聚集损失（使同一动态的样本接近）
    2. 簇间分离损失（使不同动态的样本远离）
    3. 平衡和正则化项
    """
    def __init__(self, temperature=0.07, lambda_sep=1.0, lambda_reg=0.01):
        super().__init__()
        self.temperature = temperature
        self.lambda_sep = lambda_sep
        self.lambda_reg = lambda_reg
    
    def forward(self, 
                embeddings: torch.Tensor,
                proximities: torch.Tensor,
                cluster_centers: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings: (batch, embedding_dim) 样本的动态特征嵌入
            proximities: (batch,) 与动态中心的接近度
            cluster_centers: (num_clusters, embedding_dim) 聚类中心
            
        Returns:
            包含各损失分量的字典
        """
        batch_size = embeddings.shape[0]
        
        # 1. 对比学习损失 - 簇内聚集
        # 使用温度缩放的余弦相似度
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature  # (batch, batch)
        
        # 对角线掩码（排除自身）
        mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
        
        # 正样本对（批次内的同类）
        # 简单策略：相邻的样本视为同类（时间序列连续性）
        pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
        for i in range(batch_size - 1):
            pos_mask[i, i + 1] = True
            pos_mask[i + 1, i] = True
        
        # 计算对比损失
        pos_sim = torch.exp(sim_matrix) * pos_mask
        neg_sim = torch.exp(sim_matrix) * ~mask
        
        contrastive_loss = -torch.log(
            pos_sim.sum(dim=1) / (neg_sim.sum(dim=1) + 1e-6) + 1e-6
        ).mean()
        
        # 2. 簇间分离损失 - 与聚类中心的距离
        center_dist = torch.cdist(embeddings, cluster_centers)  # (batch, num_clusters)
        
        # 最小化到最近中心的距离，最大化到其他中心的距离
        min_center_dist, min_center_idx = center_dist.min(dim=1)
        separation_loss = min_center_dist.mean()
        
        # 3. 接近度正则化 - 鼓励多样的接近度分布
        proximity_loss = -proximities.mean()  # 最大化平均接近度
        
        # 4. 嵌入正则化 - 防止嵌入退化
        embedding_norm = torch.norm(embeddings, dim=1).mean()
        regularization_loss = torch.abs(embedding_norm - 1.0)
        
        # 组合损失
        total_loss = (
            contrastive_loss + 
            self.lambda_sep * separation_loss + 
            proximity_loss + 
            self.lambda_reg * regularization_loss
        )
        
        return {
            'total': total_loss,
            'contrastive': contrastive_loss.detach(),
            'separation': separation_loss.detach(),
            'proximity': proximity_loss.detach(),
            'regularization': regularization_loss.detach(),
        }


class PretrainingTrainer:
    """预训练模型训练器"""
    
    def __init__(self,
                 model: SpikformerPretrainModel,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5):
        
        self.model = model.to(device)
        self.device = device
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100
        )
        
        # 损失函数
        self.loss_fn = RepresentationLearningLoss(
            temperature=0.07,
            lambda_sep=1.0,
            lambda_reg=0.01
        )
        
        # 训练状态
        self.epoch = 0
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_losses = {
            'total': 0.0,
            'contrastive': 0.0,
            'separation': 0.0,
            'proximity': 0.0,
            'regularization': 0.0,
        }
        
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"训练 Epoch {self.epoch + 1}")
        for batch in pbar:
            if isinstance(batch, tuple):
                x = batch[0]  # DataLoader返回(tensor,)或(x, info)
            elif isinstance(batch, list):
                x = batch[0] if len(batch) > 0 else batch
            else:
                x = batch
            
            # 确保是torch张量
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
            else:
                x = x.float()
            
            x = x.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(x)
            
            # 计算损失
            loss_dict = self.loss_fn(
                output['embedding'],
                output['proximity'],
                self.model.cluster_centers
            )
            
            loss = loss_dict['total']
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 累积损失
            for key in total_losses:
                total_losses[key] += loss_dict[key].item()
            
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'contrastive': loss_dict['contrastive'].item(),
                'separation': loss_dict['separation'].item(),
            })
        
        # 计算平均损失
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        self.train_losses.append(avg_losses['total'])
        
        return avg_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        
        total_losses = {
            'total': 0.0,
            'contrastive': 0.0,
            'separation': 0.0,
            'proximity': 0.0,
            'regularization': 0.0,
        }
        
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="验证")
            for batch in pbar:
                if isinstance(batch, tuple):
                    x = batch[0]  # DataLoader返回(tensor,)或(x, info)
                elif isinstance(batch, list):
                    x = batch[0] if len(batch) > 0 else batch
                else:
                    x = batch
                
                # 确保是torch张量
                if not isinstance(x, torch.Tensor):
                    x = torch.FloatTensor(x)
                else:
                    x = x.float()
                
                x = x.to(self.device)
                
                # 前向传播
                output = self.model(x)
                
                # 计算损失
                loss_dict = self.loss_fn(
                    output['embedding'],
                    output['proximity'],
                    self.model.cluster_centers
                )
                
                # 累积损失
                for key in total_losses:
                    total_losses[key] += loss_dict[key].item()
                
                num_batches += 1
        
        # 计算平均损失
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        self.val_losses.append(avg_losses['total'])
        
        return avg_losses
    
    def save_checkpoint(self, save_dir: str):
        """保存检查点"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        torch.save(checkpoint, save_path / f"checkpoint_epoch_{self.epoch}.pt")
        torch.save(self.model.state_dict(), save_path / "latest_model.pt")
        
        logger.info(f"保存检查点到 {save_path}")
    
    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int = 20,
            save_dir: str = "./checkpoints"):
        """训练完整的epoch循环"""
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # 训练
            train_losses = self.train_epoch(train_loader)
            
            # 验证
            val_losses = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step()
            
            # 日志
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_losses['total']:.4f} | "
                f"Val Loss: {val_losses['total']:.4f}"
            )
            
            # 早停
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0
                self.save_checkpoint(save_dir)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"早停在 epoch {epoch + 1}")
                    break
            
            # 定期保存
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(save_dir)
        
        logger.info("预训练完成")
        return self.train_losses, self.val_losses


def create_dataloaders(data_root: str,
                       zones: list = None,
                       seq_len: int = 1440,
                       stride: int = 360,
                       batch_size: int = 32,
                       train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        data_root: 数据根目录
        zones: 要使用的电网区域列表
        seq_len: 序列长度
        stride: 滑动窗口步长
        batch_size: 批次大小
        train_split: 训练集比例
        
    Returns:
        train_loader, val_loader
    """
    
    # 加载数据
    loader = LoadRenewableDataLoader(data_root)
    
    if zones is None:
        zones = list(loader.zones.keys())[:2]  # 默认使用前2个区域
    
    all_datasets = []
    for zone in zones:
        df = loader.load_zone(zone)
        dataset = TimeSeriesDataset(df, seq_len, stride, normalize='zscore')
        all_datasets.append(dataset)
    
    # 合并数据集
    all_samples = []
    for dataset in all_datasets:
        for i in range(len(dataset)):
            all_samples.append(dataset[i])
    
    # 转换为张量
    x_list = []
    for x, _ in all_samples:
        x_list.append(x)
    
    x_all = np.array(x_list)
    
    # 分割训练/验证集
    train_size = int(len(x_all) * train_split)
    x_train = x_all[:train_size]
    x_val = x_all[train_size:]
    
    # 创建数据加载器
    train_dataset = TensorDataset(torch.FloatTensor(x_train))
    val_dataset = TensorDataset(torch.FloatTensor(x_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    import sys
    
    # 配置
    data_root = "/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"使用设备: {device}")
    
    # 创建数据加载器（使用前2个区域作为演示）
    logger.info("创建数据加载器...")
    train_loader, val_loader = create_dataloaders(
        data_root,
        zones=['CAISO_zone_1_', 'CAISO_zone_2_'],
        seq_len=1440,
        batch_size=16,
    )
    
    # 获取输入维度
    sample_batch = next(iter(train_loader))
    if isinstance(sample_batch, tuple):
        sample_batch = sample_batch[0]
    input_dim = sample_batch.shape[-1]
    logger.info(f"输入维度: {input_dim}")
    
    # 创建模型
    logger.info("创建预训练模型...")
    model = SpikformerPretrainModel(
        input_dim=input_dim,
        hidden_dim=256,
        embedding_dim=128,
        num_encoder_layers=4,
    )
    
    # 创建训练器
    trainer = PretrainingTrainer(model, device=device, learning_rate=1e-3)
    
    # 开始训练
    logger.info("开始预训练...")
    train_losses, val_losses = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=10,
        save_dir="./checkpoints/spikformer_pretrain"
    )
    
    logger.info("预训练完成！")
