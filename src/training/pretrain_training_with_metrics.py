"""
改进的预训练训练脚本 - 包含实时评估指标（AUC、Precision、Recall等）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
import logging
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from src.models.spikformer_pretrain import SpikformerPretrainModel
from src.data.load_renewable_dataset import TimeSeriesDataset, LoadRenewableDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MetricsTracker:
    """训练指标追踪器"""
    
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.val_auc = []
        self.val_precision = []
        self.val_recall = []
        self.val_f1 = []
        self.val_silhouette = []  # 聚类质量指标
        self.train_proximity = []  # 训练时的接近度
        self.val_proximity = []    # 验证时的接近度
    
    def add_train(self, loss, proximity):
        self.train_loss.append(loss)
        self.train_proximity.append(proximity)
    
    def add_val(self, loss, auc=None, precision=None, recall=None, f1=None, silhouette=None, proximity=None):
        self.val_loss.append(loss)
        if auc is not None:
            self.val_auc.append(auc)
        if precision is not None:
            self.val_precision.append(precision)
        if recall is not None:
            self.val_recall.append(recall)
        if f1 is not None:
            self.val_f1.append(f1)
        if silhouette is not None:
            self.val_silhouette.append(silhouette)
        if proximity is not None:
            self.val_proximity.append(proximity)
    
    def to_dict(self):
        """转换为字典"""
        return {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'val_auc': self.val_auc,
            'val_precision': self.val_precision,
            'val_recall': self.val_recall,
            'val_f1': self.val_f1,
            'val_silhouette': self.val_silhouette,
            'train_proximity': self.train_proximity,
            'val_proximity': self.val_proximity,
        }
    
    def plot(self, output_dir='./results'):
        """绘制所有指标"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 损失曲线
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(self.train_loss, label='训练损失', linewidth=2, marker='o', markersize=4)
        ax.plot(self.val_loss, label='验证损失', linewidth=2, marker='s', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('训练和验证损失', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'loss_curve.png', dpi=150)
        plt.close()
        logger.info(f"✅ 损失曲线已保存: {output_dir / 'loss_curve.png'}")
        
        # 2. 评估指标
        if self.val_auc or self.val_precision or self.val_recall or self.val_f1:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # AUC
            if self.val_auc:
                axes[0, 0].plot(self.val_auc, marker='o', linewidth=2, color='#1f77b4')
                axes[0, 0].set_title('AUC (Area Under Curve)', fontsize=12, fontweight='bold')
                axes[0, 0].set_ylabel('AUC Score', fontsize=11)
                axes[0, 0].set_ylim([0, 1.05])
                axes[0, 0].grid(True, alpha=0.3)
            else:
                axes[0, 0].text(0.5, 0.5, '未计算AUC', ha='center', va='center')
                axes[0, 0].set_title('AUC', fontsize=12, fontweight='bold')
            
            # Precision
            if self.val_precision:
                axes[0, 1].plot(self.val_precision, marker='s', linewidth=2, color='#ff7f0e')
                axes[0, 1].set_title('Precision (精确率)', fontsize=12, fontweight='bold')
                axes[0, 1].set_ylabel('Precision', fontsize=11)
                axes[0, 1].set_ylim([0, 1.05])
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, '未计算Precision', ha='center', va='center')
                axes[0, 1].set_title('Precision', fontsize=12, fontweight='bold')
            
            # Recall
            if self.val_recall:
                axes[1, 0].plot(self.val_recall, marker='^', linewidth=2, color='#2ca02c')
                axes[1, 0].set_title('Recall (召回率)', fontsize=12, fontweight='bold')
                axes[1, 0].set_ylabel('Recall', fontsize=11)
                axes[1, 0].set_ylim([0, 1.05])
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, '未计算Recall', ha='center', va='center')
                axes[1, 0].set_title('Recall', fontsize=12, fontweight='bold')
            
            # F1
            if self.val_f1:
                axes[1, 1].plot(self.val_f1, marker='d', linewidth=2, color='#d62728')
                axes[1, 1].set_title('F1-Score', fontsize=12, fontweight='bold')
                axes[1, 1].set_ylabel('F1 Score', fontsize=11)
                axes[1, 1].set_ylim([0, 1.05])
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, '未计算F1', ha='center', va='center')
                axes[1, 1].set_title('F1-Score', fontsize=12, fontweight='bold')
            
            for ax in axes.flat:
                ax.set_xlabel('Epoch', fontsize=11)
            
            plt.suptitle('评估指标曲线', fontsize=14, fontweight='bold', y=1.00)
            plt.tight_layout()
            plt.savefig(output_dir / 'metrics_curve.png', dpi=150)
            plt.close()
            logger.info(f"✅ 评估指标已保存: {output_dir / 'metrics_curve.png'}")
        
        # 3. 聚类质量和接近度
        if self.val_silhouette or self.val_proximity:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Silhouette Score
            if self.val_silhouette:
                axes[0].plot(self.val_silhouette, marker='o', linewidth=2, color='purple')
                axes[0].set_title('Silhouette Score (聚类质量)', fontsize=12, fontweight='bold')
                axes[0].set_ylabel('Silhouette Score', fontsize=11)
                axes[0].set_ylim([-1, 1])
                axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
                axes[0].grid(True, alpha=0.3)
            else:
                axes[0].text(0.5, 0.5, '未计算Silhouette', ha='center', va='center')
                axes[0].set_title('Silhouette Score', fontsize=12, fontweight='bold')
            
            # Proximity
            if self.train_proximity and self.val_proximity:
                axes[1].plot(self.train_proximity, label='训练接近度', marker='o', linewidth=2)
                axes[1].plot(self.val_proximity, label='验证接近度', marker='s', linewidth=2)
                axes[1].set_title('接近度 (Proximity)', fontsize=12, fontweight='bold')
                axes[1].set_ylabel('Proximity Score', fontsize=11)
                axes[1].legend(fontsize=10)
                axes[1].grid(True, alpha=0.3)
            else:
                axes[1].text(0.5, 0.5, '未计算接近度', ha='center', va='center')
                axes[1].set_title('接近度 (Proximity)', fontsize=12, fontweight='bold')
            
            axes[0].set_xlabel('Epoch', fontsize=11)
            axes[1].set_xlabel('Epoch', fontsize=11)
            
            plt.suptitle('聚类和异常检测指标', fontsize=14, fontweight='bold', y=1.00)
            plt.tight_layout()
            plt.savefig(output_dir / 'clustering_metrics.png', dpi=150)
            plt.close()
            logger.info(f"✅ 聚类指标已保存: {output_dir / 'clustering_metrics.png'}")


def compute_auc_from_proximity(proximities: np.ndarray, labels: np.ndarray = None) -> float:
    """
    从接近度计算AUC
    低接近度表示异常，高接近度表示正常
    """
    if labels is None:
        # 自动生成标签：使用接近度的阈值
        threshold = proximities.mean()
        labels = (proximities > threshold).astype(int)
    
    if SKLEARN_AVAILABLE and len(np.unique(labels)) > 1:
        try:
            auc = roc_auc_score(labels, proximities)
            return auc
        except:
            return None
    return None


def compute_metrics_from_clustering(embeddings: np.ndarray, n_clusters: int = 10) -> Dict[str, float]:
    """
    从聚类结果计算指标
    """
    metrics = {}
    
    if not SKLEARN_AVAILABLE:
        return metrics
    
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        silhouette = silhouette_score(embeddings, labels)
        metrics['silhouette'] = silhouette
        
        # 计算伪标签的精确率和召回率
        # 简单方式：将最大簇作为正类，其他为负类
        cluster_counts = np.bincount(labels)
        largest_cluster = np.argmax(cluster_counts)
        binary_labels = (labels == largest_cluster).astype(int)
        
        if np.sum(binary_labels) > 0 and np.sum(1 - binary_labels) > 0:
            precision = precision_score(binary_labels, binary_labels, zero_division=0)
            recall = recall_score(binary_labels, binary_labels, zero_division=0)
            f1 = f1_score(binary_labels, binary_labels, zero_division=0)
            
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
    
    except Exception as e:
        logger.warning(f"计算聚类指标时出错: {e}")
    
    return metrics


class TrainerWithMetrics:
    """包含评估指标的训练器"""
    
    def __init__(self, model, device='cuda', learning_rate=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.metrics_tracker = MetricsTracker()
    
    def train_epoch(self, train_loader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_proximity = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for batch in pbar:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
            else:
                x = x.float()
            
            x = x.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(x)
            
            # 简单的重建损失
            loss = self.loss_fn(output['encoded'], x.unsqueeze(-1).expand_as(output['encoded']))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_proximity += output['proximity'].mean().item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item():.4f})
        
        avg_loss = total_loss / num_batches
        avg_proximity = total_proximity / num_batches
        
        return avg_loss, avg_proximity
    
    def validate(self, val_loader) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0
        total_proximity = 0
        all_embeddings = []
        all_proximities = []
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validating', leave=False)
            for batch in pbar:
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch
                
                if not isinstance(x, torch.Tensor):
                    x = torch.FloatTensor(x)
                else:
                    x = x.float()
                
                x = x.to(self.device)
                
                output = self.model(x)
                
                loss = self.loss_fn(output['encoded'], x.unsqueeze(-1).expand_as(output['encoded']))
                
                total_loss += loss.item()
                total_proximity += output['proximity'].mean().item()
                
                all_embeddings.append(output['embedding'].cpu().numpy())
                all_proximities.append(output['proximity'].cpu().numpy())
                
                num_batches += 1
                pbar.set_postfix({'loss': loss.item():.4f})
        
        avg_loss = total_loss / num_batches
        avg_proximity = total_proximity / num_batches
        
        # 计算额外指标
        embeddings = np.vstack(all_embeddings)
        proximities = np.hstack(all_proximities)
        
        metrics = {
            'loss': avg_loss,
            'proximity': avg_proximity,
        }
        
        # 计算AUC
        auc = compute_auc_from_proximity(proximities)
        if auc is not None:
            metrics['auc'] = auc
        
        # 计算聚类指标
        clustering_metrics = compute_metrics_from_clustering(embeddings, n_clusters=10)
        metrics.update(clustering_metrics)
        
        return metrics
    
    def fit(self, train_loader, val_loader, num_epochs=10, checkpoint_dir='./checkpoints'):
        """完整训练循环"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"\n{'='*70}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*70}")
            
            # 训练
            train_loss, train_proximity = self.train_epoch(train_loader)
            self.metrics_tracker.add_train(train_loss, train_proximity)
            
            # 验证
            val_metrics = self.validate(val_loader)
            self.metrics_tracker.add_val(
                val_metrics['loss'],
                auc=val_metrics.get('auc'),
                precision=val_metrics.get('precision'),
                recall=val_metrics.get('recall'),
                f1=val_metrics.get('f1'),
                silhouette=val_metrics.get('silhouette'),
                proximity=val_metrics.get('proximity')
            )
            
            # 打印指标
            logger.info(f"训练损失: {train_loss:.4f}, 训练接近度: {train_proximity:.4f}")
            logger.info(f"验证损失: {val_metrics['loss']:.4f}, 验证接近度: {val_metrics.get('proximity', 0):.4f}")
            
            if 'auc' in val_metrics:
                logger.info(f"AUC: {val_metrics['auc']:.4f}")
            if 'precision' in val_metrics:
                logger.info(f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics.get('recall', 0):.4f}, F1: {val_metrics.get('f1', 0):.4f}")
            if 'silhouette' in val_metrics:
                logger.info(f"Silhouette Score: {val_metrics['silhouette']:.4f}")
            
            # 保存检查点
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                checkpoint_path = checkpoint_dir / f'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_metrics['loss'],
                    'metrics': self.metrics_tracker.to_dict(),
                }, checkpoint_path)
                logger.info(f"✅ 最佳模型已保存: {checkpoint_path}")
            
            # 定期保存检查点
            if (epoch + 1) % 5 == 0:
                checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_metrics['loss'],
                    'metrics': self.metrics_tracker.to_dict(),
                }, checkpoint_path)
                logger.info(f"✅ 检查点已保存: {checkpoint_path}")
        
        # 保存指标和绘图
        metrics_json = checkpoint_dir / 'metrics.json'
        with open(metrics_json, 'w') as f:
            json.dump(self.metrics_tracker.to_dict(), f, indent=2)
        logger.info(f"✅ 指标已保存: {metrics_json}")
        
        # 绘制图表
        results_dir = Path('./results')
        self.metrics_tracker.plot(results_dir)
        
        return self.metrics_tracker
