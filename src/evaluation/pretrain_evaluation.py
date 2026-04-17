"""
预训练模型评估和可视化脚本
分析动态特征表示的质量
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List
import json

try:
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn未安装，跳过部分可视化功能")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib未安装，跳过绘图功能")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingEvaluator:
    """嵌入空间质量评估器"""
    
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray = None):
        """
        Args:
            embeddings: (N, embedding_dim) 样本的嵌入向量
            labels: (N,) 样本的标签（可选，用于有监督指标）
        """
        self.embeddings = embeddings
        self.labels = labels
        self.n_samples = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]
    
    def compute_clustering_metrics(self, n_clusters: int = 10) -> Dict[str, float]:
        """计算聚类指标"""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn未安装，无法计算聚类指标")
            return {}
        
        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.embeddings)
        
        # 轮廓系数（越接近1越好）
        silhouette = silhouette_score(self.embeddings, cluster_labels)
        
        # Davies-Bouldin指数（越小越好）
        davies_bouldin = davies_bouldin_score(self.embeddings, cluster_labels)
        
        return {
            'silhouette_score': float(silhouette),
            'davies_bouldin_score': float(davies_bouldin),
            'cluster_labels': cluster_labels,
        }
    
    def compute_diversity_metrics(self) -> Dict[str, float]:
        """计算嵌入空间的多样性指标"""
        
        # 计算对角线距离矩阵的均值
        distances = []
        for i in range(min(1000, self.n_samples)):  # 采样以加快计算
            for j in range(i + 1, min(1000, self.n_samples)):
                dist = np.linalg.norm(
                    self.embeddings[i] - self.embeddings[j]
                )
                distances.append(dist)
        
        distances = np.array(distances)
        
        return {
            'mean_distance': float(distances.mean()),
            'std_distance': float(distances.std()),
            'min_distance': float(distances.min()),
            'max_distance': float(distances.max()),
        }
    
    def compute_all_metrics(self, n_clusters: int = 10) -> Dict[str, float]:
        """计算所有评估指标"""
        metrics = {}
        
        # 多样性指标
        diversity = self.compute_diversity_metrics()
        metrics.update(diversity)
        
        # 聚类指标
        clustering = self.compute_clustering_metrics(n_clusters)
        if clustering:
            metrics.update({
                k: v for k, v in clustering.items()
                if not isinstance(v, np.ndarray)
            })
        
        return metrics
    
    def visualize_tsne(self, output_path: str = "./embeddings_tsne.png"):
        """使用t-SNE可视化嵌入空间"""
        if not SKLEARN_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            logger.warning("需要sklearn和matplotlib来可视化")
            return
        
        logger.info("计算t-SNE投影...")
        tsne = TSNE(n_components=2, random_state=42, n_iter=1000)
        embeddings_2d = tsne.fit_transform(self.embeddings)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=20)
        plt.title("嵌入空间 t-SNE 可视化")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        logger.info(f"保存可视化到 {output_path}")
        plt.close()
    
    def visualize_clustering(self, n_clusters: int = 10, 
                            output_path: str = "./embeddings_clustering.png"):
        """可视化聚类结果"""
        if not SKLEARN_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            logger.warning("需要sklearn和matplotlib来可视化")
            return
        
        logger.info("计算聚类...")
        clustering = self.compute_clustering_metrics(n_clusters)
        cluster_labels = clustering['cluster_labels']
        
        # 使用t-SNE投影
        logger.info("计算t-SNE投影...")
        tsne = TSNE(n_components=2, random_state=42, n_iter=1000)
        embeddings_2d = tsne.fit_transform(self.embeddings)
        
        # 绘图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=cluster_labels, cmap='tab10', alpha=0.6, s=20)
        plt.colorbar(scatter, label='簇标签')
        plt.title(f"聚类结果 (K={n_clusters})")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        logger.info(f"保存聚类可视化到 {output_path}")
        plt.close()


class ModelEvaluator:
    """完整的模型评估器"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def extract_embeddings(self, data_loader) -> np.ndarray:
        """从数据加载器提取所有样本的嵌入"""
        all_embeddings = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, tuple):
                    x, _ = batch
                else:
                    x = batch
                
                if isinstance(x, np.ndarray):
                    x = torch.FloatTensor(x)
                
                x = x.to(self.device)
                output = self.model(x)
                embeddings = output['embedding'].cpu().numpy()
                all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def evaluate(self, data_loader, output_dir: str = "./evaluation"):
        """完整的评估流程"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 提取嵌入
        logger.info("提取嵌入向量...")
        embeddings = self.extract_embeddings(data_loader)
        
        # 创建评估器
        evaluator = EmbeddingEvaluator(embeddings)
        
        # 计算指标
        logger.info("计算评估指标...")
        metrics = evaluator.compute_all_metrics(n_clusters=10)
        
        # 保存指标
        metrics_file = output_path / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"评估指标：")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
        
        # 可视化
        logger.info("生成可视化...")
        evaluator.visualize_tsne(str(output_path / "tsne.png"))
        evaluator.visualize_clustering(10, str(output_path / "clustering.png"))
        
        return metrics, embeddings


def run_full_evaluation(model_path: str, data_loader, output_dir: str = "./evaluation"):
    """运行完整的评估流程"""
    from src.models.spikformer_pretrain import SpikformerPretrainModel
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    
    # 重建模型（需要知道配置）
    # 从checkpoint的state_dict推断配置
    state_dict = checkpoint if isinstance(checkpoint, dict) and 'weight' in list(checkpoint.values())[0] else checkpoint['model_state_dict']
    
    model = SpikformerPretrainModel(
        input_dim=10,  # 需要根据实际数据修改
        hidden_dim=256,
        embedding_dim=128,
    )
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 评估
    evaluator = ModelEvaluator(model, device)
    metrics, embeddings = evaluator.evaluate(data_loader, output_dir)
    
    return metrics, embeddings


if __name__ == "__main__":
    # 示例：如果已有保存的模型和数据
    logger.info("评估脚本已就绪")
    logger.info("使用: python -m src.evaluation.pretrain_evaluation")
