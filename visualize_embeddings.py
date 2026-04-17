"""
嵌入空间可视化脚本 - 使用t-SNE和聚类可视化预训练模型的特征表示
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging

try:
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn未安装，部分功能不可用")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def extract_embeddings(model, data_loader, device='cuda'):
    """从模型中提取嵌入向量"""
    model.eval()
    all_embeddings = []
    
    logger.info("提取嵌入向量...")
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
            else:
                x = x.float()
            
            x = x.to(device)
            output = model(x)
            embeddings = output['embedding'].cpu().numpy()
            all_embeddings.append(embeddings)
            
            if (i + 1) % 100 == 0:
                logger.info(f"  已处理 {i + 1} 批次")
    
    return np.vstack(all_embeddings)


def visualize_embeddings_tsne(embeddings, output_path='embedding_tsne.png'):
    """使用t-SNE可视化嵌入空间"""
    if not SKLEARN_AVAILABLE:
        logger.error("需要scikit-learn来进行t-SNE可视化")
        return False
    
    logger.info("计算t-SNE投影...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=np.arange(len(embeddings_2d)), 
                         cmap='viridis', alpha=0.6, s=30)
    plt.colorbar(scatter, label='样本索引')
    plt.title('嵌入空间 t-SNE 可视化\n(颜色代表时间顺序)', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE 维度 1', fontsize=12)
    plt.ylabel('t-SNE 维度 2', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✅ t-SNE可视化已保存到: {output_path}")
    plt.close()
    
    return True


def visualize_clustering(embeddings, n_clusters=10, output_path='embedding_clustering.png'):
    """可视化聚类结果"""
    if not SKLEARN_AVAILABLE:
        logger.error("需要scikit-learn来进行聚类")
        return False
    
    logger.info(f"进行K-means聚类 (K={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # 计算轮廓系数
    silhouette = silhouette_score(embeddings, cluster_labels)
    logger.info(f"  轮廓系数: {silhouette:.4f}")
    
    # t-SNE投影
    logger.info("计算t-SNE投影...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图: 聚类标签
    scatter1 = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                              c=cluster_labels, cmap='tab10', alpha=0.6, s=30)
    axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                   c='red', s=300, alpha=0.8, marker='*', edgecolors='black', linewidth=2,
                   label='聚类中心')
    plt.colorbar(scatter1, ax=axes[0], label='簇标签')
    axes[0].set_title(f'K-Means聚类结果 (K={n_clusters})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('t-SNE 维度 1', fontsize=11)
    axes[0].set_ylabel('t-SNE 维度 2', fontsize=11)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 右图: 聚类分布直方图
    unique, counts = np.unique(cluster_labels, return_counts=True)
    axes[1].bar(unique, counts, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('簇标签', fontsize=11)
    axes[1].set_ylabel('样本数', fontsize=11)
    axes[1].set_title('各簇的样本分布', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'聚类质量评估 (轮廓系数: {silhouette:.4f})', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✅ 聚类可视化已保存到: {output_path}")
    plt.close()
    
    return True


def visualize_embedding_distribution(embeddings, output_path='embedding_distribution.png'):
    """可视化嵌入向量的统计分布"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 计算统计信息
    embedding_norms = np.linalg.norm(embeddings, axis=1)
    embedding_mean = embeddings.mean()
    embedding_std = embeddings.std()
    
    # 1. 嵌入范数分布
    axes[0, 0].hist(embedding_norms, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(embedding_norms.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {embedding_norms.mean():.3f}')
    axes[0, 0].set_xlabel('嵌入向量范数', fontsize=11)
    axes[0, 0].set_ylabel('频数', fontsize=11)
    axes[0, 0].set_title('嵌入向量范数分布', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 各维度的均值和标准差
    dimension_means = embeddings.mean(axis=0)
    dimension_stds = embeddings.std(axis=0)
    x = np.arange(min(20, len(dimension_means)))  # 只显示前20维
    axes[0, 1].bar(x, dimension_means[:20], alpha=0.7, label='均值', edgecolor='black')
    axes[0, 1].set_xlabel('维度', fontsize=11)
    axes[0, 1].set_ylabel('均值', fontsize=11)
    axes[0, 1].set_title('嵌入向量各维度的均值 (前20维)', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. 样本间距离分布（采样）
    sample_size = min(1000, len(embeddings))
    sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
    sample_embeddings = embeddings[sample_indices]
    
    distances = []
    for i in range(min(500, len(sample_embeddings))):
        for j in range(i + 1, min(i + 10, len(sample_embeddings))):
            dist = np.linalg.norm(sample_embeddings[i] - sample_embeddings[j])
            distances.append(dist)
    
    axes[1, 0].hist(distances, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(np.mean(distances), color='red', linestyle='--', linewidth=2, label=f'均值: {np.mean(distances):.3f}')
    axes[1, 0].set_xlabel('欧氏距离', fontsize=11)
    axes[1, 0].set_ylabel('频数', fontsize=11)
    axes[1, 0].set_title('样本间距离分布', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 统计信息文本
    axes[1, 1].axis('off')
    stats_text = f"""
    📊 嵌入空间统计信息
    
    样本数: {len(embeddings)}
    维度: {embeddings.shape[1]}
    
    向量范数:
      均值: {embedding_norms.mean():.4f}
      标准差: {embedding_norms.std():.4f}
      最小值: {embedding_norms.min():.4f}
      最大值: {embedding_norms.max():.4f}
    
    逐元素统计:
      均值: {embedding_mean:.4f}
      标准差: {embedding_std:.4f}
      最小值: {embeddings.min():.4f}
      最大值: {embeddings.max():.4f}
    
    样本间距离:
      均值: {np.mean(distances):.4f}
      标准差: {np.std(distances):.4f}
      最小值: {np.min(distances):.4f}
      最大值: {np.max(distances):.4f}
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                   verticalalignment='center', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5))
    
    plt.suptitle('嵌入空间分布分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✅ 分布可视化已保存到: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="嵌入空间可视化工具")
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints/spikformer_pretrain_low_mem',
        help='检查点目录'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=10,
        help='聚类数量'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./visualization',
        help='输出目录'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎨 开始嵌入空间可视化...")
    print()
    
    if SKLEARN_AVAILABLE:
        # 加载模型和数据加载器
        logger.info("加载预训练模型...")
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        
        from src.models.spikformer_pretrain import SpikformerPretrainModel
        from src.training.pretrain_training import create_dataloaders
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 加载检查点
        checkpoint_path = sorted(Path(args.checkpoint_dir).glob('checkpoint_epoch_*.pt'))[-1]
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 创建模型
        model = SpikformerPretrainModel(input_dim=11, hidden_dim=128, embedding_dim=64)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # 创建数据加载器
        _, val_loader = create_dataloaders(
            "/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable",
            zones=['CAISO_zone_1_'],
            batch_size=32,
        )
        
        # 提取嵌入
        embeddings = extract_embeddings(model, val_loader, device)
        
        # 可视化
        visualize_embeddings_tsne(embeddings, str(output_dir / 'embedding_tsne.png'))
        visualize_clustering(embeddings, args.n_clusters, str(output_dir / 'embedding_clustering.png'))
        visualize_embedding_distribution(embeddings, str(output_dir / 'embedding_distribution.png'))
        
        print("\n✅ 所有可视化已完成！")
        print(f"📁 输出目录: {output_dir}")
    else:
        print("❌ 请先安装scikit-learn: pip install scikit-learn")


if __name__ == "__main__":
    main()
