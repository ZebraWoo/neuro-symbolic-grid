#!/usr/bin/env python
"""
快速演示：模型输出和应用示例
"""

import torch
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.models.spikformer_pretrain import SpikformerPretrainModel
from src.training.pretrain_training import create_dataloaders


def demo_model_outputs():
    """演示：查看模型的实际输出"""
    print("=" * 70)
    print("🎯 演示1：模型输出内容和形状")
    print("=" * 70)
    print()
    
    # 1. 创建模型
    model = SpikformerPretrainModel(
        input_dim=11,           # 11个特征（电网数据）
        hidden_dim=128,         # 隐藏层维度
        embedding_dim=64        # 最终嵌入维度
    )
    
    # 2. 创建虚拟输入
    batch_size = 4
    seq_len = 720
    x = torch.randn(batch_size, seq_len, 11)
    
    print(f"📥 输入张量形状: {x.shape}")
    print(f"   解释: (batch_size={batch_size}, seq_len={seq_len}, features=11)")
    print()
    
    # 3. 前向传播
    with torch.no_grad():
        output = model(x)
    
    print(f"📤 模型输出是一个字典，包含以下键值对：")
    print()
    
    for key, value in output.items():
        print(f"  '{key}':")
        print(f"    形状: {value.shape}")
        print(f"    数据类型: {value.dtype}")
        
        if key == 'embedding':
            print(f"    含义: 动态特征表示向量（每个样本的特征压缩表示）")
            print(f"    范围: [{value.min():.4f}, {value.max():.4f}]")
            print(f"    均值: {value.mean():.4f}, 标准差: {value.std():.4f}")
            
        elif key == 'proximity':
            print(f"    含义: 接近度评分（0-1），表示与原型的相似度")
            print(f"    范围: [{value.min():.4f}, {value.max():.4f}]")
            print(f"    均值: {value.mean():.4f}")
            
        elif key == 'encoded':
            print(f"    含义: 完整的时间编码（每个时间步的隐藏表示）")
            print(f"    范围: [{value.min():.4f}, {value.max():.4f}]")
        
        print()


def demo_anomaly_detection():
    """演示2：使用proximity进行异常检测"""
    print("=" * 70)
    print("🎯 演示2：异常检测")
    print("=" * 70)
    print()
    
    # 创建模型
    model = SpikformerPretrainModel(input_dim=11, hidden_dim=128, embedding_dim=64)
    
    # 模拟批次数据
    batch_size = 32
    x = torch.randn(batch_size, 720, 11)
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    proximity = output['proximity'].numpy()
    
    # 计算异常阈值
    mean_proximity = proximity.mean()
    std_proximity = proximity.std()
    normal_threshold = mean_proximity - 1.5 * std_proximity
    
    print(f"接近度统计:")
    print(f"  均值: {mean_proximity:.4f}")
    print(f"  标准差: {std_proximity:.4f}")
    print(f"  最小值: {proximity.min():.4f}")
    print(f"  最大值: {proximity.max():.4f}")
    print()
    
    # 检测异常
    is_anomaly = proximity < normal_threshold
    num_anomalies = is_anomaly.sum()
    
    print(f"异常检测结果:")
    print(f"  阈值: {normal_threshold:.4f}")
    print(f"  异常样本数: {num_anomalies} / {batch_size}")
    print(f"  异常率: {num_anomalies / batch_size * 100:.2f}%")
    print()
    
    print(f"异常样本详情:")
    for i, (is_anom, prox) in enumerate(zip(is_anomaly, proximity)):
        if is_anom:
            print(f"  样本 {i}: 接近度={prox:.4f} ⚠️ 异常")
    print()


def demo_feature_extraction():
    """演示3：特征提取用于聚类"""
    print("=" * 70)
    print("🎯 演示3：特征提取和聚类")
    print("=" * 70)
    print()
    
    # 创建模型
    model = SpikformerPretrainModel(input_dim=11, hidden_dim=128, embedding_dim=64)
    
    # 模拟批次数据
    batch_size = 64
    x = torch.randn(batch_size, 720, 11)
    
    # 前向传播获取嵌入
    with torch.no_grad():
        output = model(x)
    
    embeddings = output['embedding'].numpy()  # (64, 64)
    
    print(f"提取的嵌入向量:")
    print(f"  形状: {embeddings.shape}")
    print(f"  解释: {embeddings.shape[0]}个样本，每个样本{embeddings.shape[1]}维特征")
    print()
    
    # 尝试聚类（需要sklearn）
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        print(f"对嵌入向量进行K-means聚类...")
        
        # 不同K值的聚类效果
        for n_clusters in [3, 5, 8, 10]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            silhouette = silhouette_score(embeddings, labels)
            
            print(f"  K={n_clusters:2d}: 轮廓系数={silhouette:.4f}, " + 
                  f"各簇大小={np.bincount(labels).tolist()}")
        
        print()
        print("💡 提示:")
        print("  - 轮廓系数越接近1越好（完美聚类）")
        print("  - 轮廓系数 > 0.5 表示质量良好")
        print("  - 轮廓系数 < 0 表示样本被分配到错误的簇")
        
    except ImportError:
        print("⚠️ 需要安装sklearn来进行聚类")
        print("  运行: pip install scikit-learn")
    
    print()


def demo_embedding_properties():
    """演示4：嵌入向量的性质分析"""
    print("=" * 70)
    print("🎯 演示4：嵌入向量性质分析")
    print("=" * 70)
    print()
    
    # 创建模型
    model = SpikformerPretrainModel(input_dim=11, hidden_dim=128, embedding_dim=64)
    
    # 生成多批数据
    all_embeddings = []
    for _ in range(5):
        x = torch.randn(32, 720, 11)
        with torch.no_grad():
            output = model(x)
        all_embeddings.append(output['embedding'].numpy())
    
    embeddings = np.vstack(all_embeddings)  # (160, 64)
    
    print(f"收集了 {embeddings.shape[0]} 个样本的嵌入向量")
    print()
    
    # 分析1：向量范数分布
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"向量范数分析:")
    print(f"  均值: {norms.mean():.4f}")
    print(f"  标准差: {norms.std():.4f}")
    print(f"  范围: [{norms.min():.4f}, {norms.max():.4f}]")
    print()
    
    # 分析2：维度利用情况
    dim_means = embeddings.mean(axis=0)
    dim_stds = embeddings.std(axis=0)
    print(f"维度利用情况（前10维）:")
    print(f"  维度    平均值      标准差")
    for i in range(min(10, embeddings.shape[1])):
        print(f"  {i:2d}    {dim_means[i]:8.4f}   {dim_stds[i]:8.4f}")
    print()
    
    # 分析3：样本间相似性
    from scipy.spatial.distance import pdist, squareform
    distances = pdist(embeddings[:50], metric='euclidean')  # 只用前50个
    
    print(f"样本间距离分析（前50个样本）:")
    print(f"  均值: {distances.mean():.4f}")
    print(f"  标准差: {distances.std():.4f}")
    print(f"  范围: [{distances.min():.4f}, {distances.max():.4f}]")
    print()
    
    # 分析4：相似度矩阵
    sim_matrix = 1 / (1 + squareform(distances))  # 转换为相似度
    print(f"相似度统计：")
    print(f"  平均相似度: {sim_matrix.mean():.4f}")
    print(f"  (相似度范围: [0, 1], 1表示完全相同)")
    print()


def main():
    """运行所有演示"""
    print("\n")
    print("🚀 Spikformer 模型输出演示程序")
    print("=" * 70)
    print()
    
    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print()
    
    # 运行演示
    demo_model_outputs()
    print()
    
    demo_anomaly_detection()
    print()
    
    demo_feature_extraction()
    print()
    
    demo_embedding_properties()
    print()
    
    print("=" * 70)
    print("✅ 演示完成！")
    print()
    print("📚 更多信息，请查看: MODEL_OUTPUT_GUIDE.md")
    print()


if __name__ == "__main__":
    main()
