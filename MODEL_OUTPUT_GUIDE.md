# 模型输出和可视化指南 📊

## 1. 模型输出内容 🎯

### SpikformerPretrainModel 的输出

```python
output = model(x)  # x: (batch_size, seq_len, num_features)

# 返回值是一个字典，包含3个关键输出：
{
    'embedding': Tensor,    # (batch_size, embedding_dim)
    'proximity': Tensor,    # (batch_size,)
    'encoded': Tensor       # (batch_size, seq_len, hidden_dim)
}
```

### 输出详解

| 输出字段 | 形状 | 数据类型 | 含义 |
|---------|------|---------|------|
| **embedding** | (B, 64) | float32 | **动态特征表示** - 24小时负荷数据的128维压缩表示，捕捉电网的关键动态特征 |
| **proximity** | (B,) | float32 | **接近度评分** - [0-1] 范围，表示该样本与学习到的原型的相似度。高值=正常模式，低值=异常模式 |
| **encoded** | (B, T, 256) | float32 | **完整时间编码** - 每个时间步的隐藏状态(B=batch, T=1440分钟, 256=hidden_dim) |

### 输出用途

```python
# 场景1: 特征提取（用于下游任务）
embeddings = output['embedding']  # (32, 64)
# 用于: 聚类、异常检测、故障预测等

# 场景2: 异常检测
proximity = output['proximity']   # (32,)
is_anomaly = proximity < threshold  # 低接近度表示异常

# 场景3: 解释性分析
encoded = output['encoded']       # (32, 1440, 256)
# 用于: 时间步级别的特征分析、注意力可视化等
```

---

## 2. 可视化工具 🎨

我为你创建了两个强大的可视化脚本：

### A. 训练过程监控 (`visualize_training.py`)

**功能：** 绘制训练/验证损失曲线和改善趋势

**用法：**
```bash
# 使用默认检查点目录
python visualize_training.py

# 使用自定义检查点目录
python visualize_training.py --checkpoint-dir ./checkpoints/my_checkpoint
```

**输出：**
- `training_curves.png` - 损失曲线图表
- `training_report.json` - 训练统计数据

**生成的图表：**
1. **训练和验证损失** - 双曲线图，显示学习过程
2. **损失改善百分比** - 展示相对于初始损失的改善幅度

---

### B. 嵌入空间可视化 (`visualize_embeddings.py`)

**功能：** 将学到的特征表示可视化到2D空间，分析嵌入质量

**用法：**
```bash
# 使用默认配置
python visualize_embeddings.py

# 自定义聚类数和输出目录
python visualize_embeddings.py --n-clusters 15 --output-dir ./results
```

**输出：** （保存在 `visualization/` 目录）
- `embedding_tsne.png` - t-SNE降维可视化，按时间顺序着色
- `embedding_clustering.png` - K-means聚类结果（轮廓系数评估）
- `embedding_distribution.png` - 嵌入统计分析（4个小图）

**生成的图表：**

1. **t-SNE 可视化**
   - 显示所有样本在2D空间的分布
   - 颜色表示时间顺序（样本索引）
   - 相近的点表示相似的动态模式

2. **聚类可视化**
   - 左图：K-means聚类结果，红星表示聚类中心
   - 右图：各簇的样本数分布
   - 轮廓系数显示聚类质量（1=完美，-1=很差）

3. **分布分析** （4个小图）
   - 向量范数分布：显示特征缩放是否均衡
   - 各维度均值：检测维度是否都有效利用
   - 样本间距离：评估特征空间的覆盖范围
   - 统计汇总：数值统计信息

---

## 3. 使用示例 📝

### 完整工作流程

```bash
# 1. 启动训练
python train_pretrain.py \
    --zones CAISO_zone_1_ CAISO_zone_2_ \
    --batch-size 16 \
    --seq-len 720 \
    --num-epochs 50

# 2. 训练完成后，可视化训练过程
python visualize_training.py --checkpoint-dir ./checkpoints/spikformer_pretrain_low_mem

# 3. 可视化学到的特征表示
python visualize_embeddings.py --n-clusters 12 --output-dir ./visualization

# 4. 查看所有输出
ls -la ./visualization/
ls -la ./checkpoints/spikformer_pretrain_low_mem/
```

### Python API 使用

```python
import torch
from src.models.spikformer_pretrain import SpikformerPretrainModel
from src.training.pretrain_training import create_dataloaders

# 1. 加载模型和数据
model = SpikformerPretrainModel(input_dim=11, hidden_dim=128, embedding_dim=64)
checkpoint = torch.load('./checkpoints/spikformer_pretrain_low_mem/checkpoint_epoch_50.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.cuda()
model.eval()

_, val_loader = create_dataloaders(
    "/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable",
    zones=['CAISO_zone_1_'],
    batch_size=32
)

# 2. 提取嵌入和其他输出
with torch.no_grad():
    for batch in val_loader:
        x = batch[0].cuda()  # (32, 1440, 11)
        outputs = model(x)
        
        embeddings = outputs['embedding']  # (32, 64)
        proximity = outputs['proximity']   # (32,)
        encoded = outputs['encoded']       # (32, 1440, 256)
        
        # 异常检测示例
        anomaly_threshold = proximity.mean() - proximity.std()
        anomalies = proximity < anomaly_threshold
        print(f"异常样本数: {anomalies.sum()} / {len(anomalies)}")
        break

# 3. 特征提取用于下游任务
all_embeddings = []
for batch in val_loader:
    x = batch[0].cuda()
    outputs = model(x)
    all_embeddings.append(outputs['embedding'].cpu().numpy())

import numpy as np
embeddings_array = np.vstack(all_embeddings)  # (N, 64)

# 用于聚类
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(embeddings_array)
```

---

## 4. 解释性分析 🔍

### 什么是"好"的嵌入空间？

✅ **好的特征表示应该有：**
1. **聚类性** - 相似的动态模式聚集在一起（轮廓系数 > 0.4）
2. **分离性** - 不同类型的模式距离远（簇间距离大）
3. **稳定性** - 相近时间的样本特征相近（时间序列特性）
4. **多样性** - 充分利用所有64个维度（不是退化到低秩）

### 如何从可视化中读取信息

**从t-SNE图：**
- 如果看到明显的聚簇 → 特征学习有效 ✅
- 如果分布很均匀 → 特征可能退化 ⚠️
- 如果颜色（时间）有明显顺序 → 捕捉了时间动态 ✅

**从聚类图：**
- 轮廓系数 > 0.5 → 聚类质量优秀
- 轮廓系数 0.3-0.5 → 聚类质量中等
- 轮廓系数 < 0.3 → 需要调整模型或聚类参数

**从分布分析：**
- 向量范数的标准差小 → 特征缩放均衡 ✅
- 多个维度平均值为0 → 可能未充分激活
- 样本距离集中在小范围 → 可能需要增大embedding_dim

---

## 5. 快速故障排除 🔧

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 可视化脚本报错"需要scikit-learn" | sklearn未安装 | `pip install scikit-learn` |
| t-SNE计算很慢 | 样本数太多 | 减少样本数或增加perplexity参数 |
| 聚类结果不明显 | 特征学习不足 | 增加训练轮数或调整学习率 |
| embedding_distribution.py运行失败 | 数据加载问题 | 检查数据路径和批次大小 |
| 显存不足 | 模型/数据太大 | 使用`visualize_embeddings.py --help`查看参数优化选项 |

---

## 6. 下一步应用 🚀

学到的特征表示 (`embedding`) 可以用于：

### 1. **异常检测**
```python
# 使用proximity评分作为异常指标
normal_threshold = proximity.mean() - 2 * proximity.std()
anomaly_scores = 1 - proximity  # 反转：低接近度=高异常分数
```

### 2. **故障预测**
```python
# 将embedding作为特征输入到分类器
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(embeddings[:-1], labels[1:])  # 预测下一时刻是否故障
```

### 3. **负荷聚类和分析**
```python
# 自动发现不同的运行工况
clusters = KMeans(n_clusters=10).fit_predict(embeddings)
# 分析各工况的特性
```

### 4. **迁移学习**
```python
# 在新数据上微调
embeddings = model.encoder(new_data)  # 直接使用预训练编码器
# 只需要少量标注数据就能适应新任务
```

---

## 7. 文件说明 📁

```
electric-power-regulation/
├── visualize_training.py           # 训练过程可视化
├── visualize_embeddings.py         # 嵌入空间可视化
├── checkpoints/
│   └── spikformer_pretrain_low_mem/
│       ├── checkpoint_epoch_*.pt   # 检查点（包含模型和损失）
│       ├── training_curves.png     # 训练曲线图
│       └── training_report.json    # 训练报告
└── visualization/                  # 嵌入可视化输出
    ├── embedding_tsne.png
    ├── embedding_clustering.png
    └── embedding_distribution.png
```

---

## 总结

🎯 **核心要点：**
1. 模型输出3个关键量：嵌入(特征)、接近度(异常分数)、完整编码(过程)
2. 两个可视化脚本自动生成训练和特征分析图表
3. 嵌入向量(64维)可直接用于下游任务
4. 接近度评分可用于实时异常检测

🚀 **立即可做的事：**
```bash
# 训练后运行（取决于硬件）
python visualize_training.py
python visualize_embeddings.py

# 或直接在Python中使用模型输出
# 见上面的"Python API使用"部分
```
