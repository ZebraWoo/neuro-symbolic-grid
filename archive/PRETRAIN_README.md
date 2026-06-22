# Spikformer预训练模型 - 电网动态特征表示学习

## 项目概述

本项目为电网时间序列数据构建Spikformer脉冲神经网络预训练模型，专注于**分钟级负荷和可再生能源数据**的**动态特征表示学习**。

### 核心创新
- 🧠 **脉冲神经网络(SNN)** 自注意机制：实现能效与精度的平衡
- 📊 **多源数据融合**：CAISO、ERCOT、MISO等电网区域的统一建模
- 🎯 **表示学习优化**：通过对比学习和聚类目标学习电网动态特征
- ⚡ **长期依赖捕捉**：通过Transformer架构处理24小时负荷变化

## 数据集

### PSML数据来源
```
/home/wuzuoxu/Data/PSML/
├── Millisecond-level PMU Measurements/     (毫秒级电压/功率)
│   ├── Natural Oscillation/                (自然振荡)
│   └── Forced Oscillation/                 (强制振荡)
├── Minute-level Load and Renewable/        ✅ (本项目主要使用)
│   ├── CAISO_zone_*.csv                    (4个区域)
│   ├── ERCOT_zone_*.csv                    (8个区域)
│   └── MISO_zone_*.csv                     (6个区域)
└── Minute-level PMU Measurements/          (分钟级电压/功率)
```

### 数据特性
- **时间分辨率**：分钟级（每条记录代表1分钟的统计数据）
- **时间跨度**：多年历史数据
- **特征数量**：根据区域而异（通常10-20个特征）
- **特征类型**：负荷功率、风电、太阳能、天气等

## 项目结构

```
electric-power-regulation/
├── train_pretrain.py                       # 🚀 快速启动脚本
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── load_renewable_dataset.py      # 数据加载和预处理
│   ├── models/
│   │   └── spikformer_pretrain.py         # Spikformer预训练模型
│   ├── training/
│   │   └── pretrain_training.py           # 训练循环和损失函数
│   └── evaluation/
│       ├── __init__.py
│       └── pretrain_evaluation.py         # 评估和可视化
├── checkpoints/                            # 模型检查点
├── README.md                               # 本文档
└── requirements.txt                        # 依赖包
```

## 快速开始

### 1. 安装依赖

```bash
# 基础依赖
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn matplotlib seaborn
pip install tqdm

# 可选（用于更好的性能）
pip install tensorboard wandb
```

### 2. 启动预训练

#### 最小配置（快速测试）
```bash
cd /home/wuzuoxu/electric-power-regulation

python train_pretrain.py \
    --zones CAISO_zone_1_ CAISO_zone_2_ \
    --batch-size 16 \
    --num-epochs 5 \
    --hidden-dim 128
```

#### 完整配置（推荐）
```bash
python train_pretrain.py \
    --zones CAISO_zone_1_ CAISO_zone_2_ CAISO_zone_3_ CAISO_zone_4_ \
    --batch-size 32 \
    --seq-len 1440 \
    --hidden-dim 256 \
    --embedding-dim 128 \
    --learning-rate 1e-3 \
    --num-epochs 20 \
    --checkpoint-dir ./checkpoints/spikformer_pretrain_full \
    --device cuda
```

#### 使用所有区域（完整训练）
```bash
python train_pretrain.py \
    --zones CAISO_zone_1_ CAISO_zone_2_ CAISO_zone_3_ CAISO_zone_4_ \
           ERCOT_zone_1_ ERCOT_zone_2_ ERCOT_zone_3_ ERCOT_zone_4_ \
           ERCOT_zone_5_ ERCOT_zone_6_ ERCOT_zone_7_ ERCOT_zone_8_ \
           MISO_zone_1_ MISO_zone_2_ MISO_zone_3_ MISO_zone_4_ \
           MISO_zone_5_ MISO_zone_6_ \
    --batch-size 64 \
    --num-epochs 50 \
    --checkpoint-dir ./checkpoints/spikformer_pretrain_all_zones
```

## 核心模块说明

### 数据加载（src/data/load_renewable_dataset.py）

```python
from src.data.load_renewable_dataset import LoadRenewableDataLoader, TimeSeriesDataset

# 加载特定区域数据
loader = LoadRenewableDataLoader("/path/to/PSML/Minute-level Load and Renewable")
df = loader.load_zone('CAISO_zone_1_')

# 创建时间序列数据集
dataset = TimeSeriesDataset(
    df,
    seq_len=1440,          # 24小时序列
    stride=360,            # 6小时步长
    normalize='zscore'     # Z-score标准化
)
```

**主要类和函数：**
- `LoadRenewableDataLoader`: 管理多个区域的数据加载
- `TimeSeriesDataset`: 将长序列转换为固定长度样本
- `create_pretrain_dataloader`: 便捷函数创建PyTorch数据加载器

### 预训练模型（src/models/spikformer_pretrain.py）

```python
from src.models.spikformer_pretrain import SpikformerPretrainModel

# 创建模型
model = SpikformerPretrainModel(
    input_dim=10,           # 输入特征数
    hidden_dim=256,         # 隐藏层维度
    embedding_dim=128,      # 嵌入维度
    num_encoder_layers=4,   # Spikformer编码块数
)

# 前向传播
output = model(x)  # x: (batch, seq_len, input_dim)
# output['embedding']: (batch, embedding_dim) - 动态特征表示
# output['proximity']: (batch,) - 与动态中心的接近度
# output['encoded']: (batch, seq_len, hidden_dim) - 完整编码
```

**核心组件：**
- `SpikeNeuron`: 脉冲激活层
- `SpikingSelfAttention`: 脉冲自注意机制
- `SpikformerBlock`: 完整的Spikformer编码块
- `TimeSeriesEncoder`: 输入投影和位置编码
- `RepresentationLearningHead`: 动态特征提取头

### 训练流程（src/training/pretrain_training.py）

```python
from src.training.pretrain_training import PretrainingTrainer

# 创建训练器
trainer = PretrainingTrainer(
    model,
    device='cuda',
    learning_rate=1e-3,
)

# 训练
train_losses, val_losses = trainer.fit(
    train_loader,
    val_loader,
    num_epochs=20,
)
```

**损失函数成分：**
1. **对比学习损失** (Contrastive Loss): 使时间序列相邻样本接近
2. **簇间分离损失** (Separation Loss): 与聚类中心的距离最小化
3. **接近度正则化** (Proximity Loss): 鼓励多样化的动态特征
4. **嵌入正则化** (Regularization): 防止嵌入退化

### 评估和可视化（src/evaluation/pretrain_evaluation.py）

```python
from src.evaluation.pretrain_evaluation import ModelEvaluator

# 创建评估器
evaluator = ModelEvaluator(model, device='cuda')

# 评估并可视化
metrics, embeddings = evaluator.evaluate(
    val_loader,
    output_dir='./evaluation_results'
)
```

**评估指标：**
- Silhouette Score (轮廓系数): 聚类紧凑度
- Davies-Bouldin Index: 簇间距离度量
- 嵌入空间多样性: 样本间平均距离、标准差等

## 训练说明

### 模型配置建议

| 场景 | Hidden Dim | Embedding Dim | Batch Size | LR | Epochs |
|------|-----------|---------------|-----------|-----|--------|
| 快速测试 | 128 | 64 | 16 | 1e-3 | 5 |
| 单区域 | 256 | 128 | 32 | 1e-3 | 20 |
| 多区域 | 512 | 256 | 64 | 5e-4 | 50 |
| 全部区域 | 768 | 512 | 128 | 1e-4 | 100+ |

### 训练监控

训练过程中会输出：
```
训练 Epoch 1
  loss: 2.3456
  contrastive: 1.2345
  separation: 0.8901
  
验证
Epoch 1/20 | Train Loss: 2.3456 | Val Loss: 2.1234
```

保存的模型：
- `latest_model.pt`: 最新模型权重
- `checkpoint_epoch_N.pt`: 第N个epoch的完整检查点

### 数据处理细节

**序列长度选择：**
- `seq_len=1440`: 24小时（推荐，捕捉日周期性）
- `seq_len=10080`: 7天（学习周周期性）
- `seq_len=43200`: 30天（学习月周期性）

**步长（stride）选择：**
- 较小的步长（如60分钟）: 更多样本，训练时间长
- 较大的步长（如360分钟）: 样本少但信息差异大（推荐）

**归一化方法：**
- `zscore`: 零均值单位方差（推荐用于神经网络）
- `minmax`: 缩放到[0,1]范围
- `none`: 不归一化

## 模型下游应用示例

### 1. 故障检测（异常检测）

```python
import torch
from src.models.spikformer_pretrain import SpikformerPretrainModel

# 加载预训练模型
model = SpikformerPretrainModel(...)
checkpoint = torch.load('./checkpoints/spikformer_pretrain/latest_model.pt')
model.load_state_dict(checkpoint)

# 提取特征用于异常检测
embeddings = model.get_embeddings(normal_data)
threshold = embeddings.std() + 3 * embeddings.mean()

# 检测异常
anomaly_embeddings = model.get_embeddings(test_data)
anomaly_scores = (anomaly_embeddings - embeddings.mean()).norm(dim=1)
is_anomaly = anomaly_scores > threshold
```

### 2. 负荷预测（微调）

```python
# 在预训练模型基础上添加预测头
class LoadForecastingModel(nn.Module):
    def __init__(self, pretrained_encoder):
        super().__init__()
        self.encoder = pretrained_encoder
        self.forecast_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 24)  # 预测未来24小时
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        forecast = self.forecast_head(encoded[:, -1, :])  # 使用最后一个时间步
        return forecast

# 微调训练
model = LoadForecastingModel(pretrained_encoder)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
```

## 性能基准

### 硬件要求

| GPU | 推荐 | 单区域 | 多区域 |
|-----|------|--------|--------|
| RTX 3090 (24GB) | ✅ | 32-64 | 64-128 |
| RTX 4090 (24GB) | ✅✅ | 64-128 | 128-256 |
| A100 (40GB) | ⭐⭐⭐ | 128+ | 256+ |
| CPU | ❌ | 不推荐 | 不可用 |

### 预期训练时间

- **单区域**（CAISO_zone_1_）：~2-4小时（20 epochs）
- **多区域**（4个CAISO区域）：~8-12小时（20 epochs）
- **全部区域**（18个区域）：~3-7天（50 epochs）

## 常见问题

### Q1: 如何处理缺失数据？
```python
# 时间序列数据集会自动跳过包含缺失值的窗口
dataset = TimeSeriesDataset(df_with_nans)  # 自动处理NaN
```

### Q2: 如何调整学习曲线？
```bash
# 减小学习率应对不稳定
python train_pretrain.py --learning-rate 5e-4

# 增大批次大小以加速收敛
python train_pretrain.py --batch-size 64
```

### Q3: CUDA内存不足怎么办？
```bash
# 减小批次大小
python train_pretrain.py --batch-size 8

# 减小序列长度
python train_pretrain.py --seq-len 720

# 减小隐藏维度
python train_pretrain.py --hidden-dim 128
```

### Q4: 如何恢复中断的训练？
```python
# 加载检查点
checkpoint = torch.load('checkpoints/spikformer_pretrain/checkpoint_epoch_10.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
trainer.epoch = checkpoint['epoch']
trainer.train_losses = checkpoint['train_losses']
```

## 扩展方向

### 1. 多模态融合
- 融合PMU毫秒级数据和负荷分钟级数据
- 支持天气、外部事件等异构数据

### 2. 多任务学习
- 同时学习重建、异常检测、预测等任务
- 动态任务权重调整

### 3. 联邦学习
- 多个电网运营商的协作训练
- 隐私保护的模型聚合

### 4. 在线学习
- 实时模型更新
- 概念漂移适应

## 参考资源

- [Spikformer: The First Globally Event-driven Transformer for Vision](https://arxiv.org/abs/2404.09310)
- [Spiking Neural Networks (SNN)](https://snntorch.readthedocs.io/)
- [Open-source Power Dataset](https://github.com/tamu-engineering-research/Open-source-power-dataset)

## 贡献和反馈

欢迎提交问题、建议或改进！

## 许可证

本项目遵循MIT许可证。

---

**最后更新**：2026年4月
