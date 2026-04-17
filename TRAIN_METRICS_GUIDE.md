# Spikformer 预训练 - 评估指标和可视化 📊

## 快速开始

三个shell脚本提供了完整的训练和可视化工作流：

### 1️⃣ **完整工作流**（推荐）
```bash
bash run_all.sh
```
一键启动 **训练 → 评估 → 可视化** 全流程

### 2️⃣ **仅训练**
```bash
bash train_with_metrics.sh
```
启动训练，自动计算：
- ✓ AUC (Area Under Curve)
- ✓ Precision (精确率)
- ✓ Recall (召回率)
- ✓ F1-Score
- ✓ Silhouette Score (聚类质量)
- ✓ Proximity Score (异常检测)

### 3️⃣ **仅可视化**
```bash
bash plot_metrics.sh
```
使用现有的训练结果生成可视化图表

---

## 脚本详解

### `train_with_metrics.sh` - 训练脚本

**功能：** 训练Spikformer模型，实时计算评估指标

**使用示例：**
```bash
# 默认配置
bash train_with_metrics.sh

# 自定义配置
bash train_with_metrics.sh \
    --data-root /path/to/data \
    --zones CAISO_zone_1_ CAISO_zone_2_ \
    --batch-size 16 \
    --seq-len 720 \
    --num-epochs 30 \
    --device cuda
```

**参数说明：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-root` | `/home/wuzuoxu/Data/PSML/...` | 数据集根目录 |
| `--zones` | `CAISO_zone_1_ CAISO_zone_2_ CAISO_zone_3_` | 电网区域列表 |
| `--batch-size` | 16 | 批次大小 |
| `--seq-len` | 720 | 序列长度（分钟数） |
| `--hidden-dim` | 128 | 隐藏层维度 |
| `--embedding-dim` | 64 | 嵌入维度 |
| `--learning-rate` | 0.001 | 学习率 |
| `--num-epochs` | 30 | 训练轮数 |
| `--checkpoint-dir` | `./checkpoints/spikformer_with_metrics` | 检查点保存目录 |
| `--device` | `auto` | 计算设备 (auto/cuda/cpu) |

**训练过程中输出：**
```
✅ Python环境检查通过
✅ 数据目录存在
Epoch 1/30
  Training: 100%|████████| 125/125 [00:45<00:00, 2.75it/s]
  Validating: 100%|████████| 32/32 [00:12<00:00, 2.65it/s]
  
训练损失: 0.2456, 训练接近度: 0.6234
验证损失: 0.2134, 验证接近度: 0.6456
AUC: 0.7234
Precision: 0.7100, Recall: 0.6890, F1: 0.6993
Silhouette Score: 0.4567

✅ 最佳模型已保存: ./checkpoints/spikformer_with_metrics/best_model.pt
```

---

### `plot_metrics.sh` - 可视化脚本

**功能：** 从训练指标生成可视化图表

**使用示例：**
```bash
# 使用默认路径
bash plot_metrics.sh

# 自定义路径
bash plot_metrics.sh \
    --metrics-file ./results/metrics.json \
    --output-dir ./my_results
```

**生成的图表：**

1. **1_loss_curve.png** - 损失曲线
   - 显示训练和验证损失随轮数的变化
   - 用于判断过拟合情况

2. **2_metrics_curve.png** - 分类评估指标（2×2网格）
   - **AUC**: 分类器区分能力（0-1，越高越好）
   - **Precision**: 预测正样本中的准确率
   - **Recall**: 找出实际正样本的能力
   - **F1-Score**: Precision和Recall的调和均值

3. **3_clustering_metrics.png** - 聚类和异常检测指标
   - **Silhouette Score**: 聚类紧密度 ([-1, 1])
     - 接近1: 聚类质量优秀
     - 接近0: 聚类欠佳
     - 接近-1: 样本被错误分类
   - **Proximity**: 样本与原型的接近度
     - 用于实时异常检测

4. **4_summary_report.png** - 训练统计总结
   - 所有指标的最佳值和最终值
   - 训练时间统计

---

### `run_all.sh` - 完整工作流

**功能：** 一键启动完整的训练→评估→可视化流程

**使用示例：**
```bash
# 默认配置
bash run_all.sh

# 跳过训练，只可视化已有结果
bash run_all.sh --skip-training

# 跳过可视化，只训练
bash run_all.sh --skip-visualization

# 自定义参数
bash run_all.sh \
    --zones CAISO_zone_1_ CAISO_zone_2_ \
    --num-epochs 50 \
    --batch-size 32
```

**流程输出示例：**
```
╔════════════════════════════════════════════════════════════════════╗
║            Spikformer 预训练完整工作流
╚════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 配置检查
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 数据目录验证通过

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 第一步：启动训练
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[训练过程...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎨 第二步：生成可视化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ loss_curve.png
✅ metrics_curve.png
✅ clustering_metrics.png
✅ summary_report.png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ 工作流完成
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 输出目录结构

```
project_root/
├── checkpoints/
│   └── spikformer_with_metrics/
│       ├── best_model.pt          ← 最佳模型
│       ├── checkpoint_epoch_5.pt   ← 检查点
│       ├── checkpoint_epoch_10.pt
│       └── ...
├── results/
│   ├── metrics.json               ← 原始指标数据（JSON）
│   ├── 1_loss_curve.png           ← 损失曲线
│   ├── 2_metrics_curve.png        ← AUC、Precision、Recall、F1
│   ├── 3_clustering_metrics.png   ← Silhouette、Proximity
│   └── 4_summary_report.png       ← 训练统计总结
└── ...
```

---

## 评估指标说明

### 📈 损失函数相关
- **训练损失**: 模型在训练集上的重建误差
- **验证损失**: 模型在验证集上的重建误差
  - 若验证损失 > 训练损失：表现过拟合
  - 若验证损失 < 训练损失：表现欠拟合

### 🎯 分类指标
- **AUC (Area Under Curve)**: [0, 1]
  - 衡量二分类器的分类性能
  - 0.5: 随机分类；1.0: 完美分类
  
- **Precision (精确率)**: [0, 1]
  - 预测为正的样本中，实际为正的比例
  - `Precision = TP / (TP + FP)`

- **Recall (召回率)**: [0, 1]
  - 实际正样本中，被正确预测的比例
  - `Recall = TP / (TP + FN)`

- **F1-Score**: [0, 1]
  - Precision和Recall的调和均值
  - `F1 = 2 * (Precision * Recall) / (Precision + Recall)`

### 🔷 聚类指标
- **Silhouette Score**: [-1, 1]
  - 衡量聚类的紧密性和分离度
  - 接近1: 聚类质量优秀
  - 接近0: 聚类质量中等
  - 接近-1: 聚类质量很差

### 📍 异常检测指标
- **Proximity Score**: [0, 1]
  - 样本与学习到的"动态原型"的相似度
  - 高值: 样本符合正常动态模式
  - 低值: 样本为异常/罕见模式
  - 可直接用于异常检测阈值设定

---

## 常见问题

### Q1: 训练很慢怎么办？
**A:** 可以调整以下参数：
```bash
bash train_with_metrics.sh \
    --batch-size 32 \
    --seq-len 360 \
    --num-epochs 10
```

### Q2: 显存不足怎么办？
**A:** 减少批次大小和序列长度：
```bash
bash train_with_metrics.sh \
    --batch-size 8 \
    --seq-len 360
```

### Q3: 指标都是NaN或0怎么办？
**A:** 检查数据和模型配置：
- 确保数据路径正确
- 检查数据是否被正确加载
- 尝试增加训练轮数

### Q4: 如何使用已训练的模型进行推理？
**A:** 参见下面的示例代码

---

## Python API 使用

### 加载训练好的模型
```python
import torch
from src.models.spikformer_pretrain import SpikformerPretrainModel

# 创建模型
model = SpikformerPretrainModel(
    input_dim=11, 
    hidden_dim=128, 
    embedding_dim=64
)

# 加载检查点
checkpoint = torch.load(
    './checkpoints/spikformer_with_metrics/best_model.pt'
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理
x = torch.randn(4, 720, 11)  # (batch, seq_len, features)
with torch.no_grad():
    output = model(x)

# 获取输出
embeddings = output['embedding']        # (4, 64) - 特征表示
proximity = output['proximity']          # (4,) - 接近度评分
encoded = output['encoded']              # (4, 720, 256) - 编码

# 异常检测示例
threshold = proximity.mean() - proximity.std()
is_anomaly = proximity < threshold
```

### 访问训练指标
```python
import json

with open('./results/metrics.json', 'r') as f:
    metrics = json.load(f)

# 获取各指标
train_loss = metrics['train_loss']
auc_scores = metrics['val_auc']
silhouette_scores = metrics['val_silhouette']

# 绘制自定义图表
import matplotlib.pyplot as plt
plt.plot(train_loss, label='Loss')
plt.legend()
plt.show()
```

---

## 使用建议

1. **首次运行**: 使用 `bash run_all.sh` 完整流程
2. **参数调优**: 根据任务需求调整学习率、轮数等
3. **监控指标**: 重点关注 **F1-Score** 和 **Silhouette Score**
4. **异常检测**: 使用 **Proximity** 进行实时异常检测
5. **特征提取**: 使用 **embedding** 作为下游任务的特征

---

## 最后

有问题？检查：
- ✓ 数据路径是否正确
- ✓ conda环境是否激活
- ✓ 依赖包是否安装 (`pip install torch scikit-learn matplotlib seaborn`)
- ✓ 显存是否充足

祝训练顺利！🚀
