# 🚀 Spikformer 预训练 - 评估指标和可视化 

## 👋 欢迎！这是你需要知道的一切

### ✨ 刚创建的功能

✅ **3个Shell脚本** - 一键启动训练和可视化  
✅ **2个Python脚本** - 支持所有评估指标计算  
✅ **1个Python模块** - 完整的训练器实现  
✅ **4份文档** - 详细的使用指南和参考  

### 🎯 支持的评估指标

| 指标 | 说明 |
|------|------|
| **AUC** | 分类性能 (Area Under Curve) |
| **Precision** | 精确率 |
| **Recall** | 召回率 |
| **F1-Score** | Precision和Recall的调和均值 |
| **Silhouette Score** | 聚类质量评估 |
| **Proximity** | 异常检测评分 |

---

## 🎬 快速开始（3步）

### 第1步：启动训练
```bash
bash run_all.sh
```
**就这样！** 脚本会自动：
- 启动训练
- 计算所有指标
- 生成可视化图表

### 第2步：等待完成
- 训练时间：~2-3小时（取决于显卡和数据量）
- 显示进度条和实时指标

### 第3步：查看结果
```bash
cd ./results
ls -lh *.png
```

生成的4张图表：
1. **1_loss_curve.png** - 损失曲线
2. **2_metrics_curve.png** - AUC、Precision、Recall、F1
3. **3_clustering_metrics.png** - Silhouette和Proximity
4. **4_summary_report.png** - 训练统计总结

---

## 📚 文档导航

### 🔰 新手入门
👉 **先看这个：** `bash QUICK_REFERENCE.sh`

快速参考指南，涵盖：
- 核心脚本用法
- 常见参数
- 常见问题解决

### 📖 详细指南
👉 **深入学习：** `TRAIN_METRICS_GUIDE.md`

包含：
- 所有参数的详细说明
- Python API 用法
- 评估指标的数学定义
- 建议和最佳实践

### 🔍 模型输出
👉 **了解输出：** `MODEL_OUTPUT_GUIDE.md`

包含：
- 模型输出的具体含义
- 如何使用嵌入向量
- 下游应用示例

### 📋 文件清单
👉 **全面了解：** `FILES_SUMMARY.md`

包含：
- 所有创建的文件列表
- 文件关系图
- 输出目录结构

---

## 💻 Shell脚本速查

### `run_all.sh` - 完整工作流（推荐！）
```bash
# 默认配置
bash run_all.sh

# 自定义参数
bash run_all.sh --num-epochs 50 --batch-size 32

# 只训练，不可视化
bash run_all.sh --skip-visualization

# 只可视化，不训练
bash run_all.sh --skip-training
```

### `train_with_metrics.sh` - 仅训练
```bash
# 默认配置（30轮，批次16）
bash train_with_metrics.sh

# 快速验证（5轮）
bash train_with_metrics.sh --num-epochs 5

# 显存不足时
bash train_with_metrics.sh --batch-size 8 --seq-len 360

# 查看所有参数
bash train_with_metrics.sh --help
```

### `plot_metrics.sh` - 仅可视化
```bash
# 从默认路径读取指标
bash plot_metrics.sh

# 自定义路径
bash plot_metrics.sh --metrics-file ./path/to/metrics.json

# 帮助
bash plot_metrics.sh --help
```

---

## 🎯 常见场景

### 场景1：我想立即开始训练
```bash
bash run_all.sh
```
✅ 自动做好一切

### 场景2：我要快速验证（5分钟）
```bash
bash train_with_metrics.sh --num-epochs 1 --batch-size 32
bash plot_metrics.sh
```

### 场景3：我有足够的时间（完整训练）
```bash
bash train_with_metrics.sh --num-epochs 50 --batch-size 16
bash plot_metrics.sh
```

### 场景4：我的显存不足
```bash
bash train_with_metrics.sh \
    --batch-size 8 \
    --seq-len 360 \
    --num-epochs 20
bash plot_metrics.sh
```

### 场景5：我已经有训练结果，只要可视化
```bash
bash plot_metrics.sh
```

---

## 📊 生成的文件

### 检查点目录
```
./checkpoints/spikformer_with_metrics/
├── best_model.pt          ← 最佳模型（用于推理）
├── checkpoint_epoch_5.pt  ← 断点继续训练用
├── checkpoint_epoch_10.pt
└── ...
```

### 结果目录
```
./results/
├── metrics.json                ← 原始数据（可用于自定义分析）
├── 1_loss_curve.png            ← 损失曲线
├── 2_metrics_curve.png         ← 分类指标
├── 3_clustering_metrics.png    ← 聚类指标
└── 4_summary_report.png        ← 统计总结
```

---

## 🔥 评估指标解读

### AUC (Area Under Curve)
- **范围:** 0 ~ 1
- **说明:** 分类器区分能力
- **目标:** > 0.7 为优秀
- **应用:** 二分类性能评估

### Precision (精确率)
- **范围:** 0 ~ 1
- **说明:** 预测的正样本中有多少真的是正样本
- **目标:** 越高越好
- **应用:** 当误报成本高时重视此指标

### Recall (召回率)
- **范围:** 0 ~ 1
- **说明:** 真实正样本中有多少被成功找出
- **目标:** 越高越好
- **应用:** 当漏报成本高时重视此指标

### F1-Score
- **范围:** 0 ~ 1
- **说明:** Precision 和 Recall 的调和均值
- **目标:** > 0.7 为优秀
- **应用:** 综合评估分类性能

### Silhouette Score
- **范围:** -1 ~ 1
- **说明:** 聚类的紧密性和分离度
- **目标:** > 0.5 为优秀，0.3-0.5 中等，< 0.3 较差
- **应用:** 评估特征表示的聚类质量

### Proximity (接近度)
- **范围:** 0 ~ 1
- **说明:** 样本与动态原型的相似度
- **目标:** 高值=正常，低值=异常
- **应用:** 实时异常检测

---

## 🐍 Python API 用法

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

# 加载最佳模型
checkpoint = torch.load('./checkpoints/spikformer_with_metrics/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 进行推理
x = torch.randn(4, 720, 11)
with torch.no_grad():
    output = model(x)

# 获取特征
embeddings = output['embedding']      # (4, 64) - 特征表示
proximity = output['proximity']        # (4,) - 接近度
encoded = output['encoded']            # (4, 720, 256) - 编码
```

### 异常检测
```python
# 计算接近度阈值
threshold = proximity.mean() - proximity.std()

# 检测异常
is_anomaly = proximity < threshold

# 获取异常指标
anomaly_scores = 1 - proximity  # 1=异常，0=正常
```

### 查看指标数据
```python
import json

with open('./results/metrics.json', 'r') as f:
    metrics = json.load(f)

# 查看各种指标
print(f"AUC: {metrics['val_auc'][-1]:.4f}")
print(f"F1: {metrics['val_f1'][-1]:.4f}")
print(f"Silhouette: {metrics['val_silhouette'][-1]:.4f}")
```

---

## ⚙️ 主要参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--batch-size` | 16 | 批次大小（显存小改成8） |
| `--seq-len` | 720 | 序列长度（分钟数） |
| `--num-epochs` | 30 | 训练轮数 |
| `--learning-rate` | 0.001 | 学习率 |
| `--hidden-dim` | 128 | 隐藏维度 |
| `--embedding-dim` | 64 | 嵌入维度 |
| `--device` | auto | 设备(cuda/cpu/auto) |

---

## 🆘 常见问题

### Q: 训练太慢了
**A:** 减少数据量或轮数：
```bash
bash train_with_metrics.sh --num-epochs 5
```

### Q: 显存不足
**A:** 减小batch和seq_len：
```bash
bash train_with_metrics.sh --batch-size 8 --seq-len 360
```

### Q: 指标都是NaN
**A:** 检查数据路径和学习率：
```bash
bash train_with_metrics.sh --learning-rate 0.0001 --num-epochs 5
```

### Q: 绘图失败
**A:** 确保指标文件存在：
```bash
ls -la ./results/metrics.json
bash plot_metrics.sh
```

---

## 📱 推荐工作流

1. **快速验证** (5分钟)
   ```bash
   bash train_with_metrics.sh --num-epochs 1
   bash plot_metrics.sh
   ```

2. **完整训练** (2-3小时)
   ```bash
   bash run_all.sh
   ```

3. **查看结果**
   ```bash
   cd ./results && ls -lh *.png
   ```

4. **模型评估**
   ```python
   # 加载最佳模型
   model.load_state_dict(torch.load('./checkpoints/.../best_model.pt')['model_state_dict'])
   # 进行推理...
   ```

5. **异常检测**
   ```python
   proximity = model(x)['proximity']
   is_anomaly = proximity < threshold
   ```

---

## 📞 需要帮助？

### 查看快速参考
```bash
bash QUICK_REFERENCE.sh
```

### 查看详细指南
```bash
cat TRAIN_METRICS_GUIDE.md
cat MODEL_OUTPUT_GUIDE.md
cat FILES_SUMMARY.md
```

### 查看脚本帮助
```bash
bash train_with_metrics.sh --help
bash plot_metrics.sh --help
bash run_all.sh --help
```

---

## ✅ 检查清单

在开始前，确保：
- [ ] 数据路径正确 (`/home/wuzuoxu/Data/PSML/...`)
- [ ] conda环境已激活 (`conda activate snn`)
- [ ] 依赖已安装 (`pip install torch scikit-learn matplotlib seaborn`)
- [ ] 显卡/显存充足（20GB+ 推荐）
- [ ] 脚本有执行权限 (`chmod +x *.sh`)

---

## 🎉 你已准备好了！

现在运行：
```bash
bash run_all.sh
```

然后等待结果...

祝您训练顺利！🚀

---

**相关文件：**
- QUICK_REFERENCE.sh - 快速参考
- TRAIN_METRICS_GUIDE.md - 详细指南  
- MODEL_OUTPUT_GUIDE.md - 模型输出
- FILES_SUMMARY.md - 文件清单
