# 🚀 Spikformer预训练 - 5分钟快速开始

## 一句话总结
利用PSML电网负荷数据，训练Spikformer脉冲神经网络学习动态特征表示，为故障检测、负荷预测等任务提供预训练模型。

## 快速命令

### 1️⃣ 安装依赖（1分钟）
```bash
cd /home/wuzuoxu/electric-power-regulation
pip install -r requirements_pretrain.txt
```

### 2️⃣ 快速测试（2分钟）
```bash
# 验证环境和数据
python test_pretrain.py
```

### 3️⃣ 启动训练（立即开始）

**最快版本**（5 epochs, 测试用，~10分钟）
```bash
python train_pretrain.py \
    --zones CAISO_zone_1_ \
    --batch-size 8 \
    --num-epochs 5 \
    --hidden-dim 128
```

**推荐版本**（20 epochs, 生产级，~2-3小时）
```bash
python train_pretrain.py \
    --zones CAISO_zone_1_ CAISO_zone_2_ CAISO_zone_3_ CAISO_zone_4_ \
    --batch-size 32 \
    --num-epochs 20 \
    --learning-rate 1e-3
```

**完整版本**（50 epochs, 所有区域，~3-7天）
```bash
python train_pretrain.py \
    --num-epochs 50 \
    --batch-size 64
```

## 输出和检查点

训练完成后，在 `./checkpoints/spikformer_pretrain/` 找到：
```
checkpoints/spikformer_pretrain/
├── latest_model.pt              # 最新模型权重
├── checkpoint_epoch_5.pt        # 第5个epoch检查点
├── checkpoint_epoch_10.pt
└── ...
```

## 模型使用示例

```python
import torch
from src.models.spikformer_pretrain import SpikformerPretrainModel

# 加载模型
model = SpikformerPretrainModel(input_dim=11, hidden_dim=256, embedding_dim=128)
checkpoint = torch.load('checkpoints/spikformer_pretrain/latest_model.pt')
model.load_state_dict(checkpoint)
model.eval()

# 提取特征（11维时间序列 -> 128维动态特征）
with torch.no_grad():
    x = torch.randn(4, 1440, 11)  # 4个样本，24小时（1440分钟），11个特征
    output = model(x)
    embeddings = output['embedding']  # (4, 128)
```

## 数据说明

### 可用电网区域（共66个）

**CAISO (4区域)**
- CAISO_zone_1_, CAISO_zone_2_, CAISO_zone_3_, CAISO_zone_4_

**ERCOT (8区域)**
- ERCOT_zone_1_ 到 ERCOT_zone_8_

**MISO (6区域)**
- MISO_zone_1_ 到 MISO_zone_6_

**其他（SPP, PJM, NYISO等）**
- 总计66个区域可选

### 特征说明

每个区域包含以下特征（11维）：
- `load_power`: 总负荷功率
- `wind_power`: 风电输出
- `solar_power`: 太阳能输出
- `DHI`: 散射水平辐照度
- `DNI`: 直接法向辐照度
- 等其他气象和电力特征

## 常见问题

### Q: 内存不足怎么办？
```bash
# 减小批次大小
python train_pretrain.py --batch-size 8

# 减小序列长度（从1440分钟改为720分钟）
python train_pretrain.py --seq-len 720

# 减小隐藏维度
python train_pretrain.py --hidden-dim 128
```

### Q: 训练速度太慢？
```bash
# 增大批次大小（如果显存允许）
python train_pretrain.py --batch-size 128

# 减少使用的区域
python train_pretrain.py --zones CAISO_zone_1_

# 使用更多GPU
# 需要修改代码支持分布式训练
```

### Q: 如何恢复中断的训练？
```python
# 手动加载检查点并继续
checkpoint = torch.load('checkpoints/spikformer_pretrain/checkpoint_epoch_10.pt')
model.load_state_dict(checkpoint['model_state_dict'])
trainer.epoch = checkpoint['epoch']
# 然后继续fit()
```

## 预期输出

### 训练日志示例
```
INFO: 使用设备: cuda
INFO: 创建数据加载器...
INFO: 加载 CAISO_zone_1_ 数据... (文件大小: 0.20 GB)
INFO: 加载完成: 形状 (1573923, 11)
INFO: 输入维度: 11
INFO: 创建预训练模型...
INFO: 模型参数: 总计 2,567,456, 可训练 2,567,456

==================================================
开始预训练
==================================================

训练 Epoch 1
  loss: 2.3456
  contrastive: 1.2345
  separation: 0.8901
  
验证
Epoch 1/20 | Train Loss: 2.3456 | Val Loss: 2.1234

... (重复19次)

Epoch 20/20 | Train Loss: 1.0234 | Val Loss: 1.0456

==================================================
预训练完成！
模型保存到: ./checkpoints/spikformer_pretrain
==================================================

==================================================
预训练总结
==================================================
最终训练损失: 1.0234
最终验证损失: 1.0456
模型保存位置: ./checkpoints/spikformer_pretrain
==================================================
```

## 下一步

1. **评估预训练效果**
```python
from src.evaluation.pretrain_evaluation import ModelEvaluator
evaluator = ModelEvaluator(model, device='cuda')
metrics, embeddings = evaluator.evaluate(val_loader)
# 查看 Silhouette Score, Davies-Bouldin Index 等指标
```

2. **故障检测应用**
```python
# 用学到的特征做异常检测
normal_embeddings = model.get_embeddings(normal_data)
test_embeddings = model.get_embeddings(test_data)
anomaly_scores = (test_embeddings - normal_embeddings.mean()).norm(dim=1)
```

3. **负荷预测微调**
```python
# 在预训练模型基础上添加预测头
# 使用更低的学习率（1e-5）微调
```

4. **特征可视化**
```python
# t-SNE降维可视化嵌入空间
from src.evaluation.pretrain_evaluation import EmbeddingEvaluator
evaluator = EmbeddingEvaluator(embeddings)
evaluator.visualize_tsne()
evaluator.visualize_clustering()
```

## 完整文档

详见 `PRETRAIN_README.md` 获取：
- 详细的架构说明
- 所有API文档
- 高级配置选项
- 性能基准
- 扩展方向

---

**祝你训练顺利！** 🎉
