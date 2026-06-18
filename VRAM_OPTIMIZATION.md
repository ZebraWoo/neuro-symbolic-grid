# 显存管理与优化指南

## 📊 当前项目显存需求

| 配置 | Batch | Seq Len | Hidden Dim | 显存估算 | 状态 |
|------|-------|---------|-----------|--------|------|
| 完整 | 16 | 720 | 128 | ~162 MB | ⚠️ 需要清理GPU |
| 轻量级 | 4 | 144 | 64 | ~2.2 MB | ✅ 推荐日常用 |
| 超轻 | 2 | 72 | 32 | ~0.2 MB | ✅ 快速原型 |

注: 实际显存占用会是估算的10-50倍，因为还需要存储:
- 完整梯度
- 优化器状态 (Adam需要2倍参数显存)
- 数据缓冲区

## 🔧 显存优化方案

### 方案1: 梯度检查点 (Gradient Checkpointing)
**效果**: 减少60%显存 | **代价**: 速度慢10-20%

```python
import torch.utils.checkpoint as checkpoint

# 在模型前向传播中使用
output = checkpoint.checkpoint(
    self.transformer_block,
    x,
    use_reentrant=False
)
```

**适用**: 当显存充足但不足以支持完整配置时

### 方案2: 混精度训练 (Mixed Precision)
**效果**: 减少50%显存 | **代价**: 速度快20-30%

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(x)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**优势**: 既减少显存又加快速度!
**适用**: 所有情况都推荐启用

### 方案3: 减少数据加载区域
**效果**: 减少80%数据显存 | **代价**: 训练数据减少

```bash
# 只加载12个区域而不是66个
python train_with_metrics.py \
    --zones CAISO_zone_1_ CAISO_zone_2_ CAISO_zone_3_ \
            CAISO_zone_4_ ERCOT_zone_1_ ERCOT_zone_2_ \
            ERCOT_zone_3_ ERCOT_zone_4_ ERCOT_zone_5_ \
            ERCOT_zone_6_ ERCOT_zone_7_ ERCOT_zone_8_
```

**适用**: 快速原型、调试时

## 🚀 推荐工作流

### 第1阶段: 快速验证 (5-10分钟)
```bash
# 使用超轻量配置验证管道可行性
bash test_ultra_lightweight.sh
```

### 第2阶段: 充分测试 (1-2小时)
```bash
# 使用轻量级配置进行实际模型评估
bash test_lightweight.sh
```

### 第3阶段: 完整训练 (可选, 需清理GPU)
```bash
# 清理其他进程
pkill -9 python  # ⚠️ 谨慎使用

# 使用完整配置训练
python train_with_metrics.py \
    --batch-size 16 \
    --seq-len 720 \
    --hidden-dim 128 \
    --num-epochs 30
```

## 📈 性能 vs 显存权衡

```
        性能 ↑
           │
    完整配置│  ████████████ (最佳精度)
           │  大显存 (需清理GPU)
           │
轻量级配置 │  ████████ (好)
           │  中显存 (推荐)
           │
超轻配置   │  ██ (快速验证)
           │  小显存 (最优)
           └─────────────────────→ 显存 ↓
```

## 🎯 快速诊断

如果遇到OOM:

```bash
# 1. 检查当前显存使用
nvidia-smi

# 2. 立即降低batch size
python train_with_metrics.py --batch-size 2

# 3. 降低seq_len
python train_with_metrics.py --seq-len 144

# 4. 降低hidden_dim
python train_with_metrics.py --hidden-dim 64

# 5. 减少数据加载
python train_with_metrics.py --zones CAISO_zone_1_ ERCOT_zone_1_

# 6. 启用混精度 (需要代码修改)
```

## 📝 显存清理脚本

```bash
#!/bin/bash
# kill_old_gpu_processes.sh
# 安全清理僵尸GPU进程

echo "🧹 清理GPU进程..."

# 找出占用GPU显存超过1小时的进程
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    runtime=$(ps -p $pid -o etime= | awk '{print $1}')
    echo "PID: $pid, Runtime: $runtime"
done

# 清理僵尸Python进程
pkill -9 -f "DreamDojo"
pkill -9 -f "spikellm"

echo "✅ 清理完成"
```

## 📞 技术支持

如需帮助，请检查:
1. `nvidia-smi` 显示的显存占用
2. 当前批量大小和序列长度
3. 是否有其他程序占用GPU
4. 模型是否启用了梯度累积

---
最后更新: 2026-04-18
