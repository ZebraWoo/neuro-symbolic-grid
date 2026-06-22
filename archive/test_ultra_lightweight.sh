#!/bin/bash

# 超轻量级测试脚本 - 用于极速原型验证
# 显存占用: <1GB (任何GPU都能运行)

echo "=================================="
echo "⚡ 超轻量级快速测试 (Ultra-Lightweight)"
echo "=================================="
echo ""
echo "配置:"
echo "  - Batch Size: 2"
echo "  - Sequence Length: 72 (3小时)"
echo "  - Hidden Dimension: 32"
echo "  - Epochs: 1"
echo "  - Estimated VRAM: 0.2-0.5 GB"
echo ""
echo "预期运行时间: 10-20分钟"
echo "用途: 快速验证管道、调试模型"
echo ""

source activate snn

python train_with_metrics.py \
    --batch-size 2 \
    --seq-len 72 \
    --hidden-dim 32 \
    --num-epochs 1 \
    --learning-rate 0.001 \
    --device auto

echo ""
echo "✅ 快速测试完成!"
echo "日志: checkpoints/spikformer_with_metrics/training.log"
