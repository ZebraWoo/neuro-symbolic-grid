#!/bin/bash

# 轻量级测试脚本 - 用于快速验证训练管道
# 显存占用: ~2-5GB (单GPU可运行)

echo "=================================="
echo "🚀 轻量级训练测试 (Lightweight Test)"
echo "=================================="
echo ""
echo "配置:"
echo "  - Batch Size: 4"
echo "  - Sequence Length: 144 (6小时)"
echo "  - Hidden Dimension: 64"
echo "  - Epochs: 2"
echo "  - Estimated VRAM: 2-3 GB"
echo ""
echo "预期运行时间: 30-60分钟"
echo ""

source activate snn

python train_with_metrics.py \
    --batch-size 4 \
    --seq-len 144 \
    --hidden-dim 64 \
    --num-epochs 2 \
    --learning-rate 0.001 \
    --device auto

echo ""
echo "✅ 测试完成!"
echo "查看结果: checkpoints/spikformer_with_metrics/"
