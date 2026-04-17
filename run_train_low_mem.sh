#!/bin/bash
# Spikformer预训练 - 低显存版本（针对显存竞争场景）
# 批次：16，序列：720，隐藏：128，~2小时

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

echo "🚀 Spikformer预训练 - 低显存版本"
echo "=================================="
echo "配置: 单个区域 + 保守配置"
echo "显存: ~6-8GB/GPU"
echo "时间: ~2小时"
echo "=================================="
echo ""

# 检查环境
echo "✓ 检查环境..."
if ! python3 -c "import torch" 2>/dev/null; then
    echo "❌ PyTorch未安装"
    exit 1
fi

mkdir -p checkpoints/spikformer_pretrain_low_mem

echo "⏱️  开始训练..."
echo ""

python train_pretrain.py \
    --zones CAISO_zone_1_ \
    --batch-size 16 \
    --seq-len 720 \
    --hidden-dim 128 \
    --embedding-dim 64 \
    --learning-rate 1e-3 \
    --num-epochs 20 \
    --checkpoint-dir ./checkpoints/spikformer_pretrain_low_mem \
    --device cuda

echo ""
echo "✅ 训练完成！"
echo "模型保存在: ./checkpoints/spikformer_pretrain_low_mem/"
