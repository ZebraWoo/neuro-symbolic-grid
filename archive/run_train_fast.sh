#!/bin/bash
# Spikformer预训练 - 快速开始脚本（推荐首次使用）
# 配置：4个CAISO区域，30个epoch，~3-4小时
# 显存占用：~18GB/GPU

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

echo "🚀 Spikformer预训练 - 快速开始版本"
echo "=================================="
echo "配置: 4个CAISO区域 + 30 epochs"
echo "显存: ~18GB/GPU"
echo "时间: ~3-4小时"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEQ_LEN="${SEQ_LEN:-1440}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
EMBEDDING_DIM="${EMBEDDING_DIM:-256}"
echo "GPU: ${GPU_ID}"
echo "batch_size: ${BATCH_SIZE}, seq_len: ${SEQ_LEN}, hidden_dim: ${HIDDEN_DIM}, embedding_dim: ${EMBEDDING_DIM}"
echo "=================================="
echo ""

# 检查依赖
echo "✓ 检查环境..."
if ! python3 -c "import torch" 2>/dev/null; then
    echo "❌ PyTorch未安装"
    echo "请运行: pip install -r requirements_pretrain.txt"
    exit 1
fi

# 创建检查点目录
mkdir -p checkpoints/spikformer_pretrain

# 启动训练
echo "⏱️  开始训练（请稍候，这会花费3-4小时）..."
echo ""

CUDA_VISIBLE_DEVICES="${GPU_ID}" python train_pretrain.py \
    --zones CAISO_zone_1_ CAISO_zone_2_ CAISO_zone_3_ CAISO_zone_4_ \
    --batch-size "${BATCH_SIZE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --embedding-dim "${EMBEDDING_DIM}" \
    --seq-len "${SEQ_LEN}" \
    --learning-rate 5e-4 \
    --num-epochs 30 \
    --checkpoint-dir ./checkpoints/spikformer_pretrain_fast \
    --device cuda

echo ""
echo "✅ 训练完成！"
echo "模型保存在: ./checkpoints/spikformer_pretrain_fast/"
