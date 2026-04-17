#!/bin/bash
# Spikformer预训练 - 完整训练脚本
# 配置：12个区域（4个CAISO+8个ERCOT），50个epoch，~8-10小时
# 显存占用：~19GB/GPU

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

echo "🚀 Spikformer预训练 - 完整版本"
echo "=================================="
echo "配置: 12个区域(CAISO+ERCOT) + 50 epochs"
echo "显存: ~19GB/GPU"
echo "时间: ~8-10小时"
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
mkdir -p checkpoints/spikformer_pretrain_full

# 启动训练
echo "⏱️  开始训练（请稍候，这会花费8-10小时）..."
echo "💡 提示：可以在另一个终端运行 'nvidia-smi -l 1' 监控显存"
echo ""

python train_pretrain.py \
    --zones CAISO_zone_1_ CAISO_zone_2_ CAISO_zone_3_ CAISO_zone_4_ \
            ERCOT_zone_1_ ERCOT_zone_2_ ERCOT_zone_3_ ERCOT_zone_4_ \
            MISO_zone_1_ MISO_zone_2_ MISO_zone_3_ MISO_zone_4_ \
    --batch-size 96 \
    --hidden-dim 512 \
    --embedding-dim 256 \
    --seq-len 1440 \
    --learning-rate 5e-4 \
    --num-epochs 50 \
    --checkpoint-dir ./checkpoints/spikformer_pretrain_full \
    --device cuda

echo ""
echo "✅ 训练完成！"
echo "模型保存在: ./checkpoints/spikformer_pretrain_full/"
