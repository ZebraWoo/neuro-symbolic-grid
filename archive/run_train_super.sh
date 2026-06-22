#!/bin/bash
# Spikformer预训练 - 超级训练脚本
# 配置：所有66个区域，100个epoch，~24-36小时
# 显存占用：~19GB/GPU
# 适合长期后台运行（推荐用 tmux 或 screen）

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

echo "🚀 Spikformer预训练 - 超级版本"
echo "=================================="
echo "配置: 所有66个区域 + 100 epochs"
echo "显存: ~19GB/GPU"
echo "时间: ~24-36小时"
echo "建议: 用 tmux/screen 后台运行"
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
mkdir -p checkpoints/spikformer_pretrain_super

# 启动训练
echo "⏱️  开始训练（请稍候，这会花费24-36小时）..."
echo "💡 建议在 tmux 中运行："
echo "   tmux new-session -d -s pretrain './run_train_super.sh'"
echo ""

python train_pretrain.py \
    --batch-size 64 \
    --hidden-dim 768 \
    --embedding-dim 512 \
    --seq-len 1440 \
    --learning-rate 1e-4 \
    --num-epochs 100 \
    --checkpoint-dir ./checkpoints/spikformer_pretrain_super \
    --device cuda

echo ""
echo "✅ 训练完成！"
echo "模型保存在: ./checkpoints/spikformer_pretrain_super/"
