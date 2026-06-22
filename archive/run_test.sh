#!/bin/bash
# Spikformer预训练 - 测试脚本
# 快速验证环境和数据（~2分钟）

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

echo "🧪 Spikformer预训练 - 系统测试"
echo "=================================="
echo ""

# 运行测试
python test_pretrain.py

echo ""
echo "✅ 测试完成！"
echo ""
echo "接下来可以运行以下脚本启动训练："
echo "  ./run_train_fast.sh    - 快速版本（推荐首次使用）"
echo "  ./run_train_full.sh    - 完整版本（平衡速度和质量）"
echo "  ./run_train_super.sh   - 超级版本（最高质量，耗时最长）"
