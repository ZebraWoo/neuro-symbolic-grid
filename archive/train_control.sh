#!/bin/bash
# 电网调控模型训练脚本 - 支持多种神经元模型

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 配置参数
MODEL_TYPE="${1:-multimodal}"  # multimodal | lif | izh | hybrid
EPOCHS="${2:-50}"
BATCH_SIZE="${3:-32}"
LR="${4:-0.001}"
DEVICE="${5:-cuda}"

print_info "=========================================="
print_info "电网自主调控预训练系统"
print_info "=========================================="
print_info "模型类型: $MODEL_TYPE"
print_info "训练轮数: $EPOCHS"
print_info "批次大小: $BATCH_SIZE"
print_info "学习率: $LR"
print_info "计算设备: $DEVICE"

# 检查环境
print_info "检查Python环境..."
if ! command -v python &> /dev/null; then
    print_error "未找到Python"
    exit 1
fi

print_success "Python版本: $(python --version)"

# 检查关键模块
print_info "检查PyTorch..."
if ! python -c "import torch; print(f'PyTorch版本: {torch.__version__}')" 2>/dev/null; then
    print_error "PyTorch未安装"
    exit 1
fi

# 创建输出目录
print_info "创建输出目录..."
mkdir -p logs checkpoints outputs

DATA_ROOT="${DATA_ROOT:-/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable}"
ZONES="${ZONES:-ERCOT_zone_1_}"
SEQ_LEN="${SEQ_LEN:-96}"
STRIDE="${STRIDE:-96}"
MAX_ROWS="${MAX_ROWS:-}"

# 运行训练（PSML 多模态，预测下一时刻负荷）
print_info "开始训练 (PSML multimodal, target=next load_power)..."
print_info "  data_root=$DATA_ROOT"
print_info "  zones=$ZONES"

EXTRA_ROWS=()
if [ -n "$MAX_ROWS" ]; then
    EXTRA_ROWS=(--max-rows-per-zone "$MAX_ROWS")
fi

python train_control.py \
    --model-type "$MODEL_TYPE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --device "$DEVICE" \
    --data-root "$DATA_ROOT" \
    --zones $ZONES \
    --seq-len "$SEQ_LEN" \
    --stride "$STRIDE" \
    "${EXTRA_ROWS[@]}"

if [ $? -eq 0 ]; then
    print_success "训练完成!"
else
    print_error "训练失败"
    exit 1
fi

# 总结
print_info "=========================================="
print_success "工作总结："
print_success "  ✓ 第①步：神经元与突含模型库"
print_success "    - LIF (泄漏积分发火)"
print_success "    - Hodgkin-Huxley (生物学精确)"
print_success "    - 静态/动态突触"
print_success "    - STDP可塑性"
print_success ""
print_success "  ✓ 第②步：多模态自主调控模型"
print_success "    - 多模态嵌入和融合"
print_success "    - 脉冲Transformer块"
print_success "    - 调控决策头"
print_success "    - 4组分预训练损失"
print_success ""
print_success "  ✓ 第③步：Izhikevich与多室模型（接口）"
print_success "    - IzhikevichNeuronInterface"
print_success "    - MultiCompartmentNeuronInterface"
print_success "    - HybridNeuronNetwork"
print_success "    - 工厂函数和配置系统"
print_info "=========================================="
print_info "输出位置:"
print_info "  模型: checkpoints/control_model_psml.pth"
print_info "  历史: outputs/training_history_psml.json"
print_info "  代码: src/control/"
