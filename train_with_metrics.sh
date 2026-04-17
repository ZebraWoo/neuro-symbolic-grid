#!/bin/bash

###############################################################################
# Spikformer 预训练脚本 - 带完整的评估指标
# 支持 AUC、Precision、Recall、F1、Silhouette等指标
###############################################################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# 检查环境
check_environment() {
    print_info "检查Python环境..."
    
    # 激活conda环境
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate snn 2>/dev/null || {
            print_error "无法激活snn环境"
            exit 1
        }
        print_success "已激活conda环境: snn"
    else
        print_warning "未找到conda，请手动激活环境"
    fi
    
    # 检查Python
    python_version=$(python --version 2>&1 | awk '{print $2}')
    print_success "Python版本: $python_version"
    
    # 检查必需的包
    for package in torch numpy matplotlib seaborn scikit-learn tqdm; do
        if python -c "import $package" 2>/dev/null; then
            print_success "$package 已安装"
        else
            print_error "$package 未安装"
        fi
    done
}

# 显示配置
show_config() {
    print_info "训练配置："
    echo ""
    echo "  数据配置:"
    echo "    • 数据路径: ${DATA_ROOT}"
    echo "    • 电网区域: ${ZONES[@]}"
    echo ""
    echo "  模型配置:"
    echo "    • 隐藏维度: ${HIDDEN_DIM}"
    echo "    • 嵌入维度: ${EMBEDDING_DIM}"
    echo ""
    echo "  训练配置:"
    echo "    • 批次大小: ${BATCH_SIZE}"
    echo "    • 序列长度: ${SEQ_LEN}"
    echo "    • 学习率: ${LEARNING_RATE}"
    echo "    • 训练轮数: ${NUM_EPOCHS}"
    echo "    • 计算设备: ${DEVICE}"
    echo ""
    echo "  输出配置:"
    echo "    • 检查点目录: ${CHECKPOINT_DIR}"
    echo ""
}

# 默认参数
DATA_ROOT="/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable"
ZONES=("CAISO_zone_1_" "CAISO_zone_2_" "CAISO_zone_3_")
BATCH_SIZE=16
SEQ_LEN=720
HIDDEN_DIM=128
EMBEDDING_DIM=64
LEARNING_RATE=0.001
NUM_EPOCHS=30
CHECKPOINT_DIR="./checkpoints/spikformer_with_metrics"
DEVICE="auto"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --zones)
            ZONES=()
            shift
            while [[ $# -gt 0 && $1 != --* ]]; do
                ZONES+=("$1")
                shift
            done
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --seq-len)
            SEQ_LEN="$2"
            shift 2
            ;;
        --hidden-dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --embedding-dim)
            EMBEDDING_DIM="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num-epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --data-root PATH           数据集根目录 (默认: /home/wuzuoxu/Data/PSML/...)"
            echo "  --zones ZONE1 ZONE2 ...   电网区域列表 (默认: CAISO_zone_1_ CAISO_zone_2_ CAISO_zone_3_)"
            echo "  --batch-size N             批次大小 (默认: 16)"
            echo "  --seq-len N                序列长度 (默认: 720)"
            echo "  --hidden-dim N             隐藏维度 (默认: 128)"
            echo "  --embedding-dim N          嵌入维度 (默认: 64)"
            echo "  --learning-rate LR         学习率 (默认: 0.001)"
            echo "  --num-epochs N             训练轮数 (默认: 30)"
            echo "  --checkpoint-dir PATH      检查点保存目录"
            echo "  --device DEVICE            计算设备: auto/cuda/cpu (默认: auto)"
            echo "  --help                     显示帮助信息"
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            exit 1
            ;;
    esac
done

# 主函数
main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║       Spikformer 预训练 - 带完整的评估指标                          ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # 检查环境
    check_environment
    echo ""
    
    # 显示配置
    show_config
    echo ""
    
    # 检查数据目录
    if [ ! -d "$DATA_ROOT" ]; then
        print_error "数据目录不存在: $DATA_ROOT"
        exit 1
    fi
    print_success "数据目录存在"
    echo ""
    
    # 开始训练
    print_info "开始训练..."
    echo ""
    
    python train_with_metrics.py \
        --data-root "$DATA_ROOT" \
        --zones "${ZONES[@]}" \
        --batch-size "$BATCH_SIZE" \
        --seq-len "$SEQ_LEN" \
        --hidden-dim "$HIDDEN_DIM" \
        --embedding-dim "$EMBEDDING_DIM" \
        --learning-rate "$LEARNING_RATE" \
        --num-epochs "$NUM_EPOCHS" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --device "$DEVICE"
    
    if [ $? -eq 0 ]; then
        echo ""
        print_success "训练完成！"
        echo ""
        echo "📊 生成的指标和可视化："
        echo "   • 检查点保存位置: $CHECKPOINT_DIR"
        echo "   • 可视化图表位置: ./results/"
        echo ""
        echo "📈 可视化文件包括："
        echo "   1️⃣  loss_curve.png - 训练和验证损失曲线"
        echo "   2️⃣  metrics_curve.png - AUC、Precision、Recall、F1曲线"
        echo "   3️⃣  clustering_metrics.png - Silhouette和接近度曲线"
        echo "   4️⃣  summary_report.png - 训练统计总结"
        echo ""
        echo "🎨 快速查看可视化："
        echo "   bash plot_metrics.sh"
        echo ""
    else
        print_error "训练失败！"
        exit 1
    fi
}

# 执行主函数
main
