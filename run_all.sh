#!/bin/bash

###############################################################################
# 完整工作流：训练 + 可视化
# 一键启动 Spikformer 预训练，自动生成所有评估指标和可视化
###############################################################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${CYAN}"
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║            $1"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_section() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
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
SKIP_TRAINING=false
SKIP_VISUALIZATION=false

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
        --num-epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-visualization)
            SKIP_VISUALIZATION=true
            shift
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --data-root PATH           数据集根目录"
            echo "  --zones ZONE1 ZONE2 ...   电网区域列表"
            echo "  --batch-size N             批次大小 (默认: 16)"
            echo "  --seq-len N                序列长度 (默认: 720)"
            echo "  --num-epochs N             训练轮数 (默认: 30)"
            echo "  --device DEVICE            计算设备: auto/cuda/cpu (默认: auto)"
            echo "  --skip-training            跳过训练，只进行可视化"
            echo "  --skip-visualization       跳过可视化，只进行训练"
            echo "  --help                     显示帮助信息"
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            exit 1
            ;;
    esac
done

main() {
    print_header "Spikformer 预训练完整工作流"
    
    print_section "📋 配置检查"
    
    echo "数据配置:"
    echo "  • 数据路径: $DATA_ROOT"
    echo "  • 区域数量: ${#ZONES[@]}"
    echo "  • 区域列表: ${ZONES[*]}"
    echo ""
    echo "训练配置:"
    echo "  • 批次大小: $BATCH_SIZE"
    echo "  • 序列长度: $SEQ_LEN"
    echo "  • 学习率: $LEARNING_RATE"
    echo "  • 训练轮数: $NUM_EPOCHS"
    echo "  • 计算设备: $DEVICE"
    echo ""
    
    # 检查数据目录
    if [ ! -d "$DATA_ROOT" ]; then
        print_error "数据目录不存在: $DATA_ROOT"
        exit 1
    fi
    print_success "数据目录验证通过"
    echo ""
    
    # 第一步：训练
    if [ "$SKIP_TRAINING" = false ]; then
        print_section "🚀 第一步：启动训练"
        
        print_info "运行命令："
        echo "  bash train_with_metrics.sh \\"
        echo "    --data-root \"$DATA_ROOT\" \\"
        echo "    --zones ${ZONES[@]} \\"
        echo "    --batch-size $BATCH_SIZE \\"
        echo "    --seq-len $SEQ_LEN \\"
        echo "    --num-epochs $NUM_EPOCHS \\"
        echo "    --device $DEVICE"
        echo ""
        
        bash train_with_metrics.sh \
            --data-root "$DATA_ROOT" \
            --zones "${ZONES[@]}" \
            --batch-size "$BATCH_SIZE" \
            --seq-len "$SEQ_LEN" \
            --num-epochs "$NUM_EPOCHS" \
            --device "$DEVICE"
        
        if [ $? -eq 0 ]; then
            print_success "训练阶段完成"
        else
            print_error "训练失败"
            exit 1
        fi
    else
        print_section "⏭️  跳过训练（使用已有的检查点）"
    fi
    
    echo ""
    
    # 第二步：可视化
    if [ "$SKIP_VISUALIZATION" = false ]; then
        print_section "🎨 第二步：生成可视化"
        
        print_info "运行命令："
        echo "  bash plot_metrics.sh"
        echo ""
        
        bash plot_metrics.sh
        
        if [ $? -eq 0 ]; then
            print_success "可视化生成完成"
        else
            print_error "可视化生成失败"
            exit 1
        fi
    else
        print_section "⏭️  跳过可视化"
    fi
    
    echo ""
    print_section "✨ 工作流完成"
    
    echo "📊 输出位置："
    echo "  • 检查点: $CHECKPOINT_DIR"
    echo "  • 指标: ./results/metrics.json"
    echo "  • 图表: ./results/*.png"
    echo ""
    echo "📈 可查看的评估指标："
    echo "  ✓ AUC (Area Under Curve)"
    echo "  ✓ Precision (精确率)"
    echo "  ✓ Recall (召回率)"
    echo "  ✓ F1-Score"
    echo "  ✓ Silhouette Score (聚类质量)"
    echo "  ✓ Proximity Score (接近度/异常检测)"
    echo "  ✓ 损失曲线 (训练 vs 验证)"
    echo ""
    echo "💡 下一步："
    echo "  1. 查看可视化图表:"
    echo "     cd ./results && ls -lh *.png"
    echo ""
    echo "  2. 查看详细指标:"
    echo "     cat ./results/metrics.json | python -m json.tool"
    echo ""
    echo "  3. 加载训练好的模型进行推理:"
    echo "     python -c \""
    echo "     import torch"
    echo "     from src.models.spikformer_pretrain import SpikformerPretrainModel"
    echo "     model = SpikformerPretrainModel(input_dim=11, hidden_dim=128, embedding_dim=64)"
    echo "     ckpt = torch.load('$CHECKPOINT_DIR/best_model.pt')"
    echo "     model.load_state_dict(ckpt['model_state_dict'])"
    echo "     \""
    echo ""
}

main
