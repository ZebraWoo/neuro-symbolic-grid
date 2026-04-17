#!/bin/bash

###############################################################################
# 训练指标可视化脚本
# 绘制 AUC、Precision、Recall、F1、Silhouette等指标的训练曲线
###############################################################################

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

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

# 默认参数
METRICS_FILE="./results/metrics.json"
OUTPUT_DIR="./results"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --metrics-file)
            METRICS_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --metrics-file PATH    指标JSON文件路径 (默认: ./results/metrics.json)"
            echo "  --output-dir PATH      输出目录 (默认: ./results)"
            echo "  --help                 显示帮助信息"
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            exit 1
            ;;
    esac
done

main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║          训练指标可视化                                             ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # 激活conda环境
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate snn 2>/dev/null || {
            print_warning "无法激活snn环境，继续使用当前环境"
        }
    fi
    
    print_info "检查指标文件..."
    
    if [ ! -f "$METRICS_FILE" ]; then
        print_error "指标文件不存在: $METRICS_FILE"
        echo ""
        echo "📌 请先运行以下命令进行训练："
        echo "   bash train_with_metrics.sh"
        exit 1
    fi
    
    print_success "指标文件存在"
    echo ""
    
    print_info "开始生成可视化..."
    echo ""
    
    # 运行可视化脚本
    python plot_metrics.py \
        --metrics-file "$METRICS_FILE" \
        --output-dir "$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo ""
        print_success "可视化完成！"
        echo ""
        echo "📁 生成的图表位置: $OUTPUT_DIR"
        echo ""
        echo "📊 生成的可视化文件："
        echo "   1️⃣  1_loss_curve.png"
        echo "       • 训练和验证损失曲线"
        echo "       • 显示模型学习过程"
        echo ""
        echo "   2️⃣  2_metrics_curve.png"
        echo "       • AUC (Area Under Curve) - 分类性能"
        echo "       • Precision (精确率) - 预测正样本的准确性"
        echo "       • Recall (召回率) - 找出所有正样本的能力"
        echo "       • F1-Score - Precision和Recall的调和均值"
        echo ""
        echo "   3️⃣  3_clustering_metrics.png"
        echo "       • Silhouette Score - 聚类质量 ([-1, 1], 越接近1越好)"
        echo "       • Proximity - 接近度评分 (用于异常检测)"
        echo ""
        echo "   4️⃣  4_summary_report.png"
        echo "       • 训练统计总结"
        echo "       • 所有指标的最佳值和最终值"
        echo ""
        echo "💡 打开文件位置："
        echo "   cd $OUTPUT_DIR && ls -lh *.png"
        echo ""
    else
        print_error "可视化失败！"
        exit 1
    fi
}

main
