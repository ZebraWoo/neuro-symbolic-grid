#!/bin/bash
# Spikformer预训练 - 交互式菜单脚本
# 可视化选择要运行的训练方案

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

clear

echo -e "${BLUE}"
echo "╔════════════════════════════════════════╗"
echo "║  Spikformer预训练 - 交互式启动菜单    ║"
echo "║        (80GB显存 × 4张GPU)             ║"
echo "╚════════════════════════════════════════╝"
echo -e "${NC}"
echo ""

echo "选择训练方案："
echo ""
echo "  1️⃣  快速开始 (推荐首次使用)"
echo "      - 4个CAISO区域"
echo "      - 30个epoch，~3-4小时"
echo "      - 显存占用：~18GB/GPU"
echo ""
echo "  2️⃣  完整训练 (推荐实用)"
echo "      - 12个区域 (CAISO + ERCOT)"
echo "      - 50个epoch，~8-10小时"
echo "      - 显存占用：~19GB/GPU"
echo ""
echo "  3️⃣  超级训练 (最高质量)"
echo "      - 全部66个区域"
echo "      - 100个epoch，~24-36小时"
echo "      - 显存占用：~19GB/GPU"
echo ""
echo "  4️⃣  系统测试 (验证环境)"
echo "      - 快速测试（~2分钟）"
echo ""
echo "  0️⃣  退出"
echo ""

read -p "请选择 (0-4): " choice

case $choice in
    1)
        echo ""
        echo -e "${GREEN}▶ 启动快速开始训练...${NC}"
        echo ""
        bash ./run_train_fast.sh
        ;;
    2)
        echo ""
        echo -e "${GREEN}▶ 启动完整训练...${NC}"
        echo ""
        bash ./run_train_full.sh
        ;;
    3)
        echo ""
        echo -e "${YELLOW}⚠️  这会运行24-36小时${NC}"
        read -p "确认启动超级训练吗? (y/n): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            echo ""
            echo -e "${GREEN}▶ 启动超级训练...${NC}"
            echo ""
            bash ./run_train_super.sh
        else
            echo "已取消"
        fi
        ;;
    4)
        echo ""
        echo -e "${GREEN}▶ 运行系统测试...${NC}"
        echo ""
        bash ./run_test.sh
        ;;
    0)
        echo "已退出"
        exit 0
        ;;
    *)
        echo -e "${RED}❌ 无效选择${NC}"
        exit 1
        ;;
esac
