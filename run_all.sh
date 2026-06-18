#!/bin/bash

###############################################################################
# Full workflow: training + visualization
# One-command Spikformer pretraining with metrics and plots
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${CYAN}"
    echo "===================================================================="
    echo "$1"
    echo "===================================================================="
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
    echo -e "${GREEN}[OK] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

# Default parameters
DATA_ROOT="/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable"
ZONES=("CAISO" "ERCOT" "MISO" "NYISO" "PJM" "SPP")
BATCH_SIZE=8
SEQ_LEN=720
HIDDEN_DIM=128
EMBEDDING_DIM=64
LEARNING_RATE=0.0001
NUM_EPOCHS=8
CHECKPOINT_DIR="./checkpoints/spikformer_with_metrics"
DEVICE="auto"
GPU_ID="${GPU_ID:-0}"
SKIP_TRAINING=false
SKIP_VISUALIZATION=false

# Parse CLI arguments
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
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --data-root PATH           Dataset root path"
            echo "  --zones ZONE1 ZONE2 ...    Grid zone list"
            echo "  --batch-size N             Batch size (default: 16)"
            echo "  --seq-len N                Sequence length (default: 720)"
            echo "  --num-epochs N             Number of epochs (default: 30)"
            echo "  --device DEVICE            Device: auto/cuda/cpu (default: auto)"
            echo "  --skip-training            Skip training and plot only"
            echo "  --skip-visualization       Skip plotting and train only"
            echo "  --help                     Show help"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

main() {
    print_header "Spikformer Pretraining Full Workflow"
    
    print_section "Configuration Check"
    
    echo "Data:"
    echo "  - Data root: $DATA_ROOT"
    echo "  - Zone count: ${#ZONES[@]}"
    echo "  - Zones: ${ZONES[*]}"
    echo ""
    echo "Training:"
    echo "  - Batch size: $BATCH_SIZE"
    echo "  - Sequence length: $SEQ_LEN"
    echo "  - Learning rate: $LEARNING_RATE"
    echo "  - Epochs: $NUM_EPOCHS"
    echo "  - Device: $DEVICE"
    echo "  - GPU_ID: $GPU_ID"
    echo ""
    
    # Check data directory
    if [ ! -d "$DATA_ROOT" ]; then
        print_error "Data directory does not exist: $DATA_ROOT"
        exit 1
    fi
    print_success "Data directory check passed"
    echo ""
    
    # Step 1: training
    if [ "$SKIP_TRAINING" = false ]; then
        print_section "Step 1: Training"
        
        print_info "Running command:"
        echo "  bash train_with_metrics.sh \\"
        echo "    --data-root \"$DATA_ROOT\" \\"
        echo "    --zones ${ZONES[@]} \\"
        echo "    --batch-size $BATCH_SIZE \\"
        echo "    --seq-len $SEQ_LEN \\"
        echo "    --num-epochs $NUM_EPOCHS \\"
        echo "    --device $DEVICE"
        echo ""
        
        CUDA_VISIBLE_DEVICES="$GPU_ID" bash train_with_metrics.sh \
            --data-root "$DATA_ROOT" \
            --zones "${ZONES[@]}" \
            --batch-size "$BATCH_SIZE" \
            --seq-len "$SEQ_LEN" \
            --num-epochs "$NUM_EPOCHS" \
            --device "$DEVICE"
        
        if [ $? -eq 0 ]; then
            print_success "Training completed"
        else
            print_error "Training failed"
            exit 1
        fi
    else
        print_section "Skip training (use existing checkpoints)"
    fi
    
    echo ""
    
    # Step 2: visualization
    if [ "$SKIP_VISUALIZATION" = false ]; then
        print_section "Step 2: Visualization"
        
        print_info "Running command:"
        echo "  bash plot_metrics.sh --metrics-file \"$CHECKPOINT_DIR/metrics.json\" --output-dir \"./results\""
        echo ""
        
        bash plot_metrics.sh --metrics-file "$CHECKPOINT_DIR/metrics.json" --output-dir "./results"
        
        if [ $? -eq 0 ]; then
            print_success "Visualization completed"
        else
            print_error "Visualization failed"
            exit 1
        fi
    else
        print_section "Skip visualization"
    fi
    
    echo ""
    print_section "Workflow Completed"
    
    echo "Outputs:"
    echo "  - Checkpoints: $CHECKPOINT_DIR"
    echo "  - Metrics: $CHECKPOINT_DIR/metrics.json"
    echo "  - Plots: ./results/*.png"
    echo ""
    echo "Available metrics:"
    echo "  - AUC (Area Under Curve)"
    echo "  - Precision"
    echo "  - Recall"
    echo "  - F1-Score"
    echo "  - Silhouette Score"
    echo "  - Proximity Score"
    echo "  - Loss curves (train vs val)"
    echo ""
    echo "Next steps:"
    echo "  1. Check generated plots:"
    echo "     cd ./results && ls -lh *.png"
    echo ""
    echo "  2. Inspect metrics JSON:"
    echo "     cat $CHECKPOINT_DIR/metrics.json | python -m json.tool"
    echo ""
    echo "  3. Load trained model for inference:"
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
