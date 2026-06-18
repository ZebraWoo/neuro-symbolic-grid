#!/bin/bash

###############################################################################
# Spikformer pretraining script with full metrics
# Supports AUC, Precision, Recall, F1, and Silhouette
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Colored logs
print_info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[OK] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

# Check environment
check_environment() {
    print_info "Checking Python environment..."
    
    # Activate conda environment
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate snn 2>/dev/null || {
            print_error "Failed to activate conda env: snn"
            exit 1
        }
        print_success "Activated conda env: snn"
    else
        print_warning "Conda not found, using current shell environment"
    fi
    
    # Check Python version
    python_version=$(python --version 2>&1 | awk '{print $2}')
    print_success "Python version: $python_version"
    
    # Check required packages
    for package in torch numpy matplotlib seaborn sklearn tqdm; do
        if python -c "import $package" 2>/dev/null; then
            print_success "$package installed"
        else
            print_error "$package not installed"
        fi
    done
}

# Show configuration
show_config() {
    print_info "Training configuration:"
    echo ""
    echo "  Data:"
    echo "    - Data root: ${DATA_ROOT}"
    echo "    - Zones: ${ZONES[@]}"
    echo ""
    echo "  Model:"
    echo "    - Hidden dim: ${HIDDEN_DIM}"
    echo "    - Embedding dim: ${EMBEDDING_DIM}"
    echo ""
    echo "  Training:"
    echo "    - Batch size: ${BATCH_SIZE}"
    echo "    - Sequence length: ${SEQ_LEN}"
    echo "    - Learning rate: ${LEARNING_RATE}"
    echo "    - Epochs: ${NUM_EPOCHS}"
    echo "    - Device: ${DEVICE}"
    echo ""
    echo "  Output:"
    echo "    - Checkpoint dir: ${CHECKPOINT_DIR}"
    echo ""
}

# Limit BLAS threads to avoid OpenBLAS crashes
configure_thread_limits() {
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-32}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-32}"
    export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-32}"
    print_info "Thread limits set: OPENBLAS/OMP/MKL/NUMEXPR=${OPENBLAS_NUM_THREADS}"
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
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --data-root PATH           Dataset root path"
            echo "  --zones ZONE1 ZONE2 ...    Grid zone list"
            echo "  --batch-size N             Batch size (default: 16)"
            echo "  --seq-len N                Sequence length (default: 720)"
            echo "  --hidden-dim N             Hidden dimension (default: 128)"
            echo "  --embedding-dim N          Embedding dimension (default: 64)"
            echo "  --learning-rate LR         Learning rate (default: 0.001)"
            echo "  --num-epochs N             Number of epochs (default: 30)"
            echo "  --checkpoint-dir PATH      Checkpoint directory"
            echo "  --device DEVICE            Device: auto/cuda/cpu (default: auto)"
            echo "  --help                     Show help"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Main function
main() {
    echo ""
    echo "===================================================================="
    echo "Spikformer Pretraining with Full Metrics"
    echo "===================================================================="
    echo ""
    
    # Check environment
    check_environment
    configure_thread_limits
    echo ""
    
    # Show configuration
    show_config
    echo ""
    
    # Check data directory
    if [ ! -d "$DATA_ROOT" ]; then
        print_error "Data directory does not exist: $DATA_ROOT"
        exit 1
    fi
    print_success "Data directory found"
    echo ""
    
    # Start training
    print_info "Starting training..."
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
        print_success "Training completed"
        echo ""
        echo "Generated outputs:"
        echo "   - Checkpoints: $CHECKPOINT_DIR"
        echo "   - Plots: ./results/"
        echo ""
        echo "Expected plot files:"
        echo "   1) loss_curve.png - Train/validation loss"
        echo "   2) metrics_curve.png - AUC/Precision/Recall/F1"
        echo "   3) clustering_metrics.png - Silhouette and proximity"
        echo "   4) summary_report.png - Training summary"
        echo ""
        echo "Quick plotting command:"
        echo "   bash plot_metrics.sh"
        echo ""
    else
        print_error "Training failed"
        exit 1
    fi
}

# Run main
main
