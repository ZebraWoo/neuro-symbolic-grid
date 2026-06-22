#!/bin/bash

###############################################################################
# Training metrics visualization script
# Plots training curves for AUC, Precision, Recall, F1, and Silhouette
###############################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

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

# Default parameters
METRICS_FILE="./results/metrics.json"
OUTPUT_DIR="./results"

# Parse CLI arguments
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
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --metrics-file PATH    Metrics JSON path (default: ./results/metrics.json)"
            echo "  --output-dir PATH      Output directory (default: ./results)"
            echo "  --help                 Show help"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

main() {
    echo ""
    echo "===================================================================="
    echo "Training Metrics Visualization"
    echo "===================================================================="
    echo ""
    
    # Activate conda environment
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate snn 2>/dev/null || {
            print_warning "Failed to activate snn env; using current environment"
        }
    fi
    
    print_info "Checking metrics file..."
    
    if [ ! -f "$METRICS_FILE" ]; then
        print_error "Metrics file not found: $METRICS_FILE"
        echo ""
        echo "Please run training first:"
        echo "   bash train_with_metrics.sh"
        exit 1
    fi
    
    print_success "Metrics file found"
    echo ""
    
    print_info "Generating plots..."
    echo ""
    
    # Run plotting script
    python plot_metrics.py \
        --metrics-file "$METRICS_FILE" \
        --output-dir "$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo ""
        print_success "Visualization completed"
        echo ""
        echo "Output directory: $OUTPUT_DIR"
        echo ""
        echo "Generated plot files:"
        echo "   1) loss_curve.png"
        echo "      - Train and validation loss curves"
        echo "      - Learning trend overview"
        echo ""
        echo "   2) metrics_curve.png"
        echo "      - AUC (Area Under Curve) for classification quality"
        echo "      - Precision: positive prediction accuracy"
        echo "      - Recall: positive sample coverage"
        echo "      - F1-Score: harmonic mean of Precision and Recall"
        echo ""
        echo "   3) clustering_metrics.png"
        echo "      - Silhouette Score for clustering quality ([-1, 1])"
        echo "      - Proximity score for anomaly detection"
        echo ""
        echo "   4) summary_report.png"
        echo "      - Training summary statistics"
        echo "      - Best and final values of all metrics"
        echo ""
        echo "List output files:"
        echo "   cd $OUTPUT_DIR && ls -lh *.png"
        echo ""
    else
        print_error "Visualization failed"
        exit 1
    fi
}

main
