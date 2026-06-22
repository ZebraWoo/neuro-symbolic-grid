#!/bin/bash
# ============================================================================
# Run all experiments for the paper:
# "A Neuro-symbolic Closed-loop Learning Framework for
#  Intelligent Power Grid Decision Support"
#
# Usage:
#   bash experiments/run_all_experiments.sh          # Full run
#   bash experiments/run_all_experiments.sh --quick  # Quick test (few epochs, few zones)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default settings
EPOCHS=50
BATCH_SIZE=32
ZONES_PER_SPLIT=0  # 0 = all
DEVICE="cuda"

# Parse flags
QUICK=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK=true
            EPOCHS=5
            BATCH_SIZE=16
            ZONES_PER_SPLIT=2
            shift
            ;;
        --cpu)
            DEVICE="cpu"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quick] [--cpu]"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "  Paper Experiment Suite"
echo "  Neuro-symbolic Decision Support"
echo "============================================"
echo "Mode: $([ "$QUICK" = true ] && echo 'QUICK TEST' || echo 'FULL')"
echo "Epochs: $EPOCHS | Batch: $BATCH_SIZE | Zones/Split: $ZONES_PER_SPLIT"
echo "Device: $DEVICE"
echo "============================================"

# ------------------------------------------------------------------
# Step 1: Generate decision labels
# ------------------------------------------------------------------
echo ""
echo "=== Step 1: Generate Decision Labels ==="
python experiments/label_decision_intents.py \
    --seq-len 96 --stride 96 \
    $([ "$QUICK" = true ] && echo "--zones-per-split 2" || echo "")

# ------------------------------------------------------------------
# Step 2: Train baselines (E1)
# ------------------------------------------------------------------
echo ""
echo "=== Step 2: Train Baselines (E1) ==="
BASELINES=("lstm" "transformer" "tcn" "snn_lif" "snn_izh")

if [ "$QUICK" = true ]; then
    # Quick: just run LSTM + SNN-LIF
    BASELINES=("lstm" "snn_lif")
fi

for model in "${BASELINES[@]}"; do
    echo ""
    echo "--- Training baseline: $model ---"
    python experiments/exp_e1_decision_accuracy.py \
        --model "$model" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --zones-per-split "$ZONES_PER_SPLIT" \
        --device "$DEVICE" \
        --checkpoint-dir checkpoints \
        --output-dir outputs
done

# ------------------------------------------------------------------
# Step 3: Train Ours (full model)
# ------------------------------------------------------------------
echo ""
echo "=== Step 3: Train Ours Full Model ==="
python experiments/exp_ours_full.py \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --zones-per-split "$ZONES_PER_SPLIT" \
    --device "$DEVICE" \
    --tag ours_full \
    --checkpoint-dir checkpoints \
    --output-dir outputs

# ------------------------------------------------------------------
# Step 4: Ablation Study (E6)
# ------------------------------------------------------------------
echo ""
echo "=== Step 4: Ablation Study (E6) ==="

declare -A ABLATIONS=(
    ["ours_no_sym"]="--no-symbolic"
    ["ours_no_phys"]="--no-physics"
    ["ours_no_cl"]="--no-closed-loop"
    ["ours_single_comp"]="--no-multi-comp"
    ["ours_no_spike"]="--no-spike"
)

if [ "$QUICK" = true ]; then
    # Quick: only run 2 ablations
    ABLATIONS=( ["ours_no_sym"]="--no-symbolic" ["ours_no_spike"]="--no-spike" )
fi

for tag in "${!ABLATIONS[@]}"; do
    echo ""
    echo "--- Ablation: $tag ---"
    python experiments/exp_ours_full.py \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --zones-per-split "$ZONES_PER_SPLIT" \
        --device "$DEVICE" \
        --tag "$tag" \
        --checkpoint-dir checkpoints \
        --output-dir outputs \
        ${ABLATIONS[$tag]}
done

# ------------------------------------------------------------------
# Step 5: Evaluation Experiments (E2-E5, E7)
# These use pre-trained checkpoints for inference only
# ------------------------------------------------------------------
echo ""
echo "=== Step 5: Evaluation Experiments ==="

# E2: Rule Satisfaction Rate
echo "--- E2: Rule Satisfaction ---"
python experiments/exp_e2_rule_satisfaction.py \
    --checkpoint checkpoints/ours_full_best.pth \
    --output-dir outputs \
    $([ "$QUICK" = true ] && echo "--max-samples 100" || echo "")

# E3: Physics Violation
echo "--- E3: Physics Violation ---"
python experiments/exp_e3_physics_violation.py \
    --checkpoint checkpoints/ours_full_best.pth \
    --output-dir outputs \
    $([ "$QUICK" = true ] && echo "--max-samples 100" || echo "")

# E4: Closed-loop Convergence
echo "--- E4: Closed-loop ---"
python experiments/exp_e4_closed_loop.py \
    --checkpoint checkpoints/ours_full_best.pth \
    --output-dir outputs \
    $([ "$QUICK" = true ] && echo "--max-samples 50" || echo "")

# E5: Robustness
echo "--- E5: Robustness ---"
python experiments/exp_e5_robustness.py \
    --checkpoint checkpoints/ours_full_best.pth \
    --output-dir outputs \
    $([ "$QUICK" = true ] && echo "--max-samples 100" || echo "")

# E7: Case Study
echo "--- E7: Case Study ---"
python experiments/exp_e7_case_study.py \
    --checkpoint checkpoints/ours_full_best.pth \
    --output-dir outputs

# ------------------------------------------------------------------
# Step 6: Generate Results Summary
# ------------------------------------------------------------------
echo ""
echo "=== Step 6: Results Summary ==="
python experiments/plot_results.py --output-dir outputs/paper_figures/

echo ""
echo "============================================"
echo "  All experiments complete!"
echo "  Check outputs/ for results"
echo "  Check outputs/paper_figures/ for figures"
echo "============================================"
