#!/bin/bash
# ============================================================================
# Launch all 8 GORS paper experiments in parallel (1 per GPU)
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="logs/experiments"
mkdir -p "$LOG_DIR"

EPOCHS=50
BATCH_SIZE=32

echo "============================================"
echo "  Launching 8 GORS Paper Experiments"
echo "  Epochs: $EPOCHS | Batch: $BATCH_SIZE"
echo "  Logs: $LOG_DIR/"
echo "============================================"
echo ""

launch() {
    local gpu=$1
    local model=$2
    local tag=$3
    shift 3
    local extra_args="$@"

    local logfile="$LOG_DIR/${tag}.log"
    echo "  GPU$gpu → $tag (log: $logfile)"

    CUDA_VISIBLE_DEVICES=$gpu \
    nohup python experiments/train_all.py \
        --model "$model" \
        --tag "$tag" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        $extra_args \
        > "$logfile" 2>&1 &
}

# ---- Baselines (Table I) ----
launch 0 lstm        g1_lstm
launch 1 transformer g1_transformer
launch 2 tcn         g1_tcn
launch 3 snn_lif     g1_snn_lif

# ---- GORS Full + Ablations (Table I & II) ----
launch 4 gors gors_full
launch 5 gors gors_no_sym   --no-symbolic
launch 6 gors gors_no_phy   --no-physics
launch 7 gors gors_no_fb    --no-feedback

echo ""
echo "All 8 experiments launched. Monitor:"
echo "  watch -n 5 'tail -5 $LOG_DIR/*.log'"
echo "  python experiments/check_progress.py"
