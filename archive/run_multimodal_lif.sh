#!/bin/bash
# Phase-2: PSML multimodal + Temporal LIF (4 modalities, load forecasting)
# Quick test:  MAX_ROWS=8000 EPOCHS=5 bash run_multimodal_lif.sh
# Main run:    bash run_multimodal_lif.sh
# 4-GPU DDP:   bash run_multimodal_lif_ddp.sh

set -e
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ROOT="${DATA_ROOT:-/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable}"
ZONES="${ZONES:-ERCOT_zone_1_ CAISO_zone_1_ MISO_zone_1_}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.001}"
SEQ_LEN="${SEQ_LEN:-96}"
STRIDE="${STRIDE:-96}"
DEVICE="${DEVICE:-cuda}"
MAX_ROWS="${MAX_ROWS:-}"

EXTRA=()
[ -n "$MAX_ROWS" ] && EXTRA+=(--max-rows-per-zone "$MAX_ROWS")

echo "[INFO] Multimodal + Temporal LIF | zones=$ZONES | epochs=$EPOCHS | device=$DEVICE"

python train_control.py \
  --model-type lif \
  --use-lif \
  --data-root "$DATA_ROOT" \
  --zones $ZONES \
  --seq-len "$SEQ_LEN" --stride "$STRIDE" \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "$LR" \
  --device "$DEVICE" \
  "${EXTRA[@]}"

python plot_control_psml.py \
  --history-file outputs/training_history_psml_lif.json \
  --output-dir outputs

cp -f outputs/psml_load_pred_curves.png outputs/psml_load_pred_curves_lif.png 2>/dev/null || true

echo "[OK] checkpoints/control_model_psml_lif_best.pth"
echo "[OK] outputs/training_history_psml_lif.json"
echo "[OK] outputs/psml_load_pred_curves_lif.png"
