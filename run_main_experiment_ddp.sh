#!/bin/bash
# Multi-GPU DDP. Use physical cards 0,1,2,3:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 bash run_main_experiment_ddp.sh

set -e
cd "$(dirname "$0")"

# Physical GPU ids (comma-separated). Empty = all visible GPUs on the machine.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC="${NPROC:-4}"
DATA_ROOT="${DATA_ROOT:-/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable}"
ZONES="${ZONES:-ERCOT_zone_1_ CAISO_zone_1_ MISO_zone_1_}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"   # per GPU → global batch = BATCH_SIZE * NPROC
LR="${LR:-0.001}"
SEQ_LEN="${SEQ_LEN:-96}"
STRIDE="${STRIDE:-96}"

echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] torchrun nproc=$NPROC | per-GPU batch=$BATCH_SIZE | global≈$((BATCH_SIZE * NPROC))"

torchrun --standalone --nproc_per_node="$NPROC" train_control_ddp.py \
  --data-root "$DATA_ROOT" \
  --zones $ZONES \
  --seq-len "$SEQ_LEN" \
  --stride "$STRIDE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --num-workers 4

echo "[OK] checkpoints/control_model_psml_ddp_best.pth"
echo "[OK] outputs/training_history_psml_ddp.json"
