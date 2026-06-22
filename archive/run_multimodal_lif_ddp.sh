#!/bin/bash
# Multimodal + LIF on GPUs 0,1,2,3 (DDP)

set -e
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC="${NPROC:-4}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"

echo "[INFO] DDP multimodal+LIF | CUDA=$CUDA_VISIBLE_DEVICES | nproc=$NPROC"

torchrun --standalone --nproc_per_node="$NPROC" train_control_ddp.py \
  --model-type lif \
  --use-lif \
  --zones ERCOT_zone_1_ CAISO_zone_1_ MISO_zone_1_ \
  --seq-len 96 --stride 96 \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr 0.001 \
  --num-workers 4

# rank0 saves training_history_psml_ddp_lif.json; plot on rank0 only happens in script
python plot_control_psml.py \
  --history-file outputs/training_history_psml_ddp_lif.json \
  --output-dir outputs
cp -f outputs/psml_load_pred_curves.png outputs/psml_load_pred_curves_lif_ddp.png 2>/dev/null || true

echo "[OK] checkpoints/control_model_psml_ddp_lif_best.pth"
