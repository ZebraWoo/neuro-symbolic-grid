#!/bin/bash
# Main experiment: 3 PSML zones, 50 epochs, GPU, full CSV (no MAX_ROWS).
# Outputs:
#   checkpoints/control_model_psml.pth
#   checkpoints/control_model_psml_best.pth
#   outputs/training_history_psml.json
#   outputs/psml_load_pred_curves.png

set -e

cd "$(dirname "$0")"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
err()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# -------- defaults (override via env) --------
DATA_ROOT="${DATA_ROOT:-/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable}"
ZONES="${ZONES:-ERCOT_zone_1_ CAISO_zone_1_ MISO_zone_1_}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.001}"
SEQ_LEN="${SEQ_LEN:-96}"
STRIDE="${STRIDE:-96}"
DEVICE="${DEVICE:-cuda}"
VAL_RATIO="${VAL_RATIO:-0.1}"

# GPU selection: export CUDA_VISIBLE_DEVICES=0 if needed
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

info "=========================================="
info "PSML multimodal main experiment"
info "=========================================="
info "data_root: $DATA_ROOT"
info "zones:     $ZONES"
info "epochs:    $EPOCHS  batch: $BATCH_SIZE  lr: $LR"
info "seq_len:   $SEQ_LEN  stride: $STRIDE"
info "device:    $DEVICE  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
info "=========================================="

command -v python >/dev/null || err "python not found"
python -c "import torch; print('PyTorch', torch.__version__, '| cuda:', torch.cuda.is_available())" \
  || err "PyTorch not installed"

mkdir -p checkpoints outputs logs

LOG_FILE="logs/main_experiment_$(date +%Y%m%d_%H%M%S).log"
info "Logging to $LOG_FILE"

# shellcheck disable=SC2086
python train_control.py \
  --data-root "$DATA_ROOT" \
  --zones $ZONES \
  --seq-len "$SEQ_LEN" \
  --stride "$STRIDE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --device "$DEVICE" \
  --val-ratio "$VAL_RATIO" \
  2>&1 | tee "$LOG_FILE"

ok "Training finished"

info "Plotting curves..."
python plot_control_psml.py \
  --history-file outputs/training_history_psml.json \
  --output-dir outputs

info "=========================================="
ok "Artifacts for report:"
ok "  checkpoints/control_model_psml.pth      (last epoch)"
ok "  checkpoints/control_model_psml_best.pth (best val RMSE)"
ok "  outputs/training_history_psml.json"
ok "  outputs/psml_load_pred_curves.png"
ok "  $LOG_FILE"
info "=========================================="
