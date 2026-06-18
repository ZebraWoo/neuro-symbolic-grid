#!/usr/bin/env bash
# Run 3-layer SNN-MLP anomaly classification demo with real PSML data.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

EPOCHS="${EPOCHS:-12}"
BATCH="${BATCH_SIZE:-64}"
DATA_FRACTION="${DATA_FRACTION:-0.33}"

echo "[INFO] epochs=$EPOCHS batch=$BATCH data_fraction=$DATA_FRACTION"
python demo/snn_mlp_demo.py \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --data-fraction "$DATA_FRACTION" \
  --output-dir demo/snn_results

python demo/plot_snn_mlp_demo.py

echo "[OK] Results: demo/snn_results/snn_history.json"
echo "[OK] Plots:   demo/snn_results/loss_curve.png"
echo "[OK]          demo/snn_results/accuracy_curve.png"
echo "[OK]          demo/snn_results/summary_report.png"
