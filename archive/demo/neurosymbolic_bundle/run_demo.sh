#!/usr/bin/env bash
# Run unified neuro-symbolic grid control demo (SNN + control joint training).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH_SIZE:-64}"
BRANCH="${BRANCH:-both}"
DATA_FRACTION="${DATA_FRACTION:-0.33}"

echo "[INFO] branch=$BRANCH epochs=$EPOCHS batch=$BATCH data_fraction=$DATA_FRACTION"
python demo/neurosymbolic_grid_demo.py \
  --branch "$BRANCH" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --data-fraction "$DATA_FRACTION" \
  --output-dir demo/ns_results

python demo/plot_neurosymbolic_demo.py

echo "[OK] Results: demo/ns_results/ns_history.json"
echo "[OK] Plots:   demo/ns_results/ns_framework_curves.png"
echo "[OK]          demo/ns_results/ns_penalty_curve.png"
echo "[OK]          demo/ns_results/ns_truth_curve.png"
echo "[OK]          demo/ns_results/summary_report.png"
