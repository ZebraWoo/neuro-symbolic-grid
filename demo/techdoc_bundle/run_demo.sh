#!/usr/bin/env bash
# Run unified tech-doc demo (multimodal + neuro-symbolic) with real PSML data.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

EPOCHS="${EPOCHS:-40}"
BATCH="${BATCH_SIZE:-64}"
BRANCH="${BRANCH:-both}"
DATA_FRACTION="${DATA_FRACTION:-0.33}"

echo "[INFO] branch=$BRANCH epochs=$EPOCHS batch=$BATCH data_fraction=$DATA_FRACTION"
python demo/techdoc_framework_demo.py \
  --branch "$BRANCH" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --data-fraction "$DATA_FRACTION" \
  --output-dir demo/techdoc_results

python demo/plot_techdoc_framework_demo.py

echo "[OK] Results: demo/techdoc_results/techdoc_history.json"
echo "[OK] Plot:    demo/techdoc_results/techdoc_framework_curves.png"
