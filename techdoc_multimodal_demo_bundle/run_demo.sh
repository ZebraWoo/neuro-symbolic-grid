#!/usr/bin/env bash
# Run unified tech-doc demo (multimodal + neuro-symbolic) from bundle root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

EPOCHS="${EPOCHS:-40}"
BATCH="${BATCH_SIZE:-64}"
BRANCH="${BRANCH:-both}"

echo "[INFO] branch=$BRANCH epochs=$EPOCHS batch=$BATCH"
python demo/techdoc_framework_demo.py \
  --branch "$BRANCH" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --output-dir demo/techdoc_results

python demo/plot_techdoc_framework_demo.py \
  --history-file demo/techdoc_results/techdoc_history.json \
  --output-dir demo/techdoc_results

echo "[OK] Results: demo/techdoc_results/techdoc_history.json"
echo "[OK] Plot:    demo/techdoc_results/techdoc_framework_curves.png"
