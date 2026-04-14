#!/usr/bin/env bash
set -euo pipefail

# One-command pipeline for Chinese-header power-grid pseudo data.
# Usage:
#   bash experiments/pseudo_data_tools/run_pseudo_pipeline.sh "data/raw"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input_file_or_dir> [output_dir]"
  echo "Example: $0 data/raw"
  exit 1
fi

INPUT_PATH="$1"
OUTPUT_ROOT="${2:-data}"

PROFILE_DIR="${OUTPUT_ROOT}/processed/pseudo_profile"
SYN_FILE="${OUTPUT_ROOT}/pseudo/synthetic_grid_data.csv"
VALIDATION_FILE="${PROFILE_DIR}/validation_report.json"

echo "[1/3] 数据画像..."
python experiments/pseudo_data_tools/01_data_profile_cn.py \
  --input "${INPUT_PATH}" \
  --out-dir "${PROFILE_DIR}"

echo "[2/3] 伪数据生成..."
python experiments/pseudo_data_tools/02_generate_synthetic_cn.py \
  --input "${INPUT_PATH}" \
  --output "${SYN_FILE}" \
  --samples-per-hour 30 \
  --seed 42

echo "[3/3] 质量校验..."
python experiments/pseudo_data_tools/03_validate_synthetic_cn.py \
  --real "${INPUT_PATH}" \
  --synthetic "${SYN_FILE}" \
  --out "${VALIDATION_FILE}"

echo "完成。"
echo "Synthetic file: ${SYN_FILE}"
echo "Validation report: ${VALIDATION_FILE}"
