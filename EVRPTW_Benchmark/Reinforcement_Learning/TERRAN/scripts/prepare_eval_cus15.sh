#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=(conda run -n maojie python)

NUM_INSTANCES="${1:-200}"
SEED="${2:-20260522}"
SAVE_PATH="${3:-EVRPTW_Dataset/AC_v1/AC_Small_15}"

"${PYTHON_BIN[@]}" -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.prepare_eval_data \
  --config-path EVRPTW_Dataset_Generator/configs/amazon_hierarchy.yaml \
  --save-path "$SAVE_PATH" \
  --num-instances "$NUM_INSTANCES" \
  --num-customers 15 \
  --num-charging-stations 3 \
  --num-regions 8 \
  --mother-num-customers 5000 \
  --mother-num-charging-stations 120 \
  --region-reuse-limit 200 \
  --seed "$SEED"
