#!/usr/bin/env bash
set -euo pipefail

# Cus50 test benchmark: 30 independent Gurobi processes, one thread/model,
# two charging-station copies, and a two-hour wall-clock limit per instance.
#
# Usage:
#   bash EVRPTW_Benchmark/Exact/Gurobi_Solver/run_cus50_test_30proc.sh
#   bash EVRPTW_Benchmark/Exact/Gurobi_Solver/run_cus50_test_30proc.sh /path/to/output

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

DATASET_PATH="${REPO_ROOT}/EVRPTW_Dataset/Instances_v1/us_11city/generation_plan/compatibility_cus50/test/test1_new_seed_same_cities/view_index.parquet"
FAMILY_ROOT="${REPO_ROOT}/EVRPTW_Dataset/Instances_v1/us_11city/materialized/families"
SAVE_PATH="${1:-${REPO_ROOT}/EVRPTW_Benchmark/results/CLE_EVRPTW_v1/compatibility_cus50/test1/Gurobi_Solver_cs2_2h}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Missing Cus50 test view index: ${DATASET_PATH}" >&2
  exit 1
fi

if [[ ! -d "${FAMILY_ROOT}" ]]; then
  echo "Missing Stage-2 family root: ${FAMILY_ROOT}" >&2
  exit 1
fi

mkdir -p "${SAVE_PATH}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}/EVRPTW_Core:${REPO_ROOT}/EVRPTW_Dataset_Generator/src:${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"

conda run -n maojie --no-capture-output \
  python "${SCRIPT_DIR}/run_gurobi.py" \
  --dataset_path "${DATASET_PATH}" \
  --family_root "${FAMILY_ROOT}" \
  --save_path "${SAVE_PATH}" \
  --time_limit_s 7200 \
  --checkpoints_s 60,300,900,3600,7200 \
  --cs_copies 2 \
  --workers 30 \
  --threads 1 \
  --no-tie_break_vehicle_count \
  --skip_completed \
  --save_traceback \
  --verbose
