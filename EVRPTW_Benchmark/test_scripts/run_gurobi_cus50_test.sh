#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cus50_common.sh"

readonly SOLVER_DIR="${REPO_ROOT}/EVRPTW_Benchmark/Exact/Gurobi_Solver"
base_save_path="${RESULTS_ROOT}/${TEST_RESULT_RELATIVE_ROOT}/Gurobi_Solver_cs2_2h"
partition_output SAVE_PATH "${base_save_path}"
readonly SAVE_PATH

prepare_output "${SAVE_PATH}"
print_contract "Gurobi exact (cs_copies=2, threads=1)" "test1_new_seed_same_cities" "${TEST_INDEX}" "${SAVE_PATH}"

run_python "${SOLVER_DIR}/run_gurobi.py" \
  --dataset_path "${TEST_INDEX}" \
  --family_root "${FAMILY_ROOT}" \
  --save_path "${SAVE_PATH}" \
  --scales "${CUS_SCALE}" \
  --time_limit_s "${TIME_LIMIT_S}" \
  --checkpoints_s "${CHECKPOINTS_S}" \
  --cs_copies 2 \
  --mip_gap 0 \
  --workers "${WORKERS}" \
  --threads 1 \
  --output_flag 0 \
  --no-tie_break_vehicle_count \
  --skip_completed \
  --save_traceback \
  --verbose \
  "${EXACT_SELECTION_ARGS[@]}"
