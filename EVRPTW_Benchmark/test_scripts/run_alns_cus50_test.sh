#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cus50_common.sh"

readonly SOLVER_DIR="${REPO_ROOT}/EVRPTW_Benchmark/MetaHeuristics/ALNS_Solver"
base_save_path="${RESULTS_ROOT}/${TEST_RESULT_RELATIVE_ROOT}/ALNS_Solver_2h"
partition_output SAVE_PATH "${base_save_path}"
readonly SAVE_PATH

prepare_output "${SAVE_PATH}"
print_contract "ALNS" "test1_new_seed_same_cities" "${TEST_INDEX}" "${SAVE_PATH}"

run_python "${SOLVER_DIR}/run_alns.py" \
  --dataset_path "${TEST_INDEX}" \
  --family_root "${FAMILY_ROOT}" \
  --save_path "${SAVE_PATH}" \
  --scales "${CUS_SCALE}" \
  --time_limit_s "${TIME_LIMIT_S}" \
  --checkpoints_s "${CHECKPOINTS_S}" \
  --seed "${BASE_SEED}" \
  --num_workers "${WORKERS}" \
  --max_in_flight "${MAX_IN_FLIGHT}" \
  --csv_flush_interval "${CSV_FLUSH_INTERVAL}" \
  --skip_completed \
  --save_traceback \
  --verbose \
  "${META_SELECTION_ARGS[@]}"
