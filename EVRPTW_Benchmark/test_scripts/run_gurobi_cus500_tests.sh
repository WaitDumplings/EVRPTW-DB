#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cus500_common.sh"

readonly SOLVER_DIR="${REPO_ROOT}/EVRPTW_Benchmark/Exact/Gurobi_Solver"
for track_index in "${!CUS500_TRACK_IDS[@]}"; do
  track_id="${CUS500_TRACK_IDS[${track_index}]}"
  test_index="${DATASET_ROOT}/generation_plan/${CUS500_RELATIVE_INDICES[${track_index}]}"
  require_test_index "${test_index}"
  base_save_path="${RESULTS_ROOT}/core/${track_id}/Gurobi_Solver_Cus500_cs2_2h"
  partition_output save_path "${base_save_path}"
  prepare_output "${save_path}"
  print_contract "Gurobi exact (cs_copies=2, threads=1)" "${track_id}" "${test_index}" "${save_path}"

  run_python "${SOLVER_DIR}/run_gurobi.py" \
    --dataset_path "${test_index}" \
    --family_root "${FAMILY_ROOT}" \
    --save_path "${save_path}" \
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
done
