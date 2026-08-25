#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cus500_common.sh"

readonly SOLVER_DIR="${REPO_ROOT}/EVRPTW_Benchmark/MetaHeuristics/VNS_TS_Solver"
for track_index in "${!CUS500_TRACK_IDS[@]}"; do
  track_id="${CUS500_TRACK_IDS[${track_index}]}"
  test_index="${DATASET_ROOT}/generation_plan/${CUS500_RELATIVE_INDICES[${track_index}]}"
  require_test_index "${test_index}"
  base_save_path="${RESULTS_ROOT}/core/${track_id}/VNS_TS_Solver_Cus500_2h"
  partition_output save_path "${base_save_path}"
  prepare_output "${save_path}"
  print_contract "VNS-TS adaptive-fast" "${track_id}" "${test_index}" "${save_path}"

  run_python "${SOLVER_DIR}/run_vns_ts.py" \
    --dataset_path "${test_index}" \
    --family_root "${FAMILY_ROOT}" \
    --save_path "${save_path}" \
    --scales "${CUS_SCALE}" \
    --time_limit_s "${TIME_LIMIT_S}" \
    --checkpoints_s "${CHECKPOINTS_S}" \
    --seed "${BASE_SEED}" \
    --num_workers "${WORKERS}" \
    --max_in_flight "${MAX_IN_FLIGHT}" \
    --csv_flush_interval "${CSV_FLUSH_INTERVAL}" \
    --predefine_route_number 3 \
    --eta_feas 20 \
    --tabu_iter 10 \
    --tabu_tenure 30 \
    --k_max 15 \
    --search_mode fast \
    --move_candidate_limit 40 \
    --route_neighbor_limit 4 \
    --position_neighbor_limit 4 \
    --exchange_neighbor_limit 6 \
    --station_candidate_limit 5 \
    --skip_completed \
    --save_traceback \
    --verbose \
    "${META_SELECTION_ARGS[@]}"
done
