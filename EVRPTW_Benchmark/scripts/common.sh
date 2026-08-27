#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "common.sh is a shared library; run a shell below Gurobi/, ALNS/, or VNSTS/." >&2
  exit 2
fi

readonly BENCHMARK_SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

run_frozen_test() {
  if (($# != 5)); then
    echo "run_frozen_test requires: solver scale track relative-index result-root" >&2
    return 2
  fi

  local solver_kind="$1"
  local CUS_SCALE="$2"
  local track_id="$3"
  local relative_index="$4"
  local result_relative_root="$5"
  local TEST_VIEW_COUNT="500"

  if [[ "${EVRPTW_NOHUP_CHILD:-0}" == "1" && -n "${EVRPTW_NOHUP_EXIT_FILE:-}" ]]; then
    trap 'printf "%s\n" "$?" > "${EVRPTW_NOHUP_EXIT_FILE}"' EXIT
  fi

  # Reuse the validated restore discovery, shard, resume, and timing contract.
  source "${BENCHMARK_SCRIPT_ROOT}/../test_scripts/common.sh"

  local test_index="${DATASET_ROOT}/generation_plan/${relative_index}"
  require_test_index "${test_index}"

  local solver_dir
  local solver_label
  case "${solver_kind}" in
    Gurobi)
      solver_dir="Gurobi_Solver_${CUS_SCALE}_cs2_2h"
      solver_label="Gurobi exact (cs_copies=2, threads=1)"
      ;;
    ALNS)
      solver_dir="ALNS_Solver_${CUS_SCALE}_2h"
      solver_label="ALNS"
      ;;
    VNSTS)
      solver_dir="VNS_TS_Solver_${CUS_SCALE}_2h"
      solver_label="VNS-TS adaptive-fast"
      ;;
    *)
      echo "Unknown solver kind: ${solver_kind}" >&2
      return 2
      ;;
  esac

  local base_save_path="${RESULTS_ROOT}/${result_relative_root}/${solver_dir}"
  local save_path
  partition_output save_path "${base_save_path}"

  local foreground="${EVRPTW_FOREGROUND:-0}"
  if [[ "${foreground}" != "0" && "${foreground}" != "1" ]]; then
    echo "EVRPTW_FOREGROUND must be 0 or 1, got: ${foreground}" >&2
    return 2
  fi
  if [[ "${DRY_RUN}" == "0" && "${foreground}" == "0" && "${EVRPTW_NOHUP_CHILD:-0}" != "1" ]]; then
    command -v nohup >/dev/null 2>&1 || {
      echo "nohup is required for detached benchmark launch." >&2
      return 2
    }
    local launcher_path
    launcher_path="$(cd -- "$(dirname -- "$0")" && pwd)/$(basename -- "$0")"
    local job_slug="${solver_kind,,}_${CUS_SCALE,,}_${track_id}"
    if [[ -n "${RESULT_PARTITION_DIR}" ]]; then
      job_slug+="_${RESULT_PARTITION_DIR}"
    fi
    local nohup_log_root="${EVRPTW_NOHUP_LOG_ROOT:-logs/benchmarks/nohup}"
    local job_dir="${nohup_log_root}/${job_slug}"
    local log_file="${job_dir}/run.log"
    local pid_file="${job_dir}/pid.txt"
    local exit_file="${job_dir}/exit_code.txt"
    mkdir -p "${job_dir}"

    if [[ -f "${pid_file}" ]]; then
      local existing_pid=""
      local existing_command=""
      read -r existing_pid < "${pid_file}" || true
      if [[ "${existing_pid}" =~ ^[1-9][0-9]*$ ]]; then
        existing_command="$(ps -p "${existing_pid}" -o args= 2>/dev/null || true)"
      fi
      if [[ "${existing_command}" == *"${launcher_path}"* ]]; then
        echo "Benchmark is already running."
        echo "  pid:  ${existing_pid}"
        echo "  log:  ${log_file}"
        echo "  exit: ${exit_file}"
        return 0
      fi
    fi

    printf '%s\n' "RUNNING" > "${exit_file}"
    printf '\n[%s] starting %s %s %s\n' "$(date -Is)" "${solver_kind}" "${CUS_SCALE}" "${track_id}" >> "${log_file}"
    nohup env EVRPTW_NOHUP_CHILD=1 EVRPTW_NOHUP_EXIT_FILE="${exit_file}" bash "${launcher_path}" >> "${log_file}" 2>&1 < /dev/null &
    local launched_pid=$!
    printf '%s\n' "${launched_pid}" > "${pid_file}"
    echo "Benchmark started with nohup."
    echo "  pid:  ${launched_pid}"
    echo "  log:  ${log_file}"
    echo "  exit: ${exit_file}"
    echo "  data: ${DATASET_ROOT}"
    return 0
  fi

  prepare_output "${save_path}"
  print_contract "${solver_label}" "${track_id}" "${test_index}" "${save_path}"

  case "${solver_kind}" in
    Gurobi)
      run_python "${REPO_ROOT}/EVRPTW_Benchmark/Exact/Gurobi_Solver/run_gurobi.py" \
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
      ;;
    ALNS)
      run_python "${REPO_ROOT}/EVRPTW_Benchmark/MetaHeuristics/ALNS_Solver/run_alns.py" \
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
        --skip_completed \
        --save_traceback \
        --verbose \
        "${META_SELECTION_ARGS[@]}"
      ;;
    VNSTS)
      run_python "${REPO_ROOT}/EVRPTW_Benchmark/MetaHeuristics/VNS_TS_Solver/run_vns_ts.py" \
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
      ;;
  esac
}
