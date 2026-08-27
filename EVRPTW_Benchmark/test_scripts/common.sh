#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "common.sh is a shared library; run one of the run_*_test.sh scripts." >&2
  exit 2
fi

: "${CUS_SCALE:?CUS_SCALE must be set before sourcing common.sh}"
: "${TEST_VIEW_COUNT:?TEST_VIEW_COUNT must be set before sourcing common.sh}"
readonly CUS_SCALE
readonly TEST_VIEW_COUNT

readonly TEST_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${TEST_SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
readonly CHECKPOINTS_S="300,1800,3600,7200"
readonly TIME_LIMIT_S="7200"
readonly BASE_SEED="2026"

WORKERS="${EVRPTW_TEST_WORKERS:-30}"
if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVRPTW_TEST_WORKERS must be a positive integer, got: ${WORKERS}" >&2
  exit 2
fi
readonly WORKERS

if [[ -n "${EVRPTW_MAX_IN_FLIGHT:-}" ]]; then
  MAX_IN_FLIGHT="${EVRPTW_MAX_IN_FLIGHT}"
else
  MAX_IN_FLIGHT="$((WORKERS * 2))"
fi
if ! [[ "${MAX_IN_FLIGHT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVRPTW_MAX_IN_FLIGHT must be a positive integer, got: ${MAX_IN_FLIGHT}" >&2
  exit 2
fi
readonly MAX_IN_FLIGHT

readonly CSV_FLUSH_INTERVAL="${EVRPTW_CSV_FLUSH_INTERVAL:-25}"
readonly CONDA_ENV="${EVRPTW_CONDA_ENV:-maojie}"
readonly DRY_RUN="${EVRPTW_DRY_RUN:-0}"

if ! [[ "${TEST_VIEW_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TEST_VIEW_COUNT must be a positive integer, got: ${TEST_VIEW_COUNT}" >&2
  exit 2
fi
if ! [[ "${CSV_FLUSH_INTERVAL}" =~ ^[0-9]+$ ]]; then
  echo "EVRPTW_CSV_FLUSH_INTERVAL must be a non-negative integer." >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "EVRPTW_DRY_RUN must be 0 or 1, got: ${DRY_RUN}" >&2
  exit 2
fi

readonly CANONICAL_DATASET_RELATIVE_ROOT="EVRPTW_Dataset/Instances_v2/us_11city"
readonly SOURCE_DATASET_RELATIVE_ROOT="EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823"
dataset_candidates=(
  "${CANONICAL_DATASET_RELATIVE_ROOT}"
  "${SOURCE_DATASET_RELATIVE_ROOT}"
  "../evrptw_runtime/EVRPTW_Dataset/Instances_v2/us_11city"
  "../../evrptw_runtime/EVRPTW_Dataset/Instances_v2/us_11city"
  "../../../evrptw_runtime/EVRPTW_Dataset/Instances_v2/us_11city"
)

if [[ -n "${EVRPTW_DATASET_ROOT:-}" ]]; then
  if [[ "${EVRPTW_DATASET_ROOT}" == /* ]]; then
    echo "EVRPTW_DATASET_ROOT must be relative to the repository root." >&2
    echo "Example: EVRPTW_DATASET_ROOT=../evrptw_runtime/EVRPTW_Dataset/Instances_v2/us_11city" >&2
    exit 2
  fi
  DATASET_ROOT="${EVRPTW_DATASET_ROOT}"
else
  DATASET_ROOT=""
  for candidate in "${dataset_candidates[@]}"; do
    if [[ -d "${candidate}/materialized/families" && -d "${candidate}/generation_plan" ]]; then
      DATASET_ROOT="${candidate}"
      break
    fi
  done
  if [[ -z "${DATASET_ROOT}" && "${DRY_RUN}" == "1" ]]; then
    DATASET_ROOT="${CANONICAL_DATASET_RELATIVE_ROOT}"
  fi
fi
if [[ "${DRY_RUN}" == "0" && ! -d "${DATASET_ROOT}" ]]; then
  echo "No Stage-2 v7 dataset root was found." >&2
  echo "Checked repository-relative roots:" >&2
  printf '  %s\n' "${dataset_candidates[@]}" >&2
  echo "Set EVRPTW_DATASET_ROOT to a repository-relative path if needed." >&2
  exit 2
fi
readonly DATASET_ROOT
readonly FAMILY_ROOT="${DATASET_ROOT}/materialized/families"
results_root_raw="${EVRPTW_TEST_RESULTS_ROOT:-EVRPTW_Benchmark/results/CLE_EVRPTW_v2_test_2h}"
readonly RESULTS_ROOT="${results_root_raw}"

if [[ "${DRY_RUN}" == "0" ]]; then
  if [[ ! -d "${FAMILY_ROOT}" ]]; then
    echo "Missing Stage-2 materialized family root: ${FAMILY_ROOT}" >&2
    exit 2
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required; set up the ${CONDA_ENV} environment first." >&2
    exit 2
  fi
fi

export PYTHONDONTWRITEBYTECODE=1
export EVRPTW_META_THREADS_PER_WORKER=1
export PYTHONPATH="${REPO_ROOT}/EVRPTW_Core:${REPO_ROOT}/EVRPTW_Dataset_Generator/src:${REPO_ROOT}/EVRPTW_Benchmark/Exact/Gurobi_Solver:${REPO_ROOT}/EVRPTW_Benchmark/MetaHeuristics:${REPO_ROOT}/EVRPTW_Benchmark/MetaHeuristics/ALNS_Solver:${REPO_ROOT}/EVRPTW_Benchmark/MetaHeuristics/VNS_TS_Solver${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${EVRPTW_MAX_INSTANCES:-}" ]] && \
  ! [[ "${EVRPTW_MAX_INSTANCES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVRPTW_MAX_INSTANCES must be a positive integer." >&2
  exit 2
fi

shard_count_raw="${EVRPTW_SHARD_COUNT:-}"
shard_index_raw="${EVRPTW_SHARD_INDEX:-}"
start_raw="${EVRPTW_START_INDEX:-}"
end_raw="${EVRPTW_END_INDEX:-}"
if [[ -n "${shard_count_raw}" || -n "${shard_index_raw}" ]]; then
  if [[ -z "${shard_count_raw}" || -z "${shard_index_raw}" ]]; then
    echo "Set both EVRPTW_SHARD_COUNT and EVRPTW_SHARD_INDEX." >&2
    exit 2
  fi
  if [[ -n "${start_raw}" || -n "${end_raw}" ]]; then
    echo "Do not combine shard selection with explicit start/end indices." >&2
    exit 2
  fi
  if ! [[ "${shard_count_raw}" =~ ^[1-9][0-9]*$ ]] || \
    ! [[ "${shard_index_raw}" =~ ^[0-9]+$ ]]; then
    echo "Shard count must be positive and shard index must be non-negative." >&2
    exit 2
  fi
  if ((shard_count_raw > TEST_VIEW_COUNT)); then
    echo "EVRPTW_SHARD_COUNT cannot exceed ${TEST_VIEW_COUNT}." >&2
    exit 2
  fi
  if ((shard_index_raw >= shard_count_raw)); then
    echo "EVRPTW_SHARD_INDEX must satisfy 0 <= index < count." >&2
    exit 2
  fi
  RANGE_START="$((TEST_VIEW_COUNT * shard_index_raw / shard_count_raw))"
  RANGE_END="$((TEST_VIEW_COUNT * (shard_index_raw + 1) / shard_count_raw))"
  if ((shard_count_raw > 1)); then
    RESULT_PARTITION_DIR="shard_${shard_index_raw}_of_${shard_count_raw}"
  else
    RESULT_PARTITION_DIR=""
  fi
else
  RANGE_START="${start_raw:-0}"
  RANGE_END="${end_raw:-${TEST_VIEW_COUNT}}"
  if ! [[ "${RANGE_START}" =~ ^[0-9]+$ ]] || \
    ! [[ "${RANGE_END}" =~ ^[0-9]+$ ]]; then
    echo "EVRPTW_START_INDEX and EVRPTW_END_INDEX must be non-negative integers." >&2
    exit 2
  fi
  if ((RANGE_START < 0 || RANGE_START >= RANGE_END || RANGE_END > TEST_VIEW_COUNT)); then
    echo "Required range: 0 <= start < end <= ${TEST_VIEW_COUNT}." >&2
    exit 2
  fi
  if [[ -n "${start_raw}" || -n "${end_raw}" ]]; then
    RESULT_PARTITION_DIR="range_${RANGE_START}_${RANGE_END}"
  else
    RESULT_PARTITION_DIR=""
  fi
fi
readonly RANGE_START
readonly RANGE_END
readonly RESULT_PARTITION_DIR

SELECTED_VIEW_COUNT="$((RANGE_END - RANGE_START))"
if [[ -n "${EVRPTW_MAX_INSTANCES:-}" ]] && \
  ((EVRPTW_MAX_INSTANCES < SELECTED_VIEW_COUNT)); then
  SELECTED_VIEW_COUNT="${EVRPTW_MAX_INSTANCES}"
fi
readonly SELECTED_VIEW_COUNT

EXACT_SELECTION_ARGS=()
META_SELECTION_ARGS=()
if ((RANGE_START != 0 || RANGE_END != TEST_VIEW_COUNT)); then
  EXACT_SELECTION_ARGS+=(--start_index "${RANGE_START}" --end_index "${RANGE_END}")
  META_SELECTION_ARGS+=(--start_index "${RANGE_START}" --end_index "${RANGE_END}")
fi
if [[ -n "${EVRPTW_MAX_INSTANCES:-}" ]]; then
  EXACT_SELECTION_ARGS+=(--limit "${EVRPTW_MAX_INSTANCES}")
  META_SELECTION_ARGS+=(--max_instances "${EVRPTW_MAX_INSTANCES}")
fi

require_test_index() {
  local index_path="$1"
  if [[ "${DRY_RUN}" == "0" && ! -f "${index_path}" ]]; then
    echo "Missing test view index: ${index_path}" >&2
    exit 2
  fi
}

partition_output() {
  local variable_name="$1"
  local base_path="$2"
  if [[ -n "${RESULT_PARTITION_DIR}" ]]; then
    printf -v "${variable_name}" '%s/%s' "${base_path}" "${RESULT_PARTITION_DIR}"
  else
    printf -v "${variable_name}" '%s' "${base_path}"
  fi
}

prepare_output() {
  local output_path="$1"
  if [[ "${DRY_RUN}" == "0" ]]; then
    mkdir -p "${output_path}"
  fi
}

print_contract() {
  local solver_name="$1"
  local track_name="$2"
  local index_path="$3"
  local output_path="$4"
  printf '%s\n' \
    "solver=${solver_name}" \
    "track=${track_name}" \
    "dataset=${index_path}" \
    "scale=${CUS_SCALE}" \
    "views=${SELECTED_VIEW_COUNT}/${TEST_VIEW_COUNT}" \
    "range=[${RANGE_START},${RANGE_END})" \
    "checkpoints_s=${CHECKPOINTS_S}" \
    "time_limit_s=${TIME_LIMIT_S}" \
    "workers=${WORKERS}" \
    "output=${output_path}"
}

run_python() {
  local command=(conda run -n "${CONDA_ENV}" --no-capture-output python "$@")
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY-RUN:'
    printf ' %q' "${command[@]}"
    printf '\n'
    return 0
  fi
  "${command[@]}"
}
