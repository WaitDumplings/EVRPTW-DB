#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "common.sh is a shared library; run one of the run_*_cus50_test.sh scripts." >&2
  exit 2
fi

readonly TEST_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${TEST_SCRIPT_DIR}/../.." && pwd)"
readonly CHECKPOINTS_S="300,1800,3600,7200"
readonly TIME_LIMIT_S="7200"
readonly CUS_SCALE="Cus50"
readonly TEST_RELATIVE_INDEX="compatibility_cus50/test/test1_new_seed_same_cities/view_index.parquet"
readonly TEST_RESULT_RELATIVE_ROOT="compatibility_cus50/test1_new_seed_same_cities"
readonly BASE_SEED="2026"

readonly WORKERS="${EVRPTW_TEST_WORKERS:-30}"
readonly MAX_IN_FLIGHT="${EVRPTW_MAX_IN_FLIGHT:-$((WORKERS * 2))}"
readonly CSV_FLUSH_INTERVAL="${EVRPTW_CSV_FLUSH_INTERVAL:-25}"
readonly CONDA_ENV="${EVRPTW_CONDA_ENV:-maojie}"
readonly DRY_RUN="${EVRPTW_DRY_RUN:-0}"

for value_name in WORKERS MAX_IN_FLIGHT; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer, got: ${value}" >&2
    exit 2
  fi
done
if ! [[ "${CSV_FLUSH_INTERVAL}" =~ ^[0-9]+$ ]]; then
  echo "CSV_FLUSH_INTERVAL must be a non-negative integer, got: ${CSV_FLUSH_INTERVAL}" >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "EVRPTW_DRY_RUN must be 0 or 1, got: ${DRY_RUN}" >&2
  exit 2
fi

canonical_root="${REPO_ROOT}/EVRPTW_Dataset/Instances_v2/us_11city"
generated_root="${REPO_ROOT}/EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823"
if [[ -n "${EVRPTW_DATASET_ROOT:-}" ]]; then
  DATASET_ROOT="${EVRPTW_DATASET_ROOT}"
elif [[ -d "${canonical_root}" ]]; then
  DATASET_ROOT="${canonical_root}"
elif [[ -d "${generated_root}" ]]; then
  DATASET_ROOT="${generated_root}"
else
  echo "No Stage-2 v7 dataset root was found." >&2
  echo "Restore it to ${canonical_root} or set EVRPTW_DATASET_ROOT." >&2
  exit 2
fi
readonly DATASET_ROOT
readonly TEST_INDEX="${DATASET_ROOT}/generation_plan/${TEST_RELATIVE_INDEX}"
readonly FAMILY_ROOT="${DATASET_ROOT}/materialized/families"
readonly RESULTS_ROOT="${EVRPTW_TEST_RESULTS_ROOT:-${REPO_ROOT}/EVRPTW_Benchmark/results/CLE_EVRPTW_v2_test_2h}"

if [[ "${DRY_RUN}" == "0" ]]; then
  if [[ ! -f "${TEST_INDEX}" ]]; then
    echo "Missing Cus50 Test-1 view index: ${TEST_INDEX}" >&2
    exit 2
  fi
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

EXACT_SELECTION_ARGS=()
META_SELECTION_ARGS=()
if [[ -n "${EVRPTW_MAX_INSTANCES:-}" ]]; then
  if ! [[ "${EVRPTW_MAX_INSTANCES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVRPTW_MAX_INSTANCES must be a positive integer." >&2
    exit 2
  fi
  EXACT_SELECTION_ARGS+=(--limit "${EVRPTW_MAX_INSTANCES}")
  META_SELECTION_ARGS+=(--max_instances "${EVRPTW_MAX_INSTANCES}")
fi
if [[ -n "${EVRPTW_START_INDEX:-}" ]]; then
  EXACT_SELECTION_ARGS+=(--start_index "${EVRPTW_START_INDEX}")
  META_SELECTION_ARGS+=(--start_index "${EVRPTW_START_INDEX}")
fi
if [[ -n "${EVRPTW_END_INDEX:-}" ]]; then
  EXACT_SELECTION_ARGS+=(--end_index "${EVRPTW_END_INDEX}")
  META_SELECTION_ARGS+=(--end_index "${EVRPTW_END_INDEX}")
fi

prepare_output() {
  local output_path="$1"
  if [[ "${DRY_RUN}" == "0" ]]; then
    mkdir -p "${output_path}"
  fi
}

print_contract() {
  local solver_name="$1"
  local output_path="$2"
  printf '%s\n' \
    "solver=${solver_name}" \
    "dataset=${TEST_INDEX}" \
    "scale=${CUS_SCALE}" \
    "instances=500" \
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
  cd "${REPO_ROOT}"
  "${command[@]}"
}
