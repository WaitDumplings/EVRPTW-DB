#!/usr/bin/env bash
set -euo pipefail

SERVER_SCRIPT_DIR="${SERVER_SCRIPT_DIR:?SERVER_SCRIPT_DIR must be set by the server wrapper}"
export DRL_SERVER_ID="$(basename "$SERVER_SCRIPT_DIR")"
export EVRPTW_REPO_ROOT="${EVRPTW_REPO_ROOT:-$(cd "$SERVER_SCRIPT_DIR/../../../../.." && pwd)}"

if [[ -n "${EVRPTW_DATASET_ROOT:-}" ]]; then
  DATASET_ROOT="$EVRPTW_DATASET_ROOT"
  [[ "$DATASET_ROOT" = /* ]] || DATASET_ROOT="$EVRPTW_REPO_ROOT/$DATASET_ROOT"
else
  DATASET_CANDIDATES=(
    "$EVRPTW_REPO_ROOT/EVRPTW_Dataset/Instances_v2/us_11city"
    "$EVRPTW_REPO_ROOT/EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823"
    "$EVRPTW_REPO_ROOT/../EVRPTW-DB/EVRPTW_Dataset/Instances_v2/us_11city"
    "$EVRPTW_REPO_ROOT/../EVRPTW-DB/EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823"
  )
  DATASET_ROOT="${DATASET_CANDIDATES[0]}"
  for candidate in "${DATASET_CANDIDATES[@]}"; do
    if [[ -f "$candidate/generation_plan/core/train/view_index.parquet" ]]; then
      DATASET_ROOT="$candidate"
      break
    fi
  done
fi
export EVRPTW_DATASET_ROOT="$(realpath -m "$DATASET_ROOT")"
export EVRPTW_OUTPUT_ROOT="${EVRPTW_OUTPUT_ROOT:-$EVRPTW_REPO_ROOT/EVRPTW_Benchmark/results/DRL_rq_v1}"
export EVRPTW_CONDA_ENV="${EVRPTW_CONDA_ENV:-maojie}"
export DRL_MANIFEST="$SERVER_SCRIPT_DIR/jobs.jsonl"
export DRL_EXPECTED_BRANCH="${DRL_EXPECTED_BRANCH:-drl-benchmark-adapters}"

case "$DRL_SERVER_ID" in
  2080ti_4_1|2080ti_4_2)
    export DRL_SLOTS="0,1,2,3"
    export DRL_LOCAL_GPU_COUNT="4"
    export DRL_GPU_NAME_PATTERN="RTX 2080 Ti"
    ;;
  2080ti_3_1)
    export DRL_SLOTS="0,1,2"
    export DRL_LOCAL_GPU_COUNT="3"
    export DRL_GPU_NAME_PATTERN="RTX 2080 Ti"
    ;;
  a6000_2_1)
    export DRL_SLOTS="0,1"
    export DRL_LOCAL_GPU_COUNT="2"
    export DRL_GPU_NAME_PATTERN="RTX A6000"
    ;;
  *) echo "Unknown RQ server bundle: $DRL_SERVER_ID" >&2; exit 2 ;;
esac
