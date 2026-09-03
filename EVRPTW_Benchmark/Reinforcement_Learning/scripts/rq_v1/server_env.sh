#!/usr/bin/env bash
set -euo pipefail

SERVER_SCRIPT_DIR="${SERVER_SCRIPT_DIR:?SERVER_SCRIPT_DIR must be set by the server wrapper}"
export DRL_SERVER_ID="$(basename "$SERVER_SCRIPT_DIR")"
export EVRPTW_REPO_ROOT="${EVRPTW_REPO_ROOT:-$(cd "$SERVER_SCRIPT_DIR/../../../../.." && pwd)}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/dataset_root.sh"
export EVRPTW_DATASET_ROOT="$(resolve_evrptw_dataset_root "$EVRPTW_REPO_ROOT")"
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
