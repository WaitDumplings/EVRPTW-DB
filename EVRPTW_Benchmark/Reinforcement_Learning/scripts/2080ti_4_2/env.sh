#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DEFAULT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export EVRPTW_REPO_ROOT="${EVRPTW_REPO_ROOT:-$REPO_DEFAULT}"
export EVRPTW_DATASET_ROOT="${EVRPTW_DATASET_ROOT:-$EVRPTW_REPO_ROOT/EVRPTW_Dataset/Instances_v2/us_11city}"
export EVRPTW_OUTPUT_ROOT="${EVRPTW_OUTPUT_ROOT:-$EVRPTW_REPO_ROOT/EVRPTW_Benchmark/results/DRL_protocol_v1}"
export DRL_MANIFEST="$SCRIPT_DIR/jobs.jsonl"
export DRL_SLOTS="0,1,2,3"
export DRL_LOCAL_GPU_COUNT="4"
export DRL_GPU_NAME_PATTERN="RTX 2080 Ti"
export DRL_SERVER_ID="2080ti_4_2"
