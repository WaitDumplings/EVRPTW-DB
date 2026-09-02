#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DEFAULT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export EVRPTW_REPO_ROOT="${EVRPTW_REPO_ROOT:-$REPO_DEFAULT}"
export EVRPTW_DATASET_ROOT="${EVRPTW_DATASET_ROOT:-$EVRPTW_REPO_ROOT/EVRPTW_Dataset/Instances_v2/us_11city}"
export EVRPTW_OUTPUT_ROOT="${EVRPTW_OUTPUT_ROOT:-$EVRPTW_REPO_ROOT/EVRPTW_Benchmark/results/DRL_protocol_v1}"
export EVRPTW_CONDA_ENV="${EVRPTW_CONDA_ENV:-maojie}"
export DRL_MANIFEST="$SCRIPT_DIR/jobs.jsonl"
export DRL_SLOTS="0,1"
export DRL_LOCAL_GPU_COUNT="2"
export DRL_GPU_NAME_PATTERN="RTX A6000"
export DRL_SERVER_ID="a6000_2_1"
