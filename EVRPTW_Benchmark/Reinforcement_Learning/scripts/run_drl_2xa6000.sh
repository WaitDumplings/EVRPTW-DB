#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DRL_MANIFEST="$SCRIPT_DIR/../manifests/drl_a6000_jobs_v1.jsonl"
export DRL_SLOTS="0,1"
export DRL_LOCAL_GPU_COUNT=2
export DRL_GPU_NAME_PATTERN="RTX A6000"
exec bash "$SCRIPT_DIR/drl_job_runner.sh" "$@"
