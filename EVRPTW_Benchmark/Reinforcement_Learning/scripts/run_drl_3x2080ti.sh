#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DRL_MANIFEST="$SCRIPT_DIR/../manifests/drl_2080ti_jobs_v1.jsonl"
export DRL_SLOTS="8,9,10"
export DRL_LOCAL_GPU_COUNT=3
export DRL_GPU_NAME_PATTERN="RTX 2080 Ti"
exec bash "$SCRIPT_DIR/drl_job_runner.sh" "$@"
