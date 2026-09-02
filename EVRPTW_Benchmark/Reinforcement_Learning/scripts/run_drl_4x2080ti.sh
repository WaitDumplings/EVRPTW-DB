#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SERVER_INDEX:?SERVER_INDEX must be 0 or 1}"
case "$SERVER_INDEX" in
  0) export DRL_SLOTS="0,1,2,3" ;;
  1) export DRL_SLOTS="4,5,6,7" ;;
  *) echo "SERVER_INDEX must be 0 or 1" >&2; exit 2 ;;
esac
export DRL_MANIFEST="$SCRIPT_DIR/../manifests/drl_2080ti_jobs_v1.jsonl"
export DRL_LOCAL_GPU_COUNT=4
export DRL_GPU_NAME_PATTERN="RTX 2080 Ti"
exec bash "$SCRIPT_DIR/drl_job_runner.sh" "$@"
