#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DRL_MANIFEST:?DRL_MANIFEST must name a frozen manifest}"
: "${DRL_SLOTS:?DRL_SLOTS must name the owned global slots}"
: "${DRL_LOCAL_GPU_COUNT:?DRL_LOCAL_GPU_COUNT is required}"
: "${DRL_GPU_NAME_PATTERN:?DRL_GPU_NAME_PATTERN is required}"

exec python "$SCRIPT_DIR/drl_job_runtime.py" "${1:?mode is required}" \
  --manifest "$DRL_MANIFEST" \
  --slots "$DRL_SLOTS" \
  --local-gpu-count "$DRL_LOCAL_GPU_COUNT" \
  --gpu-name-pattern "$DRL_GPU_NAME_PATTERN" \
  "${@:2}"
