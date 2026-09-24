#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${EVRPTW_PYTHON:-python3}" "${SCRIPT_DIR}/bks_refine.py" \
  --part lower --workers 30 --time-limit-s 7200 \
  --checkpoints-s 900,1800,2700,3600,4500,5400,6300,7200 "$@"
