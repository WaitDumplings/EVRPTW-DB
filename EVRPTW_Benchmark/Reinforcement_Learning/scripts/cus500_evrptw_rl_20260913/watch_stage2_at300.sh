#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CUS500_STAGE2_PYTHON="${CUS500_STAGE2_PYTHON:-/home/npg/miniconda3/envs/maojie/bin/python}"
if [[ ! -x "$CUS500_STAGE2_PYTHON" ]]; then
  CUS500_STAGE2_PYTHON=python3
fi
exec "$CUS500_STAGE2_PYTHON" "$SCRIPT_DIR/watch_stage2.py" "$@"
