#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CUS500_STOP_PYTHON="${CUS500_STOP_PYTHON:-/home/npg/miniconda3/envs/maojie/bin/python}"
exec "$CUS500_STOP_PYTHON" "$SCRIPT_DIR/watch_stop_epoch.py" "$@"
