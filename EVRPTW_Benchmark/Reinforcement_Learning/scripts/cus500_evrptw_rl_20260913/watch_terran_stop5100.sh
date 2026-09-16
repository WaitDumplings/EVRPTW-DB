#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STOP_PYTHON="${STOP_PYTHON:-/home/npg/miniconda3/envs/maojie/bin/python}"
exec "$STOP_PYTHON" "$SCRIPT_DIR/watch_terran_stop5100.py" "$@"
