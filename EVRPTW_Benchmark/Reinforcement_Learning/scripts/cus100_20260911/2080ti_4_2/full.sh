#!/usr/bin/env bash
set -euo pipefail
# TR17/TR18 numerical-stability retraining only: use ./evrptw_rl_stable.sh.
# This original ten-job deployment keeps its historical behavior and outputs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
PYTHON_BIN="${CUS100_PYTHON:-python}"
cd "$REPO_ROOT"
exec "$PYTHON_BIN" "$SCRIPT_DIR/../launch.py" --server 2080ti_4_2 --mode start "$@"
