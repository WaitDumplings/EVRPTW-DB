#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
PYTHON_BIN="${CUS100_PYTHON:-python}"
cd "$REPO_ROOT"
exec "$PYTHON_BIN" "$SCRIPT_DIR/../launch.py" --server 2080ti_4_1 --mode start "$@"
