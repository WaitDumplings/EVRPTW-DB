#!/usr/bin/env bash
set -euo pipefail
SERVER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SERVER_SCRIPT_DIR
source "$SERVER_SCRIPT_DIR/../server_env.sh"
LOG_DIR="$EVRPTW_OUTPUT_ROOT/launcher_logs/$DRL_SERVER_ID"
[[ -f "$LOG_DIR/current.log.path" ]] || { echo "No launcher log yet." >&2; exit 2; }
exec tail -n 100 -f "$(cat "$LOG_DIR/current.log.path")"
