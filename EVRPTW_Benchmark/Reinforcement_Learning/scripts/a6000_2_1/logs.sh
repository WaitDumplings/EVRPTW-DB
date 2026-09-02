#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
PATH_FILE="$EVRPTW_OUTPUT_ROOT/launcher_logs/$DRL_SERVER_ID/current.log.path"
[[ -f "$PATH_FILE" ]] || { echo "no launcher log has been created" >&2; exit 2; }
exec tail -n "${LINES:-100}" -F "$(cat "$PATH_FILE")"
