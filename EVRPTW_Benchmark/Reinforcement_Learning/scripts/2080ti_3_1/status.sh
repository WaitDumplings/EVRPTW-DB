#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
LOG_DIR="$EVRPTW_OUTPUT_ROOT/launcher_logs/$DRL_SERVER_ID"
for MODE in pilot full resume; do
  PID_FILE="$LOG_DIR/$MODE.pid"
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "$MODE: running pid=$(cat "$PID_FILE")"
  fi
done
bash "$SCRIPT_DIR/run.sh" status --skip-gpu-preflight "$@"
