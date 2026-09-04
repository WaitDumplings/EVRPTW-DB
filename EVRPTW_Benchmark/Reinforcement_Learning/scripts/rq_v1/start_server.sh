#!/usr/bin/env bash
set -euo pipefail
SERVER_SCRIPT_DIR="${SERVER_SCRIPT_DIR:?SERVER_SCRIPT_DIR must be set}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/server_env.sh"
MODE="${1:?usage: start_server.sh full|resume}"
shift
case "$MODE" in full|resume) ;; *) echo "invalid mode: $MODE" >&2; exit 2 ;; esac
LOG_DIR="$EVRPTW_OUTPUT_ROOT/launcher_logs/$DRL_SERVER_ID"
mkdir -p "$LOG_DIR"
PID_FILE="$LOG_DIR/$MODE.pid"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "$DRL_SERVER_ID/$MODE is already running with pid $(cat "$PID_FILE")" >&2
  exit 3
fi
STAMP="$(date +%Y%m%dT%H%M%S)"
LOG_FILE="$LOG_DIR/${MODE}_${STAMP}.log"
nohup setsid env SERVER_SCRIPT_DIR="$SERVER_SCRIPT_DIR" \
  bash "$SCRIPT_DIR/run_server.sh" "$MODE" "$@" \
  >"$LOG_FILE" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" >"$PID_FILE"
printf '%s\n' "$LOG_FILE" >"$LOG_DIR/current.log.path"
echo "started with nohup: server=$DRL_SERVER_ID mode=$MODE pid=$PID"
echo "log: $LOG_FILE"
