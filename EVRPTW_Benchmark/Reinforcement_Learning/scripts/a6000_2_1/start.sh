#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
MODE="${1:?usage: start.sh pilot|full|resume|evaluate [runner options]}"
case "$MODE" in pilot|full|resume|evaluate) ;; *) echo "invalid mode: $MODE" >&2; exit 2 ;; esac
LOG_DIR="$EVRPTW_OUTPUT_ROOT/launcher_logs/$DRL_SERVER_ID"
mkdir -p "$LOG_DIR"
PID_FILE="$LOG_DIR/$MODE.pid"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "$DRL_SERVER_ID/$MODE is already running with pid $(cat "$PID_FILE")" >&2
  exit 3
fi
STAMP="$(date +%Y%m%dT%H%M%S)"
LOG_FILE="$LOG_DIR/${MODE}_${STAMP}.log"
nohup setsid bash "$SCRIPT_DIR/run.sh" "$@" >"$LOG_FILE" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" >"$PID_FILE"
printf '%s\n' "$LOG_FILE" >"$LOG_DIR/current.log.path"
echo "started: server=$DRL_SERVER_ID mode=$MODE pid=$PID"
echo "log: $LOG_FILE"
