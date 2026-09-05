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

# Serialize the PID check and PID-file write so simultaneous full/resume
# requests cannot both get past the check before either launcher is recorded.
exec 9>"$LOG_DIR/launcher.lock"
flock -x 9
for ACTIVE_MODE in full resume; do
  ACTIVE_PID_FILE="$LOG_DIR/$ACTIVE_MODE.pid"
  [[ -f "$ACTIVE_PID_FILE" ]] || continue
  ACTIVE_PID="$(tr -d '[:space:]' < "$ACTIVE_PID_FILE")"
  if [[ "$ACTIVE_PID" =~ ^[1-9][0-9]*$ ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
    echo "$DRL_SERVER_ID/$ACTIVE_MODE is already running with pid $ACTIVE_PID" >&2
    exit 3
  fi
  rm -f -- "$ACTIVE_PID_FILE"
done

PID_FILE="$LOG_DIR/$MODE.pid"
STAMP="$(date +%Y%m%dT%H%M%S)"
LOG_FILE="$LOG_DIR/${MODE}_${STAMP}.log"
nohup setsid env SERVER_SCRIPT_DIR="$SERVER_SCRIPT_DIR" \
  bash "$SCRIPT_DIR/run_server.sh" "$MODE" "$@" \
  >"$LOG_FILE" 2>&1 < /dev/null 9>&- &
PID=$!
printf '%s\n' "$PID" >"$PID_FILE"
printf '%s\n' "$LOG_FILE" >"$LOG_DIR/current.log.path"
flock -u 9
echo "started with nohup: server=$DRL_SERVER_ID mode=$MODE pid=$PID"
echo "log: $LOG_FILE"
