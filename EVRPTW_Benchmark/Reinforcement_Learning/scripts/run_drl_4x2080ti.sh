#!/usr/bin/env bash
set -euo pipefail
: "${SERVER_INDEX:?SERVER_INDEX must be 0 or 1}"
case "$SERVER_INDEX" in
  0) SERVER_ID="2080ti_4_1" ;;
  1) SERVER_ID="2080ti_4_2" ;;
  *) echo "SERVER_INDEX must be 0 or 1" >&2; exit 2 ;;
esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/$SERVER_ID/run.sh" "$@"
