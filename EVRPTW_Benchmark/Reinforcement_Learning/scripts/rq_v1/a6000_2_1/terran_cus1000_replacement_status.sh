#!/usr/bin/env bash
set -euo pipefail
SERVER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SERVER_SCRIPT_DIR
export DRL_MANIFEST="$SERVER_SCRIPT_DIR/terran_cus1000_replacement_jobs.jsonl"
export DRL_SCALES="Cus1000"
exec bash "$SERVER_SCRIPT_DIR/../run_server.sh" status "$@" \
  --launcher-id terran_cus1000_replacement_v1 \
  --methods terran \
  --slots 1 \
  --slot-gpu-map 1:1 \
  --skip-gpu-preflight
