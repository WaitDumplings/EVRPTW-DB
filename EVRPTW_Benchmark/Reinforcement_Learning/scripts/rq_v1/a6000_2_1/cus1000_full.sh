#!/usr/bin/env bash
set -euo pipefail
SERVER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SERVER_SCRIPT_DIR
export DRL_MANIFEST="$SERVER_SCRIPT_DIR/cus1000_jobs.jsonl"
export DRL_SCALES="Cus1000"
exec bash "$SERVER_SCRIPT_DIR/../start_server.sh" full "$@"
