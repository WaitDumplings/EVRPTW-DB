#!/usr/bin/env bash
set -euo pipefail
SERVER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SERVER_SCRIPT_DIR
export DRL_MANIFEST="$SERVER_SCRIPT_DIR/jobs_preverified.jsonl"
exec bash "$SERVER_SCRIPT_DIR/../start_server.sh" resume \
  --reuse-preverified-training-streams "$@"
