#!/usr/bin/env bash
set -euo pipefail
SERVER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SERVER_SCRIPT_DIR
source "$SERVER_SCRIPT_DIR/env.sh"
export DRL_MANIFEST="$SERVER_SCRIPT_DIR/jobs_preverified.jsonl"
exec bash "$SERVER_SCRIPT_DIR/../start_server.sh" full \
  --reuse-preverified-training-streams "$@"
