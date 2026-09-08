#!/usr/bin/env bash
set -euo pipefail
SERVER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SERVER_SCRIPT_DIR
exec bash "$SERVER_SCRIPT_DIR/../start_server.sh" resume "$@"
