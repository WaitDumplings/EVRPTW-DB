#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Use the current registered-stream queue and its machine-specific profiles.
exec bash "$SCRIPT_DIR/../rq_v1/2080ti_3_1/resume.sh" "$@"
