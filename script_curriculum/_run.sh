#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${CURRICULUM_PYTHON:-python}" "$HERE/launch.py" "$@"
