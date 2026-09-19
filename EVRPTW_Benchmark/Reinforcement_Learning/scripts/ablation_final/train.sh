#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${ABLATION_PYTHON:-python3}" -u "${HERE}/launch.py" "$@"
