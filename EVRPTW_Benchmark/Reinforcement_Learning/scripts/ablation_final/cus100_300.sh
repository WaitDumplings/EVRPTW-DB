#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/train.sh" --scale 100 --epochs 300 --validation-every 100 "$@"
