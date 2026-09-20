#!/usr/bin/env bash
# Compatibility entry: defaults to GPUs 0/1; source is now /data/cus100_ckpt/am.ckpt.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/_launch.sh" am_evrptw "$@"
