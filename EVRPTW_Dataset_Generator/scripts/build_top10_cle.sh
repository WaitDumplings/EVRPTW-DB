#!/usr/bin/env bash
set -euo pipefail

GENERATOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$GENERATOR_DIR"

python scripts/build_top10_cle.py \
  --profile configs/us_top10_cle_v1.json \
  "$@"
