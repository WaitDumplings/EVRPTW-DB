#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../common.sh"
run_frozen_test "ALNS" "Cus1000" "test1_new_seed" \
  "core/test/test1_new_seed/view_index.parquet" \
  "core/test1_new_seed"
