#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../common.sh"
run_frozen_test "VNSTS" "Cus50" "test1_new_seed" \
  "compatibility_cus50/test/test1_new_seed_same_cities/view_index.parquet" \
  "compatibility_cus50/test1_new_seed_same_cities"
