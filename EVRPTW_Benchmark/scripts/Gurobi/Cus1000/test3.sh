#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../common.sh"
run_frozen_test "Gurobi" "Cus1000" "test3_heldout_city" \
  "core/test/test3_heldout_city/view_index.parquet" \
  "core/test3_heldout_city"
