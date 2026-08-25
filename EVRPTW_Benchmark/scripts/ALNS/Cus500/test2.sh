#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../common.sh"
run_frozen_test "ALNS" "Cus500" "test2_heldout_locations" \
  "core/test/test2_heldout_locations/view_index.parquet" \
  "core/test2_heldout_locations"
