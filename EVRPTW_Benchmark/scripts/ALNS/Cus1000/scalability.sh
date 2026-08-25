#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../common.sh"
run_frozen_test "ALNS" "Cus1000" "unseen_scale_same_cities" \
  "scalability_cus2000/test/unseen_scale_same_cities/view_index.parquet" \
  "scalability_cus2000/unseen_scale_same_cities"
