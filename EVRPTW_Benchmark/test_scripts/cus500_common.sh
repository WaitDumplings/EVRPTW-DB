#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "cus500_common.sh is a shared library." >&2
  exit 2
fi

CUS_SCALE="Cus500"
TEST_VIEW_COUNT="500"
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

readonly -a CUS500_TRACK_IDS=(
  "test1_new_seed"
  "test2_heldout_locations"
  "test3_heldout_city"
)
readonly -a CUS500_RELATIVE_INDICES=(
  "core/test/test1_new_seed/view_index.parquet"
  "core/test/test2_heldout_locations/view_index.parquet"
  "core/test/test3_heldout_city/view_index.parquet"
)
