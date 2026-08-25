#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "cus50_common.sh is a shared library." >&2
  exit 2
fi

CUS_SCALE="Cus50"
TEST_VIEW_COUNT="500"
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

readonly TEST_RELATIVE_INDEX="compatibility_cus50/test/test1_new_seed_same_cities/view_index.parquet"
readonly TEST_RESULT_RELATIVE_ROOT="compatibility_cus50/test1_new_seed_same_cities"
readonly TEST_INDEX="${DATASET_ROOT}/generation_plan/${TEST_RELATIVE_INDEX}"
require_test_index "${TEST_INDEX}"
