#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Run sequentially so each solver receives the requested process count without
# three process pools competing for the same CPUs and memory.
bash "${SCRIPT_DIR}/run_gurobi_cus50_test.sh"
bash "${SCRIPT_DIR}/run_alns_cus50_test.sh"
bash "${SCRIPT_DIR}/run_vnsts_cus50_test.sh"
