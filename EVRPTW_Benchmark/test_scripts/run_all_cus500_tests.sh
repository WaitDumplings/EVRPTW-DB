#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Run sequentially to avoid three 30-process pools competing for one server.
bash "${SCRIPT_DIR}/run_gurobi_cus500_tests.sh"
bash "${SCRIPT_DIR}/run_alns_cus500_tests.sh"
bash "${SCRIPT_DIR}/run_vnsts_cus500_tests.sh"
