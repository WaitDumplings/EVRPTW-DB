#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
# Do not source the multi-instance common.sh: it forces threads=1/MIPGap=0.
# Keep the activated conda environment and let Gurobi choose its own threads.
exec "${PYTHON_BIN:-python3}" -u \
  "${REPO_ROOT}/EVRPTW_Benchmark/Exact/Gurobi_Solver/run_single_cus500.py" "$@"
