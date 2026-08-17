#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATOR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPOSITORY_DIR="$(cd "$GENERATOR_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CLE_ROOT="${CLE_ROOT:-$REPOSITORY_DIR/EVRPTW_Dataset/CLE_v2/us_11city}"
INSTANCE_ROOT="${INSTANCE_ROOT:-$REPOSITORY_DIR/EVRPTW_Dataset/Instances_v2/us_11city}"
WORKERS="${WORKERS:-1}"
FAMILIES_PER_WORKER_TASK="${FAMILIES_PER_WORKER_TASK:-25}"

if [[ ! -d "$CLE_ROOT/cities" ]]; then
  echo "CLE root is missing its cities directory: $CLE_ROOT" >&2
  exit 2
fi
if [[ ! -d "$INSTANCE_ROOT/materialized/families" ]]; then
  echo "Slim/full instance parameters are missing: $INSTANCE_ROOT" >&2
  exit 2
fi

cd "$GENERATOR_DIR"
export PYTHONPATH="$GENERATOR_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

selection_args=("$@")
"$PYTHON_BIN" scripts/reconstruct_stage2_instances.py restore \
  --dataset-root "$INSTANCE_ROOT" \
  --cle-root "$CLE_ROOT" \
  --workers "$WORKERS" \
  --families-per-worker-task "$FAMILIES_PER_WORKER_TASK" \
  --validation exact \
  --report "$INSTANCE_ROOT/matrix_restore_report.json" \
  "${selection_args[@]}"

echo "Restored Stage-2 matrix cache: $INSTANCE_ROOT"
