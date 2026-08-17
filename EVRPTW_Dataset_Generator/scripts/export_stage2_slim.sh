#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATOR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPOSITORY_DIR="$(cd "$GENERATOR_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CLE_ROOT="${CLE_ROOT:-$REPOSITORY_DIR/EVRPTW_Dataset/CLE_v2/us_11city}"
SOURCE_INSTANCE_ROOT="${SOURCE_INSTANCE_ROOT:-$REPOSITORY_DIR/EVRPTW_Dataset/Instances_v2/us_11city}"
SLIM_INSTANCE_ROOT="${SLIM_INSTANCE_ROOT:-$REPOSITORY_DIR/EVRPTW_Dataset/Instances_v2_slim/us_11city}"
PROFILE_PATH="${PROFILE_PATH:-$GENERATOR_DIR/configs/us_reference_instance_profile_v2.json}"

if [[ ! -d "$SOURCE_INSTANCE_ROOT/materialized/families" ]]; then
  echo "Full Stage-2 instance tree is missing: $SOURCE_INSTANCE_ROOT" >&2
  exit 2
fi
if [[ ! -d "$CLE_ROOT/cities" ]]; then
  echo "CLE root is missing: $CLE_ROOT" >&2
  exit 2
fi
if [[ -e "$SLIM_INSTANCE_ROOT" ]]; then
  echo "Refusing to overwrite slim output: $SLIM_INSTANCE_ROOT" >&2
  exit 2
fi

mkdir -p "$(dirname "$SLIM_INSTANCE_ROOT")"
cd "$GENERATOR_DIR"
export PYTHONPATH="$GENERATOR_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" scripts/reconstruct_stage2_instances.py export-slim \
  --source-root "$SOURCE_INSTANCE_ROOT" \
  --output-root "$SLIM_INSTANCE_ROOT" \
  --cle-root "$CLE_ROOT" \
  --profile "$PROFILE_PATH" \
  --report "$SLIM_INSTANCE_ROOT.export_report.json"

echo "Slim Stage-2 parameters: $SLIM_INSTANCE_ROOT"
