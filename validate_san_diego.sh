#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATION_ROOT="${SD_VALIDATION_ROOT:-$ROOT_DIR/EVRPTW_Dataset/Validation/san-diego}"
SD_CLE_ROOT="${SD_CLE_ROOT:-$VALIDATION_ROOT/CLE_v1}"
SD_INSTANCE_ROOT="${SD_INSTANCE_ROOT:-$VALIDATION_ROOT/Instances_v1}"
SD_WORK_ROOT="${SD_WORK_ROOT:-$ROOT_DIR/EVRPTW_Dataset_Generator/work/san-diego-validation-v1}"
AMAZON_MODEL_BUILD_INPUTS="${AMAZON_MODEL_BUILD_INPUTS:-$ROOT_DIR/EVRPTW_Dataset_Generator/data/sources/amazon-last-mile-2021/model_build_inputs}"
AMAZON_ARTIFACT_ROOT="${AMAZON_ARTIFACT_ROOT:-$ROOT_DIR/EVRPTW_Dataset/Calibration_v1/amazon_stage2_v2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$SD_WORK_ROOT/matplotlib}"

mkdir -p "$MPLCONFIGDIR"
export MPLCONFIGDIR

for amazon_file in route_data.json package_data.json travel_times.json; do
  if [[ ! -f "$AMAZON_MODEL_BUILD_INPUTS/$amazon_file" && ! -f "$AMAZON_ARTIFACT_ROOT/manifest.json" ]]; then
    echo "Missing Amazon input: $AMAZON_MODEL_BUILD_INPUTS/$amazon_file" >&2
    echo "Run EVRPTW_Dataset_Generator/scripts/download_amazon_last_mile_2021.sh first." >&2
    exit 2
  fi
done

echo "[1/2] Rebuilding the San Diego CLE in an isolated validation root"
CLE_WORK_ROOT="$SD_WORK_ROOT" \
CLE_RELEASE_ROOT="$SD_CLE_ROOT" \
KEEP_CLE_WORK=1 \
PYTHON_BIN="$PYTHON_BIN" \
  "$ROOT_DIR/generate_cle.sh" --cities san-diego

echo "[2/2] Generating and verifying one San Diego Stage-2 family"
INSTANCE_MODE=non_release_pilot \
WORKERS=1 \
CLE_ROOT="$SD_CLE_ROOT" \
INSTANCE_OUTPUT_ROOT="$SD_INSTANCE_ROOT" \
AMAZON_MODEL_BUILD_INPUTS="$AMAZON_MODEL_BUILD_INPUTS" \
AMAZON_ARTIFACT_ROOT="$AMAZON_ARTIFACT_ROOT" \
PYTHON_BIN="$PYTHON_BIN" \
  "$ROOT_DIR/generate_instances.sh" \
    --cities san-diego \
    --tracks train \
    --max-families 1

test -f "$SD_CLE_ROOT/cities/san-diego/manifest.json"
test -f "$SD_INSTANCE_ROOT/stage2_run_report.json"
test -f "$SD_INSTANCE_ROOT/reports/phase1/phase1_corpus_metrics.json"

echo "San Diego validation passed."
echo "CLE: $SD_CLE_ROOT/cities/san-diego"
echo "Stage-2 report: $SD_INSTANCE_ROOT/stage2_run_report.json"
echo "Phase-1 metrics: $SD_INSTANCE_ROOT/reports/phase1"
