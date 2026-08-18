#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATOR_DIR="$ROOT_DIR/EVRPTW_Dataset_Generator"
PYTHON_BIN="${PYTHON_BIN:-python}"
WORKERS="${WORKERS:-12}"
FAMILIES_PER_WORKER_TASK="${FAMILIES_PER_WORKER_TASK:-25}"
INSTANCE_MODE="${INSTANCE_MODE:-research}"
INSTANCE_METHOD="${INSTANCE_METHOD:-stage2}"
CLE_ROOT="${CLE_ROOT:-$ROOT_DIR/EVRPTW_Dataset/CLE_v2/us_11city}"
INSTANCE_OUTPUT_ROOT="${INSTANCE_OUTPUT_ROOT:-$ROOT_DIR/EVRPTW_Dataset/Instances_v2/us_11city}"
AMAZON_MODEL_BUILD_INPUTS="${AMAZON_MODEL_BUILD_INPUTS:-$GENERATOR_DIR/data/sources/amazon-last-mile-2021/model_build_inputs}"
AMAZON_ARTIFACT_ROOT="${AMAZON_ARTIFACT_ROOT:-$ROOT_DIR/EVRPTW_Dataset/Calibration_v2/amazon_stage2_v3}"
MAX_ATTEMPTS_PER_FAMILY="${MAX_ATTEMPTS_PER_FAMILY:-4}"
RUN_DISCIPLINE="${RUN_DISCIPLINE:-}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  echo "Activate the evrptw-cle conda environment first." >&2
  exit 2
fi
if [[ ! -d "$CLE_ROOT/cities" ]]; then
  echo "CLE packages are missing: $CLE_ROOT" >&2
  echo "Run ./generate_cle.sh first." >&2
  exit 2
fi
case "$INSTANCE_METHOD" in
  stage2|restore) ;;
  *)
    echo "INSTANCE_METHOD must be stage2 or restore." >&2
    exit 2
    ;;
esac

if [[ "$INSTANCE_METHOD" == "stage2" && ! -f "$AMAZON_ARTIFACT_ROOT/manifest.json" ]]; then
  amazon_missing=0
  for amazon_file in route_data.json package_data.json travel_times.json; do
    if [[ ! -s "$AMAZON_MODEL_BUILD_INPUTS/$amazon_file" ]]; then
      amazon_missing=1
    fi
  done
  if [[ "$amazon_missing" -eq 1 ]]; then
    if [[ "$(basename "$AMAZON_MODEL_BUILD_INPUTS")" != "model_build_inputs" ]]; then
      echo "Cannot infer the Amazon source root from: $AMAZON_MODEL_BUILD_INPUTS" >&2
      echo "Use a path ending in model_build_inputs or provide the three JSON files." >&2
      exit 2
    fi
    echo "Amazon model-build inputs are missing; downloading the three public files"
    "$GENERATOR_DIR/scripts/download_amazon_last_mile_2021.sh" \
      "$(dirname "$AMAZON_MODEL_BUILD_INPUTS")"
  fi
fi

if [[ "$INSTANCE_METHOD" == "stage2" ]]; then
  case "$INSTANCE_MODE" in
    official|research|non_release_pilot) ;;
    *)
      echo "INSTANCE_MODE must be official, research, or non_release_pilot." >&2
      exit 2
      ;;
  esac
fi

mkdir -p "$INSTANCE_OUTPUT_ROOT"
cd "$GENERATOR_DIR"
export PYTHONPATH="$GENERATOR_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ "$INSTANCE_METHOD" == "stage2" ]]; then
  "$PYTHON_BIN" scripts/check_candidate_revision.py \
    --repo-root "$ROOT_DIR" \
    --require-branch stage2-repair-candidate
fi

if [[ "$INSTANCE_METHOD" == "stage2" && ! -f "$AMAZON_ARTIFACT_ROOT/manifest.json" ]]; then
  "$PYTHON_BIN" scripts/build_amazon_stage2_artifacts.py \
    --model-build-inputs "$AMAZON_MODEL_BUILD_INPUTS" \
    --output-dir "$AMAZON_ARTIFACT_ROOT"
fi

extra_args=("$@")
if [[ -n "${FROZEN_SPLIT_ROOT:-}" ]]; then
  extra_args+=(--frozen-split-root "${FROZEN_SPLIT_ROOT:-}")
fi
if [[ -n "$RUN_DISCIPLINE" ]]; then
  extra_args+=(--run-discipline "$RUN_DISCIPLINE")
fi
if [[ "$INSTANCE_METHOD" == "stage2" && "$INSTANCE_MODE" == "non_release_pilot" ]]; then
  pilot_count_seen=0
  for argument in "$@"; do
    if [[ "$argument" == "--pilot-families-per-city" ]]; then
      pilot_count_seen=1
      break
    fi
  done
  if [[ "$pilot_count_seen" -eq 0 ]]; then
    extra_args+=(--pilot-families-per-city "${PILOT_FAMILIES_PER_CITY:-1}")
  fi
fi

if [[ "$INSTANCE_METHOD" == "stage2" ]]; then
  "$PYTHON_BIN" scripts/build_stage2_instances.py \
    --config configs/cle_evrptw_stage2_v2.json \
    --profile configs/us_reference_instance_profile_v2.json \
    --cle-root "$CLE_ROOT" \
    --block-group-preset configs/us_census_block_groups_v1.json \
    --block-group-source-dir data/sources/census_block_groups_2025 \
    --output-root "$INSTANCE_OUTPUT_ROOT" \
    --amazon-artifact-root "$AMAZON_ARTIFACT_ROOT" \
    --mode "$INSTANCE_MODE" \
    --workers "$WORKERS" \
    --families-per-worker-task "$FAMILIES_PER_WORKER_TASK" \
    --max-attempts-per-family "$MAX_ATTEMPTS_PER_FAMILY" \
    "${extra_args[@]}"
else
  INSTANCE_ROOT="$INSTANCE_OUTPUT_ROOT" \
  CLE_ROOT="$CLE_ROOT" \
  WORKERS="$WORKERS" \
  FAMILIES_PER_WORKER_TASK="$FAMILIES_PER_WORKER_TASK" \
    bash scripts/restore_stage2_instances.sh "${extra_args[@]}"
fi

echo "Instance output: $INSTANCE_OUTPUT_ROOT"
if [[ "$INSTANCE_METHOD" == "stage2" ]]; then
  echo "Phase-1 metrics: $INSTANCE_OUTPUT_ROOT/reports/phase1"
fi
