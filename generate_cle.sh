#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATOR_DIR="$ROOT_DIR/EVRPTW_Dataset_Generator"
PYTHON_BIN="${PYTHON_BIN:-python}"
CLE_PROFILE="${CLE_PROFILE:-$GENERATOR_DIR/configs/us_11city_cle_v1.json}"
WORK_ROOT="${CLE_WORK_ROOT:-$GENERATOR_DIR/work/us-11city-v1}"
RELEASE_ROOT="${CLE_RELEASE_ROOT:-$ROOT_DIR/EVRPTW_Dataset/CLE_v1/us_11city}"
NSI_CACHE_ROOT="${NSI_CACHE_ROOT:-$GENERATOR_DIR/data/sources/nsi-us-11city}"
NSI_WORKERS="${NSI_WORKERS:-4}"
KEEP_CLE_WORK="${KEEP_CLE_WORK:-0}"
PREPARE_CLE_SOURCES="${PREPARE_CLE_SOURCES:-1}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  echo "Activate the evrptw-cle conda environment first." >&2
  exit 2
fi
if ! command -v osmium >/dev/null 2>&1; then
  echo "osmium is required. Activate the environment from environment.yml." >&2
  exit 2
fi
if [[ ! -f "$CLE_PROFILE" ]]; then
  echo "CLE profile is missing: $CLE_PROFILE" >&2
  exit 2
fi

export PYTHONPATH="$GENERATOR_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ "$PREPARE_CLE_SOURCES" == "1" ]]; then
  echo "Preparing/reusing the frozen 11-city public source inputs"
  cd "$GENERATOR_DIR"
  "$PYTHON_BIN" scripts/prepare_us11_sources.py
elif [[ "$PREPARE_CLE_SOURCES" != "0" ]]; then
  echo "PREPARE_CLE_SOURCES must be 0 or 1." >&2
  exit 2
fi

mkdir -p "$WORK_ROOT" "$RELEASE_ROOT"

# Frozen NSI API responses are source data, not generated CLE state. Pre-seed
# them into the resumable work tree so the server build does not need network.
if [[ -d "$NSI_CACHE_ROOT" ]]; then
  for city_cache in "$NSI_CACHE_ROOT"/*; do
    [[ -d "$city_cache/raw_tiles" ]] || continue
    city_slug="$(basename "$city_cache")"
    destination="$WORK_ROOT/customers/$city_slug/raw_tiles"
    mkdir -p "$destination"
    cp -R "$city_cache/raw_tiles/." "$destination/"
  done
fi

cd "$GENERATOR_DIR"

echo "CLE profile: $CLE_PROFILE"
echo "CLE work root: $WORK_ROOT"

"$PYTHON_BIN" scripts/build_cle_cohort.py \
  --profile "$CLE_PROFILE" \
  --work-root "$WORK_ROOT" \
  --release-root "$RELEASE_ROOT" \
  --nsi-workers "$NSI_WORKERS" \
  --replace-release-package \
  "$@"

# A no-argument invocation is the complete production build. Remove only its
# validated generator-owned work tree after packaging; partial/debug runs stay
# resumable unless the caller explicitly removes them.
if [[ "$#" -eq 0 && "$KEEP_CLE_WORK" != "1" ]]; then
  case "$WORK_ROOT" in
    "$GENERATOR_DIR"/work/*)
      rm -rf -- "$WORK_ROOT"
      ;;
    *)
      echo "Refusing to remove non-generator work root: $WORK_ROOT" >&2
      exit 2
      ;;
  esac
fi

echo "CLE output: $RELEASE_ROOT"
