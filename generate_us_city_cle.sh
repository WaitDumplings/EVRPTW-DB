#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATOR_DIR="$ROOT_DIR/EVRPTW_Dataset_Generator"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat >&2 <<'EOF'
Usage:
  NLR_API_KEY=... ./generate_us_city_cle.sh --city "San Diego" --state CA

Optional smaller official OSM extract:
  ... --geofabrik-region california/socal

The command resolves a 2025 Census Place, downloads/reuses public U.S. source
adapters, generates a single-city configuration, builds the CLE, and packages
it under EVRPTW_Dataset/CLE_v1/us_custom/<city-slug>.
EOF
}

if [[ "$#" -eq 0 ]]; then
  usage
  exit 2
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
PYTHON_PATH="$(command -v "$PYTHON_BIN")"
PYTHON_ENV_BIN="$(cd "$(dirname "$PYTHON_PATH")" && pwd)"
if [[ -x "$PYTHON_ENV_BIN/osmium" ]]; then
  export PATH="$PYTHON_ENV_BIN:$PATH"
fi
if ! command -v osmium >/dev/null 2>&1; then
  echo "osmium is required. Create/activate EVRPTW_Dataset_Generator/environment.yml." >&2
  exit 2
fi

export PYTHONPATH="$GENERATOR_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$GENERATOR_DIR/work/.matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$GENERATOR_DIR/work/.cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

cd "$GENERATOR_DIR"
exec "$PYTHON_BIN" scripts/generate_us_city_cle.py "$@"
