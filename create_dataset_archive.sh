#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL="$ROOT_DIR/EVRPTW_Dataset_Generator/scripts/dataset_archive_tool.py"
CLE_ROOT="${CLE_ROOT:-$ROOT_DIR/EVRPTW_Dataset/CLE_v2/us_11city}"
INSTANCE_ROOT="${INSTANCE_ROOT:-$ROOT_DIR/EVRPTW_Dataset/Instances_v2/us_11city}"
PROFILE_PATH="${PROFILE_PATH:-$ROOT_DIR/EVRPTW_Dataset_Generator/configs/us_reference_instance_profile_v2.json}"
COMPRESSION_THREADS="${COMPRESSION_THREADS:-12}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-9}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ZSTD_BIN="${ZSTD_BIN:-zstd}"

usage() {
  cat <<'EOF'
Create an accepted CLE + slim Stage-2 release archive.

Usage:
  create_dataset_archive.sh --archive FILE.tar.zst [options]

Options:
  --archive FILE                Required output; must not already exist.
  --sha256-file FILE            Default: FILE.sha256.
  --cle-root DIR                Default: repository EVRPTW_Dataset CLE.
  --instance-root DIR           Default: repository EVRPTW_Dataset instances.
  --profile FILE                Default: frozen US reference profile.
  --compression-threads N       Default: 12.
  --compression-level N         zstd level 1..19 (default: 9).

Creation is refused unless CLE, Stage-2, Phase-1, and V2.1 Q90 acceptance pass.
The output contains one EVRPTW_Dataset/ root and no dense matrix cache files.
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

archive=""
checksum=""
while (($#)); do
  case "$1" in
    --archive)
      (($# >= 2)) || fail "--archive requires a value"
      archive="$2"
      shift 2
      ;;
    --sha256-file)
      (($# >= 2)) || fail "--sha256-file requires a value"
      checksum="$2"
      shift 2
      ;;
    --cle-root)
      (($# >= 2)) || fail "--cle-root requires a value"
      CLE_ROOT="$2"
      shift 2
      ;;
    --instance-root)
      (($# >= 2)) || fail "--instance-root requires a value"
      INSTANCE_ROOT="$2"
      shift 2
      ;;
    --profile)
      (($# >= 2)) || fail "--profile requires a value"
      PROFILE_PATH="$2"
      shift 2
      ;;
    --compression-threads)
      (($# >= 2)) || fail "--compression-threads requires a value"
      COMPRESSION_THREADS="$2"
      shift 2
      ;;
    --compression-level)
      (($# >= 2)) || fail "--compression-level requires a value"
      COMPRESSION_LEVEL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
done

[[ -n "$archive" ]] || fail "--archive FILE.tar.zst is required"
[[ "$COMPRESSION_THREADS" =~ ^[1-9][0-9]*$ ]] || fail "--compression-threads must be positive"
[[ "$COMPRESSION_LEVEL" =~ ^([1-9]|1[0-9])$ ]] || fail "--compression-level must be 1..19"
PYTHON_RESOLVED="$(command -v "$PYTHON_BIN" || true)"
[[ -n "$PYTHON_RESOLVED" ]] || fail "Python executable not found: $PYTHON_BIN"
ZSTD_RESOLVED="$(command -v "$ZSTD_BIN" || true)"
[[ -n "$ZSTD_RESOLVED" ]] || fail "zstd executable not found: $ZSTD_BIN"

arguments=(
  create
  --cle-root "$CLE_ROOT"
  --instance-root "$INSTANCE_ROOT"
  --profile "$PROFILE_PATH"
  --archive "$archive"
  --repo-root "$ROOT_DIR"
  --zstd-bin "$ZSTD_RESOLVED"
  --compression-threads "$COMPRESSION_THREADS"
  --compression-level "$COMPRESSION_LEVEL"
)
if [[ -n "$checksum" ]]; then
  arguments+=(--sha256-file "$checksum")
fi
export PYTHONPATH="$ROOT_DIR/EVRPTW_Dataset_Generator/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_RESOLVED" "$TOOL" "${arguments[@]}"
