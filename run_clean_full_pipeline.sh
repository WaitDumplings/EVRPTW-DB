#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATOR_DIR="$ROOT_DIR/EVRPTW_Dataset_Generator"
PYTHON_BIN="${PYTHON_BIN:-/home/npg/miniconda3/envs/evrptw-cle/bin/python}"
CLE_ROOT="${CLE_ROOT:-/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset/CLE_v2/us_11city}"
AMAZON_ARTIFACT_ROOT="${AMAZON_ARTIFACT_ROOT:-/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset/Calibration_v2/amazon_stage2_v3}"
PROFILE="$GENERATOR_DIR/configs/us_reference_instance_profile_v2_release.json"
COHORT_SPLIT="$GENERATOR_DIR/configs/amazon_cohort_split_v1.json"
ACCEPTANCE_CONFIG="$GENERATOR_DIR/configs/stage2_acceptance_v3_full_7500.json"
WORKERS="${WORKERS:-30}"
FAMILIES_PER_TASK="${FAMILIES_PER_WORKER_TASK:-1}"
C3_FAMILIES_PER_TASK="${C3_FAMILIES_PER_TASK:-25}"
RESTORE_VALIDATION=1
PUSH_AFTER_SUCCESS=1
OUTPUT_ROOT=""
ARCHIVE=""
RESTORE_DESTINATION=""

usage() {
  cat <<'EOF'
Run the clean EVRPTW-DB full pipeline from a fresh Stage-2 root.

Usage:
  run_clean_full_pipeline.sh --output-root DIR --archive FILE.tar.zst \
    --restore-destination DIR [--no-restore] [--no-push]

The existing frozen 11-city CLE is treated as the unchanged master. The fresh
run starts at Stage-2 C0 and executes C0/C1-C2 preflight, C3, 7,500-family
materialization, full verification/feasibility, Phase-1, construct-valid v3,
slim archive creation/inspection, and (by default) a full matrix restore test.
No SHA or file-hash validation is performed.
EOF
}

fail() { echo "ERROR: $*" >&2; exit 2; }

while (($#)); do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --archive) ARCHIVE="$2"; shift 2 ;;
    --restore-destination) RESTORE_DESTINATION="$2"; shift 2 ;;
    --no-restore) RESTORE_VALIDATION=0; shift ;;
    --no-push) PUSH_AFTER_SUCCESS=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

[[ -n "$OUTPUT_ROOT" ]] || fail "--output-root is required"
[[ -n "$ARCHIVE" ]] || fail "--archive is required"
if [[ "$RESTORE_VALIDATION" == 1 ]]; then
  [[ -n "$RESTORE_DESTINATION" ]] || fail "--restore-destination is required"
fi
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || fail "WORKERS must be positive"
[[ "$FAMILIES_PER_TASK" == 1 ]] || fail "full generation requires one family per worker task"
[[ ! -e "$OUTPUT_ROOT" ]] || fail "fresh output root already exists: $OUTPUT_ROOT"
[[ ! -e "$ARCHIVE" ]] || fail "archive already exists: $ARCHIVE"
if [[ "$RESTORE_VALIDATION" == 1 ]]; then
  [[ ! -e "$RESTORE_DESTINATION/EVRPTW_Dataset" ]] || \
    fail "restore dataset target already exists: $RESTORE_DESTINATION/EVRPTW_Dataset"
  [[ ! -e "$RESTORE_DESTINATION/.evrptw_restore_us11city" ]] || \
    fail "restore job already exists: $RESTORE_DESTINATION/.evrptw_restore_us11city"
fi
[[ -d "$CLE_ROOT/cities" ]] || fail "frozen CLE master is missing: $CLE_ROOT"
[[ -f "$AMAZON_ARTIFACT_ROOT/manifest.json" ]] || fail "Amazon artifact is missing"
[[ -x "$PYTHON_BIN" ]] || fail "Python environment is unavailable: $PYTHON_BIN"

LOG_ROOT="${PIPELINE_LOG_ROOT:-${OUTPUT_ROOT}.pipeline_logs}"
[[ ! -e "$LOG_ROOT" ]] || fail "pipeline log root already exists: $LOG_ROOT"
mkdir -p "$LOG_ROOT" "$(dirname "$ARCHIVE")"
STATUS_JSON="$LOG_ROOT/clean_full_pipeline_status.json"
CURRENT_STEP="initializing"
STARTED_EPOCH="$(date +%s)"

write_status() {
  local status="$1"
  local message="$2"
  "$PYTHON_BIN" - "$STATUS_JSON" "$status" "$CURRENT_STEP" "$message" \
    "$OUTPUT_ROOT" "$ARCHIVE" "$RESTORE_DESTINATION" "$STARTED_EPOCH" <<'PY_STATUS'
import json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
path=Path(sys.argv[1]); status,step,message=sys.argv[2:5]
payload={
  "schema":"evrptw_clean_full_pipeline_status_v1",
  "status":status,
  "current_step":step,
  "message":message,
  "instance_root":sys.argv[5],
  "archive":sys.argv[6],
  "restore_destination":sys.argv[7],
  "started_epoch_s":int(sys.argv[8]),
  "elapsed_seconds":time.time()-int(sys.argv[8]),
  "updated_at":datetime.now(timezone.utc).isoformat(),
  "workers":30,
  "file_hash_validation_performed":False,
}
path.parent.mkdir(parents=True,exist_ok=True)
tmp=path.with_name(f".{path.name}.tmp-{os.getpid()}")
with tmp.open("w",encoding="utf-8") as h:
 json.dump(payload,h,indent=2,sort_keys=True,ensure_ascii=False); h.write("\n")
 h.flush(); os.fsync(h.fileno())
os.replace(tmp,path)
PY_STATUS
}

failed() {
  local code=$?
  write_status failed "step failed with exit code $code"
  exit "$code"
}
trap failed ERR INT TERM

run_step() {
  CURRENT_STEP="$1"
  shift
  write_status running "started"
  local step_started
  step_started="$(date +%s)"
  "$@" >"$LOG_ROOT/${CURRENT_STEP}.log" 2>&1
  local elapsed=$(( $(date +%s) - step_started ))
  printf '%s\t%s\n' "$CURRENT_STEP" "$elapsed" >> "$LOG_ROOT/step_timings.tsv"
  write_status running "completed in ${elapsed}s"
}

cd "$ROOT_DIR"
run_step code_preflight "$PYTHON_BIN" "$GENERATOR_DIR/scripts/check_candidate_revision.py" \
  --repo-root "$ROOT_DIR" --require-branch stage2-repair-candidate
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || fail "repository is not clean"
run_step tests env PYTHONPATH="$GENERATOR_DIR/src" "$PYTHON_BIN" -m pytest -q \
  "$GENERATOR_DIR/tests"

run_step stage2_from_zero env \
  PYTHON_BIN="$PYTHON_BIN" INSTANCE_MODE=official WORKERS="$WORKERS" \
  C3_WORKERS="$WORKERS" C3_FAMILIES_PER_TASK="$C3_FAMILIES_PER_TASK" \
  FAMILIES_PER_WORKER_TASK="$FAMILIES_PER_TASK" \
  CLE_ROOT="$CLE_ROOT" INSTANCE_OUTPUT_ROOT="$OUTPUT_ROOT" \
  AMAZON_ARTIFACT_ROOT="$AMAZON_ARTIFACT_ROOT" \
  REFERENCE_PROFILE="$PROFILE" \
  "$ROOT_DIR/generate_instances.sh" --full-run-approved

run_step feasibility_gate "$PYTHON_BIN" \
  "$GENERATOR_DIR/scripts/watch_full_corpus_feasibility.py" \
  --root "$OUTPUT_ROOT" --expected 7500 --poll-seconds 5

run_step construct_valid_v3 env PYTHONPATH="$GENERATOR_DIR/src" "$PYTHON_BIN" \
  "$GENERATOR_DIR/scripts/evaluate_stage2_construct_valid_acceptance_v3.py" \
  --instance-root "$OUTPUT_ROOT" \
  --amazon-artifact-root "$AMAZON_ARTIFACT_ROOT" \
  --cohort-split "$COHORT_SPLIT" \
  --config "$ACCEPTANCE_CONFIG" \
  --acceptance-output "$OUTPUT_ROOT/reports/stage2_repair/stage2_acceptance_v3_construct_valid.json" \
  --diagnostics-output "$OUTPUT_ROOT/reports/stage2_repair/amazon_operational_diagnostics_v3.json"

run_step archive_create env PYTHON_BIN="$PYTHON_BIN" COMPRESSION_THREADS="$WORKERS" \
  CLE_ROOT="$CLE_ROOT" INSTANCE_ROOT="$OUTPUT_ROOT" PROFILE_PATH="$PROFILE" \
  "$ROOT_DIR/create_dataset_archive.sh" --archive "$ARCHIVE" \
  --compression-threads "$WORKERS"
run_step archive_inspect env PYTHONPATH="$GENERATOR_DIR/src" "$PYTHON_BIN" \
  "$GENERATOR_DIR/scripts/dataset_archive_tool.py" inspect \
  --archive "$ARCHIVE" --zstd-bin "$(command -v zstd)"

if [[ "$RESTORE_VALIDATION" == 1 ]]; then
  mkdir -p "$RESTORE_DESTINATION"
  run_step full_restore env PYTHON_BIN="$PYTHON_BIN" \
    "$ROOT_DIR/restore_dataset_archive.sh" start \
    --archive "$ARCHIVE" --destination "$RESTORE_DESTINATION" \
    --workers "$WORKERS" --families-per-worker-task 25 --foreground
fi

[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || fail "repository became dirty"
if [[ "$PUSH_AFTER_SUCCESS" == 1 ]]; then
  run_step github_push git push origin stage2-repair-candidate
fi
CURRENT_STEP="complete"
write_status passed "clean full pipeline completed"
trap - ERR INT TERM
printf 'Complete. Status: %s\n' "$STATUS_JSON"
printf 'Instances: %s\n' "$OUTPUT_ROOT"
printf 'Archive: %s\n' "$ARCHIVE"
