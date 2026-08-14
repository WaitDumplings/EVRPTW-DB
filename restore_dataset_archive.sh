#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL="$ROOT_DIR/EVRPTW_Dataset_Generator/scripts/dataset_archive_tool.py"
DEFAULT_DESTINATION="/data"
DEFAULT_WORKERS=12
DEFAULT_FAMILIES_PER_TASK=25

usage() {
  cat <<'EOF'
Safely unpack a slim EVRPTW dataset archive and restore all matrix families.

Usage:
  restore_dataset_archive.sh start --archive FILE.tar.zst [options]
  restore_dataset_archive.sh status [--destination DIR]
  restore_dataset_archive.sh logs [--destination DIR] [--follow]
  restore_dataset_archive.sh wait [--destination DIR]

Start options:
  --archive FILE                 Required slim .tar.zst archive.
  --sha256-file FILE             Default: FILE.sha256 (required).
  --destination DIR              Parent for EVRPTW_Dataset (default: /data).
  --workers N                    Matrix restore processes (default: 12).
  --families-per-worker-task N   Family chunk size (default: 25).
  --foreground                   Run synchronously instead of in tmux.

The default start command returns after creating a persistent tmux job.  It
never overwrites an unrelated destination tree.  Re-running start for the same
archive safely reuses an extracted slim tree and all complete matrix families.
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

resolve_command() {
  local variable_name="$1"
  local candidate="$2"
  local resolved
  if [[ "$candidate" == */* ]]; then
    [[ -x "$candidate" ]] || fail "Executable not found: $candidate"
    resolved="$(cd "$(dirname "$candidate")" && pwd)/$(basename "$candidate")"
  else
    resolved="$(command -v "$candidate" || true)"
    [[ -n "$resolved" ]] || fail "Required command not found: $candidate"
  fi
  printf -v "$variable_name" '%s' "$resolved"
}

canonical_destination() {
  "$PYTHON_RESOLVED" -c \
    'import sys; from pathlib import Path; print(Path(sys.argv[1]).expanduser().resolve())' \
    "$1"
}

check_restore_python_dependencies() {
  local generator_src="$ROOT_DIR/EVRPTW_Dataset_Generator/src"
  local import_error
  if ! import_error="$(
    PYTHONPATH="$generator_src${PYTHONPATH:+:$PYTHONPATH}" \
      "$PYTHON_RESOLVED" -c \
      'from evrptw_stage2.reconstruction import restore_dataset_matrices' 2>&1
  )"; then
    [[ -z "$import_error" ]] || printf '%s\n' "$import_error" >&2
    fail "Python restore dependencies are unavailable for $PYTHON_RESOLVED. Run: $PYTHON_RESOLVED -m pip install -r $ROOT_DIR/requirements.txt"
  fi
}

job_paths() {
  DESTINATION_RESOLVED="$(canonical_destination "$1")"
  JOB_DIR="$DESTINATION_RESOLVED/.evrptw_restore_us11city"
  local destination_hash
  destination_hash="$(printf '%s' "$DESTINATION_RESOLVED" | sha256sum | cut -c1-12)"
  SESSION="evrptw-restore-$destination_hash"
}

resolve_command PYTHON_RESOLVED "${PYTHON_BIN:-python}"

command_name="${1:-}"
if [[ -z "$command_name" || "$command_name" == "-h" || "$command_name" == "--help" ]]; then
  usage
  exit 0
fi
shift

case "$command_name" in
  start)
    archive=""
    checksum=""
    destination="$DEFAULT_DESTINATION"
    workers="$DEFAULT_WORKERS"
    families_per_task="$DEFAULT_FAMILIES_PER_TASK"
    foreground=0
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
        --destination)
          (($# >= 2)) || fail "--destination requires a value"
          destination="$2"
          shift 2
          ;;
        --workers)
          (($# >= 2)) || fail "--workers requires a value"
          workers="$2"
          shift 2
          ;;
        --families-per-worker-task)
          (($# >= 2)) || fail "--families-per-worker-task requires a value"
          families_per_task="$2"
          shift 2
          ;;
        --foreground)
          foreground=1
          shift
          ;;
        -h|--help)
          usage
          exit 0
          ;;
        *)
          fail "Unknown start option: $1"
          ;;
      esac
    done
    [[ -n "$archive" ]] || fail "start requires --archive FILE.tar.zst"
    [[ -f "$archive" ]] || fail "Archive not found: $archive"
    if [[ -z "$checksum" ]]; then
      checksum="$archive.sha256"
    fi
    [[ -f "$checksum" ]] || fail "Checksum sidecar not found: $checksum"
    [[ "$workers" =~ ^[1-9][0-9]*$ ]] || fail "--workers must be positive"
    [[ "$families_per_task" =~ ^[1-9][0-9]*$ ]] || \
      fail "--families-per-worker-task must be positive"
    check_restore_python_dependencies
    resolve_command ZSTD_RESOLVED "${ZSTD_BIN:-zstd}"
    resolve_command FLOCK_RESOLVED "${FLOCK_BIN:-flock}"
    if [[ "$foreground" -eq 0 ]]; then
      resolve_command TMUX_RESOLVED "${TMUX_BIN:-tmux}"
    else
      TMUX_RESOLVED="$(command -v "${TMUX_BIN:-tmux}" || true)"
    fi
    job_paths "$destination"
    [[ "$DESTINATION_RESOLVED" != "/" ]] || fail "Refusing to use / as destination"
    mkdir -p "$JOB_DIR"
    exec 9>"$JOB_DIR/launcher.lock"
    "$FLOCK_RESOLVED" -n 9 || \
      fail "Another archive launcher is active for $DESTINATION_RESOLVED"

    if [[ -n "$TMUX_RESOLVED" ]] && \
      "$TMUX_RESOLVED" has-session -t "$SESSION" 2>/dev/null; then
      [[ -f "$JOB_DIR/job.json" ]] || \
        fail "tmux session name collision without a matching restore job: $SESSION"
      configured_archive="$("$PYTHON_RESOLVED" -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["archive"])' \
        "$JOB_DIR/job.json")"
      requested_archive="$("$PYTHON_RESOLVED" -c \
        'import sys; from pathlib import Path; print(Path(sys.argv[1]).expanduser().resolve())' \
        "$archive")"
      [[ "$configured_archive" == "$requested_archive" ]] || \
        fail "A different archive restore is already running for $DESTINATION_RESOLVED"
      echo "Restore is already running in tmux session: $SESSION"
      "$PYTHON_RESOLVED" "$TOOL" status --job-dir "$JOB_DIR" || true
      exit 0
    fi

    "$PYTHON_RESOLVED" "$TOOL" init \
      --archive "$archive" \
      --sha256-file "$checksum" \
      --destination "$DESTINATION_RESOLVED" \
      --repo-root "$ROOT_DIR" \
      --job-dir "$JOB_DIR" \
      --python-bin "$PYTHON_RESOLVED" \
      --zstd-bin "$ZSTD_RESOLVED" \
      --workers "$workers" \
      --families-per-worker-task "$families_per_task" \
      --session "$SESSION"

    log_file="$JOB_DIR/restore.log"
    touch "$log_file"
    if [[ "$foreground" -eq 1 ]]; then
      echo "Running restore in the foreground. Log: $log_file"
      set +e
      "$PYTHON_RESOLVED" "$TOOL" run --job-dir "$JOB_DIR" 2>&1 | tee -a "$log_file"
      result=${PIPESTATUS[0]}
      set -e
      exit "$result"
    fi

    printf -v worker_command 'exec %q %q run --job-dir %q >>%q 2>&1' \
      "$PYTHON_RESOLVED" "$TOOL" "$JOB_DIR" "$log_file"
    # Do not leak the launcher flock into a newly created tmux server. The
    # parent shell retains it until new-session has returned.
    "$TMUX_RESOLVED" new-session -d -s "$SESSION" "$worker_command" 9>&-
    echo "Restore started in background."
    echo "  tmux session: $SESSION"
    echo "  status: ./restore_dataset_archive.sh status --destination $(printf '%q' "$DESTINATION_RESOLVED")"
    echo "  logs:   ./restore_dataset_archive.sh logs --destination $(printf '%q' "$DESTINATION_RESOLVED") --follow"
    ;;

  status|logs|wait)
    destination="$DEFAULT_DESTINATION"
    follow=0
    while (($#)); do
      case "$1" in
        --destination)
          (($# >= 2)) || fail "--destination requires a value"
          destination="$2"
          shift 2
          ;;
        --follow)
          [[ "$command_name" == "logs" ]] || fail "--follow is only valid with logs"
          follow=1
          shift
          ;;
        -h|--help)
          usage
          exit 0
          ;;
        *)
          fail "Unknown $command_name option: $1"
          ;;
      esac
    done
    job_paths "$destination"
    [[ -d "$JOB_DIR" ]] || fail "No restore job exists for $DESTINATION_RESOLVED"
    if [[ "$command_name" != "logs" ]]; then
      TMUX_RESOLVED="$(command -v "${TMUX_BIN:-tmux}" || true)"
    fi
    if [[ "$command_name" == "status" ]]; then
      tmux_running=0
      if [[ -n "$TMUX_RESOLVED" ]] && \
        "$TMUX_RESOLVED" has-session -t "$SESSION" 2>/dev/null; then
        echo "tmux: running ($SESSION)"
        tmux_running=1
      else
        echo "tmux: not running ($SESSION)"
      fi
      set +e
      "$PYTHON_RESOLVED" "$TOOL" status --job-dir "$JOB_DIR"
      status_result=$?
      phase="$("$PYTHON_RESOLVED" "$TOOL" status --job-dir "$JOB_DIR" --field phase 2>/dev/null)"
      set -e
      if [[ "$tmux_running" -eq 0 ]]; then
        case "$phase" in
          succeeded|failed|interrupted) ;;
          *)
            echo "ERROR: restore worker stopped before recording a terminal phase" >&2
            exit 1
            ;;
        esac
      fi
      exit "$status_result"
    elif [[ "$command_name" == "logs" ]]; then
      log_file="$JOB_DIR/restore.log"
      [[ -f "$log_file" ]] || fail "Restore log does not exist yet: $log_file"
      if [[ "$follow" -eq 1 ]]; then
        exec tail -n 100 -F "$log_file"
      else
        tail -n 100 "$log_file"
      fi
    else
      last_phase=""
      while true; do
        phase="$("$PYTHON_RESOLVED" "$TOOL" status --job-dir "$JOB_DIR" --field phase || true)"
        if [[ "$phase" != "$last_phase" ]]; then
          echo "Restore phase: $phase"
          last_phase="$phase"
        fi
        case "$phase" in
          succeeded) exit 0 ;;
          failed|interrupted) exit 1 ;;
        esac
        if [[ -z "$TMUX_RESOLVED" ]] || \
          ! "$TMUX_RESOLVED" has-session -t "$SESSION" 2>/dev/null; then
          # The worker writes its terminal state atomically before exiting, so
          # re-read after observing the tmux session disappear. This closes the
          # race where a successful job ended between the first read and the
          # has-session check above.
          phase="$("$PYTHON_RESOLVED" "$TOOL" status --job-dir "$JOB_DIR" --field phase || true)"
          case "$phase" in
            succeeded) exit 0 ;;
            failed|interrupted) exit 1 ;;
            *)
              echo "Restore worker is no longer running; inspect status and logs." >&2
              exit 1
              ;;
          esac
        fi
        sleep 5
      done
    fi
    ;;

  *)
    fail "Unknown command: $command_name (expected start, status, logs, or wait)"
    ;;
esac
