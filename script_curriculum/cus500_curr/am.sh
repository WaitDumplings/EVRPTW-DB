#!/usr/bin/env bash
# Road Cus100 best -> Road Cus500 AM, two GPUs (0/1), +3000 logical epochs.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
foreground=0
if [ "${1:-}" = --foreground ]; then
    foreground=1
    shift
fi
for arg in "$@"; do
    case "$arg" in --dry-run|--help|-h) foreground=1 ;; esac
done
if [ -z "${CURRICULUM_PYTHON:-}" ]; then
    target_env="${CURRICULUM_CONDA_ENV:-maojie}"
    if [ "${CONDA_DEFAULT_ENV:-}" != "$target_env" ]; then
        conda_cmd="${CONDA_EXE:-conda}"
        if ! conda_base="$("$conda_cmd" info --base)"; then
            echo "Activate $target_env or set CURRICULUM_PYTHON=/path/to/env/bin/python." >&2
            exit 1
        fi
        source "$conda_base/etc/profile.d/conda.sh"
        conda activate "$target_env"
    fi
    export CURRICULUM_PYTHON="$(command -v python)"
fi
# Only the explicit --pull option updates code; long-lived jobs may be using
# this checkout, and the experiment records the current source hashes.
if [ "${1:-}" = --pull ]; then
    git pull --ff-only origin ablation
    shift
fi
if [ "$foreground" = 1 ]; then
    exec "$CURRICULUM_PYTHON" "$HERE/launch.py" "$@"
fi
log_root="${CURRICULUM_CUS500_OUTPUT_ROOT:-/data/curriculum_stage2_cus500}/launchers/am"
log_dir="$log_root/$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$log_dir"
if ! "$CURRICULUM_PYTHON" "$HERE/launch.py" --dry-run "$@" > "$log_dir/preflight.log" 2>&1; then
    cat "$log_dir/preflight.log" >&2
    exit 1
fi
nohup "$CURRICULUM_PYTHON" "$HERE/launch.py" "$@" \
    > "$log_dir/launcher.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$log_dir/launcher.pid"
echo "Submitted AM Road Cus500 two-GPU curriculum; launcher PID=$pid"
echo "Launcher log: $log_dir/launcher.log"
echo "The launcher checks GPU ownership and prints its result directory there."
