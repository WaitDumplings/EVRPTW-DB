#!/usr/bin/env bash
# Shared launcher; public scripts supply the method and physical GPU indices.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
method="$1"
shift
foreground=0
pull=0
args=()
for arg in "$@"; do
    case "$arg" in
        --foreground) foreground=1 ;;
        --pull) pull=1 ;;
        --dry-run|--help|-h) foreground=1; args+=("$arg") ;;
        *) args+=("$arg") ;;
    esac
done
if [ -z "${CURRICULUM_PYTHON:-}" ]; then
    target_env="${CURRICULUM_CONDA_ENV:-${CONDA_DEFAULT_ENV:-maojie}}"
    if [ "${CONDA_DEFAULT_ENV:-}" != "$target_env" ]; then
        conda_cmd="${CONDA_EXE:-conda}"
        if ! conda_base="$("$conda_cmd" info --base)"; then
            echo "Activate your training environment or set CURRICULUM_PYTHON=/path/to/env/bin/python." >&2
            exit 1
        fi
        source "$conda_base/etc/profile.d/conda.sh"
        conda activate "$target_env"
    fi
    export CURRICULUM_PYTHON="$(command -v python)"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Code updates are opt-in because other live jobs may use this checkout.
if [ "$pull" = 1 ]; then
    git pull --ff-only origin ablation
fi
if [ "$foreground" = 1 ]; then
    exec "$CURRICULUM_PYTHON" "$HERE/launch.py" --method "$method" "${args[@]}"
fi
log_root="${CURRICULUM_CUS500_OUTPUT_ROOT:-/data/curriculum_stage2_cus500}/launchers/$method"
log_dir="$log_root/$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$log_dir"
if ! "$CURRICULUM_PYTHON" "$HERE/launch.py" --method "$method" --dry-run "${args[@]}" > "$log_dir/preflight.log" 2>&1; then
    cat "$log_dir/preflight.log" >&2
    exit 1
fi
"$CURRICULUM_PYTHON" - "$HERE/launch.py" "$method" "$log_dir" "${args[@]}" <<'PYLAUNCH'
from pathlib import Path
import subprocess
import sys
entry, method, log_path, *args = sys.argv[1:]
logs = Path(log_path)
with (logs / 'launcher.log').open('w') as log:
    child = subprocess.Popen([sys.executable, entry, '--method', method, *args],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, close_fds=True)
(logs / 'launcher.pid').write_text(str(child.pid) + '\n')
print(f'Submitted {method} Road Cus500 distributed curriculum; launcher PID={child.pid}')
print(f"Launcher log: {logs / 'launcher.log'}")
print('The launcher checks GPU ownership and prints its result directory there.')
PYLAUNCH
