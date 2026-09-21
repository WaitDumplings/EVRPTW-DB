#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
python_bin="${CURRICULUM_PYTHON:-python}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
args=()
if [[ $# -ge 4 && "$1" =~ ^[0-9]+$ && "$2" =~ ^[0-9]+$ && "$3" =~ ^[0-9]+$ && "$4" =~ ^[0-9]+$ ]]; then
    args+=(--gpus "$1,$2,$3,$4")
    shift 4
fi
foreground=0
for arg in "$@"; do
    case "$arg" in
        --foreground) foreground=1 ;;
        --dry-run|--help|-h) foreground=1; args+=("$arg") ;;
        *) args+=("$arg") ;;
    esac
done
if [[ "$foreground" = 1 ]]; then
    exec "$python_bin" "$HERE/launch.py" "${args[@]}"
fi
log_dir="${CURRICULUM_CUS1000_OUTPUT_ROOT:-/data/curriculum_stage3_cus1000}/launchers/am/$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$log_dir"
if ! "$python_bin" "$HERE/launch.py" --dry-run "${args[@]}" > "$log_dir/preflight.log" 2>&1; then
    cat "$log_dir/preflight.log" >&2
    exit 1
fi
"$python_bin" - "$HERE/launch.py" "$log_dir" "${args[@]}" <<'PYLAUNCH'
from pathlib import Path
import signal,subprocess,sys
entry,logs,*args=sys.argv[1:]
logs=Path(logs)
with (logs/'launcher.log').open('w') as stream:
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    child=subprocess.Popen([sys.executable,entry,*args],stdin=subprocess.DEVNULL,
        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
(logs/'launcher.pid').write_text(str(child.pid)+'\n')
print(f'AM Road Cus1000 four-GPU curriculum: launcher PID {child.pid}')
print(f'Launcher log: {logs / "launcher.log"}')
PYLAUNCH
