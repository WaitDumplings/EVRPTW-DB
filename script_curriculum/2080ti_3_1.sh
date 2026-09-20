#!/usr/bin/env bash
# RRNCO G/E on GPU 0/1.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
SERVER=2080ti_3_1
DEFAULT_ENV=caliroute
MODE="${1:-run}"
if [ "$#" -gt 1 ] || { [ "$MODE" != run ] && [ "$MODE" != --dry-run ] && [ "$MODE" != --help ]; }; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi
if [ "$MODE" = --help ]; then
    echo "RRNCO G/E on GPU 0/1."
    echo "Usage: $0 [--dry-run]"
    echo "Default conda env: $DEFAULT_ENV; override CURRICULUM_CONDA_ENV or CURRICULUM_PYTHON."
    echo "Pulls origin/ablation first; CURRICULUM_SKIP_PULL=1 uses the current checkout."
    exit 0
fi
if [ "${CURRICULUM_SKIP_PULL:-0}" != 1 ]; then
    git pull --ff-only origin ablation
fi
if [ -z "${CURRICULUM_PYTHON:-}" ]; then
    target_env="${CURRICULUM_CONDA_ENV:-$DEFAULT_ENV}"
    if [ "${CONDA_DEFAULT_ENV:-}" != "$target_env" ]; then
        conda_cmd="${CONDA_EXE:-conda}"
        if ! conda_base="$("$conda_cmd" info --base)"; then
            echo "Activate conda env $target_env or set CURRICULUM_PYTHON=/path/to/env/bin/python." >&2
            exit 1
        fi
        source "$conda_base/etc/profile.d/conda.sh"
        conda activate "$target_env"
    fi
    export CURRICULUM_PYTHON="$(command -v python)"
fi
export CURRICULUM_OUTPUT_ROOT="${CURRICULUM_OUTPUT_ROOT:-/data/curriculum_stage1}"
export CURRICULUM_CKPT_ROOT="${CURRICULUM_CKPT_ROOT:-/data/best_ckpt}"
models=(rrnco rrnco)
domains=(G E)
gpus=(0 1)
log_dir="$CURRICULUM_OUTPUT_ROOT/launchers/$SERVER/$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$log_dir"
echo "Server: $SERVER; launcher logs: $log_dir"

# Verify every source before submitting any of this server's tasks.
for i in "${!models[@]}"; do
    echo "Check ${models[$i]} ${domains[$i]} -> GPU ${gpus[$i]}"
    if ! "$HERE/${models[$i]}.sh" "${domains[$i]}" "${gpus[$i]}" --dry-run \
        > "$log_dir/${models[$i]}_${domains[$i]}.preflight.log" 2>&1; then
        cat "$log_dir/${models[$i]}_${domains[$i]}.preflight.log" >&2
        exit 1
    fi
done
# Physical GPU IDs, respecting CUDA_VISIBLE_DEVICES; per-job locks are checked
# again by the actual launcher to handle races with other submissions.
"$CURRICULUM_PYTHON" - "${gpus[@]}" <<'PY_GPU'
import sys
import pandas, pyarrow, torch, yaml  # Validate the selected environment before launch.
if not torch.cuda.is_available():
    raise SystemExit('CUDA is not available in the selected Python environment')
from script_curriculum.launch import compute_busy_uuids, gpu_inventory
available = {gpu['index']: gpu for gpu in gpu_inventory()}
busy = compute_busy_uuids()
for raw in sys.argv[1:]:
    index = int(raw)
    if index not in available:
        raise SystemExit(f'GPU {index} unavailable or excluded by CUDA_VISIBLE_DEVICES')
    if available[index]['uuid'] in busy:
        raise SystemExit(f'GPU {index} occupied; no tasks from this server script were started')
print('Assigned GPUs are available.')
PY_GPU
if [ "$MODE" = --dry-run ]; then
    echo "Preflight passed. No training started. Logs: $log_dir"
    exit 0
fi
printf 'model\tdomain\tgpu\tlauncher_pid\tlog_path\n' > "$log_dir/jobs.tsv"
for i in "${!models[@]}"; do
    log="$log_dir/${models[$i]}_${domains[$i]}.log"
    nohup "$HERE/${models[$i]}.sh" "${domains[$i]}" "${gpus[$i]}" \
        > "$log" 2>&1 < /dev/null &
    pid=$!
    printf '%s\t%s\t%s\t%s\t%s\n' "${models[$i]}" "${domains[$i]}" "${gpus[$i]}" "$pid" "$log" >> "$log_dir/jobs.tsv"
    echo "Submitted ${models[$i]} ${domains[$i]} on GPU ${gpus[$i]}; launcher PID=$pid"
done
echo "Each task runs +2000 epochs, validates every 100; see jobs.tsv and per-task logs in $log_dir"
