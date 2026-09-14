#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cus500_python="${CUS500_PYTHON:-${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}}"
if [[ -z "${cus500_python}" ]]; then
    cus500_python="$(command -v python3 || command -v python)"
fi
if [[ ! -x "${cus500_python}" ]]; then
    echo "Python not executable: ${cus500_python}; activate your conda environment or set CUS500_PYTHON." >&2
    exit 1
fi
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
# A separate output keeps this fresh mean run apart from the old sum run.
cus500_output="${CUS500_EVRPTW_GPU23_OUTPUT_ROOT:-${repo_root}/EVRPTW_Benchmark/results/cus500_evrptw_rl_mean_gpu23_20260914}"
# Use the new config defaults; explicit CLI batch/cache overrides still work.
unset CUS500_BATCH_SIZE CUS500_ACCUMULATION_STEPS CUS500_INSTANCE_CACHE_SIZE
# Both cards train one shared model; the command never launches the watcher.
exec "${cus500_python}" "${script_dir}/launch.py" \
    --output-root "${cus500_output}" "$@" \
    --config "${script_dir}/config_gpu23_mean.json" \
    --model evrptw_rl --gpus 2,3
