#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd -- "${script_dir}/.." && pwd)"
repo_dir="$(cd -- "${experiment_dir}/../../../.." && pwd)"
cus1000_python="${CUS1000_PYTHON:-${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}}"
if [[ -z "${cus1000_python}" ]]; then
    cus1000_python="$(command -v python3 || command -v python)"
fi
if [[ ! -x "${cus1000_python}" ]]; then
    echo "Python not executable: ${cus1000_python}; activate your conda environment or set CUS1000_PYTHON." >&2
    exit 1
fi
cd -- "${repo_dir}"
# Keep the measured deployment recipe fixed, including on status/resume.
exec "${cus1000_python}" "${experiment_dir}/launch.py" \
    --config "${experiment_dir}/config.json" "$@" \
    --model evrptw_rl --gpus 0,1,2,3 --batch-size 12 --accumulation-steps 1
