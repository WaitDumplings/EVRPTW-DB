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
# Keep this entry's model and physical cards fixed, including on status/resume.
exec "${cus500_python}" "${script_dir}/launch.py" \
    --config "${script_dir}/configs/rrnco_gpu01.json" "$@" \
    --model rrnco --gpus 0,1
