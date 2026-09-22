#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
gpus=()
while [ "$#" -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; do
    gpus+=("$1")
    shift
done
if [ "${#gpus[@]}" -ne 2 ] && [ "${#gpus[@]}" -ne 4 ]; then
    echo "Usage: $0 GPU0 GPU1 [GPU2 GPU3] [--dry-run|--foreground] [training options]" >&2
    exit 2
fi
for ((i=0; i<${#gpus[@]}; i++)); do
    for ((j=0; j<i; j++)); do
        if [ "${gpus[i]}" = "${gpus[j]}" ]; then
            echo "Choose distinct physical GPU indices" >&2
            exit 2
        fi
    done
done
gpu_list="$(IFS=,; echo "${gpus[*]}")"
exec "$HERE/_launch.sh" evrptw_rl --gpus "$gpu_list" "$@"
