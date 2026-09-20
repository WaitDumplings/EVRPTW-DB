#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 GPU0 GPU1 [--dry-run|--foreground] [training options]" >&2
    exit 2
fi
if ! [[ "$1" =~ ^[0-9]+$ && "$2" =~ ^[0-9]+$ ]] || [ "$1" = "$2" ]; then
    echo "Choose two different physical GPU indices, for example: $0 0 1" >&2
    exit 2
fi
first_gpu="$1"
second_gpu="$2"
shift 2
exec "$HERE/_launch.sh" rrnco --gpus "$first_gpu,$second_gpu" "$@"
