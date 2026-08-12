#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_METHOD="${DATA_METHOD:-stage2}"
if [[ "${1:-}" == "stage2" || "${1:-}" == "restore" ]]; then
  DATA_METHOD="$1"
  shift
fi

case "$DATA_METHOD" in
  stage2)
    if [[ "${SKIP_CLE_BUILD:-0}" != "1" ]]; then
      "$ROOT_DIR/generate_cle.sh"
    fi
    INSTANCE_METHOD=stage2 "$ROOT_DIR/generate_instances.sh" "$@"
    ;;
  restore)
    # The caller supplies a portable CLE at CLE_ROOT and a slim parameter tree
    # at INSTANCE_OUTPUT_ROOT. Optional --view-id/--view-id-file arguments
    # reconstruct only the selected parent families.
    INSTANCE_METHOD=restore "$ROOT_DIR/generate_instances.sh" "$@"
    ;;
  *)
    echo "Usage: $0 {stage2|restore} [Stage-2/restore arguments]" >&2
    exit 2
    ;;
esac
