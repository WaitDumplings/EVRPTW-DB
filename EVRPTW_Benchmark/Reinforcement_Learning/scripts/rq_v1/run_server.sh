#!/usr/bin/env bash
set -euo pipefail
SERVER_SCRIPT_DIR="${SERVER_SCRIPT_DIR:?SERVER_SCRIPT_DIR must be set}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/server_env.sh"
MODE="${1:?usage: run_server.sh pilot|full|resume|status}"
shift
case "$MODE" in
  pilot|full|resume)
    bash "$SCRIPT_DIR/prepare_artifacts.sh"
    ;;
  status) ;;
  *) echo "invalid mode: $MODE" >&2; exit 2 ;;
esac
cd "$EVRPTW_REPO_ROOT"
exec python -m EVRPTW_Benchmark.Reinforcement_Learning.scripts.drl_job_runtime \
  "$MODE" \
  --manifest "$DRL_MANIFEST" \
  --slots "$DRL_SLOTS" \
  --local-gpu-count "$DRL_LOCAL_GPU_COUNT" \
  --gpu-name-pattern "$DRL_GPU_NAME_PATTERN" \
  --expected-branch "$DRL_EXPECTED_BRANCH" \
  --seeds "$DRL_SEEDS" \
  --scales "$DRL_SCALES" \
  "$@"
