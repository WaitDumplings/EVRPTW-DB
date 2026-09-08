#!/usr/bin/env bash
set -euo pipefail
SERVER_SCRIPT_DIR="${SERVER_SCRIPT_DIR:?SERVER_SCRIPT_DIR must be set}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/server_env.sh"
MODE="${1:?usage: run_server.sh full|resume|status}"
shift
SEED_SELECTION="${DRL_SEEDS:-1234}"
REUSE_PREVERIFIED_TRAINING_STREAMS=0
FORWARD_ARGS=()
while (( $# )); do
  case "$1" in
    --seed)
      (( $# >= 2 )) || { echo "--seed requires one integer" >&2; exit 2; }
      SEED_SELECTION="$2"
      shift 2
      ;;
    --seed=*)
      SEED_SELECTION="${1#--seed=}"
      shift
      ;;
    --seeds)
      (( $# >= 2 )) || { echo "--seeds requires a comma-separated list" >&2; exit 2; }
      SEED_SELECTION="$2"
      shift 2
      ;;
    --seeds=*)
      SEED_SELECTION="${1#--seeds=}"
      shift
      ;;
    --reuse-preverified-training-streams)
      (( REUSE_PREVERIFIED_TRAINING_STREAMS == 0 )) || {
        echo "--reuse-preverified-training-streams may be specified only once" >&2
        exit 2
      }
      REUSE_PREVERIFIED_TRAINING_STREAMS=1
      FORWARD_ARGS+=("$1")
      shift
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done
[[ "$SEED_SELECTION" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
  echo "invalid seed selection: $SEED_SELECTION" >&2
  exit 2
}
export DRL_SEEDS="$SEED_SELECTION"
case "$MODE" in
  full|resume)
    if (( REUSE_PREVERIFIED_TRAINING_STREAMS == 0 )); then
      bash "$SCRIPT_DIR/prepare_artifacts.sh"
    fi
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
  "${FORWARD_ARGS[@]}"
