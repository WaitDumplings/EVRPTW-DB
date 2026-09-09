#!/usr/bin/env bash
# Run the paired RRNCO experiment on GPUs 1/2 and start all benchmark queues
# as their GPUs become available. Existing jobs must be stopped by their owner.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
RUN_TAG="${RUN_TAG:-rrnco_v2_$(date +%Y%m%dT%H%M%S)}"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,44}$ ]] || { echo "Invalid RUN_TAG" >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHON_BIN
cd "$REPO_ROOT"
[[ -z "$(git status --porcelain)" ]] || { echo "Commit the reviewed experiment source before launching" >&2; exit 2; }
SOURCE_COMMIT="$(git rev-parse HEAD)"
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
[[ "$GPU_COUNT" -eq 4 ]] || { echo "This pipeline requires the 4-GPU 2080ti_4_1 server" >&2; exit 2; }
BUSY="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "$BUSY" ]] || { echo "GPU compute jobs are still active; preserve and stop only the intended old jobs before launching: $BUSY" >&2; exit 3; }
LOG_ROOT="${PIPELINE_LOG_ROOT:-$REPO_ROOT/EVRPTW_Benchmark/results/2080ti_update_20260909/pipeline/$RUN_TAG}"
[[ ! -e "$LOG_ROOT" ]] || { echo "Pipeline log directory must be fresh: $LOG_ROOT" >&2; exit 2; }
mkdir -p "$LOG_ROOT"
printf '%s\n' "$$" > "$LOG_ROOT/supervisor.pid"
printf '%s\n' "$SOURCE_COMMIT" > "$LOG_ROOT/source_commit.txt"
RRNCO_OUTPUT_ROOT="${RRNCO_OUTPUT_ROOT:-$REPO_ROOT/EVRPTW_Benchmark/results/RRNCO_EV_optimized_long_v2/$RUN_TAG}"
export RUN_TAG
# Detached benchmark launchers keep their own PID, lock, and provenance records.
bash "$SCRIPT_DIR/full.sh" --launcher-id "${RUN_TAG}_main" \
  --slots 0,3 --slot-gpu-map 0:0,3:3 > "$LOG_ROOT/benchmark_0_3_launch.log" 2>&1
run_pair_member() {
  local graph_mode="$1" gpu="$2" result=0
  GPU="$gpu" GRAPH_MODE="$graph_mode" OUTPUT_ROOT="$RRNCO_OUTPUT_ROOT" \
    bash "$REPO_ROOT/EVRPTW_Benchmark/Reinforcement_Learning/RRNCO_EVRPTW/run_optimized_long_training.sh" \
    > "$LOG_ROOT/rrnco_${graph_mode}.log" 2>&1 || result=$?
  printf '%s\n' "$result" > "$LOG_ROOT/rrnco_${graph_mode}.exit_code"
  # Refuse to silently run the deferred benchmark from changed source.
  if [[ "$(git rev-parse HEAD)" != "$SOURCE_COMMIT" || -n "$(git status --porcelain)" ]]; then
    echo "Deferred GPU $gpu benchmark needs the original clean source $SOURCE_COMMIT" \
      > "$LOG_ROOT/benchmark_${gpu}_blocked.txt"
    return 4
  fi
  bash "$SCRIPT_DIR/full.sh" --launcher-id "${RUN_TAG}_gpu${gpu}" \
    --slots "$gpu" --slot-gpu-map "$gpu:$gpu" > "$LOG_ROOT/benchmark_${gpu}_launch.log" 2>&1
  return "$result"
}
run_pair_member full 1 &
FULL_WORKER=$!
run_pair_member node_only 2 &
NODE_WORKER=$!
printf '%s\n' "$FULL_WORKER" > "$LOG_ROOT/full_worker.pid"
printf '%s\n' "$NODE_WORKER" > "$LOG_ROOT/node_only_worker.pid"
echo "Started: benchmark GPUs 0/3; RRNCO full/node_only GPUs 1/2; logs=$LOG_ROOT"
full_result=0
node_result=0
wait "$FULL_WORKER" || full_result=$?
wait "$NODE_WORKER" || node_result=$?
printf 'full=%s node_only=%s\n' "$full_result" "$node_result" > "$LOG_ROOT/scheduling_result.txt"
(( full_result == 0 && node_result == 0 ))
