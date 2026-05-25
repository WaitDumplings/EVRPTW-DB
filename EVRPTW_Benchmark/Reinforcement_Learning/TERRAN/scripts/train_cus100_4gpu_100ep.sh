#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=(conda run -n maojie python)
LOG_DIR="EVRPTW_Benchmark/Reinforcement_Learning/TERRAN/logs/launch_cus100_100ep"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --epochs 100
  --num-envs-per-gpu 96
  --n-traj 64
  --rollout-steps 140
  --num-minibatches 16
  --eval-interval 10
  --eval-n-traj 50
  --debug
  --debug-log-every 1
)

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN[@]}" -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.train   --config cus100_terran.yaml   --seed 101   "${COMMON_ARGS[@]}"   > "$LOG_DIR/terran_seed_101.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN[@]}" -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.train   --config cus100_terran.yaml   --seed 202   "${COMMON_ARGS[@]}"   > "$LOG_DIR/terran_seed_202.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 "${PYTHON_BIN[@]}" -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.train   --config cus100_terran_pbrs.yaml   --seed 101   "${COMMON_ARGS[@]}"   > "$LOG_DIR/terran_pbrs_seed_101.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 "${PYTHON_BIN[@]}" -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.train   --config cus100_terran_pbrs.yaml   --seed 202   "${COMMON_ARGS[@]}"   > "$LOG_DIR/terran_pbrs_seed_202.log" 2>&1 &

wait
