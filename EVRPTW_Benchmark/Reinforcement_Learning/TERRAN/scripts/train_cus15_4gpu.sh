#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=(conda run -n maojie python)

LOG_DIR="EVRPTW_Benchmark/Reinforcement_Learning/TERRAN/logs/launch_cus15"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN[@]}" -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.train \
  --config cus15_terran.yaml \
  --seed 1002 \
  > "$LOG_DIR/terran_seed_1002.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN[@]}" -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.train \
  --config cus15_terran.yaml \
  --seed 1003 \
  > "$LOG_DIR/terran_seed_1003.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 "${PYTHON_BIN[@]}" -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.train \
  --config cus15_terran_pbrs.yaml \
  --seed 1002 \
  > "$LOG_DIR/terran_pbrs_seed_1002.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 "${PYTHON_BIN[@]}" -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.train \
  --config cus15_terran_pbrs.yaml \
  --seed 1003 \
  > "$LOG_DIR/terran_pbrs_seed_1003.log" 2>&1 &

wait
