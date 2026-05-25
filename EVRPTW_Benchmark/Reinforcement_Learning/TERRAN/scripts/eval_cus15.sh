#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=(conda run -n maojie python)

EVAL_PATH="${1:-EVRPTW_Dataset/Amazon_Calibrated_v1/Cus_15/CS_3/eval_200}"
N_TRAJ="${2:-50}"
DEVICE="${3:-cuda}"

run_eval() {
  local solver_name="$1"
  local seed="$2"
  local checkpoint="$3"
  local output_dir="$4"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Skip missing checkpoint: $checkpoint" >&2
    return 0
  fi
  "${PYTHON_BIN[@]}" -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.eval \
    --checkpoint-path "$checkpoint" \
    --eval-path "$EVAL_PATH" \
    --output-dir "$output_dir" \
    --solver-name "$solver_name" \
    --seed "$seed" \
    --device "$DEVICE" \
    --num-customers 15 \
    --num-charging-stations 3 \
    --n-traj "$N_TRAJ" \
    --decode-mode sample
}

run_eval \
  TERRAN \
  1234 \
  EVRPTW_Benchmark/Reinforcement_Learning/TERRAN/checkpoints/Cus_15_CS_3/TERRAN/seed_1234/checkpoint_final.pt \
  EVRPTW_Benchmark/results/Amazon_Calibrated_v1/Cus_15/CS_3/TERRAN/seed_1234

run_eval \
  TERRAN \
  2026 \
  EVRPTW_Benchmark/Reinforcement_Learning/TERRAN/checkpoints/Cus_15_CS_3/TERRAN/seed_2026/checkpoint_final.pt \
  EVRPTW_Benchmark/results/Amazon_Calibrated_v1/Cus_15/CS_3/TERRAN/seed_2026

run_eval \
  TERRAN_PBRS \
  1234 \
  EVRPTW_Benchmark/Reinforcement_Learning/TERRAN/checkpoints/Cus_15_CS_3/TERRAN_PBRS/seed_1234/checkpoint_final.pt \
  EVRPTW_Benchmark/results/Amazon_Calibrated_v1/Cus_15/CS_3/TERRAN_PBRS/seed_1234

run_eval \
  TERRAN_PBRS \
  2026 \
  EVRPTW_Benchmark/Reinforcement_Learning/TERRAN/checkpoints/Cus_15_CS_3/TERRAN_PBRS/seed_2026/checkpoint_final.pt \
  EVRPTW_Benchmark/results/Amazon_Calibrated_v1/Cus_15/CS_3/TERRAN_PBRS/seed_2026
