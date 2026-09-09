#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/dataset_root.sh"
DATASET_ROOT="$(resolve_evrptw_dataset_root "$REPO_ROOT")"

SEED="${SEED:-1234}"
EPOCHS="${EPOCHS:-500}"
RRNCO_GPU="${RRNCO_GPU:-1}"
AM_GPU="${AM_GPU:-2}"
VALIDATION_LIMIT="${VALIDATION_LIMIT:-500}"
RRNCO_LEARNING_RATE="${RRNCO_LEARNING_RATE:-0.0001}"
RRNCO_WEIGHT_DECAY="${RRNCO_WEIGHT_DECAY:-0.01}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/EVRPTW_Benchmark/results/RRNCO_EV_Cus100_screen_$(date +%Y%m%dT%H%M%S)}"

TRAIN_INDEX="$DATASET_ROOT/generation_plan/core/train/view_index.parquet"
VAL_INDEX="$DATASET_ROOT/generation_plan/core/val/view_index.parquet"
FAMILY_ROOT="$DATASET_ROOT/materialized/families"
for required in "$TRAIN_INDEX" "$VAL_INDEX" "$FAMILY_ROOT"; do
  [[ -e "$required" ]] || { echo "Missing required dataset path: $required" >&2; exit 1; }
done

RRNCO_OUT="$OUTPUT_ROOT/rrnco_ev_seed${SEED}"
AM_OUT="$OUTPUT_ROOT/am_evrptw_seed${SEED}"
[[ ! -e "$RRNCO_OUT" && ! -e "$AM_OUT" ]] || {
  echo "Comparison output must be fresh: $OUTPUT_ROOT" >&2
  exit 1
}
mkdir -p "$OUTPUT_ROOT"

common=(
  --dataset-path "$TRAIN_INDEX"
  --family-root "$FAMILY_ROOT"
  --scale Cus100 --split-ids train --track-ids train
  --seed "$SEED" --device cuda
  --training-epochs "$EPOCHS" --minimum-training-epochs "$EPOCHS"
  --training-rollout-steps 120 --validation-rollout-steps 180
  --physical-batch-size 4 --effective-batch-size 4 --batch-size 4
  --samples-per-instance 5 --baseline-eval-size 0
  --validation-dataset-path "$VAL_INDEX"
  --validation-family-root "$FAMILY_ROOT"
  --validation-limit "$VALIDATION_LIMIT"
  --validation-decode-type sampling --validation-candidates 100
  --validation-seed $((910000000 + SEED))
  --validation-every-epochs 100
  --post-minimum-validation-every-epochs 100
  --validation-checkpoints $(((EPOCHS + 99) / 100))
  --protocol-id rrnco_ev_cus100_screen_v3_full_val
  --objective-config "$REPO_ROOT/EVRPTW_Benchmark/Reinforcement_Learning/configs/rivian_energy_vehicle_cost_v2.json"
  --reward-contract "$REPO_ROOT/EVRPTW_Benchmark/Reinforcement_Learning/configs/drl_reward_contract_energy_vehicle_v3.json"
  --optimizer adamw
)

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$RRNCO_GPU" python -m \
  EVRPTW_Benchmark.Reinforcement_Learning.RRNCO_EVRPTW.train \
  "${common[@]}" --learning-rate "$RRNCO_LEARNING_RATE" --weight-decay "$RRNCO_WEIGHT_DECAY" \
  --output-dir "$RRNCO_OUT" >"$OUTPUT_ROOT/rrnco.log" 2>&1 &
rrnco_pid=$!
CUDA_VISIBLE_DEVICES="$AM_GPU" python -m \
  EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.train \
  "${common[@]}" --learning-rate 0.0001 --weight-decay 0.01 \
  --output-dir "$AM_OUT" >"$OUTPUT_ROOT/am.log" 2>&1 &
am_pid=$!

printf 'RRNCO-EV pid=%s GPU=%s\nAM-EVRPTW pid=%s GPU=%s\noutput=%s\n' \
  "$rrnco_pid" "$RRNCO_GPU" "$am_pid" "$AM_GPU" "$OUTPUT_ROOT"
rrnco_rc=0
am_rc=0
wait "$rrnco_pid" || rrnco_rc=$?
wait "$am_pid" || am_rc=$?
printf 'RRNCO-EV exit=%s; AM-EVRPTW exit=%s\n' "$rrnco_rc" "$am_rc"
[[ "$rrnco_rc" -eq 0 && "$am_rc" -eq 0 ]]
