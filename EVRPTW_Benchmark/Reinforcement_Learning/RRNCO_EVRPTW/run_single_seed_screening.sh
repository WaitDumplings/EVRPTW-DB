#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/dataset_root.sh"
DATASET_ROOT="$(resolve_evrptw_dataset_root "$REPO_ROOT")"

SCALE="${SCALE:-Cus100}"
SEED="${SEED:-1234}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-}"
TRAIN_TRAJECTORIES="${TRAIN_TRAJECTORIES:-5}"
VALIDATION_LIMIT="${VALIDATION_LIMIT:-500}"
VALIDATION_CANDIDATES="${VALIDATION_CANDIDATES:-100}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

case "$SCALE" in
  Cus50)
    PLAN="compatibility_cus50"
    TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-65}"
    VALIDATION_MAX_STEPS="${VALIDATION_MAX_STEPS:-98}"
    DEFAULT_BATCH_SIZE=24
    ;;
  Cus100)
    PLAN="core"
    TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-120}"
    VALIDATION_MAX_STEPS="${VALIDATION_MAX_STEPS:-180}"
    DEFAULT_BATCH_SIZE=6
    ;;
  *)
    echo "This frozen screening launcher supports only Cus50 or Cus100: $SCALE" >&2
    exit 2
    ;;
esac
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"

TRAIN_INDEX="$DATASET_ROOT/generation_plan/$PLAN/train/view_index.parquet"
VAL_INDEX="$DATASET_ROOT/generation_plan/$PLAN/val/view_index.parquet"
FAMILY_ROOT="$DATASET_ROOT/materialized/families"
for required in "$TRAIN_INDEX" "$VAL_INDEX" "$FAMILY_ROOT"; do
  [[ -e "$required" ]] || { echo "Missing required dataset path: $required" >&2; exit 1; }
done

EXECUTABLE_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/EVRPTW_Benchmark/results/RRNCO_EV_single_seed_screening_v1/$SCALE/seed_${SEED}/${EXECUTABLE_COMMIT}}"
[[ ! -e "$OUTPUT_DIR" ]] || {
  echo "Output must be fresh; refusing overwrite: $OUTPUT_DIR" >&2
  exit 1
}
mkdir -p "$OUTPUT_DIR"

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPU" python -m \
  EVRPTW_Benchmark.Reinforcement_Learning.RRNCO_EVRPTW.train \
  --dataset-path "$TRAIN_INDEX" \
  --family-root "$FAMILY_ROOT" \
  --scale "$SCALE" --split-ids train --track-ids train \
  --seed "$SEED" --device cuda \
  --training-epochs "$EPOCHS" --minimum-training-epochs "$EPOCHS" \
  --training-rollout-steps "$TRAIN_MAX_STEPS" \
  --validation-rollout-steps "$VALIDATION_MAX_STEPS" \
  --physical-batch-size "$BATCH_SIZE" \
  --effective-batch-size "$BATCH_SIZE" --batch-size "$BATCH_SIZE" \
  --samples-per-instance "$TRAIN_TRAJECTORIES" --baseline-eval-size 0 \
  --validation-dataset-path "$VAL_INDEX" \
  --validation-family-root "$FAMILY_ROOT" \
  --validation-limit "$VALIDATION_LIMIT" \
  --validation-decode-type sampling \
  --validation-candidates "$VALIDATION_CANDIDATES" \
  --validation-seed $((910000000 + SEED)) \
  --validation-every-epochs 100 \
  --post-minimum-validation-every-epochs 100 \
  --validation-checkpoints $(((EPOCHS + 99) / 100)) \
  --protocol-id rrnco_ev_single_seed_500_update_screening_v2_calibrated_batch \
  --objective-config "$REPO_ROOT/EVRPTW_Benchmark/Reinforcement_Learning/configs/rivian_energy_vehicle_cost_v2.json" \
  --reward-contract "$REPO_ROOT/EVRPTW_Benchmark/Reinforcement_Learning/configs/drl_reward_contract_energy_vehicle_v3.json" \
  --optimizer adamw --learning-rate "$LEARNING_RATE" \
  --weight-decay "$WEIGHT_DECAY" \
  --output-dir "$OUTPUT_DIR"
