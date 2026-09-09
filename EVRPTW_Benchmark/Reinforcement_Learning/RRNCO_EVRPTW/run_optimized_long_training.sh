#!/usr/bin/env bash
# Explicit RRNCO-EV v2 experiment; the original long launcher remains unchanged.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/dataset_root.sh"
DATASET_ROOT="$(resolve_evrptw_dataset_root "$REPO_ROOT")"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCALE="${SCALE:-Cus50}"
SEED="${SEED:-1234}"
GPU="${GPU:-1}"
GRAPH_MODE="${GRAPH_MODE:-full}"
EPOCHS="${EPOCHS:-10000}"
MINIMUM_EPOCHS="${MINIMUM_EPOCHS:-5000}"
VALIDATION_EVERY_EPOCHS="${VALIDATION_EVERY_EPOCHS:-100}"
POST_MINIMUM_VALIDATION_EVERY_EPOCHS="${POST_MINIMUM_VALIDATION_EVERY_EPOCHS:-250}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
TRAIN_TRAJECTORIES="${TRAIN_TRAJECTORIES:-5}"
RELATION_CHUNK_SIZE="${RELATION_CHUNK_SIZE:-32}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-2}"
case "$SCALE" in
  Cus50)
    PLAN=compatibility_cus50; DEFAULT_BATCH_SIZE=128
    TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-65}"
    VALIDATION_MAX_STEPS="${VALIDATION_MAX_STEPS:-98}"
    ;;
  Cus100)
    # This is a configurable starting point; run a full train/validation memory
    # gate on the target server before scheduling a Cus100 long experiment.
    PLAN=core; DEFAULT_BATCH_SIZE=32
    TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-120}"
    VALIDATION_MAX_STEPS="${VALIDATION_MAX_STEPS:-180}"
    ;;
  *) echo "Supported scales: Cus50, Cus100" >&2; exit 2 ;;
esac
case "$GRAPH_MODE" in
  full|distance|distance_time|node_only) ;;
  *) echo "Unsupported GRAPH_MODE: $GRAPH_MODE" >&2; exit 2 ;;
esac
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-$BATCH_SIZE}"
for variable in EPOCHS MINIMUM_EPOCHS VALIDATION_EVERY_EPOCHS POST_MINIMUM_VALIDATION_EVERY_EPOCHS EARLY_STOP_PATIENCE TRAIN_TRAJECTORIES RELATION_CHUNK_SIZE BATCH_SIZE EFFECTIVE_BATCH_SIZE TRAIN_MAX_STEPS VALIDATION_MAX_STEPS; do
  [[ "${!variable}" =~ ^[1-9][0-9]*$ ]] || { echo "$variable must be a positive integer" >&2; exit 2; }
done
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a nonnegative integer" >&2; exit 2; }
(( EPOCHS >= MINIMUM_EPOCHS && TRAIN_TRAJECTORIES >= 2 && EFFECTIVE_BATCH_SIZE >= BATCH_SIZE && EFFECTIVE_BATCH_SIZE % BATCH_SIZE == 0 )) || {
  echo "Require epochs >= minimum, trajectories >= 2, and effective batch divisible by physical batch" >&2; exit 2;
}
EARLY_STOP_START="$MINIMUM_EPOCHS"
if (( EPOCHS == MINIMUM_EPOCHS )); then
  # A fixed short gate has no post-minimum window in which to stop early.
  EARLY_STOP_START=0
  EARLY_STOP_PATIENCE=0
fi
# Include the minimum and maximum endpoints even for non-divisible intervals.
PRE_CHECKPOINTS=$(((MINIMUM_EPOCHS + VALIDATION_EVERY_EPOCHS - 1) / VALIDATION_EVERY_EPOCHS))
POST_CHECKPOINTS=$(((EPOCHS - MINIMUM_EPOCHS + POST_MINIMUM_VALIDATION_EVERY_EPOCHS - 1) / POST_MINIMUM_VALIDATION_EVERY_EPOCHS))
VALIDATION_CHECKPOINTS=$((PRE_CHECKPOINTS + POST_CHECKPOINTS))
CUSTOMER_EXPOSURE_BUDGET=$((EPOCHS * EFFECTIVE_BATCH_SIZE * ${SCALE#Cus}))
TRAIN_INDEX="$DATASET_ROOT/generation_plan/$PLAN/train/view_index.parquet"
VAL_INDEX="$DATASET_ROOT/generation_plan/$PLAN/val/view_index.parquet"
FAMILY_ROOT="$DATASET_ROOT/materialized/families"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/EVRPTW_Benchmark/results/RRNCO_EV_optimized_long_v2}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%dT%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/runs/$SCALE/$GRAPH_MODE/seed_${SEED}/$RUN_TAG}"
# Representation variants intentionally share the exact ordered stream.
TRAINING_STREAM="${TRAINING_STREAM:-$OUTPUT_ROOT/artifacts/streams/${EPOCHS}e_b${EFFECTIVE_BATCH_SIZE}/$SCALE/seed_${SEED}.parquet}"
command=(
  "$PYTHON_BIN" -m EVRPTW_Benchmark.Reinforcement_Learning.RRNCO_EVRPTW.train
  --dataset-path "$TRAIN_INDEX" --family-root "$FAMILY_ROOT"
  --scale "$SCALE" --split-ids train --track-ids train --seed "$SEED" --device cuda
  --training-epochs "$EPOCHS" --minimum-training-epochs "$MINIMUM_EPOCHS"
  --training-stream-path "$TRAINING_STREAM" --customer-exposure-budget "$CUSTOMER_EXPOSURE_BUDGET"
  --early-stop-start-epoch "$EARLY_STOP_START" --early-stop-patience-validations "$EARLY_STOP_PATIENCE"
  --training-rollout-steps "$TRAIN_MAX_STEPS" --validation-rollout-steps "$VALIDATION_MAX_STEPS"
  --physical-batch-size "$BATCH_SIZE" --effective-batch-size "$EFFECTIVE_BATCH_SIZE" --batch-size "$BATCH_SIZE"
  --samples-per-instance "$TRAIN_TRAJECTORIES" --baseline-eval-size 0
  --reinforce-baseline leave_one_out --graph-mode "$GRAPH_MODE"
  --aft-mode stable --distance-sampling nearest --relation-temperature 5
  --relation-chunk-size "$RELATION_CHUNK_SIZE" --checkpoint-bias
  --validation-dataset-path "$VAL_INDEX" --validation-family-root "$FAMILY_ROOT"
  --validation-limit 500 --validation-decode-type sampling --validation-candidates 100
  --validation-seed "$((910000000 + SEED))" --validation-every-epochs "$VALIDATION_EVERY_EPOCHS"
  --post-minimum-validation-every-epochs "$POST_MINIMUM_VALIDATION_EVERY_EPOCHS"
  --validation-checkpoints "$VALIDATION_CHECKPOINTS" --protocol-id rrnco_ev_optimized_long_v2
  --objective-config "$REPO_ROOT/EVRPTW_Benchmark/Reinforcement_Learning/configs/rivian_energy_vehicle_cost_v2.json"
  --reward-contract "$REPO_ROOT/EVRPTW_Benchmark/Reinforcement_Learning/configs/drl_reward_contract_energy_vehicle_v3.json"
  --optimizer adamw --learning-rate "${LEARNING_RATE:-0.0001}" --weight-decay "${WEIGHT_DECAY:-0.01}"
  --output-dir "$OUTPUT_DIR"
)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'Frozen stream SHA256 will be appended after preparation/verification.\n'
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
for required in "$TRAIN_INDEX" "$VAL_INDEX" "$FAMILY_ROOT"; do
  [[ -e "$required" ]] || { echo "Missing dataset path: $required" >&2; exit 1; }
done
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Output must be fresh: $OUTPUT_DIR" >&2; exit 1; }
cd "$REPO_ROOT"
mkdir -p "$(dirname "$TRAINING_STREAM")"
# Concurrent full/node_only launches must not rewrite a shared stream.
exec 8>"$TRAINING_STREAM.lock"
flock -x 8
if [[ ! -e "$TRAINING_STREAM" ]]; then
  [[ ! -e "$TRAINING_STREAM.manifest.json" ]] || { echo "Orphan stream manifest: $TRAINING_STREAM.manifest.json" >&2; exit 1; }
  "$PYTHON_BIN" -m EVRPTW_Benchmark.Reinforcement_Learning.scripts.build_training_stream \
    --index "$TRAIN_INDEX" --scale "$SCALE" --seed "$SEED" \
    --customer-exposures "$CUSTOMER_EXPOSURE_BUDGET" --output "$TRAINING_STREAM"
fi
STREAM_SHA256="$("$PYTHON_BIN" - "$TRAINING_STREAM" "$TRAIN_INDEX" "$SCALE" "$SEED" "$((EPOCHS * EFFECTIVE_BATCH_SIZE))" <<'PY'
import sys
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_stream import file_sha256, load_training_stream_contract
stream, source, scale, seed, count = sys.argv[1:]
contract = load_training_stream_contract(stream)
if (contract['scale'], contract['seed'], contract['sample_count']) != (scale, int(seed), int(count)):
    raise ValueError('Frozen stream scale, seed, or exposure count does not match this run')
if contract['source_index_sha256'] != file_sha256(source):
    raise ValueError('Frozen stream source-index digest does not match this dataset')
print(contract['sha256'])
PY
)"
flock -u 8
exec 8>&-
command+=(--training-stream-contract-sha256 "$STREAM_SHA256")
mkdir -p "$OUTPUT_DIR"
git rev-parse HEAD > "$OUTPUT_DIR/source_commit.txt"
git diff --binary HEAD > "$OUTPUT_DIR/source_changes.patch"
sha256sum "$OUTPUT_DIR/source_changes.patch" > "$OUTPUT_DIR/source_changes.sha256"
cp "$SCRIPT_DIR/run_optimized_long_training.sh" "$OUTPUT_DIR/launcher_snapshot.sh"
printf '#!/usr/bin/env bash\nset -euo pipefail\n' > "$OUTPUT_DIR/command.sh"
printf 'cd %q\n' "$REPO_ROOT" >> "$OUTPUT_DIR/command.sh"
printf 'export OMP_NUM_THREADS=%q MKL_NUM_THREADS=%q OPENBLAS_NUM_THREADS=%q NUMBA_NUM_THREADS=%q\n' \
  "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "$NUMBA_NUM_THREADS" >> "$OUTPUT_DIR/command.sh"
printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU" >> "$OUTPUT_DIR/command.sh"
printf '%q ' "${command[@]}" >> "$OUTPUT_DIR/command.sh"
printf '\n' >> "$OUTPUT_DIR/command.sh"
if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then
  echo "Prepared frozen stream and executable command: $OUTPUT_DIR/command.sh"
  exit 0
fi
set +e
CUDA_VISIBLE_DEVICES="$GPU" "${command[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rc" > "$OUTPUT_DIR/exit_code.txt"
exit "$rc"
