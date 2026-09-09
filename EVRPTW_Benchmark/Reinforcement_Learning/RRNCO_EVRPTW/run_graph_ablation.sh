#!/usr/bin/env bash
# Same-backbone road-information screening. Training views and budgets are shared.
set -euo pipefail
# Bash reads script input incrementally. Execute an immutable private copy so
# future edits/replacements of the launcher cannot corrupt a running wrapper.
if [[ "${RRNCO_SCREEN_SNAPSHOT_PATH:-}" != "${BASH_SOURCE[0]}" ]]; then
  source_script="$(realpath "${BASH_SOURCE[0]}")"
  snapshot_script="$(mktemp "${TMPDIR:-/tmp}/rrnco_graph_screen.XXXXXX.sh")"
  cp "$source_script" "$snapshot_script"
  RRNCO_SCREEN_SOURCE_SCRIPT="$source_script" RRNCO_SCREEN_SNAPSHOT_PATH="$snapshot_script" \
    exec bash "$snapshot_script" "$@"
fi
trap 'rm -f -- "$RRNCO_SCREEN_SNAPSHOT_PATH"' EXIT
SCRIPT_DIR="$(cd "$(dirname "$RRNCO_SCREEN_SOURCE_SCRIPT")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$REPO_ROOT/EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/dataset_root.sh"
DATASET_ROOT="$(resolve_evrptw_dataset_root "$REPO_ROOT")"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCALE="${SCALE:-Cus50}"
SEED="${SEED:-1234}"
GPU="${GPU:-1}"
GRAPH_MODE="${GRAPH_MODE:-full}"
AFT_MODE="${AFT_MODE:-stable}"
REINFORCE_BASELINE="${REINFORCE_BASELINE:-leave_one_out}"
DISTANCE_SAMPLING="${DISTANCE_SAMPLING:-nearest}"
RELATION_TEMPERATURE="${RELATION_TEMPERATURE:-5}"
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-24}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-$BATCH_SIZE}"
TRAIN_TRAJECTORIES="${TRAIN_TRAJECTORIES:-5}"
VALIDATION_EVERY="${VALIDATION_EVERY:-$EPOCHS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-2}"
case "$SCALE" in
  Cus50) PLAN=compatibility_cus50; TRAIN_STEPS=65; VAL_STEPS=98 ;;
  Cus100) PLAN=core; TRAIN_STEPS=120; VAL_STEPS=180 ;;
  *) echo "This small-scale screening supports Cus50 and Cus100" >&2; exit 2 ;;
esac
(( EPOCHS > 0 && BATCH_SIZE > 0 && EFFECTIVE_BATCH_SIZE >= BATCH_SIZE && EFFECTIVE_BATCH_SIZE % BATCH_SIZE == 0 && VALIDATION_EVERY > 0 )) || {
  echo "Invalid epochs, physical/effective batch, or validation interval" >&2; exit 2;
}
RUN_TAG="${RUN_TAG:-$(date +%Y%m%dT%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/EVRPTW_Benchmark/results/RRNCO_graph_v3/$RUN_TAG}"
OUTPUT_DIR="$OUTPUT_ROOT/${SCALE}_${GRAPH_MODE}_${AFT_MODE}_${DISTANCE_SAMPLING}_${REINFORCE_BASELINE}_seed${SEED}"
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Output must be fresh: $OUTPUT_DIR" >&2; exit 1; }
cd "$REPO_ROOT"
TRAIN_INDEX="$DATASET_ROOT/generation_plan/$PLAN/train/view_index.parquet"
STREAM="${TRAINING_STREAM:-$OUTPUT_ROOT/streams_v3/${SCALE}_seed${SEED}_${EPOCHS}e_b${EFFECTIVE_BATCH_SIZE}.parquet}"
mkdir -p "$(dirname "$STREAM")"
# Two paired jobs may prepare the same stream concurrently.
exec 8>"$STREAM.lock"
flock -x 8
if [[ ! -e "$STREAM" ]]; then
  [[ ! -e "$STREAM.manifest.json" && ! -e "$STREAM.metadata.json" ]] || {
    echo "Existing stream metadata without its stream; refusing overwrite: $STREAM" >&2; exit 1;
  }
  "$PYTHON_BIN" -m EVRPTW_Benchmark.Reinforcement_Learning.scripts.build_training_stream \
    --index "$TRAIN_INDEX" --scale "$SCALE" --seed "$SEED" \
    --customer-exposures "$((EPOCHS * EFFECTIVE_BATCH_SIZE * ${SCALE#Cus}))" --output "$STREAM"
fi
# v3 requires a hashed manifest. Older exploratory artifacts are left untouched;
# explicitly pointing TRAINING_STREAM at one must fail, never silently migrate.
STREAM_SHA256="$("$PYTHON_BIN" - "$STREAM" "$TRAIN_INDEX" "$SCALE" "$SEED" "$((EPOCHS * EFFECTIVE_BATCH_SIZE))" <<'PY_STREAM'
import sys
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_stream import file_sha256, load_training_stream_contract
stream, source, scale, seed, count = sys.argv[1:]
contract = load_training_stream_contract(stream)
if (contract['scale'], contract['seed'], contract['sample_count']) != (scale, int(seed), int(count)):
    raise ValueError('Frozen stream scale, seed, or exposure count does not match this run')
if contract['source_index_sha256'] != file_sha256(source):
    raise ValueError('Frozen stream source-index digest does not match this dataset')
print(contract['sha256'])
PY_STREAM
)"
flock -u 8
exec 8>&-
mkdir -p "$OUTPUT_DIR"
# Preserve exact source identity even for a diagnostic before the formal commit.
git rev-parse HEAD > "$OUTPUT_DIR/source_commit.txt"
git diff --binary HEAD > "$OUTPUT_DIR/source_changes.patch"
sha256sum "$OUTPUT_DIR/source_changes.patch" > "$OUTPUT_DIR/source_changes.sha256"
command=(
  "$PYTHON_BIN" -m EVRPTW_Benchmark.Reinforcement_Learning.RRNCO_EVRPTW.train
  --dataset-path "$DATASET_ROOT/generation_plan/$PLAN/train/view_index.parquet"
  --family-root "$DATASET_ROOT/materialized/families" --scale "$SCALE" --split-ids train --track-ids train
  --seed "$SEED" --device cuda --training-epochs "$EPOCHS" --minimum-training-epochs "$EPOCHS"
  --training-stream-path "$STREAM" --training-stream-contract-sha256 "$STREAM_SHA256" --customer-exposure-budget "$((EPOCHS * EFFECTIVE_BATCH_SIZE * ${SCALE#Cus}))"
  --physical-batch-size "$BATCH_SIZE" --effective-batch-size "$EFFECTIVE_BATCH_SIZE" --batch-size "$BATCH_SIZE"
  --samples-per-instance "$TRAIN_TRAJECTORIES" --baseline-eval-size 0
  --training-rollout-steps "$TRAIN_STEPS" --validation-rollout-steps "$VAL_STEPS"
  --validation-dataset-path "$DATASET_ROOT/generation_plan/$PLAN/val/view_index.parquet"
  --validation-family-root "$DATASET_ROOT/materialized/families"
  --validation-limit 500 --validation-decode-type sampling --validation-candidates 100
  --validation-seed "$((910000000 + SEED))"
  --validation-every-epochs "$VALIDATION_EVERY" --post-minimum-validation-every-epochs "$VALIDATION_EVERY"
  --validation-checkpoints "$(((EPOCHS + VALIDATION_EVERY - 1) / VALIDATION_EVERY))"
  --protocol-id rrnco_ev_same_backbone_graph_screen_v3_sha256
  --objective-config "$REPO_ROOT/EVRPTW_Benchmark/Reinforcement_Learning/configs/rivian_energy_vehicle_cost_v2.json"
  --reward-contract "$REPO_ROOT/EVRPTW_Benchmark/Reinforcement_Learning/configs/drl_reward_contract_energy_vehicle_v3.json"
  --optimizer adamw --learning-rate "${LEARNING_RATE:-0.0001}" --weight-decay 0.01
  --reinforce-baseline "$REINFORCE_BASELINE"
  --graph-mode "$GRAPH_MODE" --aft-mode "$AFT_MODE" --distance-sampling "$DISTANCE_SAMPLING"
  --relation-temperature "$RELATION_TEMPERATURE" --relation-chunk-size "${RELATION_CHUNK_SIZE:-32}" --checkpoint-bias
  --output-dir "$OUTPUT_DIR"
)
cp "${BASH_SOURCE[0]}" "$OUTPUT_DIR/launcher_snapshot.sh"
printf '#!/usr/bin/env bash\nset -euo pipefail\n' > "$OUTPUT_DIR/command.sh"
printf 'cd %q\n' "$REPO_ROOT" >> "$OUTPUT_DIR/command.sh"
printf 'export OMP_NUM_THREADS=%q MKL_NUM_THREADS=%q OPENBLAS_NUM_THREADS=%q NUMBA_NUM_THREADS=%q\n' \
  "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "$NUMBA_NUM_THREADS" >> "$OUTPUT_DIR/command.sh"
printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU" >> "$OUTPUT_DIR/command.sh"
printf '%q ' "${command[@]}" >> "$OUTPUT_DIR/command.sh"
printf '\n' >> "$OUTPUT_DIR/command.sh"
if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then
  echo "Prepared verified v3 stream and command: $OUTPUT_DIR/command.sh"
  exit 0
fi
set +e
CUDA_VISIBLE_DEVICES="$GPU" "${command[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rc" > "$OUTPUT_DIR/exit_code.txt"
exit "$rc"
