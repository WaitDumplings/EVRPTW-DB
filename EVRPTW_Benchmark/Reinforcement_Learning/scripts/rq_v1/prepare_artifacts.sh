#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${EVRPTW_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
source "$SCRIPT_DIR/dataset_root.sh"
DATASET_ROOT="$(resolve_evrptw_dataset_root "$REPO_ROOT")"
OUTPUT_ROOT="${EVRPTW_OUTPUT_ROOT:-$REPO_ROOT/EVRPTW_Benchmark/results/DRL_rq_v1}"
ARTIFACT_ROOT="$OUTPUT_ROOT/artifacts"
RUNTIME_BUDGET_ID="drl_rq_runtime_budget_v4_cus1000_b2_val100"
STREAM_ROOT="$ARTIFACT_ROOT/streams/$RUNTIME_BUDGET_ID"
TRAIN_CORE="$DATASET_ROOT/generation_plan/core/train/view_index.parquet"
TRAIN_CUS50="$DATASET_ROOT/generation_plan/compatibility_cus50/train/view_index.parquet"
FAMILY_ROOT="$DATASET_ROOT/materialized/families"
FAMILY_METRICS="$DATASET_ROOT/reports/phase1/family_metrics.parquet"
SUPPORT_ROOT="$ARTIFACT_ROOT/supports"
E_MANIFEST="$ARTIFACT_ROOT/euclidean/euclidean_calibration_manifest.json"
MARKER="$ARTIFACT_ROOT/preparation_${RUNTIME_BUDGET_ID}.json"

cd "$REPO_ROOT"
for required in "$TRAIN_CORE" "$TRAIN_CUS50" "$FAMILY_METRICS"; do
  [[ -f "$required" ]] || { echo "Missing required release-data file: $required" >&2; exit 2; }
done
[[ -d "$FAMILY_ROOT" ]] || { echo "Missing materialized family root: $FAMILY_ROOT" >&2; exit 2; }

if [[ -f "$MARKER" ]] && python - "$MARKER" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
paths = [pathlib.Path(item) for item in d.get("required_artifacts", [])]
valid_budget = d.get("runtime_budget_id") == "drl_rq_runtime_budget_v4_cus1000_b2_val100"
raise SystemExit(0 if valid_budget and paths and all(path.is_file() for path in paths) else 1)
PY
then
  echo "RQ artifacts already complete: $MARKER"
  exit 0
fi

mkdir -p "$SUPPORT_ROOT" "$ARTIFACT_ROOT/euclidean"
python -m EVRPTW_Benchmark.Reinforcement_Learning.scripts.build_support_sets \
  --train-index "$TRAIN_CORE" \
  --family-metrics "$FAMILY_METRICS" \
  --fraction 0.10 \
  --seed 73129 \
  --output-dir "$SUPPORT_ROOT"

python -m EVRPTW_Benchmark.Reinforcement_Learning.scripts.calibrate_euclidean_representation \
  --train-index "$TRAIN_CORE" \
  --family-root "$FAMILY_ROOT" \
  --scale Cus100 \
  --seed 24680 \
  --views-per-day-type 100 \
  --pairs-per-view 100 \
  --output "$E_MANIFEST"

declare -A INDEX=(
  [Cus50]="$TRAIN_CUS50"
  [Cus100]="$TRAIN_CORE"
  [Cus500]="$TRAIN_CORE"
  [Cus1000]="$TRAIN_CORE"
)
declare -A FORMAL_EXPOSURE=(
  [Cus50]=10000000
  [Cus100]=5000000
  [Cus500]=2000000
  [Cus1000]=2000000
)
declare -A PILOT_EXPOSURE=(
  [Cus50]=20000
  [Cus100]=10000
  [Cus500]=4000
  [Cus1000]=4000
)
SEEDS=(1234 2345 3456)
REQUIRED=("$E_MANIFEST" "$SUPPORT_ROOT/support_selection_manifest.json")

for scale in Cus50 Cus100 Cus500 Cus1000; do
  for seed in "${SEEDS[@]}"; do
    formal="$STREAM_ROOT/formal/Full-support/$scale/seed_${seed}.parquet"
    mkdir -p "$(dirname "$formal")"
    python -m EVRPTW_Benchmark.Reinforcement_Learning.scripts.build_training_stream \
      --index "${INDEX[$scale]}" --scale "$scale" --seed "$seed" \
      --customer-exposures "${FORMAL_EXPOSURE[$scale]}" \
      --output "$formal"
    REQUIRED+=("$formal" "$formal.manifest.json")
  done
  pilot="$STREAM_ROOT/pilot/Full-support/$scale/seed_1234.parquet"
  mkdir -p "$(dirname "$pilot")"
  python -m EVRPTW_Benchmark.Reinforcement_Learning.scripts.build_training_stream \
    --index "${INDEX[$scale]}" --scale "$scale" --seed 1234 \
    --customer-exposures "${PILOT_EXPOSURE[$scale]}" \
    --output "$pilot"
  REQUIRED+=("$pilot" "$pilot.manifest.json")
done

for support in Random-10%-support Coverage-10%-support; do
  for seed in "${SEEDS[@]}"; do
    stream="$STREAM_ROOT/formal/$support/Cus100/seed_${seed}.parquet"
    mkdir -p "$(dirname "$stream")"
    python -m EVRPTW_Benchmark.Reinforcement_Learning.scripts.build_training_stream \
      --index "$TRAIN_CORE" --scale Cus100 --seed "$seed" \
      --customer-exposures 5000000 \
      --allowed-family-ids "$SUPPORT_ROOT/$support.txt" \
      --output "$stream"
    REQUIRED+=("$stream" "$stream.manifest.json")
  done
done

python - "$MARKER" "${REQUIRED[@]}" <<'PY'
import json, os, pathlib, sys, tempfile
target = pathlib.Path(sys.argv[1])
payload = {
    "schema": "drl_rq_artifact_preparation_v2",
    "runtime_budget_id": "drl_rq_runtime_budget_v4_cus1000_b2_val100",
    "status": "passed",
    "required_artifacts": sys.argv[2:],
    "validation_or_test_used_for_selection": False,
    "file_hash_validation_performed": False,
}
target.parent.mkdir(parents=True, exist_ok=True)
fd, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
os.replace(name, target)
PY
echo "RQ artifact preparation PASS: $MARKER"
