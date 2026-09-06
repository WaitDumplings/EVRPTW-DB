#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${EVRPTW_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
source "$SCRIPT_DIR/dataset_root.sh"
DATASET_ROOT="$(resolve_evrptw_dataset_root "$REPO_ROOT")"
OUTPUT_ROOT="${EVRPTW_OUTPUT_ROOT:-$REPO_ROOT/EVRPTW_Benchmark/results/DRL_rq_v1}"
ARTIFACT_ROOT="$OUTPUT_ROOT/artifacts"
RUNTIME_BUDGET_ID="drl_rq_runtime_budget_v13_am5_min5000_max10000_tailval50"
STREAM_ROOT="$ARTIFACT_ROOT/streams/$RUNTIME_BUDGET_ID"
STREAM_REGISTRY="$REPO_ROOT/EVRPTW_Benchmark/Reinforcement_Learning/configs/drl_training_stream_registry_v1.json"
TRAIN_CORE="$DATASET_ROOT/generation_plan/core/train/view_index.parquet"
SEED_SELECTION="${DRL_SEEDS:-1234}"
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
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done
[[ "$SEED_SELECTION" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
  echo "invalid seed selection: $SEED_SELECTION" >&2
  exit 2
}
export DRL_SEEDS="$SEED_SELECTION"
SEED_TAG="${SEED_SELECTION//,/_}"
MARKER="$ARTIFACT_ROOT/preparation_${RUNTIME_BUDGET_ID}_seeds_${SEED_TAG}.json"

cd "$REPO_ROOT"
for required in "$TRAIN_CORE"; do
  [[ -f "$required" ]] || { echo "Missing required release-data file: $required" >&2; exit 2; }
done

if [[ -f "$MARKER" ]] && python - "$MARKER" "$DATASET_ROOT" "$STREAM_REGISTRY" <<'PY'
import hashlib, json, pathlib, sys
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_stream import load_training_stream_contract
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
paths = [pathlib.Path(item) for item in d.get("required_artifacts", [])]
valid_budget = d.get("runtime_budget_id") == "drl_rq_runtime_budget_v13_am5_min5000_max10000_tailval50"
valid_dataset = pathlib.Path(d.get("dataset_root", "")).resolve() == pathlib.Path(sys.argv[2]).resolve()
canonical = {
    key: value for key, value in d.items()
    if key not in {"marker_sha256", "dataset_root"}
}
valid_marker = d.get("marker_sha256") == hashlib.sha256(
    json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
).hexdigest()
contracts = d.get("training_stream_contracts", [])
registry = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
valid_contracts = bool(contracts)
for item in contracts:
    try:
        actual = load_training_stream_contract(pathlib.Path.cwd() / item["relative_path"])
        valid_contracts &= actual == item["snapshot"] and actual["sha256"] == item["sha256"]
    except Exception:
        valid_contracts = False
registered = registry.get("streams", {})
valid_contracts &= d.get("marker_sha256") == registry.get("artifact_preparation_marker_sha256")
valid_contracts &= all(
    any(entry.get("path") == item.get("relative_path") and entry.get("snapshot") == item.get("snapshot")
        for entry in registered.values())
    for item in contracts
)
raise SystemExit(
    0 if valid_budget and valid_dataset and valid_marker and valid_contracts
    and paths and all((pathlib.Path.cwd() / path).is_file() for path in paths) else 1
)
PY
then
  echo "RQ artifacts already complete: $MARKER"
  exit 0
fi

declare -A INDEX=(
  [Cus500]="$TRAIN_CORE"
  [Cus1000]="$TRAIN_CORE"
)
declare -A FORMAL_EXPOSURE=(
  [Cus500]=320000000
  [Cus1000]=20000000
)
IFS=',' read -r -a SEEDS <<< "$SEED_SELECTION"
for seed in "${SEEDS[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || {
    echo "Invalid DRL seed selection: $SEED_SELECTION" >&2; exit 2;
  }
done
REQUIRED=()

for scale in Cus500 Cus1000; do
  for seed in "${SEEDS[@]}"; do
    formal="$STREAM_ROOT/formal/Full-support/$scale/seed_${seed}.parquet"
    mkdir -p "$(dirname "$formal")"
    python -m EVRPTW_Benchmark.Reinforcement_Learning.scripts.build_training_stream \
      --index "${INDEX[$scale]}" --scale "$scale" --seed "$seed" \
      --customer-exposures "${FORMAL_EXPOSURE[$scale]}" \
      --output "$formal"
    REQUIRED+=("$formal" "$formal.manifest.json")
  done
done

python - "$MARKER" "$DATASET_ROOT" "$STREAM_REGISTRY" "${REQUIRED[@]}" <<'PY'
import hashlib, json, os, pathlib, sys, tempfile
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_stream import load_training_stream_contract
target = pathlib.Path(sys.argv[1])
repo = pathlib.Path.cwd().resolve()
registry = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
required = [str(pathlib.Path(path).resolve().relative_to(repo)) for path in sys.argv[4:]]
stream_paths = [path for path in required if path.endswith(".parquet")]
contracts = []
for path in stream_paths:
    snapshot = load_training_stream_contract(repo / path)
    contracts.append({"relative_path": path, "sha256": snapshot["sha256"], "snapshot": snapshot})
payload = {
    "schema": "drl_rq_artifact_preparation_v2",
    "runtime_budget_id": "drl_rq_runtime_budget_v13_am5_min5000_max10000_tailval50",
    "status": "passed",
    "dataset_root": sys.argv[2],
    "required_artifacts": required,
    "training_stream_contracts": contracts,
    "active_scales": ["Cus500", "Cus1000"],
    "validation_or_test_used_for_selection": False,
    "file_hash_validation_performed": True,
}
canonical = {key: value for key, value in payload.items() if key != "dataset_root"}
payload["marker_sha256"] = hashlib.sha256(
    json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
).hexdigest()
if payload["marker_sha256"] != registry.get("artifact_preparation_marker_sha256"):
    raise RuntimeError("prepared artifact marker does not match frozen registry")
registered = registry.get("streams", {})
for item in contracts:
    if not any(
        entry.get("path") == item["relative_path"]
        and entry.get("snapshot") == item["snapshot"]
        for entry in registered.values()
    ):
        raise RuntimeError("prepared training stream does not match frozen registry")
target.parent.mkdir(parents=True, exist_ok=True)
fd, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
os.replace(name, target)
PY
echo "RQ artifact preparation PASS: $MARKER"
