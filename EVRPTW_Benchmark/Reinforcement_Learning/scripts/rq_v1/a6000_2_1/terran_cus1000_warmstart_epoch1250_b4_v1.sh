#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_PATH="$SCRIPT_DIR/terran_cus1000_warmstart_epoch1250_b4_v1.json"
export SERVER_SCRIPT_DIR="$SCRIPT_DIR"
source "$SCRIPT_DIR/../server_env.sh"

PYTHON_BIN="${EVRPTW_PYTHON_BIN:-$(command -v python)}"
[[ -x "$PYTHON_BIN" ]] || {
  echo "Python executable is unavailable: $PYTHON_BIN" >&2
  exit 2
}

# Load the versioned profile as shell-safe assignments.  Keeping the experiment
# values in one JSON document makes the detached command and its audit copy use
# exactly the same source of truth.
PROFILE_EXPORTS="$($PYTHON_BIN - "$PROFILE_PATH" <<'PY'
import json
import shlex
import sys
from pathlib import Path

profile_path = Path(sys.argv[1])
profile = json.loads(profile_path.read_text(encoding="utf-8"))
if profile.get("schema") != "drl_terran_warmstart_launcher_profile_v1":
    raise SystemExit(f"invalid launcher profile schema: {profile.get('schema')!r}")

source = profile["source_checkpoint"]
training = profile["training"]
validation = profile["validation"]
stream = profile["training_stream"]
dataset = profile["dataset"]
files = profile["files"]

expected_future_epochs = int(training["maximum_global_epoch"]) - int(source["epoch"])
expected_stream_samples = expected_future_epochs * int(training["effective_batch_size"])
expected_exposures = expected_stream_samples * int(profile["num_customers"])
expected_validation_steps = (3 * int(training["rollout_steps"]) + 1) // 2
initial = int(validation["base_every_epochs"])
minimum = int(training["minimum_training_epoch"])
tail = int(validation["post_minimum_every_epochs"])
maximum = int(training["maximum_global_epoch"])
scheduled = set(range(initial, minimum + 1, initial))
scheduled.add(minimum)
scheduled.update(range(minimum + tail, maximum + 1, tail))
scheduled.add(maximum)
future_validations = [epoch for epoch in sorted(scheduled) if epoch > int(source["epoch"])]

checks = {
    "planned future epochs": (int(training["planned_future_epochs"]), expected_future_epochs),
    "stream samples": (int(stream["sample_count"]), expected_stream_samples),
    "customer exposures": (int(stream["customer_exposures"]), expected_exposures),
    "validation rollout": (int(validation["rollout_steps"]), expected_validation_steps),
    "future validation checkpoints": (int(validation["future_checkpoints"]), len(future_validations)),
}
for label, (actual, expected) in checks.items():
    if actual != expected:
        raise SystemExit(f"profile {label} mismatch: {actual} != {expected}")
if int(training["physical_batch_size"]) != int(training["effective_batch_size"]):
    raise SystemExit("this profile requires physical batch == effective batch")
if source.get("load_policy") != "model_weights_only_fresh_adamw":
    raise SystemExit("this profile requires a fresh AdamW optimizer warm start")

values = {
    "PROFILE_ID": profile["profile_id"],
    "LAUNCHER_ID": profile["launcher_id"],
    "PROTOCOL_ID": profile["protocol_id"],
    "OUTPUT_SUFFIX": profile["output_suffix"],
    "EXPECTED_BRANCH": profile["expected_branch"],
    "GPU_INDEX": profile["gpu_index"],
    "REPRESENTATION": profile["representation"],
    "CONDITION": profile["condition"],
    "SCALE": profile["scale"],
    "NUM_CUSTOMERS": profile["num_customers"],
    "SEED": profile["seed"],
    "SOURCE_RELATIVE": source["relative_path"],
    "SOURCE_SHA256": source["sha256"],
    "SOURCE_EPOCH": source["epoch"],
    "SOURCE_SEED": source["seed"],
    "MAXIMUM_EPOCH": training["maximum_global_epoch"],
    "PHYSICAL_BATCH": training["physical_batch_size"],
    "EFFECTIVE_BATCH": training["effective_batch_size"],
    "N_TRAJ": training["n_traj"],
    "TRAIN_ROLLOUT_STEPS": training["rollout_steps"],
    "PPO_STEP_CHUNK_SIZE": training["ppo_step_chunk_size"],
    "NUM_MINIBATCHES": training["num_minibatches"],
    "GRADIENT_ACCUMULATION_STEPS": training["gradient_accumulation_steps"],
    "LEARNING_RATE": training["learning_rate"],
    "WEIGHT_DECAY": training["weight_decay"],
    "MINIMUM_EPOCH": training["minimum_training_epoch"],
    "EARLY_STOP_START_EPOCH": training["early_stop_start_epoch"],
    "EARLY_STOP_PATIENCE": training["early_stop_patience_validations"],
    "TERMINAL_SUCCESS_BONUS": training["terminal_success_bonus"],
    "VAL_ROLLOUT_STEPS": validation["rollout_steps"],
    "VAL_LIMIT": validation["limit"],
    "VAL_DECODE_TYPE": validation["decode_type"],
    "VAL_CANDIDATES": validation["candidates"],
    "VAL_SEED": validation["seed"],
    "VAL_EVERY": validation["base_every_epochs"],
    "POST_MINIMUM_VAL_EVERY": validation["post_minimum_every_epochs"],
    "VAL_CHECKPOINTS": validation["future_checkpoints"],
    "FINAL_VAL_LIMIT": validation["final_validation_limit"],
    "STREAM_RELATIVE": stream["relative_path"],
    "STREAM_CONTRACT_SHA256": stream["contract_sha256"],
    "STREAM_SAMPLE_COUNT": stream["sample_count"],
    "CUSTOMER_EXPOSURES": stream["customer_exposures"],
    "EXPOSURE_CHECKPOINTS": ",".join(str(value) for value in stream["exposure_checkpoints"]),
    "GPU_HOUR_CHECKPOINTS": ",".join(str(value) for value in stream["gpu_hour_checkpoints"]),
    "TRAIN_INDEX_RELATIVE": dataset["train_index"],
    "TRAIN_INDEX_SHA256": dataset["train_index_sha256"],
    "VAL_INDEX_RELATIVE": dataset["validation_index"],
    "VAL_INDEX_SHA256": dataset["validation_index_sha256"],
    "FAMILY_ROOT_RELATIVE": dataset["family_root"],
    "TERRAN_CONFIG_RELATIVE": files["terran_config"],
    "TERRAN_CONFIG_SHA256": files["terran_config_sha256"],
    "OBJECTIVE_CONFIG_RELATIVE": files["objective_config"],
    "OBJECTIVE_CONFIG_SHA256": files["objective_config_sha256"],
    "REWARD_CONTRACT_RELATIVE": files["reward_contract"],
    "REWARD_CONTRACT_FILE_SHA256": files["reward_contract_file_sha256"],
    "REWARD_CONTRACT_SHA256": files["reward_contract_sha256"],
}
for name, value in values.items():
    print(f"{name}={shlex.quote(str(value))}")
PY
)"
eval "$PROFILE_EXPORTS"

SOURCE_CHECKPOINT="$EVRPTW_REPO_ROOT/$SOURCE_RELATIVE"
TRAINING_STREAM="$EVRPTW_REPO_ROOT/$STREAM_RELATIVE"
TRAIN_INDEX="$EVRPTW_DATASET_ROOT/$TRAIN_INDEX_RELATIVE"
VAL_INDEX="$EVRPTW_DATASET_ROOT/$VAL_INDEX_RELATIVE"
FAMILY_ROOT="$EVRPTW_DATASET_ROOT/$FAMILY_ROOT_RELATIVE"
TERRAN_CONFIG="$EVRPTW_REPO_ROOT/$TERRAN_CONFIG_RELATIVE"
OBJECTIVE_CONFIG="$EVRPTW_REPO_ROOT/$OBJECTIVE_CONFIG_RELATIVE"
REWARD_CONTRACT="$EVRPTW_REPO_ROOT/$REWARD_CONTRACT_RELATIVE"
BASE_LOG_DIR="$EVRPTW_OUTPUT_ROOT/launcher_logs/$DRL_SERVER_ID/launchers/$LAUNCHER_ID"
PID_FILE="$BASE_LOG_DIR/train.pid"
CURRENT_LOG_PATH="$BASE_LOG_DIR/current.log.path"
CURRENT_OUTPUT_PATH="$BASE_LOG_DIR/current.output.path"

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

require_file_sha256() {
  local path="${1:?path is required}"
  local expected="${2:?expected SHA256 is required}"
  [[ -f "$path" ]] || { echo "required file is missing: $path" >&2; exit 4; }
  local actual
  actual="$(sha256_file "$path")"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA256 mismatch for $path: $actual != $expected" >&2
    exit 4
  }
}

preflight() {
  [[ -d "$EVRPTW_REPO_ROOT/.git" ]] || {
    echo "repository is missing: $EVRPTW_REPO_ROOT" >&2
    exit 4
  }
  local branch dirty
  branch="$(git -C "$EVRPTW_REPO_ROOT" branch --show-current)"
  [[ "$branch" == "$EXPECTED_BRANCH" ]] || {
    echo "wrong branch: $branch; expected $EXPECTED_BRANCH" >&2
    exit 4
  }
  dirty="$(git -C "$EVRPTW_REPO_ROOT" status --porcelain --untracked-files=normal)"
  [[ -z "$dirty" ]] || {
    echo "working tree must be clean before launching $PROFILE_ID" >&2
    echo "$dirty" >&2
    exit 4
  }
  git -C "$EVRPTW_REPO_ROOT" ls-files --error-unmatch \
    "${PROFILE_PATH#"$EVRPTW_REPO_ROOT/"}" >/dev/null

  require_file_sha256 "$SOURCE_CHECKPOINT" "$SOURCE_SHA256"
  require_file_sha256 "$TRAIN_INDEX" "$TRAIN_INDEX_SHA256"
  require_file_sha256 "$VAL_INDEX" "$VAL_INDEX_SHA256"
  require_file_sha256 "$TERRAN_CONFIG" "$TERRAN_CONFIG_SHA256"
  require_file_sha256 "$OBJECTIVE_CONFIG" "$OBJECTIVE_CONFIG_SHA256"
  require_file_sha256 "$REWARD_CONTRACT" "$REWARD_CONTRACT_FILE_SHA256"
  [[ -d "$FAMILY_ROOT" ]] || {
    echo "Stage-2 family root is missing: $FAMILY_ROOT" >&2
    exit 4
  }

  PYTHONPATH="$EVRPTW_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" - "$PROFILE_PATH" "$SOURCE_CHECKPOINT" "$TRAINING_STREAM" "$REWARD_CONTRACT" <<'PY'
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

from EVRPTW_Benchmark.Reinforcement_Learning.common.reward_contract import RewardContract
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_stream import load_training_stream_contract

profile = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
source_path = Path(sys.argv[2]).resolve(strict=True)
stream_path = Path(sys.argv[3]).resolve(strict=True)
reward_path = Path(sys.argv[4]).resolve(strict=True)
source = profile["source_checkpoint"]
stream = profile["training_stream"]

payload = torch.load(source_path, map_location="cpu", weights_only=False)
if not isinstance(payload, Mapping):
    raise SystemExit("warm-start checkpoint payload is not a mapping")
if int(payload.get("epoch", -1)) != int(source["epoch"]):
    raise SystemExit(
        f"warm-start epoch mismatch: {payload.get('epoch')} != {source['epoch']}"
    )
if int(payload.get("seed", -1)) != int(source["seed"]):
    raise SystemExit(
        f"warm-start seed mismatch: {payload.get('seed')} != {source['seed']}"
    )
if not isinstance(payload.get("model_state_dict"), Mapping):
    raise SystemExit("warm-start checkpoint has no model_state_dict")

contract = load_training_stream_contract(stream_path)
expected = str(stream["contract_sha256"])
if contract["sha256"] != expected:
    raise SystemExit(f"training stream contract mismatch: {contract['sha256']} != {expected}")
if int(contract["sample_count"]) != int(stream["sample_count"]):
    raise SystemExit("training stream sample count mismatch")
if contract["scale"] != profile["scale"] or int(contract["seed"]) != int(profile["seed"]):
    raise SystemExit("training stream scale/seed mismatch")

reward = RewardContract.load(reward_path)
if reward.digest != profile["files"]["reward_contract_sha256"]:
    raise SystemExit("reward contract semantic SHA256 mismatch")
print(
    "preflight artifacts verified: "
    f"source_epoch={source['epoch']} stream_samples={contract['sample_count']} "
    f"stream_contract={contract['sha256']}"
)
PY
}

show_status() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(tr -d '[:space:]' < "$PID_FILE")"
    if [[ "$pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "$LAUNCHER_ID: running pid=$pid gpu=$GPU_INDEX"
    else
      echo "$LAUNCHER_ID: stale pid file (${pid:-empty})"
    fi
  else
    echo "$LAUNCHER_ID: not running"
  fi
  [[ -f "$CURRENT_LOG_PATH" ]] && echo "log: $(cat "$CURRENT_LOG_PATH")"
  [[ -f "$CURRENT_OUTPUT_PATH" ]] && echo "output: $(cat "$CURRENT_OUTPUT_PATH")"
}

MODE="${1:-start}"
case "$MODE" in
  --check|check)
    preflight
    echo "profile: $PROFILE_ID"
    echo "GPU: physical $GPU_INDEX (GPU 0 is left available for the default Cus500 launcher)"
    echo "batch: physical=$PHYSICAL_BATCH effective=$EFFECTIVE_BATCH n_traj=$N_TRAJ"
    echo "rollout: train=$TRAIN_ROLLOUT_STEPS validation=$VAL_ROLLOUT_STEPS chunk=$PPO_STEP_CHUNK_SIZE"
    echo "epochs: source=$SOURCE_EPOCH maximum=$MAXIMUM_EPOCH future=$((MAXIMUM_EPOCH - SOURCE_EPOCH))"
    exit 0
    ;;
  --status|status)
    show_status
    exit 0
    ;;
  start)
    ;;
  *)
    echo "usage: $0 [start|--check|--status]" >&2
    exit 2
    ;;
esac

mkdir -p "$BASE_LOG_DIR"
exec 9>"$BASE_LOG_DIR/launcher.lock"
flock -x 9
if [[ -f "$PID_FILE" ]]; then
  ACTIVE_PID="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$ACTIVE_PID" =~ ^[1-9][0-9]*$ ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
    echo "$LAUNCHER_ID is already running with pid $ACTIVE_PID" >&2
    exit 3
  fi
  rm -f -- "$PID_FILE"
fi

preflight

GPU_NAME="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=name --format=csv,noheader | head -n 1)"
[[ "$GPU_NAME" =~ RTX[[:space:]](A6000|6000[[:space:]]Ada[[:space:]]Generation) ]] || {
  echo "GPU $GPU_INDEX is not the expected A6000-class device: $GPU_NAME" >&2
  exit 4
}
GPU_PIDS="$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
[[ -z "$GPU_PIDS" ]] || {
  echo "GPU $GPU_INDEX already has compute processes; refusing to overlap: $GPU_PIDS" >&2
  exit 3
}

COMMIT="$(git -C "$EVRPTW_REPO_ROOT" rev-parse HEAD)"
OUTPUT_DIR="$EVRPTW_OUTPUT_ROOT/$REPRESENTATION/$CONDITION/terran/$SCALE/seed_$SEED/${COMMIT}__${OUTPUT_SUFFIX}"
[[ ! -e "$OUTPUT_DIR" ]] || {
  echo "dedicated output already exists; refusing to overwrite: $OUTPUT_DIR" >&2
  exit 3
}
mkdir -p "$OUTPUT_DIR"
cp "$PROFILE_PATH" "$OUTPUT_DIR/launcher_profile.json"

COMMAND=(
  "$PYTHON_BIN" -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.train
  --config "$TERRAN_CONFIG"
  --seed "$SEED"
  --device cuda
  --stage2-dataset-path "$TRAIN_INDEX"
  --stage2-family-root "$FAMILY_ROOT"
  --stage2-scale "$SCALE"
  --num-customers "$NUM_CUSTOMERS"
  --num-envs-per-gpu "$PHYSICAL_BATCH"
  --n-traj "$N_TRAJ"
  --learning-rate "$LEARNING_RATE"
  --warm-start-checkpoint "$SOURCE_CHECKPOINT"
  --warm-start-epoch-mode continue_global
  --training-epochs "$MAXIMUM_EPOCH"
  --minimum-training-epochs "$MINIMUM_EPOCH"
  --post-minimum-validation-every-epochs "$POST_MINIMUM_VAL_EVERY"
  --training-rollout-steps "$TRAIN_ROLLOUT_STEPS"
  --validation-rollout-steps "$VAL_ROLLOUT_STEPS"
  --physical-batch-size "$PHYSICAL_BATCH"
  --effective-batch-size "$EFFECTIVE_BATCH"
  --validation-dataset-path "$VAL_INDEX"
  --validation-family-root "$FAMILY_ROOT"
  --validation-limit "$VAL_LIMIT"
  --validation-decode-type "$VAL_DECODE_TYPE"
  --validation-candidates "$VAL_CANDIDATES"
  --validation-seed "$VAL_SEED"
  --early-stop-patience-validations "$EARLY_STOP_PATIENCE"
  --early-stop-start-epoch "$EARLY_STOP_START_EPOCH"
  --final-validation-limit "$FINAL_VAL_LIMIT"
  --validation-every-epochs "$VAL_EVERY"
  --validation-checkpoints "$VAL_CHECKPOINTS"
  --protocol-id "$PROTOCOL_ID"
  --output-dir "$OUTPUT_DIR"
  --objective-config "$OBJECTIVE_CONFIG"
  --reward-contract "$REWARD_CONTRACT"
  --optimizer adamw
  --weight-decay "$WEIGHT_DECAY"
  --num-minibatches "$NUM_MINIBATCHES"
  --ppo-step-chunk-size "$PPO_STEP_CHUNK_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --terminal-success-bonus "$TERMINAL_SUCCESS_BONUS"
  --training-stream-path "$TRAINING_STREAM"
  --training-stream-contract-sha256 "$STREAM_CONTRACT_SHA256"
  --customer-exposure-budget "$CUSTOMER_EXPOSURES"
  --exposure-checkpoints "$EXPOSURE_CHECKPOINTS"
  --gpu-hour-checkpoints "$GPU_HOUR_CHECKPOINTS"
  --training-representation "$REPRESENTATION"
)

COMMAND_PATH="$OUTPUT_DIR/launch_command.sh"
{
  printf '#!/usr/bin/env bash\n'
  printf 'CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q ' "$GPU_INDEX" "$EVRPTW_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
} > "$COMMAND_PATH"
chmod 0444 "$COMMAND_PATH"

STAMP="$(date +%Y%m%dT%H%M%S)"
LOG_FILE="$BASE_LOG_DIR/train_${STAMP}.log"
nohup setsid env \
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
  PYTHONPATH="$EVRPTW_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "${COMMAND[@]}" >"$LOG_FILE" 2>&1 < /dev/null 9>&- &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
printf '%s\n' "$LOG_FILE" > "$CURRENT_LOG_PATH"
printf '%s\n' "$OUTPUT_DIR" > "$CURRENT_OUTPUT_PATH"
flock -u 9

echo "started with nohup: launcher=$LAUNCHER_ID pid=$PID physical_gpu=$GPU_INDEX"
echo "log: $LOG_FILE"
echo "output: $OUTPUT_DIR"
