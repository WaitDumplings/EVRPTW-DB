#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "drl_experiment_protocol_v1.yaml"
DEFAULT_MANIFEST_DIR = ROOT / "manifests"
DEFAULT_SERVER_SCRIPT_DIR = ROOT / "scripts"
METHOD_ORDER = ("am_evrptw", "evrptw_rl", "drl_ts", "terran")
SEEN_SCALES = ("Cus100", "Cus500", "Cus1000")
SERVER_SPECS = {
    "2080ti_4_1": {
        "hardware": "2080ti",
        "canonical_slots": (0, 3, 6, 9),
        "gpu_count": 4,
        "gpu_name_pattern": "RTX 2080 Ti",
    },
    "2080ti_4_2": {
        "hardware": "2080ti",
        "canonical_slots": (1, 4, 7, 10),
        "gpu_count": 4,
        "gpu_name_pattern": "RTX 2080 Ti",
    },
    "2080ti_3_1": {
        "hardware": "2080ti",
        "canonical_slots": (2, 5, 8),
        "gpu_count": 3,
        "gpu_name_pattern": "RTX 2080 Ti",
    },
    "a6000_2_1": {
        "hardware": "a6000",
        "canonical_slots": (0, 1),
        "gpu_count": 2,
        "gpu_name_pattern": "RTX A6000",
    },
}


def load_protocol(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        protocol = yaml.safe_load(stream)
    if protocol.get("objective") != "verified_directed_road_total_distance":
        raise ValueError("canonical verified-distance objective is required")
    if protocol.get("training", {}).get("checkpoint_selection_uses_test_metrics"):
        raise ValueError("test metrics cannot select checkpoints")
    if protocol.get("scale_transfer", {}).get("train_cus2000"):
        raise ValueError("Cus2000 training is forbidden")
    rollout_steps = protocol.get("training", {}).get("rollout_steps_by_scale", {})
    if set(rollout_steps) != set(protocol.get("scales", {})):
        raise ValueError("training rollout-step budgets must cover every training scale")
    if any(int(value) <= 0 for value in rollout_steps.values()):
        raise ValueError("training rollout-step budgets must be positive")
    training = protocol.get("training", {})
    if training.get("budget_mode") != "fixed_logical_epochs":
        raise ValueError("formal training must use fixed_logical_epochs")
    logical_epochs = training.get("logical_epochs_by_scale", {})
    environments_per_epoch = training.get("environments_per_epoch_by_scale", {})
    scale_ids = set(protocol.get("scales", {}))
    if set(logical_epochs) != scale_ids or set(environments_per_epoch) != scale_ids:
        raise ValueError("logical epoch budgets must cover every training scale")
    if any(int(value) <= 0 for value in logical_epochs.values()):
        raise ValueError("logical epoch counts must be positive")
    if any(int(value) <= 0 for value in environments_per_epoch.values()):
        raise ValueError("environments per logical epoch must be positive")
    if int(training.get("validation_checkpoints", 0)) <= 0:
        raise ValueError("validation checkpoints must be positive")
    disabled = set(protocol.get("disabled_tracks", []))
    if {"E_to_R", "R_to_Inject_to_R"}.difference(disabled):
        raise ValueError("E->R and R->Inject->R must remain disabled")
    deployment = protocol.get("hardware", {}).get("deployment", {})
    for server_id, spec in SERVER_SPECS.items():
        frozen = deployment.get(server_id, {})
        if int(frozen.get("gpu_count", -1)) != int(spec["gpu_count"]):
            raise ValueError(f"hardware deployment GPU count drifted for {server_id}")
        if tuple(frozen.get("canonical_slots", ())) != tuple(spec["canonical_slots"]):
            raise ValueError(f"hardware deployment slots drifted for {server_id}")
    return protocol


def _base_job(
    protocol: dict[str, Any],
    *,
    job_id: str,
    kind: str,
    run_mode: str,
    hardware: str,
    slot: int,
    queue_position: int,
    method: str,
    scale: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema": "drl_job_manifest_v1",
        "protocol_id": protocol["protocol_id"],
        "job_id": job_id,
        "enabled": True,
        "representation": "R",
        "kind": kind,
        "run_mode": run_mode,
        "hardware": hardware,
        "global_slot": int(slot),
        "queue_position": int(queue_position),
        "method": method,
        "scale": scale,
        "seed": int(seed),
        "objective": protocol["objective"],
    }


def _append_round_robin(
    queues: dict[int, list[dict[str, Any]]],
    jobs: Iterable[dict[str, Any]],
    slot_count: int,
) -> None:
    for index, job in enumerate(jobs):
        slot = index % int(slot_count)
        job["global_slot"] = slot
        job["queue_position"] = len(queues[slot])
        queues[slot].append(job)


def _largest_divisor_at_most(value: int, limit: int) -> int:
    if value <= 0 or limit <= 0:
        raise ValueError("batch values must be positive")
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    raise AssertionError("one always divides a positive integer")


def _train_job(
    protocol: dict[str, Any],
    *,
    method: str,
    scale: str,
    seed: int,
    hardware: str,
    run_mode: str = "full",
    pilot_kind: str | None = None,
) -> dict[str, Any]:
    prefix = "pilot" if run_mode == "pilot" else "train"
    suffix = f"__{pilot_kind}" if pilot_kind else ""
    job_id = f"{prefix}__R__{method}__{scale}__seed{seed}{suffix}"
    method_cfg = protocol["methods"][method]
    customer_count = int(protocol["scales"][scale]["customer_count"])
    calibrated_physical_cap = int(method_cfg["physical_batch"][scale])
    if run_mode == "pilot":
        physical_batch = calibrated_physical_cap
        effective_batch = int(method_cfg["effective_batch"][scale])
        if pilot_kind == "short_optimization":
            training_epochs = int(protocol["pilot"]["short_optimization_batches"])
        elif hardware == "2080ti":
            training_epochs = int(protocol["pilot"]["rtx2080ti_memory_batches"])
        else:
            training_epochs = int(protocol["pilot"]["a6000_memory_batches"])
        environments_per_epoch = effective_batch
    else:
        training_epochs = int(protocol["training"]["logical_epochs_by_scale"][scale])
        environments_per_epoch = int(
            protocol["training"]["environments_per_epoch_by_scale"][scale]
        )
        physical_batch = _largest_divisor_at_most(
            environments_per_epoch, calibrated_physical_cap
        )
        effective_batch = environments_per_epoch
    target_environments = training_epochs * environments_per_epoch
    actual_environments = target_environments
    job = _base_job(
        protocol,
        job_id=job_id,
        kind="pilot" if run_mode == "pilot" else "train",
        run_mode=run_mode,
        hardware=hardware,
        slot=0,
        queue_position=0,
        method=method,
        scale=scale,
        seed=seed,
    )
    job.update(
        {
            "stage": pilot_kind or "optimization",
            "train_module": method_cfg["train_module"],
            "train_index": protocol["scales"][scale]["train_index"],
            "validation_index": protocol["scales"][scale]["validation_index"],
            "train_views_per_pass": protocol["scales"][scale]["train_views"],
            "budget_mode": "fixed_logical_epochs",
            "training_epochs": training_epochs,
            "logical_environments_per_epoch": environments_per_epoch,
            "calibrated_physical_batch_cap": calibrated_physical_cap,
            "target_environments": target_environments,
            "actual_environments": actual_environments,
            "target_customer_exposures": target_environments * customer_count,
            "actual_customer_exposures": actual_environments * customer_count,
            "training_rollout_steps": int(
                protocol["training"]["rollout_steps_by_scale"][scale]
            ),
            "validation_checkpoints": int(
                protocol["training"]["validation_checkpoints"]
            ),
            "validation_views": int(
                protocol["pilot"]["validation_limit"]
                if run_mode == "pilot"
                else protocol["training"]["validation_views"]
            ),
            "physical_batch_size": physical_batch,
            "effective_batch_size": effective_batch,
            "memory_gate_gib": (
                float(protocol["hardware"]["rtx_2080_ti"]["memory_gate_gib"])
                if hardware == "2080ti" and scale == "Cus500"
                else None
            ),
            "requires_pilot_gate": run_mode == "full",
        }
    )
    return job


def _eval_job(
    protocol: dict[str, Any],
    *,
    method: str,
    scale: str,
    seed: int,
    test_id: str,
    decode_budget: str,
    hardware: str,
    transfer: bool = False,
) -> dict[str, Any]:
    decode = protocol["decoding"][decode_budget]
    if transfer:
        cohort = protocol["scale_transfer"]["cohorts"][test_id]
        target_scale = cohort["scale"]
        job_id = (
            f"transfer__R__{method}__Cus1000_to_{target_scale}__seed{seed}"
            f"__{decode_budget}"
        )
        index = protocol["scale_transfer"]["index"]
        track_id = protocol["scale_transfer"]["track_id"]
        expected_views = cohort["expected_views"]
        scale_filter = target_scale
        kind = "transfer"
    else:
        spec = protocol["scales"][scale]["tests"][test_id]
        job_id = (
            f"eval__R__{method}__{scale}__seed{seed}__{test_id}"
            f"__{decode_budget}"
        )
        index = spec["index"]
        track_id = spec["track_id"]
        expected_views = spec["expected_views"]
        scale_filter = scale
        kind = "eval"
    job = _base_job(
        protocol,
        job_id=job_id,
        kind=kind,
        run_mode="evaluate",
        hardware=hardware,
        slot=0,
        queue_position=0,
        method=method,
        scale=scale_filter,
        seed=seed,
    )
    job.update(
        {
            "stage": "inference",
            "split": "test",
            "test_id": test_id,
            "track_id": track_id,
            "dataset_index": index,
            "expected_views": int(expected_views),
            "decode_budget": decode_budget,
            "decode_type": decode["decode_type"],
            "candidate_count": int(decode["candidates"]),
            "candidate_chunk_size": int(decode["candidate_chunk_size"]),
            "eval_module": protocol["methods"][method]["eval_module"],
            "checkpoint_job_id": f"train__R__{method}__Cus1000__seed{seed}"
            if transfer
            else f"train__R__{method}__{scale}__seed{seed}",
            "source_scale": "Cus1000" if transfer else scale,
        }
    )
    return job


def build_manifests(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seeds = tuple(int(seed) for seed in protocol["seeds"])
    q2080: dict[int, list[dict[str, Any]]] = defaultdict(list)
    qa6000: dict[int, list[dict[str, Any]]] = defaultdict(list)

    pilot_2080 = []
    for scale, stage in (("Cus100", "short_optimization"), ("Cus500", "memory")):
        for method in METHOD_ORDER:
            pilot_2080.append(
                _train_job(
                    protocol,
                    method=method,
                    scale=scale,
                    seed=int(protocol["pilot"]["seed"]),
                    hardware="2080ti",
                    run_mode="pilot",
                    pilot_kind=stage,
                )
            )
    _append_round_robin(q2080, pilot_2080, 11)

    pilot_a6000 = [
        _train_job(
            protocol,
            method=method,
            scale="Cus1000",
            seed=int(protocol["pilot"]["seed"]),
            hardware="a6000",
            run_mode="pilot",
            pilot_kind="memory",
        )
        for method in METHOD_ORDER
    ]
    for index, job in enumerate(pilot_a6000):
        slot = index % 2
        job["global_slot"] = slot
        job["queue_position"] = len(qa6000[slot])
        qa6000[slot].append(job)

    full_2080: list[dict[str, Any]] = []
    for scale, methods in (
        ("Cus100", METHOD_ORDER),
        ("Cus500", ("am_evrptw", "evrptw_rl")),
        ("Cus500", ("drl_ts", "terran")),
        ("Cus50", METHOD_ORDER),
    ):
        for seed in seeds:
            for method in methods:
                full_2080.append(
                    _train_job(
                        protocol,
                        method=method,
                        scale=scale,
                        seed=seed,
                        hardware="2080ti",
                    )
                )
    _append_round_robin(q2080, full_2080, 11)

    # Frozen A6000 seed-round-robin queues from the directive.
    a6000_methods = {
        0: ("drl_ts", "am_evrptw"),
        1: ("terran", "evrptw_rl"),
    }
    for seed in seeds:
        for slot, methods in a6000_methods.items():
            for method in methods:
                job = _train_job(
                    protocol,
                    method=method,
                    scale="Cus1000",
                    seed=seed,
                    hardware="a6000",
                )
                job["global_slot"] = slot
                job["queue_position"] = len(qa6000[slot])
                qa6000[slot].append(job)

    greedy_jobs: list[dict[str, Any]] = []
    best_jobs: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        for seed in seeds:
            for scale in ("Cus50", *SEEN_SCALES):
                for test_id in protocol["scales"][scale]["tests"]:
                    greedy_jobs.append(
                        _eval_job(
                            protocol,
                            method=method,
                            scale=scale,
                            seed=seed,
                            test_id=test_id,
                            decode_budget="greedy",
                            hardware="2080ti",
                        )
                    )
                    best_jobs.append(
                        _eval_job(
                            protocol,
                            method=method,
                            scale=scale,
                            seed=seed,
                            test_id=test_id,
                            decode_budget="best_of_50",
                            hardware="a6000",
                        )
                    )
    _append_round_robin(q2080, greedy_jobs, 11)
    _append_round_robin(qa6000, best_jobs, 2)

    transfer_jobs = []
    for seed in seeds:
        for method in METHOD_ORDER:
            for cohort in ("paired_Cus1000", "zero_shot_Cus2000"):
                for budget in ("greedy", "best_of_50"):
                    transfer_jobs.append(
                        _eval_job(
                            protocol,
                            method=method,
                            scale="Cus1000",
                            seed=seed,
                            test_id=cohort,
                            decode_budget=budget,
                            hardware="a6000",
                            transfer=True,
                        )
                    )
    _append_round_robin(qa6000, transfer_jobs, 2)

    jobs2080 = [job for slot in range(11) for job in q2080[slot]]
    jobsa6000 = [job for slot in range(2) for job in qa6000[slot]]
    validate_jobs(protocol, jobs2080, jobsa6000)
    return jobs2080, jobsa6000


def validate_jobs(
    protocol: dict[str, Any],
    jobs2080: list[dict[str, Any]],
    jobsa6000: list[dict[str, Any]],
) -> None:
    jobs = jobs2080 + jobsa6000
    ids = [job["job_id"] for job in jobs]
    duplicate_ids = [job_id for job_id, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"duplicate stable job IDs: {duplicate_ids[:5]}")
    if any(job["representation"] != "R" for job in jobs):
        raise ValueError("unapproved representation job was generated")
    if any(job["kind"] == "train" and job["scale"] == "Cus2000" for job in jobs):
        raise ValueError("Cus2000 training job was generated")
    if any(
        job["kind"] == "train" and job.get("split") == "test" for job in jobs
    ):
        raise ValueError("test split was used for training")
    full_training = [job for job in jobs if job["kind"] == "train"]
    if len(full_training) != 48:
        raise ValueError(f"expected 48 full training jobs, got {len(full_training)}")
    cus1000 = [job for job in full_training if job["scale"] == "Cus1000"]
    if len(cus1000) != 12 or any(job["hardware"] != "a6000" for job in cus1000):
        raise ValueError("A6000 manifest must contain all 12 Cus1000 train jobs")
    cus50_eval = [job for job in jobs if job["kind"] == "eval" and job["scale"] == "Cus50"]
    if {job["test_id"] for job in cus50_eval} != {"T1"}:
        raise ValueError("Cus50 may contain T1 evaluation only")
    transfer = [job for job in jobs if job["kind"] == "transfer"]
    if {job["test_id"] for job in transfer} != {
        "paired_Cus1000",
        "zero_shot_Cus2000",
    }:
        raise ValueError("scale transfer must contain paired control and Cus2000")


def build_server_bundles(
    jobs2080: list[dict[str, Any]],
    jobsa6000: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Project checkpoint-producing queues onto four physical servers.

    The two four-GPU and one three-GPU 2080 Ti servers receive interleaved
    canonical slots.  This keeps the frozen global queue unchanged while
    balancing the 36 non-Cus1000 training jobs at 13/13/10.
    """

    sources = [
        job
        for job in jobs2080 + jobsa6000
        if job["kind"] in {"pilot", "train"}
    ]
    bundles: dict[str, list[dict[str, Any]]] = {
        server_id: [] for server_id in SERVER_SPECS
    }
    canonical_locations: dict[tuple[str, int], tuple[str, int]] = {}
    for server_id, spec in SERVER_SPECS.items():
        for local, canonical in enumerate(spec["canonical_slots"]):
            key = (str(spec["hardware"]), int(canonical))
            if key in canonical_locations:
                raise ValueError(f"duplicate physical ownership for {key}")
            canonical_locations[key] = (server_id, local)
    for source in sources:
        canonical_slot = int(source["global_slot"])
        destination = canonical_locations[(str(source["hardware"]), canonical_slot)]
        server_id, local_slot = destination
        job = dict(source)
        job["canonical_global_slot"] = canonical_slot
        job["assigned_server"] = server_id
        job["global_slot"] = local_slot
        bundles[server_id].append(job)
    for server_id, selected in bundles.items():
        by_local: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for job in selected:
            by_local[int(job["global_slot"])].append(job)
        for local_slot, queue in by_local.items():
            for position, job in enumerate(
                sorted(queue, key=lambda row: int(row["queue_position"]))
            ):
                job["queue_position"] = position
        bundles[server_id] = sorted(
            selected,
            key=lambda row: (int(row["global_slot"]), int(row["queue_position"])),
        )
    canonical_ids = {job["job_id"] for job in sources}
    assigned_ids = [
        job["job_id"] for jobs in bundles.values() for job in jobs
    ]
    if len(assigned_ids) != len(set(assigned_ids)):
        raise ValueError("server bundles overlap")
    if set(assigned_ids) != canonical_ids:
        missing = sorted(canonical_ids.difference(assigned_ids))
        extra = sorted(set(assigned_ids).difference(canonical_ids))
        raise ValueError(f"server bundle coverage mismatch: missing={missing[:5]} extra={extra[:5]}")
    return bundles


def _assignment_summary(
    server_id: str,
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "drl_server_assignment_v1",
        "protocol_id": jobs[0]["protocol_id"],
        "server_id": server_id,
        "hardware": SERVER_SPECS[server_id]["hardware"],
        "gpu_count": SERVER_SPECS[server_id]["gpu_count"],
        "canonical_slots": list(SERVER_SPECS[server_id]["canonical_slots"]),
        "job_count": len(jobs),
        "counts_by_run_mode": dict(sorted(Counter(job["run_mode"] for job in jobs).items())),
        "counts_by_kind": dict(sorted(Counter(job["kind"] for job in jobs).items())),
        "full_training_jobs": sum(job["kind"] == "train" for job in jobs),
        "counts_by_local_gpu": {
            str(slot): dict(
                sorted(
                    Counter(
                        job["run_mode"]
                        for job in jobs
                        if int(job["global_slot"]) == slot
                    ).items()
                )
            )
            for slot in range(int(SERVER_SPECS[server_id]["gpu_count"]))
        },
        "purpose": "checkpoint_generation_only",
    }


def _env_script(server_id: str) -> str:
    spec = SERVER_SPECS[server_id]
    slots = ",".join(str(index) for index in range(int(spec["gpu_count"])))
    return f"""#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO_DEFAULT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export EVRPTW_REPO_ROOT="${{EVRPTW_REPO_ROOT:-$REPO_DEFAULT}}"
export EVRPTW_DATASET_ROOT="${{EVRPTW_DATASET_ROOT:-$EVRPTW_REPO_ROOT/EVRPTW_Dataset/Instances_v2/us_11city}}"
export EVRPTW_OUTPUT_ROOT="${{EVRPTW_OUTPUT_ROOT:-$EVRPTW_REPO_ROOT/EVRPTW_Benchmark/results/DRL_protocol_v1}}"
export DRL_MANIFEST="$SCRIPT_DIR/jobs.jsonl"
export DRL_SLOTS="{slots}"
export DRL_LOCAL_GPU_COUNT="{spec['gpu_count']}"
export DRL_GPU_NAME_PATTERN="{spec['gpu_name_pattern']}"
export DRL_SERVER_ID="{server_id}"
"""


def _run_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
exec bash "$SCRIPT_DIR/../drl_job_runner.sh" "$@"
"""


def _start_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
MODE="${1:?usage: start.sh pilot|full|resume [runner options]}"
case "$MODE" in pilot|full|resume) ;; *) echo "invalid mode: $MODE" >&2; exit 2 ;; esac
LOG_DIR="$EVRPTW_OUTPUT_ROOT/launcher_logs/$DRL_SERVER_ID"
mkdir -p "$LOG_DIR"
PID_FILE="$LOG_DIR/$MODE.pid"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "$DRL_SERVER_ID/$MODE is already running with pid $(cat "$PID_FILE")" >&2
  exit 3
fi
STAMP="$(date +%Y%m%dT%H%M%S)"
LOG_FILE="$LOG_DIR/${MODE}_${STAMP}.log"
nohup setsid bash "$SCRIPT_DIR/run.sh" "$@" >"$LOG_FILE" 2>&1 < /dev/null &
PID=$!
printf '%s\\n' "$PID" >"$PID_FILE"
printf '%s\\n' "$LOG_FILE" >"$LOG_DIR/current.log.path"
echo "started: server=$DRL_SERVER_ID mode=$MODE pid=$PID"
echo "log: $LOG_FILE"
"""


def _status_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
LOG_DIR="$EVRPTW_OUTPUT_ROOT/launcher_logs/$DRL_SERVER_ID"
for MODE in pilot full resume; do
  PID_FILE="$LOG_DIR/$MODE.pid"
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "$MODE: running pid=$(cat "$PID_FILE")"
  fi
done
bash "$SCRIPT_DIR/run.sh" status --skip-gpu-preflight "$@"
"""


def _logs_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
PATH_FILE="$EVRPTW_OUTPUT_ROOT/launcher_logs/$DRL_SERVER_ID/current.log.path"
[[ -f "$PATH_FILE" ]] || { echo "no launcher log has been created" >&2; exit 2; }
exec tail -n "${LINES:-100}" -F "$(cat "$PATH_FILE")"
"""


def _mode_script(mode: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
exec bash "$SCRIPT_DIR/start.sh" {mode} "$@"
"""


def _readme(server_id: str, summary: dict[str, Any]) -> str:
    return f"""# {server_id} DRL queue

Hardware: {summary['gpu_count']} × {SERVER_SPECS[server_id]['gpu_name_pattern']}

Assigned checkpoint jobs: {summary['job_count']} total;
pilot={summary['counts_by_run_mode'].get('pilot', 0)},
full={summary['counts_by_run_mode'].get('full', 0)}.

From the repository root, activate any Python environment containing the project's required dependencies and run:

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/{server_id}/pilot.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/{server_id}/status.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/{server_id}/logs.sh
```

`pilot.sh`, `full.sh`, and `resume.sh` detach with nohup.
`run.sh MODE` is the foreground/debug entrypoint. Environment paths may be
overridden before launch; committed defaults are repository-relative.

These four bundles intentionally contain no T1/T2/T3, best-of-50, or Cus2000
test jobs. Collect their `checkpoint_selected.pt`, validation, training result,
and provenance artifacts on the future central test server.
"""


def write_server_bundles(
    root: Path,
    bundles: dict[str, list[dict[str, Any]]],
    *,
    check: bool,
) -> None:
    for server_id, jobs in bundles.items():
        directory = root / server_id
        summary = _assignment_summary(server_id, jobs)
        files = {
            "jobs.jsonl": (_serialize(jobs), False),
            "assignment_summary.json": (
                json.dumps(summary, sort_keys=True, indent=2) + "\n",
                False,
            ),
            "env.sh": (_env_script(server_id), True),
            "run.sh": (_run_script(), True),
            "start.sh": (_start_script(), True),
            "status.sh": (_status_script(), True),
            "logs.sh": (_logs_script(), True),
            "pilot.sh": (_mode_script("pilot"), True),
            "full.sh": (_mode_script("full"), True),
            "resume.sh": (_mode_script("resume"), True),
            "README.md": (_readme(server_id, summary), False),
        }
        for name, (content, executable) in files.items():
            _write_or_check(directory / name, content, check, executable=executable)


def validate_dataset(protocol: dict[str, Any], dataset_root: Path) -> None:
    for scale, spec in protocol["scales"].items():
        train = pd.read_parquet(dataset_root / spec["train_index"])
        train_count = int((train["scale_id"].str.lower() == scale.lower()).sum())
        if train_count != int(spec["train_views"]):
            raise ValueError(f"{scale} train count {train_count} != {spec['train_views']}")
        validation = pd.read_parquet(dataset_root / spec["validation_index"])
        val_count = int((validation["scale_id"].str.lower() == scale.lower()).sum())
        if val_count != int(protocol["training"]["validation_views"]):
            raise ValueError(f"{scale} validation count {val_count} != 500")
        for test_id, test in spec["tests"].items():
            frame = pd.read_parquet(dataset_root / test["index"])
            selected = frame[
                (frame["scale_id"].str.lower() == scale.lower())
                & (frame["track_id"] == test["track_id"])
            ]
            if len(selected) != int(test["expected_views"]):
                raise ValueError(f"{scale}/{test_id} count {len(selected)} != 500")
    transfer = protocol["scale_transfer"]
    frame = pd.read_parquet(dataset_root / transfer["index"])
    for cohort in transfer["cohorts"].values():
        count = int((frame["scale_id"].str.lower() == cohort["scale"].lower()).sum())
        if count != int(cohort["expected_views"]):
            raise ValueError(f"transfer {cohort['scale']} count {count} != 500")


def _serialize(jobs: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs)


def _write_or_check(
    path: Path,
    content: str,
    check: bool,
    *,
    executable: bool = False,
) -> None:
    if check:
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"manifest is stale: {path}")
        if executable and not path.stat().st_mode & 0o111:
            raise SystemExit(f"generated launcher is not executable: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    temporary.replace(path)
    if executable:
        path.chmod(0o755)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen DRL job manifests.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--server-script-dir", type=Path, default=DEFAULT_SERVER_SCRIPT_DIR)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    if args.dataset_root:
        validate_dataset(protocol, args.dataset_root)
    jobs2080, jobsa6000 = build_manifests(protocol)
    bundles = build_server_bundles(jobs2080, jobsa6000)
    _write_or_check(
        args.output_dir / "drl_2080ti_jobs_v1.jsonl",
        _serialize(jobs2080),
        args.check,
    )
    write_server_bundles(args.server_script_dir, bundles, check=args.check)
    _write_or_check(
        args.output_dir / "drl_a6000_jobs_v1.jsonl",
        _serialize(jobsa6000),
        args.check,
    )
    print(
        json.dumps(
            {
                "protocol_id": protocol["protocol_id"],
                "jobs_2080ti": len(jobs2080),
                "jobs_a6000": len(jobsa6000),
                "full_training_jobs": sum(
                    job["kind"] == "train" for job in jobs2080 + jobsa6000
                ),
                "servers": {
                    server_id: {
                        "jobs": len(jobs),
                        "full_training_jobs": sum(
                            job["kind"] == "train" for job in jobs
                        ),
                    }
                    for server_id, jobs in bundles.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
