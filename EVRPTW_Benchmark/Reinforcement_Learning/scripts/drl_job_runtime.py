#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import yaml

from EVRPTW_Benchmark.Reinforcement_Learning.common.method_auxiliary import (
    MethodAuxiliaryProfile,
    load_method_auxiliary_profile,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.reward_contract import (
    RewardContract,
    load_reward_contract,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_stream import (
    file_sha256,
    load_training_stream_contract,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_protocol import (
    resolved_training_signature_from_args,
    validation_epochs,
    validation_rollout_steps,
)


ROOT = Path(__file__).resolve().parents[1]
TERRAN_CONFIG = ROOT / "TERRAN" / "configs" / "stage2_cus100_terran.yaml"
RUNTIME_CONFIG = ROOT / "configs" / "drl_rq_runtime_candidates_v2.yaml"
FROZEN_PROTOCOL_CONFIG = ROOT / "configs" / "drl_rq_protocol_frozen_v1.yaml"
AUTHORIZED_LAUNCH_POLICY = "reward_contract_v2_formal_user_authorized"
METHODS = {"am_evrptw", "evrptw_rl", "drl_ts", "terran"}
LAUNCHER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
REQUIRED_ENV = (
    "EVRPTW_REPO_ROOT",
    "EVRPTW_DATASET_ROOT",
    "EVRPTW_OUTPUT_ROOT",
)
STOP = threading.Event()
CHILDREN: dict[int, subprocess.Popen[str]] = {}
CHILD_LOCK = threading.Lock()
OVERFLOW_LOCK = threading.Lock()


def validate_launcher_id(value: str) -> str:
    launcher_id = str(value).strip()
    if not LAUNCHER_ID_PATTERN.fullmatch(launcher_id):
        raise ValueError(
            "--launcher-id must contain 1-64 letters, digits, '.', '_' or '-' "
            "and start with a letter or digit"
        )
    return launcher_id


def parse_slot_gpu_map(
    raw: str | None,
    *,
    slots: set[int],
    local_gpu_count: int,
) -> dict[int, int]:
    """Resolve an explicit logical-slot to physical-GPU mapping.

    The legacy behavior remains the default.  A replacement launcher can
    select only slot 1 while still targeting CUDA device 1 via ``1:1`` rather
    than silently remapping the sole selected slot to device 0.
    """

    if not slots:
        raise ValueError("--slots must select at least one logical slot")
    if local_gpu_count <= 0:
        raise ValueError("--local-gpu-count must be positive")
    if raw is None or not raw.strip():
        mapping = {
            slot: local_gpu
            for local_gpu, slot in enumerate(sorted(slots))
        }
    else:
        mapping: dict[int, int] = {}
        for item in raw.split(","):
            fields = item.strip().split(":")
            if len(fields) != 2:
                raise ValueError(
                    "--slot-gpu-map entries must use SLOT:LOCAL_GPU"
                )
            try:
                slot, local_gpu = (int(value) for value in fields)
            except ValueError as error:
                raise ValueError(
                    "--slot-gpu-map entries must contain integers"
                ) from error
            if slot in mapping:
                raise ValueError(f"duplicate logical slot in --slot-gpu-map: {slot}")
            mapping[slot] = local_gpu
    if set(mapping) != slots:
        raise ValueError(
            "--slot-gpu-map must map exactly the selected slots: "
            f"mapped={sorted(mapping)}, selected={sorted(slots)}"
        )
    invalid_gpus = sorted(
        local_gpu
        for local_gpu in mapping.values()
        if not 0 <= local_gpu < local_gpu_count
    )
    if invalid_gpus:
        raise ValueError(
            "--slot-gpu-map references GPU indexes outside "
            f"[0, {local_gpu_count - 1}]: {invalid_gpus}"
        )
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("--slot-gpu-map cannot assign concurrent slots to one GPU")
    return mapping


def validate_job_routing(
    jobs: list[dict[str, Any]],
    *,
    launcher_id: str,
    slot_gpu_map: Mapping[int, int],
) -> None:
    """Fail closed when a dedicated manifest is launched in another namespace/GPU."""

    for job in jobs:
        slot = int(job["global_slot"])
        required_launcher_id = job.get("required_launcher_id")
        if (
            required_launcher_id is not None
            and str(required_launcher_id) != launcher_id
        ):
            raise ValueError(
                f"job {job['job_id']} requires launcher "
                f"{required_launcher_id!r}, got {launcher_id!r}"
            )
        required_local_gpu = job.get("required_local_gpu")
        if (
            required_local_gpu is not None
            and int(required_local_gpu) != slot_gpu_map[slot]
        ):
            raise ValueError(
                f"job {job['job_id']} requires local GPU "
                f"{required_local_gpu}, got {slot_gpu_map[slot]}"
            )


def process_rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        pass
    return 0


def process_gpu_memory_bytes(pid: int) -> int:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return 0
    total_mib = 0
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) == pid:
            try:
                total_mib += int(fields[1])
            except ValueError:
                continue
    return total_mib * 1024 * 1024


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def record_a6000_overflow(job: dict[str, Any], context: dict[str, Any]) -> Path:
    overflow = dict(job)
    overflow.update(
        {
            "hardware": "a6000",
            "global_slot": int(job["seed"]) % 2,
            "queue_position": -1,
            "overflow_reason": "OOM_UNCHANGED_CONFIG",
            "source_hardware": job.get("hardware"),
        }
    )
    path = context["output"] / "a6000_overflow_jobs_v1.jsonl"
    with OVERFLOW_LOCK:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(overflow, sort_keys=True) + "\n")
            stream.flush()
    return path


def load_jobs(
    path: Path,
    slots: set[int],
    mode: str,
    *,
    seeds: set[int] | None = None,
    scales: set[str] | None = None,
    methods: set[str] | None = None,
) -> list[dict[str, Any]]:
    jobs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    jobs = [job for job in jobs if int(job["global_slot"]) in slots and job.get("enabled", True)]
    if scales is not None:
        jobs = [job for job in jobs if str(job.get("scale")) in scales]
    if seeds is not None:
        jobs = [job for job in jobs if int(job.get("seed", -1)) in seeds]
    if methods is not None:
        jobs = [job for job in jobs if str(job.get("method")) in methods]
    if mode in {"full", "resume"}:
        jobs = [job for job in jobs if job["run_mode"] == "full"]
    elif mode == "evaluate":
        jobs = [job for job in jobs if job["run_mode"] == "evaluate"]
    return sorted(jobs, key=lambda job: (int(job["global_slot"]), int(job["queue_position"])))


def git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def gpu_name_matches(name: str, accepted_patterns: str) -> bool:
    patterns = [
        pattern.strip().lower()
        for pattern in accepted_patterns.split("|")
        if pattern.strip()
    ]
    return bool(patterns) and any(pattern in name.lower() for pattern in patterns)


def validate_formal_launch_gates(
    repo: Path,
    jobs: list[dict[str, Any]],
    *,
    require_open: bool,
) -> None:
    """Validate three independent formal decisions and optionally require approval.

    A mutable gate JSON is not sufficient launch authority.  The checked-in
    runtime candidate and frozen protocol must carry the same decision, policy,
    and complete G1--G8 evidence.  This makes flipping a single boolean fail
    closed.
    """

    resolved_repo = repo.resolve()
    grouped: dict[Path, list[dict[str, Any]]] = {}
    for job in jobs:
        raw_path = str(job.get("formal_gate_file", "")).strip()
        if not raw_path:
            raise RuntimeError(
                f"formal job {job.get('job_id')} is missing formal_gate_file"
            )
        candidate = Path(raw_path)
        gate_path = (
            candidate.resolve()
            if candidate.is_absolute()
            else (resolved_repo / candidate).resolve()
        )
        if gate_path != resolved_repo and resolved_repo not in gate_path.parents:
            raise RuntimeError(
                f"formal gate must remain inside the repository: {gate_path}"
            )
        grouped.setdefault(gate_path, []).append(job)

    runtime_path = resolved_repo / RUNTIME_CONFIG.relative_to(ROOT.parents[1])
    protocol_path = resolved_repo / FROZEN_PROTOCOL_CONFIG.relative_to(ROOT.parents[1])
    try:
        runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
        protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError("cannot read frozen formal launch configuration") from error
    if not isinstance(runtime, Mapping) or not isinstance(protocol, Mapping):
        raise RuntimeError("invalid frozen formal launch configuration")

    def decision(
        document: Mapping[str, Any], label: str
    ) -> tuple[str, bool, str, dict[str, str], tuple[str, ...]]:
        protocol_id = str(document.get("protocol_id", ""))
        allowed = document.get("formal_launch_allowed")
        policy = document.get("launch_policy")
        gates_raw = document.get("formal_launch_gates")
        authorized_raw = document.get("authorized_job_ids")
        if not protocol_id or not isinstance(allowed, bool) or not isinstance(policy, str):
            raise RuntimeError(f"{label} has an incomplete formal launch decision")
        if (
            not isinstance(authorized_raw, list)
            or not authorized_raw
            or any(not isinstance(value, str) or not value.strip() for value in authorized_raw)
            or len(authorized_raw) != len(set(authorized_raw))
        ):
            raise RuntimeError(
                f"{label} must define a nonempty, unique authorized_job_ids list"
            )
        if not isinstance(gates_raw, Mapping) or set(gates_raw) != {
            f"G{index}" for index in range(1, 9)
        }:
            raise RuntimeError(f"{label} must define exactly G1-G8")
        gates: dict[str, str] = {}
        for key, raw in gates_raw.items():
            value = raw.get("status") if isinstance(raw, Mapping) else raw
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError(f"{label} has an invalid {key} status")
            gates[str(key)] = value.strip()
        return (
            protocol_id,
            allowed,
            policy,
            gates,
            tuple(sorted(value.strip() for value in authorized_raw)),
        )

    runtime_decision = decision(runtime, "runtime config")
    protocol_decision = decision(protocol, "frozen protocol")
    for field in (
        "training_stream_registry_path",
        "training_stream_registry_sha256",
    ):
        if not runtime.get(field) or runtime.get(field) != protocol.get(field):
            raise RuntimeError(
                f"runtime/frozen protocol {field} contracts disagree"
            )

    def gate_status_complete(status: str) -> bool:
        upper = status.upper()
        return upper in {
            "PILOT_WAIVED_BY_USER",
            "IMPLEMENTED_UNIT_TESTED",
            "IMPLEMENTED_RELEASE_TESTED",
            "PASSED",
            "COMPLETED",
            "NOT_APPLICABLE_G_ONLY",
        }

    for gate_path, gated_jobs in grouped.items():
        try:
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cannot read formal launch gate: {gate_path}") from error
        if gate.get("schema") != "drl_rq_formal_launch_gate_v1":
            raise RuntimeError(f"invalid formal launch gate schema: {gate_path}")
        protocols = {str(job.get("protocol_id", "")) for job in gated_jobs}
        if protocols != {str(gate.get("protocol_id", ""))}:
            raise RuntimeError(
                "formal launch gate protocol does not match manifest jobs: "
                f"gate={gate.get('protocol_id')}, jobs={sorted(protocols)}"
            )
        gate_decision = decision(gate, f"formal launch gate {gate_path}")
        decisions = (gate_decision, runtime_decision, protocol_decision)
        if len(set((item[0], item[1], item[2]) for item in decisions)) != 1:
            raise RuntimeError(
                "formal launch gate/runtime/protocol decisions disagree"
            )
        if any(item[3] != gate_decision[3] for item in decisions[1:]):
            raise RuntimeError(
                "formal launch gate/runtime/protocol G1-G8 evidence disagrees"
            )
        if any(item[4] != gate_decision[4] for item in decisions[1:]):
            raise RuntimeError(
                "formal launch gate/runtime/protocol authorized_job_ids disagree"
            )
        if protocols != {gate_decision[0]}:
            raise RuntimeError(
                "formal launch decision protocol does not match manifest jobs"
            )
        requested_job_ids = [str(job.get("job_id", "")) for job in gated_jobs]
        if (
            not requested_job_ids
            or any(not job_id for job_id in requested_job_ids)
            or len(requested_job_ids) != len(set(requested_job_ids))
        ):
            raise RuntimeError(
                "formal launch requires a nonempty selection of unique job IDs"
            )
        unauthorized = sorted(set(requested_job_ids).difference(gate_decision[4]))
        if unauthorized:
            raise RuntimeError(
                "formal job selection is outside authorized_job_ids: "
                f"{unauthorized}"
            )
        if require_open and not gate_decision[1]:
            raise RuntimeError(
                "formal launch gate is closed; complete short-training validation "
                "and obtain explicit user authorization before full/resume"
            )
        if require_open and gate_decision[2] != AUTHORIZED_LAUNCH_POLICY:
            raise RuntimeError(
                "formal launch policy does not record explicit user authorization"
            )
        incomplete = {
            key: value
            for key, value in gate_decision[3].items()
            if not gate_status_complete(value)
        }
        if require_open and incomplete:
            raise RuntimeError(f"formal launch gates are incomplete: {incomplete}")


def training_contract(job: dict[str, Any]) -> dict[str, Any]:
    """Version the scientific objective independently of unchanged data streams."""
    contract = {}
    if "objective_config" in job:
        contract["objective_config"] = job["objective_config"]
    formal_scientific = (
        job.get("protocol_id") == "drl_rq_protocol_frozen_v1"
        or "training_stream_contract_sha256" in job
        or "objective_config" in job
    )
    scientific_fields = {
        "soft_stage_end_epoch", "training_stream_contract_sha256",
        "training_stream_contract_snapshot", "training_stream_path",
        "protocol_id", "runtime_budget_id", "representation", "condition",
        "train_index", "training_representation", "scale", "seed",
        "artifact_preparation_marker_path", "artifact_preparation_marker_sha256",
        "training_stream_registry_path", "training_stream_registry_sha256",
        "training_epochs",
        "minimum_training_epochs", "training_rollout_steps",
        "validation_rollout_steps",
        "physical_batch_size", "effective_batch_size",
        "warm_start_source_commit", "warm_start_scope",
        "warm_start_checkpoint_name", "warm_start_missing_policy",
        "training_trajectory_count", "customer_exposure_budget",
        "target_environments", "validation_index", "validation_views",
        "validation_decode_type", "validation_candidate_count",
        "validation_seed", "validation_every_epochs",
        "post_minimum_validation_every_epochs", "validation_checkpoints",
        "early_stop_patience_validations", "early_stop_start_epoch",
        "final_validation_views", "num_minibatches", "ppo_step_chunk_size",
        "terran_terminal_success_bonus",
    }
    for field in (
        "reward_contract_config_path",
        "reward_contract_id",
        "reward_contract_sha256",
        "reward_objective_scale",
        "reward_failure_base",
        "reward_unserved_coefficient",
        "training_gamma",
        "method_auxiliary_profile_path",
        "method_auxiliary_profile_id",
        "method_auxiliary_sha256",
        "method_auxiliary_method",
        "method_auxiliary_applicability",
        "method_auxiliary_aggregation",
        "method_auxiliary_denominator",
        "method_auxiliary_step_clip",
        "method_auxiliary_component_clip",
        "method_auxiliary_weights",
        "soft_stage_end_epoch",
        "training_stream_contract_sha256",
        "training_stream_contract_snapshot",
        "training_stream_path",
        "protocol_id",
        "runtime_budget_id",
        "representation",
        "condition",
        "train_index",
        "artifact_preparation_marker_path",
        "artifact_preparation_marker_sha256",
        "training_stream_registry_path",
        "training_stream_registry_sha256",
        "training_representation",
        "scale",
        "seed",
        "training_epochs",
        "minimum_training_epochs",
        "training_rollout_steps",
        "validation_rollout_steps",
        "physical_batch_size",
        "effective_batch_size",
        "warm_start_source_commit",
        "warm_start_scope",
        "warm_start_checkpoint_name",
        "warm_start_missing_policy",
        "training_trajectory_count",
        "customer_exposure_budget",
        "target_environments",
        "validation_index",
        "validation_views",
        "validation_decode_type",
        "validation_candidate_count",
        "validation_seed",
        "validation_every_epochs",
        "post_minimum_validation_every_epochs",
        "validation_checkpoints",
        "early_stop_patience_validations",
        "early_stop_start_epoch",
        "final_validation_views",
        "num_minibatches",
        "ppo_step_chunk_size",
        "terran_terminal_success_bonus",
    ):
        if field in scientific_fields and not formal_scientific:
            continue
        if field in job:
            contract[field] = job[field]
    for field in ("optimizer_name", "optimizer_weight_decay"):
        if field in job:
            contract[field] = job[field]
    return contract


def validate_optimizer_contracts(jobs: list[dict[str, Any]]) -> None:
    cfg = yaml.safe_load(RUNTIME_CONFIG.read_text(encoding="utf-8"))
    optimizer = cfg["training_optimizer"]
    expected_name = str(optimizer["name"]).lower()
    expected_weight_decay = float(optimizer["weight_decay"])
    if (
        expected_name != "adamw"
        or not math.isfinite(expected_weight_decay)
        or expected_weight_decay < 0.0
    ):
        raise RuntimeError("formal training requires a valid AdamW contract")
    for job in jobs:
        if job.get("kind") != "train":
            continue
        if (
            str(job.get("optimizer_name", "")).lower() != expected_name
            or float(job.get("optimizer_weight_decay", -1.0))
            != expected_weight_decay
        ):
            raise RuntimeError(
                f"manifest/config optimizer contract mismatch for {job['job_id']}; "
                "regenerate the RQ manifests before launching"
            )
    terran_training = yaml.safe_load(
        TERRAN_CONFIG.read_text(encoding="utf-8")
    )["training"]
    if (
        str(terran_training.get("optimizer", "")).lower() != expected_name
        or float(terran_training.get("weight_decay", -1.0))
        != expected_weight_decay
    ):
        raise RuntimeError("TERRAN config does not match the formal optimizer contract")


def validate_terran_training_contracts(jobs: list[dict[str, Any]]) -> None:
    terran_jobs = [
        job for job in jobs if job.get("method") == "terran" and job["kind"] == "train"
    ]
    if not terran_jobs:
        return
    training = yaml.safe_load(TERRAN_CONFIG.read_text(encoding="utf-8"))["training"]
    runtime_config = yaml.safe_load(RUNTIME_CONFIG.read_text(encoding="utf-8"))
    frozen_protocol = yaml.safe_load(
        FROZEN_PROTOCOL_CONFIG.read_text(encoding="utf-8")
    )
    expected = {
        "reward_contract_id": training["reward_contract_id"],
        "training_gamma": float(training["gamma"]),
    }
    for job in terran_jobs:
        if any(job.get(key) != value for key, value in expected.items()):
            raise RuntimeError(
                f"TERRAN manifest/config reward contract mismatch for {job['job_id']}: "
                f"manifest={training_contract(job)} config={expected}; "
                "regenerate the RQ manifests before launching"
            )
        if job.get("protocol_id") == "drl_rq_protocol_frozen_v1":
            try:
                completion_bonus = float(job["terran_terminal_success_bonus"])
                runtime_bonus = float(
                    runtime_config["training_overrides_by_method_scale"]
                    ["terran"][job["scale"]]["terran_terminal_success_bonus"]
                )
                protocol_bonus = float(
                    frozen_protocol["training_overrides_by_method_scale"]
                    ["terran"][job["scale"]]["terran_terminal_success_bonus"]
                )
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    "formal TERRAN manifest/runtime/protocol must freeze "
                    "terran_terminal_success_bonus per scale"
                ) from error
            if (
                not all(
                    math.isfinite(value) and value >= 0.0
                    for value in (completion_bonus, runtime_bonus, protocol_bonus)
                )
                or completion_bonus != runtime_bonus
                or completion_bonus != protocol_bonus
            ):
                raise RuntimeError(
                    "formal TERRAN terminal success-bonus contract mismatch: "
                    f"manifest={completion_bonus}, runtime={runtime_bonus}, "
                    f"protocol={protocol_bonus}"
                )


def validate_objective_contracts(jobs: list[dict[str, Any]]) -> None:
    """Reject stale/formally distance-only manifests before any launch side effects."""
    scoped_jobs = [job for job in jobs if job["kind"] == "train"
                   or "objective_config" in job or "objective_config_path" in job]
    if not scoped_jobs:
        return
    cfg = yaml.safe_load(RUNTIME_CONFIG.read_text(encoding="utf-8"))
    relative_path = cfg["objective_config_path"]
    expected = json.loads(
        (ROOT.parents[1] / relative_path).read_text(encoding="utf-8")
    )["objective"]
    for job in scoped_jobs:
        if (job.get("objective_config_path") != relative_path
                or job.get("objective_config") != expected
                or job.get("candidate_selection") != cfg["evaluation"]["selection"]):
            raise RuntimeError(
                f"manifest/config objective contract mismatch for {job['job_id']}; "
                "regenerate the RQ manifests and start cost training in a fresh root"
            )


def validate_reward_contracts(jobs: list[dict[str, Any]]) -> None:
    """Fail closed if a manifest does not match the frozen on-disk contract."""

    training_jobs = [job for job in jobs if job.get("kind") == "train"]
    if not training_jobs:
        return
    cfg = yaml.safe_load(RUNTIME_CONFIG.read_text(encoding="utf-8"))
    relative_path = cfg["reward_contract_config_path"]
    contract = load_reward_contract(ROOT.parents[1] / relative_path)
    if contract.contract_id != cfg["reward_contract_revision"]:
        raise RuntimeError("runtime config and reward contract id disagree")
    for job in training_jobs:
        terms = contract.for_scale(job["scale"], job["objective_config"])
        expected = {
            "reward_contract_config_path": relative_path,
            "reward_contract_id": terms.contract_id,
            "reward_contract_sha256": terms.digest,
            "reward_objective_scale": terms.objective_scale,
            "reward_failure_base": terms.failure_base,
            "reward_unserved_coefficient": terms.unserved_coefficient,
        }
        if any(job.get(key) != value for key, value in expected.items()):
            raise RuntimeError(
                f"manifest/config reward contract mismatch for {job['job_id']}; "
                "regenerate manifests and start in a fresh output directory"
            )


def validate_method_auxiliary_contracts(jobs: list[dict[str, Any]]) -> None:
    """Keep method-only shaping profiles separate and fail closed on drift."""

    cfg = yaml.safe_load(RUNTIME_CONFIG.read_text(encoding="utf-8"))
    configured = cfg.get("method_auxiliary_profiles", {})
    for job in jobs:
        if job.get("kind") != "train":
            continue
        relative_path = configured.get(job.get("method"))
        manifest_path = job.get("method_auxiliary_profile_path")
        if relative_path is None:
            if manifest_path is not None:
                raise RuntimeError(
                    f"unexpected method auxiliary profile for {job['job_id']}"
                )
            continue
        profile = load_method_auxiliary_profile(
            ROOT.parents[1] / relative_path
        ).require_method(job["method"])
        expected = {
            "method_auxiliary_profile_path": relative_path,
            "method_auxiliary_profile_id": profile.profile_id,
            "method_auxiliary_sha256": profile.digest,
            "method_auxiliary_method": profile.method,
            "method_auxiliary_applicability": profile.applicability,
            "method_auxiliary_aggregation": profile.aggregation,
            "method_auxiliary_denominator": profile.denominator,
            "method_auxiliary_step_clip": profile.step_clip,
            "method_auxiliary_component_clip": profile.component_clip,
            "method_auxiliary_weights": dict(profile.weights),
        }
        if any(job.get(field) != value for field, value in expected.items()):
            raise RuntimeError(
                f"manifest/config method auxiliary mismatch for {job['job_id']}; "
                "regenerate manifests and start in a fresh output directory"
            )


def validate_training_stream_contracts(
    jobs: list[dict[str, Any]],
    repo: Path,
    dataset: Path,
) -> None:
    """Bind every formal job to one exact, training-only ordered ID stream."""

    grouped: dict[tuple[str, str, str, str, int], set[tuple[str, str]]] = {}
    verified_markers: dict[Path, Mapping[str, Any]] = {}
    verified_registries: dict[Path, Mapping[str, Any]] = {}
    for job in jobs:
        if job.get("kind") != "train":
            continue
        relative_path = job.get("training_stream_path")
        expected_sha = job.get("training_stream_contract_sha256")
        expected_snapshot = job.get("training_stream_contract_snapshot")
        if not relative_path or not expected_sha or not isinstance(expected_snapshot, Mapping):
            raise RuntimeError(
                f"formal job {job.get('job_id')} is missing its frozen training-stream contract"
            )
        actual = load_training_stream_contract(repo / str(relative_path))
        if actual != expected_snapshot or actual["sha256"] != expected_sha:
            raise RuntimeError(
                f"training-stream contract mismatch for {job.get('job_id')}"
            )
        registry_relative = job.get("training_stream_registry_path")
        registry_expected_sha = job.get("training_stream_registry_sha256")
        if not registry_relative or not registry_expected_sha:
            raise RuntimeError("formal job is missing its training-stream registry")
        registry_path = (repo / str(registry_relative)).resolve()
        if repo.resolve() not in registry_path.parents:
            raise RuntimeError("training-stream registry must remain inside the repository")
        if registry_path not in verified_registries:
            try:
                registry = json.loads(registry_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError("cannot read training-stream registry") from error
            if not isinstance(registry, Mapping):
                raise RuntimeError("training-stream registry must contain an object")
            registry_canonical = {
                key: value for key, value in registry.items() if key != "sha256"
            }
            registry_sha = hashlib.sha256(
                json.dumps(
                    registry_canonical,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            if (
                registry.get("schema") != "drl_training_stream_registry_v1"
                or registry.get("source_scope")
                != "training_split_and_track_only"
                or registry.get("sha256") != registry_sha
            ):
                raise RuntimeError("training-stream registry contract mismatch")
            verified_registries[registry_path] = registry
        registry = verified_registries[registry_path]
        if registry.get("sha256") != registry_expected_sha:
            raise RuntimeError("manifest training-stream registry SHA256 mismatch")
        registry_key = (
            f"{job.get('representation')}/{job.get('condition')}/"
            f"{job.get('method')}/"
            f"{job.get('scale')}/seed_{int(job.get('seed'))}"
        )
        if (registry.get("streams") or {}).get(registry_key) != {
            "path": str(relative_path),
            "snapshot": actual,
        }:
            raise RuntimeError("registry training-stream entry mismatch")
        marker_relative = job.get("artifact_preparation_marker_path")
        marker_expected_sha = job.get("artifact_preparation_marker_sha256")
        if not marker_relative or not marker_expected_sha:
            raise RuntimeError(
                f"formal job {job.get('job_id')} is missing its artifact marker contract"
            )
        marker_path = (repo / str(marker_relative)).resolve()
        if repo.resolve() not in marker_path.parents:
            raise RuntimeError("artifact preparation marker must remain inside the repository")
        if marker_path not in verified_markers:
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError("cannot read artifact preparation marker") from error
            if not isinstance(marker, Mapping):
                raise RuntimeError("artifact preparation marker must contain an object")
            canonical = {
                key: value
                for key, value in marker.items()
                if key not in {"marker_sha256", "dataset_root"}
            }
            marker_sha = hashlib.sha256(
                json.dumps(
                    canonical,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            if (
                marker.get("schema") != "drl_rq_artifact_preparation_v2"
                or marker.get("status") != "passed"
                or marker.get("file_hash_validation_performed") is not True
                or marker.get("marker_sha256") != marker_sha
                or Path(str(marker.get("dataset_root", ""))).resolve()
                != dataset.resolve()
            ):
                raise RuntimeError("artifact preparation marker contract mismatch")
            verified_markers[marker_path] = marker
        marker = verified_markers[marker_path]
        if marker.get("marker_sha256") != marker_expected_sha:
            raise RuntimeError("manifest artifact marker SHA256 mismatch")
        if marker_expected_sha != registry.get(
            "artifact_preparation_marker_sha256"
        ):
            raise RuntimeError("registry/manifest artifact marker SHA256 mismatch")
        marker_streams = [
            item
            for item in marker.get("training_stream_contracts", ())
            if item.get("relative_path") == str(relative_path)
        ]
        if marker_streams != [{
            "relative_path": str(relative_path),
            "sha256": actual["sha256"],
            "snapshot": actual,
        }]:
            raise RuntimeError("artifact marker training-stream contract mismatch")
        expected_count = int(job.get("target_environments", -1))
        if (
            actual["scale"] != str(job.get("scale"))
            or actual["seed"] != int(job.get("seed", -1))
            or actual["sample_count"] != expected_count
        ):
            raise RuntimeError(
                f"training-stream scope/budget mismatch for {job.get('job_id')}"
            )
        source_index = dataset / str(job.get("train_index", ""))
        if not source_index.is_file():
            raise FileNotFoundError(
                f"training source index is missing for {job.get('job_id')}: {source_index}"
            )
        if file_sha256(source_index) != actual["source_index_sha256"]:
            raise RuntimeError(
                f"training source index SHA256 mismatch for {job.get('job_id')}"
            )
        if (
            str(job.get("condition")) == "Full-support"
            and actual.get("allowed_family_ids_sha256") is not None
        ):
            raise RuntimeError("Full-support stream unexpectedly freezes a support subset")
        key = (
            str(job.get("representation")),
            str(job.get("condition")),
            str(job.get("method")),
            str(job.get("scale")),
            int(job.get("seed")),
        )
        grouped.setdefault(key, set()).add((str(relative_path), str(expected_sha)))
    disagreeing = {key: values for key, values in grouped.items() if len(values) != 1}
    if disagreeing:
        raise RuntimeError(
            f"one exact method-specific job maps to multiple streams: {disagreeing}"
        )


def validate_completed_training_stream_contract(
    job: dict[str, Any],
    context: dict[str, Any],
    training_result: Mapping[str, Any],
    checkpoint: Path,
) -> None:
    relative_path = job.get("training_stream_path")
    if relative_path is None:
        return
    actual = load_training_stream_contract(context["repo"] / str(relative_path))
    expected_snapshot = job.get("training_stream_contract_snapshot")
    expected_sha = job.get("training_stream_contract_sha256")
    if actual != expected_snapshot or actual["sha256"] != expected_sha:
        raise RuntimeError("completed run training-stream artifact/manifest mismatch")
    if (
        training_result.get("training_stream_contract_snapshot") != actual
        or training_result.get("training_stream_contract_sha256") != actual["sha256"]
    ):
        raise RuntimeError("completed training_result training-stream contract mismatch")

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if job.get("method") == "terran":
        config = payload.get("config")
        if not isinstance(config, Mapping):
            raise RuntimeError("selected TERRAN checkpoint is missing its config")
        protocol = config.get("protocol")
        if not isinstance(protocol, Mapping):
            raise RuntimeError("selected TERRAN checkpoint is missing protocol provenance")
        checkpoint_snapshot = protocol.get("training_stream_contract_snapshot")
        checkpoint_sha = protocol.get("training_stream_contract_sha256")
    else:
        checkpoint_snapshot = payload.get("training_stream_contract")
        saved_args = payload.get("args", {}) or {}
        if not isinstance(saved_args, Mapping):
            saved_args = vars(saved_args)
        if checkpoint_snapshot != saved_args.get("training_stream_contract_snapshot"):
            raise RuntimeError("selected checkpoint training-stream snapshots disagree")
        checkpoint_sha = saved_args.get("training_stream_contract_sha256")
    if checkpoint_snapshot != actual or checkpoint_sha != actual["sha256"]:
        raise RuntimeError("selected checkpoint training-stream contract mismatch")


def expected_resolved_training_signature(
    job: Mapping[str, Any], context: Mapping[str, Any]
) -> dict[str, Any]:
    validation_steps = int(
        job.get(
            "validation_rollout_steps",
            validation_rollout_steps(int(job["training_rollout_steps"])),
        )
    )
    method_specific = None
    if job.get("method") == "terran":
        base = yaml.safe_load(TERRAN_CONFIG.read_text(encoding="utf-8"))
        training = base.get("training", {}) or {}
        evaluation = base.get("evaluation", {}) or {}
        epochs = int(job["training_epochs"])
        physical = int(job["physical_batch_size"])
        effective = int(job["effective_batch_size"])
        rollout_steps = int(job["training_rollout_steps"])
        scheduled = list(
            validation_epochs(
                epochs,
                initial_interval=int(job["validation_every_epochs"]),
                minimum_epochs=int(job["minimum_training_epochs"]),
                post_minimum_interval=int(
                    job["post_minimum_validation_every_epochs"]
                ),
            )
        )
        raw_decode = str(job["validation_decode_type"])
        method_specific = {
            "method": "TERRAN",
            "task_reward": {
                "terminal_success_bonus": float(
                    job["terran_terminal_success_bonus"]
                ),
                "unit": "normalized_objective_cost",
            },
            "training": {
                "epochs": epochs,
                "num_envs_per_gpu": physical,
                "n_traj": int(job["training_trajectory_count"]),
                "rollout_steps": rollout_steps,
                "logical_microbatches_per_epoch": effective // physical,
                "ppo_update_epochs": int(training.get("ppo_update_epochs", 4)),
                "num_minibatches": max(
                    1, int(job.get("num_minibatches", training.get("num_minibatches", 1)))
                ),
                "gradient_accumulation_steps": max(
                    1, int(training.get("gradient_accumulation_steps", 1))
                ),
                "ppo_step_chunk_size": max(
                    0,
                    int(
                        job.get(
                            "ppo_step_chunk_size",
                            training.get("ppo_step_chunk_size", 0),
                        )
                        or 0
                    ),
                ),
                "clip_coef": float(training.get("clip_coef", 0.2)),
                "vf_coef": float(training.get("vf_coef", 0.5)),
                "ent_coef": float(training.get("ent_coef", 0.01)),
                "learning_rate": float(training.get("learning_rate", 1e-4)),
                "max_grad_norm": float(training.get("max_grad_norm", 1.0)),
                "gamma": float(job["training_gamma"]),
            },
            "evaluation": {
                "seed": int(job["validation_seed"]),
                "decode_mode": "sample" if raw_decode == "sampling" else "greedy",
                "n_traj": int(job["validation_candidate_count"]),
                "limit": int(job["validation_views"]),
                "max_steps": validation_steps,
                "batch_size": 1,
                "num_batches": (
                    int(evaluation["eval_num_batches"])
                    if evaluation.get("eval_num_batches") is not None
                    else None
                ),
                "interval": int(job["validation_every_epochs"]),
            },
            "protocol": {
                "protocol_id": str(job["protocol_id"]),
                "physical_batch_size": physical,
                "effective_batch_size": effective,
                "training_rollout_steps": rollout_steps,
                "validation_rollout_steps": validation_steps,
                "training_stream_contract_sha256": job[
                    "training_stream_contract_sha256"
                ],
                "minimum_training_epochs": int(job["minimum_training_epochs"]),
                "validation_every_epochs": int(job["validation_every_epochs"]),
                "post_minimum_validation_every_epochs": int(
                    job["post_minimum_validation_every_epochs"]
                ),
                "scheduled_validation_epochs": scheduled,
                "validation_checkpoints": int(job["validation_checkpoints"]),
                "early_stop_patience_validations": int(
                    job["early_stop_patience_validations"]
                ),
                "early_stop_start_epoch": int(job["early_stop_start_epoch"]),
            },
        }
    arguments = argparse.Namespace(
        protocol_id=job.get("protocol_id"),
        seed=job.get("seed"),
        scale=job.get("scale"),
        training_representation=job.get("training_representation", "G"),
        training_epochs=job.get("training_epochs"),
        minimum_training_epochs=job.get("minimum_training_epochs"),
        training_rollout_steps=job.get("training_rollout_steps"),
        validation_rollout_steps=validation_steps,
        physical_batch_size=job.get("physical_batch_size"),
        effective_batch_size=job.get("effective_batch_size"),
        samples_per_instance=job.get("training_trajectory_count"),
        customer_exposure_budget=job.get("customer_exposure_budget"),
        training_stream_contract_sha256=job.get(
            "training_stream_contract_sha256"
        ),
        training_stream_path=(
            Path(context["repo"]) / str(job["training_stream_path"])
            if job.get("training_stream_path") is not None
            else None
        ),
        validation_dataset_path=(
            Path(context["dataset"]) / str(job["validation_index"])
            if job.get("validation_index") is not None
            else None
        ),
        validation_limit=job.get("validation_views"),
        validation_decode_type=job.get("validation_decode_type"),
        validation_candidates=job.get("validation_candidate_count"),
        validation_seed=job.get("validation_seed"),
        validation_every_epochs=job.get("validation_every_epochs"),
        post_minimum_validation_every_epochs=job.get(
            "post_minimum_validation_every_epochs"
        ),
        validation_checkpoints=job.get("validation_checkpoints"),
        early_stop_patience_validations=job.get(
            "early_stop_patience_validations"
        ),
        early_stop_start_epoch=job.get("early_stop_start_epoch"),
        final_validation_limit=job.get("final_validation_views"),
        soft_stage_end_epoch=job.get("soft_stage_end_epoch"),
        optimizer=job.get("optimizer_name"),
        weight_decay=job.get("optimizer_weight_decay"),
        reward_contract_sha256=job.get("reward_contract_sha256"),
        method_auxiliary_sha256=job.get("method_auxiliary_sha256"),
        euclidean_manifest=(
            Path(context["repo"]) / str(job["euclidean_manifest"])
            if job.get("euclidean_manifest") is not None
            else None
        ),
        resolved_training_method_fields=method_specific,
    )
    return resolved_training_signature_from_args(arguments)


def validate_completed_training_signature(
    job: dict[str, Any],
    context: dict[str, Any],
    training_result: Mapping[str, Any],
    checkpoint: Path,
) -> None:
    if job.get("training_stream_contract_sha256") is None:
        return
    expected = expected_resolved_training_signature(job, context)
    if (
        training_result.get("resolved_training_signature") != expected
        or training_result.get("resolved_training_signature_sha256")
        != expected["sha256"]
    ):
        raise RuntimeError("completed training_result scientific signature mismatch")
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if job.get("method") == "terran":
        config = payload.get("config")
        protocol = config.get("protocol") if isinstance(config, Mapping) else None
        if not isinstance(protocol, Mapping):
            raise RuntimeError("selected TERRAN checkpoint lacks protocol signature")
        saved = protocol.get("resolved_training_signature")
        saved_sha = protocol.get("resolved_training_signature_sha256")
    else:
        saved = payload.get("resolved_training_signature")
        raw_args = payload.get("args", {}) or {}
        saved_args = raw_args if isinstance(raw_args, Mapping) else vars(raw_args)
        if saved != saved_args.get("resolved_training_signature"):
            raise RuntimeError("selected checkpoint scientific signatures disagree")
        saved_sha = saved_args.get("resolved_training_signature_sha256")
    if saved != expected or saved_sha != expected["sha256"]:
        raise RuntimeError("selected checkpoint scientific signature mismatch")


def validate_training_result_outcome(
    job: Mapping[str, Any], training_result: Mapping[str, Any]
) -> None:
    if job.get("protocol_id") != "drl_rq_protocol_frozen_v1":
        return
    status = training_result.get("status")
    requested = int(job["training_epochs"])
    minimum = int(job["minimum_training_epochs"])
    if training_result.get("protocol_id") != job.get("protocol_id"):
        raise RuntimeError("training_result protocol mismatch")
    if training_result.get("requested_training_epochs") != requested:
        raise RuntimeError("training_result requested epoch budget mismatch")
    try:
        completed = int(training_result["completed_training_epochs"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("training_result lacks completed epoch evidence") from error
    if status == "passed":
        if completed != requested or bool(training_result.get("early_stopped")):
            raise RuntimeError("passed training_result did not finish its hard cap")
    elif status == "early_stopped":
        if not minimum <= completed <= requested or not bool(
            training_result.get("early_stopped")
        ):
            raise RuntimeError("early-stopped training_result violates its budget")
    else:
        raise RuntimeError(f"invalid formal training outcome: {status!r}")


def validate_completed_training_reward_contract(
    job: dict[str, Any],
    context: dict[str, Any],
    training_result: Mapping[str, Any],
    checkpoint: Path,
) -> None:
    """Cross-check launch intent against the report and selected checkpoint.

    ``provenance.json`` and ``job_result.json`` originate in this launcher, so
    they cannot prove that the trainer consumed the requested contract.  The
    completed trainer report and its selected checkpoint are independent
    post-run evidence and must both agree with the frozen on-disk contract.
    """

    relative_path = job.get("reward_contract_config_path")
    if relative_path is None:
        return
    contract = load_reward_contract(context["repo"] / str(relative_path))
    terms = contract.for_scale(job["scale"], job["objective_config"])
    expected_result = {
        "objective_config": terms.objective_config.to_dict(),
        "reward_contract_id": terms.contract_id,
        "reward_contract_sha256": terms.digest,
        "reward_contract_scale": terms.scale_label,
        "reward_contract_snapshot": contract.to_dict(),
        "reward_objective_scale": terms.objective_scale,
        "reward_failure_base": terms.failure_base,
        "reward_unserved_coefficient": terms.unserved_coefficient,
    }
    if job.get("method") == "terran":
        completion_bonus = float(job["terran_terminal_success_bonus"])
        expected_result.update(
            terran_terminal_success_bonus=completion_bonus,
            terran_terminal_success_bonus_unit="normalized_objective_cost",
            terran_terminal_success_bonus_equivalent_usd=(
                completion_bonus * terms.objective_scale
            ),
        )
    mismatched = {
        field: (training_result.get(field), expected)
        for field, expected in expected_result.items()
        if training_result.get(field) != expected
    }
    if mismatched:
        raise RuntimeError(
            "completed training_result reward contract mismatch: "
            f"{mismatched}"
        )

    # Import lazily so status/preflight operations do not pay PyTorch startup
    # cost.  map_location keeps this verification independent of GPU state.
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if job.get("method") == "terran":
        config = payload.get("config")
        if not isinstance(config, Mapping):
            raise RuntimeError("selected TERRAN checkpoint is missing its config")
        checkpoint_objective = config.get("objective")
        checkpoint_snapshot = config.get("reward_contract")
        derived = {
            "reward_contract_id": (config.get("training") or {}).get(
                "reward_contract_id"
            ),
            "reward_contract_sha256": (config.get("normalization") or {}).get(
                "reward_contract_sha256"
            ),
            "reward_contract_scale": (config.get("normalization") or {}).get(
                "reward_contract_scale"
            ),
            "reward_objective_scale": (config.get("normalization") or {}).get(
                "reward_objective_scale"
            ),
            "reward_failure_base": (config.get("normalization") or {}).get(
                "failure_base"
            ),
            "reward_unserved_coefficient": (
                config.get("normalization") or {}
            ).get("unserved_coefficient"),
            "terran_terminal_success_bonus": (config.get("pbrs") or {}).get(
                "terminal_success_bonus"
            ),
            "terran_terminal_success_bonus_unit": (
                config.get("normalization") or {}
            ).get("terran_terminal_success_bonus_unit"),
            "terran_terminal_success_bonus_equivalent_usd": (
                config.get("normalization") or {}
            ).get("terran_terminal_success_bonus_equivalent_usd"),
        }
    else:
        checkpoint_objective = payload.get("objective_config")
        saved_args = payload.get("args")
        if not isinstance(saved_args, Mapping):
            saved_args = vars(saved_args) if saved_args is not None else {}
        checkpoint_snapshot = payload.get("reward_contract")
        if checkpoint_snapshot != saved_args.get("reward_contract_snapshot"):
            raise RuntimeError(
                "selected checkpoint reward contract snapshots disagree"
            )
        derived = {
            "reward_contract_id": saved_args.get("reward_contract_id"),
            "reward_contract_sha256": saved_args.get("reward_contract_sha256"),
            "reward_contract_scale": saved_args.get("reward_contract_scale"),
            "reward_objective_scale": saved_args.get("reward_objective_scale"),
            "reward_failure_base": saved_args.get("reward_failure_base"),
            "reward_unserved_coefficient": saved_args.get(
                "reward_unserved_coefficient"
            ),
        }
    if checkpoint_objective != terms.objective_config.to_dict():
        raise RuntimeError(
            "selected checkpoint objective config does not match the requested contract"
        )
    try:
        checkpoint_contract = RewardContract.from_payload(checkpoint_snapshot)
        checkpoint_terms = checkpoint_contract.for_scale(
            terms.scale_label, terms.objective_config
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "selected checkpoint contains an invalid reward contract"
        ) from error
    if checkpoint_contract.to_dict() != contract.to_dict():
        raise RuntimeError(
            "selected checkpoint does not contain the requested frozen contract"
        )
    expected_derived = {
        "reward_contract_id": checkpoint_terms.contract_id,
        "reward_contract_sha256": checkpoint_terms.digest,
        "reward_contract_scale": checkpoint_terms.scale_label,
        "reward_objective_scale": checkpoint_terms.objective_scale,
        "reward_failure_base": checkpoint_terms.failure_base,
        "reward_unserved_coefficient": checkpoint_terms.unserved_coefficient,
    }
    if job.get("method") == "terran":
        completion_bonus = float(job["terran_terminal_success_bonus"])
        expected_derived.update(
            terran_terminal_success_bonus=completion_bonus,
            terran_terminal_success_bonus_unit="normalized_objective_cost",
            terran_terminal_success_bonus_equivalent_usd=(
                completion_bonus * checkpoint_terms.objective_scale
            ),
        )
    if derived != expected_derived:
        raise RuntimeError(
            "selected checkpoint derived reward-contract fields are inconsistent: "
            f"checkpoint={derived}, expected={expected_derived}"
        )


def validate_completed_method_auxiliary_contract(
    job: dict[str, Any],
    context: dict[str, Any],
    training_result: Mapping[str, Any],
    checkpoint: Path,
) -> None:
    relative_path = job.get("method_auxiliary_profile_path")
    if relative_path is None:
        forbidden = (
            "method_auxiliary_profile_id",
            "method_auxiliary_sha256",
            "method_auxiliary_snapshot",
            "method_auxiliary_method",
            "method_auxiliary_applicability",
            "method_auxiliary_aggregation",
            "method_auxiliary_denominator",
            "method_auxiliary_step_clip",
            "method_auxiliary_component_clip",
            "method_auxiliary_weights",
        )
        polluted = {
            field: training_result.get(field)
            for field in forbidden
            if training_result.get(field) is not None
        }
        if polluted:
            raise RuntimeError(
                f"method without an auxiliary profile reported one: {polluted}"
            )
        if (
            job.get("protocol_id") != "drl_rq_protocol_frozen_v1"
            and job.get("reward_contract_config_path") is None
        ):
            return
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if job.get("method") == "terran":
            config = payload.get("config", {}) or {}
            checkpoint_snapshot = (
                config.get("method_auxiliary_profile")
                if isinstance(config, Mapping)
                else None
            )
            saved_args: Mapping[str, Any] = {}
        else:
            checkpoint_snapshot = payload.get("method_auxiliary_profile")
            raw_args = payload.get("args", {}) or {}
            saved_args = (
                raw_args if isinstance(raw_args, Mapping) else vars(raw_args)
            )
        checkpoint_polluted = {
            field: saved_args.get(field)
            for field in forbidden
            if saved_args.get(field) is not None
        }
        if checkpoint_snapshot is not None or checkpoint_polluted:
            raise RuntimeError(
                "method without an auxiliary profile contains checkpoint auxiliary metadata"
            )
        return
    profile = load_method_auxiliary_profile(
        context["repo"] / str(relative_path)
    ).require_method(job["method"])
    expected = {
        "method_auxiliary_profile_id": profile.profile_id,
        "method_auxiliary_sha256": profile.digest,
        "method_auxiliary_snapshot": profile.to_dict(),
        "method_auxiliary_method": profile.method,
        "method_auxiliary_applicability": profile.applicability,
        "method_auxiliary_aggregation": profile.aggregation,
        "method_auxiliary_denominator": profile.denominator,
        "method_auxiliary_step_clip": profile.step_clip,
        "method_auxiliary_component_clip": profile.component_clip,
        "method_auxiliary_weights": dict(profile.weights),
    }
    mismatched = {
        field: (training_result.get(field), value)
        for field, value in expected.items()
        if training_result.get(field) != value
    }
    if mismatched:
        raise RuntimeError(
            f"completed training_result method auxiliary mismatch: {mismatched}"
        )
    if job["method"] == "drl_ts" and training_result.get(
        "soft_stage_end_epoch"
    ) != job.get("soft_stage_end_epoch"):
        raise RuntimeError(
            "completed training_result DRL-TS soft-stage boundary mismatch: "
            f"trainer={training_result.get('soft_stage_end_epoch')}, "
            f"manifest={job.get('soft_stage_end_epoch')}"
        )

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_args = payload.get("args", {}) or {}
    if not isinstance(saved_args, Mapping):
        saved_args = vars(saved_args)
    top_level = payload.get("method_auxiliary_profile")
    saved_snapshot = saved_args.get("method_auxiliary_snapshot")
    if top_level != saved_snapshot:
        raise RuntimeError("selected checkpoint auxiliary snapshots disagree")
    try:
        checkpoint_profile = MethodAuxiliaryProfile.from_payload(
            top_level
        ).require_method(job["method"])
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "selected checkpoint contains an invalid method auxiliary profile"
        ) from error
    if checkpoint_profile.to_dict() != profile.to_dict():
        raise RuntimeError(
            "selected checkpoint does not contain the requested auxiliary profile"
        )
    checkpoint_derived = {
        field: saved_args.get(field) for field in expected if field != "method_auxiliary_snapshot"
    }
    if checkpoint_derived != {
        field: value for field, value in expected.items()
        if field != "method_auxiliary_snapshot"
    }:
        raise RuntimeError(
            "selected checkpoint derived method auxiliary fields are inconsistent"
        )
    if job["method"] == "drl_ts":
        expected_runtime = {
            "soft_violation_contract_id": profile.profile_id,
            "soft_violation_step_clip": profile.step_clip,
            "soft_violation_component_clip": profile.component_clip,
            "soft_violation_denominator": profile.denominator,
            "capacity_penalty": profile.weights["capacity"],
            "time_penalty": profile.weights["time_window"],
            "energy_penalty": profile.weights["energy"],
            "soft_stage_end_epoch": job.get("soft_stage_end_epoch"),
        }
        actual_runtime = {
            field: saved_args.get(field) for field in expected_runtime
        }
        if actual_runtime != expected_runtime:
            raise RuntimeError(
                "selected DRL-TS checkpoint did not consume its auxiliary profile: "
                f"checkpoint={actual_runtime}, expected={expected_runtime}"
            )


def preflight(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    validate_terran_training_contracts(jobs)
    validate_objective_contracts(jobs)
    validate_reward_contracts(jobs)
    validate_method_auxiliary_contracts(jobs)
    validate_optimizer_contracts(jobs)
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing required environment variables: {', '.join(missing)}")
    repo = Path(os.environ["EVRPTW_REPO_ROOT"]).resolve()
    dataset = Path(os.environ["EVRPTW_DATASET_ROOT"]).resolve()
    output = Path(os.environ["EVRPTW_OUTPUT_ROOT"]).resolve()
    if not (repo / ".git").exists() or not dataset.is_dir():
        raise RuntimeError("repository or dataset root does not exist")
    if args.mode in {"full", "resume"}:
        validate_formal_launch_gates(
            repo,
            jobs,
            require_open=not args.dry_run,
        )
    validate_training_stream_contracts(jobs, repo, dataset)
    active_python_env = os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name
    output.mkdir(parents=True, exist_ok=True)
    probe = output / f".write_probe_{os.getpid()}"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    free = shutil.disk_usage(output).free
    if free < int(args.minimum_free_gib * 1024**3):
        raise RuntimeError(f"output free space is below {args.minimum_free_gib} GiB")
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repo, text=True
    ).strip()
    commit = git_commit(repo)
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).strip()
    if dirty and not args.dry_run:
        raise RuntimeError("working tree must be clean for a non-dry run")
    expected_branch = args.expected_branch
    if expected_branch and branch != expected_branch:
        raise RuntimeError(f"wrong branch: {branch}; expected {expected_branch}")
    gpu_names: list[str] = []
    if not args.skip_gpu_preflight:
        query = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).splitlines()
        if len(query) != args.local_gpu_count:
            raise RuntimeError(f"expected {args.local_gpu_count} GPUs, found {len(query)}")
        gpu_names = query
        bad = [name for name in query if not gpu_name_matches(name, args.gpu_name_pattern)]
        if bad:
            accepted = [
                pattern.strip()
                for pattern in args.gpu_name_pattern.split("|")
                if pattern.strip()
            ]
            raise RuntimeError(f"unexpected GPU model(s): {bad}; expected one of {accepted}")
    for job in jobs:
        index_key = "train_index" if job["kind"] == "train" else "dataset_index"
        path = dataset / job[index_key]
        if not path.is_file():
            raise FileNotFoundError(f"dataset index is missing for {job['job_id']}: {path}")
    context = {
        "repo": repo,
        "dataset": dataset,
        "output": output,
        "branch": branch,
        "commit": commit,
        "free_bytes": free,
        "conda_env": active_python_env,
        "configured_conda_env": os.environ.get("EVRPTW_CONDA_ENV"),
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "gpu_names": gpu_names,
        "launcher_id": validate_launcher_id(args.launcher_id),
        "slot_gpu_map": {
            str(slot): local_gpu
            for slot, local_gpu in parse_slot_gpu_map(
                args.slot_gpu_map,
                slots={
                    int(value)
                    for value in args.slots.split(",")
                    if value.strip()
                },
                local_gpu_count=args.local_gpu_count,
            ).items()
        },
    }
    if args.mode == "evaluate" and not args.dry_run:
        missing_checkpoints = [
            str(checkpoint_dir(job, context) / "checkpoint_selected.pt")
            for job in jobs
            if not (checkpoint_dir(job, context) / "checkpoint_selected.pt").is_file()
        ]
        if missing_checkpoints:
            raise FileNotFoundError(
                "evaluation checkpoint dependencies are missing; sync the listed "
                f"training outputs first: {missing_checkpoints[:5]}"
            )
    return context


def output_dir(job: dict[str, Any], context: dict[str, Any]) -> Path:
    root = context["output"] / job["representation"]
    if job.get("condition"):
        root = root / str(job["condition"])
    root = root / job["method"] / job["scale"] / f"seed_{job['seed']}" / context["commit"]
    if job["kind"] in {"eval", "transfer"}:
        root = root / job["test_id"] / job["decode_budget"]
    return root


def resolve_warm_start_checkpoint(
    job: Mapping[str, Any], context: Mapping[str, Any]
) -> Path | None:
    """Resolve an exact-job source checkpoint, never across task conditions."""

    source_commit = str(job.get("warm_start_source_commit", "")).strip()
    if not source_commit:
        return None
    if str(job.get("warm_start_scope", "")) != "exact_job_only":
        raise RuntimeError("warm-start scope must be exact_job_only")
    source_context = dict(context)
    source_context["commit"] = source_commit
    candidate = output_dir(dict(job), source_context) / str(
        job.get("warm_start_checkpoint_name", "best.ckpt")
    )
    if candidate.is_file():
        return candidate.resolve()
    policy = str(job.get("warm_start_missing_policy", "error"))
    if policy == "fresh":
        return None
    raise FileNotFoundError(
        f"exact-job warm-start checkpoint is missing: {candidate}"
    )


def training_command(job: dict[str, Any], context: dict[str, Any], out: Path, resume: bool) -> list[str]:
    dataset = context["dataset"]
    validation_steps = int(
        job.get(
            "validation_rollout_steps",
            validation_rollout_steps(int(job["training_rollout_steps"])),
        )
    )
    if job["method"] == "terran":
        command = [
            sys.executable,
            "-m",
            job["train_module"],
            "--config",
            str(TERRAN_CONFIG),
            "--seed",
            str(job["seed"]),
            "--device",
            "cuda",
            "--stage2-dataset-path",
            str(dataset / job["train_index"]),
            "--stage2-family-root",
            str(dataset / "materialized" / "families"),
            "--stage2-scale",
            job["scale"],
            "--num-customers",
            job["scale"].removeprefix("Cus"),
            "--num-envs-per-gpu",
            str(job["physical_batch_size"]),
            "--n-traj",
            str(job["training_trajectory_count"]),
            "--training-epochs",
            str(job["training_epochs"]),
            "--minimum-training-epochs",
            str(job.get("minimum_training_epochs", job["training_epochs"])),
            "--post-minimum-validation-every-epochs",
            str(job.get("post_minimum_validation_every_epochs", job.get("validation_every_epochs", job["training_epochs"]))),
            "--training-rollout-steps",
            str(job["training_rollout_steps"]),
            "--validation-rollout-steps",
            str(validation_steps),
            "--physical-batch-size",
            str(job["physical_batch_size"]),
            "--effective-batch-size",
            str(job["effective_batch_size"]),
            "--validation-dataset-path",
            str(dataset / job["validation_index"]),
            "--validation-family-root",
            str(dataset / "materialized" / "families"),
            "--validation-limit",
            str(job["validation_views"]),
            "--validation-decode-type",
            str(job["validation_decode_type"]),
            "--validation-candidates",
            str(job["validation_candidate_count"]),
            "--validation-seed",
            str(job.get("validation_seed", int(job["seed"]) + 910_000_000)),
            "--early-stop-patience-validations",
            str(job.get("early_stop_patience_validations", 0)),
            "--early-stop-start-epoch",
            str(job.get("early_stop_start_epoch", 0)),
            "--final-validation-limit",
            str(job.get("final_validation_views", 0)),
            "--validation-every-epochs",
            str(job.get("validation_every_epochs", job["training_epochs"])),
            "--validation-checkpoints",
            str(job["validation_checkpoints"]),
            "--protocol-id",
            job["protocol_id"],
            "--output-dir",
            str(out),
        ]
    else:
        command = [
            sys.executable,
            "-m",
            job["train_module"],
            "--dataset-path",
            str(dataset / job["train_index"]),
            "--family-root",
            str(dataset / "materialized" / "families"),
            "--scale",
            job["scale"],
            "--split-ids",
            "train",
            "--track-ids",
            "train",
            "--seed",
            str(job["seed"]),
            "--device",
            "cuda",
            "--training-epochs",
            str(job["training_epochs"]),
            "--minimum-training-epochs",
            str(job.get("minimum_training_epochs", job["training_epochs"])),
            "--post-minimum-validation-every-epochs",
            str(job.get("post_minimum_validation_every_epochs", job.get("validation_every_epochs", job["training_epochs"]))),
            "--training-rollout-steps",
            str(job["training_rollout_steps"]),
            "--validation-rollout-steps",
            str(validation_steps),
            "--physical-batch-size",
            str(job["physical_batch_size"]),
            "--effective-batch-size",
            str(job["effective_batch_size"]),
            "--samples-per-instance",
            str(job["training_trajectory_count"]),
            "--validation-dataset-path",
            str(dataset / job["validation_index"]),
            "--validation-family-root",
            str(dataset / "materialized" / "families"),
            "--validation-limit",
            str(job["validation_views"]),
            "--validation-decode-type",
            str(job["validation_decode_type"]),
            "--validation-candidates",
            str(job["validation_candidate_count"]),
            "--validation-seed",
            str(job.get("validation_seed", int(job["seed"]) + 910_000_000)),
            "--early-stop-patience-validations",
            str(job.get("early_stop_patience_validations", 0)),
            "--early-stop-start-epoch",
            str(job.get("early_stop_start_epoch", 0)),
            "--final-validation-limit",
            str(job.get("final_validation_views", 0)),
            "--validation-every-epochs",
            str(job.get("validation_every_epochs", job["training_epochs"])),
            "--validation-checkpoints",
            str(job["validation_checkpoints"]),
            "--protocol-id",
            job["protocol_id"],
            "--output-dir",
            str(out),
        ]
    if job.get("objective_config_path"):
        command.extend(["--objective-config", str(context["repo"] / job["objective_config_path"])])
    if job.get("reward_contract_config_path"):
        command.extend(
            [
                "--reward-contract",
                str(context["repo"] / job["reward_contract_config_path"]),
            ]
        )
    if job.get("method_auxiliary_profile_path"):
        command.extend(
            [
                "--method-auxiliary-profile",
                str(context["repo"] / job["method_auxiliary_profile_path"]),
            ]
        )
    command.extend(
        [
            "--optimizer",
            str(job["optimizer_name"]),
            "--weight-decay",
            str(job["optimizer_weight_decay"]),
        ]
    )
    if job["method"] == "drl_ts" and job.get("soft_stage_end_epoch") is not None:
        command.extend(["--soft-stage-end-epoch", str(job["soft_stage_end_epoch"])])
    if job["method"] == "terran":
        for field, option in (
            ("num_minibatches", "--num-minibatches"),
            ("ppo_step_chunk_size", "--ppo-step-chunk-size"),
            ("terran_terminal_success_bonus", "--terminal-success-bonus"),
        ):
            if field in job:
                command.extend([option, str(job[field])])
    if job.get("training_stream_path"):
        command.extend(
            [
                "--training-stream-path",
                str(context["repo"] / job["training_stream_path"]),
            ]
        )
        if job.get("training_stream_contract_sha256") is not None:
            command.extend(
                [
                    "--training-stream-contract-sha256",
                    str(job["training_stream_contract_sha256"]),
                ]
            )
        command.extend(
            [
                "--customer-exposure-budget",
                str(job["customer_exposure_budget"]),
                "--exposure-checkpoints",
                ",".join(str(value) for value in job.get("exposure_checkpoints", [])),
                "--gpu-hour-checkpoints",
                ",".join(str(value) for value in job.get("gpu_hour_checkpoints", [])),
            ]
        )
    command.extend(
        ["--training-representation", str(job.get("training_representation", "G"))]
    )
    if job.get("euclidean_manifest"):
        command.extend(
            [
                "--euclidean-manifest",
                str(context["repo"] / job["euclidean_manifest"]),
            ]
        )
    if resume:
        command.append("--resume")
    else:
        warm_start = resolve_warm_start_checkpoint(job, context)
        if warm_start is not None:
            command.extend(["--warm-start-checkpoint", str(warm_start)])
    return command


def checkpoint_dir(job: dict[str, Any], context: dict[str, Any]) -> Path:
    train = dict(job)
    train["scale"] = job.get("source_scale", job["scale"])
    train["kind"] = "train"
    return output_dir(train, context)


def evaluation_command(job: dict[str, Any], context: dict[str, Any], out: Path) -> list[str]:
    dataset = context["dataset"]
    checkpoint = checkpoint_dir(job, context) / "checkpoint_selected.pt"
    command = [
        sys.executable,
        "-m",
        job["eval_module"],
        "--dataset-path",
        str(dataset / job["dataset_index"]),
        "--family-root",
        str(dataset / "materialized" / "families"),
        "--checkpoint",
        str(checkpoint),
        "--scale",
        job["scale"],
        "--split-ids",
        "test",
        "--track-ids",
        job["track_id"],
        "--candidates",
        str(job["candidate_count"]),
        "--candidate-chunk-size",
        str(job["candidate_chunk_size"]),
        "--seed",
        str(job["seed"]),
        "--limit",
        str(job["expected_views"]),
        "--device",
        "cuda",
        "--output-dir",
        str(out),
    ]
    if job.get("objective_config_path"):
        command.extend(["--objective-config", str(context["repo"] / job["objective_config_path"])])
    if job["method"] == "terran":
        command.extend(["--decode-mode", "greedy" if job["decode_type"] == "greedy" else "sample"])
    else:
        command.extend(["--decode-type", job["decode_type"]])
    return command


def command_for(job: dict[str, Any], context: dict[str, Any], out: Path, resume: bool) -> list[str]:
    if "test_command" in job:
        return list(job["test_command"])
    if job["kind"] == "train":
        return training_command(job, context, out, resume)
    return evaluation_command(job, context, out)


def required_training_artifacts(job: dict[str, Any], out: Path) -> list[Path]:
    artifacts = [
        out / "checkpoint_selected.pt",
        out / "validation_summary.json",
        out / "training_result.json",
    ]
    for field in (
        "primary_checkpoint",
        "minimum_budget_checkpoint",
        "extended_checkpoint",
    ):
        if field in job:
            artifacts.append(out / str(job[field]))
    if int(job.get("final_validation_views", 0) or 0) > 0:
        artifacts.append(out / "validation_final_audit.json")
    return list(dict.fromkeys(artifacts))


def job_complete(
    job: dict[str, Any],
    out: Path,
    context: dict[str, Any] | None = None,
) -> bool:
    """Revalidate trainer-owned evidence every time a job is considered done."""

    try:
        result_path = out / "job_result.json"
        if not result_path.is_file():
            return False
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("status") != "passed":
            return False
        contract = training_contract(job)
        if contract:
            provenance_path = out / "provenance.json"
            if not provenance_path.is_file():
                return False
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if not isinstance(provenance, Mapping):
                return False
            if any(payload.get(key) != value for key, value in contract.items()):
                return False
            if any(provenance.get(key) != value for key, value in contract.items()):
                return False
            if training_contract(provenance.get("job", {})) != contract:
                return False
        if job["kind"] == "train":
            if not all(path.is_file() for path in required_training_artifacts(job, out)):
                return False
            training_result = json.loads(
                (out / "training_result.json").read_text(encoding="utf-8")
            )
            if not isinstance(training_result, Mapping):
                return False
            validate_training_result_outcome(job, training_result)
            validation_context = context or {
                "repo": ROOT.parents[1],
                "dataset": Path(os.environ.get("EVRPTW_DATASET_ROOT", ".")),
            }
            selected = out / "checkpoint_selected.pt"
            validate_completed_training_reward_contract(
                job, validation_context, training_result, selected
            )
            validate_completed_method_auxiliary_contract(
                job, validation_context, training_result, selected
            )
            validate_completed_training_stream_contract(
                job, validation_context, training_result, selected
            )
            validate_completed_training_signature(
                job, validation_context, training_result, selected
            )
            return True
        return (out / "summary.csv").is_file() and (out / "routes.jsonl").is_file()
    except Exception:
        return False


def should_resume_job(job: dict[str, Any], out: Path, requested: bool) -> bool:
    if not requested or job["kind"] != "train":
        return False
    state = (out / "data_pass_state.json").is_file()
    checkpoint = (out / "checkpoint_latest.pt").is_file()
    if state != checkpoint:
        raise RuntimeError(
            f"incomplete resume evidence for {job['job_id']}: "
            f"state={state} checkpoint={checkpoint}"
        )
    return state and checkpoint


def existing_training_state(out: Path) -> list[Path]:
    """Ignore launcher-only files while protecting checkpoints and training history."""
    patterns = (
        "data_pass_state.json", "training_result.json", "checkpoint*.pt", "best*.ckpt",
        "*history.jsonl", "validation_summary*.json", "early_stop_state.json",
        "logs/train_log.csv", "checkpoints/*",
    )
    return sorted({path for pattern in patterns for path in out.glob(pattern) if path.is_file()})


def run_job(job: dict[str, Any], context: dict[str, Any], local_gpu: int, resume: bool, dry_run: bool) -> bool:
    out = output_dir(job, context)
    out.mkdir(parents=True, exist_ok=True)
    if job_complete(job, out, context):
        return True
    resume_this_job = should_resume_job(job, out, resume)
    contract = training_contract(job)
    provenance_path = out / "provenance.json"
    if resume_this_job and "objective_config" in contract and not provenance_path.is_file():
        raise RuntimeError(
            f"missing objective provenance for resume of {job['job_id']}; "
            "start in a fresh directory instead of importing a historical checkpoint"
        )
    if resume_this_job and contract and provenance_path.is_file():
        previous_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if training_contract(previous_provenance.get("job", {})) != contract or any(
            previous_provenance.get(key) != value for key, value in contract.items()
        ):
            raise RuntimeError(
                f"resume provenance reward contract mismatch for {job['job_id']}; "
                "start fresh without reusing the old training directory"
            )
    warm_start_checkpoint = (
        resolve_warm_start_checkpoint(job, context)
        if job["kind"] == "train" and not resume_this_job else None
    )
    if job["kind"] == "train" and not resume_this_job:
        previous_state = existing_training_state(out)
        if previous_state:
            raise RuntimeError(
                f"refusing fresh training over existing state for {job['job_id']}: "
                f"{previous_state[:3]}; use matching-config resume or a fresh run directory"
            )
    command = command_for(job, context, out, resume_this_job)
    provenance = {
        "schema": "drl_job_provenance_v1",
        "job": job,
        "command": command,
        "git_commit": context["commit"],
        "git_branch": context["branch"],
        "warm_started_from_checkpoint": warm_start_checkpoint is not None,
        "warm_start_checkpoint": str(warm_start_checkpoint) if warm_start_checkpoint else None,
        "dataset_release_id": (context["dataset"] / "release_manifest.json").read_text(encoding="utf-8")[:4096]
        if (context["dataset"] / "release_manifest.json").exists()
        else "unavailable",
        "conda_env": context["conda_env"],
        "resume_requested": bool(resume),
        "resumed_from_checkpoint": bool(resume_this_job),
        "launcher_id": context.get("launcher_id", "default"),
        "logical_slot": int(job["global_slot"]),
        "local_gpu": int(local_gpu),
        **training_contract(job),
        "started_at": time.time(),
    }
    atomic_json(out / "provenance.json", provenance)
    if dry_run:
        print(json.dumps({"job_id": job["job_id"], "gpu": local_gpu, "command": command}, sort_keys=True))
        return True
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(local_gpu)
    started = time.perf_counter()
    with (out / "stdout.log").open("a", encoding="utf-8") as stdout, (out / "stderr.log").open("a", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=context["repo"], env=env, text=True, stdout=stdout, stderr=stderr, start_new_session=True)
        with CHILD_LOCK:
            CHILDREN[local_gpu] = process
        peak_cpu_memory_bytes = 0
        peak_gpu_memory_bytes = 0
        while process.poll() is None:
            peak_cpu_memory_bytes = max(
                peak_cpu_memory_bytes, process_rss_bytes(process.pid)
            )
            peak_gpu_memory_bytes = max(
                peak_gpu_memory_bytes, process_gpu_memory_bytes(process.pid)
            )
            time.sleep(0.5)
        returncode = process.wait()
        with CHILD_LOCK:
            CHILDREN.pop(local_gpu, None)
    stderr_text = (out / "stderr.log").read_text(encoding="utf-8", errors="replace")
    oom = returncode in {137, -9} or "out of memory" in stderr_text.lower()
    overflow_manifest = None
    if oom and job.get("hardware") == "2080ti" and job.get("scale") == "Cus500":
        overflow_manifest = record_a6000_overflow(job, context)
    artifacts_present = (
        all(path.is_file() for path in required_training_artifacts(job, out))
        if job["kind"] == "train"
        else ((out / "summary.csv").is_file() and (out / "routes.jsonl").is_file())
    )
    training_result_path = out / "training_result.json"
    training_result: Mapping[str, Any] = {}
    training_result_validation_error = None
    if job["kind"] == "train":
        try:
            loaded_training_result = json.loads(
                training_result_path.read_text(encoding="utf-8")
            )
            if not isinstance(loaded_training_result, Mapping):
                raise ValueError("training_result.json must contain an object")
            training_result = loaded_training_result
            validate_training_result_outcome(job, training_result)
        except Exception as error:
            training_result_validation_error = str(error)
    reward_contract_validation_error = None
    auxiliary_contract_validation_error = None
    training_stream_validation_error = None
    training_signature_validation_error = None
    if (
        returncode == 0
        and artifacts_present
        and job["kind"] == "train"
        and training_result_validation_error is None
    ):
        try:
            validate_completed_training_reward_contract(
                job,
                context,
                training_result,
                out / "checkpoint_selected.pt",
            )
        except Exception as error:
            reward_contract_validation_error = str(error)
        try:
            validate_completed_method_auxiliary_contract(
                job,
                context,
                training_result,
                out / "checkpoint_selected.pt",
            )
        except Exception as error:
            auxiliary_contract_validation_error = str(error)
        try:
            validate_completed_training_stream_contract(
                job,
                context,
                training_result,
                out / "checkpoint_selected.pt",
            )
        except Exception as error:
            training_stream_validation_error = str(error)
        try:
            validate_completed_training_signature(
                job,
                context,
                training_result,
                out / "checkpoint_selected.pt",
            )
        except Exception as error:
            training_signature_validation_error = str(error)
    passed = (
        returncode == 0
        and artifacts_present
        and training_result_validation_error is None
        and reward_contract_validation_error is None
        and auxiliary_contract_validation_error is None
        and training_stream_validation_error is None
        and training_signature_validation_error is None
    )
    result = {
        "schema": "drl_job_result_v1",
        "job_id": job["job_id"],
        "launcher_id": context.get("launcher_id", "default"),
        "logical_slot": int(job["global_slot"]),
        "local_gpu": int(local_gpu),
        "status": "passed" if passed else "failed",
        "returncode": returncode,
        "training_outcome": training_result.get("status"),
        "completed_training_epochs": training_result.get("completed_training_epochs"),
        "early_stopped": bool(training_result.get("early_stopped", False)),
        "early_stop_epoch": training_result.get("early_stop_epoch"),
        "wall_time_s": time.perf_counter() - started,
        "peak_cpu_memory_bytes": peak_cpu_memory_bytes,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "completed_at": time.time(),
        "failure_reason": (
            "OOM_UNCHANGED_CONFIG"
            if oom
            else (
                "TRAINING_METHOD_AUXILIARY_CONTRACT_MISMATCH"
                if auxiliary_contract_validation_error is not None
                else (
                    "TRAINING_SCIENTIFIC_SIGNATURE_MISMATCH"
                    if training_signature_validation_error is not None
                    else (
                        "TRAINING_STREAM_CONTRACT_MISMATCH"
                        if training_stream_validation_error is not None
                        else (
                            "TRAINING_REWARD_CONTRACT_MISMATCH"
                            if reward_contract_validation_error is not None
                            else (
                                "TRAINING_RESULT_INVALID"
                                if training_result_validation_error is not None
                                else None
                            )
                        )
                    )
                )
            )
        ),
        "training_result_validation_error": training_result_validation_error,
        "reward_contract_validation_error": reward_contract_validation_error,
        "method_auxiliary_validation_error": auxiliary_contract_validation_error,
        "training_stream_validation_error": training_stream_validation_error,
        "training_signature_validation_error": training_signature_validation_error,
        "overflow_manifest": str(overflow_manifest) if overflow_manifest else None,
        **training_contract(job),
    }
    atomic_json(out / "job_result.json", result)
    return passed


def worker(slot: int, jobs: list[dict[str, Any]], context: dict[str, Any], local_gpu: int, resume: bool, dry_run: bool, failures: list[str]) -> None:
    for job in jobs:
        if STOP.is_set():
            return
        try:
            passed = run_job(job, context, local_gpu, resume, dry_run)
        except Exception as exc:
            print(f"job {job['job_id']} refused or failed: {exc}", file=sys.stderr)
            passed = False
        if not passed:
            failures.append(job["job_id"])
            return


def handle_signal(signum: int, _frame: Any) -> None:
    STOP.set()
    with CHILD_LOCK:
        children = list(CHILDREN.values())
    for process in children:
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen DRL manifest queues.")
    parser.add_argument("mode", choices=("full", "evaluate", "status", "resume"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--slots", required=True)
    parser.add_argument(
        "--slot-gpu-map",
        default=None,
        help=(
            "Optional explicit logical-slot to local CUDA-device mapping, for "
            "example 1:1. It must map exactly --slots."
        ),
    )
    parser.add_argument(
        "--launcher-id",
        default="default",
        help="Safe launcher namespace identifier recorded in run provenance.",
    )
    parser.add_argument("--local-gpu-count", type=int, required=True)
    parser.add_argument("--gpu-name-pattern", required=True)
    parser.add_argument("--expected-branch", default="drl-benchmark-adapters")
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--seeds", default="1234")
    parser.add_argument("--scales", default="Cus50,Cus100,Cus500,Cus1000")
    parser.add_argument("--methods", default=None, help="Optional comma-separated method filter.")
    parser.add_argument("--skip-gpu-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slots = {int(value) for value in args.slots.split(",") if value.strip()}
    launcher_id = validate_launcher_id(args.launcher_id)
    slot_gpu_map = parse_slot_gpu_map(
        args.slot_gpu_map,
        slots=slots,
        local_gpu_count=args.local_gpu_count,
    )
    seeds = {int(value) for value in args.seeds.split(",") if value.strip()}
    scales = {value.strip() for value in args.scales.split(",") if value.strip()}
    methods = (
        {value.strip() for value in args.methods.split(",") if value.strip()}
        if args.methods is not None else None
    )
    if not seeds:
        raise ValueError("--seeds must select at least one seed")
    if not scales:
        raise ValueError("--scales must select at least one scale")
    if methods is not None and (not methods or methods - METHODS):
        raise ValueError(f"--methods must select from {sorted(METHODS)}")
    jobs = load_jobs(args.manifest, slots, args.mode, seeds=seeds, scales=scales, methods=methods)
    if args.mode != "status" and not jobs:
        raise ValueError(
            f"no {args.mode} jobs match seeds={sorted(seeds)} "
            f"and scales={sorted(scales)} methods={sorted(methods) if methods is not None else 'all'} "
            f"in {args.manifest}"
        )
    validate_job_routing(
        jobs,
        launcher_id=launcher_id,
        slot_gpu_map=slot_gpu_map,
    )
    context = preflight(args, jobs)
    # Tests and downstream callers may provide a minimal context, so populate
    # the launcher routing provenance here as the single authoritative fallback.
    context.setdefault("launcher_id", launcher_id)
    context.setdefault(
        "slot_gpu_map",
        {str(slot): local_gpu for slot, local_gpu in slot_gpu_map.items()},
    )
    if args.mode == "status":
        rows = []
        by_mode: dict[str, dict[str, int]] = {}
        for mode in ("full", "evaluate"):
            mode_rows = []
            for job in load_jobs(
                args.manifest, slots, mode, seeds=seeds, scales=scales, methods=methods
            ):
                row = {
                    "job_id": job["job_id"],
                    "complete": job_complete(job, output_dir(job, context), context),
                }
                rows.append(row)
                mode_rows.append(row)
            by_mode[mode] = {
                "planned": len(mode_rows),
                "completed": sum(row["complete"] for row in mode_rows),
            }
        print(
            json.dumps(
                {
                    "jobs": len(rows),
                    "completed": sum(row["complete"] for row in rows),
                    "by_mode": by_mode,
                    "rows": rows,
                },
                sort_keys=True,
            )
        )
        return
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    by_slot = {slot: [job for job in jobs if int(job["global_slot"]) == slot] for slot in slots}
    failures: list[str] = []
    threads = []
    for slot in sorted(slots):
        local_gpu = slot_gpu_map[slot]
        thread = threading.Thread(target=worker, args=(slot, by_slot[slot], context, local_gpu, args.mode == "resume", args.dry_run, failures), daemon=False)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit(f"failed queues: {failures}")


if __name__ == "__main__":
    main()
