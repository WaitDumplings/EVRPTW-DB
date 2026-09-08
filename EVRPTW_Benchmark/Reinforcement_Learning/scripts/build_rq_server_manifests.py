#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from EVRPTW_Benchmark.Reinforcement_Learning.common.method_auxiliary import (
    load_method_auxiliary_profile,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.reward_contract import (
    load_reward_contract,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_stream import (
    load_training_stream_contract,
    training_stream_contract_digest,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_protocol import (
    VALIDATION_ROLLOUT_STEPS_DENOMINATOR,
    VALIDATION_ROLLOUT_STEPS_NUMERATOR,
    validation_rollout_steps,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "drl_rq_runtime_candidates_v2.yaml"
STREAM_REGISTRY = ROOT / "configs" / "drl_training_stream_registry_v1.json"
TERRAN_CONFIG = ROOT / "TERRAN" / "configs" / "stage2_cus100_terran.yaml"
TERRAN_CUS1000_REPLACEMENT_CONFIG = (
    ROOT / "configs" / "drl_terran_cus1000_replacement_v1.yaml"
)
SCRIPT_ROOT = ROOT / "scripts" / "rq_v1"
GATE = "EVRPTW_Benchmark/Reinforcement_Learning/configs/drl_rq_formal_launch_gate_v1.json"
ARTIFACTS = "EVRPTW_Benchmark/results/DRL_rq_v1/artifacts"
METHODS = {
    "am_evrptw": "EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.train",
    "evrptw_rl": "EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.train",
    "drl_ts": "EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.train",
    "terran": "EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.train",
}
CUS1000_PRIORITY_SCHEDULE = (
    ("terran", 1),
    ("drl_ts", 0),
    ("evrptw_rl", 0),
    ("am_evrptw", 0),
)
TERRAN_FORMAL_SCHEDULE = (
    ("Cus500", 0),
    ("Cus1000", 1),
)
SERVERS = {
    "2080ti_4_1": ("2080ti", 4, "RTX 2080 Ti"),
    "2080ti_4_2": ("2080ti", 4, "RTX 2080 Ti"),
    "2080ti_3_1": ("2080ti", 3, "RTX 2080 Ti"),
    "a6000_2_1": ("a6000", 2, "RTX A6000|RTX 6000 Ada Generation"),
}
SPECIAL_WARM_START_COMMIT = "aa114d06995cdd35429bcc793bee4cff14590eb5"
SPECIAL_WARM_START_SERVERS = {"2080ti_4_2", "2080ti_3_1"}
SPECIAL_WARM_START_MANIFEST = "jobs_warm_start_aa114d0.jsonl"
SPECIAL_WARM_START_OBJECTIVE_PROFILE_ID = "rivian_energy_vehicle_cost_v1"

TRAIN_INDEX = {
    "Cus50": "generation_plan/compatibility_cus50/train/view_index.parquet",
    "Cus100": "generation_plan/core/train/view_index.parquet",
    "Cus500": "generation_plan/core/train/view_index.parquet",
    "Cus1000": "generation_plan/core/train/view_index.parquet",
}
VALIDATION_INDEX = {
    "Cus50": "generation_plan/compatibility_cus50/val/view_index.parquet",
    "Cus100": "generation_plan/core/val/view_index.parquet",
    "Cus500": "generation_plan/core/val/view_index.parquet",
    "Cus1000": "generation_plan/core/val/view_index.parquet",
}


def frozen_validation_rollout_steps(
    cfg: Mapping[str, Any], training_steps: int
) -> int:
    contract = cfg.get("validation_rollout_step_limit")
    expected_contract = {
        "relative_to": "training_rollout_steps",
        "numerator": VALIDATION_ROLLOUT_STEPS_NUMERATOR,
        "denominator": VALIDATION_ROLLOUT_STEPS_DENOMINATOR,
        "rounding": "ceiling",
    }
    if contract != expected_contract:
        raise ValueError(
            "runtime validation rollout-step contract mismatch: "
            f"{contract!r} != {expected_contract!r}"
        )
    return validation_rollout_steps(training_steps)


def load_training_stream_registry(cfg: Mapping[str, Any]) -> dict[str, Any]:
    configured_path = str(cfg.get("training_stream_registry_path", ""))
    if configured_path != str(STREAM_REGISTRY.relative_to(ROOT.parents[1])):
        raise ValueError("runtime config training-stream registry path mismatch")
    try:
        payload = json.loads(STREAM_REGISTRY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "frozen training-stream registry is missing or invalid"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("frozen artifact marker must contain an object")
    canonical = {
        key: value
        for key, value in payload.items()
        if key != "sha256"
    }
    digest = hashlib.sha256(
        json.dumps(
            canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    if (
        payload.get("schema") != "drl_training_stream_registry_v1"
        or payload.get("runtime_budget_id") != cfg["runtime_budget_id"]
        or payload.get("source_scope") != "training_split_and_track_only"
        or payload.get("sha256") != digest
        or payload.get("sha256") != cfg.get("training_stream_registry_sha256")
    ):
        raise ValueError("frozen training-stream registry contract mismatch")
    return payload


@lru_cache(maxsize=32)
def _load_training_stream_contract_for_file_identity(
    path: Path, mtime_ns: int, size: int,
) -> dict[str, Any]:
    # mtime/size are intentional cache-key fields. A replaced artifact is
    # revalidated, while repeated manifest builds avoid rehashing 10M rows.
    del mtime_ns, size
    return load_training_stream_contract(path)


def job(
    cfg: dict[str, Any], *, method: str, scale: str, seed: int,
    representation: str, condition: str, hardware: str,
    registry: Mapping[str, Any] | None = None,
    stream_contract_cache: dict[Path, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    training_overrides = (
        cfg.get("training_overrides_by_method_scale", {})
        .get(method, {})
        .get(scale, {})
    )
    if training_overrides and method != "terran":
        raise ValueError(
            "method/scale training overrides are supported only for TERRAN; "
            f"found {method}/{scale}"
        )
    allowed_training_overrides = {
        "training_rollout_steps",
        "validation_rollout_steps",
        "num_minibatches",
        "ppo_step_chunk_size",
        "terran_terminal_success_bonus",
    }
    unexpected_training_overrides = set(training_overrides).difference(
        allowed_training_overrides
    )
    if unexpected_training_overrides:
        raise ValueError(
            f"unsupported training override(s) for {method}/{scale}: "
            f"{sorted(unexpected_training_overrides)}"
        )
    logical = int(cfg["candidate_logical_batch_by_method_scale"][method][scale])
    environments_per_epoch = logical
    cap = int(cfg["physical_batch_caps"][method][scale])
    physical = cap
    if not 0 < physical <= logical:
        raise ValueError(
            f"physical batch must be in [1, logical batch]: {method} {scale}"
        )
    if method == "terran" and logical % physical:
        raise ValueError("TERRAN physical batch must divide its logical batch")
    updates = int(cfg["candidate_logical_epochs"][scale])
    minimum_updates = int(cfg["candidate_minimum_logical_epochs"][scale])
    if not 0 < minimum_updates <= updates:
        raise ValueError("minimum logical epochs must not exceed the hard cap")
    target_environments = updates * environments_per_epoch
    exposure = int(
        cfg["candidate_customer_exposure_budget_by_method_scale"][method][scale]
    )
    derived_exposure = target_environments * int(scale.removeprefix("Cus"))
    if exposure != derived_exposure:
        raise ValueError(
            f"epoch/environment schedule does not match exposure budget for {scale}: "
            f"{derived_exposure} != {exposure}"
        )
    validation_views = int(
        cfg["formal_candidate"]["selection_validation_views"][scale]
    )
    final_validation_views = int(
        cfg["formal_candidate"].get("final_validation_views", {}).get(scale, 0)
    )
    planning_wall_time_hours = cfg["formal_candidate"].get(
        "planning_wall_time_hours", {}
    ).get(scale)
    stream = (
        f"{ARTIFACTS}/streams/{cfg['runtime_budget_id']}/"
        f"formal/{condition}/{method}/{scale}/seed_{seed}.parquet"
    )
    if registry is None:
        registry = load_training_stream_registry(cfg)
    registry_key = f"{representation}/{condition}/{method}/{scale}/seed_{seed}"
    try:
        registered_stream = registry["streams"][registry_key]
        stream_contract = registered_stream["snapshot"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"training stream is not frozen in the registry: {registry_key}"
        ) from error
    if registered_stream.get("path") != stream:
        raise ValueError("training-stream registry path mismatch")
    if (
        not isinstance(stream_contract, dict)
        or stream_contract.get("sha256")
        != training_stream_contract_digest(stream_contract)
    ):
        raise ValueError("training-stream registry contains an invalid snapshot")
    artifact_path = ROOT.parents[1] / stream
    if artifact_path.is_file():
        actual_stream_contract = (
            stream_contract_cache.get(artifact_path)
            if stream_contract_cache is not None
            else None
        )
        if actual_stream_contract is None:
            stat = artifact_path.stat()
            actual_stream_contract = _load_training_stream_contract_for_file_identity(
                artifact_path, stat.st_mtime_ns, stat.st_size
            )
            if stream_contract_cache is not None:
                stream_contract_cache[artifact_path] = actual_stream_contract
        if actual_stream_contract != stream_contract:
            raise ValueError("local training stream disagrees with frozen registry")
    if (
        stream_contract["scale"] != scale
        or stream_contract["seed"] != int(seed)
        or stream_contract["sample_count"] != target_environments
    ):
        raise ValueError(
            "frozen training stream does not match the requested formal job: "
            f"{stream_contract}"
        )
    seed_tag = "_".join(str(int(value)) for value in cfg["seeds"])
    marker_path = (
        f"{ARTIFACTS}/preparation_{cfg['runtime_budget_id']}_seeds_{seed_tag}.json"
    )
    training_steps = int(
        training_overrides.get("training_rollout_steps", cfg["rollout_steps"][scale])
    )
    if training_steps <= 0:
        raise ValueError(f"training_rollout_steps must be positive for {method}/{scale}")
    derived_validation_steps = frozen_validation_rollout_steps(cfg, training_steps)
    configured_validation_steps = int(
        training_overrides.get("validation_rollout_steps", derived_validation_steps)
    )
    if configured_validation_steps != derived_validation_steps:
        raise ValueError(
            f"validation_rollout_steps must equal ceil(3/2 * training) for "
            f"{method}/{scale}: {configured_validation_steps} != "
            f"{derived_validation_steps}"
        )
    payload = {
        "schema": "drl_rq_job_manifest_v1",
        "protocol_id": "drl_rq_protocol_frozen_v1",
        "runtime_budget_id": cfg["runtime_budget_id"],
        "job_id": f"full__{representation}__{condition}__{method}__{scale}__seed{seed}",
        "enabled": True,
        "kind": "train",
        "run_mode": "full",
        "stage": "formal_training",
        "hardware": hardware,
        "representation": representation,
        "training_representation": representation,
        "condition": condition,
        "method": method,
        "scale": scale,
        "seed": seed,
        "warm_start_source_commit": str(cfg["warm_start"]["source_commit"]),
        "warm_start_scope": str(cfg["warm_start"]["scope"]),
        "warm_start_checkpoint_name": str(cfg["warm_start"]["checkpoint_name"]),
        "warm_start_missing_policy": str(cfg["warm_start"]["missing_exact_job"]),
        "train_module": METHODS[method],
        "train_index": TRAIN_INDEX[scale],
        "validation_index": VALIDATION_INDEX[scale],
        "training_stream_path": stream,
        "training_stream_contract_sha256": stream_contract["sha256"],
        "training_stream_contract_snapshot": stream_contract,
        "artifact_preparation_marker_path": marker_path,
        "artifact_preparation_marker_sha256": registry[
            "artifact_preparation_marker_sha256"
        ],
        "training_stream_registry_path": str(
            STREAM_REGISTRY.relative_to(ROOT.parents[1])
        ),
        "training_stream_registry_sha256": registry["sha256"],
        "customer_exposure_budget": exposure,
        "target_environments": target_environments,
        "exposure_checkpoints": [
            int(exposure * float(fraction))
            for fraction in cfg["formal_candidate"]["exposure_checkpoints_fraction"]
        ],
        "gpu_hour_checkpoints": cfg["formal_candidate"]["gpu_hour_checkpoints"],
        "training_epochs": updates,
        "minimum_training_epochs": minimum_updates,
        "minimum_target_environments": minimum_updates * environments_per_epoch,
        "minimum_customer_exposure_budget": (
            minimum_updates * environments_per_epoch * int(scale.removeprefix("Cus"))
        ),
        "logical_environments_per_epoch": environments_per_epoch,
        "planned_logical_epochs": updates,
        "training_rollout_steps": training_steps,
        "validation_rollout_steps": configured_validation_steps,
        "optimizer_name": str(cfg["training_optimizer"]["name"]),
        "optimizer_weight_decay": float(
            cfg["training_optimizer"]["weight_decay"]
        ),
        "training_trajectory_count": int(
            cfg.get("training_trajectory_count_by_method_scale", {})
            .get(method, {})
            .get(scale, cfg["training_trajectory_count_by_method"][method])
        ),
        "physical_batch_size": physical,
        "effective_batch_size": logical,
        "validation_views": validation_views,
        "validation_decode_type": cfg["evaluation"]["validation_decode_type"],
        "validation_candidate_count": int(
            cfg["evaluation"]["validation_candidate_count"]
        ),
        "validation_seed": int(seed)
        + int(cfg["formal_candidate"]["validation_seed_offset"]),
        "test_decode_type": cfg["evaluation"]["test_decode_type"],
        "test_candidate_count": int(cfg["evaluation"]["test_candidate_count"]),
        "candidate_selection": cfg["evaluation"]["selection"],
        "early_stop_patience_validations": int(cfg["formal_candidate"].get("early_stop_patience_validations", 0)),
        "early_stop_start_epoch": int(cfg["formal_candidate"].get("early_stop_start_epoch", 0)),
        "post_minimum_validation_every_epochs": int(
            cfg["formal_candidate"].get(
                "post_minimum_validation_every_epochs",
                cfg["formal_candidate"]["validation_every_epochs"],
            )
        ),
        "soft_stage_end_epoch": (
            int(cfg["formal_candidate"]["drl_ts_soft_stage_end_epoch"])
            if method == "drl_ts" else None
        ),
        "primary_checkpoint": "best_overall.ckpt",
        "minimum_budget_checkpoint": "best_within_5000.ckpt",
        "extended_checkpoint": "best_overall.ckpt",
        "final_validation_views": final_validation_views,
        "validation_every_epochs": int(
            cfg["formal_candidate"]["validation_every_epochs"]
        ),
        "validation_checkpoints": len(
            set(
                range(
                    int(cfg["formal_candidate"]["validation_every_epochs"]),
                    minimum_updates + 1,
                    int(cfg["formal_candidate"]["validation_every_epochs"]),
                )
            )
            | {minimum_updates}
            | set(
                range(
                    minimum_updates + int(cfg["formal_candidate"]["post_minimum_validation_every_epochs"]),
                    updates + 1,
                    int(cfg["formal_candidate"]["post_minimum_validation_every_epochs"]),
                )
            )
            | {updates}
        ),
        "planning_wall_time_hours": planning_wall_time_hours,
        "formal_gate_file": GATE,
        "euclidean_manifest": (
            f"{ARTIFACTS}/euclidean/euclidean_calibration_manifest.json"
            if representation == "E" else None
        ),
        "file_hash_validation_performed": True,
    }
    objective_path = cfg["objective_config_path"]
    objective = json.loads(
        (ROOT.parents[1] / objective_path).read_text(encoding="utf-8")
    )["objective"]
    payload.update(objective_config_path=objective_path, objective_config=objective)
    reward_contract_path = cfg["reward_contract_config_path"]
    reward_contract = load_reward_contract(ROOT.parents[1] / reward_contract_path)
    reward_terms = reward_contract.for_scale(scale, objective)
    payload.update(
        reward_contract_config_path=reward_contract_path,
        reward_contract_id=reward_terms.contract_id,
        reward_contract_sha256=reward_terms.digest,
        reward_objective_scale=reward_terms.objective_scale,
        reward_failure_base=reward_terms.failure_base,
        reward_unserved_coefficient=reward_terms.unserved_coefficient,
    )
    auxiliary_path = cfg.get("method_auxiliary_profiles", {}).get(method)
    if auxiliary_path is not None:
        auxiliary = load_method_auxiliary_profile(
            ROOT.parents[1] / auxiliary_path
        ).require_method(method)
        payload.update(
            method_auxiliary_profile_path=auxiliary_path,
            method_auxiliary_profile_id=auxiliary.profile_id,
            method_auxiliary_sha256=auxiliary.digest,
            method_auxiliary_method=auxiliary.method,
            method_auxiliary_applicability=auxiliary.applicability,
            method_auxiliary_aggregation=auxiliary.aggregation,
            method_auxiliary_denominator=auxiliary.denominator,
            method_auxiliary_step_clip=auxiliary.step_clip,
            method_auxiliary_component_clip=auxiliary.component_clip,
            method_auxiliary_weights=dict(auxiliary.weights),
        )
    if method == "terran":
        training = yaml.safe_load(TERRAN_CONFIG.read_text(encoding="utf-8"))["training"]
        if training["reward_contract_id"] != reward_terms.contract_id:
            raise ValueError("TERRAN config and common reward contract disagree")
        payload.update(
            training_gamma=float(training["gamma"]),
        )
    for field in (
        "num_minibatches",
        "ppo_step_chunk_size",
        "terran_terminal_success_bonus",
    ):
        if field not in training_overrides:
            continue
        raw_value = training_overrides[field]
        value = (
            float(raw_value)
            if field == "terran_terminal_success_bonus"
            else int(raw_value)
        )
        if field == "terran_terminal_success_bonus":
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{field} must be non-negative for {method}/{scale}: {value}"
                )
        elif value <= 0:
            raise ValueError(
                f"{field} must be positive for {method}/{scale}: {value}"
            )
        payload[field] = value
    if method == "terran" and "terran_terminal_success_bonus" not in payload:
        raise ValueError(
            f"TERRAN {scale} must explicitly freeze terran_terminal_success_bonus"
        )
    return payload


def assign_round_robin(
    queues: dict[str, list[dict[str, Any]]],
    jobs: list[dict[str, Any]],
    servers: list[str],
) -> None:
    slots = [(server, slot) for server in servers for slot in range(SERVERS[server][1])]
    for index, payload in enumerate(jobs):
        server, slot = slots[index % len(slots)]
        payload["hardware"] = SERVERS[server][0]
        payload["global_slot"] = slot
        payload["queue_position"] = sum(
            row.get("global_slot") == slot for row in queues[server]
        )
        queues[server].append(payload)


def validate_reward_contract_scope(cfg: dict[str, Any]) -> None:
    """Fail closed before generating jobs for an uncalibrated scale."""

    reward_contract_path = cfg["reward_contract_config_path"]
    reward_contract = load_reward_contract(
        ROOT.parents[1] / reward_contract_path
    )
    configured = tuple(cfg.get("reward_contract_calibrated_scales", ()))
    if not configured or len(configured) != len(set(configured)):
        raise ValueError(
            "reward_contract_calibrated_scales must be a nonempty unique list"
        )
    actual = set(reward_contract.scales)
    if set(configured) != actual:
        raise ValueError(
            "reward_contract_calibrated_scales must exactly match the frozen "
            f"contract; configured={sorted(configured)}, contract={sorted(actual)}"
        )
    if cfg.get("reward_contract_revision") != reward_contract.contract_id:
        raise ValueError(
            "reward_contract_revision must match the frozen contract_id"
        )

    enabled = set(cfg.get("enabled_scales", ()))
    uncalibrated = enabled.difference(actual)
    if uncalibrated:
        raise ValueError(
            "enabled scale(s) have no frozen reward calibration: "
            f"{sorted(uncalibrated)}"
        )
    blocked = set(cfg.get("reward_contract_blocked_scales", ()))
    if enabled.intersection(blocked):
        raise ValueError(
            "reward-contract-blocked scales cannot be enabled: "
            f"{sorted(enabled.intersection(blocked))}"
        )
    incorrectly_blocked = blocked.intersection(actual)
    if incorrectly_blocked:
        raise ValueError(
            "calibrated reward-contract scales cannot also be blocked: "
            f"{sorted(incorrectly_blocked)}"
        )


def build() -> dict[str, list[dict[str, Any]]]:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_reward_contract_scope(cfg)
    registry = load_training_stream_registry(cfg)
    stream_contract_cache: dict[Path, dict[str, Any]] = {}
    # Materialize every server queue, including any legitimately empty queue,
    # so checked-in manifests remain a complete scheduling contract.
    queues: dict[str, list[dict[str, Any]]] = {
        server: [] for server in SERVERS
    }
    scales = tuple(cfg["enabled_scales"])
    assigned_scales = [
        scale
        for hardware_scales in cfg["scale_hardware"].values()
        for scale in hardware_scales
    ]
    scale_hardware = {
        scale: hardware
        for hardware, hardware_scales in cfg["scale_hardware"].items()
        for scale in hardware_scales
    }
    if (
        len(assigned_scales) != len(set(assigned_scales))
        or set(scale_hardware) != set(scales)
    ):
        raise ValueError("scale_hardware must assign every enabled scale exactly once")
    servers_by_hardware = {
        hardware: [server for server, spec in SERVERS.items() if spec[0] == hardware]
        for hardware in {spec[0] for spec in SERVERS.values()}
    }

    formal_by_hardware: dict[str, list[dict[str, Any]]] = {
        hardware: [] for hardware in servers_by_hardware
    }
    for seed in cfg["seeds"]:
        for scale in scales:
            hardware = scale_hardware[scale]
            formal_by_hardware[hardware].extend(
                job(
                    cfg,
                    method=method,
                    scale=scale,
                    seed=seed,
                    representation="G",
                    condition="Full-support",
                    hardware=hardware,
                    registry=registry,
                    stream_contract_cache=stream_contract_cache,
                )
                for method in METHODS
            )
        if "Cus100" in scales:
            cus100_hardware = scale_hardware["Cus100"]
            formal_by_hardware[cus100_hardware].extend(
                job(
                    cfg,
                    method=method,
                    scale="Cus100",
                    seed=seed,
                    representation="G",
                    condition=condition,
                    hardware=cus100_hardware,
                    registry=registry,
                    stream_contract_cache=stream_contract_cache,
                )
                for condition in ("Random-10%-support", "Coverage-10%-support")
                for method in ("am_evrptw", "terran")
            )
            formal_by_hardware[cus100_hardware].extend(
                job(
                    cfg,
                    method=method,
                    scale="Cus100",
                    seed=seed,
                    representation="E",
                    condition="Full-support",
                    hardware=cus100_hardware,
                    registry=registry,
                    stream_contract_cache=stream_contract_cache,
                )
                for method in METHODS
            )
    for hardware, formal in formal_by_hardware.items():
        assign_round_robin(queues, formal, servers_by_hardware[hardware])
    return queues


def build_a6000_cus1000_priority_queue(
    queues: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Project the canonical A6000 jobs into the approved Cus1000 priority queue."""
    source = (queues or build())["a6000_2_1"]
    selected = {
        str(row["method"]): row
        for row in source
        if row["run_mode"] == "full"
        and row["representation"] == "G"
        and row["condition"] == "Full-support"
        and row["scale"] == "Cus1000"
        and int(row["seed"]) == 1234
    }
    expected = {method for method, _slot in CUS1000_PRIORITY_SCHEDULE}
    if set(selected) != expected:
        raise ValueError(
            "Cus1000 priority queue requires exactly one seed-1234 job for "
            f"each method; found {sorted(selected)}"
        )

    queue_positions: dict[int, int] = defaultdict(int)
    priority: list[dict[str, Any]] = []
    for method, slot in CUS1000_PRIORITY_SCHEDULE:
        payload = dict(selected[method])
        payload["global_slot"] = slot
        payload["queue_position"] = queue_positions[slot]
        queue_positions[slot] += 1
        priority.append(payload)
    return priority


def build_a6000_terran_formal_queue(
    queues: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Place the two explicitly authorized TERRAN jobs on separate GPUs."""

    source = (queues or build())["a6000_2_1"]
    selected: dict[str, dict[str, Any]] = {}
    for row in source:
        if (
            row["run_mode"] == "full"
            and row["representation"] == "G"
            and row["condition"] == "Full-support"
            and row["method"] == "terran"
            and int(row["seed"]) == 1234
            and row["scale"] in {scale for scale, _slot in TERRAN_FORMAL_SCHEDULE}
        ):
            scale = str(row["scale"])
            if scale in selected:
                raise ValueError(f"duplicate authorized TERRAN job for {scale}")
            selected[scale] = row

    expected_scales = {scale for scale, _slot in TERRAN_FORMAL_SCHEDULE}
    if set(selected) != expected_scales:
        raise ValueError(
            "TERRAN formal queue requires exactly one seed-1234 job for each "
            f"authorized scale; found {sorted(selected)}"
        )

    rows: list[dict[str, Any]] = []
    for scale, slot in TERRAN_FORMAL_SCHEDULE:
        payload = dict(selected[scale])
        payload["global_slot"] = slot
        payload["queue_position"] = 0
        rows.append(payload)

    runtime = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    authorized = set(runtime.get("authorized_job_ids", ()))
    actual = {str(row["job_id"]) for row in rows}
    if (
        bool(runtime.get("formal_launch_allowed"))
        and not actual.issubset(authorized)
    ):
        raise ValueError(
            "dedicated TERRAN queue contains job IDs outside authorized_job_ids: "
            f"queue={sorted(actual)}, authorized={sorted(authorized)}"
        )
    return rows


def build_a6000_terran_cus1000_replacement_queue(
    queues: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Build the isolated one-job replacement queue for physical GPU 1."""

    profile = yaml.safe_load(
        TERRAN_CUS1000_REPLACEMENT_CONFIG.read_text(encoding="utf-8")
    )
    if profile.get("schema") != "drl_terran_cus1000_replacement_v1":
        raise ValueError("invalid TERRAN Cus1000 replacement profile schema")
    source = (queues or build())["a6000_2_1"]
    matches = [row for row in source if row["job_id"] == profile["job_id"]]
    if len(matches) != 1:
        raise ValueError(
            "replacement profile must resolve exactly one canonical job; "
            f"found {len(matches)}"
        )
    payload = dict(matches[0])
    expected = {
        "method": str(profile["method"]),
        "scale": str(profile["scale"]),
        "seed": int(profile["seed"]),
        "training_rollout_steps": int(profile["training_rollout_steps"]),
        "validation_rollout_steps": int(profile["validation_rollout_steps"]),
        "num_minibatches": int(profile["num_minibatches"]),
        "ppo_step_chunk_size": int(profile["ppo_step_chunk_size"]),
        "terran_terminal_success_bonus": float(
            profile["terran_terminal_success_bonus"]
        ),
    }
    actual = {field: payload.get(field) for field in expected}
    if actual != expected:
        raise ValueError(
            "canonical TERRAN Cus1000 job disagrees with replacement profile: "
            f"actual={actual}, expected={expected}"
        )
    if expected["validation_rollout_steps"] != validation_rollout_steps(
        expected["training_rollout_steps"]
    ):
        raise ValueError("replacement validation horizon violates the 3/2 contract")
    runtime = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if (
        bool(runtime.get("formal_launch_allowed"))
        and payload["job_id"] not in set(runtime.get("authorized_job_ids", ()))
    ):
        raise ValueError("replacement job is outside authorized_job_ids")
    payload.update(
        global_slot=int(profile["global_slot"]),
        queue_position=0,
        launcher_profile_id=str(profile["profile_id"]),
        required_launcher_id=str(profile["launcher_id"]),
        required_local_gpu=int(profile["local_gpu"]),
    )
    return [payload]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build four frozen RQ training queues.")
    parser.add_argument("--output-root", type=Path, default=SCRIPT_ROOT)
    args = parser.parse_args()
    runtime_config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    formal_launch_allowed = bool(runtime_config["formal_launch_allowed"])
    launch_policy = str(runtime_config["launch_policy"])
    authorized_job_ids = set(runtime_config["authorized_job_ids"])
    objective_profile_id = str(
        json.loads(
            (
                ROOT.parents[1] / runtime_config["objective_config_path"]
            ).read_text(encoding="utf-8")
        )["objective"]["profile_id"]
    )
    queues = build()
    for server, rows in queues.items():
        destination = args.output_root / server
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "jobs.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        # This manifest is historical evidence tied to objective v1. Once the
        # active objective migrates, leave any checked-in copy byte-for-byte
        # untouched rather than silently relabelling aa114d0 checkpoints.
        if (
            server in SPECIAL_WARM_START_SERVERS
            and objective_profile_id == SPECIAL_WARM_START_OBJECTIVE_PROFILE_ID
        ):
            warm_rows = []
            for row in rows:
                payload = dict(row)
                payload["warm_start_source_commit"] = SPECIAL_WARM_START_COMMIT
                payload["warm_start_missing_policy"] = "error"
                warm_rows.append(payload)
            (destination / SPECIAL_WARM_START_MANIFEST).write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n" for row in warm_rows
                ),
                encoding="utf-8",
            )
        summary = {
            "schema": "drl_rq_server_assignment_v1",
            "server": server,
            "hardware": SERVERS[server][0],
            "gpu_count": SERVERS[server][1],
            "pilot_jobs": 0,
            "formal_jobs": sum(row["run_mode"] == "full" for row in rows),
            "formal_launch_allowed": bool(rows)
            and formal_launch_allowed
            and {str(row["job_id"]) for row in rows}.issubset(
                authorized_job_ids
            ),
            "authorized_formal_job_ids": sorted(
                authorized_job_ids.intersection(
                    str(row["job_id"]) for row in rows
                )
            ),
            "launch_policy": (
                launch_policy if rows else "blocked_no_calibrated_reward_scale"
            ),
        }
        (destination / "assignment_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    cus1000_priority = build_a6000_cus1000_priority_queue(queues)
    a6000_destination = args.output_root / "a6000_2_1"
    (a6000_destination / "cus1000_jobs.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in cus1000_priority),
        encoding="utf-8",
    )
    priority_summary = {
        "schema": "drl_rq_server_assignment_v1",
        "server": "a6000_2_1",
        "profile": "cus1000_priority",
        "hardware": "a6000",
        "gpu_count": 2,
        "pilot_jobs": 0,
        "formal_jobs": len(cus1000_priority),
        "formal_launch_allowed": formal_launch_allowed
        and {str(row["job_id"]) for row in cus1000_priority}.issubset(
            authorized_job_ids
        ),
        "authorized_formal_job_ids": sorted(
            authorized_job_ids.intersection(
                str(row["job_id"]) for row in cus1000_priority
            )
        ),
        "launch_policy": launch_policy,
        "slot_queues": {
            "0": ["drl_ts", "evrptw_rl", "am_evrptw"],
            "1": ["terran"],
        },
    }
    (a6000_destination / "cus1000_assignment_summary.json").write_text(
        json.dumps(priority_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    terran_formal = build_a6000_terran_formal_queue(queues)
    (a6000_destination / "terran_jobs.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in terran_formal),
        encoding="utf-8",
    )
    terran_summary = {
        "schema": "drl_rq_server_assignment_v1",
        "server": "a6000_2_1",
        "profile": "terran_formal",
        "hardware": "a6000",
        "gpu_count": 2,
        "pilot_jobs": 0,
        "formal_jobs": len(terran_formal),
        "formal_launch_allowed": formal_launch_allowed
        and {str(row["job_id"]) for row in terran_formal}.issubset(
            authorized_job_ids
        ),
        "launch_policy": launch_policy,
        "authorized_formal_job_ids": sorted(
            authorized_job_ids.intersection(
                str(row["job_id"]) for row in terran_formal
            )
        ),
        "slot_queues": {
            "0": ["terran/Cus500"],
            "1": ["terran/Cus1000"],
        },
    }
    (a6000_destination / "terran_assignment_summary.json").write_text(
        json.dumps(terran_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    terran_replacement = build_a6000_terran_cus1000_replacement_queue(queues)
    (a6000_destination / "terran_cus1000_replacement_jobs.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in terran_replacement),
        encoding="utf-8",
    )
    replacement_profile = yaml.safe_load(
        TERRAN_CUS1000_REPLACEMENT_CONFIG.read_text(encoding="utf-8")
    )
    replacement_summary = {
        "schema": "drl_rq_server_assignment_v1",
        "server": "a6000_2_1",
        "profile": str(replacement_profile["profile_id"]),
        "launcher_id": str(replacement_profile["launcher_id"]),
        "hardware": "a6000",
        "gpu_count": 2,
        "pilot_jobs": 0,
        "formal_jobs": 1,
        "formal_launch_allowed": formal_launch_allowed
        and {str(row["job_id"]) for row in terran_replacement}.issubset(
            authorized_job_ids
        ),
        "launch_policy": launch_policy,
        "authorized_formal_job_ids": sorted(
            authorized_job_ids.intersection(
                str(row["job_id"]) for row in terran_replacement
            )
        ),
        "slot_gpu_map": {
            str(replacement_profile["global_slot"]): int(
                replacement_profile["local_gpu"]
            )
        },
        "slot_queues": {"1": ["terran/Cus1000"]},
    }
    (
        a6000_destination
        / "terran_cus1000_replacement_assignment_summary.json"
    ).write_text(
        json.dumps(replacement_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({server: len(rows) for server, rows in queues.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
