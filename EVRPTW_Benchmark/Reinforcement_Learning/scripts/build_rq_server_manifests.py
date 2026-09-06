#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
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


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "drl_rq_runtime_candidates_v2.yaml"
STREAM_REGISTRY = ROOT / "configs" / "drl_training_stream_registry_v1.json"
TERRAN_CONFIG = ROOT / "TERRAN" / "configs" / "stage2_cus100_terran.yaml"
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
SERVERS = {
    "2080ti_4_1": ("2080ti", 4, "RTX 2080 Ti"),
    "2080ti_4_2": ("2080ti", 4, "RTX 2080 Ti"),
    "2080ti_3_1": ("2080ti", 3, "RTX 2080 Ti"),
    "a6000_2_1": ("a6000", 2, "RTX A6000|RTX 6000 Ada Generation"),
}
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


def job(
    cfg: dict[str, Any], *, method: str, scale: str, seed: int,
    representation: str, condition: str, hardware: str,
) -> dict[str, Any]:
    logical = int(cfg["candidate_logical_batch"][scale])
    environments_per_epoch = int(cfg["candidate_environments_per_epoch"][scale])
    if logical != environments_per_epoch:
        raise ValueError(
            f"logical batch must equal environments per epoch for {scale}: "
            f"{logical} != {environments_per_epoch}"
        )
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
    exposure = int(cfg["candidate_customer_exposure_budget"][scale])
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
        f"formal/{condition}/{scale}/seed_{seed}.parquet"
    )
    registry = load_training_stream_registry(cfg)
    registry_key = f"{representation}/{condition}/{scale}/seed_{seed}"
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
        actual_stream_contract = load_training_stream_contract(artifact_path)
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
        "training_rollout_steps": int(cfg["rollout_steps"][scale]),
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
    training_overrides = (
        cfg.get("training_overrides_by_method_scale", {})
        .get(method, {})
        .get(scale, {})
    )
    if training_overrides and method != "terran":
        raise ValueError(
            "PPO training overrides are supported only for TERRAN; "
            f"found {method}/{scale}"
        )
    allowed_training_overrides = {"num_minibatches", "ppo_step_chunk_size"}
    unexpected_training_overrides = set(training_overrides).difference(
        allowed_training_overrides
    )
    if unexpected_training_overrides:
        raise ValueError(
            f"unsupported training override(s) for {method}/{scale}: "
            f"{sorted(unexpected_training_overrides)}"
        )
    for field, value in training_overrides.items():
        value = int(value)
        if value <= 0:
            raise ValueError(
                f"{field} must be positive for {method}/{scale}: {value}"
            )
        payload[field] = value
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
    # Materialize empty server queues too.  In this contract revision the
    # 2080 Ti bundles intentionally contain no jobs until Cus50/Cus100 are
    # calibrated, and their checked-in empty manifests are part of the gate.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build four frozen RQ training queues.")
    parser.add_argument("--output-root", type=Path, default=SCRIPT_ROOT)
    args = parser.parse_args()
    runtime_config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    formal_launch_allowed = bool(runtime_config["formal_launch_allowed"])
    launch_policy = str(runtime_config["launch_policy"])
    queues = build()
    for server, rows in queues.items():
        destination = args.output_root / server
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "jobs.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        summary = {
            "schema": "drl_rq_server_assignment_v1",
            "server": server,
            "hardware": SERVERS[server][0],
            "gpu_count": SERVERS[server][1],
            "pilot_jobs": 0,
            "formal_jobs": sum(row["run_mode"] == "full" for row in rows),
            "formal_launch_allowed": bool(rows) and formal_launch_allowed,
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
        "formal_launch_allowed": formal_launch_allowed,
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
    print(json.dumps({server: len(rows) for server, rows in queues.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
