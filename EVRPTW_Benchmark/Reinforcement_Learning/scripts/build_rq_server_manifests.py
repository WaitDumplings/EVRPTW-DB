#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "drl_rq_runtime_candidates_v2.yaml"
SCRIPT_ROOT = ROOT / "scripts" / "rq_v1"
GATE = "EVRPTW_Benchmark/Reinforcement_Learning/configs/drl_rq_formal_launch_gate_v1.json"
ARTIFACTS = "EVRPTW_Benchmark/results/DRL_rq_v1/artifacts"
METHODS = {
    "am_evrptw": "EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.train",
    "evrptw_rl": "EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.train",
    "drl_ts": "EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.train",
    "terran": "EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.train",
}
SERVERS = {
    "2080ti_4_1": ("2080ti", 4, "RTX 2080 Ti"),
    "2080ti_4_2": ("2080ti", 4, "RTX 2080 Ti"),
    "2080ti_3_1": ("2080ti", 3, "RTX 2080 Ti"),
    "a6000_2_1": ("a6000", 2, "RTX A6000"),
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


def largest_divisor_at_most(value: int, limit: int) -> int:
    for candidate in range(min(int(value), int(limit)), 0, -1):
        if int(value) % candidate == 0:
            return candidate
    raise AssertionError("one divides every positive integer")


def job(
    cfg: dict[str, Any], *, method: str, scale: str, seed: int,
    representation: str, condition: str, run_mode: str, hardware: str,
) -> dict[str, Any]:
    logical = int(cfg["candidate_logical_batch"][scale])
    environments_per_epoch = int(cfg["candidate_environments_per_epoch"][scale])
    if logical != environments_per_epoch:
        raise ValueError(
            f"logical batch must equal environments per epoch for {scale}: "
            f"{logical} != {environments_per_epoch}"
        )
    cap = int(cfg["physical_batch_caps"][method][scale])
    physical = logical if method == "terran" else largest_divisor_at_most(logical, cap)
    if run_mode == "pilot":
        updates = int(cfg["pilot"]["logical_updates"])
        target_environments = updates * environments_per_epoch
        exposure = target_environments * int(scale.removeprefix("Cus"))
        stream_scope = "pilot"
        validation_views = int(cfg["pilot"]["validation_views"])
        planning_wall_time_hours = None
    else:
        updates = int(cfg["candidate_logical_epochs"][scale])
        target_environments = updates * environments_per_epoch
        exposure = int(cfg["candidate_customer_exposure_budget"][scale])
        derived_exposure = target_environments * int(scale.removeprefix("Cus"))
        if exposure != derived_exposure:
            raise ValueError(
                f"epoch/environment schedule does not match exposure budget for {scale}: "
                f"{derived_exposure} != {exposure}"
            )
        stream_scope = "formal"
        validation_views = int(cfg["formal_candidate"]["validation_views"])
        planning_wall_time_hours = cfg["formal_candidate"].get(
            "planning_wall_time_hours", {}
        ).get(scale)
    stream = (
        f"{ARTIFACTS}/streams/{cfg['runtime_budget_id']}/"
        f"{stream_scope}/{condition}/{scale}/seed_{seed}.parquet"
    )
    payload = {
        "schema": "drl_rq_job_manifest_v1",
        "protocol_id": "drl_rq_protocol_frozen_v1",
        "runtime_budget_id": cfg["runtime_budget_id"],
        "job_id": f"{run_mode}__{representation}__{condition}__{method}__{scale}__seed{seed}",
        "enabled": True,
        "kind": "pilot" if run_mode == "pilot" else "train",
        "run_mode": run_mode,
        "stage": "g1_g4_g5_g6_runtime_memory" if run_mode == "pilot" else "formal_training",
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
        "customer_exposure_budget": exposure,
        "target_environments": target_environments,
        "exposure_checkpoints": (
            [exposure]
            if run_mode == "pilot"
            else [
                int(exposure * float(fraction))
                for fraction in cfg["formal_candidate"]["exposure_checkpoints_fraction"]
            ]
        ),
        "gpu_hour_checkpoints": (
            [] if run_mode == "pilot" else cfg["formal_candidate"]["gpu_hour_checkpoints"]
        ),
        "training_epochs": updates,
        "logical_environments_per_epoch": environments_per_epoch,
        "planned_optimizer_updates": updates,
        "training_rollout_steps": int(cfg["rollout_steps"][scale]),
        "physical_batch_size": physical,
        "effective_batch_size": logical,
        "validation_views": validation_views,
        "validation_checkpoints": 1,
        "planning_wall_time_hours": planning_wall_time_hours,
        "formal_gate_file": GATE if run_mode == "full" else None,
        "euclidean_manifest": (
            f"{ARTIFACTS}/euclidean/euclidean_calibration_manifest.json"
            if representation == "E" else None
        ),
        "file_hash_validation_performed": False,
    }
    return payload


def assign_round_robin(
    queues: dict[str, list[dict[str, Any]]],
    jobs: list[dict[str, Any]],
    servers: list[str],
) -> None:
    slots = [(server, slot) for server in servers for slot in range(SERVERS[server][1])]
    for index, payload in enumerate(jobs):
        server, slot = slots[index % len(slots)]
        payload["global_slot"] = slot
        payload["queue_position"] = sum(
            row.get("global_slot") == slot for row in queues[server]
        )
        queues[server].append(payload)


def build() -> dict[str, list[dict[str, Any]]]:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    queues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pilot_2080 = [
        job(cfg, method=method, scale=scale, seed=1234, representation="G",
            condition="Full-support", run_mode="pilot", hardware="2080ti")
        for scale in ("Cus50", "Cus100", "Cus500") for method in METHODS
    ]
    pilot_a6000 = [
        job(cfg, method=method, scale="Cus1000", seed=1234, representation="G",
            condition="Full-support", run_mode="pilot", hardware="a6000")
        for method in METHODS
    ] + [
        job(cfg, method=method, scale="Cus100", seed=1234, representation="E",
            condition="Full-support", run_mode="pilot", hardware="a6000")
        for method in METHODS
    ]
    assign_round_robin(queues, pilot_2080, ["2080ti_4_1", "2080ti_4_2", "2080ti_3_1"])
    assign_round_robin(queues, pilot_a6000, ["a6000_2_1"])

    formal_2080 = [
        job(cfg, method=method, scale=scale, seed=seed, representation="G",
            condition="Full-support", run_mode="full", hardware="2080ti")
        for scale in ("Cus50", "Cus100", "Cus500")
        for method in METHODS for seed in cfg["seeds"]
    ]
    formal_2080 += [
        job(cfg, method=method, scale="Cus100", seed=seed, representation="G",
            condition=condition, run_mode="full", hardware="2080ti")
        for condition in ("Random-10%-support", "Coverage-10%-support")
        for method in ("am_evrptw", "terran") for seed in cfg["seeds"]
    ]
    formal_2080 += [
        job(cfg, method=method, scale="Cus100", seed=seed, representation="E",
            condition="Full-support", run_mode="full", hardware="2080ti")
        for method in METHODS for seed in cfg["seeds"]
    ]
    formal_a6000 = [
        job(cfg, method=method, scale="Cus1000", seed=seed, representation="G",
            condition="Full-support", run_mode="full", hardware="a6000")
        for method in METHODS for seed in cfg["seeds"]
    ]
    assign_round_robin(queues, formal_2080, ["2080ti_4_1", "2080ti_4_2", "2080ti_3_1"])
    assign_round_robin(queues, formal_a6000, ["a6000_2_1"])
    return queues


def main() -> None:
    parser = argparse.ArgumentParser(description="Build four frozen RQ training queues.")
    parser.add_argument("--output-root", type=Path, default=SCRIPT_ROOT)
    args = parser.parse_args()
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
            "pilot_jobs": sum(row["run_mode"] == "pilot" for row in rows),
            "formal_jobs": sum(row["run_mode"] == "full" for row in rows),
            "formal_launch_allowed": False,
        }
        (destination / "assignment_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({server: len(rows) for server, rows in queues.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
