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
METHOD_ORDER = ("am_evrptw", "evrptw_rl", "drl_ts", "terran")
SEEN_SCALES = ("Cus100", "Cus500", "Cus1000")


def load_protocol(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        protocol = yaml.safe_load(stream)
    if protocol.get("objective") != "verified_directed_road_total_distance":
        raise ValueError("canonical verified-distance objective is required")
    if protocol.get("training", {}).get("checkpoint_selection_uses_test_metrics"):
        raise ValueError("test metrics cannot select checkpoints")
    if protocol.get("scale_transfer", {}).get("train_cus2000"):
        raise ValueError("Cus2000 training is forbidden")
    disabled = set(protocol.get("disabled_tracks", []))
    if {"E_to_R", "R_to_Inject_to_R"}.difference(disabled):
        raise ValueError("E->R and R->Inject->R must remain disabled")
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
    passes = (
        protocol["training"]["pilot_data_passes"]
        if run_mode == "pilot"
        else protocol["training"]["full_data_passes"]
    )
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
            "data_passes": int(passes),
            "validation_every_passes": int(
                protocol["training"]["validation_every_passes"]
            ),
            "validation_views": int(
                protocol["pilot"]["validation_limit"]
                if run_mode == "pilot"
                else protocol["training"]["validation_views"]
            ),
            "physical_batch_size": int(method_cfg["physical_batch"][scale]),
            "effective_batch_size": int(method_cfg["effective_batch"][scale]),
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


def _write_or_check(path: Path, content: str, check: bool) -> None:
    if check:
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"manifest is stale: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen DRL job manifests.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    if args.dataset_root:
        validate_dataset(protocol, args.dataset_root)
    jobs2080, jobsa6000 = build_manifests(protocol)
    _write_or_check(
        args.output_dir / "drl_2080ti_jobs_v1.jsonl",
        _serialize(jobs2080),
        args.check,
    )
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
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
