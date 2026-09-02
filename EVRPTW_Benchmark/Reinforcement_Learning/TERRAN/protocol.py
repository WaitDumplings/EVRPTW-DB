from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..common import Stage2TaskPool
from ..common.training_protocol import (
    atomic_json,
    require_registered_batches,
    require_training_rollout_steps,
)
from ..common.data_pass import DataPassState


def configure_protocol(args: Any, overrides: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int] | None]:
    if args.data_passes is None:
        return overrides, None
    if args.stage2_dataset_path is None or args.output_dir is None:
        raise ValueError("TERRAN protocol mode requires Stage-2 data and --output-dir")
    completed = 0
    environment_transitions = 0
    optimizer_steps = 0
    resume_checkpoint = None
    if args.resume:
        state_path = Path(args.output_dir) / "data_pass_state.json"
        resume_checkpoint = Path(args.output_dir) / "checkpoint_latest.pt"
        if not state_path.is_file() or not resume_checkpoint.is_file():
            raise FileNotFoundError(
                "TERRAN resume requires committed data_pass_state.json and checkpoint_latest.pt"
            )
        state = DataPassState.load(state_path, protocol_id=args.protocol_id)
        completed = int(state.completed_data_passes)
        environment_transitions = int(state.environment_transitions)
        optimizer_steps = int(state.optimizer_steps)
    physical, effective = require_registered_batches(args, args.num_envs_per_gpu or 1)
    training_rollout_steps = require_training_rollout_steps(args)
    if physical != effective:
        raise ValueError("TERRAN v1 registers equal physical/effective batches")
    pool = Stage2TaskPool(
        dataset_path=args.stage2_dataset_path,
        family_root=args.stage2_family_root,
        scale=args.stage2_scale,
        split_ids=args.stage2_split_ids or "train",
        track_ids=args.stage2_track_ids or "train",
        seed=args.seed,
    )
    if len(pool) % physical:
        raise ValueError(f"TERRAN pass size {len(pool)} is not divisible by {physical}")
    epochs_per_pass = len(pool) // physical
    epochs = int(args.data_passes) * epochs_per_pass
    if args.max_batches_per_pass is not None:
        if not args.pilot_mode:
            raise ValueError("partial passes are pilot-only")
        epochs = int(args.max_batches_per_pass)
    configured = dict(overrides)
    configured.setdefault("training", {})
    configured.setdefault("data", {})
    configured["data"]["stage2_completed_data_passes"] = completed
    configured["training"].update(
        {
            "epochs": epochs,
            "num_envs_per_gpu": physical,
            "rollout_steps": training_rollout_steps,
            "checkpoint_interval": max(
                1, epochs_per_pass * int(args.validation_every_passes)
            ),
        }
    )
    configured["output_dir"] = str(Path(args.output_dir).resolve())
    configured["protocol"] = {
        "protocol_id": args.protocol_id,
        "data_passes": int(args.data_passes),
        "views_per_pass": len(pool),
        "epochs_per_pass": epochs_per_pass,
        "physical_batch_size": physical,
        "effective_batch_size": effective,
        "training_rollout_steps": training_rollout_steps,
        "pilot_partial": args.max_batches_per_pass is not None,
        "completed_data_passes": completed,
        "environment_transitions": environment_transitions,
        "optimizer_steps": optimizer_steps,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
    }
    return configured, {
        "views_per_pass": len(pool),
        "epochs_per_pass": epochs_per_pass,
        "physical_batch_size": physical,
        "effective_batch_size": effective,
    }


def _validation_summary(path: Path, data_pass: int) -> dict[str, Any]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    passed = [row for row in rows if row["verifier_passed"].lower() == "true"]
    return {
        "schema": "drl_validation_summary_v1",
        "split": "validation",
        "data_pass": int(data_pass),
        "instances": len(rows),
        "complete_and_feasible": len(passed),
        "complete_and_feasible_rate": len(passed) / max(len(rows), 1),
        "mean_verified_distance_km": (
            float(np.mean([float(row["objective_distance_km"]) for row in passed]))
            if passed
            else None
        ),
        "verifier_summary_passed": len(rows) > 0 and len(passed) == len(rows),
    }


def finalize_protocol(args: Any, final_checkpoint: Path, meta: dict[str, int] | None) -> None:
    if meta is None:
        return
    output = Path(args.output_dir)
    checkpoints = sorted(final_checkpoint.parent.glob("checkpoint_epoch_*.pt"))
    if final_checkpoint not in checkpoints:
        checkpoints.append(final_checkpoint)
    records: list[tuple[tuple[float, float], Path, dict[str, Any]]] = []
    history_path = output / "validation_history.jsonl"
    for checkpoint in checkpoints:
        if checkpoint.name.startswith("checkpoint_epoch_"):
            epoch = int(checkpoint.stem.rsplit("_", 1)[1])
        else:
            epoch = int(args.data_passes) * meta["epochs_per_pass"]
        data_pass = max(1, min(int(args.data_passes), epoch // meta["epochs_per_pass"]))
        validation_dir = output / "validation" / f"pass_{data_pass:03d}"
        command = [
            sys.executable,
            "-m",
            "EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.eval_stage2",
            "--dataset-path",
            str(args.validation_dataset_path),
            "--checkpoint",
            str(checkpoint),
            "--scale",
            str(args.stage2_scale),
            "--split-ids",
            "val",
            "--track-ids",
            "validation",
            "--decode-mode",
            "greedy",
            "--candidates",
            "1",
            "--candidate-chunk-size",
            "1",
            "--limit",
            str(args.validation_limit),
            "--seed",
            str(args.seed + data_pass * 100_000),
            "--device",
            str(args.device or "cuda"),
            "--output-dir",
            str(validation_dir),
        ]
        if args.validation_family_root:
            command.extend(["--family-root", str(args.validation_family_root)])
        subprocess.run(command, check=True)
        summary = _validation_summary(validation_dir / "summary.csv", data_pass)
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, sort_keys=True) + "\n")
        distance = summary["mean_verified_distance_km"]
        key = (
            float(summary["complete_and_feasible_rate"]),
            -float(distance) if distance is not None else -float("inf"),
        )
        records.append((key, checkpoint, summary))
    _, selected, selected_summary = max(records, key=lambda row: row[0])
    shutil.copy2(selected, output / "checkpoint_selected.pt")
    shutil.copy2(final_checkpoint, output / "checkpoint_latest.pt")
    atomic_json(output / "validation_summary.json", selected_summary)
    train_log = output / "logs" / "train_log.csv"
    samples_seen = 0
    environment_transitions = 0
    optimizer_steps = 0
    wall_time_s = 0.0
    peak_gpu = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    if train_log.exists():
        with train_log.open("r", newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            samples_seen = int(float(rows[-1]["samples_seen"]))
            environment_transitions = int(float(rows[-1].get("environment_transitions_total", 0)))
            optimizer_steps = int(float(rows[-1].get("optimizer_steps_total", 0)))
            wall_time_s = sum(float(row["epoch_wall_time_s"]) for row in rows)
    expected = (
        int(args.max_batches_per_pass) * meta["physical_batch_size"]
        if args.max_batches_per_pass is not None
        else int(args.data_passes) * meta["views_per_pass"]
    )
    if samples_seen != expected:
        raise RuntimeError(f"TERRAN exposure mismatch: {samples_seen} != {expected}")
    atomic_json(
        output / "training_result.json",
        {
            "schema": "drl_training_result_v1",
            "status": "pilot_partial" if args.max_batches_per_pass is not None else "passed",
            "method": "TERRAN",
            "protocol_id": args.protocol_id,
            "requested_data_passes": int(args.data_passes),
            "completed_data_passes": 0 if args.max_batches_per_pass is not None else int(args.data_passes),
            "instances_seen": samples_seen,
            "customer_exposures": samples_seen * int(str(args.stage2_scale).removeprefix("Cus")),
            "environment_transitions": environment_transitions,
            "optimizer_steps": optimizer_steps,
            "selected_checkpoint": str(output / "checkpoint_selected.pt"),
            "peak_gpu_memory_bytes": peak_gpu,
            "completed_at": time.time(),
            "wall_time_s": wall_time_s,
        },
    )


__all__ = ["configure_protocol", "finalize_protocol"]
