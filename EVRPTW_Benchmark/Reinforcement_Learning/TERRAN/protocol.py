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
    parse_float_checkpoints,
    parse_int_checkpoints,
    require_registered_batches,
    require_training_rollout_steps,
)
from ..common.data_pass import DataPassState
from ..common.training_stream import read_stream_view_ids


def configure_protocol(args: Any, overrides: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int] | None]:
    if getattr(args, "training_epochs", None) is None and args.data_passes is None:
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
    pool = Stage2TaskPool(
        dataset_path=args.stage2_dataset_path,
        family_root=args.stage2_family_root,
        scale=args.stage2_scale,
        split_ids=args.stage2_split_ids or "train",
        track_ids=args.stage2_track_ids or "train",
        seed=args.seed,
        representation=getattr(args, "training_representation", "G"),
        euclidean_manifest=getattr(args, "euclidean_manifest", None),
    )
    fixed_epochs = getattr(args, "training_epochs", None) is not None
    stream_path = getattr(args, "training_stream_path", None)
    if fixed_epochs:
        if args.data_passes is not None or args.max_batches_per_pass is not None:
            raise ValueError("fixed TERRAN epochs cannot be combined with data-pass options")
        epochs = int(args.training_epochs)
        if epochs <= 0:
            raise ValueError("--training-epochs must be positive")
        if stream_path is None and epochs * physical > len(pool):
            raise ValueError("fixed training budget exceeds the no-replacement training pool")
        if stream_path is not None:
            expected_instances = epochs * effective
            if len(read_stream_view_ids(stream_path)) != expected_instances:
                raise ValueError("TERRAN training stream length does not match its budget")
            expected_exposures = expected_instances * int(
                str(args.stage2_scale).removeprefix("Cus")
            )
            if (
                args.customer_exposure_budget is None
                or int(args.customer_exposure_budget) != expected_exposures
            ):
                raise ValueError("TERRAN explicit stream requires an exact exposure budget")
        validation_every_epochs = int(
            getattr(args, "validation_every_epochs", None) or epochs
        )
        if validation_every_epochs <= 0:
            raise ValueError("--validation-every-epochs must be positive")
        expected_validation_checkpoints = (
            epochs + validation_every_epochs - 1
        ) // validation_every_epochs
        if int(getattr(args, "validation_checkpoints", 1)) != expected_validation_checkpoints:
            raise ValueError(
                "fixed-epoch validation checkpoint count does not match "
                "--validation-every-epochs"
            )
        epochs_per_pass = epochs
        total_passes = 1
    else:
        if len(pool) % physical:
            raise ValueError(f"TERRAN pass size {len(pool)} is not divisible by {physical}")
        epochs_per_pass = len(pool) // physical
        total_passes = int(args.data_passes)
        epochs = total_passes * epochs_per_pass
        if args.max_batches_per_pass is not None:
            if not args.pilot_mode:
                raise ValueError("partial passes are pilot-only")
            epochs = int(args.max_batches_per_pass)
    configured = dict(overrides)
    configured.setdefault("training", {})
    configured.setdefault("data", {})
    configured.setdefault("evaluation", {})
    configured["data"]["stage2_completed_data_passes"] = completed
    configured["training"].update(
        {
            "epochs": epochs,
            "num_envs_per_gpu": physical,
            "rollout_steps": training_rollout_steps,
            "logical_microbatches_per_epoch": effective // physical,
            "checkpoint_interval": (
                validation_every_epochs
                if fixed_epochs
                else max(1, epochs_per_pass * int(args.validation_every_passes))
            ),
        }
    )
    if fixed_epochs and getattr(args, "validation_dataset_path", None) is not None:
        configured["evaluation"].update(
            {
                "eval_interval": validation_every_epochs,
                "eval_path": str(args.validation_dataset_path),
                "eval_family_root": (
                    str(args.validation_family_root)
                    if args.validation_family_root is not None
                    else None
                ),
                "eval_scale": str(args.stage2_scale),
                "eval_split_ids": "val",
                "eval_track_ids": "validation",
                "eval_representation": str(
                    getattr(args, "training_representation", "G")
                ),
                "eval_euclidean_manifest": (
                    str(args.euclidean_manifest)
                    if getattr(args, "euclidean_manifest", None) is not None
                    else None
                ),
                "eval_limit": int(args.validation_limit),
                "eval_n_traj": 1,
                "eval_batch_size": 1,
                "eval_decode_mode": "greedy",
                "eval_info_level": "full",
                "eval_save_routes": False,
                "eval_require_independent_verifier": True,
            }
        )
    configured["output_dir"] = str(Path(args.output_dir).resolve())
    configured["protocol"] = {
        "protocol_id": args.protocol_id,
        "budget_mode": (
            "fixed_customer_exposure" if stream_path is not None else
            ("fixed_logical_epochs" if fixed_epochs else "complete_data_passes")
        ),
        "training_epochs": epochs if fixed_epochs else None,
        "logical_environments_per_epoch": effective if fixed_epochs else None,
        "data_passes": total_passes,
        "views_per_pass": len(pool),
        "epochs_per_pass": epochs_per_pass,
        "physical_batch_size": physical,
        "effective_batch_size": effective,
        "training_rollout_steps": training_rollout_steps,
        "validation_every_epochs": (
            validation_every_epochs if fixed_epochs else None
        ),
        "validation_checkpoints": int(getattr(args, "validation_checkpoints", 1)),
        "pilot_partial": bool(getattr(args, "pilot_mode", False)),
        "completed_data_passes": completed,
        "environment_transitions": environment_transitions,
        "optimizer_steps": optimizer_steps,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "training_stream_path": str(stream_path) if stream_path is not None else None,
        "exposure_checkpoints": list(parse_int_checkpoints(getattr(args, "exposure_checkpoints", ""))),
        "gpu_hour_checkpoints": list(parse_float_checkpoints(getattr(args, "gpu_hour_checkpoints", ""))),
    }
    return configured, {
        "views_per_pass": len(pool),
        "epochs_per_pass": epochs_per_pass,
        "physical_batch_size": physical,
        "effective_batch_size": effective,
    }


def _validation_summary(
    path: Path, data_pass: int, logical_epoch: int | None = None
) -> dict[str, Any]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    passed = [row for row in rows if row["verifier_passed"].lower() == "true"]
    return {
        "schema": "drl_validation_summary_v1",
        "split": "validation",
        "data_pass": int(data_pass),
        "logical_epoch": logical_epoch,
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
    fixed_epochs = getattr(args, "training_epochs", None) is not None
    total_passes = 1 if fixed_epochs else int(args.data_passes)
    history_path = output / "validation_history.jsonl"
    if fixed_epochs:
        # Fixed-epoch jobs validate online every N epochs. Reuse that committed
        # evidence here instead of evaluating every epoch snapshot a second time.
        best_checkpoint = output / "best.ckpt"
        summary_path = output / "validation_summary.json"
        if not best_checkpoint.is_file() or not summary_path.is_file():
            raise RuntimeError(
                "TERRAN fixed-epoch training ended without an online validation selection"
            )
        selected_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        shutil.copy2(best_checkpoint, output / "checkpoint_selected.pt")
    else:
        checkpoints = sorted(final_checkpoint.parent.glob("checkpoint_epoch_*.pt"))
        if final_checkpoint not in checkpoints:
            checkpoints.append(final_checkpoint)
        records: list[tuple[tuple[float, float], Path, dict[str, Any]]] = []
        for checkpoint in checkpoints:
            if checkpoint.name.startswith("checkpoint_epoch_"):
                epoch = int(checkpoint.stem.rsplit("_", 1)[1])
            else:
                epoch = total_passes * meta["epochs_per_pass"]
            data_pass = max(1, min(total_passes, epoch // meta["epochs_per_pass"]))
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
            command.extend(["--representation", str(args.training_representation)])
            if args.euclidean_manifest:
                command.extend(["--euclidean-manifest", str(args.euclidean_manifest)])
            subprocess.run(command, check=True)
            summary = _validation_summary(
                validation_dir / "summary.csv", data_pass
            )
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
        shutil.copy2(selected, output / "best.ckpt")
    shutil.copy2(final_checkpoint, output / "checkpoint_latest.pt")
    atomic_json(output / "validation_summary.json", selected_summary)
    final_validation_limit = int(
        getattr(args, "final_validation_limit", 0) or 0
    )
    final_validation_path = output / "validation_final_audit.json"
    if fixed_epochs and final_validation_limit > 0:
        validation_dir = output / "validation" / "final_audit"
        command = [
            sys.executable,
            "-m",
            "EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.eval_stage2",
            "--dataset-path",
            str(args.validation_dataset_path),
            "--checkpoint",
            str(output / "checkpoint_selected.pt"),
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
            str(final_validation_limit),
            "--seed",
            str(args.seed + 999_000_000),
            "--device",
            str(args.device or "cuda"),
            "--output-dir",
            str(validation_dir),
            "--representation",
            str(getattr(args, "training_representation", "G")),
        ]
        if args.validation_family_root:
            command.extend(["--family-root", str(args.validation_family_root)])
        if getattr(args, "euclidean_manifest", None):
            command.extend(["--euclidean-manifest", str(args.euclidean_manifest)])
        subprocess.run(command, check=True)
        final_validation = _validation_summary(
            validation_dir / "summary.csv",
            data_pass=1,
            logical_epoch=selected_summary.get("logical_epoch"),
        )
        if int(final_validation["instances"]) != final_validation_limit:
            raise RuntimeError(
                "final validation audit did not consume the registered view count: "
                f"{final_validation['instances']} != {final_validation_limit}"
            )
        final_validation.update(
            {
                "schema": "drl_final_validation_audit_v1",
                "selection_checkpoint": str(output / "best.ckpt"),
                "selection_logical_epoch": selected_summary.get("logical_epoch"),
                "selection_changed": False,
            }
        )
        atomic_json(final_validation_path, final_validation)
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
        int(args.training_epochs) * meta["effective_batch_size"]
        if fixed_epochs
        else (
            int(args.max_batches_per_pass) * meta["physical_batch_size"]
            if args.max_batches_per_pass is not None
            else int(args.data_passes) * meta["views_per_pass"]
        )
    )
    if samples_seen != expected:
        raise RuntimeError(f"TERRAN exposure mismatch: {samples_seen} != {expected}")
    atomic_json(
        output / "training_result.json",
        {
            "schema": "drl_training_result_v1",
            "status": "pilot_partial" if getattr(args, "pilot_mode", False) else "passed",
            "method": "TERRAN",
            "protocol_id": args.protocol_id,
            "budget_mode": (
                "fixed_customer_exposure" if getattr(args, "training_stream_path", None) is not None else
                ("fixed_logical_epochs" if fixed_epochs else "complete_data_passes")
            ),
            "requested_training_epochs": int(args.training_epochs) if fixed_epochs else None,
            "completed_training_epochs": int(args.training_epochs) if fixed_epochs else None,
            "logical_environments_per_epoch": meta["effective_batch_size"] if fixed_epochs else None,
            "requested_data_passes": int(args.data_passes) if args.data_passes is not None else None,
            "completed_data_passes": 1 if fixed_epochs else (0 if args.max_batches_per_pass is not None else int(args.data_passes)),
            "training_rollout_steps": int(args.training_rollout_steps),
            "instances_seen": samples_seen,
            "customer_exposures": samples_seen * int(str(args.stage2_scale).removeprefix("Cus")),
            "environment_transitions": environment_transitions,
            "optimizer_steps": optimizer_steps,
            "training_stream_path": str(args.training_stream_path) if args.training_stream_path else None,
            "saved_exposure_checkpoint_files": [
                str(path)
                for path in sorted(final_checkpoint.parent.glob("checkpoint_customer_exposure_*.pt"))
            ],
            "saved_gpu_hour_checkpoint_files": [
                str(path)
                for path in sorted(final_checkpoint.parent.glob("checkpoint_gpu_hours_*.pt"))
            ],
            "selected_checkpoint": str(output / "checkpoint_selected.pt"),
            "best_checkpoint": str(output / "best.ckpt"),
            "validation_every_epochs": (
                int(
                    getattr(args, "validation_every_epochs", None)
                    or args.training_epochs
                )
                if fixed_epochs
                else None
            ),
            "validation_checkpoints": int(getattr(args, "validation_checkpoints", 1)),
            "final_validation_limit": final_validation_limit,
            "final_validation_audit": (
                str(final_validation_path) if final_validation_limit > 0 else None
            ),
            "peak_gpu_memory_bytes": peak_gpu,
            "completed_at": time.time(),
            "wall_time_s": wall_time_s,
        },
    )


__all__ = ["configure_protocol", "finalize_protocol"]
