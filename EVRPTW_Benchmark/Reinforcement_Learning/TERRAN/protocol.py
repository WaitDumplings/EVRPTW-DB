from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..common import Stage2TaskPool
from ..common.training_protocol import (
    atomic_json,
    parse_float_checkpoints,
    parse_int_checkpoints,
    require_registered_batches,
    require_training_rollout_steps,
    require_validation_decoding,
    resolved_training_signature_digest,
    validation_epochs,
    validation_key,
)
from ..common.objective import resolve_objective
from ..common.reward_contract import RewardContract
from ..common.data_pass import DataPassState
from ..common.training_stream import (
    read_stream_view_ids,
    training_stream_contract_digest,
    training_stream_contract_from_args,
)


def _checkpoint_reward_contract_provenance(
    checkpoint: Path,
    *,
    scale: str,
    objective: Any,
) -> dict[str, Any] | None:
    """Read the reward contract actually frozen into a TERRAN checkpoint.

    The launcher manifest is an intent record.  This helper turns the completed
    checkpoint into the post-run source of truth and rejects internally
    inconsistent derived fields before ``training_result.json`` is published.
    """

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError(f"TERRAN checkpoint is missing its frozen config: {checkpoint}")
    checkpoint_objective = resolve_objective(config.get("objective"))
    if checkpoint_objective.to_dict() != resolve_objective(objective).to_dict():
        raise RuntimeError(
            f"TERRAN checkpoint objective does not match the completed run: {checkpoint}"
        )
    snapshot = config.get("reward_contract")
    if snapshot is None:
        return None
    try:
        contract = RewardContract.from_payload(snapshot)
        terms = contract.for_scale(scale, resolve_objective(objective))
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"TERRAN checkpoint contains an invalid reward contract: {checkpoint}"
        ) from error

    expected = {
        ("training", "reward_contract_id"): terms.contract_id,
        ("normalization", "reward_contract_id"): terms.contract_id,
        ("normalization", "reward_contract_sha256"): terms.digest,
        ("normalization", "reward_contract_scale"): terms.scale_label,
        ("normalization", "reward_objective_scale"): terms.objective_scale,
        ("normalization", "failure_base"): terms.failure_base,
        ("normalization", "unserved_coefficient"): terms.unserved_coefficient,
        ("env", "normalize_reward"): True,
        ("env", "reward_objective_scale"): terms.objective_scale,
        ("env", "invalid_action_penalty"): 0.0,
        ("env", "success_bonus"): 0.0,
        ("pbrs", "use_terminal_heuristic"): False,
        ("pbrs", "use_terminal_task_penalty"): True,
        ("pbrs", "success_bonus"): 0.0,
        ("pbrs", "failure_base"): terms.failure_base,
        ("pbrs", "unserved_coefficient"): terms.unserved_coefficient,
    }
    for (section, field), expected_value in expected.items():
        actual = (config.get(section) or {}).get(field)
        if actual != expected_value:
            raise RuntimeError(
                "TERRAN checkpoint reward-contract field mismatch: "
                f"{section}.{field}={actual!r}, expected {expected_value!r}"
            )
    return {
        "reward_contract_id": terms.contract_id,
        "reward_contract_sha256": terms.digest,
        "reward_contract_scale": terms.scale_label,
        "reward_contract_snapshot": contract.to_dict(),
        "reward_objective_scale": terms.objective_scale,
        "reward_failure_base": terms.failure_base,
        "reward_unserved_coefficient": terms.unserved_coefficient,
    }


def _checkpoint_training_stream_provenance(checkpoint: Path) -> dict[str, Any] | None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError(f"TERRAN checkpoint is missing its frozen config: {checkpoint}")
    protocol = config.get("protocol")
    if protocol is None:
        return None
    if not isinstance(protocol, Mapping):
        raise RuntimeError(f"TERRAN checkpoint is missing protocol provenance: {checkpoint}")
    snapshot = protocol.get("training_stream_contract_snapshot")
    digest = protocol.get("training_stream_contract_sha256")
    if snapshot is None and digest is None:
        return None
    if not isinstance(snapshot, Mapping):
        raise RuntimeError(f"TERRAN checkpoint has an invalid stream snapshot: {checkpoint}")
    if snapshot.get("sha256") != training_stream_contract_digest(snapshot):
        raise RuntimeError(f"TERRAN checkpoint stream snapshot is corrupt: {checkpoint}")
    if digest != snapshot["sha256"]:
        raise RuntimeError(f"TERRAN checkpoint stream digest is inconsistent: {checkpoint}")
    return {
        "training_stream_contract_sha256": digest,
        "training_stream_contract_snapshot": dict(snapshot),
    }


def _checkpoint_training_signature_provenance(
    checkpoint: Path,
) -> dict[str, Any] | None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError(f"TERRAN checkpoint is missing its frozen config: {checkpoint}")
    protocol = config.get("protocol")
    if protocol is None:
        return None
    if not isinstance(protocol, Mapping):
        raise RuntimeError(f"TERRAN checkpoint is missing protocol provenance: {checkpoint}")
    signature = protocol.get("resolved_training_signature")
    digest = protocol.get("resolved_training_signature_sha256")
    method_fields = protocol.get("resolved_training_method_fields")
    if signature is None and digest is None and method_fields is None:
        return None
    if not isinstance(signature, Mapping) or not isinstance(digest, str):
        raise RuntimeError(
            f"TERRAN checkpoint has an incomplete resolved training signature: {checkpoint}"
        )
    signature_dict = dict(signature)
    if (
        signature.get("sha256") != resolved_training_signature_digest(signature_dict)
        or digest != signature.get("sha256")
        or method_fields != signature.get("method_specific")
    ):
        raise RuntimeError(
            f"TERRAN checkpoint resolved training signature is inconsistent: {checkpoint}"
        )
    return {
        "resolved_training_signature_sha256": digest,
        "resolved_training_signature": signature_dict,
    }


def configure_protocol(args: Any, overrides: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if getattr(args, "training_epochs", None) is None and args.data_passes is None:
        return overrides, None
    if args.stage2_dataset_path is None or args.output_dir is None:
        raise ValueError("TERRAN protocol mode requires Stage-2 data and --output-dir")
    completed = 0
    completed_samples = 0
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
        completed_samples = int(state.instances_seen)
        environment_transitions = int(state.environment_transitions)
        optimizer_steps = int(state.optimizer_steps)
    physical, effective = require_registered_batches(args, args.num_envs_per_gpu or 1)
    if effective % physical:
        raise ValueError("TERRAN requires an exact physical-batch divisor")
    training_rollout_steps = require_training_rollout_steps(args)
    validation_decode_type, validation_candidates = require_validation_decoding(args)
    validation_seed = int(
        getattr(args, "validation_seed", None)
        if getattr(args, "validation_seed", None) is not None
        else int(args.seed) + 910_000_000
    )
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
    early_stop_patience = int(
        getattr(args, "early_stop_patience_validations", 0) or 0
    )
    if early_stop_patience < 0:
        raise ValueError("--early-stop-patience-validations cannot be negative")
    if early_stop_patience and not fixed_epochs:
        raise ValueError("early stopping is supported only with --training-epochs")
    early_stop_start_epoch = int(
        getattr(args, "early_stop_start_epoch", 0) or 0
    )
    if early_stop_start_epoch < 0:
        raise ValueError("--early-stop-start-epoch cannot be negative")
    if early_stop_start_epoch and not fixed_epochs:
        raise ValueError("delayed early stopping requires --training-epochs")
    stream_path = getattr(args, "training_stream_path", None)
    stream_contract = training_stream_contract_from_args(
        args,
        required=(args.protocol_id == "drl_rq_protocol_frozen_v1"),
    )
    if fixed_epochs:
        if args.data_passes is not None or args.max_batches_per_pass is not None:
            raise ValueError("fixed TERRAN epochs cannot be combined with data-pass options")
        epochs = int(args.training_epochs)
        if epochs <= 0:
            raise ValueError("--training-epochs must be positive")
        if early_stop_start_epoch >= epochs:
            raise ValueError(
                "--early-stop-start-epoch must be smaller than --training-epochs"
            )
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
        minimum_training_epochs = int(
            getattr(args, "minimum_training_epochs", None) or epochs
        )
        post_minimum_validation_every_epochs = int(
            getattr(args, "post_minimum_validation_every_epochs", None)
            or validation_every_epochs
        )
        scheduled_validation_epochs = validation_epochs(
            epochs,
            initial_interval=validation_every_epochs,
            minimum_epochs=minimum_training_epochs,
            post_minimum_interval=post_minimum_validation_every_epochs,
        )
        if int(getattr(args, "validation_checkpoints", 1)) != len(scheduled_validation_epochs):
            raise ValueError(
                "fixed-epoch validation checkpoint count does not match "
                "the configured two-phase validation schedule"
            )
        if early_stop_patience and early_stop_start_epoch < minimum_training_epochs:
            raise ValueError(
                "early stopping cannot start before --minimum-training-epochs"
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
    configured["data"]["stage2_completed_samples"] = completed_samples
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
            "early_stop_patience_validations": early_stop_patience,
            "early_stop_start_epoch": early_stop_start_epoch,
            "minimum_training_epochs": (minimum_training_epochs if fixed_epochs else None),
            "post_minimum_validation_every_epochs": (
                post_minimum_validation_every_epochs if fixed_epochs else None
            ),
            "validation_epochs": (list(scheduled_validation_epochs) if fixed_epochs else []),
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
                "eval_n_traj": validation_candidates,
                "eval_seed": validation_seed,
                "eval_batch_size": 1,
                "eval_decode_mode": (
                    "sample"
                    if validation_decode_type == "sampling"
                    else "greedy"
                ),
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
        "minimum_training_epochs": (
            minimum_training_epochs if fixed_epochs else None
        ),
        "post_minimum_validation_every_epochs": (
            post_minimum_validation_every_epochs if fixed_epochs else None
        ),
        "scheduled_validation_epochs": (
            list(scheduled_validation_epochs) if fixed_epochs else []
        ),
        "early_stop_patience_validations": early_stop_patience,
        "early_stop_start_epoch": early_stop_start_epoch,
        "validation_checkpoints": int(getattr(args, "validation_checkpoints", 1)),
        "validation_decode_type": validation_decode_type,
        "validation_candidates": validation_candidates,
        "validation_seed": validation_seed,
        "pilot_partial": bool(getattr(args, "pilot_mode", False)),
        "completed_data_passes": completed,
        "completed_samples": completed_samples,
        "environment_transitions": environment_transitions,
        "optimizer_steps": optimizer_steps,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "training_stream_path": str(stream_path) if stream_path is not None else None,
        "training_stream_contract_sha256": (
            stream_contract["sha256"] if stream_contract is not None else None
        ),
        "training_stream_contract_snapshot": stream_contract,
        "final_validation_limit": int(
            getattr(args, "final_validation_limit", 0) or 0
        ),
        "exposure_checkpoints": list(parse_int_checkpoints(getattr(args, "exposure_checkpoints", ""))),
        "gpu_hour_checkpoints": list(parse_float_checkpoints(getattr(args, "gpu_hour_checkpoints", ""))),
    }
    return configured, {
        "objective": resolve_objective(configured.get("objective")).to_dict(),
        "views_per_pass": len(pool),
        "epochs_per_pass": epochs_per_pass,
        "physical_batch_size": physical,
        "effective_batch_size": effective,
        "scheduled_validation_epochs": (
            list(scheduled_validation_epochs) if fixed_epochs else []
        ),
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
        "mean_verified_objective": (
            float(np.mean([float(row.get("objective_value") or row["objective_distance_km"]) for row in passed]))
            if passed else None
        ),
        "objective_mode": rows[0].get("objective_mode", "distance") if rows else "distance",
        "objective_unit": rows[0].get("objective_unit", "km") if rows else "km",
        "mean_verified_cost_usd": (
            float(np.mean([float(row["objective_cost_usd"]) for row in passed]))
            if passed and all(row.get("objective_cost_usd") not in (None, "") for row in passed) else None
        ),
        **{
            f"mean_verified_{name}": float(np.mean([float(row[name]) for row in passed]))
            if passed and all(row.get(name) not in (None, "") for row in passed) else None
            for name in ("objective_cost_usd", "electricity_cost_usd", "vehicle_cost_usd", "vehicle_count")
        },
    }


def finalize_protocol(args: Any, final_checkpoint: Path, meta: dict[str, Any] | None) -> None:
    if meta is None:
        return
    validation_decode_type, validation_candidates = require_validation_decoding(args)
    validation_seed = int(
        getattr(args, "validation_seed", None)
        if getattr(args, "validation_seed", None) is not None
        else int(args.seed) + 910_000_000
    )
    output = Path(args.output_dir)
    fixed_epochs = getattr(args, "training_epochs", None) is not None
    total_passes = 1 if fixed_epochs else int(args.data_passes)
    history_path = output / "validation_history.jsonl"
    if fixed_epochs:
        # Fixed-epoch jobs validate online every N epochs. Reuse that committed
        # evidence here instead of evaluating every epoch snapshot a second time.
        # The formal aliases are always republished from the best checkpoint over
        # the complete run; fixed-minimum evidence remains separately named.
        best_overall_checkpoint = output / "best_overall.ckpt"
        best_within_minimum_checkpoint = output / "best_within_5000.ckpt"
        overall_summary_path = output / "validation_summary_overall.json"
        within_minimum_summary_path = (
            output / "validation_summary_within_5000.json"
        )
        if (
            not best_overall_checkpoint.is_file()
            or not best_within_minimum_checkpoint.is_file()
            or not overall_summary_path.is_file()
            or not within_minimum_summary_path.is_file()
        ):
            raise RuntimeError(
                "TERRAN fixed-epoch training ended without an online validation selection"
            )
        selected_summary = json.loads(
            overall_summary_path.read_text(encoding="utf-8")
        )
        shutil.copy2(best_overall_checkpoint, output / "checkpoint_selected.pt")
        shutil.copy2(best_overall_checkpoint, output / "best.ckpt")
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
                "sample" if validation_decode_type == "sampling" else "greedy",
                "--candidates",
                str(validation_candidates),
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
            key = validation_key(summary)
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
            "sample" if validation_decode_type == "sampling" else "greedy",
            "--candidates",
            str(validation_candidates),
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
    early_stop_state_path = output / "early_stop_state.json"
    early_stop_state = (
        json.loads(early_stop_state_path.read_text(encoding="utf-8"))
        if fixed_epochs and early_stop_state_path.is_file()
        else {}
    )
    completed_training_epochs = (
        int(early_stop_state.get("completed_training_epochs", args.training_epochs))
        if fixed_epochs
        else None
    )
    early_stopped = bool(early_stop_state.get("early_stopped", False))
    if train_log.exists():
        with train_log.open("r", newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            samples_seen = int(float(rows[-1]["samples_seen"]))
            environment_transitions = int(float(rows[-1].get("environment_transitions_total", 0)))
            optimizer_steps = int(float(rows[-1].get("optimizer_steps_total", 0)))
            wall_time_s = sum(float(row["epoch_wall_time_s"]) for row in rows)
    expected = (
        int(completed_training_epochs) * meta["effective_batch_size"]
        if fixed_epochs
        else (
            int(args.max_batches_per_pass) * meta["physical_batch_size"]
            if args.max_batches_per_pass is not None
            else int(args.data_passes) * meta["views_per_pass"]
        )
    )
    if samples_seen != expected:
        raise RuntimeError(f"TERRAN exposure mismatch: {samples_seen} != {expected}")
    active_objective = resolve_objective(meta.get("objective"))
    reward_contract_provenance = _checkpoint_reward_contract_provenance(
        final_checkpoint,
        scale=str(args.stage2_scale),
        objective=active_objective,
    )
    selected_reward_contract_provenance = _checkpoint_reward_contract_provenance(
        output / "checkpoint_selected.pt",
        scale=str(args.stage2_scale),
        objective=active_objective,
    )
    if reward_contract_provenance != selected_reward_contract_provenance:
        raise RuntimeError(
            "TERRAN final and selected checkpoints disagree on the reward contract"
        )
    if (
        getattr(args, "reward_contract", None) is not None
        and reward_contract_provenance is None
    ):
        raise RuntimeError(
            "TERRAN was launched with a reward contract, but completed checkpoints "
            "do not contain one"
        )
    reward_contract_fields = reward_contract_provenance or {
        "reward_contract_id": None,
        "reward_contract_sha256": None,
        "reward_contract_scale": None,
        "reward_contract_snapshot": None,
        "reward_objective_scale": None,
        "reward_failure_base": None,
        "reward_unserved_coefficient": None,
    }
    stream_contract_provenance = _checkpoint_training_stream_provenance(
        final_checkpoint
    )
    selected_stream_contract_provenance = _checkpoint_training_stream_provenance(
        output / "checkpoint_selected.pt"
    )
    if stream_contract_provenance != selected_stream_contract_provenance:
        raise RuntimeError(
            "TERRAN final and selected checkpoints disagree on the training stream"
        )
    expected_stream_snapshot = getattr(
        args, "training_stream_contract_snapshot", None
    )
    expected_stream_sha = getattr(args, "training_stream_contract_sha256", None)
    if expected_stream_snapshot is not None and stream_contract_provenance != {
        "training_stream_contract_sha256": expected_stream_sha,
        "training_stream_contract_snapshot": expected_stream_snapshot,
    }:
        raise RuntimeError(
            "TERRAN completed checkpoints do not contain the requested training stream"
        )
    stream_contract_fields = stream_contract_provenance or {
        "training_stream_contract_sha256": None,
        "training_stream_contract_snapshot": None,
    }
    training_signature_provenance = _checkpoint_training_signature_provenance(
        final_checkpoint
    )
    selected_training_signature_provenance = (
        _checkpoint_training_signature_provenance(
            output / "checkpoint_selected.pt"
        )
    )
    if training_signature_provenance != selected_training_signature_provenance:
        raise RuntimeError(
            "TERRAN final and selected checkpoints disagree on the resolved "
            "training signature"
        )
    if (
        args.protocol_id == "drl_rq_protocol_frozen_v1"
        and training_signature_provenance is None
    ):
        raise RuntimeError(
            "formal TERRAN checkpoints are missing the resolved training signature"
        )
    training_signature_fields = training_signature_provenance or {
        "resolved_training_signature_sha256": None,
        "resolved_training_signature": None,
    }
    atomic_json(
        output / "training_result.json",
        {
            "schema": "drl_training_result_v1",
            "status": "pilot_partial" if getattr(args, "pilot_mode", False) else ("early_stopped" if early_stopped else "passed"),
            "method": "TERRAN",
            "protocol_id": args.protocol_id,
            "objective_config": active_objective.to_dict(),
            "objective_mode": active_objective.mode,
            "objective_unit": active_objective.unit,
            **reward_contract_fields,
            **stream_contract_fields,
            **training_signature_fields,
            "budget_mode": (
                "fixed_customer_exposure" if getattr(args, "training_stream_path", None) is not None else
                ("fixed_logical_epochs" if fixed_epochs else "complete_data_passes")
            ),
            "requested_training_epochs": int(args.training_epochs) if fixed_epochs else None,
            "completed_training_epochs": completed_training_epochs,
            "early_stopped": early_stopped,
            "early_stop_epoch": early_stop_state.get("early_stop_epoch"),
            "logical_environments_per_epoch": meta["effective_batch_size"] if fixed_epochs else None,
            "requested_data_passes": int(args.data_passes) if args.data_passes is not None else None,
            "completed_data_passes": (0 if early_stopped else 1) if fixed_epochs else (0 if args.max_batches_per_pass is not None else int(args.data_passes)),
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
            "best_within_5000_checkpoint": str(output / "best_within_5000.ckpt"),
            "best_overall_checkpoint": str(output / "best_overall.ckpt"),
            "minimum_training_epochs": (
                int(getattr(args, "minimum_training_epochs", None) or args.training_epochs)
                if fixed_epochs else None
            ),
            "validation_every_epochs": (
                int(
                    getattr(args, "validation_every_epochs", None)
                    or args.training_epochs
                )
                if fixed_epochs
                else None
            ),
            "post_minimum_validation_every_epochs": (
                int(
                    getattr(args, "post_minimum_validation_every_epochs", None)
                    or getattr(args, "validation_every_epochs", None)
                    or args.training_epochs
                )
                if fixed_epochs else None
            ),
            "scheduled_validation_epochs": list(
                meta.get("scheduled_validation_epochs", [])
            ),
            "validation_checkpoints": int(getattr(args, "validation_checkpoints", 1)),
            "completed_validation_checkpoints": int(early_stop_state.get("completed_validation_checkpoints", 0)),
            "early_stop_patience_validations": int(getattr(args, "early_stop_patience_validations", 0) or 0),
            "early_stop_start_epoch": int(getattr(args, "early_stop_start_epoch", 0) or 0),
            "validation_seed": validation_seed,
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
