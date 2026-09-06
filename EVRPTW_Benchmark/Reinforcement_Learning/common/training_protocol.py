from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

from .data_pass import DataPassState
from .evaluation import select_min_verified_objective
from .objective import resolve_objective
from .stage2_data import Stage2TaskPool


def add_data_pass_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--objective-config", type=Path, help="Versioned objective JSON; omitted means legacy distance.")
    parser.add_argument(
        "--reward-contract",
        type=Path,
        help="Self-verifying frozen reward scale and terminal-failure contract.",
    )
    parser.add_argument(
        "--method-auxiliary-profile",
        type=Path,
        help=(
            "Self-verifying method-specific auxiliary objective profile; "
            "independent of the shared task reward contract."
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=("adamw",),
        default="adamw",
        help="Training optimizer. Formal DRL runs use AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Decoupled AdamW weight decay.",
    )
    parser.add_argument("--data-passes", type=int)
    parser.add_argument("--training-epochs", type=int)
    parser.add_argument("--training-rollout-steps", type=int)
    parser.add_argument("--training-stream-path", type=Path)
    parser.add_argument(
        "--training-stream-contract-sha256",
        help=(
            "Expected self-verifying logical training-stream contract SHA256. "
            "Required by the frozen formal protocol."
        ),
    )
    parser.add_argument("--customer-exposure-budget", type=int)
    parser.add_argument("--exposure-checkpoints", default="")
    parser.add_argument("--gpu-hour-checkpoints", default="")
    parser.add_argument("--training-representation", choices=("E", "G"), default="G")
    parser.add_argument("--euclidean-manifest", type=Path)
    parser.add_argument("--physical-batch-size", type=int)
    parser.add_argument("--effective-batch-size", type=int)
    parser.add_argument("--validation-dataset-path", type=Path)
    parser.add_argument("--validation-family-root", type=Path)
    parser.add_argument("--validation-limit", type=int, default=500)
    parser.add_argument(
        "--validation-decode-type",
        choices=("greedy", "sampling"),
        default="greedy",
    )
    parser.add_argument("--validation-candidates", type=int, default=1)
    parser.add_argument("--validation-seed", type=int)
    parser.add_argument("--final-validation-limit", type=int, default=0)
    parser.add_argument("--validation-every-passes", type=int, default=5)
    parser.add_argument("--validation-every-epochs", type=int)
    parser.add_argument("--minimum-training-epochs", type=int)
    parser.add_argument("--post-minimum-validation-every-epochs", type=int)
    parser.add_argument("--validation-checkpoints", type=int, default=1)
    parser.add_argument("--early-stop-patience-validations", type=int, default=0)
    parser.add_argument("--early-stop-start-epoch", type=int, default=0)
    parser.add_argument("--protocol-id", default="legacy_cli_defaults")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-batches-per-pass", type=int)
    parser.add_argument("--pilot-mode", action="store_true")


TRAINING_SIGNATURE_SCHEMA = "drl_resolved_training_signature_v1"


def _signature_sha256(payload: dict[str, Any]) -> str:
    canonical = {key: value for key, value in payload.items() if key != "sha256"}
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def resolved_training_signature_digest(payload: dict[str, Any]) -> str:
    """Hash a resolved signature, excluding only its self-hash field."""

    return _signature_sha256(payload)


def resolved_training_signature_from_args(args: Any) -> dict[str, Any]:
    """Freeze trainer-resolved fields that can change scientific conclusions."""

    scale = getattr(args, "scale", None) or getattr(args, "stage2_scale", None)
    trajectories = getattr(args, "samples_per_instance", None)
    if trajectories is None:
        trajectories = getattr(args, "n_traj", None)
    path_fields = (
        "training_stream_path",
        "validation_dataset_path",
        "euclidean_manifest",
    )
    raw_method_fields = getattr(args, "resolved_training_method_fields", None)
    if raw_method_fields is not None and not isinstance(raw_method_fields, dict):
        raise ValueError("resolved training method fields must be a JSON object")
    # JSON round-trip both detaches the snapshot from mutable config objects and
    # rejects tensors, NaN, and other values that cannot be frozen portably.
    method_fields = (
        json.loads(json.dumps(raw_method_fields, sort_keys=True, allow_nan=False))
        if raw_method_fields is not None
        else None
    )
    payload: dict[str, Any] = {
        "schema": TRAINING_SIGNATURE_SCHEMA,
        "protocol_id": str(getattr(args, "protocol_id", "")),
        "seed": (
            int(getattr(args, "seed"))
            if getattr(args, "seed", None) is not None
            else None
        ),
        "scale": str(scale),
        "training_representation": str(
            getattr(args, "training_representation", "G")
        ),
        "training_epochs": getattr(args, "training_epochs", None),
        "minimum_training_epochs": getattr(args, "minimum_training_epochs", None),
        "training_rollout_steps": getattr(args, "training_rollout_steps", None),
        "physical_batch_size": getattr(args, "physical_batch_size", None),
        "effective_batch_size": getattr(args, "effective_batch_size", None),
        "training_trajectory_count": trajectories,
        "customer_exposure_budget": getattr(args, "customer_exposure_budget", None),
        "training_stream_contract_sha256": getattr(
            args, "training_stream_contract_sha256", None
        ),
        "validation_limit": getattr(args, "validation_limit", None),
        "validation_decode_type": getattr(args, "validation_decode_type", None),
        "validation_candidates": getattr(args, "validation_candidates", None),
        "validation_seed": getattr(args, "validation_seed", None),
        "validation_every_epochs": getattr(args, "validation_every_epochs", None),
        "post_minimum_validation_every_epochs": getattr(
            args, "post_minimum_validation_every_epochs", None
        ),
        "validation_checkpoints": getattr(args, "validation_checkpoints", None),
        "early_stop_patience_validations": getattr(
            args, "early_stop_patience_validations", None
        ),
        "early_stop_start_epoch": getattr(args, "early_stop_start_epoch", None),
        "final_validation_limit": getattr(args, "final_validation_limit", None),
        "soft_stage_end_epoch": getattr(args, "soft_stage_end_epoch", None),
        "optimizer": getattr(args, "optimizer", None),
        "weight_decay": getattr(args, "weight_decay", None),
        "reward_contract_sha256": getattr(args, "reward_contract_sha256", None),
        "method_auxiliary_sha256": getattr(args, "method_auxiliary_sha256", None),
        "method_specific": method_fields,
    }
    for field in path_fields:
        value = getattr(args, field, None)
        payload[field] = str(Path(value).resolve()) if value is not None else None
    for field in (
        "training_epochs",
        "minimum_training_epochs",
        "training_rollout_steps",
        "physical_batch_size",
        "effective_batch_size",
        "training_trajectory_count",
        "customer_exposure_budget",
        "validation_limit",
        "validation_candidates",
        "validation_seed",
        "validation_every_epochs",
        "post_minimum_validation_every_epochs",
        "validation_checkpoints",
        "early_stop_patience_validations",
        "early_stop_start_epoch",
        "final_validation_limit",
        "soft_stage_end_epoch",
    ):
        if payload[field] is not None:
            payload[field] = int(payload[field])
    if payload["weight_decay"] is not None:
        payload["weight_decay"] = float(payload["weight_decay"])
    payload["sha256"] = resolved_training_signature_digest(payload)
    return payload


def freeze_resolved_training_signature(args: Any) -> dict[str, Any]:
    signature = resolved_training_signature_from_args(args)
    setattr(args, "resolved_training_signature", signature)
    setattr(args, "resolved_training_signature_sha256", signature["sha256"])
    return signature


def assert_checkpoint_training_signature(payload: dict[str, Any], args: Any) -> None:
    expected = resolved_training_signature_from_args(args)
    saved = payload.get("resolved_training_signature")
    if not isinstance(saved, dict):
        raise ValueError("checkpoint resolved training signature mismatch")
    if saved.get("sha256") != resolved_training_signature_digest(saved):
        raise ValueError("checkpoint resolved training signature SHA256 mismatch")
    saved_args = payload.get("args", {}) or {}
    if not isinstance(saved_args, dict):
        saved_args = vars(saved_args)
    if (
        saved_args.get("resolved_training_signature") != saved
        or saved_args.get("resolved_training_signature_sha256") != saved["sha256"]
    ):
        raise ValueError("checkpoint args resolved training signature mismatch")
    if saved == expected:
        return
    # Historical custom protocols explicitly support extending a deterministic
    # stream prefix.  Keep that narrow compatibility while the frozen formal
    # protocol remains immutable.  Every non-budget field, including the
    # stream digest, validation seed/decoding, and method-specific semantics,
    # must still match byte-for-byte.
    if expected.get("protocol_id") == "drl_rq_protocol_frozen_v1":
        raise ValueError("checkpoint resolved training signature mismatch")
    extension_fields = {
        "training_epochs",
        "customer_exposure_budget",
        "validation_checkpoints",
    }
    saved_without_budget = {
        key: value
        for key, value in saved.items()
        if key not in extension_fields | {"sha256"}
    }
    expected_without_budget = {
        key: value
        for key, value in expected.items()
        if key not in extension_fields | {"sha256"}
    }
    if saved_without_budget != expected_without_budget:
        raise ValueError("checkpoint resolved training signature mismatch")
    for field in extension_fields:
        saved_value = saved.get(field)
        expected_value = expected.get(field)
        if (
            saved_value is None
            or expected_value is None
            or int(expected_value) < int(saved_value)
        ):
            raise ValueError("checkpoint resolved training signature mismatch")


def require_adamw(args: argparse.Namespace) -> float:
    optimizer = str(getattr(args, "optimizer", "adamw")).lower()
    if optimizer != "adamw":
        raise ValueError(f"unsupported optimizer: {optimizer}; expected adamw")
    weight_decay = float(getattr(args, "weight_decay", 0.01))
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("--weight-decay must be finite and non-negative")
    return weight_decay


def build_adamw_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    *,
    learning_rate: float,
    weight_decay: float,
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    learning_rate = float(learning_rate)
    weight_decay = float(weight_decay)
    eps = float(eps)
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("AdamW learning rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("AdamW weight decay must be finite and non-negative")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("AdamW epsilon must be finite and positive")
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        eps=eps,
        weight_decay=weight_decay,
    )


def require_registered_batches(args: argparse.Namespace, legacy_batch: int) -> tuple[int, int]:
    physical = int(args.physical_batch_size or legacy_batch)
    effective = int(args.effective_batch_size or physical)
    if physical <= 0 or effective <= 0:
        raise ValueError("physical/effective batch sizes must be positive")
    if physical > effective:
        raise ValueError("physical batch size cannot exceed effective batch size")
    return physical, effective


def require_training_rollout_steps(args: argparse.Namespace) -> int:
    value = getattr(args, "training_rollout_steps", None)
    if value is None:
        raise ValueError("protocol mode requires --training-rollout-steps")
    value = int(value)
    if value <= 0:
        raise ValueError("training rollout steps must be positive")
    return value


def require_validation_decoding(args: argparse.Namespace) -> tuple[str, int]:
    decode_type = str(getattr(args, "validation_decode_type", "greedy"))
    candidates = int(getattr(args, "validation_candidates", 1))
    if decode_type not in {"greedy", "sampling"}:
        raise ValueError(f"unsupported validation decode type: {decode_type}")
    if candidates <= 0:
        raise ValueError("--validation-candidates must be positive")
    if decode_type == "greedy" and candidates != 1:
        raise ValueError("greedy validation has exactly one candidate")
    return decode_type, candidates


def validation_epochs(
    maximum_epochs: int,
    *,
    initial_interval: int,
    minimum_epochs: int | None = None,
    post_minimum_interval: int | None = None,
) -> tuple[int, ...]:
    """Return the exact fixed-budget validation schedule.

    The minimum-budget checkpoint belongs to the initial phase. A denser
    post-minimum schedule starts strictly after that checkpoint.
    """
    maximum = int(maximum_epochs)
    initial = int(initial_interval)
    minimum = maximum if minimum_epochs is None else int(minimum_epochs)
    tail = initial if post_minimum_interval is None else int(post_minimum_interval)
    if maximum <= 0 or initial <= 0 or tail <= 0:
        raise ValueError("epoch limits and validation intervals must be positive")
    if not 0 < minimum <= maximum:
        raise ValueError("minimum training epochs must be in [1, maximum epochs]")
    scheduled = set(range(initial, minimum + 1, initial))
    scheduled.add(minimum)
    if minimum < maximum:
        scheduled.update(range(minimum + tail, maximum + 1, tail))
        scheduled.add(maximum)
    return tuple(sorted(scheduled))


def parse_int_checkpoints(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    parsed = tuple(sorted({int(item) for item in str(value).split(",") if item.strip()}))
    if any(item <= 0 for item in parsed):
        raise ValueError("exposure checkpoints must be positive integers")
    return parsed


def parse_float_checkpoints(value: str | None) -> tuple[float, ...]:
    if not value:
        return ()
    parsed = tuple(sorted({float(item) for item in str(value).split(",") if item.strip()}))
    if any(not math.isfinite(item) or item <= 0.0 for item in parsed):
        raise ValueError("GPU-hour checkpoints must be finite and positive")
    return parsed


def grouped_batches(
    batches: Iterable[list[Any]],
    *,
    effective_batch_size: int,
    max_batches: int | None = None,
) -> Iterable[list[list[Any]]]:
    """Group and, when needed, split microbatches at logical boundaries."""

    if int(effective_batch_size) <= 0:
        raise ValueError("effective_batch_size must be positive")
    group: list[list[Any]] = []
    count = 0
    seen_batches = 0
    for incoming in batches:
        if max_batches is not None and seen_batches >= int(max_batches):
            break
        seen_batches += 1
        offset = 0
        while offset < len(incoming):
            room = int(effective_batch_size) - count
            batch = incoming[offset : offset + room]
            offset += len(batch)
            if not batch:
                break
            group.append(batch)
            count += len(batch)
            if count == int(effective_batch_size):
                yield group
                group = []
                count = 0
    if group:
        yield group


def make_validation_pool(
    args: argparse.Namespace,
    *,
    scale: str,
    seed: int,
) -> Stage2TaskPool | None:
    if args.validation_dataset_path is None:
        return None
    return Stage2TaskPool(
        dataset_path=args.validation_dataset_path,
        family_root=args.validation_family_root,
        scale=scale,
        split_ids="val",
        track_ids="validation",
        seed=int(seed) + 900_000,
        representation=getattr(args, "training_representation", "G"),
        euclidean_manifest=getattr(args, "euclidean_manifest", None),
    )


def verified_validation(
    instances: Iterable[Any],
    solve: Callable[[Any, int], dict[str, Any]],
    *,
    seed: int,
    objective_config=None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    active_objective = resolve_objective(objective_config)
    # Sampling decoders draw from Torch's process-global generators.  Scope
    # those draws to the registered validation seed so validation is
    # repeatable and cannot advance the training RNG stream.  Saving every
    # visible CUDA generator is intentional: ``torch.manual_seed`` seeds all
    # of them, and ``fork_rng`` restores them even when ``solve`` raises.
    cuda_rng_devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    for index, instance in enumerate(instances):
        instance_seed = int(seed) + index
        # Validation is selection-only. Retaining an autograd graph for every
        # sampled trajectory wastes GPU memory and can make best-of-K OOM even
        # though the corresponding training batch fits.
        with torch.random.fork_rng(devices=cuda_rng_devices, enabled=True):
            torch.manual_seed(instance_seed)
            with torch.no_grad():
                info = solve(instance, instance_seed)
        current_objective = resolve_objective(
            objective_config if objective_config is not None else info.get("objective_config")
        )
        if rows and current_objective != active_objective:
            raise ValueError("validation cohort mixes objective configurations")
        active_objective = current_objective
        selected, routes, verification = select_min_verified_objective(instance, info, current_objective)
        rows.append(
            {
                "instance_id": instance.instance_id,
                "selected_traj_idx": selected,
                "environment_success": bool(info["success"][selected]),
                "verifier_passed": bool(verification["passed"]),
                "objective_distance_km": float(verification["objective_distance_km"]),
                "vehicle_count": len(routes),
                **current_objective.fields(
                    verification["objective_distance_km"], verification["vehicles_started"]
                ),
            }
        )
    passed = [row for row in rows if row["verifier_passed"]]
    return {
        "schema": "drl_validation_summary_v1",
        "instances": len(rows),
        "complete_and_feasible": len(passed),
        "complete_and_feasible_rate": len(passed) / max(len(rows), 1),
        "mean_verified_distance_km": (
            float(np.mean([row["objective_distance_km"] for row in passed]))
            if passed
            else None
        ),
        "objective_mode": active_objective.mode,
        "objective_profile_id": active_objective.profile_id,
        "objective_unit": active_objective.unit,
        "objective_config": active_objective.to_dict(),
        "mean_verified_objective": (
            float(np.mean([row["objective_value"] for row in passed])) if passed else None
        ),
        "mean_verified_cost_usd": (
            float(np.mean([row["objective_cost_usd"] for row in passed]))
            if passed and active_objective.is_cost else None
        ),
        "mean_verified_vehicle_count": (
            float(np.mean([row["vehicles_started"] for row in passed])) if passed else None
        ),
        "mean_verified_electricity_cost_usd": (
            float(np.mean([row["electricity_cost_usd"] for row in passed]))
            if passed and active_objective.is_cost else None
        ),
        "mean_verified_vehicle_cost_usd": (
            float(np.mean([row["vehicle_cost_usd"] for row in passed]))
            if passed and active_objective.is_cost else None
        ),
        "verifier_summary_passed": len(rows) > 0 and len(passed) == len(rows),
        "rows": rows,
    }


def validation_key(summary: dict[str, Any]) -> tuple[float, float]:
    rate = float(summary["complete_and_feasible_rate"])
    value = summary.get("mean_verified_objective")
    if value is None and summary.get("objective_mode", "distance") == "distance":
        value = summary.get("mean_verified_distance_km")
    return rate, -math.inf if value is None else -float(value)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_state(output_dir: Path, protocol_id: str, resume: bool) -> DataPassState:
    state_path = output_dir / "data_pass_state.json"
    if state_path.exists() and not resume:
        raise FileExistsError(
            f"existing data-pass state requires --resume: {state_path}"
        )
    return DataPassState.load(state_path, protocol_id=protocol_id)


__all__ = [
    "add_data_pass_arguments",
    "append_jsonl",
    "atomic_json",
    "grouped_batches",
    "load_state",
    "make_validation_pool",
    "parse_float_checkpoints",
    "parse_int_checkpoints",
    "resolved_training_signature_digest",
    "require_registered_batches",
    "require_training_rollout_steps",
    "validation_epochs",
    "validation_key",
    "verified_validation",
]
