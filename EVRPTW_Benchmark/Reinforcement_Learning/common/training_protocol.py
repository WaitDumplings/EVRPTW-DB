from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

from .data_pass import DataPassState
from .evaluation import select_min_verified_distance
from .stage2_data import Stage2TaskPool


def add_data_pass_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-passes", type=int)
    parser.add_argument("--training-epochs", type=int)
    parser.add_argument("--training-rollout-steps", type=int)
    parser.add_argument("--training-stream-path", type=Path)
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


def require_registered_batches(args: argparse.Namespace, legacy_batch: int) -> tuple[int, int]:
    physical = int(args.physical_batch_size or legacy_batch)
    effective = int(args.effective_batch_size or physical)
    if physical <= 0 or effective <= 0:
        raise ValueError("physical/effective batch sizes must be positive")
    if effective % physical:
        raise ValueError("effective batch size must be a multiple of physical batch size")
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
    group: list[list[Any]] = []
    count = 0
    seen_batches = 0
    for batch in batches:
        if max_batches is not None and seen_batches >= int(max_batches):
            break
        seen_batches += 1
        if group and count + len(batch) > effective_batch_size:
            yield group
            group = []
            count = 0
        group.append(batch)
        count += len(batch)
        if count == effective_batch_size:
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
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, instance in enumerate(instances):
        # Validation is selection-only. Retaining an autograd graph for every
        # sampled trajectory wastes GPU memory and can make best-of-K OOM even
        # though the corresponding training batch fits.
        with torch.no_grad():
            info = solve(instance, int(seed) + index)
        selected, routes, verification = select_min_verified_distance(instance, info)
        rows.append(
            {
                "instance_id": instance.instance_id,
                "selected_traj_idx": selected,
                "environment_success": bool(info["success"][selected]),
                "verifier_passed": bool(verification["passed"]),
                "objective_distance_km": float(verification["objective_distance_km"]),
                "vehicle_count": len(routes),
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
        "verifier_summary_passed": len(rows) > 0 and len(passed) == len(rows),
        "rows": rows,
    }


def validation_key(summary: dict[str, Any]) -> tuple[float, float]:
    rate = float(summary["complete_and_feasible_rate"])
    distance = summary.get("mean_verified_distance_km")
    return rate, -math.inf if distance is None else -float(distance)


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
    "require_registered_batches",
    "require_training_rollout_steps",
    "validation_epochs",
    "validation_key",
    "verified_validation",
]
