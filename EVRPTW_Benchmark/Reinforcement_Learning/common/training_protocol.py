from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .data_pass import DataPassState
from .evaluation import select_min_verified_distance
from .stage2_data import Stage2TaskPool


def add_data_pass_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-passes", type=int)
    parser.add_argument("--physical-batch-size", type=int)
    parser.add_argument("--effective-batch-size", type=int)
    parser.add_argument("--validation-dataset-path", type=Path)
    parser.add_argument("--validation-family-root", type=Path)
    parser.add_argument("--validation-limit", type=int, default=500)
    parser.add_argument("--validation-every-passes", type=int, default=5)
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
    )


def verified_validation(
    instances: Iterable[Any],
    solve: Callable[[Any, int], dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, instance in enumerate(instances):
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
    "require_registered_batches",
    "validation_key",
    "verified_validation",
]
