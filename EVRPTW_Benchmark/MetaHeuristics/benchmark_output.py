from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from evrptw_core.io import save_solution
from evrptw_core.schema import EVRPTWSolution

from benchmark_common import checkpoint_label


TIME_TRACE_FIELDNAMES = [
    "instance_id",
    "file",
    "family_id",
    "checkpoint_s",
    "elapsed_s",
    "reached_checkpoint",
    "status",
    "benchmark_status",
    "has_incumbent",
    "first_feasible_time_s",
    "incumbent_event_time_s",
    "objective_distance_km",
    "vehicle_count",
    "routes_json",
    "route_sequence_json",
    "checkpoint_solution_path",
    "source",
    "errors",
]


def snapshot_rows(
    instance_id: str,
    source_info: dict[str, str],
    snapshots: list[dict[str, Any]],
    first_feasible_time_s: float | None,
) -> list[dict[str, Any]]:
    return [
        {
            "instance_id": instance_id,
            "file": source_info.get("file", ""),
            "family_id": source_info.get("family_id", ""),
            "checkpoint_s": snapshot["checkpoint_s"],
            "elapsed_s": snapshot["elapsed_s"],
            "reached_checkpoint": snapshot["reached_checkpoint"],
            "status": snapshot["status"],
            "benchmark_status": snapshot["benchmark_status"],
            "has_incumbent": snapshot["has_incumbent"],
            "first_feasible_time_s": "" if first_feasible_time_s is None else first_feasible_time_s,
            "incumbent_event_time_s": (
                "" if snapshot["incumbent_event_time_s"] is None else snapshot["incumbent_event_time_s"]
            ),
            "objective_distance_km": (
                "" if snapshot["objective_distance_km"] is None else snapshot["objective_distance_km"]
            ),
            "vehicle_count": "" if snapshot["vehicle_count"] is None else snapshot["vehicle_count"],
            "routes_json": json.dumps(snapshot["routes"]),
            "route_sequence_json": json.dumps(snapshot["route_sequence"]),
            "checkpoint_solution_path": "",
            "source": snapshot["source"],
            "errors": "",
        }
        for snapshot in snapshots
    ]


def error_snapshot_rows(
    instance_id: str,
    source_info: dict[str, str],
    checkpoints_s: tuple[float, ...],
    status: str,
    errors: str,
) -> list[dict[str, Any]]:
    return [
        {
            "instance_id": instance_id,
            "file": source_info.get("file", ""),
            "family_id": source_info.get("family_id", ""),
            "checkpoint_s": checkpoint,
            "elapsed_s": 0.0,
            "reached_checkpoint": False,
            "status": status,
            "benchmark_status": status,
            "has_incumbent": False,
            "first_feasible_time_s": "",
            "incumbent_event_time_s": "",
            "objective_distance_km": "",
            "vehicle_count": "",
            "routes_json": "[]",
            "route_sequence_json": "[]",
            "checkpoint_solution_path": "",
            "source": "error",
            "errors": errors,
        }
        for checkpoint in checkpoints_s
    ]


def save_result_artifacts(
    result: dict[str, Any],
    *,
    solver_name: str,
    solutions_dir: Path,
    checkpoints_dir: Path,
) -> None:
    solution_dict = result.get("solution")
    if solution_dict is not None:
        solution = EVRPTWSolution.from_dict(solution_dict)
        path = solutions_dir / f"{solution.instance_id}_solution.pkl"
        save_solution(path, solution)
        result["summary_row"]["solution_path"] = str(path)

    for row, snapshot in zip(result.get("time_rows", []), result.get("snapshots", [])):
        if not snapshot.get("has_incumbent"):
            continue
        checkpoint_solution = EVRPTWSolution(
            instance_id=result["summary_row"]["instance_id"],
            solver_name=solver_name,
            routes=[list(route) for route in snapshot["routes"]],
            objective_distance_km=float(snapshot["objective_distance_km"]),
            vehicle_count=int(snapshot["vehicle_count"]),
            runtime_s=float(snapshot["elapsed_s"]),
            feasible=True,
            metadata={
                "checkpoint_s": snapshot["checkpoint_s"],
                "reached_checkpoint": snapshot["reached_checkpoint"],
                "incumbent_event_time_s": snapshot["incumbent_event_time_s"],
                "source": snapshot["source"],
                "benchmark_status": snapshot["benchmark_status"],
            },
        )
        path = checkpoints_dir / (
            f"{checkpoint_solution.instance_id}_{checkpoint_label(snapshot['checkpoint_s'])}_solution.pkl"
        )
        save_solution(path, checkpoint_solution)
        row["checkpoint_solution_path"] = str(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    temporary.replace(path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
