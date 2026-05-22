from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from evrptw_core.io import load_instance, save_solution
from evrptw_core.schema import EVRPTWSolution
from evrptw_core.validation import validate_instance_structure
from gurobi_solver import GurobiEVRPTWSolver, GurobiSolverConfig


DEFAULT_EXACT_TIME_LIMIT_S = 7200.0


def iter_instance_files(dataset_path: Path) -> list[Path]:
    if dataset_path.is_file():
        return [dataset_path]
    return sorted((dataset_path / "instances").glob("Cus_*_CS_*/*.pkl"))


def parse_checkpoints(raw: str) -> tuple[float, ...]:
    if not raw.strip():
        return tuple()
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0:
            raise ValueError(f"Checkpoint must be non-negative, got {value}")
        values.append(value)
    return tuple(sorted(set(values)))


def resolve_time_schedule(
    requested_checkpoints_s: tuple[float, ...],
    requested_time_limit_s: float | None,
) -> tuple[tuple[float, ...], float]:
    if not requested_checkpoints_s:
        time_limit_s = float(requested_time_limit_s) if requested_time_limit_s is not None else DEFAULT_EXACT_TIME_LIMIT_S
        return (time_limit_s,), time_limit_s

    last_checkpoint_s = max(requested_checkpoints_s)
    if requested_time_limit_s is None:
        time_limit_s = last_checkpoint_s
    else:
        time_limit_s = max(float(requested_time_limit_s), last_checkpoint_s)
    return requested_checkpoints_s, time_limit_s


def checkpoint_label(checkpoint_s: float | int | None) -> str:
    if checkpoint_s is None:
        return "final"
    value = float(checkpoint_s)
    if value.is_integer():
        return f"{int(value)}s"
    return f"{value:g}s".replace(".", "p")


def write_checkpoint_solution(
    instance_id: str,
    snapshot: dict[str, Any],
    solver_name: str,
    checkpoint_dir: Path,
) -> str:
    if not snapshot.get("has_incumbent"):
        return ""
    label = checkpoint_label(snapshot.get("checkpoint_s"))
    solution = EVRPTWSolution(
        instance_id=instance_id,
        solver_name=solver_name,
        routes=snapshot.get("routes", []),
        objective_distance_km=snapshot.get("objective_distance_km"),
        vehicle_count=snapshot.get("vehicle_count"),
        runtime_s=snapshot.get("elapsed_s"),
        feasible=True,
        metadata={
            "checkpoint_s": snapshot.get("checkpoint_s"),
            "reached_checkpoint": snapshot.get("reached_checkpoint"),
            "best_bound": snapshot.get("best_bound"),
            "mip_gap": snapshot.get("mip_gap"),
            "solver_status": snapshot.get("solver_status"),
            "source": snapshot.get("source"),
        },
    )
    path = checkpoint_dir / f"{instance_id}_{label}_solution.pkl"
    save_solution(path, solution)
    return str(path)


def append_time_rows(
    rows: list[dict[str, Any]],
    instance_file: Path,
    instance_id: str,
    solution: EVRPTWSolution,
    checkpoint_dir: Path,
) -> None:
    first_feasible_time_s = solution.metadata.get("first_feasible_time_s")
    for snapshot in solution.metadata.get("checkpoint_snapshots", []):
        checkpoint_solution_path = write_checkpoint_solution(
            instance_id=instance_id,
            snapshot=snapshot,
            solver_name=solution.solver_name,
            checkpoint_dir=checkpoint_dir,
        )
        rows.append({
            "instance_id": instance_id,
            "file": str(instance_file),
            "checkpoint_s": snapshot.get("checkpoint_s"),
            "elapsed_s": snapshot.get("elapsed_s"),
            "reached_checkpoint": snapshot.get("reached_checkpoint"),
            "status": snapshot.get("solver_status"),
            "has_incumbent": snapshot.get("has_incumbent"),
            "first_feasible_time_s": first_feasible_time_s,
            "objective_distance_km": snapshot.get("objective_distance_km"),
            "best_bound": snapshot.get("best_bound"),
            "mip_gap": snapshot.get("mip_gap"),
            "vehicle_count": snapshot.get("vehicle_count"),
            "routes_json": json.dumps(snapshot.get("routes", [])),
            "route_sequence_json": json.dumps(snapshot.get("route_sequence", [])),
            "checkpoint_solution_path": checkpoint_solution_path,
            "source": snapshot.get("source"),
            "errors": "",
        })


def append_error_time_rows(
    rows: list[dict[str, Any]],
    checkpoints_s: tuple[float, ...],
    instance_file: Path,
    instance_id: str,
    status: str,
    error: str,
) -> None:
    for checkpoint_s in checkpoints_s:
        rows.append({
            "instance_id": instance_id,
            "file": str(instance_file),
            "checkpoint_s": checkpoint_s,
            "elapsed_s": "",
            "reached_checkpoint": False,
            "status": status,
            "has_incumbent": False,
            "first_feasible_time_s": "",
            "objective_distance_km": "",
            "best_bound": "",
            "mip_gap": "",
            "vehicle_count": "",
            "routes_json": "[]",
            "route_sequence_json": "[]",
            "checkpoint_solution_path": "",
            "source": "error",
            "errors": error,
        })


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the exact Gurobi EVRP-TW-D solver on pickle instances.")
    parser.add_argument("--dataset_path", required=True, help="Dataset root or one instance pickle file.")
    parser.add_argument("--save_path", required=True, help="Directory for benchmark summaries and solution pickles.")
    parser.add_argument("--time_limit_s", type=float, default=None, help="Max solve time in seconds. Defaults to max checkpoint, or 7200 when checkpoints are empty.")
    parser.add_argument("--mip_gap", type=float, default=0.0)
    parser.add_argument("--cs_copies", type=int, default=3, help="Number of dummy copies per active charging station. Default: 3.")
    parser.add_argument("--output_flag", type=int, default=0)
    parser.add_argument("--checkpoints_s", default="", help="Comma-separated seconds for incumbent snapshots, e.g. 60,300,900.")
    parser.add_argument("--save_traceback", action="store_true", help="Store Python tracebacks in the summary CSV.")
    parser.add_argument("--verbose", action="store_true", help="Print per-instance progress.")
    args = parser.parse_args()

    requested_checkpoints_s = parse_checkpoints(args.checkpoints_s)
    checkpoints_s, time_limit_s = resolve_time_schedule(requested_checkpoints_s, args.time_limit_s)
    print(f"Exact benchmark schedule: time_limit_s={time_limit_s:g}, checkpoints_s={list(checkpoints_s)}, cs_copies={args.cs_copies}")

    dataset_path = Path(args.dataset_path)
    save_path = Path(args.save_path)
    solutions_dir = save_path / "solutions"
    checkpoint_dir = solutions_dir / "checkpoints"
    solutions_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    solver_config = GurobiSolverConfig(
        time_limit_s=time_limit_s,
        mip_gap=args.mip_gap,
        cs_copies=args.cs_copies,
        output_flag=args.output_flag,
        checkpoints_s=checkpoints_s,
    )

    summary_rows = []
    time_rows = []
    instance_files = iter_instance_files(dataset_path)
    if not instance_files:
        print(f"No instance pickle files found under: {dataset_path}")

    for instance_file in instance_files:
        instance_id = instance_file.stem
        try:
            instance = load_instance(instance_file)
            instance_id = instance.instance_id
            validation = validate_instance_structure(instance)
            if not validation.success:
                errors = json.dumps(validation.errors)
                summary_rows.append({
                    "instance_id": instance.instance_id,
                    "file": str(instance_file),
                    "status": "INVALID_INSTANCE",
                    "status_name": "INVALID_INSTANCE",
                    "feasible": False,
                    "objective_distance_km": "",
                    "vehicle_count": "",
                    "runtime_s": "",
                    "first_feasible_time_s": "",
                    "mip_gap": "",
                    "best_bound": "",
                    "routes_json": "",
                    "route_sequence_json": "",
                    "solution_path": "",
                    "time_trace_path": "",
                    "errors": errors,
                    "traceback": "",
                })
                append_error_time_rows(time_rows, checkpoints_s, instance_file, instance_id, "INVALID_INSTANCE", errors)
                continue

            solver = GurobiEVRPTWSolver(solver_config)
            solution = solver.solve(instance)
            solution_path = solutions_dir / f"{instance.instance_id}_solution.pkl"
            save_solution(solution_path, solution)
            append_time_rows(time_rows, instance_file, instance.instance_id, solution, checkpoint_dir)

            summary_rows.append({
                "instance_id": instance.instance_id,
                "file": str(instance_file),
                "status": solution.metadata.get("gurobi_status"),
                "status_name": solution.metadata.get("gurobi_status_name"),
                "feasible": solution.feasible,
                "objective_distance_km": solution.objective_distance_km,
                "vehicle_count": solution.vehicle_count,
                "runtime_s": solution.runtime_s,
                "first_feasible_time_s": solution.metadata.get("first_feasible_time_s"),
                "mip_gap": solution.metadata.get("mip_gap"),
                "best_bound": solution.metadata.get("best_bound"),
                "routes_json": json.dumps(solution.routes),
                "route_sequence_json": json.dumps([node for route in solution.routes for node in route]),
                "solution_path": str(solution_path),
                "time_trace_path": str(save_path / "gurobi_time_trace.csv"),
                "errors": json.dumps(solution.violations),
                "traceback": "",
            })
            if args.verbose:
                print(
                    f"{instance.instance_id}: feasible={solution.feasible} "
                    f"obj={solution.objective_distance_km} vehicles={solution.vehicle_count} "
                    f"first_feasible={solution.metadata.get('first_feasible_time_s')}"
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            summary_rows.append({
                "instance_id": instance_id,
                "file": str(instance_file),
                "status": "ERROR",
                "status_name": "ERROR",
                "feasible": False,
                "objective_distance_km": "",
                "vehicle_count": "",
                "runtime_s": "",
                "first_feasible_time_s": "",
                "mip_gap": "",
                "best_bound": "",
                "routes_json": "",
                "route_sequence_json": "",
                "solution_path": "",
                "time_trace_path": str(save_path / "gurobi_time_trace.csv"),
                "errors": error,
                "traceback": traceback.format_exc() if args.save_traceback else "",
            })
            append_error_time_rows(time_rows, checkpoints_s, instance_file, instance_id, "ERROR", error)
            print(f"{instance_id}: ERROR {type(exc).__name__}: {exc}")

    summary_path = save_path / "gurobi_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "instance_id", "file", "status", "status_name", "feasible", "objective_distance_km",
            "vehicle_count", "runtime_s", "first_feasible_time_s", "mip_gap", "best_bound",
            "routes_json", "route_sequence_json", "solution_path", "time_trace_path", "errors", "traceback",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    trace_path = save_path / "gurobi_time_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "instance_id", "file", "checkpoint_s", "elapsed_s", "reached_checkpoint", "status",
            "has_incumbent", "first_feasible_time_s", "objective_distance_km", "best_bound", "mip_gap",
            "vehicle_count", "routes_json", "route_sequence_json", "checkpoint_solution_path", "source", "errors",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in time_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"Saved summary: {summary_path}")
    print(f"Saved time trace: {trace_path}")


if __name__ == "__main__":
    main()
