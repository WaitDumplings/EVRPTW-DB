from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from evrptw_core.io import load_instance, save_solution
from evrptw_core.schema import EVRPTWSolution
from evrptw_core.validation import validate_instance_structure
from instance_adapter import flatten_routes, to_alns_tensor_instance
from solver import ALNS_Solver


def iter_instance_files(dataset_path: Path) -> list[Path]:
    if dataset_path.is_file():
        return [dataset_path]
    return sorted((dataset_path / "instances").glob("Cus_*_CS_*/*.pkl"))


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def solve_one(task: dict[str, Any]) -> dict[str, Any]:
    instance_file = Path(task["instance_file"])
    seed = int(task["seed"])
    max_iters = task.get("max_iters")
    delta_iters = task.get("delta_iters")
    verbose = bool(task.get("verbose", False))
    save_traceback = bool(task.get("save_traceback", False))

    instance_id = instance_file.stem
    try:
        set_random_seed(seed)
        instance = load_instance(instance_file)
        instance_id = instance.instance_id
        validation = validate_instance_structure(instance)
        if not validation.success:
            return {
                "instance_id": instance_id,
                "file": str(instance_file),
                "status": "INVALID_INSTANCE",
                "feasible": False,
                "errors": json.dumps(validation.errors),
                "traceback": "",
                "summary_row": None,
                "route_rows": [],
                "solution": None,
            }

        alns_instance = to_alns_tensor_instance(instance)
        solver = ALNS_Solver(alns_instance, seed=seed, format="tensor")
        if max_iters is not None:
            solver.max_iters = int(max_iters)

        start = time.perf_counter()
        if verbose:
            solver.solve(delta_iters=delta_iters, resume=False)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                solver.solve(delta_iters=delta_iters, resume=False)
        runtime_s = time.perf_counter() - start

        routes = [list(map(int, route)) for route in solver.best_routes]
        objective = finite_or_none(float(getattr(solver, "global_value", float("inf"))))
        feasible = bool(objective is not None and routes and solver.is_solution_feasible(routes))
        visited_all = bool(all(bool(x) for x in solver.visited))
        unvisited_count = int(len(solver.visited) - sum(bool(x) for x in solver.visited))
        vehicle_count = len(routes) if feasible else None
        route_sequence = flatten_routes(routes)

        solution = EVRPTWSolution(
            instance_id=instance_id,
            solver_name="alns_multi",
            routes=routes,
            objective_distance_km=objective,
            vehicle_count=vehicle_count,
            runtime_s=runtime_s,
            feasible=feasible,
            violations={} if feasible else {"visited_all": visited_all, "unvisited_count": unvisited_count},
            metadata={
                "seed": seed,
                "cur_iter": int(getattr(solver, "cur_iter", -1)),
                "max_iters": int(getattr(solver, "max_iters", -1)),
                "delta_iters": delta_iters,
                "visited_all": visited_all,
                "unvisited_count": unvisited_count,
            },
        )

        summary_row = {
            "instance_id": instance_id,
            "file": str(instance_file),
            "status": "FEASIBLE" if feasible else "INFEASIBLE",
            "feasible": feasible,
            "objective_distance_km": objective if objective is not None else "",
            "vehicle_count": vehicle_count if vehicle_count is not None else "",
            "runtime_s": runtime_s,
            "seed": seed,
            "cur_iter": int(getattr(solver, "cur_iter", -1)),
            "max_iters": int(getattr(solver, "max_iters", -1)),
            "delta_iters": delta_iters if delta_iters is not None else "",
            "visited_all": visited_all,
            "unvisited_count": unvisited_count,
            "routes_json": json.dumps(routes),
            "route_sequence_json": json.dumps(route_sequence),
            "solution_path": "",
            "errors": "",
            "traceback": "",
        }

        route_rows = []
        for route_idx, route in enumerate(routes):
            route_rows.append({
                "instance_id": instance_id,
                "file": str(instance_file),
                "route_idx": route_idx,
                "route_json": json.dumps(route),
                "route_sequence_json": json.dumps(route),
            })

        return {
            "instance_id": instance_id,
            "file": str(instance_file),
            "status": summary_row["status"],
            "feasible": feasible,
            "summary_row": summary_row,
            "route_rows": route_rows,
            "solution": solution.to_dict(),
            "errors": "",
            "traceback": "",
        }
    except Exception as exc:
        return {
            "instance_id": instance_id,
            "file": str(instance_file),
            "status": "ERROR",
            "feasible": False,
            "summary_row": {
                "instance_id": instance_id,
                "file": str(instance_file),
                "status": "ERROR",
                "feasible": False,
                "objective_distance_km": "",
                "vehicle_count": "",
                "runtime_s": "",
                "seed": seed,
                "cur_iter": "",
                "max_iters": "",
                "delta_iters": delta_iters if delta_iters is not None else "",
                "visited_all": False,
                "unvisited_count": "",
                "routes_json": "[]",
                "route_sequence_json": "[]",
                "solution_path": "",
                "errors": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc() if save_traceback else "",
            },
            "route_rows": [],
            "solution": None,
            "errors": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc() if save_traceback else "",
        }


def run_tasks(tasks: list[dict[str, Any]], num_workers: int) -> list[dict[str, Any]]:
    if num_workers <= 1:
        return [solve_one(task) for task in tasks]

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(solve_one, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pickle-native multiprocess ALNS benchmark.")
    parser.add_argument("--dataset_path", required=True, help="Dataset root or one instance pickle file.")
    parser.add_argument("--save_path", required=True, help="Directory for ALNS summaries and solution pickles.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--max_iters", type=int, default=None, help="Override ALNS max iterations. Default uses solver profile.")
    parser.add_argument("--delta_iters", type=int, default=None, help="Run only this many ALNS iterations.")
    parser.add_argument("--save_traceback", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    save_path = Path(args.save_path)
    solutions_dir = save_path / "solutions"
    solutions_dir.mkdir(parents=True, exist_ok=True)

    instance_files = iter_instance_files(dataset_path)
    if args.max_instances is not None:
        instance_files = instance_files[: args.max_instances]
    if not instance_files:
        print(f"No instance pickle files found under: {dataset_path}")

    num_workers = max(1, int(args.num_workers))
    print(
        f"ALNS benchmark schedule: instances={len(instance_files)}, num_workers={num_workers}, "
        f"max_iters={args.max_iters}, delta_iters={args.delta_iters}"
    )

    tasks = []
    for idx, instance_file in enumerate(instance_files):
        tasks.append({
            "instance_file": str(instance_file),
            "seed": int(args.seed) + idx,
            "max_iters": args.max_iters,
            "delta_iters": args.delta_iters,
            "verbose": args.verbose,
            "save_traceback": args.save_traceback,
        })

    results = run_tasks(tasks, num_workers=num_workers)
    results.sort(key=lambda item: item.get("instance_id", ""))

    summary_rows = []
    route_rows = []
    for result in results:
        row = result.get("summary_row")
        if row is None:
            continue
        solution_dict = result.get("solution")
        if solution_dict is not None:
            solution = EVRPTWSolution.from_dict(solution_dict)
            solution_path = solutions_dir / f"{solution.instance_id}_solution.pkl"
            save_solution(solution_path, solution)
            row["solution_path"] = str(solution_path)
        summary_rows.append(row)
        route_rows.extend(result.get("route_rows", []))
        print(
            f"{row['instance_id']}: status={row['status']} obj={row['objective_distance_km']} "
            f"vehicles={row['vehicle_count']} runtime={row['runtime_s']}"
        )

    summary_fieldnames = [
        "instance_id", "file", "status", "feasible", "objective_distance_km", "vehicle_count",
        "runtime_s", "seed", "cur_iter", "max_iters", "delta_iters", "visited_all", "unvisited_count",
        "routes_json", "route_sequence_json", "solution_path", "errors", "traceback",
    ]
    route_fieldnames = ["instance_id", "file", "route_idx", "route_json", "route_sequence_json"]

    write_csv(save_path / "alns_summary.csv", summary_rows, summary_fieldnames)
    write_csv(save_path / "alns_routes.csv", route_rows, route_fieldnames)
    print(f"Saved summary: {save_path / 'alns_summary.csv'}")
    print(f"Saved routes: {save_path / 'alns_routes.csv'}")


if __name__ == "__main__":
    main()
