from __future__ import annotations

import argparse
import csv
import json
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
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))
sys.path.insert(0, str(THIS_DIR))

from evrptw_core.io import iter_instance_dicts, load_instance, save_solution
from evrptw_core.schema import EVRPTWInstance, EVRPTWSolution
from evrptw_core.validation import validate_instance_structure
from solver import GreedyEVRPTWSolver, flatten_routes


def iter_instance_files(dataset_path: Path) -> list[Path]:
    if dataset_path.is_file():
        return [dataset_path]
    return sorted((dataset_path / "instances").glob("Cus_*_CS_*/*.pkl"))


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def solve_one(task: dict[str, Any]) -> dict[str, Any]:
    instance_source = str(task.get("instance_file", ""))
    instance_file = Path(instance_source) if instance_source else Path(".")
    instance_payload = task.get("instance_payload")
    seed = int(task["seed"])
    save_traceback = bool(task.get("save_traceback", False))
    customer_order = str(task.get("customer_order", "nearest"))

    instance_id = str(task.get("instance_id", instance_file.stem))
    try:
        set_random_seed(seed)
        instance = EVRPTWInstance.from_dict(instance_payload) if instance_payload is not None else load_instance(instance_file)
        instance_id = instance.instance_id
        validation = validate_instance_structure(instance)
        if not validation.success:
            errors = json.dumps(validation.errors)
            return {
                "summary_row": {
                    "instance_id": instance_id,
                    "file": str(instance_file),
                    "status": "INVALID_INSTANCE",
                    "feasible": False,
                    "objective_distance_km": "",
                    "vehicle_count": "",
                    "runtime_s": "",
                    "seed": seed,
                    "customer_order": customer_order,
                    "visited_all": False,
                    "unvisited_count": "",
                    "failed_vehicle_starts": "",
                    "routes_json": "[]",
                    "route_sequence_json": "[]",
                    "solution_path": "",
                    "errors": errors,
                    "traceback": "",
                },
                "route_rows": [],
                "solution": None,
            }

        solver = GreedyEVRPTWSolver(instance, customer_order=customer_order)
        start = time.perf_counter()
        routes, audit = solver.solve()
        runtime_s = time.perf_counter() - start

        objective = solver.route_distance_km(routes) if routes else None
        feasible = bool(audit["visited_all"] and routes)
        vehicle_count = len(routes) if feasible else None
        route_sequence = flatten_routes(routes)

        solution = EVRPTWSolution(
            instance_id=instance_id,
            solver_name=solver.name,
            routes=routes,
            objective_distance_km=objective,
            vehicle_count=vehicle_count,
            runtime_s=runtime_s,
            feasible=feasible,
            violations={} if feasible else audit,
            metadata={
                "seed": seed,
                "customer_order": customer_order,
                **audit,
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
            "customer_order": customer_order,
            "visited_all": audit["visited_all"],
            "unvisited_count": audit["unvisited_count"],
            "failed_vehicle_starts": audit["failed_vehicle_starts"],
            "routes_json": json.dumps(routes),
            "route_sequence_json": json.dumps(route_sequence),
            "solution_path": "",
            "errors": "" if feasible else json.dumps(audit),
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

        return {"summary_row": summary_row, "route_rows": route_rows, "solution": solution.to_dict()}
    except Exception as exc:
        return {
            "summary_row": {
                "instance_id": instance_id,
                "file": str(instance_file),
                "status": "ERROR",
                "feasible": False,
                "objective_distance_km": "",
                "vehicle_count": "",
                "runtime_s": "",
                "seed": seed,
                "customer_order": customer_order,
                "visited_all": False,
                "unvisited_count": "",
                "failed_vehicle_starts": "",
                "routes_json": "[]",
                "route_sequence_json": "[]",
                "solution_path": "",
                "errors": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc() if save_traceback else "",
            },
            "route_rows": [],
            "solution": None,
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
    parser = argparse.ArgumentParser(description="Run the pickle-native Greedy EVRP-TW-D benchmark.")
    parser.add_argument("--dataset_path", required=True, help="Dataset root or one instance pickle file.")
    parser.add_argument("--save_path", required=True, help="Directory for Greedy summaries and solution pickles.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--customer_order", choices=["nearest", "earliest_due", "hybrid"], default="nearest")
    parser.add_argument("--save_traceback", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    save_path = Path(args.save_path)
    solutions_dir = save_path / "solutions"
    solutions_dir.mkdir(parents=True, exist_ok=True)

    instance_payloads = list(iter_instance_dicts(dataset_path))
    if args.max_instances is not None:
        instance_payloads = instance_payloads[: args.max_instances]
    if not instance_payloads:
        print(f"No EVRPTW instances found under: {dataset_path}")

    num_workers = max(1, int(args.num_workers))
    print(
        f"Greedy benchmark schedule: instances={len(instance_payloads)}, num_workers={num_workers}, "
        f"customer_order={args.customer_order}"
    )

    tasks = []
    for idx, payload in enumerate(instance_payloads):
        tasks.append({
            "instance_file": str(dataset_path),
            "instance_payload": payload,
            "instance_id": str(payload.get("instance_id", f"instance_{idx:06d}")),
            "seed": int(args.seed) + idx,
            "customer_order": args.customer_order,
            "save_traceback": args.save_traceback,
        })

    results = run_tasks(tasks, num_workers=num_workers)
    results.sort(key=lambda item: item.get("summary_row", {}).get("instance_id", ""))

    summary_rows = []
    route_rows = []
    for result in results:
        row = result["summary_row"]
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
        "runtime_s", "seed", "customer_order", "visited_all", "unvisited_count",
        "failed_vehicle_starts", "routes_json", "route_sequence_json", "solution_path",
        "errors", "traceback",
    ]
    route_fieldnames = ["instance_id", "file", "route_idx", "route_json", "route_sequence_json"]

    write_csv(save_path / "greedy_summary.csv", summary_rows, summary_fieldnames)
    write_csv(save_path / "greedy_routes.csv", route_rows, route_fieldnames)
    print(f"Saved summary: {save_path / 'greedy_summary.csv'}")
    print(f"Saved routes: {save_path / 'greedy_routes.csv'}")


if __name__ == "__main__":
    main()
