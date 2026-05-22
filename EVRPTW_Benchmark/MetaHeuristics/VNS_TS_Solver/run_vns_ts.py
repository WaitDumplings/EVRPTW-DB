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
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))
sys.path.insert(0, str(THIS_DIR))

from evrptw_core.io import load_instance, save_solution
from evrptw_core.schema import EVRPTWSolution
from evrptw_core.validation import validate_instance_structure
from solver import VNSTSolver
from vnst_adapter import flatten_routes, route_distance_km, routes_to_terminal_ids, to_vnst_instance


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
    save_traceback = bool(task.get("save_traceback", False))
    verbose = bool(task.get("verbose", False))

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
                "summary_row": {
                    "instance_id": instance_id,
                    "file": str(instance_file),
                    "status": "INVALID_INSTANCE",
                    "feasible": False,
                    "objective_distance_km": "",
                    "vehicle_count": "",
                    "runtime_s": "",
                    "seed": seed,
                    "predefine_route_number": task.get("predefine_route_number", ""),
                    "eta_feas": task.get("eta_feas", ""),
                    "eta_dist": task.get("eta_dist", ""),
                    "tabu_iter": task.get("tabu_iter", ""),
                    "tabu_tenure": task.get("tabu_tenure", ""),
                    "k_max": task.get("k_max", ""),
                    "search_mode": task.get("search_mode", ""),
                    "move_candidate_limit": task.get("move_candidate_limit", ""),
                    "route_neighbor_limit": task.get("route_neighbor_limit", ""),
                    "position_neighbor_limit": task.get("position_neighbor_limit", ""),
                    "exchange_neighbor_limit": task.get("exchange_neighbor_limit", ""),
                    "station_candidate_limit": task.get("station_candidate_limit", ""),
                    "raw_global_value": "",
                    "routes_json": "[]",
                    "route_sequence_json": "[]",
                    "solution_path": "",
                    "errors": json.dumps(validation.errors),
                    "traceback": "",
                },
                "route_rows": [],
                "solution": None,
                "errors": json.dumps(validation.errors),
                "traceback": "",
            }

        vnst_instance = to_vnst_instance(instance)
        solver = VNSTSolver(
            vnst_instance,
            predefine_route_number=int(task.get("predefine_route_number", 3)),
            show_progress=verbose,
            search_mode=str(task.get("search_mode", "fast")),
            move_candidate_limit=int(task.get("move_candidate_limit", 80)),
            route_neighbor_limit=int(task.get("route_neighbor_limit", 4)),
            position_neighbor_limit=int(task.get("position_neighbor_limit", 4)),
            exchange_neighbor_limit=int(task.get("exchange_neighbor_limit", 6)),
            station_candidate_limit=int(task.get("station_candidate_limit", 5)),
        )
        if task.get("eta_feas") is not None:
            solver.η_feas = int(task["eta_feas"])
        if task.get("eta_dist") is not None:
            solver.η_dist = int(task["eta_dist"])
        if task.get("tabu_iter") is not None:
            solver.tabu_iter = int(task["tabu_iter"])
        if task.get("tabu_tenure") is not None:
            solver.tabu_tenure = int(task["tabu_tenure"])
        if task.get("k_max") is not None:
            solver.k_max = int(task["k_max"])

        start = time.perf_counter()
        if verbose:
            solution_obj = solver.solve()
        else:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                solution_obj = solver.solve()
        runtime_s = time.perf_counter() - start

        routes = routes_to_terminal_ids(solution_obj)
        feasible = bool(solution_obj is not None and routes and solver.is_solution_feasible(solution_obj))
        objective = route_distance_km(routes, instance) if feasible else finite_or_none(float(getattr(solver, "global_value", float("inf"))))
        raw_global_value = finite_or_none(float(getattr(solver, "global_value", float("inf"))))
        vehicle_count = len(routes) if feasible else None
        route_sequence = flatten_routes(routes)

        solution = EVRPTWSolution(
            instance_id=instance_id,
            solver_name="vns_tabu_search",
            routes=routes,
            objective_distance_km=objective,
            vehicle_count=vehicle_count,
            runtime_s=runtime_s,
            feasible=feasible,
            violations={} if feasible else {"solver_returned_routes": bool(routes)},
            metadata={
                "seed": seed,
                "predefine_route_number": int(task.get("predefine_route_number", 3)),
                "eta_feas": int(solver.η_feas),
                "eta_dist": int(solver.η_dist),
                "tabu_iter": int(solver.tabu_iter),
                "tabu_tenure": int(solver.tabu_tenure),
                "k_max": int(solver.k_max),
                "search_mode": solver.search_mode,
                "move_candidate_limit": int(solver.move_candidate_limit),
                "route_neighbor_limit": int(solver.route_neighbor_limit),
                "position_neighbor_limit": int(solver.position_neighbor_limit),
                "exchange_neighbor_limit": int(solver.exchange_neighbor_limit),
                "station_candidate_limit": int(solver.station_candidate_limit),
                "raw_global_value": raw_global_value,
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
            "predefine_route_number": int(task.get("predefine_route_number", 3)),
            "eta_feas": int(solver.η_feas),
            "eta_dist": int(solver.η_dist),
            "tabu_iter": int(solver.tabu_iter),
            "tabu_tenure": int(solver.tabu_tenure),
            "k_max": int(solver.k_max),
            "search_mode": solver.search_mode,
            "move_candidate_limit": int(solver.move_candidate_limit),
            "route_neighbor_limit": int(solver.route_neighbor_limit),
            "position_neighbor_limit": int(solver.position_neighbor_limit),
            "exchange_neighbor_limit": int(solver.exchange_neighbor_limit),
            "station_candidate_limit": int(solver.station_candidate_limit),
            "raw_global_value": raw_global_value if raw_global_value is not None else "",
            "routes_json": json.dumps(routes),
            "route_sequence_json": json.dumps(route_sequence),
            "solution_path": "",
            "errors": "" if feasible else json.dumps(solution.violations),
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
                "predefine_route_number": task.get("predefine_route_number", ""),
                "eta_feas": task.get("eta_feas", ""),
                "eta_dist": task.get("eta_dist", ""),
                "tabu_iter": task.get("tabu_iter", ""),
                "tabu_tenure": task.get("tabu_tenure", ""),
                "k_max": task.get("k_max", ""),
                "search_mode": task.get("search_mode", ""),
                "move_candidate_limit": task.get("move_candidate_limit", ""),
                "route_neighbor_limit": task.get("route_neighbor_limit", ""),
                "position_neighbor_limit": task.get("position_neighbor_limit", ""),
                "exchange_neighbor_limit": task.get("exchange_neighbor_limit", ""),
                "station_candidate_limit": task.get("station_candidate_limit", ""),
                "raw_global_value": "",
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
    parser = argparse.ArgumentParser(description="Run the pickle-native VNS-Tabu Search benchmark.")
    parser.add_argument("--dataset_path", required=True, help="Dataset root or one instance pickle file.")
    parser.add_argument("--save_path", required=True, help="Directory for VNS-TS summaries and solution pickles.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--predefine_route_number", type=int, default=3)
    parser.add_argument("--eta_feas", type=int, default=20, help="Feasibility-phase iteration budget. Legacy default: 700.")
    parser.add_argument("--eta_dist", type=int, default=20, help="Distance-phase iteration budget. Legacy default: 100.")
    parser.add_argument("--tabu_iter", type=int, default=10, help="Inner tabu iterations. Legacy default: 100.")
    parser.add_argument("--tabu_tenure", type=int, default=30)
    parser.add_argument("--k_max", type=int, default=15)
    parser.add_argument("--search_mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--move_candidate_limit", type=int, default=40)
    parser.add_argument("--route_neighbor_limit", type=int, default=4)
    parser.add_argument("--position_neighbor_limit", type=int, default=4)
    parser.add_argument("--exchange_neighbor_limit", type=int, default=6)
    parser.add_argument("--station_candidate_limit", type=int, default=5)
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
        f"VNS-TS benchmark schedule: instances={len(instance_files)}, num_workers={num_workers}, "
        f"eta_feas={args.eta_feas}, eta_dist={args.eta_dist}, tabu_iter={args.tabu_iter}, "
        f"predefine_route_number={args.predefine_route_number}, search_mode={args.search_mode}, "
        f"move_candidate_limit={args.move_candidate_limit}"
    )

    tasks = []
    for idx, instance_file in enumerate(instance_files):
        tasks.append({
            "instance_file": str(instance_file),
            "seed": int(args.seed) + idx,
            "predefine_route_number": args.predefine_route_number,
            "eta_feas": args.eta_feas,
            "eta_dist": args.eta_dist,
            "tabu_iter": args.tabu_iter,
            "tabu_tenure": args.tabu_tenure,
            "k_max": args.k_max,
            "search_mode": args.search_mode,
            "move_candidate_limit": args.move_candidate_limit,
            "route_neighbor_limit": args.route_neighbor_limit,
            "position_neighbor_limit": args.position_neighbor_limit,
            "exchange_neighbor_limit": args.exchange_neighbor_limit,
            "station_candidate_limit": args.station_candidate_limit,
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
        "runtime_s", "seed", "predefine_route_number", "eta_feas", "eta_dist", "tabu_iter",
        "tabu_tenure", "k_max", "search_mode", "move_candidate_limit", "route_neighbor_limit",
        "position_neighbor_limit", "exchange_neighbor_limit", "station_candidate_limit",
        "raw_global_value", "routes_json", "route_sequence_json",
        "solution_path", "errors", "traceback",
    ]
    route_fieldnames = ["instance_id", "file", "route_idx", "route_json", "route_sequence_json"]

    write_csv(save_path / "vns_ts_summary.csv", summary_rows, summary_fieldnames)
    write_csv(save_path / "vns_ts_routes.csv", route_rows, route_fieldnames)
    print(f"Saved summary: {save_path / 'vns_ts_summary.csv'}")
    print(f"Saved routes: {save_path / 'vns_ts_routes.csv'}")


if __name__ == "__main__":
    main()
