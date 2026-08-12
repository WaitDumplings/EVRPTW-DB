from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
META_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Dataset_Generator" / "src"))
sys.path.insert(0, str(META_ROOT))

from evrptw_core.schema import EVRPTWSolution, merge_route_sequences
from evrptw_core.validation import validate_instance_structure

from benchmark_common import (
    IncumbentEventRecorder,
    SolverTimeLimit,
    build_input_tasks,
    charging_profile,
    hard_time_limit,
    load_input_task,
    parse_checkpoints,
    parse_scales,
    resolve_schedule,
    validate_routes,
)
from benchmark_output import (
    TIME_TRACE_FIELDNAMES,
    error_snapshot_rows,
    read_csv_rows,
    save_result_artifacts,
    snapshot_rows,
    write_csv,
)
from solver import VNSTSolver
from vnst_adapter import to_vnst_instance


SUMMARY_FIELDNAMES = [
    "instance_id", "file", "family_id", "city_slug", "split_id", "track_id", "scale_id",
    "day_type", "status", "benchmark_status", "benchmark_completed", "has_incumbent",
    "feasible", "objective_distance_km", "vehicle_count", "runtime_s",
    "first_feasible_time_s", "time_limit_s", "terminated_by_time_limit", "seed",
    "predefine_route_number", "eta_feas", "eta_dist", "tabu_iter", "tabu_tenure", "k_max",
    "search_mode", "move_candidate_limit", "route_neighbor_limit", "position_neighbor_limit",
    "exchange_neighbor_limit", "station_candidate_limit", "route_validation_passed",
    "charging_visit_count", "total_charging_time_s", "charging_power_min_kw",
    "charging_power_max_kw", "routes_json", "route_sequence_json", "solution_path",
    "time_trace_path", "errors", "traceback",
]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def source_info_from_task(task: dict[str, Any]) -> tuple[str, dict[str, str]]:
    ref = task.get("stage2_task", {})
    return str(ref.get("view_id", "unknown")), {
        "file": str(ref.get("index_path", "")),
        "family_id": str(ref.get("family_id", "")),
        "city_slug": str(ref.get("city_slug", "")),
        "split_id": str(ref.get("split_id", "")),
        "track_id": str(ref.get("track_id", "")),
        "scale_id": str(ref.get("scale_id", "")),
    }


def failed_result(
    task: dict[str, Any],
    *,
    status: str,
    errors: str,
    trace: str = "",
) -> dict[str, Any]:
    instance_id, info = source_info_from_task(task)
    summary = {key: "" for key in SUMMARY_FIELDNAMES}
    summary.update(
        {
            "instance_id": instance_id,
            **info,
            "status": status,
            "benchmark_status": status,
            "benchmark_completed": False,
            "has_incumbent": False,
            "feasible": False,
            "time_limit_s": task["time_limit_s"],
            "seed": task["seed"],
            "routes_json": "[]",
            "route_sequence_json": "[]",
            "errors": errors,
            "traceback": trace,
        }
    )
    return {
        "summary_row": summary,
        "time_rows": error_snapshot_rows(
            instance_id, info, tuple(task["checkpoints_s"]), status, errors
        ),
        "snapshots": [],
        "solution": None,
    }


def solve_one(task: dict[str, Any]) -> dict[str, Any]:
    seed = int(task["seed"])
    set_random_seed(seed)
    try:
        instance, info = load_input_task(task)
        structural = validate_instance_structure(instance)
        if not structural.success:
            return failed_result(
                task,
                status="INVALID_INSTANCE",
                errors=json.dumps(structural.errors),
            )

        power_kw, _, _ = charging_profile(instance)
        adapted = to_vnst_instance(instance)
        solver = VNSTSolver(
            adapted,
            predefine_route_number=int(task["predefine_route_number"]),
            show_progress=bool(task.get("verbose")),
            search_mode=str(task["search_mode"]),
            move_candidate_limit=int(task["move_candidate_limit"]),
            route_neighbor_limit=int(task["route_neighbor_limit"]),
            position_neighbor_limit=int(task["position_neighbor_limit"]),
            exchange_neighbor_limit=int(task["exchange_neighbor_limit"]),
            station_candidate_limit=int(task["station_candidate_limit"]),
        )
        solver.η_feas = int(task["eta_feas"])
        solver.η_dist = int(task["eta_dist"])
        solver.tabu_iter = int(task["tabu_iter"])
        solver.tabu_tenure = int(task["tabu_tenure"])
        solver.k_max = int(task["k_max"])

        recorder = IncumbentEventRecorder(task["checkpoints_s"], task["time_limit_s"])
        start = time.perf_counter()

        def observe(_elapsed_s: float, _objective: float, routes: list[list[int]]) -> None:
            audit = validate_routes(instance, routes)
            if audit["passed"]:
                recorder.observe(
                    time.perf_counter() - start,
                    float(audit["objective_distance_km"]),
                    routes,
                )

        def run_solver() -> None:
            solver.solve(
                time_limit_s=task["time_limit_s"],
                incumbent_callback=observe,
            )

        try:
            with hard_time_limit(task["time_limit_s"]):
                if task.get("verbose"):
                    run_solver()
                else:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        run_solver()
        except SolverTimeLimit:
            solver.terminated_by_time_limit = True
        runtime_s = time.perf_counter() - start
        natural_completion = not bool(solver.terminated_by_time_limit)
        best = recorder.best_event
        status = "COMPLETED_WITH_INCUMBENT" if best is not None else "UNFINISHED_NO_INCUMBENT"
        snapshots = recorder.snapshots(
            runtime_s=runtime_s,
            natural_completion=natural_completion,
            final_status=status,
        )

        audit = validate_routes(instance, best["routes"]) if best is not None else {
            "passed": False,
            "violations": ["no feasible incumbent"],
            "charging_visit_count": 0,
            "total_charging_time_s": 0.0,
        }
        routes = [] if best is None else best["routes"]
        objective = None if best is None else float(best["objective_distance_km"])
        solution = None
        if best is not None:
            solution = EVRPTWSolution(
                instance_id=instance.instance_id,
                solver_name="vns_ts_stage2_anytime",
                routes=routes,
                objective_distance_km=objective,
                vehicle_count=len(routes),
                runtime_s=runtime_s,
                feasible=True,
                metadata={
                    "first_feasible_time_s": recorder.first_feasible_time_s,
                    "time_limit_s": task["time_limit_s"],
                    "terminated_by_time_limit": solver.terminated_by_time_limit,
                    "checkpoint_snapshots": snapshots,
                    "charging_model": "full_charge_per_station_power",
                    "benchmark_status": status,
                    "benchmark_completed": True,
                    "has_incumbent": True,
                },
            ).to_dict()

        summary = {
            "instance_id": instance.instance_id,
            **info,
            "day_type": instance.day_type,
            "status": status,
            "benchmark_status": status,
            "benchmark_completed": best is not None,
            "has_incumbent": best is not None,
            "feasible": best is not None,
            "objective_distance_km": "" if objective is None else objective,
            "vehicle_count": "" if best is None else len(routes),
            "runtime_s": runtime_s,
            "first_feasible_time_s": (
                "" if recorder.first_feasible_time_s is None else recorder.first_feasible_time_s
            ),
            "time_limit_s": task["time_limit_s"],
            "terminated_by_time_limit": solver.terminated_by_time_limit,
            "seed": seed,
            "predefine_route_number": task["predefine_route_number"],
            "eta_feas": solver.η_feas,
            "eta_dist": solver.η_dist,
            "tabu_iter": solver.tabu_iter,
            "tabu_tenure": solver.tabu_tenure,
            "k_max": solver.k_max,
            "search_mode": solver.search_mode,
            "move_candidate_limit": solver.move_candidate_limit,
            "route_neighbor_limit": solver.route_neighbor_limit,
            "position_neighbor_limit": solver.position_neighbor_limit,
            "exchange_neighbor_limit": solver.exchange_neighbor_limit,
            "station_candidate_limit": solver.station_candidate_limit,
            "route_validation_passed": audit["passed"],
            "charging_visit_count": audit["charging_visit_count"],
            "total_charging_time_s": audit["total_charging_time_s"],
            "charging_power_min_kw": float(np.min(power_kw)) if len(power_kw) else "",
            "charging_power_max_kw": float(np.max(power_kw)) if len(power_kw) else "",
            "routes_json": json.dumps(routes),
            "route_sequence_json": json.dumps(merge_route_sequences(routes)),
            "solution_path": "",
            "time_trace_path": "",
            "errors": "" if best is not None else json.dumps(audit["violations"]),
            "traceback": "",
        }
        return {
            "summary_row": summary,
            "time_rows": snapshot_rows(
                instance.instance_id, info, snapshots, recorder.first_feasible_time_s
            ),
            "snapshots": snapshots,
            "solution": solution,
        }
    except Exception as exc:
        return failed_result(
            task,
            status="ERROR",
            errors=f"{type(exc).__name__}: {exc}",
            trace=traceback.format_exc() if task.get("save_traceback") else "",
        )


def run_tasks(tasks: list[dict[str, Any]], workers: int) -> Iterator[dict[str, Any]]:
    if workers <= 1:
        for task in tasks:
            yield solve_one(task)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(solve_one, task) for task in tasks]
        for future in as_completed(futures):
            yield future.result()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VNS-TS on Stage-2 view_index/materialized-family instances."
    )
    parser.add_argument("--dataset_path", required=True, help="Stage-2 root or view_index.parquet")
    parser.add_argument("--family_root", default=None, help="Optional materialized/families override")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--scales", default="", help="Optional list such as Cus50,Cus100")
    parser.add_argument("--time_limit_s", type=float, default=None)
    parser.add_argument(
        "--checkpoints_s",
        default="",
        help="Comma-separated checkpoints. Default follows --time_limit_s; otherwise 60,300,900,3600,7200.",
    )
    parser.add_argument("--predefine_route_number", type=int, default=3)
    parser.add_argument("--eta_feas", type=int, default=20)
    parser.add_argument("--eta_dist", type=int, default=20)
    parser.add_argument("--tabu_iter", type=int, default=10)
    parser.add_argument("--tabu_tenure", type=int, default=30)
    parser.add_argument("--k_max", type=int, default=15)
    parser.add_argument("--search_mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--move_candidate_limit", type=int, default=40)
    parser.add_argument("--route_neighbor_limit", type=int, default=4)
    parser.add_argument("--position_neighbor_limit", type=int, default=4)
    parser.add_argument("--exchange_neighbor_limit", type=int, default=6)
    parser.add_argument("--station_candidate_limit", type=int, default=5)
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--save_traceback", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    checkpoints_s, time_limit_s = resolve_schedule(
        parse_checkpoints(args.checkpoints_s), args.time_limit_s
    )
    save_path = Path(args.save_path)
    solutions_dir = save_path / "solutions"
    checkpoints_dir = solutions_dir / "checkpoints"
    solutions_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    summary_path = save_path / "vns_ts_summary.csv"
    trace_path = save_path / "vns_ts_time_trace.csv"

    input_tasks = build_input_tasks(
        args.dataset_path,
        family_root=args.family_root,
        scales=parse_scales(args.scales),
        max_instances=args.max_instances,
    )
    existing_summary = read_csv_rows(summary_path) if args.skip_completed else []
    completed = {
        row["instance_id"]
        for row in existing_summary
        if row.get("status") in {"COMPLETED_WITH_INCUMBENT", "UNFINISHED_NO_INCUMBENT"}
    }
    tasks = []
    for index, task in enumerate(input_tasks):
        instance_id, _ = source_info_from_task(task)
        if instance_id in completed:
            continue
        task.update(
            {
                "seed": int(args.seed) + index,
                "time_limit_s": time_limit_s,
                "checkpoints_s": checkpoints_s,
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
            }
        )
        tasks.append(task)

    summary_rows: list[dict[str, Any]] = list(existing_summary)
    time_rows: list[dict[str, Any]] = read_csv_rows(trace_path) if args.skip_completed else []
    print(
        f"VNS-TS Stage-2 schedule: instances={len(tasks)}, workers={max(1, args.num_workers)}, "
        f"time_limit_s={time_limit_s:g}, checkpoints_s={list(checkpoints_s)}"
    )
    for result in run_tasks(tasks, max(1, int(args.num_workers))):
        save_result_artifacts(
            result,
            solver_name="vns_ts_stage2_anytime",
            solutions_dir=solutions_dir,
            checkpoints_dir=checkpoints_dir,
        )
        result["summary_row"]["time_trace_path"] = str(trace_path)
        summary_rows = [
            row for row in summary_rows if row.get("instance_id") != result["summary_row"]["instance_id"]
        ]
        time_rows = [
            row for row in time_rows if row.get("instance_id") != result["summary_row"]["instance_id"]
        ]
        summary_rows.append(result["summary_row"])
        time_rows.extend(result["time_rows"])
        summary_rows.sort(key=lambda row: str(row.get("instance_id", "")))
        time_rows.sort(
            key=lambda row: (str(row.get("instance_id", "")), float(row.get("checkpoint_s", 0.0)))
        )
        write_csv(summary_path, summary_rows, SUMMARY_FIELDNAMES)
        write_csv(trace_path, time_rows, TIME_TRACE_FIELDNAMES)
        row = result["summary_row"]
        print(
            f"{row['instance_id']}: status={row['status']} "
            f"objective={row['objective_distance_km']} runtime_s={row['runtime_s']}"
        )

    if not tasks:
        write_csv(summary_path, summary_rows, SUMMARY_FIELDNAMES)
        write_csv(trace_path, time_rows, TIME_TRACE_FIELDNAMES)
    print(f"Saved summary: {summary_path}")
    print(f"Saved time trace: {trace_path}")


if __name__ == "__main__":
    main()
