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
from pathlib import Path
from typing import Any, Iterator

# Prevent 30 worker processes from each creating a BLAS/OpenMP thread pool.
# Set EVRPTW_META_THREADS_PER_WORKER before launch to make an explicit override.
_THREADS_PER_WORKER = os.environ.get("EVRPTW_META_THREADS_PER_WORKER", "1")
if int(_THREADS_PER_WORKER) <= 0:
    raise ValueError("EVRPTW_META_THREADS_PER_WORKER must be positive")
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_thread_variable] = _THREADS_PER_WORKER

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
META_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Dataset_Generator" / "src"))
sys.path.insert(0, str(META_ROOT))

from evrptw_core.schema import EVRPTWSolution, merge_route_sequences
from evrptw_core.validation import validate_instance_structure

from benchmark_common import (
    ALGORITHM_TIMING_SCOPE,
    IncumbentEventRecorder,
    IncumbentReplayCache,
    SEED_SCHEME,
    TIME_BUDGET_ITERATION_CEILING,
    SolverTimeLimit,
    bounded_process_map,
    build_run_contract,
    build_input_tasks,
    charging_profile,
    hard_time_limit,
    load_input_task,
    parse_checkpoints,
    parse_scales,
    resolve_schedule,
    stable_view_seed,
)
from benchmark_output import (
    IncrementalCsvStore,
    TIME_TRACE_FIELDNAMES,
    error_snapshot_rows,
    save_result_artifacts,
    snapshot_rows,
)
from instance_adapter import to_alns_tensor_instance
from solver import ALNS_Solver


SOLVER_NAME = "alns_stage2_anytime"
ALGORITHM_PROFILE_ID = "alns_stage2_scalable_v2"


SUMMARY_FIELDNAMES = [
    "instance_id", "file", "family_id", "city_slug", "split_id", "track_id", "scale_id",
    "day_type", "status", "benchmark_status", "benchmark_completed", "has_incumbent",
    "feasible", "objective_distance_km", "vehicle_count", "runtime_s",
    "first_feasible_time_s", "time_limit_s", "terminated_by_time_limit", "timing_scope",
    "seed", "seed_scheme",
    "run_contract_fingerprint", "run_contract_json",
    "cur_iter",
    "max_iters", "delta_iters", "route_validation_passed", "charging_visit_count",
    "total_charging_time_s", "charging_power_min_kw", "charging_power_max_kw",
    "charging_power_derating_factor", "routes_json",
    "route_sequence_json", "incumbent_replay_hits", "incumbent_replay_misses",
    "algorithm_profile_id", "initial_construction_strategy", "initial_solution_source",
    "initial_construction_time_s", "initial_route_count", "algorithm_profile_json",
    "solution_path", "time_trace_path", "errors", "traceback",
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
            "timing_scope": ALGORITHM_TIMING_SCOPE,
            "seed": task["seed"],
            "seed_scheme": task.get("seed_scheme", SEED_SCHEME),
            "run_contract_fingerprint": task.get("run_contract_fingerprint", ""),
            "run_contract_json": task.get("run_contract_json", ""),
            "algorithm_profile_id": ALGORITHM_PROFILE_ID,
            "routes_json": "[]",
            "route_sequence_json": "[]",
            "errors": errors,
            "traceback": trace,
        }
    )
    return {
        "summary_row": summary,
        "time_rows": error_snapshot_rows(
            instance_id,
            info,
            tuple(task["checkpoints_s"]),
            status,
            errors,
            provenance={
                "solver_name": SOLVER_NAME,
                "algorithm_profile_id": ALGORITHM_PROFILE_ID,
                "seed": task.get("seed", ""),
                "seed_scheme": task.get("seed_scheme", SEED_SCHEME),
                "run_contract_fingerprint": task.get(
                    "run_contract_fingerprint", ""
                ),
            },
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

        recorder = IncumbentEventRecorder(task["checkpoints_s"], task["time_limit_s"])
        replay_cache = IncumbentReplayCache(instance)
        solver: ALNS_Solver | None = None
        power_kw = np.asarray([], dtype=np.float32)
        charging_power_factor = 1.0
        configured_max_iters = (
            int(task["max_iters"])
            if task.get("max_iters") is not None
            else (
                TIME_BUDGET_ITERATION_CEILING
                if task.get("delta_iters") is None
                else None
            )
        )
        start = time.perf_counter()

        def observe(_elapsed_s: float, _objective: float, routes: list[list[int]]) -> None:
            audit = replay_cache.validate(routes)
            if audit["passed"]:
                # Use the runner's single experiment clock for both events and
                # finalization; the solver-supplied value is informational.
                recorder.observe(
                    time.perf_counter() - start,
                    float(audit["objective_distance_km"]),
                    routes,
                )

        def construct_and_run_solver() -> None:
            nonlocal charging_power_factor, power_kw, solver
            power_kw, charging_power_factor, _ = charging_profile(instance)
            adapted = to_alns_tensor_instance(instance)
            solver = ALNS_Solver(adapted, seed=seed, format="tensor")
            if configured_max_iters is not None:
                solver.max_iters = configured_max_iters
            remaining_s = float(task["time_limit_s"]) - (
                time.perf_counter() - start
            )
            if remaining_s <= 0.0:
                raise SolverTimeLimit(
                    "adapter/solver construction exhausted the time budget"
                )
            solver.solve(
                delta_iters=task.get("delta_iters"),
                time_limit_s=remaining_s,
                incumbent_callback=observe,
            )

        runner_timeout = False
        try:
            remaining_before_run_s = float(task["time_limit_s"]) - (
                time.perf_counter() - start
            )
            if remaining_before_run_s <= 0.0:
                raise SolverTimeLimit("runner setup exhausted the time budget")
            with hard_time_limit(remaining_before_run_s):
                if task.get("verbose"):
                    construct_and_run_solver()
                else:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        construct_and_run_solver()
        except SolverTimeLimit:
            runner_timeout = True
            if solver is not None:
                solver.terminated_by_time_limit = True
        runtime_s = time.perf_counter() - start
        terminated_by_time_limit = runner_timeout or bool(
            getattr(solver, "terminated_by_time_limit", False)
        )
        run_metadata = (
            solver.get_run_metadata()
            if solver is not None and hasattr(solver, "get_run_metadata")
            else {}
        )
        construction_metadata = dict(run_metadata.get("initial_construction", {}))
        natural_completion = not terminated_by_time_limit
        best = recorder.best_event
        status = "COMPLETED_WITH_INCUMBENT" if best is not None else "UNFINISHED_NO_INCUMBENT"
        snapshots = recorder.snapshots(
            runtime_s=runtime_s,
            natural_completion=natural_completion,
            final_status=status,
        )

        audit = replay_cache.validate(best["routes"]) if best is not None else {
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
                solver_name=SOLVER_NAME,
                routes=routes,
                objective_distance_km=objective,
                vehicle_count=len(routes),
                runtime_s=runtime_s,
                feasible=True,
                metadata={
                    "first_feasible_time_s": recorder.first_feasible_time_s,
                    "time_limit_s": task["time_limit_s"],
                    "terminated_by_time_limit": terminated_by_time_limit,
                    "timing_scope": ALGORITHM_TIMING_SCOPE,
                    "checkpoint_snapshots": snapshots,
                    "charging_model": "full_charge_linear_derated_v2",
                    "charging_power_derating_factor": charging_power_factor,
                    "benchmark_status": status,
                    "benchmark_completed": True,
                    "has_incumbent": True,
                    "algorithm_profile": run_metadata,
                    "algorithm_profile_id": run_metadata.get(
                        "algorithm_profile_id", ALGORITHM_PROFILE_ID
                    ),
                    "seed": seed,
                    "seed_scheme": task.get("seed_scheme", SEED_SCHEME),
                    "run_contract_fingerprint": task.get(
                        "run_contract_fingerprint", ""
                    ),
                    "run_contract_json": task.get("run_contract_json", ""),
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
            "terminated_by_time_limit": terminated_by_time_limit,
            "timing_scope": ALGORITHM_TIMING_SCOPE,
            "seed": seed,
            "seed_scheme": task.get("seed_scheme", SEED_SCHEME),
            "run_contract_fingerprint": task.get("run_contract_fingerprint", ""),
            "run_contract_json": task.get("run_contract_json", ""),
            "cur_iter": int(getattr(solver, "cur_iter", 0)),
            "max_iters": (
                int(solver.max_iters)
                if solver is not None
                else ("" if configured_max_iters is None else configured_max_iters)
            ),
            "delta_iters": "" if task.get("delta_iters") is None else task["delta_iters"],
            "route_validation_passed": audit["passed"],
            "charging_visit_count": audit["charging_visit_count"],
            "total_charging_time_s": audit["total_charging_time_s"],
            "charging_power_min_kw": float(np.min(power_kw)) if len(power_kw) else "",
            "charging_power_max_kw": float(np.max(power_kw)) if len(power_kw) else "",
            "charging_power_derating_factor": charging_power_factor,
            "routes_json": json.dumps(routes),
            "route_sequence_json": json.dumps(merge_route_sequences(routes)),
            "incumbent_replay_hits": replay_cache.hits,
            "incumbent_replay_misses": replay_cache.misses,
            "algorithm_profile_id": run_metadata.get(
                "algorithm_profile_id", ALGORITHM_PROFILE_ID
            ),
            "initial_construction_strategy": run_metadata.get(
                "initial_construction_strategy", ""
            ),
            "initial_solution_source": run_metadata.get("singleton_source", ""),
            "initial_construction_time_s": construction_metadata.get("elapsed_s", ""),
            "initial_route_count": construction_metadata.get("result_route_count", ""),
            "algorithm_profile_json": json.dumps(
                run_metadata, sort_keys=True, separators=(",", ":")
            ),
            "solution_path": "",
            "time_trace_path": "",
            "errors": "" if best is not None else json.dumps(audit["violations"]),
            "traceback": "",
        }
        return {
            "summary_row": summary,
            "time_rows": snapshot_rows(
                instance.instance_id,
                info,
                snapshots,
                recorder.first_feasible_time_s,
                provenance={
                    "solver_name": SOLVER_NAME,
                    "algorithm_profile_id": run_metadata.get(
                        "algorithm_profile_id", ALGORITHM_PROFILE_ID
                    ),
                    "seed": seed,
                    "seed_scheme": task.get("seed_scheme", SEED_SCHEME),
                    "run_contract_fingerprint": task.get(
                        "run_contract_fingerprint", ""
                    ),
                },
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


def run_tasks(
    tasks: list[dict[str, Any]],
    workers: int,
    max_in_flight: int | None = None,
) -> Iterator[dict[str, Any]]:
    yield from bounded_process_map(
        solve_one,
        tasks,
        workers=workers,
        max_in_flight=max_in_flight,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ALNS on Stage-2 view_index/materialized-family instances."
    )
    parser.add_argument("--dataset_path", required=True, help="Stage-2 root or view_index.parquet")
    parser.add_argument("--family_root", default=None, help="Optional materialized/families override")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0, help="Inclusive filtered index")
    parser.add_argument("--end_index", type=int, default=None, help="Exclusive filtered index")
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument(
        "--max_in_flight",
        type=int,
        default=None,
        help="Bound queued worker tasks; default is 2x num_workers",
    )
    parser.add_argument(
        "--csv_flush_interval",
        type=int,
        default=0,
        help="Materialize canonical CSV every N results; 0 writes only at the end",
    )
    parser.add_argument("--scales", default="", help="Optional list such as Cus50,Cus100")
    parser.add_argument("--time_limit_s", type=float, default=None)
    parser.add_argument(
        "--checkpoints_s",
        default="",
        help="Comma-separated checkpoints. Default follows --time_limit_s; otherwise 60,300,900,3600,7200.",
    )
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--delta_iters", type=int, default=None, help="Smoke-test iteration budget")
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
    summary_path = save_path / "alns_summary.csv"
    trace_path = save_path / "alns_time_trace.csv"

    input_tasks = build_input_tasks(
        args.dataset_path,
        family_root=args.family_root,
        scales=parse_scales(args.scales),
        max_instances=args.max_instances,
        start_index=args.start_index,
        end_index=args.end_index,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    if args.csv_flush_interval < 0:
        parser.error("--csv_flush_interval must be non-negative")
    store = IncrementalCsvStore(
        summary_path=summary_path,
        trace_path=trace_path,
        summary_fieldnames=SUMMARY_FIELDNAMES,
        trace_fieldnames=TIME_TRACE_FIELDNAMES,
        solver_key="alns",
        resume=args.skip_completed,
    )
    terminal_statuses = {"COMPLETED_WITH_INCUMBENT", "UNFINISHED_NO_INCUMBENT"}
    tasks = []
    for task in input_tasks:
        instance_id, _ = source_info_from_task(task)
        instance_seed = stable_view_seed(args.seed, instance_id)
        effective_max_iters = (
            int(args.max_iters)
            if args.max_iters is not None
            else (
                TIME_BUDGET_ITERATION_CEILING
                if args.delta_iters is None
                else None
            )
        )
        search_budget_mode = (
            "delta_iterations"
            if args.delta_iters is not None
            else (
                "max_iterations_and_wall_clock"
                if args.max_iters is not None
                else "wall_clock"
            )
        )
        task.update(
            {
                "seed": instance_seed,
                "seed_scheme": SEED_SCHEME,
                "time_limit_s": time_limit_s,
                "checkpoints_s": checkpoints_s,
                "max_iters": args.max_iters,
                "delta_iters": args.delta_iters,
                "verbose": args.verbose,
                "save_traceback": args.save_traceback,
            }
        )
        fingerprint, contract_json = build_run_contract(
            task,
            algorithm_name=SOLVER_NAME,
            algorithm_profile_id=ALGORITHM_PROFILE_ID,
            base_seed=args.seed,
            solver_parameters={
                "search_budget_mode": search_budget_mode,
                "max_iters_requested": args.max_iters,
                "max_iters_effective": effective_max_iters,
                "delta_iters": args.delta_iters,
                "certificate_warm_start": "canonical_replayed_if_available",
                "charging_model": "full_charge_per_station_power",
            },
        )
        task["run_contract_fingerprint"] = fingerprint
        task["run_contract_json"] = contract_json
        if args.skip_completed and store.has_completed_contract(
            instance_id,
            fingerprint,
            terminal_statuses,
        ):
            continue
        tasks.append(task)

    print(
        f"ALNS Stage-2 schedule: instances={len(tasks)}, workers={max(1, args.num_workers)}, "
        f"max_in_flight={args.max_in_flight or max(1, args.num_workers) * 2}, "
        f"time_limit_s={time_limit_s:g}, checkpoints_s={list(checkpoints_s)}, "
        f"seed_scheme={SEED_SCHEME}, shard={args.shard_index}/{args.shard_count}"
    )
    try:
        for result in run_tasks(
            tasks,
            max(1, int(args.num_workers)),
            max_in_flight=args.max_in_flight,
        ):
            save_result_artifacts(
                result,
                solver_name=SOLVER_NAME,
                solutions_dir=solutions_dir,
                checkpoints_dir=checkpoints_dir,
            )
            result["summary_row"]["time_trace_path"] = str(trace_path)
            store.record_result(result)
            if (
                args.csv_flush_interval > 0
                and store.records_since_flush >= args.csv_flush_interval
            ):
                store.flush_canonical()
            row = result["summary_row"]
            print(
                f"{row['instance_id']}: status={row['status']} "
                f"objective={row['objective_distance_km']} runtime_s={row['runtime_s']}"
            )

        store.flush_canonical()
    finally:
        store.close()
    print(f"Saved summary: {summary_path}")
    print(f"Saved time trace: {trace_path}")


if __name__ == "__main__":
    main()
