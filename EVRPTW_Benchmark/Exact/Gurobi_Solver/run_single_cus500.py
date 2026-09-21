"""One random Road Cus500 test instance, monetary D_time objective, native Gurobi defaults.

No process pool or thread caps. Logs native output plus a durable callback CSV.
Every callback-observed objective/bound/gap change is written without throttling.
By default neither TimeLimit, MIPGap nor Threads is set. An optional TimeLimit
applies to optimization only; elapsed wall time is logged separately.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import secrets
import signal
import subprocess
import sys
import time
import traceback
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Dataset_Generator/src"))

from evrptw_core.objective import load_objective, select_objective_distance
from evrptw_core.io import save_solution
from gurobipy import GRB
import gurobipy as gp
from gurobi_solver import GurobiEVRPTWSolver, GurobiSolverConfig
from stage2_adapter import read_stage2_tasks, load_stage2_instance

TRACKS = {
    "T1": "test1_new_seed",
    "T2": "test2_heldout_locations",
    "T3": "test3_heldout_city",
}
DEFAULT_RELEASE = "us_11city_full_clean_v7_bbde5db_20260823"
FIELDS = [
    "timestamp_utc",
    "event",
    "wall_runtime_s",
    "gurobi_runtime_s",
    "objective_value_usd",
    "best_bound_usd",
    "mip_gap",
    "mip_gap_percent",
    "gap_status",
    "candidate_objective_usd",
    "node_count",
    "solution_count",
    "solver_status",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and abs(value) < GRB.INFINITY else None


def relative_gap(obj, bound):
    if obj is None:
        return None, "no_incumbent"
    if bound is None:
        return None, "bound_unavailable"
    if obj == 0.0:
        return (0.0, "finite") if bound == 0.0 else (None, "infinite")
    return abs(obj - bound) / abs(obj), "finite"


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(jsonable(obj), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProgressRecorder:
    def __init__(self, output: Path, interval_s: float = 1.0):
        self.output = output
        self.interval_s = interval_s
        self.started = time.perf_counter()
        self.last_write = -math.inf
        self.last_recorded_state = None
        self.best = None
        self.bound = None
        self.error = None
        self.stop_requested = False
        self.csv_file = (output / "progress.csv").open("w", newline="")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=FIELDS)
        self.writer.writeheader()
        self.csv_file.flush()
        self.json_file = (output / "progress.jsonl").open("w")

    def emit(
        self,
        event,
        *,
        runtime=None,
        obj=None,
        bound=None,
        candidate=None,
        nodes=None,
        solutions=None,
        status=None,
        gap_override=None,
    ):
        gap, kind = relative_gap(obj, bound)
        if gap_override is not None:
            gap = gap_override
            kind = "finite"
        row = dict(
            timestamp_utc=now(),
            event=event,
            wall_runtime_s=time.perf_counter() - self.started,
            gurobi_runtime_s=runtime,
            objective_value_usd=obj,
            best_bound_usd=bound,
            mip_gap=gap,
            mip_gap_percent=None if gap is None else 100 * gap,
            gap_status=kind,
            candidate_objective_usd=candidate,
            node_count=nodes,
            solution_count=solutions,
            solver_status=status,
        )
        self.writer.writerow(row)
        self.csv_file.flush()
        self.json_file.write(json.dumps(row, allow_nan=False) + "\n")
        self.json_file.flush()
        write_json(self.output / "latest_progress.json", row)
        self.last_write = time.perf_counter()
        self.last_recorded_state = (obj, bound, gap, kind)
        display = lambda v: "NA" if v is None else f"{v:.10g}"
        print(
            f"[progress] {event} wall={row['wall_runtime_s']:.3f}s "
            f"gurobi={display(runtime)}s objective={display(obj)} USD "
            f"bound={display(bound)} gap={display(gap)} "
            f"({display(row['mip_gap_percent'])}%)",
            flush=True,
        )
        return row

    def callback(self, model, where):
        if where == GRB.Callback.POLLING:
            return
        try:
            if self.stop_requested:
                model.terminate()
            get = lambda code: finite(model.cbGet(code))
            runtime = get(GRB.Callback.RUNTIME)
            candidate = None
            nodes = None
            solutions = None
            event = {
                GRB.Callback.PRESOLVE: "presolve",
                GRB.Callback.SIMPLEX: "simplex",
                GRB.Callback.BARRIER: "barrier",
                GRB.Callback.MIP: "mip",
                GRB.Callback.MIPSOL: "mipsol",
                GRB.Callback.MIPNODE: "mipnode",
            }.get(where)
            if event is None:
                return
            if where in (GRB.Callback.MIP, GRB.Callback.MIPSOL, GRB.Callback.MIPNODE):
                prefix = {
                    GRB.Callback.MIP: "MIP",
                    GRB.Callback.MIPSOL: "MIPSOL",
                    GRB.Callback.MIPNODE: "MIPNODE",
                }[where]
                # OBJBST is the incumbent; MIPSOL_OBJ is a candidate
                # (not necessarily an improvement).
                incumbent = get(getattr(GRB.Callback, prefix + "_OBJBST"))
                if incumbent is not None:
                    self.best = incumbent
                bound = get(getattr(GRB.Callback, prefix + "_OBJBND"))
                if bound is not None:
                    self.bound = bound
                nodes = get(getattr(GRB.Callback, prefix + "_NODCNT"))
                solutions = get(getattr(GRB.Callback, prefix + "_SOLCNT"))
                if where == GRB.Callback.MIPSOL:
                    candidate = get(GRB.Callback.MIPSOL_OBJ)
            # Never throttle a changed incumbent, bound, or gap. The interval
            # only limits repeated heartbeat rows when those values are unchanged.
            gap, kind = relative_gap(self.best, self.bound)
            state = (self.best, self.bound, gap, kind)
            if (
                state != self.last_recorded_state
                or where == GRB.Callback.MIPSOL
                or time.perf_counter() - self.last_write >= self.interval_s
            ):
                self.emit(
                    event,
                    runtime=runtime,
                    obj=self.best,
                    bound=self.bound,
                    candidate=candidate,
                    nodes=nodes,
                    solutions=solutions,
                    status="RUNNING",
                )
        except BaseException as exc:
            # Gurobi otherwise catches callback exceptions and may keep solving silently.
            self.error = repr(exc)
            model.terminate()

    def final(self, model, status):
        count = int(model.SolCount)
        obj = finite(model.ObjVal) if count else None
        bound = finite(model.ObjBound)
        gap = finite(model.MIPGap) if count else None
        return self.emit(
            "final",
            runtime=finite(model.Runtime),
            obj=obj,
            bound=bound,
            nodes=finite(model.NodeCount),
            solutions=count,
            status=status,
            gap_override=gap,
        )

    def close(self):
        self.csv_file.close()
        self.json_file.close()


class LoggedSolver(GurobiEVRPTWSolver):
    def __init__(self, config, output, recorder, manifest):
        super().__init__(config)
        self.output = output
        self.recorder = recorder
        self.manifest = manifest

    def _build_model(self, instance):
        self.recorder.emit("model_build_start", runtime=0.0, status="BUILDING")
        result = super()._build_model(instance)
        model = result[0]
        model.Params.LogFile = str(self.output / "gurobi.log")
        model.update()
        params = {}
        for name in [
            "Threads",
            "ThreadLimit",
            "MIPGap",
            "MIPGapAbs",
            "TimeLimit",
            "MIPFocus",
            "Heuristics",
            "Presolve",
            "Seed",
            "Method",
            "ConcurrentMIP",
        ]:
            info = model.getParamInfo(name)
            if info is not None:
                params[name] = {"effective": info[2], "native_default": info[5]}
        self.manifest.update(
            status="optimizing",
            gurobi_parameters=params,
            model_variables=int(model.NumVars),
            model_constraints=int(model.NumConstrs),
            model_build_wall_time_s=time.perf_counter() - self.recorder.started,
        )
        write_json(self.output / "run_config.json", self.manifest)
        self.recorder.emit("model_ready", runtime=0.0, status="READY")
        return result

    def _make_callback(self, trace, node_map, x, instance):
        original = super()._make_callback(trace, node_map, x, instance)

        def callback(model, where):
            self.recorder.callback(model, where)
            if self.recorder.error:
                return
            try:
                original(model, where)
            except BaseException as exc:
                self.recorder.error = repr(exc)
                model.terminate()

        return callback


def choose_task(tasks, seed):
    candidates = sorted(
        (t for t in tasks if t.scale_label == "Cus500" and t.split_id == "test"),
        key=lambda t: (t.view_id, t.index_path),
    )
    if not candidates:
        raise ValueError("No Road Cus500 test instances in selected index")
    if len({t.view_id for t in candidates}) != len(candidates):
        raise ValueError("Duplicate instance IDs in candidate list")
    return random.Random(seed).choice(candidates), candidates


def default_output_dir():
    return REPO_ROOT / "EVRPTW_Benchmark/results" / (
        "gurobi_single_Cus500_dtime_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + f"_{os.getpid()}"
    )


def launch_background(args):
    """Detach one solver, with no terminal descriptors or inherited SIGHUP."""
    out = (args.output_dir or default_output_dir()).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Use a NEW output directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(str(out) + ".launcher.log")
    command = [sys.executable, "-u", str(Path(__file__).resolve())]
    # Rebuild parsed options, so the child cannot inherit --background (even if
    # the caller used an argparse abbreviation). Do not set solver thread caps.
    for key, value in vars(args).items():
        if key in {"background", "output_dir"} or value is None or value is False:
            continue
        command.append("--" + key)
        if value is not True:
            command.append(str(value))
    command.extend(["--output_dir", str(out)])
    # Reserve the launch log before spawning; a second launch for the same
    # destination fails even before the worker creates its output directory.
    with log_path.open("x") as log:
        old_hup = signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, close_fds=True)
        finally:
            signal.signal(signal.SIGHUP, old_hup)
    write_json(Path(str(out) + ".launcher.json"), {
        "status": "spawned", "launched_at": now(), "pid": child.pid,
        "output_dir": str(out), "progress_csv": str(out / "progress.csv"),
        "log_file": str(log_path), "command": command,
    })
    print(f"Background PID: {child.pid}\nCSV: {out / 'progress.csv'}\nLog: {log_path}", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--background", action="store_true",
                    help="Detach solver; print PID and CSV/log paths, then exit.")
    ap.add_argument("--dataset_root", type=Path, default=None)
    ap.add_argument("--track", choices=TRACKS, default="T1")
    ap.add_argument(
        "--selection_seed",
        type=int,
        default=None,
        help="Only controls which instance is drawn; does not change Gurobi Seed.",
    )
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument(
        "--time_limit_s",
        type=float,
        default=None,
        help=(
            "Optional optimizer TimeLimit; default unset/unlimited. "
            "Model construction is logged separately."
        ),
    )
    ap.add_argument(
        "--log_interval_s",
        type=float,
        default=1.0,
        help=(
            "Heartbeat interval for unchanged state; every observed objective/bound/gap "
            "change, MIPSOL, and final is always written."
        ),
    )
    ap.add_argument("--cs_copies", type=int, default=2)
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Select and validate one instance and write its manifest; "
            "do not construct or solve a MILP."
        ),
    )
    args = ap.parse_args(argv)
    if args.time_limit_s is not None and (
        not math.isfinite(args.time_limit_s) or args.time_limit_s <= 0
    ):
        ap.error("--time_limit_s must be finite and positive")
    if not math.isfinite(args.log_interval_s) or args.log_interval_s <= 0:
        ap.error("--log_interval_s must be finite and positive")
    if args.cs_copies < 1:
        ap.error("--cs_copies must be positive")
    if args.background:
        return launch_background(args)
    dataset = (
        args.dataset_root
        or Path(
            os.environ.get("EVRPTW_DATASET_ROOT")
            or os.environ.get("CUS100_ROAD_ROOT")
            or REPO_ROOT / "EVRPTW_Dataset/Instances_v2" / DEFAULT_RELEASE
        )
    ).resolve()
    index = (
        dataset
        / "generation_plan/core/test"
        / TRACKS[args.track]
        / "view_index.parquet"
    )
    seed = (
        args.selection_seed
        if args.selection_seed is not None
        else secrets.randbits(63)
    )
    task, candidates = choose_task(
        read_stage2_tasks(index, family_root=dataset / "materialized/families"), seed
    )
    out = (args.output_dir or default_output_dir()).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    objective_path = (
        REPO_ROOT
        / "EVRPTW_Benchmark/Reinforcement_Learning/configs/rivian_energy_vehicle_cost_v2.json"
    )
    objective = load_objective(objective_path)
    instance = select_objective_distance(load_stage2_instance(task), objective)
    assert instance.num_customers == 500 and objective.is_cost
    script_dir = Path(__file__).resolve().parent
    sources = [
        Path(__file__),
        script_dir / "gurobi_solver.py",
        script_dir / "route_validator.py",
        script_dir / "stage2_adapter.py",
        REPO_ROOT / "EVRPTW_Core/evrptw_core/objective.py",
    ]
    manifest = {
        "schema": "single_cus500_dtime_cost_native_gurobi_v2",
        "created_at": now(),
        "status": "prepared",
        "instance_id": task.view_id,
        "selection_seed": seed,
        "sampling": "one uniform draw from sorted unique Road Cus500 test IDs",
        "selection_candidates": len(candidates),
        "selection_track": args.track,
        "selected_task": asdict(task),
        "index_sha256": sha(index),
        "objective": objective.to_dict(),
        "objective_config_sha256": sha(objective_path),
        "cost_distance_source": instance.metadata["objective_distance_source"],
        "time_source": "running_time_shortest_matrix_s",
        "energy_source": "running_time_path_energy_kwh = kappa D_time",
        "gurobi_version": ".".join(map(str, gp.gurobi.version())),
        "cs_copies": args.cs_copies,
        "thread_policy": (
            "Threads/ThreadLimit not set by script; "
            "native automatic/default parameters retained"
        ),
        "mipgap_policy": (
            "MIPGap/MIPGapAbs not set by script; "
            "native default termination tolerances retained"
        ),
        "time_limit_s_requested": args.time_limit_s,
        "process_policy": (
            "one instance, one direct model.optimize call, "
            "no worker pool or per-worker thread cap"
        ),
        "timing": (
            "gurobi_runtime_s is optimizer Runtime; wall_runtime_s starts before "
            "model construction and includes construction and post-solve replay; "
            "data loading precedes both"
        ),
        "gap_note": (
            "mip_gap is a ratio, mip_gap_percent is 100 times ratio; "
            "missing incumbent/bound left empty, never zero; final uses raw "
            "Gurobi MIPGap including nonzero residual at OPTIMAL"
        ),
        "progress_sampling": (
            "every callback-observed objective/bound/gap change without interval throttling; "
            "also all MIPSOL events, final, and unchanged-state heartbeats at >= requested interval; "
            "callback observations only, not unobservable internal optimizer transitions"
        ),
        "log_interval_s": args.log_interval_s,
        "cpu_logical_count": os.cpu_count(),
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "source_sha256": {
            str(p.relative_to(REPO_ROOT)): sha(p) for p in sources
        },
        "source_repo": str(REPO_ROOT),
        "inherited_numeric_thread_variables": {
            k: os.environ[k]
            for k in [
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ]
            if k in os.environ
        },
    }
    write_json(out / "run_config.json", manifest)
    print(
        f"INSTANCE {task.view_id} ({args.track}, Cus500), selection_seed={seed}\n"
        f"OUTPUT {out}",
        flush=True,
    )
    if args.dry_run:
        manifest["status"] = "dry_run_validated_no_solve"
        write_json(out / "run_config.json", manifest)
        return 0
    rec = ProgressRecorder(out, args.log_interval_s)
    config = GurobiSolverConfig(
        time_limit_s=args.time_limit_s,
        mip_gap=None,
        threads=None,
        output_flag=1,
        cs_copies=args.cs_copies,
        checkpoints_s=(),
        tie_break_vehicle_count=False,
        objective_mode=objective.mode,
        objective_profile_id=objective.profile_id,
        electricity_price_usd_per_kwh=objective.electricity_price_usd_per_kwh,
        consumption_kwh_per_km=objective.consumption_kwh_per_km,
        vehicle_fixed_cost_usd=objective.vehicle_fixed_cost_usd,
    )
    solver = LoggedSolver(config, out, rec, manifest)

    def request_stop(signum, frame):
        rec.stop_requested = True
        if solver.model is not None:
            solver.model.terminate()

    previous_handlers = {
        s: signal.signal(s, request_stop) for s in [signal.SIGINT, signal.SIGTERM]
    }
    try:
        solution = solver.solve(instance)
        final = rec.final(solver.model, solution.metadata["gurobi_status_name"])
        save_solution(out / "solution.pkl", solution)
        write_json(out / "solution.json", solution.to_dict())
        summary = {
            "instance_id": instance.instance_id,
            "status": solution.metadata["gurobi_status_name"],
            "verified_feasible": solution.feasible,
            "gurobi_objective_value_usd": final["objective_value_usd"],
            "verified_cost_usd": (
                solution.metadata.get("objective_cost_usd")
                if solution.feasible
                else None
            ),
            "best_bound_usd": final["best_bound_usd"],
            "mip_gap": final["mip_gap"],
            "mip_gap_percent": final["mip_gap_percent"],
            "gurobi_runtime_s": final["gurobi_runtime_s"],
            "wall_runtime_s": final["wall_runtime_s"],
            "vehicle_count": solution.vehicle_count if solution.feasible else None,
            "objective_distance_km": (
                solution.objective_distance_km if solution.feasible else None
            ),
            "cost_distance_source": manifest["cost_distance_source"],
            "callback_error": rec.error,
            "route_validation": solution.metadata.get("route_validation"),
        }
        write_json(out / "summary.json", summary)
        with (out / "summary.csv").open("w", newline="") as f:
            flat = {k: v for k, v in summary.items() if k != "route_validation"}
            w = csv.DictWriter(f, fieldnames=list(flat))
            w.writeheader()
            w.writerow(flat)
        manifest.update(
            status="callback_error" if rec.error else "finished",
            finished_at=now(),
            solver_status=summary["status"],
        )
        write_json(out / "run_config.json", manifest)
        if rec.error:
            raise RuntimeError("Logging/solver callback failed: " + rec.error)
        return 0
    except BaseException as exc:
        write_json(
            out / "error.json",
            {"at": now(), "error": repr(exc), "traceback": traceback.format_exc()},
        )
        manifest.update(status="failed", error=repr(exc))
        write_json(out / "run_config.json", manifest)
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        rec.close()
        if solver.model is not None:
            solver.model.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
