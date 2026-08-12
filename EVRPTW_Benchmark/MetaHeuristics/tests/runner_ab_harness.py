#!/usr/bin/env python3
"""Lightweight deterministic A/B acceptance for both metaheuristic runners.

The harness runs one fixed-iteration instance once in-process and once through
the bounded process pool, then compares every deterministic solution field. It
is intentionally an opt-in smoke tool rather than part of the normal test
suite because it loads a real Stage-2 family and starts solver processes.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ALNS_RUNNER = (
    REPO_ROOT / "EVRPTW_Benchmark" / "MetaHeuristics" / "ALNS_Solver" / "run_alns.py"
)
VNS_RUNNER = (
    REPO_ROOT
    / "EVRPTW_Benchmark"
    / "MetaHeuristics"
    / "VNS_TS_Solver"
    / "run_vns_ts.py"
)

SUMMARY_KEYS = (
    "instance_id",
    "status",
    "benchmark_status",
    "has_incumbent",
    "feasible",
    "objective_distance_km",
    "vehicle_count",
    "seed",
    "seed_scheme",
    "algorithm_profile_id",
    "run_contract_fingerprint",
    "run_contract_json",
    "routes_json",
    "route_sequence_json",
    "route_validation_passed",
)
TRACE_KEYS = (
    "instance_id",
    "solver_name",
    "algorithm_profile_id",
    "seed",
    "seed_scheme",
    "run_contract_fingerprint",
    "checkpoint_s",
    "status",
    "benchmark_status",
    "has_incumbent",
    "objective_distance_km",
    "vehicle_count",
    "routes_json",
    "route_sequence_json",
    "source",
)


def read_rows(path: Path, keys: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {key: row.get(key, "") for key in keys}
        for row in sorted(
            rows,
            key=lambda item: (
                item.get("instance_id", ""),
                float(item.get("checkpoint_s") or 0.0),
            ),
        )
    ]


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--family_root", default=None)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    common = [
        "--dataset_path",
        str(Path(args.dataset_path).resolve()),
        "--max_instances",
        "1",
        "--seed",
        str(args.seed),
        "--time_limit_s",
        "30",
        "--checkpoints_s",
        "30",
    ]
    if args.family_root:
        common.extend(["--family_root", str(Path(args.family_root).resolve())])

    solvers = {
        "alns": {
            "runner": ALNS_RUNNER,
            "extra": ["--delta_iters", "2"],
            "summary": "alns_summary.csv",
            "trace": "alns_time_trace.csv",
        },
        "vns_ts": {
            "runner": VNS_RUNNER,
            "extra": [
                "--eta_feas",
                "1",
                "--eta_dist",
                "1",
                "--tabu_iter",
                "1",
            ],
            "summary": "vns_ts_summary.csv",
            "trace": "vns_ts_time_trace.csv",
        },
    }
    report = {}
    for solver_name, config in solvers.items():
        output_a = work / f"{solver_name}_single_worker"
        output_b = work / f"{solver_name}_process_pool"
        run(
            [
                args.python,
                str(config["runner"]),
                *common,
                *config["extra"],
                "--save_path",
                str(output_a),
                "--num_workers",
                "1",
            ]
        )
        run(
            [
                args.python,
                str(config["runner"]),
                *common,
                *config["extra"],
                "--save_path",
                str(output_b),
                "--num_workers",
                "2",
                "--max_in_flight",
                "1",
            ]
        )
        summary_a = read_rows(output_a / config["summary"], SUMMARY_KEYS)
        summary_b = read_rows(output_b / config["summary"], SUMMARY_KEYS)
        trace_a = read_rows(output_a / config["trace"], TRACE_KEYS)
        trace_b = read_rows(output_b / config["trace"], TRACE_KEYS)
        if summary_a != summary_b or trace_a != trace_b:
            raise AssertionError(
                f"{solver_name} single-worker/process-pool deterministic output mismatch"
            )
        report[solver_name] = {
            "passed": True,
            "instance_id": summary_a[0]["instance_id"],
            "objective_distance_km": summary_a[0]["objective_distance_km"],
            "seed": summary_a[0]["seed"],
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
