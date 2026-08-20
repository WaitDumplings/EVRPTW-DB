#!/usr/bin/env python3
"""Run full-plan C3 by city, then atomically bind all selected pairs to the plan."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import pandas as pd

import apply_stage2_joint_support_gate as c3
from evrptw_stage2.profile import load_reference_profile
from evrptw_stage2.provenance import resolve_git_provenance
from evrptw_stage2.selection import JOINT_SUPPORT_CONTRACT_ID
from evrptw_stage2.toy import load_full_path_toy_manifest, toy_family_ids


SCHEMA = "cle_evrptw_phase_c3_parallel_full_plan_v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _terminate_all(
    running: dict[str, tuple[subprocess.Popen[str], Any]],
    grace_s: float,
) -> None:
    for process, _handle in running.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and any(
        process.poll() is None for process, _handle in running.values()
    ):
        time.sleep(0.25)
    for process, _handle in running.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()
    for _process, handle in running.values():
        handle.close()


def _valid_completed_task_report(
    report_path: Path,
    *,
    city: str,
    expected_family_ids: tuple[str, ...],
    code_commit: str,
) -> bool:
    if not report_path.is_file():
        return False
    try:
        report = _read_json(report_path)
    except (OSError, ValueError):
        return False
    rows = report.get("families", [])
    observed_ids = tuple(sorted(str(row.get("family_id")) for row in rows))
    return bool(
        report.get("schema") == c3.C3_SCHEMA
        and report.get("passed") is True
        and report.get("status") == "passed_report_only"
        and report.get("code_provenance", {}).get("code_commit") == code_commit
        and observed_ids == tuple(sorted(expected_family_ids))
        and all(str(row.get("city_slug")) == city and row.get("selected") for row in rows)
    )


def _build_tasks(
    families: pd.DataFrame,
    *,
    families_per_task: int,
) -> list[dict[str, Any]]:
    tasks_by_city: dict[str, list[dict[str, Any]]] = {}
    for city, city_rows in families.groupby("city_slug", sort=True):
        ids = sorted(city_rows["family_id"].astype(str))
        city_tasks: list[dict[str, Any]] = []
        for start in range(0, len(ids), families_per_task):
            family_ids = tuple(ids[start : start + families_per_task])
            city_tasks.append(
                {
                    "task_id": f"{city}.part-{start // families_per_task:04d}",
                    "city": str(city),
                    "family_ids": family_ids,
                }
            )
        tasks_by_city[str(city)] = city_tasks
    # Round-robin keeps every city represented in the initial worker wave and
    # lets the global queue absorb city-level runtime skew.
    tasks = []
    maximum_parts = max(map(len, tasks_by_city.values()), default=0)
    for part_index in range(maximum_parts):
        for city in sorted(tasks_by_city):
            if part_index < len(tasks_by_city[city]):
                tasks.append(tasks_by_city[city][part_index])
    return tasks


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[2]
    provenance = resolve_git_provenance(
        repo_root,
        require_clean=True,
        require_branch="stage2-repair-candidate",
    )
    parts = c3._load_family_parts(args.plan_root)
    registry = _read_json(args.plan_root / "split_registry.json")
    if registry.get("code_provenance", {}).get("code_commit") != provenance[
        "code_commit"
    ]:
        raise ValueError(
            "C3 plan belongs to a different executable commit; rebuild from a fresh root"
        )
    profile = load_reference_profile(args.profile, official=True)
    expected_contract = str(
        profile.get("source_contract", {}).get(
            "cle_eligibility_contract", "strict_release_v1"
        )
    )
    if args.official_cle_contract != expected_contract:
        raise ValueError(
            "Parallel C3 CLE contract differs from the promoted profile: "
            f"runner={args.official_cle_contract}, profile={expected_contract}"
        )
    families = pd.concat([frame for _, frame in parts], ignore_index=True)
    families = families.drop_duplicates("family_id").sort_values(
        ["city_slug", "family_id"]
    )
    if families.empty:
        raise ValueError("Full C3 plan is empty")
    toy_manifest = None
    full_plan = args.toy_manifest is None
    if args.toy_manifest is not None:
        if args.mode != "official_toy":
            raise ValueError("--toy-manifest requires --mode official_toy")
        toy_manifest = load_full_path_toy_manifest(
            args.toy_manifest,
            code_commit=str(provenance["code_commit"]),
        )
        requested_ids = set(toy_family_ids(toy_manifest))
        missing_ids = sorted(requested_ids - set(families["family_id"].astype(str)))
        if missing_ids:
            raise ValueError(f"Toy families are absent from the plan: {missing_ids}")
        families = families.loc[
            families["family_id"].astype(str).isin(requested_ids)
        ].copy()
    elif args.mode == "official_toy":
        raise ValueError("official_toy parallel C3 requires --toy-manifest")
    cities = sorted(families["city_slug"].astype(str).unique())
    expected_family_count = 7_500 if full_plan else 150
    if len(families) != expected_family_count or len(cities) != 11:
        raise ValueError(
            "Parallel C3 plan scope mismatch: "
            f"expected families={expected_family_count}, cities=11; "
            f"observed families={len(families)}, cities={len(cities)}"
        )
    city_counts = {
        city: int(families["city_slug"].astype(str).eq(city).sum()) for city in cities
    }
    tasks = _build_tasks(families, families_per_task=args.families_per_task)
    task_by_id = {str(task["task_id"]): task for task in tasks}
    work_root = args.output.parent / "c3_task_reports"
    work_root.mkdir(parents=True, exist_ok=True)
    pending: deque[dict[str, Any]] = deque()
    completed: set[str] = set()
    for task in tasks:
        task_id = str(task["task_id"])
        report_path = work_root / f"{task_id}.json"
        if _valid_completed_task_report(
            report_path,
            city=str(task["city"]),
            expected_family_ids=tuple(task["family_ids"]),
            code_commit=str(provenance["code_commit"]),
        ):
            completed.add(task_id)
        else:
            pending.append(task)

    running: dict[str, tuple[subprocess.Popen[str], Any]] = {}
    last_counts = {
        str(task["task_id"]): (
            len(task["family_ids"])
            if str(task["task_id"]) in completed
            else 0
        )
        for task in tasks
    }
    last_change = {str(task["task_id"]): time.monotonic() for task in tasks}
    worker_script = Path(__file__).with_name("apply_stage2_joint_support_gate.py")

    def start_task(task: dict[str, Any]) -> None:
        task_id = str(task["task_id"])
        output = work_root / f"{task_id}.json"
        progress = work_root / f"{task_id}.progress.json"
        log_path = work_root / f"{task_id}.log"
        handle = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(worker_script),
            "--cle-root", str(args.cle_root),
            "--plan-root", str(args.plan_root),
            "--customer-split-root", str(args.customer_split_root),
            "--amazon-artifact-root", str(args.amazon_artifact_root),
            "--amazon-cohort-split", str(args.amazon_cohort_split),
            "--profile", str(args.profile),
            "--output", str(output),
            "--mode", args.mode,
            "--official-cle-contract", args.official_cle_contract,
            "--cities", str(task["city"]),
            "--family-ids", *map(str, task["family_ids"]),
            "--targeted-gate",
            "--report-only",
            "--progress-output", str(progress),
        ]
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        running[task_id] = (process, handle)
        last_change[task_id] = time.monotonic()

    try:
        while pending or running:
            while pending and len(running) < args.workers:
                start_task(pending.popleft())
            time.sleep(1.0)
            for task_id, (process, handle) in list(running.items()):
                task = task_by_id[task_id]
                progress_path = work_root / f"{task_id}.progress.json"
                if progress_path.is_file():
                    try:
                        count = int(_read_json(progress_path).get("completed", 0))
                    except (OSError, ValueError):
                        count = last_counts[task_id]
                    if count > last_counts[task_id]:
                        last_counts[task_id] = count
                        last_change[task_id] = time.monotonic()
                if (
                    process.poll() is None
                    and time.monotonic() - last_change[task_id]
                    >= args.family_wall_timeout_s
                ):
                    raise TimeoutError(
                        f"C3 task {task_id} made no family progress for "
                        f"{args.family_wall_timeout_s} seconds"
                    )
                return_code = process.poll()
                if return_code is None:
                    continue
                handle.close()
                del running[task_id]
                if return_code != 0:
                    raise RuntimeError(
                        f"C3 task failed: task={task_id}, exit={return_code}, "
                        + "log="
                        + str(work_root / f"{task_id}.log")
                    )
                report_path = work_root / f"{task_id}.json"
                if not _valid_completed_task_report(
                    report_path,
                    city=str(task["city"]),
                    expected_family_ids=tuple(task["family_ids"]),
                    code_commit=str(provenance["code_commit"]),
                ):
                    raise RuntimeError(f"C3 task report failed validation: {report_path}")
                completed.add(task_id)
                last_counts[task_id] = len(task["family_ids"])
            city_completed_counts = {
                city: int(
                    sum(
                        last_counts[str(task["task_id"])]
                        for task in tasks
                        if str(task["city"]) == city
                    )
                )
                for city in cities
            }
            active_cities = sorted(
                {str(task_by_id[task_id]["city"]) for task_id in running}
            )
            completed_cities = sorted(
                city
                for city in cities
                if city_completed_counts[city] == city_counts[city]
            )
            c3._atomic_json(
                args.progress_output,
                {
                    "schema": "cle_evrptw_c3_parallel_progress_v1",
                    "planned": int(len(families)),
                    "completed": int(sum(last_counts.values())),
                    "completed_cities": completed_cities,
                    "active_cities": active_cities,
                    "pending_task_count": len(pending),
                    "active_task_ids": sorted(running),
                    "completed_task_count": len(completed),
                    "task_count": len(tasks),
                    "families_per_task": int(args.families_per_task),
                    "city_counts": city_counts,
                    "city_completed_counts": city_completed_counts,
                    "updated_monotonic_s": time.monotonic(),
                },
            )
    except BaseException:
        _terminate_all(running, args.termination_grace_s)
        raise

    updates: dict[str, dict[str, Any]] = {}
    family_summaries: list[dict[str, Any]] = []
    city_reason_counts = {city: Counter() for city in cities}
    city_task_reports = {city: [] for city in cities}
    for task in tasks:
        task_id = str(task["task_id"])
        city = str(task["city"])
        task_report_path = work_root / f"{task_id}.json"
        task_report = _read_json(task_report_path)
        city_task_reports[city].append(str(task_report_path.resolve()))
        for family in task_report["families"]:
            family_id = str(family["family_id"])
            selected = dict(family["selected"])
            city_reason_counts[city].update(
                family.get("rejected_pair_reason_counts", {})
            )
            updates[family_id] = {
                "joint_support_contract_id": JOINT_SUPPORT_CONTRACT_ID,
                "candidate_depot_count": int(family["candidate_depot_count"]),
                "candidate_structure_source_count": int(
                    family["candidate_structure_source_count"]
                ),
                "joint_pair_count": int(family["joint_pair_count"]),
                "aggregate_gate_pass_count": int(family["aggregate_gate_pass_count"]),
                "exact_gate_pass_count": int(family["exact_gate_pass_count"]),
                "selected_depot_id": selected["selected_depot_id"],
                "selected_structure_source_id": selected[
                    "selected_structure_source_id"
                ],
                "required_decile_counts": selected["required_decile_counts"],
                "available_decile_counts": selected["available_decile_counts"],
                "capacity_contract_fingerprint": selected[
                    "capacity_contract_fingerprint"
                ],
                "rejected_pair_reason_counts": json.dumps(
                    dict(sorted(family.get("rejected_pair_reason_counts", {}).items())),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
            family_summaries.append(
                {
                    "family_id": family_id,
                    "city_slug": city,
                    "selected_depot_id": selected["selected_depot_id"],
                    "selected_structure_source_id": selected[
                        "selected_structure_source_id"
                    ],
                    "capacity_contract_fingerprint": selected[
                        "capacity_contract_fingerprint"
                    ],
                    "elapsed_seconds": float(family["elapsed_seconds"]),
                    "attempted_pair_count": len(family.get("attempted_pairs", [])),
                }
            )
    city_summaries = [
        {
            "city_slug": city,
            "family_count": city_counts[city],
            "task_count": len(city_task_reports[city]),
            "rejected_pair_reason_counts": dict(
                sorted(city_reason_counts[city].items())
            ),
            "task_reports": city_task_reports[city],
        }
        for city in cities
    ]
    planned_ids = set(families["family_id"].astype(str))
    if set(updates) != planned_ids:
        raise RuntimeError("Parallel C3 update IDs do not equal selected plan-scope IDs")
    report = {
        "schema": SCHEMA,
        "status": "applying_plan",
        "passed": None,
        "full_plan": full_plan,
        "release_eligible": full_plan,
        "benchmark_role": (
            "full_c3" if full_plan else "non_release_full_path_toy"
        ),
        "covered_family_count": len(updates),
        "city_count": len(cities),
        "workers": int(args.workers),
        "families_per_task": int(args.families_per_task),
        "task_count": len(tasks),
        "family_wall_timeout_s": float(args.family_wall_timeout_s),
        "official_cle_contract": args.official_cle_contract,
        "benchmark_positioning": (
            "infrastructure_grounded_semi_synthetic_not_fully_real"
        ),
        "benchmark_description": "infrastructure-grounded semi-synthetic",
        "manual_cle_release_claimed": False,
        "city_summaries": city_summaries,
        "families": family_summaries,
        "code_provenance": provenance,
        "hash_validation_performed": False,
        "toy_manifest": (
            str(args.toy_manifest.resolve()) if args.toy_manifest is not None else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    c3._atomic_json(args.output, report)
    c3._apply_updates(
        args.plan_root,
        parts,
        updates,
        c3_report=args.output,
        code_provenance=provenance,
        full_plan=full_plan,
    )
    report["status"] = "passed"
    report["passed"] = True
    report["elapsed_seconds"] = time.perf_counter() - started
    c3._atomic_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cle-root", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--customer-split-root", type=Path, required=True)
    parser.add_argument("--amazon-artifact-root", type=Path, required=True)
    parser.add_argument("--amazon-cohort-split", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("official", "official_toy"),
        default="official",
    )
    parser.add_argument("--toy-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--families-per-task", type=int, default=25)
    parser.add_argument("--family-wall-timeout-s", type=float, default=7200.0)
    parser.add_argument("--termination-grace-s", type=float, default=60.0)
    parser.add_argument(
        "--official-cle-contract",
        choices=("strict_release_v1", "frozen_technical_candidate_v1"),
        default="strict_release_v1",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.families_per_task < 1:
        raise ValueError("--families-per-task must be positive")
    report = run(args)
    print(
        "Parallel C3: "
        f"status={report['status']} covered={report['covered_family_count']} "
        f"cities={report['city_count']}"
    )
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
