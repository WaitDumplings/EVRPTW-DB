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


def _valid_completed_city_report(
    report_path: Path,
    *,
    city: str,
    expected_count: int,
    code_commit: str,
) -> bool:
    if not report_path.is_file():
        return False
    try:
        report = _read_json(report_path)
    except (OSError, ValueError):
        return False
    rows = report.get("families", [])
    return bool(
        report.get("schema") == c3.C3_SCHEMA
        and report.get("passed") is True
        and report.get("status") == "passed_report_only"
        and report.get("code_provenance", {}).get("code_commit") == code_commit
        and len(rows) == expected_count
        and all(str(row.get("city_slug")) == city and row.get("selected") for row in rows)
    )


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
    cities = sorted(families["city_slug"].astype(str).unique())
    if len(families) != 7_500 or len(cities) != 11:
        raise ValueError(
            "Official parallel C3 requires the complete 7,500-family / 11-city plan; "
            f"observed families={len(families)}, cities={len(cities)}"
        )
    city_counts = {
        city: int(families["city_slug"].astype(str).eq(city).sum()) for city in cities
    }
    work_root = args.output.parent / "c3_city_reports"
    work_root.mkdir(parents=True, exist_ok=True)
    pending: deque[str] = deque()
    completed: set[str] = set()
    for city in cities:
        report_path = work_root / f"{city}.json"
        if _valid_completed_city_report(
            report_path,
            city=city,
            expected_count=city_counts[city],
            code_commit=str(provenance["code_commit"]),
        ):
            completed.add(city)
        else:
            pending.append(city)

    running: dict[str, tuple[subprocess.Popen[str], Any]] = {}
    last_counts = {city: 0 for city in cities}
    last_change = {city: time.monotonic() for city in cities}
    worker_script = Path(__file__).with_name("apply_stage2_joint_support_gate.py")

    def start_city(city: str) -> None:
        output = work_root / f"{city}.json"
        progress = work_root / f"{city}.progress.json"
        log_path = work_root / f"{city}.log"
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
            "--mode", "official",
            "--official-cle-contract", args.official_cle_contract,
            "--cities", city,
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
        running[city] = (process, handle)
        last_change[city] = time.monotonic()

    try:
        while pending or running:
            while pending and len(running) < args.workers:
                start_city(pending.popleft())
            time.sleep(1.0)
            for city, (process, handle) in list(running.items()):
                progress_path = work_root / f"{city}.progress.json"
                if progress_path.is_file():
                    try:
                        count = int(_read_json(progress_path).get("completed", 0))
                    except (OSError, ValueError):
                        count = last_counts[city]
                    if count > last_counts[city]:
                        last_counts[city] = count
                        last_change[city] = time.monotonic()
                if (
                    process.poll() is None
                    and time.monotonic() - last_change[city]
                    >= args.family_wall_timeout_s
                ):
                    raise TimeoutError(
                        f"C3 city worker {city} made no family progress for "
                        f"{args.family_wall_timeout_s} seconds"
                    )
                return_code = process.poll()
                if return_code is None:
                    continue
                handle.close()
                del running[city]
                if return_code != 0:
                    raise RuntimeError(
                        f"C3 city worker failed: city={city}, exit={return_code}, "
                        + "log="
                        + str(work_root / f"{city}.log")
                    )
                report_path = work_root / f"{city}.json"
                if not _valid_completed_city_report(
                    report_path,
                    city=city,
                    expected_count=city_counts[city],
                    code_commit=str(provenance["code_commit"]),
                ):
                    raise RuntimeError(f"C3 city report failed validation: {report_path}")
                completed.add(city)
                last_counts[city] = city_counts[city]
            c3._atomic_json(
                args.progress_output,
                {
                    "schema": "cle_evrptw_c3_parallel_progress_v1",
                    "planned": int(len(families)),
                    "completed": int(sum(last_counts.values())),
                    "completed_cities": sorted(completed),
                    "active_cities": sorted(running),
                    "pending_cities": list(pending),
                    "city_counts": city_counts,
                    "city_completed_counts": last_counts,
                    "updated_monotonic_s": time.monotonic(),
                },
            )
    except BaseException:
        _terminate_all(running, args.termination_grace_s)
        raise

    updates: dict[str, dict[str, Any]] = {}
    city_summaries: list[dict[str, Any]] = []
    family_summaries: list[dict[str, Any]] = []
    for city in cities:
        city_report = _read_json(work_root / f"{city}.json")
        reason_counts: Counter[str] = Counter()
        for family in city_report["families"]:
            family_id = str(family["family_id"])
            selected = dict(family["selected"])
            reason_counts.update(family.get("rejected_pair_reason_counts", {}))
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
        city_summaries.append(
            {
                "city_slug": city,
                "family_count": city_counts[city],
                "rejected_pair_reason_counts": dict(sorted(reason_counts.items())),
                "report": str((work_root / f"{city}.json").resolve()),
            }
        )
    planned_ids = set(families["family_id"].astype(str))
    if set(updates) != planned_ids:
        raise RuntimeError("Parallel C3 update IDs do not equal full family plan IDs")
    report = {
        "schema": SCHEMA,
        "status": "applying_plan",
        "passed": None,
        "full_plan": True,
        "covered_family_count": len(updates),
        "city_count": len(cities),
        "workers": int(args.workers),
        "family_wall_timeout_s": float(args.family_wall_timeout_s),
        "official_cle_contract": args.official_cle_contract,
        "benchmark_positioning": (
            "infrastructure_grounded_semi_synthetic_not_fully_real"
        ),
        "manual_cle_release_claimed": False,
        "city_summaries": city_summaries,
        "families": family_summaries,
        "code_provenance": provenance,
        "hash_validation_performed": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    c3._atomic_json(args.output, report)
    c3._apply_updates(
        args.plan_root,
        parts,
        updates,
        c3_report=args.output,
        code_provenance=provenance,
        full_plan=True,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=11)
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
    report = run(args)
    print(
        "Parallel C3: "
        f"status={report['status']} covered={report['covered_family_count']} "
        f"cities={report['city_count']}"
    )
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
