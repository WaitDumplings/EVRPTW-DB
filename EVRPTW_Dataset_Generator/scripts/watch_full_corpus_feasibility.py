#!/usr/bin/env python3
"""Wait for the full Stage-2 run and reconcile its feasibility evidence.

This watcher is deliberately read-only with respect to family artifacts.  It
does not rerun matrix generation and does not calculate file hashes.  The
Stage-2 verifier already recomputes every view's feasibility certificate; this
program independently reconciles those results with the generation plan,
published family directories, progress ledger, runtime contract, and Phase-1
hard-gate report.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "evrptw_full_corpus_feasibility_watcher_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def snapshot_family_dirs(root: Path) -> set[str]:
    families = root / "materialized" / "families"
    if not families.is_dir():
        return set()
    return {
        item.name
        for item in families.iterdir()
        if item.is_dir() and (item / "family_manifest.json").is_file()
    }


def successful_materialized_ids(report: dict[str, Any]) -> list[str]:
    explicit = report.get("materialized_family_ids")
    if isinstance(explicit, list):
        return list(map(str, explicit))
    return [
        str(item["family_id"])
        for item in report.get("materialized", [])
        if item.get("status") in {"materialized", "reused_verified"}
    ]


def reconcile(root: Path, expected: int) -> dict[str, Any]:
    progress = read_json(root / "stage2_progress.json")
    report = read_json(root / "stage2_run_report.json")
    phase1 = read_json(root / "reports" / "phase1" / "summary.json")
    planned = list(map(str, report.get("planned_family_ids") or []))
    materialized = successful_materialized_ids(report)
    verifications = list(report.get("verified") or [])
    verified = [
        str(item.get("family_id"))
        for item in verifications
        if item.get("family_id") is not None and item.get("passed") is True
    ]
    published = sorted(snapshot_family_dirs(root))
    runtime = dict(report.get("runtime_contract") or {})

    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(report.get("status") == "passed", "stage2 report status is not passed")
    require(report.get("passed") is True, "stage2 report passed is not true")
    require(
        report.get("terminal_report_committed") is True,
        "terminal Stage-2 report was not committed",
    )
    require(progress.get("status") == "passed", "progress status is not passed")
    for field in ("planned", "completed", "materialized", "verified"):
        require(int(progress.get(field, -1)) == expected, f"progress {field} != {expected}")
    for field in ("rejected", "timed_out", "aborted", "unresolved", "not_started"):
        require(int(progress.get(field, -1)) == 0, f"progress {field} != 0")
    require(not progress.get("active_family_ids"), "progress still has active families")

    require(len(planned) == expected, f"planned ID count != {expected}")
    require(len(set(planned)) == expected, "planned IDs are not unique")
    require(len(materialized) == expected, f"materialized ID count != {expected}")
    require(len(set(materialized)) == expected, "materialized IDs are not unique")
    require(len(verifications) == expected, f"verification record count != {expected}")
    require(len(verified) == expected, f"verified PASS ID count != {expected}")
    require(len(set(verified)) == expected, "verified IDs are not unique")
    require(len(published) == expected, f"published family directory count != {expected}")
    if len(planned) == expected:
        reference = set(planned)
        require(set(materialized) == reference, "materialized ID set differs from plan")
        require(set(verified) == reference, "verified ID set differs from plan")
        require(set(published) == reference, "published family ID set differs from plan")

    failed_verifications: list[dict[str, Any]] = []
    totals = {
        "views": 0,
        "certified_customers": 0,
        "requires_charging_customers": 0,
        "stored_full_cs_cache_views": 0,
        "unreachable_full_cs_return_entries": 0,
        "maximum_charging_visit_count": 0,
        "matrix_bytes": 0,
    }
    for item in verifications:
        family_id = str(item.get("family_id", ""))
        item_failures: list[str] = []
        if item.get("schema") != "cle_evrptw_materialized_family_verification_v1":
            item_failures.append("verification schema mismatch")
        if item.get("passed") is not True:
            item_failures.append("passed is not true")
        if item.get("errors") not in ([], None):
            item_failures.append("errors is not empty")
        view_count = int(item.get("view_count", -1))
        certified = int(item.get("certified_customer_count", -1))
        cached_views = int(item.get("stored_full_cs_cache_view_count", -1))
        unreachable = int(item.get("unreachable_full_cs_return_count", -1))
        if view_count <= 0:
            item_failures.append("view_count is not positive")
        if certified <= 0:
            item_failures.append("certified_customer_count is not positive")
        if cached_views != view_count:
            item_failures.append("not every view has a stored full-CS return cache")
        if unreachable != 0:
            item_failures.append("one or more full-CS-to-depot entries are unreachable")
        if item_failures:
            failed_verifications.append(
                {"family_id": family_id, "failures": item_failures}
            )
        totals["views"] += max(0, view_count)
        totals["certified_customers"] += max(0, certified)
        totals["requires_charging_customers"] += max(
            0, int(item.get("requires_charging_customer_count", -1))
        )
        totals["stored_full_cs_cache_views"] += max(0, cached_views)
        totals["unreachable_full_cs_return_entries"] += max(0, unreachable)
        totals["maximum_charging_visit_count"] = max(
            totals["maximum_charging_visit_count"],
            int(item.get("maximum_charging_visit_count", -1)),
        )
        totals["matrix_bytes"] += max(0, int(item.get("matrix_total_bytes", -1)))
    require(not failed_verifications, "one or more family feasibility summaries failed")
    require(
        runtime.get("hard_stop_triggered") is False,
        "materialization runtime hard stop was triggered",
    )
    require(
        int(runtime.get("remaining_process_group_count", -1)) == 0,
        "remaining process-group count is not zero",
    )
    for field in ("timed_out", "aborted", "unresolved_family_ids"):
        require(not runtime.get(field), f"runtime contract {field} is not empty")
    require(
        phase1.get("all_hard_gates_passed") is True,
        "Phase-1 hard correctness gates did not all pass",
    )
    require(
        int(phase1.get("successful_parent_family_count", -1)) == expected,
        f"Phase-1 successful family count != {expected}",
    )

    return {
        "schema": SCHEMA,
        "status": "PASS" if not failures else "RED",
        "passed": not failures,
        "checked_at": utc_now(),
        "instance_root": str(root),
        "expected_family_count": expected,
        "counts": {
            "planned": len(planned),
            "materialized": len(materialized),
            "verified_pass": len(verified),
            "published_family_directories": len(published),
            "phase1_successful": int(phase1.get("successful_parent_family_count", -1)),
        },
        "feasibility_evidence": {
            "basis": "stage2_verifier_recomputed_every_materialized_view",
            **totals,
            "failed_family_count": len(failed_verifications),
            "failed_families": failed_verifications,
        },
        "runtime_gate": {
            "hard_stop_triggered": runtime.get("hard_stop_triggered"),
            "timed_out": runtime.get("timed_out"),
            "aborted": runtime.get("aborted"),
            "unresolved_family_ids": runtime.get("unresolved_family_ids"),
            "remaining_process_group_count": runtime.get(
                "remaining_process_group_count"
            ),
        },
        "phase1_all_hard_gates_passed": phase1.get("all_hard_gates_passed"),
        "failures": failures,
        "family_artifacts_modified": False,
        "file_hashes_computed": False,
        "next_action": (
            "ready_for_reviewed_code_cleanup_and_clean_from_zero_run"
            if not failures
            else "STOP"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=7500)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    watcher_dir = root / "reports" / "post_generation"
    progress_path = watcher_dir / "watcher_progress.json"
    event_path = watcher_dir / "watcher_events.jsonl"
    gate_path = watcher_dir / "full_corpus_feasibility_gate_v1.json"
    ready_path = watcher_dir / "READY_FOR_CLEANUP_AND_CLEAN_RERUN.json"
    started_at = utc_now()
    last_observed: tuple[Any, ...] | None = None
    while True:
        try:
            progress = read_json(root / "stage2_progress.json")
            report = read_json(root / "stage2_run_report.json")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            snapshot = {
                "schema": SCHEMA,
                "status": "waiting_for_readable_reports",
                "started_at": started_at,
                "updated_at": utc_now(),
                "error": str(error),
            }
            atomic_write_json(progress_path, snapshot)
            time.sleep(args.poll_seconds)
            continue

        observed = (
            progress.get("status"),
            progress.get("materialized"),
            progress.get("verified"),
            progress.get("unresolved"),
            report.get("status"),
            report.get("terminal_report_committed"),
        )
        snapshot = {
            "schema": SCHEMA,
            "status": "waiting_for_terminal_stage2_pass",
            "started_at": started_at,
            "updated_at": utc_now(),
            "stage2_progress": {
                "status": progress.get("status"),
                "planned": progress.get("planned"),
                "materialized": progress.get("materialized"),
                "verified": progress.get("verified"),
                "rejected": progress.get("rejected"),
                "timed_out": progress.get("timed_out"),
                "aborted": progress.get("aborted"),
                "unresolved": progress.get("unresolved"),
                "active_family_ids": progress.get("active_family_ids"),
                "not_started": progress.get("not_started"),
            },
            "stage2_report_status": report.get("status"),
            "terminal_report_committed": report.get("terminal_report_committed", False),
            "file_hashes_computed": False,
        }
        atomic_write_json(progress_path, snapshot)
        if observed != last_observed:
            append_event(event_path, {"observed_at": utc_now(), **snapshot})
            last_observed = observed

        terminal = report.get("terminal_report_committed") is True
        if terminal or progress.get("status") == "failed" or report.get("status") == "failed":
            try:
                gate = reconcile(root, args.expected)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                gate = {
                    "schema": SCHEMA,
                    "status": "RED",
                    "passed": False,
                    "checked_at": utc_now(),
                    "instance_root": str(root),
                    "failures": [f"cannot reconcile terminal reports: {error}"],
                    "family_artifacts_modified": False,
                    "file_hashes_computed": False,
                    "next_action": "STOP",
                }
            atomic_write_json(gate_path, gate)
            atomic_write_json(progress_path, {**snapshot, "status": gate["status"]})
            append_event(event_path, {"event": "terminal_gate", **gate})
            if gate["passed"]:
                atomic_write_json(
                    ready_path,
                    {
                        "schema": SCHEMA,
                        "ready": True,
                        "created_at": utc_now(),
                        "feasibility_gate": str(gate_path),
                        "next_action": "reviewed_code_cleanup_then_clean_from_zero_run",
                    },
                )
                return 0
            return 2
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
