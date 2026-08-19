"""Read-only reconciliation of a completed Stage-2 pilot run."""

from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from .parallel import verify_family_path
from .progress import atomic_write_json


RECONCILIATION_REASON = "post_verification_stdout_broken_pipe"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_copy_bytes(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with source.open("rb") as input_handle, os.fdopen(
            descriptor, "wb"
        ) as output_handle:
            while chunk := input_handle.read(1024 * 1024):
                output_handle.write(chunk)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _plan_family_ids(root: Path) -> tuple[list[str], list[str]]:
    parts = sorted((root / "generation_plan").rglob("family_index.parquet"))
    if not parts:
        raise FileNotFoundError("No family_index.parquet found in generation_plan")
    raw_ids: list[str] = []
    for path in parts:
        frame = pd.read_parquet(path, columns=["family_id"])
        raw_ids.extend(map(str, frame["family_id"].tolist()))
    duplicates = sorted(
        family_id for family_id in set(raw_ids) if raw_ids.count(family_id) > 1
    )
    return sorted(set(raw_ids)), duplicates


def _family_inventory(family_root: Path) -> dict[str, tuple[int, int, int]]:
    inventory: dict[str, tuple[int, int, int]] = {}
    if not family_root.is_dir():
        return inventory
    for path in sorted(family_root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        inventory[str(path.relative_to(family_root))] = (
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_mode),
        )
    return inventory


def _duplicate_ids(values: list[str]) -> list[str]:
    return sorted(value for value in set(values) if values.count(value) > 1)


def _ledger_gate(original: Mapping[str, Any], root: Path) -> dict[str, Any]:
    failures: list[str] = []
    empty_top_level = (
        "rejected_attempts",
        "timed_out_attempts",
        "aborted_attempts",
        "not_started_family_ids",
        "unresolved_family_ids",
    )
    for key in empty_top_level:
        if list(original.get(key) or []):
            failures.append(f"original report {key} is non-empty")
    if bool(original.get("hard_stop_triggered")):
        failures.append("original report hard_stop_triggered is true")

    runtime = dict(original.get("runtime_contract") or {})
    for key in (
        "timed_out",
        "aborted",
        "cancelled_family_ids",
        "skipped_prior_timeout_family_ids",
        "unresolved_family_ids",
    ):
        if list(runtime.get(key) or []):
            failures.append(f"runtime contract {key} is non-empty")
    if bool(runtime.get("hard_stop_triggered")):
        failures.append("runtime contract hard_stop_triggered is true")
    if int(runtime.get("remaining_process_group_count", -1)) != 0:
        failures.append("runtime contract remaining_process_group_count is not zero")

    timeout_paths = sorted((root / "timeouts").glob("*.json"))
    if timeout_paths:
        failures.append(f"timeout ledgers exist: {len(timeout_paths)}")

    rejection_attempt_count = 0
    rejection_paths = sorted((root / "rejections").glob("*.json"))
    for path in rejection_paths:
        rejection_attempt_count += len(_load_json(path).get("attempts") or [])
    if rejection_attempt_count:
        failures.append(
            f"rejection ledgers contain {rejection_attempt_count} attempts"
        )

    decisions = list(runtime.get("decisions") or [])
    decision_ids = [str(item.get("family_id")) for item in decisions]
    decision_duplicate_ids = _duplicate_ids(decision_ids)
    if decision_duplicate_ids:
        failures.append(
            f"runtime decisions contain duplicate IDs: {decision_duplicate_ids}"
        )
    bad_decisions = [
        str(item.get("family_id"))
        for item in decisions
        if str(item.get("status")) != "materialized"
    ]
    if bad_decisions:
        failures.append(f"non-materialized runtime decisions: {bad_decisions}")

    return {
        "passed": not failures,
        "failures": failures,
        "timeout_ledger_count": len(timeout_paths),
        "rejection_ledger_count": len(rejection_paths),
        "rejection_attempt_count": rejection_attempt_count,
        "runtime_decision_count": len(decisions),
        "runtime_decision_duplicate_ids": decision_duplicate_ids,
        "runtime_decision_family_ids": decision_ids,
        "remaining_process_group_count": runtime.get(
            "remaining_process_group_count"
        ),
    }


def _verify_all(
    family_paths: list[Path],
    *,
    workers: int,
    verifier: Callable[[str | Path], dict[str, Any]],
) -> list[dict[str, Any]]:
    if workers <= 0:
        raise ValueError("workers must be positive")

    def failed(path: Path, error: BaseException) -> dict[str, Any]:
        return {
            "family_id": path.name,
            "passed": False,
            "errors": [f"{type(error).__name__}: {error}"],
            "warnings": [],
            "verifier_exception": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }

    if workers == 1:
        results: list[dict[str, Any]] = []
        for path in family_paths:
            try:
                result = verifier(path)
            except Exception as error:
                result = failed(path, error)
            result.setdefault("family_id", path.name)
            results.append(result)
        return results

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_path = {
            executor.submit(verifier, str(path)): path for path in family_paths
        }
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()
            except Exception as error:
                result = failed(path, error)
            result.setdefault("family_id", path.name)
            results.append(result)
    return results


def reconcile_existing_pilot(
    output_root: str | Path,
    *,
    reconciliation_code_provenance: Mapping[str, Any],
    expected_family_count: int = 140,
    expected_generation_commit: str | None = None,
    workers: int = 12,
    verifier: Callable[[str | Path], dict[str, Any]] = verify_family_path,
) -> dict[str, Any]:
    """Re-verify an existing pilot and replace only its report-control files."""

    root = Path(output_root).resolve()
    canonical_path = root / "stage2_run_report.json"
    original_path = root / "stage2_run_report.broken_pipe_original.json"
    reconciliation_path = root / "reports" / "report_reconciliation_v1.json"
    if not canonical_path.is_file() and not original_path.is_file():
        raise FileNotFoundError(f"Missing Stage-2 run report under {root}")

    if not original_path.is_file():
        candidate = _load_json(canonical_path)
        exception = dict(candidate.get("exception") or {})
        if candidate.get("status") != "failed" or exception.get("type") != "BrokenPipeError":
            raise ValueError(
                "Canonical report is not the approved post-verification BrokenPipe failure"
            )
        _atomic_copy_bytes(canonical_path, original_path)
    original = _load_json(original_path)
    original_exception = dict(original.get("exception") or {})
    if original.get("status") != "failed" or original_exception.get("type") != "BrokenPipeError":
        raise ValueError("Preserved original report is not a BrokenPipe failure")

    started = time.perf_counter()
    started_at = _utc_now()
    family_root = root / "materialized" / "families"
    before_inventory = _family_inventory(family_root)
    plan_ids, plan_duplicate_ids = _plan_family_ids(root)
    materialized_ids = sorted(
        path.name for path in family_root.iterdir() if path.is_dir()
    ) if family_root.is_dir() else []
    original_planned = list(map(str, original.get("planned_family_ids") or []))
    original_materialized = list(
        map(str, original.get("materialized_family_ids") or [])
    )
    original_verified = list(map(str, original.get("verified_family_ids") or []))
    ledger = _ledger_gate(original, root)

    set_gate_failures: list[str] = []
    generation_commit = str(
        dict(original.get("code_provenance") or {}).get("code_commit", "")
    )
    if (
        expected_generation_commit is not None
        and generation_commit != expected_generation_commit
    ):
        set_gate_failures.append(
            "generation commit differs from explicitly approved commit: "
            f"{generation_commit} != {expected_generation_commit}"
        )
    expected_sets = {
        "plan": plan_ids,
        "original_planned": original_planned,
        "original_materialized": original_materialized,
        "original_verified": original_verified,
        "published": materialized_ids,
    }
    for name, ids in expected_sets.items():
        if len(ids) != expected_family_count:
            set_gate_failures.append(
                f"{name} count {len(ids)} != {expected_family_count}"
            )
        duplicates = _duplicate_ids(ids)
        if duplicates:
            set_gate_failures.append(f"{name} has duplicate IDs: {duplicates}")
    if plan_duplicate_ids:
        set_gate_failures.append(f"plan parquet has duplicate IDs: {plan_duplicate_ids}")
    reference = set(plan_ids)
    for name, ids in expected_sets.items():
        if set(ids) != reference:
            set_gate_failures.append(f"{name} ID set differs from generation plan")
    if ledger["runtime_decision_count"] != expected_family_count:
        set_gate_failures.append(
            "runtime decision count differs from expected family count"
        )
    if ledger["runtime_decision_duplicate_ids"]:
        set_gate_failures.append("runtime decisions contain duplicate family IDs")
    if set(ledger["runtime_decision_family_ids"]) != reference:
        set_gate_failures.append("runtime decision ID set differs from generation plan")

    verifications: list[dict[str, Any]] = []
    verifier_failures: list[str] = []
    if not set_gate_failures and ledger["passed"]:
        verifications = _verify_all(
            [family_root / family_id for family_id in plan_ids],
            workers=workers,
            verifier=verifier,
        )
        verifications.sort(key=lambda item: str(item.get("family_id")))
        verified_ids = [
            str(item.get("family_id"))
            for item in verifications
            if bool(item.get("passed"))
        ]
        if len(verifications) != expected_family_count:
            verifier_failures.append(
                f"verifier result count {len(verifications)} != {expected_family_count}"
            )
        if _duplicate_ids([str(item.get("family_id")) for item in verifications]):
            verifier_failures.append("verifier returned duplicate family IDs")
        if set(verified_ids) != reference:
            verifier_failures.append("verified PASS ID set differs from generation plan")
        for item in verifications:
            if not bool(item.get("passed")):
                verifier_failures.append(
                    f"family verifier failed: {item.get('family_id')}"
                )

    after_inventory = _family_inventory(family_root)
    changed_paths = sorted(
        set(before_inventory) ^ set(after_inventory)
        | {
            path
            for path in set(before_inventory) & set(after_inventory)
            if before_inventory[path] != after_inventory[path]
        }
    )
    family_artifacts_modified = bool(changed_paths)
    failures = [
        *set_gate_failures,
        *list(ledger["failures"]),
        *verifier_failures,
    ]
    if family_artifacts_modified:
        failures.append("family artifacts changed during read-only reconciliation")

    reconciliation_commit = str(
        reconciliation_code_provenance.get("code_commit", "")
    )
    passed = not failures
    reconciliation = {
        "schema": "cle_evrptw_stage2_report_reconciliation_v1",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "reconciled": passed,
        "reconciliation_reason": RECONCILIATION_REASON,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "generation_code_commit": generation_commit,
        "reconciliation_code_commit": reconciliation_commit,
        "reconciliation_code_provenance": dict(reconciliation_code_provenance),
        "original_report_status": original.get("status"),
        "original_exception": original_exception,
        "original_report_preserved_at": str(original_path),
        "canonical_report_replaced": False,
        "last_completed_stage": "verification",
        "expected_family_count": expected_family_count,
        "planned_count": len(plan_ids),
        "materialized_count": len(materialized_ids),
        "verified_count": sum(bool(item.get("passed")) for item in verifications),
        "planned_family_ids": plan_ids,
        "materialized_family_ids": materialized_ids,
        "verified_family_ids": sorted(
            str(item.get("family_id"))
            for item in verifications
            if bool(item.get("passed"))
        ),
        "set_gate_failures": set_gate_failures,
        "ledger_gate": ledger,
        "verifier_failures": verifier_failures,
        "verifications": verifications,
        "family_artifacts_modified": family_artifacts_modified,
        "family_artifact_changed_paths": changed_paths,
        "family_artifact_inventory_method": "path_size_mtime_ns_mode_no_hash",
        "failures": failures,
    }
    reconciliation["canonical_report_replaced"] = False
    atomic_write_json(reconciliation_path, reconciliation)
    if not passed:
        return reconciliation

    canonical = deepcopy(original)
    canonical.pop("exception", None)
    canonical.update(
        {
            "status": "passed",
            "passed": True,
            "last_completed_stage": "verification",
            "planned_count": expected_family_count,
            "materialized_count": expected_family_count,
            "verified_count": expected_family_count,
            "planned_family_ids": plan_ids,
            "materialized_family_ids": materialized_ids,
            "verified_family_ids": reconciliation["verified_family_ids"],
            "verified": verifications,
            "reconciled": True,
            "reconciliation_reason": RECONCILIATION_REASON,
            "generation_code_commit": generation_commit,
            "reconciliation_code_commit": reconciliation_commit,
            "original_report_status": original.get("status"),
            "original_exception": original_exception,
            "family_artifacts_modified": False,
            "run_state_before_reconciliation": (
                "generation_complete_but_report_control_failed"
            ),
            "reconciliation_report": str(reconciliation_path),
            "terminal_report_committed": True,
            "terminal_report_committed_at": _utc_now(),
        }
    )
    atomic_write_json(canonical_path, canonical)
    reconciliation["canonical_report_replaced"] = True
    atomic_write_json(reconciliation_path, reconciliation)
    return reconciliation
