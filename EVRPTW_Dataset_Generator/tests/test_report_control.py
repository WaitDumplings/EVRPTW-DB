from __future__ import annotations

import importlib.util
import io
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from evrptw_stage2 import progress as progress_module
from evrptw_stage2.progress import Stage2ProgressWriter, atomic_write_json
from evrptw_stage2.reconciliation import reconcile_existing_pilot
from evrptw_stage2.runtime_supervisor import (
    FamilyProcessSpec,
    ProcessDecision,
    supervise_family_processes,
)


ROOT = Path(__file__).parents[1]
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "stage2_report_control_runner", ROOT / "scripts" / "build_stage2_instances.py"
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)
FIXTURE = ROOT / "tests" / "fixtures" / "runtime_family_fixture.py"
GENERATION_COMMIT = "1" * 40
RECONCILIATION_COMMIT = "2" * 40


class BrokenStream:
    def write(self, _value: str) -> None:
        raise BrokenPipeError(32, "synthetic closed stdout")

    def flush(self) -> None:
        raise AssertionError("flush must not run after write failed")


def _provenance() -> dict[str, object]:
    return {
        "schema": "evrptw_code_provenance_v1",
        "code_commit": RECONCILIATION_COMMIT,
        "code_branch": "stage2-repair-candidate",
        "working_tree_clean": True,
    }


def _pilot_fixture(tmp_path: Path, ids: tuple[str, ...] = ("f1", "f2")) -> Path:
    root = tmp_path / "pilot"
    plan_path = root / "generation_plan" / "core" / "train" / "family_index.parquet"
    plan_path.parent.mkdir(parents=True)
    pd.DataFrame({"family_id": list(ids)}).to_parquet(plan_path, index=False)
    family_root = root / "materialized" / "families"
    for family_id in ids:
        family_dir = family_root / family_id
        family_dir.mkdir(parents=True)
        (family_dir / "artifact.txt").write_text(
            f"immutable-{family_id}\n", encoding="utf-8"
        )
    report = {
        "schema": "cle_evrptw_stage2_run_report_v2",
        "status": "failed",
        "passed": False,
        "last_completed_stage": "verification",
        "exception": {"type": "BrokenPipeError", "message": "[Errno 32] Broken pipe"},
        "code_provenance": {"code_commit": GENERATION_COMMIT},
        "planned_family_ids": list(ids),
        "materialized_family_ids": list(ids),
        "verified_family_ids": list(ids),
        "materialized": [{"family_id": family_id} for family_id in ids],
        "verified": [
            {"family_id": family_id, "passed": True} for family_id in ids
        ],
        "rejected_attempts": [],
        "timed_out_attempts": [],
        "aborted_attempts": [],
        "not_started_family_ids": [],
        "unresolved_family_ids": [],
        "hard_stop_triggered": False,
        "runtime_contract": {
            "hard_stop_triggered": False,
            "timed_out": [],
            "aborted": [],
            "cancelled_family_ids": [],
            "skipped_prior_timeout_family_ids": [],
            "unresolved_family_ids": [],
            "remaining_process_group_count": 0,
            "decisions": [
                {"family_id": family_id, "status": "materialized"}
                for family_id in ids
            ],
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "stage2_run_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def _pass_verifier(path: str | Path) -> dict[str, object]:
    return {
        "family_id": Path(path).name,
        "passed": True,
        "errors": [],
        "warnings": [],
    }


def _fail_f2_verifier(path: str | Path) -> dict[str, object]:
    family_id = Path(path).name
    return {
        "family_id": family_id,
        "passed": family_id != "f2",
        "errors": [] if family_id != "f2" else ["synthetic verifier failure"],
        "warnings": [],
    }


def _reconcile(root: Path, verifier=_pass_verifier) -> dict[str, object]:
    return reconcile_existing_pilot(
        root,
        reconciliation_code_provenance=_provenance(),
        expected_family_count=2,
        expected_generation_commit=GENERATION_COMMIT,
        workers=1,
        verifier=verifier,
    )


def test_pass_report_committed_before_closed_stdout_cannot_be_downgraded(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "stage2_run_report.json"
    warning_path = tmp_path / "reports" / "observability_warnings.jsonl"
    report = {
        "status": "passed",
        "passed": True,
        "planned_family_ids": ["f1"],
        "materialized_family_ids": ["f1"],
        "verified_family_ids": ["f1"],
        "terminal_report_committed": True,
    }
    RUNNER._write_json(report_path, report)
    original = report_path.read_bytes()
    RUNNER._ACTIVE_RUN_REPORT = report
    RUNNER._ACTIVE_RUN_REPORT_PATH = report_path
    RUNNER._OBSERVABILITY_LEDGER_PATH = warning_path
    RUNNER._TERMINAL_REPORT_COMMITTED = True
    assert not RUNNER._emit_run_report(report, report_path, stream=BrokenStream())
    error = BrokenPipeError(32, "synthetic uncaught closed stdout")
    RUNNER._uncaught_run_report_hook(BrokenPipeError, error, None)
    assert report_path.read_bytes() == original
    assert json.loads(report_path.read_text())["status"] == "passed"


def test_broken_pipe_is_written_to_independent_observability_ledger(
    tmp_path: Path,
) -> None:
    warning_path = tmp_path / "reports" / "observability_warnings.jsonl"
    RUNNER._OBSERVABILITY_LEDGER_PATH = warning_path
    RUNNER._TERMINAL_REPORT_COMMITTED = True
    report = {"status": "passed"}
    assert not RUNNER._emit_run_report(
        report, tmp_path / "stage2_run_report.json", stream=BrokenStream()
    )
    warning = json.loads(warning_path.read_text().splitlines()[-1])
    assert warning["warning_type"] == "BrokenPipeError"
    assert warning["phase"] == "stdout_summary"
    assert warning["terminal_report_committed"] is True


def test_default_stdout_is_concise_and_full_json_requires_debug(tmp_path: Path) -> None:
    report = {
        "status": "passed",
        "planned_family_ids": ["f1"],
        "materialized_family_ids": ["f1"],
        "verified_family_ids": ["f1"],
        "large_private_field": "must-not-be-on-default-stdout",
    }
    concise = io.StringIO()
    assert RUNNER._emit_run_report(
        report, tmp_path / "stage2_run_report.json", stream=concise
    )
    assert "must-not-be-on-default-stdout" not in concise.getvalue()
    assert "planned=1 materialized=1 verified=1" in concise.getvalue()
    debug = io.StringIO()
    assert RUNNER._emit_run_report(
        report,
        tmp_path / "stage2_run_report.json",
        debug_full_report=True,
        stream=debug,
    )
    assert "must-not-be-on-default-stdout" in debug.getvalue()


def test_progress_counts_are_monotonic_after_each_family(tmp_path: Path) -> None:
    writer = Stage2ProgressWriter(tmp_path, ["f1", "f2"])
    snapshots = [json.loads(writer.path.read_text())]
    for family_id in ("f1", "f2"):
        writer.apply_supervisor_event(
            {
                "event_type": "family_started",
                "family_id": family_id,
                "status": "active",
                "active_family_ids": [family_id],
            }
        )
        writer.apply_supervisor_event(
            {
                "event_type": "family_terminal",
                "family_id": family_id,
                "status": "materialized",
                "result": {"materialized": [{"family_id": family_id}]},
                "active_family_ids": [],
            }
        )
        snapshots.append(json.loads(writer.path.read_text()))
        writer.record_verification(family_id, passed=True)
        snapshots.append(json.loads(writer.path.read_text()))
    writer.finalize(passed=True)
    snapshots.append(json.loads(writer.path.read_text()))
    assert [item["completed"] for item in snapshots] == sorted(
        item["completed"] for item in snapshots
    )
    assert [item["materialized"] for item in snapshots] == sorted(
        item["materialized"] for item in snapshots
    )
    assert [item["verified"] for item in snapshots] == sorted(
        item["verified"] for item in snapshots
    )
    assert snapshots[-1]["completed"] == 2
    assert snapshots[-1]["materialized"] == 2
    assert snapshots[-1]["status"] == "passed"
    assert snapshots[-1]["verified"] == 2


def test_interrupted_atomic_progress_write_preserves_previous_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "stage2_progress.json"
    atomic_write_json(path, {"completed": 1})
    original = path.read_bytes()

    def fail_replace(_source: str | Path, _destination: str | Path) -> None:
        raise OSError("synthetic interrupted replace")

    monkeypatch.setattr(progress_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic interrupted"):
        atomic_write_json(path, {"completed": 2})
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".stage2_progress.json.*.tmp"))


def test_reconciliation_generates_pass_for_exact_complete_pilot(
    tmp_path: Path,
) -> None:
    root = _pilot_fixture(tmp_path)
    result = _reconcile(root)
    canonical = json.loads((root / "stage2_run_report.json").read_text())
    assert result["passed"] is True
    assert result["planned_count"] == result["materialized_count"] == 2
    assert result["verified_count"] == 2
    assert canonical["status"] == "passed"
    assert canonical["reconciled"] is True
    assert canonical["terminal_report_committed"] is True


@pytest.mark.parametrize("failure_mode", ["missing", "verifier"])
def test_reconciliation_refuses_missing_or_failing_family(
    tmp_path: Path, failure_mode: str
) -> None:
    root = _pilot_fixture(tmp_path)
    if failure_mode == "missing":
        shutil.rmtree(root / "materialized" / "families" / "f2")
        verifier = _pass_verifier
    else:
        verifier = _fail_f2_verifier
    result = _reconcile(root, verifier=verifier)
    canonical = json.loads((root / "stage2_run_report.json").read_text())
    assert result["passed"] is False
    assert result["canonical_report_replaced"] is False
    assert canonical["status"] == "failed"


def test_reconciliation_preserves_original_red_report_exactly(tmp_path: Path) -> None:
    root = _pilot_fixture(tmp_path)
    original = (root / "stage2_run_report.json").read_bytes()
    result = _reconcile(root)
    assert result["passed"] is True
    assert (root / "stage2_run_report.broken_pipe_original.json").read_bytes() == original


def test_reconciliation_does_not_modify_family_artifacts(tmp_path: Path) -> None:
    root = _pilot_fixture(tmp_path)
    family_root = root / "materialized" / "families"
    before = {
        str(path.relative_to(family_root)): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in family_root.rglob("*")
        if path.is_file()
    }
    result = _reconcile(root)
    after = {
        str(path.relative_to(family_root)): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in family_root.rglob("*")
        if path.is_file()
    }
    assert result["family_artifacts_modified"] is False
    assert before == after


def test_closed_stdout_after_supervisor_completion_leaves_no_orphan(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    staging = tmp_path / ".inflight" / "f1" / "attempt-000"
    marker = tmp_path / "started.json"
    command = (
        sys.executable,
        str(FIXTURE),
        "--sleep-s",
        "0.05",
        "--started-marker",
        str(marker),
        "--heartbeat",
        str(runtime / "heartbeat.json"),
        "--partial",
        str(staging / "partial.json"),
        "--result",
        str(runtime / "worker-result.json"),
        "--completion-marker",
        str(tmp_path / "materialized" / "families" / "f1" / "completed.json"),
        "--staging-completion-marker",
        str(staging / "family_manifest.json"),
    )
    spec = FamilyProcessSpec(
        run_id="closed-stdout",
        family_id="f1",
        attempt_id="attempt-000",
        attempt_number=0,
        city_slug="fixture",
        track_id="train",
        day_type="weekday",
        scale_id="cus50",
        seed=1,
        command=command,
        cwd=str(tmp_path),
        result_path=str(runtime / "worker-result.json"),
        heartbeat_path=str(runtime / "heartbeat.json"),
        stdout_path=str(runtime / "stdout.log"),
        stderr_path=str(runtime / "stderr.log"),
        partial_artifact_path=str(staging),
        timeout_ledger_path=str(tmp_path / "timeouts" / "f1.json"),
    )
    supervised = supervise_family_processes(
        [spec],
        max_workers=1,
        on_normal_exit=lambda _spec, returncode: ProcessDecision(
            status="materialized", result={"returncode": returncode}
        ),
        family_wall_timeout_s=2,
        termination_grace_s=0.2,
        runner_exit_slack_s=0.2,
        poll_interval_s=0.02,
    )
    pid = int(json.loads(marker.read_text())["pid"])
    RUNNER._OBSERVABILITY_LEDGER_PATH = tmp_path / "observability.jsonl"
    RUNNER._TERMINAL_REPORT_COMMITTED = True
    assert not RUNNER._emit_run_report(
        {"status": "passed"}, tmp_path / "report.json", stream=BrokenStream()
    )
    assert supervised["remaining_process_group_count"] == 0
    assert not Path(f"/proc/{pid}").exists()
