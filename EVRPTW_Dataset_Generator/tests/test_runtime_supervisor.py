from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from evrptw_stage2.runtime_supervisor import (
    RUNTIME_CONTRACT_ID,
    STOP_POLICY,
    FamilyProcessSpec,
    ProcessDecision,
    supervise_family_processes,
)


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "runtime_family_fixture.py"
DRIVER = ROOT / "tests" / "fixtures" / "runtime_supervisor_driver.py"


def _wait_for(path: Path, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for {path}")


def _pid_running(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    if not stat.is_file():
        return False
    fields = stat.read_text(encoding="utf-8").split()
    return len(fields) >= 3 and fields[2] != "Z"


def _spec(
    root: Path,
    family_id: str,
    *,
    sleep_s: float,
    ignore_term: bool = False,
    spawn_grandchild: bool = False,
) -> FamilyProcessSpec:
    runtime = root / "runtime" / family_id
    staging = root / ".inflight" / family_id / "attempt-000"
    command = [
        sys.executable,
        str(FIXTURE),
        "--sleep-s",
        str(sleep_s),
        "--started-marker",
        str(root / f"{family_id}.started.json"),
        "--heartbeat",
        str(runtime / "heartbeat.json"),
        "--partial",
        str(staging / "partial.json"),
        "--result",
        str(runtime / "worker-result.json"),
        "--completion-marker",
        str(root / "materialized" / "families" / family_id / "completed.json"),
        "--staging-completion-marker",
        str(
            staging / "materialized" / "families" / family_id / "family_manifest.json"
        ),
    ]
    if ignore_term:
        command.append("--ignore-term")
    if spawn_grandchild:
        command.extend(
            [
                "--spawn-grandchild",
                "--grandchild-pid",
                str(root / f"{family_id}.grandchild.json"),
            ]
        )
    return FamilyProcessSpec(
        run_id="fixture-run",
        family_id=family_id,
        attempt_id="attempt-000",
        attempt_number=0,
        city_slug="fixture-city",
        track_id="train",
        day_type="weekday",
        scale_id="cus50",
        seed=7,
        command=tuple(command),
        cwd=str(root),
        result_path=str(runtime / "worker-result.json"),
        heartbeat_path=str(runtime / "heartbeat.json"),
        stdout_path=str(runtime / "stdout.log"),
        stderr_path=str(runtime / "stderr.log"),
        partial_artifact_path=str(staging),
        timeout_ledger_path=str(root / "timeouts" / f"{family_id}.json"),
    )


def _complete(_spec: FamilyProcessSpec, returncode: int) -> ProcessDecision:
    return ProcessDecision(status="complete", result={"returncode": returncode})


def _run(
    specs: list[FamilyProcessSpec],
    *,
    workers: int = 1,
    timeout_s: float = 2.0,
    grace_s: float = 0.5,
    slack_s: float = 0.5,
    callback=_complete,
) -> dict[str, object]:
    return supervise_family_processes(
        specs,
        max_workers=workers,
        on_normal_exit=callback,
        family_wall_timeout_s=timeout_s,
        termination_grace_s=grace_s,
        runner_exit_slack_s=slack_s,
        poll_interval_s=0.02,
    )


def test_family_sleep_past_timeout_is_terminated_on_time(tmp_path: Path) -> None:
    started = time.monotonic()
    report = _run([_spec(tmp_path, "slow", sleep_s=30)])
    elapsed = time.monotonic() - started
    assert elapsed < 3.5
    assert report["hard_stop_triggered"]
    assert report["hard_stop_reason"]["reason_code"] == "family_wall_timeout"
    assert report["unresolved_family_ids"] == ["slow"]
    assert len(report["timed_out"]) == 1


def test_timeout_kills_grandchild_in_same_process_group(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        "tree",
        sleep_s=30,
        ignore_term=True,
        spawn_grandchild=True,
    )
    report = _run([spec])
    grandchild = json.loads((tmp_path / "tree.grandchild.json").read_text())["pid"]
    parent = json.loads((tmp_path / "tree.started.json").read_text())["pid"]
    assert not _pid_running(parent)
    assert not _pid_running(grandchild)
    assert report["remaining_process_group_count"] == 0


def test_supervisor_writes_complete_ledger_after_sigkill(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        "ledger",
        sleep_s=30,
        ignore_term=True,
        spawn_grandchild=True,
    )
    _run([spec])
    ledger = json.loads(Path(spec.timeout_ledger_path).read_text())
    assert ledger["runtime_contract_id"] == RUNTIME_CONTRACT_ID
    assert ledger["outcome"] == "timed_out"
    assert ledger["reason_code"] == "family_wall_timeout"
    assert ledger["retryable"] is False
    assert ledger["attempt_consumed"] is True
    assert ledger["retry_stopped_early"] is True
    assert ledger["deadline_timestamp"] > ledger["start_timestamp"]
    assert ledger["latest_heartbeat"]["stage"] == "fixture_sleep"
    assert ledger["termination"]["sigterm_sent"] is True
    assert ledger["termination"]["sigkill_sent"] is True
    assert ledger["termination"]["process_group_alive_after_cleanup"] is False


def test_timeout_partial_artifact_is_not_completed(tmp_path: Path) -> None:
    spec = _spec(tmp_path, "partial", sleep_s=30)
    _run([spec])
    partial = Path(spec.partial_artifact_path) / "partial.json"
    final = tmp_path / "materialized" / "families" / "partial" / "completed.json"
    assert partial.is_file()
    assert not final.exists()
    assert not (Path(spec.partial_artifact_path) / "completed.json").exists()
    staged_family = (
        Path(spec.partial_artifact_path) / "materialized" / "families" / "partial"
    )
    assert not (staged_family / "family_manifest.json").exists()
    assert (staged_family / "family_manifest.timeout_partial.json").is_file()
    ledger = json.loads(Path(spec.timeout_ledger_path).read_text())
    assert ledger["partial_completion_markers_quarantined"] == [
        str(staged_family / "family_manifest.timeout_partial.json")
    ]


def test_hard_stop_prevents_queued_family_from_starting(tmp_path: Path) -> None:
    first = _spec(tmp_path, "first", sleep_s=30)
    queued = _spec(tmp_path, "queued", sleep_s=0.1)
    report = _run([first, queued], workers=1)
    assert report["cancelled_family_ids"] == ["queued"]
    assert not (tmp_path / "queued.started.json").exists()
    assert report["hard_stop_reason"]["queue_state_at_trigger"] == {
        "queued_count": 1,
        "inflight_count": 1,
        "cancelled_count": 1,
    }


def test_other_inflight_family_is_aborted_after_grace(tmp_path: Path) -> None:
    trigger = _spec(tmp_path, "trigger", sleep_s=0.1)
    peer = _spec(tmp_path, "peer", sleep_s=30, ignore_term=True)

    def decision(spec: FamilyProcessSpec, returncode: int) -> ProcessDecision:
        if spec.family_id == "trigger":
            return ProcessDecision(
                status="hard_stop",
                result={"returncode": returncode},
                hard_stop_reason={"reason_code": "fixture_hard_gate"},
            )
        return _complete(spec, returncode)

    started = time.monotonic()
    report = _run(
        [trigger, peer], workers=2, timeout_s=20, grace_s=0.5, slack_s=0.5,
        callback=decision,
    )
    assert time.monotonic() - started < 2.0
    assert report["hard_stop_reason"]["reason_code"] == "fixture_hard_gate"
    assert [item["family_id"] for item in report["aborted"]] == ["peer"]


def test_runner_exits_within_timeout_grace_plus_slack(tmp_path: Path) -> None:
    timeout_s, grace_s, slack_s = 2.0, 0.5, 0.5
    started = time.monotonic()
    _run(
        [_spec(tmp_path, "bound", sleep_s=30, ignore_term=True)],
        timeout_s=timeout_s,
        grace_s=grace_s,
        slack_s=slack_s,
    )
    elapsed = time.monotonic() - started
    assert elapsed <= timeout_s + grace_s + slack_s + 0.75


def test_resume_does_not_retry_timed_out_family(tmp_path: Path) -> None:
    spec = _spec(tmp_path, "resume", sleep_s=30)
    _run([spec])
    marker = tmp_path / "resume.started.json"
    first_mtime = marker.stat().st_mtime_ns
    started = time.monotonic()
    report = _run([spec])
    assert time.monotonic() - started < 0.5
    assert marker.stat().st_mtime_ns == first_mtime
    assert report["skipped_prior_timeout_family_ids"] == ["resume"]
    assert report["unresolved_family_ids"] == ["resume"]


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_runner_signals_cleanup_complete_family_tree(
    tmp_path: Path, signum: signal.Signals
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    process = subprocess.Popen(
        [
            sys.executable,
            str(DRIVER),
            "--root",
            str(tmp_path),
            "--fixture",
            str(FIXTURE),
        ],
        env=env,
    )
    _wait_for(tmp_path / "started.json")
    _wait_for(tmp_path / "grandchild.json")
    os.kill(process.pid, signum)
    assert process.wait(timeout=8) == 0
    parent = json.loads((tmp_path / "started.json").read_text())["pid"]
    grandchild = json.loads((tmp_path / "grandchild.json").read_text())["pid"]
    assert not _pid_running(parent)
    assert not _pid_running(grandchild)
    report = json.loads((tmp_path / "supervisor_report.json").read_text())
    assert report["hard_stop_reason"]["reason_code"] == "runner_signal"
    assert report["remaining_process_group_count"] == 0


def test_normal_worker_exit_with_orphan_grandchild_is_hard_failure(tmp_path: Path) -> None:
    spec = _spec(tmp_path, "orphan", sleep_s=0.1, spawn_grandchild=True)
    report = _run([spec], timeout_s=20)
    child = json.loads((tmp_path / "orphan.grandchild.json").read_text())["pid"]
    assert report["hard_stop_reason"]["reason_code"] == (
        "orphan_process_group_after_worker_exit"
    )
    assert not _pid_running(child)
    assert report["remaining_process_group_count"] == 0


def test_runtime_contract_and_stop_policy_are_frozen(tmp_path: Path) -> None:
    report = _run([_spec(tmp_path, "contract", sleep_s=0.05)], timeout_s=5)
    assert report["runtime_contract_id"] == RUNTIME_CONTRACT_ID
    assert report["stop_policy"] == STOP_POLICY
    assert not report["hard_stop_triggered"]
