"""Killable family-process supervision for Stage-2 materialization.

Each family attempt owns a new POSIX session/process group.  The supervisor,
not the worker, enforces wall-clock deadlines and persists timeout evidence.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


RUNTIME_CONTRACT_ID = "family_process_timeout_and_abort_v2"
STOP_POLICY = "abort_all_inflight_after_grace"
FAMILY_WALL_TIMEOUT_S = 7_200.0
TERMINATION_GRACE_S = 60.0
RUNNER_EXIT_SLACK_S = 30.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "unreadable", "path": str(path)}


@dataclass(frozen=True)
class FamilyProcessSpec:
    """One independently killable family attempt."""

    run_id: str
    family_id: str
    attempt_id: str
    attempt_number: int
    city_slug: str
    track_id: str
    day_type: str
    scale_id: str
    seed: int
    command: tuple[str, ...]
    cwd: str
    result_path: str
    heartbeat_path: str
    stdout_path: str
    stderr_path: str
    partial_artifact_path: str
    timeout_ledger_path: str
    environment: Mapping[str, str] | None = None


@dataclass(frozen=True)
class ProcessDecision:
    """Domain decision after a worker exits normally."""

    status: str
    result: Mapping[str, Any] | None = None
    follow_up: FamilyProcessSpec | None = None
    hard_stop_reason: Mapping[str, Any] | None = None


@dataclass
class _ActiveProcess:
    spec: FamilyProcessSpec
    process: subprocess.Popen[bytes]
    pid: int
    pgid: int
    started_monotonic: float
    deadline_monotonic: float
    started_at: str
    deadline_at: str
    stdout_handle: Any
    stderr_handle: Any
    termination_reason: str | None = None
    sigterm_sent_at_monotonic: float | None = None
    sigterm_sent_at: str | None = None
    sigkill_sent_at_monotonic: float | None = None
    sigkill_sent_at: str | None = None
    ledger: dict[str, Any] | None = None


@dataclass
class _SignalState:
    signum: int | None = None
    received_at: str | None = None


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_process_group(pgid: int, signum: int) -> bool:
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        return False
    return True


def _close_logs(active: _ActiveProcess) -> None:
    for handle in (active.stdout_handle, active.stderr_handle):
        if not handle.closed:
            handle.close()


def _base_termination_ledger(
    active: _ActiveProcess,
    *,
    outcome: str,
    reason_code: str,
    now_monotonic: float,
    global_stop_trigger: Mapping[str, Any],
    queued_count: int,
    inflight_count: int,
    cancelled_count: int,
) -> dict[str, Any]:
    trigger_queue_state = global_stop_trigger.get("queue_state_at_trigger")
    queue_state = (
        dict(trigger_queue_state)
        if isinstance(trigger_queue_state, Mapping)
        else {
            "queued_count": int(queued_count),
            "inflight_count": int(inflight_count),
            "cancelled_count": int(cancelled_count),
        }
    )
    return {
        "schema": "cle_evrptw_family_timeout_ledger_v2",
        "runtime_contract_id": RUNTIME_CONTRACT_ID,
        "stop_policy": STOP_POLICY,
        "run_id": active.spec.run_id,
        "family_id": active.spec.family_id,
        "attempt_id": active.spec.attempt_id,
        "attempt_number": int(active.spec.attempt_number),
        "city_slug": active.spec.city_slug,
        "track_id": active.spec.track_id,
        "day_type": active.spec.day_type,
        "scale_id": active.spec.scale_id,
        "seed": int(active.spec.seed),
        "pid": int(active.pid),
        "pgid": int(active.pgid),
        "start_timestamp": active.started_at,
        "deadline_timestamp": active.deadline_at,
        "deadline_monotonic": float(active.deadline_monotonic),
        "termination_event_timestamp": _utc_now(),
        "elapsed_seconds": float(now_monotonic - active.started_monotonic),
        "latest_heartbeat": _read_json_if_present(Path(active.spec.heartbeat_path)),
        "stdout_path": active.spec.stdout_path,
        "stderr_path": active.spec.stderr_path,
        "partial_artifact_path": active.spec.partial_artifact_path,
        "outcome": outcome,
        "reason_code": reason_code,
        "retryable": False,
        "attempt_consumed": True,
        "retry_stopped_early": True,
        "global_stop_trigger": dict(global_stop_trigger),
        "queue_state_at_trigger": queue_state,
        "termination": {
            "sigterm_sent": False,
            "sigterm_timestamp": None,
            "sigkill_sent": False,
            "sigkill_timestamp": None,
            "returncode": None,
            "process_group_alive_after_cleanup": None,
        },
    }


def _persist_ledger(active: _ActiveProcess) -> None:
    if active.ledger is not None:
        _atomic_write_json(Path(active.spec.timeout_ledger_path), active.ledger)


def _send_term(active: _ActiveProcess, now_monotonic: float) -> None:
    if active.sigterm_sent_at_monotonic is not None:
        return
    sent = _signal_process_group(active.pgid, signal.SIGTERM)
    active.sigterm_sent_at_monotonic = now_monotonic
    active.sigterm_sent_at = _utc_now()
    if active.ledger is not None:
        active.ledger["termination"]["sigterm_sent"] = bool(sent)
        active.ledger["termination"]["sigterm_timestamp"] = active.sigterm_sent_at
        _persist_ledger(active)


def _send_kill(active: _ActiveProcess, now_monotonic: float) -> None:
    if active.sigkill_sent_at_monotonic is not None:
        return
    sent = _signal_process_group(active.pgid, signal.SIGKILL)
    active.sigkill_sent_at_monotonic = now_monotonic
    active.sigkill_sent_at = _utc_now()
    if active.ledger is not None:
        active.ledger["termination"]["sigkill_sent"] = bool(sent)
        active.ledger["termination"]["sigkill_timestamp"] = active.sigkill_sent_at
        _persist_ledger(active)


def _quarantine_partial_completion_markers(active: _ActiveProcess) -> list[str]:
    """Prevent staged timeout artifacts from advertising completed status."""

    root = Path(active.spec.partial_artifact_path)
    quarantined: list[str] = []
    if not root.is_dir():
        return quarantined
    for marker in root.rglob("family_manifest.json"):
        payload = _read_json_if_present(marker)
        if not payload or payload.get("materialization_status") != "complete":
            continue
        destination = marker.with_name("family_manifest.timeout_partial.json")
        os.replace(marker, destination)
        quarantined.append(str(destination))
    return quarantined


def _finalize_terminated(active: _ActiveProcess) -> dict[str, Any]:
    returncode = active.process.poll()
    if returncode is None:
        try:
            returncode = active.process.wait(timeout=0)
        except subprocess.TimeoutExpired:
            returncode = None
    _close_logs(active)
    if active.ledger is None:
        raise RuntimeError("Terminated family is missing its supervisor ledger")
    active.ledger["termination"]["returncode"] = returncode
    active.ledger["termination"]["process_group_alive_after_cleanup"] = (
        _process_group_alive(active.pgid)
    )
    active.ledger["partial_completion_markers_quarantined"] = (
        _quarantine_partial_completion_markers(active)
    )
    active.ledger["completed_at"] = _utc_now()
    _persist_ledger(active)
    return active.ledger


def supervise_family_processes(
    specs: Sequence[FamilyProcessSpec],
    *,
    max_workers: int,
    on_normal_exit: Callable[[FamilyProcessSpec, int], ProcessDecision],
    family_wall_timeout_s: float = FAMILY_WALL_TIMEOUT_S,
    termination_grace_s: float = TERMINATION_GRACE_S,
    runner_exit_slack_s: float = RUNNER_EXIT_SLACK_S,
    poll_interval_s: float = 1.0,
    install_signal_handlers: bool = True,
) -> dict[str, Any]:
    """Run family attempts and abort the complete process tree on a hard stop."""

    if os.name != "posix":
        raise RuntimeError(f"{RUNTIME_CONTRACT_ID} requires POSIX process groups")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    for name, value in (
        ("family_wall_timeout_s", family_wall_timeout_s),
        ("termination_grace_s", termination_grace_s),
        ("runner_exit_slack_s", runner_exit_slack_s),
        ("poll_interval_s", poll_interval_s),
    ):
        if float(value) <= 0:
            raise ValueError(f"{name} must be positive")

    queued = deque(specs)
    active: dict[str, _ActiveProcess] = {}
    decisions: list[dict[str, Any]] = []
    timed_out: list[dict[str, Any]] = []
    aborted: list[dict[str, Any]] = []
    skipped_prior_timeout: list[str] = []
    cancelled: list[str] = []
    hard_stop_reason: dict[str, Any] | None = None
    hard_stop_monotonic: float | None = None
    signal_state = _SignalState()
    previous_handlers: dict[int, Any] = {}

    def receive_signal(signum: int, _frame: Any) -> None:
        if signal_state.signum is None:
            signal_state.signum = int(signum)
            signal_state.received_at = _utc_now()

    if install_signal_handlers:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, receive_signal)

    def trigger_hard_stop(reason: Mapping[str, Any], now_monotonic: float) -> None:
        nonlocal hard_stop_reason, hard_stop_monotonic
        if hard_stop_reason is not None:
            return
        queued_count = len(queued)
        hard_stop_reason = {
            **dict(reason),
            "queue_state_at_trigger": {
                "queued_count": queued_count,
                "inflight_count": len(active),
                "cancelled_count": queued_count,
            },
        }
        hard_stop_monotonic = now_monotonic
        while queued:
            cancelled.append(queued.popleft().family_id)

    def launch(spec: FamilyProcessSpec) -> None:
        ledger_path = Path(spec.timeout_ledger_path)
        if ledger_path.is_file():
            skipped_prior_timeout.append(spec.family_id)
            decisions.append(
                {
                    "family_id": spec.family_id,
                    "attempt_id": spec.attempt_id,
                    "status": "unresolved_prior_timeout_not_retried",
                }
            )
            return
        Path(spec.partial_artifact_path).mkdir(parents=True, exist_ok=True)
        Path(spec.stdout_path).parent.mkdir(parents=True, exist_ok=True)
        Path(spec.stderr_path).parent.mkdir(parents=True, exist_ok=True)
        stdout_handle = Path(spec.stdout_path).open("wb")
        stderr_handle = Path(spec.stderr_path).open("wb")
        environment = os.environ.copy()
        if spec.environment is not None:
            environment.update({str(key): str(value) for key, value in spec.environment.items()})
        started_monotonic = time.monotonic()
        try:
            process = subprocess.Popen(
                list(spec.command),
                cwd=spec.cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
            )
        except BaseException:
            stdout_handle.close()
            stderr_handle.close()
            raise
        pgid = os.getpgid(process.pid)
        if pgid != process.pid:
            process.terminate()
            raise RuntimeError("Family worker did not become its own process-group leader")
        started_at = datetime.now(timezone.utc)
        active[spec.family_id] = _ActiveProcess(
            spec=spec,
            process=process,
            pid=int(process.pid),
            pgid=int(pgid),
            started_monotonic=started_monotonic,
            deadline_monotonic=started_monotonic + float(family_wall_timeout_s),
            started_at=started_at.isoformat(),
            deadline_at=(started_at + timedelta(seconds=family_wall_timeout_s)).isoformat(),
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
        )

    try:
        while queued or active:
            now = time.monotonic()
            if signal_state.signum is not None and hard_stop_reason is None:
                trigger_hard_stop(
                    {
                        "reason_code": "runner_signal",
                        "signal": int(signal_state.signum),
                        "received_at": signal_state.received_at,
                    },
                    now,
                )
                for item in active.values():
                    if item.termination_reason is None:
                        item.termination_reason = "runner_signal"
                        item.ledger = _base_termination_ledger(
                            item,
                            outcome="aborted",
                            reason_code="runner_signal",
                            now_monotonic=now,
                            global_stop_trigger=hard_stop_reason,
                            queued_count=0,
                            inflight_count=len(active),
                            cancelled_count=len(cancelled),
                        )
                    _send_term(item, now)

            while hard_stop_reason is None and queued and len(active) < max_workers:
                before = len(active)
                launch(queued.popleft())
                if len(active) == before:
                    continue

            for family_id, item in list(active.items()):
                returncode = item.process.poll()
                group_alive = _process_group_alive(item.pgid)
                if item.termination_reason is None and returncode is not None:
                    if group_alive:
                        reason = {
                            "reason_code": "orphan_process_group_after_worker_exit",
                            "family_id": family_id,
                            "pgid": item.pgid,
                        }
                        trigger_hard_stop(reason, now)
                        item.termination_reason = "orphan_process_group_after_worker_exit"
                        item.ledger = _base_termination_ledger(
                            item,
                            outcome="aborted",
                            reason_code=item.termination_reason,
                            now_monotonic=now,
                            global_stop_trigger=hard_stop_reason,
                            queued_count=len(queued),
                            inflight_count=len(active),
                            cancelled_count=len(cancelled),
                        )
                        _send_term(item, now)
                        continue
                    _close_logs(item)
                    decision = on_normal_exit(item.spec, int(returncode))
                    decisions.append(
                        {
                            "family_id": family_id,
                            "attempt_id": item.spec.attempt_id,
                            "status": decision.status,
                            "result": dict(decision.result or {}),
                        }
                    )
                    del active[family_id]
                    if decision.hard_stop_reason is not None:
                        trigger_hard_stop(decision.hard_stop_reason, now)
                    elif decision.follow_up is not None:
                        queued.appendleft(decision.follow_up)
                    continue

                if item.termination_reason is None and now >= item.deadline_monotonic:
                    reason = {
                        "reason_code": "family_wall_timeout",
                        "family_id": family_id,
                        "attempt_id": item.spec.attempt_id,
                        "limit_seconds": float(family_wall_timeout_s),
                    }
                    trigger_hard_stop(reason, now)
                    item.termination_reason = "family_wall_timeout"
                    item.ledger = _base_termination_ledger(
                        item,
                        outcome="timed_out",
                        reason_code="family_wall_timeout",
                        now_monotonic=now,
                        global_stop_trigger=hard_stop_reason,
                        queued_count=len(queued),
                        inflight_count=len(active),
                        cancelled_count=len(cancelled),
                    )
                    _persist_ledger(item)
                    _send_term(item, now)

            if hard_stop_reason is not None and hard_stop_monotonic is not None:
                grace_deadline = hard_stop_monotonic + float(termination_grace_s)
                kill_deadline = grace_deadline + float(runner_exit_slack_s)
                for item in active.values():
                    if item.termination_reason == "family_wall_timeout":
                        if now >= grace_deadline:
                            _send_kill(item, now)
                        continue
                    if item.termination_reason is None and now >= grace_deadline:
                        item.termination_reason = "global_hard_stop"
                        item.ledger = _base_termination_ledger(
                            item,
                            outcome="aborted",
                            reason_code="global_hard_stop",
                            now_monotonic=now,
                            global_stop_trigger=hard_stop_reason,
                            queued_count=0,
                            inflight_count=len(active),
                            cancelled_count=len(cancelled),
                        )
                        _persist_ledger(item)
                        _send_term(item, now)
                    if item.termination_reason is not None and now >= kill_deadline:
                        _send_kill(item, now)

            for family_id, item in list(active.items()):
                if item.termination_reason is None:
                    continue
                if _process_group_alive(item.pgid):
                    continue
                ledger = _finalize_terminated(item)
                if item.termination_reason == "family_wall_timeout":
                    timed_out.append(ledger)
                else:
                    aborted.append(ledger)
                decisions.append(
                    {
                        "family_id": family_id,
                        "attempt_id": item.spec.attempt_id,
                        "status": ledger["outcome"],
                        "result": ledger,
                    }
                )
                del active[family_id]

            if queued or active:
                time.sleep(float(poll_interval_s))
    finally:
        for item in active.values():
            _send_kill(item, time.monotonic())
            try:
                item.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            _close_logs(item)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    unresolved = sorted(
        {
            item["family_id"]
            for item in decisions
            if item["status"]
            in {"timed_out", "aborted", "unresolved_prior_timeout_not_retried"}
        }
    )
    return {
        "schema": "cle_evrptw_family_process_supervision_v2",
        "runtime_contract_id": RUNTIME_CONTRACT_ID,
        "stop_policy": STOP_POLICY,
        "family_wall_timeout_s": float(family_wall_timeout_s),
        "termination_grace_s": float(termination_grace_s),
        "runner_exit_slack_s": float(runner_exit_slack_s),
        "hard_stop_triggered": hard_stop_reason is not None,
        "hard_stop_reason": hard_stop_reason,
        "decisions": decisions,
        "timed_out": timed_out,
        "aborted": aborted,
        "cancelled_family_ids": sorted(set(cancelled)),
        "skipped_prior_timeout_family_ids": sorted(set(skipped_prior_timeout)),
        "unresolved_family_ids": unresolved,
        "remaining_process_group_count": 0,
    }
