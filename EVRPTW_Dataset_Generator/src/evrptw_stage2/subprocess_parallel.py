"""Production adapter for one-attempt-per-process Stage-2 materialization."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .planning import derive_seed
from .release_discipline import PilotStopController
from .runtime_supervisor import (
    FamilyProcessSpec,
    ProcessDecision,
    supervise_family_processes,
)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_rejection_ledger(output_root: Path, family_id: str) -> dict[str, Any]:
    path = output_root / "rejections" / f"{family_id}.json"
    if not path.is_file():
        return {"schema": "cle_evrptw_family_rejection_ledger_v2", "attempts": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_spec(
    task: Mapping[str, Any],
    *,
    attempt_number: int,
    run_id: str,
    python_executable: str,
    working_directory: Path,
) -> FamilyProcessSpec:
    family = dict(task["families"][0]["family"])
    family_id = str(family["family_id"])
    output_root = Path(task["output_root"])
    attempt_id = f"attempt-{attempt_number:03d}"
    runtime_dir = output_root / ".runtime" / run_id / family_id / attempt_id
    staging_root = output_root / ".inflight" / family_id / attempt_id
    result_path = runtime_dir / "result.json"
    heartbeat_path = runtime_dir / "heartbeat.json"
    envelope_path = runtime_dir / "envelope.json"
    envelope = {
        "schema": "cle_evrptw_family_process_envelope_v2",
        "runtime_contract_id": "family_process_timeout_and_abort_v2",
        "run_id": run_id,
        "family_id": family_id,
        "attempt_number": int(attempt_number),
        "attempt_id": attempt_id,
        "output_root": str(output_root),
        "staging_root": str(staging_root),
        "result_path": str(result_path),
        "heartbeat_path": str(heartbeat_path),
        "task": dict(task),
    }
    _atomic_write_json(envelope_path, envelope)
    return FamilyProcessSpec(
        run_id=run_id,
        family_id=family_id,
        attempt_id=attempt_id,
        attempt_number=int(attempt_number),
        city_slug=str(family["city_slug"]),
        track_id=str(family["track_id"]),
        day_type=str(family["day_type"]),
        scale_id=str(family["parent_scale_id"]),
        seed=derive_seed(
            int(family["family_seed"]), "materialization_attempt", attempt_number
        ),
        command=(
            python_executable,
            "-m",
            "evrptw_stage2.family_process_worker",
            "--envelope",
            str(envelope_path),
        ),
        cwd=str(working_directory),
        result_path=str(result_path),
        heartbeat_path=str(heartbeat_path),
        stdout_path=str(runtime_dir / "stdout.log"),
        stderr_path=str(runtime_dir / "stderr.log"),
        partial_artifact_path=str(staging_root),
        timeout_ledger_path=str(output_root / "timeouts" / f"{family_id}.json"),
    )


def run_supervised_materialization(
    tasks: list[dict[str, Any]],
    *,
    workers: int,
    max_attempts_per_family: int,
    family_wall_timeout_s: float,
    termination_grace_s: float,
    runner_exit_slack_s: float,
    pilot_controller: PilotStopController | None,
    python_executable: str | None = None,
    working_directory: Path | None = None,
    poll_interval_s: float = 1.0,
) -> dict[str, Any]:
    """Run exact family attempts under the killable runtime contract."""

    if any(len(task.get("families", [])) != 1 for task in tasks):
        raise ValueError("family_process_timeout_and_abort_v2 requires one family per task")
    if max_attempts_per_family <= 0:
        raise ValueError("max_attempts_per_family must be positive")
    now = datetime.now(timezone.utc)
    run_id = f"stage2-{now.strftime('%Y%m%dT%H%M%S.%fZ')}-pid{os.getpid()}"
    executable = str(python_executable or sys.executable)
    cwd = Path(working_directory or Path.cwd()).resolve()

    task_by_family: dict[str, dict[str, Any]] = {}
    initial_specs: list[FamilyProcessSpec] = []
    preclosed_unresolved: list[str] = []
    for task in tasks:
        family = dict(task["families"][0]["family"])
        family_id = str(family["family_id"])
        task_by_family[family_id] = task
        output_root = Path(task["output_root"])
        ledger = _load_rejection_ledger(output_root, family_id)
        attempts = list(ledger.get("attempts", []))
        retry_closed = any(item.get("retryable") is False for item in attempts)
        if retry_closed or len(attempts) >= max_attempts_per_family:
            preclosed_unresolved.append(family_id)
            continue
        initial_specs.append(
            _build_spec(
                task,
                attempt_number=len(attempts),
                run_id=run_id,
                python_executable=executable,
                working_directory=cwd,
            )
        )

    def on_normal_exit(spec: FamilyProcessSpec, returncode: int) -> ProcessDecision:
        result_path = Path(spec.result_path)
        if not result_path.is_file():
            return ProcessDecision(
                status="worker_failed_without_result",
                result={"family_id": spec.family_id, "returncode": returncode},
                hard_stop_reason={
                    "reason_code": "worker_failed_without_result",
                    "family_id": spec.family_id,
                    "returncode": returncode,
                },
            )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if returncode != 0 or "fatal_error" in payload:
            return ProcessDecision(
                status="worker_fatal_error",
                result=payload,
                hard_stop_reason={
                    "reason_code": "worker_fatal_error",
                    "family_id": spec.family_id,
                    "returncode": returncode,
                    "fatal_error": payload.get("fatal_error"),
                },
            )
        result = dict(payload["result"])
        unresolved = list(result.get("unresolved_family_ids", []))
        follow_up: FamilyProcessSpec | None = None
        if unresolved:
            ledger = _load_rejection_ledger(
                Path(task_by_family[spec.family_id]["output_root"]), spec.family_id
            )
            attempts = list(ledger.get("attempts", []))
            retryable = bool(attempts and attempts[-1].get("retryable", True))
            if retryable and len(attempts) < max_attempts_per_family:
                follow_up = _build_spec(
                    task_by_family[spec.family_id],
                    attempt_number=len(attempts),
                    run_id=run_id,
                    python_executable=executable,
                    working_directory=cwd,
                )

        observed = dict(result)
        if follow_up is not None:
            observed["unresolved_family_ids"] = []
        if pilot_controller is not None:
            pilot_controller.observe_chunk(observed)
            pilot_controller.poll(time.perf_counter())
        hard_stop = (
            pilot_controller.report()["stop_reasons"]
            if pilot_controller is not None and pilot_controller.stopped
            else None
        )
        if hard_stop:
            return ProcessDecision(
                status="hard_stop_after_result",
                result=result,
                hard_stop_reason={
                    "reason_code": "pilot_stop_controller",
                    "stop_reasons": hard_stop,
                },
            )
        if follow_up is not None:
            return ProcessDecision(
                status="retry_scheduled", result=result, follow_up=follow_up
            )
        return ProcessDecision(
            status="materialized" if result.get("materialized") else "unresolved",
            result=result,
        )

    supervision = supervise_family_processes(
        initial_specs,
        max_workers=workers,
        on_normal_exit=on_normal_exit,
        family_wall_timeout_s=family_wall_timeout_s,
        termination_grace_s=termination_grace_s,
        runner_exit_slack_s=runner_exit_slack_s,
        poll_interval_s=poll_interval_s,
    )
    materialized: list[dict[str, Any]] = []
    rejected_attempts: list[dict[str, Any]] = []
    unresolved = set(preclosed_unresolved)
    fatal_results: list[dict[str, Any]] = []
    for decision in supervision["decisions"]:
        result = dict(decision.get("result", {}))
        if "result" in result:
            result = dict(result["result"])
        materialized.extend(result.get("materialized", []))
        rejected_attempts.extend(result.get("rejected_attempts", []))
        if decision["status"] in {
            "unresolved",
            "hard_stop_after_result",
        }:
            unresolved.update(result.get("unresolved_family_ids", []))
        if decision["status"].startswith("worker_"):
            unresolved.add(str(decision["family_id"]))
            fatal_results.append(result)
    unresolved.update(supervision["unresolved_family_ids"])
    return {
        "schema": "cle_evrptw_supervised_materialization_result_v2",
        "run_id": run_id,
        "materialized": materialized,
        "rejected_attempts": rejected_attempts,
        "timed_out_attempts": supervision["timed_out"],
        "aborted_attempts": supervision["aborted"],
        "unresolved_family_ids": sorted(unresolved),
        "not_started_family_ids": supervision["cancelled_family_ids"],
        "fatal_results": fatal_results,
        "runtime_contract": supervision,
    }
