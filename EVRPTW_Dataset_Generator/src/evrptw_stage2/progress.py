"""Small atomic progress snapshots for long Stage-2 runs."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace one JSON file without exposing a partial document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def append_json_event(path: Path, payload: Mapping[str, Any]) -> None:
    """Append and fsync one compact observability event."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, ensure_ascii=False))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


class Stage2ProgressWriter:
    """Maintain monotonic family-level progress outside the large run report."""

    def __init__(
        self,
        output_root: Path,
        planned_family_ids: list[str],
        *,
        write_initial: bool = True,
    ) -> None:
        self.path = output_root / "stage2_progress.json"
        self.events_path = output_root / "stage2_progress_events.jsonl"
        self.planned_ids = set(map(str, planned_family_ids))
        self.completed_ids: set[str] = set()
        self.materialized_ids: set[str] = set()
        self.verified_ids: set[str] = set()
        self.rejected_attempt_count = 0
        self.timed_out_ids: set[str] = set()
        self.aborted_ids: set[str] = set()
        self.unresolved_ids: set[str] = set()
        self.active_ids: set[str] = set()
        self.last_completed_family_id: str | None = None
        self.status = "planned"
        if write_initial:
            self._persist("progress_initialized")

    def _snapshot(self) -> dict[str, Any]:
        return {
            "schema": "cle_evrptw_stage2_progress_v1",
            "status": self.status,
            "planned": len(self.planned_ids),
            "completed": len(self.completed_ids),
            "materialized": len(self.materialized_ids),
            "verified": len(self.verified_ids),
            "rejected": int(self.rejected_attempt_count),
            "timed_out": len(self.timed_out_ids),
            "aborted": len(self.aborted_ids),
            "unresolved": len(self.unresolved_ids),
            "active_family_ids": sorted(self.active_ids),
            "not_started": len(
                self.planned_ids - self.completed_ids - self.active_ids
            ),
            "last_completed_family_id": self.last_completed_family_id,
            "updated_at": _utc_now(),
        }

    def _persist(self, event_type: str, **details: Any) -> None:
        snapshot = self._snapshot()
        atomic_write_json(self.path, snapshot)
        append_json_event(
            self.events_path,
            {
                "schema": "cle_evrptw_stage2_progress_event_v1",
                "event_type": event_type,
                "updated_at": snapshot["updated_at"],
                "completed": snapshot["completed"],
                "materialized": snapshot["materialized"],
                "verified": snapshot["verified"],
                "active_family_ids": snapshot["active_family_ids"],
                **details,
            },
        )

    def apply_supervisor_event(self, event: Mapping[str, Any]) -> None:
        family_id = str(event.get("family_id", ""))
        self.active_ids = set(map(str, event.get("active_family_ids", [])))
        event_type = str(event.get("event_type", "supervisor_event"))
        self.status = "materializing"
        if event_type == "family_terminal":
            status = str(event.get("status", "unknown"))
            result = dict(event.get("result") or {})
            if "result" in result and isinstance(result["result"], Mapping):
                result = dict(result["result"])
            if status != "retry_scheduled":
                self.completed_ids.add(family_id)
                self.last_completed_family_id = family_id
            if result.get("materialized"):
                self.materialized_ids.add(family_id)
            self.rejected_attempt_count += len(result.get("rejected_attempts", []))
            if status == "timed_out":
                self.timed_out_ids.add(family_id)
            if status == "aborted":
                self.aborted_ids.add(family_id)
            if status in {
                "unresolved",
                "timed_out",
                "aborted",
                "hard_stop_after_result",
                "worker_failed_without_result",
                "worker_fatal_error",
                "unresolved_prior_timeout_not_retried",
            }:
                self.unresolved_ids.add(family_id)
        self._persist(event_type, family_id=family_id, status=event.get("status"))

    def record_verification(self, family_id: str, *, passed: bool) -> None:
        family_id = str(family_id)
        self.status = "verifying"
        if passed:
            self.verified_ids.add(family_id)
        else:
            self.unresolved_ids.add(family_id)
        self.last_completed_family_id = family_id
        self._persist("family_verified", family_id=family_id, passed=bool(passed))

    def finalize(self, *, passed: bool) -> None:
        self.status = "passed" if passed else "failed"
        self.active_ids.clear()
        self._persist("progress_finalized", passed=bool(passed))
