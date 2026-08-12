from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import socket
import uuid
import warnings
from pathlib import Path
from typing import Any

from evrptw_core.io import save_solution
from evrptw_core.schema import EVRPTWSolution

from benchmark_common import checkpoint_label


TIME_TRACE_FIELDNAMES = [
    "instance_id",
    "file",
    "family_id",
    "solver_name",
    "algorithm_profile_id",
    "seed",
    "seed_scheme",
    "run_contract_fingerprint",
    "checkpoint_s",
    "elapsed_s",
    "reached_checkpoint",
    "status",
    "benchmark_status",
    "has_incumbent",
    "first_feasible_time_s",
    "incumbent_event_time_s",
    "objective_distance_km",
    "vehicle_count",
    "routes_json",
    "route_sequence_json",
    "checkpoint_solution_path",
    "source",
    "errors",
]


def snapshot_rows(
    instance_id: str,
    source_info: dict[str, str],
    snapshots: list[dict[str, Any]],
    first_feasible_time_s: float | None,
    *,
    provenance: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    trace_provenance = provenance or {}
    return [
        {
            "instance_id": instance_id,
            "file": source_info.get("file", ""),
            "family_id": source_info.get("family_id", ""),
            "solver_name": trace_provenance.get("solver_name", ""),
            "algorithm_profile_id": trace_provenance.get(
                "algorithm_profile_id", ""
            ),
            "seed": trace_provenance.get("seed", ""),
            "seed_scheme": trace_provenance.get("seed_scheme", ""),
            "run_contract_fingerprint": trace_provenance.get(
                "run_contract_fingerprint", ""
            ),
            "checkpoint_s": snapshot["checkpoint_s"],
            "elapsed_s": snapshot["elapsed_s"],
            "reached_checkpoint": snapshot["reached_checkpoint"],
            "status": snapshot["status"],
            "benchmark_status": snapshot["benchmark_status"],
            "has_incumbent": snapshot["has_incumbent"],
            "first_feasible_time_s": "" if first_feasible_time_s is None else first_feasible_time_s,
            "incumbent_event_time_s": (
                "" if snapshot["incumbent_event_time_s"] is None else snapshot["incumbent_event_time_s"]
            ),
            "objective_distance_km": (
                "" if snapshot["objective_distance_km"] is None else snapshot["objective_distance_km"]
            ),
            "vehicle_count": "" if snapshot["vehicle_count"] is None else snapshot["vehicle_count"],
            "routes_json": json.dumps(snapshot["routes"]),
            "route_sequence_json": json.dumps(snapshot["route_sequence"]),
            "checkpoint_solution_path": "",
            "source": snapshot["source"],
            "errors": "",
        }
        for snapshot in snapshots
    ]


def error_snapshot_rows(
    instance_id: str,
    source_info: dict[str, str],
    checkpoints_s: tuple[float, ...],
    status: str,
    errors: str,
    *,
    provenance: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    trace_provenance = provenance or {}
    return [
        {
            "instance_id": instance_id,
            "file": source_info.get("file", ""),
            "family_id": source_info.get("family_id", ""),
            "solver_name": trace_provenance.get("solver_name", ""),
            "algorithm_profile_id": trace_provenance.get(
                "algorithm_profile_id", ""
            ),
            "seed": trace_provenance.get("seed", ""),
            "seed_scheme": trace_provenance.get("seed_scheme", ""),
            "run_contract_fingerprint": trace_provenance.get(
                "run_contract_fingerprint", ""
            ),
            "checkpoint_s": checkpoint,
            "elapsed_s": 0.0,
            "reached_checkpoint": False,
            "status": status,
            "benchmark_status": status,
            "has_incumbent": False,
            "first_feasible_time_s": "",
            "incumbent_event_time_s": "",
            "objective_distance_km": "",
            "vehicle_count": "",
            "routes_json": "[]",
            "route_sequence_json": "[]",
            "checkpoint_solution_path": "",
            "source": "error",
            "errors": errors,
        }
        for checkpoint in checkpoints_s
    ]


def save_result_artifacts(
    result: dict[str, Any],
    *,
    solver_name: str,
    solutions_dir: Path,
    checkpoints_dir: Path,
) -> None:
    summary_row = result["summary_row"]
    contract_namespace = _contract_artifact_namespace(
        summary_row.get("run_contract_fingerprint", "")
    )
    solution_dict = result.get("solution")
    if solution_dict is not None:
        solution = EVRPTWSolution.from_dict(solution_dict)
        path = (
            solutions_dir
            / contract_namespace
            / f"{solution.instance_id}_solution.pkl"
        )
        atomic_save_solution(path, solution)
        result["summary_row"]["solution_path"] = str(path)

    for row, snapshot in zip(result.get("time_rows", []), result.get("snapshots", [])):
        if not snapshot.get("has_incumbent"):
            continue
        checkpoint_solution = EVRPTWSolution(
            instance_id=result["summary_row"]["instance_id"],
            solver_name=solver_name,
            routes=[list(route) for route in snapshot["routes"]],
            objective_distance_km=float(snapshot["objective_distance_km"]),
            vehicle_count=int(snapshot["vehicle_count"]),
            runtime_s=float(snapshot["elapsed_s"]),
            feasible=True,
            metadata={
                "checkpoint_s": snapshot["checkpoint_s"],
                "reached_checkpoint": snapshot["reached_checkpoint"],
                "incumbent_event_time_s": snapshot["incumbent_event_time_s"],
                "source": snapshot["source"],
                "benchmark_status": snapshot["benchmark_status"],
                "seed": summary_row.get("seed", ""),
                "seed_scheme": summary_row.get("seed_scheme", ""),
                "algorithm_profile_id": summary_row.get(
                    "algorithm_profile_id", ""
                ),
                "run_contract_fingerprint": summary_row.get(
                    "run_contract_fingerprint", ""
                ),
                "run_contract_json": summary_row.get("run_contract_json", ""),
            },
        )
        path = checkpoints_dir / contract_namespace / (
            f"{checkpoint_solution.instance_id}_"
            f"{checkpoint_label(snapshot['checkpoint_s'])}_solution.pkl"
        )
        atomic_save_solution(path, checkpoint_solution)
        row["checkpoint_solution_path"] = str(path)


def _contract_artifact_namespace(fingerprint: Any) -> str:
    """Return a path-safe namespace for one semantic run contract."""

    raw = str(fingerprint).strip().lower()
    if len(raw) == 64 and all(character in "0123456789abcdef" for character in raw):
        return raw
    if not raw:
        return "unversioned"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_save_solution(path: Path, solution: EVRPTWSolution) -> None:
    """Write a solution pickle beside its target and atomically publish it."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        save_solution(temporary, solution)
        temporary.replace(target)
    finally:
        # Preserve a previously published target when serialization fails and
        # avoid accumulating abandoned temporary artifacts.
        if temporary.exists():
            temporary.unlink()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    temporary.replace(path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _journal_json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"cannot serialize {type(value).__name__} in runner journal")


class IncrementalCsvStore:
    """Crash-recoverable result journal with canonical CSV materialization.

    Each completed instance is appended once to a JSONL journal and applied to
    O(1) in-memory indexes.  The potentially large canonical CSV files are
    written only at explicit flush points, avoiding the previous O(N^2)
    rewrite pattern.  A malformed final journal line (for example, process
    termination during append) is ignored while all complete records survive.

    Existing CSV-only result directories remain valid.  With ``resume=True``
    they are loaded first and then overlaid by newer journal records.
    """

    VERSION = 1

    def __init__(
        self,
        *,
        summary_path: Path,
        trace_path: Path,
        summary_fieldnames: list[str],
        trace_fieldnames: list[str],
        solver_key: str,
        resume: bool,
    ) -> None:
        self.summary_path = Path(summary_path)
        self.trace_path = Path(trace_path)
        self.summary_fieldnames = list(summary_fieldnames)
        self.trace_fieldnames = list(trace_fieldnames)
        self.journal_path = (
            self.summary_path.parent
            / ".runner_state"
            / f"{str(solver_key)}_results.jsonl"
        )
        self.lock_path = self.summary_path.parent / ".runner_state" / "active.lock"
        self._lock_handle = None
        self._acquire_save_path_lock(str(solver_key))
        self._summary_by_instance: dict[str, dict[str, Any]] = {}
        self._trace_by_key: dict[tuple[str, float], dict[str, Any]] = {}
        self._trace_keys_by_instance: dict[str, set[tuple[str, float]]] = {}
        self.records_since_flush = 0
        self._warned_legacy_contract = False

        if resume:
            for row in read_csv_rows(self.summary_path):
                instance_id = str(row.get("instance_id", ""))
                if instance_id:
                    self._summary_by_instance[instance_id] = row
            for row in read_csv_rows(self.trace_path):
                key = self._time_key(row)
                if key is not None:
                    self._trace_by_key[key] = row
                    self._trace_keys_by_instance.setdefault(key[0], set()).add(key)
            self._recover_journal()
        else:
            # The reset marker tells a later --skip_completed recovery not to
            # merge a stale canonical CSV from a previous fresh run.
            self._replace_journal([{"record_type": "reset", "version": self.VERSION}])

    def _acquire_save_path_lock(self, solver_key: str) -> None:
        """Reject concurrent writers to one save directory.

        Summary filenames differ between algorithms, but their solution
        artifact names do not.  A single directory therefore has to belong to
        one active runner.  ``flock`` is automatically released by the kernel
        after normal exit, exceptions, or SIGKILL, unlike a stale PID marker.
        """

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            handle.close()
            message = (
                f"save_path {self.summary_path.parent} already has an active "
                f"benchmark runner ({owner}); use a unique --save_path"
            )
            warnings.warn(message, RuntimeWarning)
            raise RuntimeError(message) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "solver_key": solver_key,
                    "summary_path": str(self.summary_path),
                },
                sort_keys=True,
            )
        )
        handle.flush()
        os.fsync(handle.fileno())
        self._lock_handle = handle

    def close(self) -> None:
        handle = self._lock_handle
        if handle is None:
            return
        self._lock_handle = None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> "IncrementalCsvStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _time_key(row: dict[str, Any]) -> tuple[str, float] | None:
        instance_id = str(row.get("instance_id", ""))
        if not instance_id:
            return None
        try:
            checkpoint = float(row.get("checkpoint_s", 0.0))
        except (TypeError, ValueError):
            return None
        return instance_id, checkpoint

    def _replace_journal(self, records: list[dict[str, Any]]) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.journal_path.with_suffix(self.journal_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        default=_journal_json_default,
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.journal_path)

    def _apply_record(self, record: dict[str, Any]) -> None:
        record_type = record.get("record_type")
        if record_type == "reset":
            self._summary_by_instance.clear()
            self._trace_by_key.clear()
            self._trace_keys_by_instance.clear()
            return
        if record_type != "result":
            return
        summary = dict(record.get("summary_row", {}))
        instance_id = str(summary.get("instance_id", record.get("instance_id", "")))
        if not instance_id:
            return
        summary["instance_id"] = instance_id
        self._summary_by_instance[instance_id] = summary
        for key in self._trace_keys_by_instance.pop(instance_id, set()):
            self._trace_by_key.pop(key, None)
        for raw_row in record.get("time_rows", []):
            row = dict(raw_row)
            row["instance_id"] = instance_id
            key = self._time_key(row)
            if key is not None:
                self._trace_by_key[key] = row
                self._trace_keys_by_instance.setdefault(instance_id, set()).add(key)

    def _recover_journal(self) -> None:
        if not self.journal_path.is_file():
            return
        with self.journal_path.open(encoding="utf-8") as handle:
            lines = list(handle)
        nonempty_indices = [index for index, line in enumerate(lines) if line.strip()]
        if not nonempty_indices:
            return
        last_nonempty_index = nonempty_indices[-1]
        valid_records: list[dict[str, Any]] = []
        repair_incomplete_tail = False
        for line_index in nonempty_indices:
            line_number = line_index + 1
            try:
                record = json.loads(lines[line_index])
            except json.JSONDecodeError as exc:
                if line_index != last_nonempty_index:
                    raise ValueError(
                        f"malformed runner journal record before the final line: "
                        f"{self.journal_path}:{line_number}"
                    ) from exc
                warnings.warn(
                    f"repairing incomplete runner journal tail "
                    f"{self.journal_path}:{line_number}",
                    RuntimeWarning,
                )
                repair_incomplete_tail = True
                break
            if not isinstance(record, dict):
                raise ValueError(
                    f"runner journal record must be an object: "
                    f"{self.journal_path}:{line_number}"
                )
            if int(record.get("version", self.VERSION)) != self.VERSION:
                raise ValueError(
                    f"unsupported runner journal version in {self.journal_path}"
                )
            valid_records.append(record)

        for record in valid_records:
            self._apply_record(record)
        if repair_incomplete_tail:
            # Leaving the partial bytes in place would make the next append
            # join the broken JSON token.  A second crash/restart would then
            # discard that newly completed result as part of the same bad
            # tail.  Atomically publish only complete records before allowing
            # any future append.
            self._replace_journal(valid_records)

    def completed_instance_ids(
        self,
        statuses: set[str],
        run_contract_fingerprints: dict[str, str] | None = None,
    ) -> set[str]:
        """Return terminal views whose stored run contract exactly matches.

        Legacy CSVs do not carry a semantic run identity, so they are never
        assumed compatible with a new invocation.  This prevents a short pilot
        or an old seed/profile from silently satisfying a formal run.
        """

        expected = run_contract_fingerprints or {}
        matched: set[str] = set()
        saw_legacy_terminal = False
        for instance_id, fingerprint in expected.items():
            row = self._summary_by_instance.get(str(instance_id))
            if row is None or str(row.get("status", "")) not in statuses:
                continue
            stored = str(row.get("run_contract_fingerprint", "")).strip()
            if not stored:
                saw_legacy_terminal = True
                continue
            if stored == str(fingerprint):
                matched.add(str(instance_id))
        if saw_legacy_terminal and not self._warned_legacy_contract:
            warnings.warn(
                "terminal legacy results without run_contract_fingerprint "
                "cannot be safely resumed and will be rerun",
                RuntimeWarning,
            )
            self._warned_legacy_contract = True
        return matched

    def has_completed_contract(
        self,
        instance_id: str,
        fingerprint: str,
        statuses: set[str],
    ) -> bool:
        return str(instance_id) in self.completed_instance_ids(
            statuses,
            {str(instance_id): str(fingerprint)},
        )

    def terminal_instance_ids(self, statuses: set[str]) -> set[str]:
        """Diagnostic-only status lookup that deliberately ignores contracts."""

        return {
            instance_id
            for instance_id, row in self._summary_by_instance.items()
            if str(row.get("status", "")) in statuses
        }

    def record_result(self, result: dict[str, Any]) -> None:
        summary = dict(result["summary_row"])
        instance_id = str(summary["instance_id"])
        record = {
            "record_type": "result",
            "version": self.VERSION,
            "instance_id": instance_id,
            "summary_row": summary,
            "time_rows": [dict(row) for row in result.get("time_rows", [])],
        }
        self._apply_record(record)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    record,
                    default=_journal_json_default,
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        self.records_since_flush += 1

    @property
    def summary_rows(self) -> list[dict[str, Any]]:
        return sorted(
            self._summary_by_instance.values(),
            key=lambda row: str(row.get("instance_id", "")),
        )

    @property
    def time_rows(self) -> list[dict[str, Any]]:
        return sorted(
            self._trace_by_key.values(),
            key=lambda row: (
                str(row.get("instance_id", "")),
                float(row.get("checkpoint_s", 0.0)),
            ),
        )

    def flush_canonical(self) -> None:
        # Keep the journal intact until both atomic CSV replacements succeed.
        # It is therefore sufficient for recovery if the process dies between
        # these two writes.
        write_csv(self.summary_path, self.summary_rows, self.summary_fieldnames)
        write_csv(self.trace_path, self.time_rows, self.trace_fieldnames)
        self._replace_journal([])
        self.records_since_flush = 0
