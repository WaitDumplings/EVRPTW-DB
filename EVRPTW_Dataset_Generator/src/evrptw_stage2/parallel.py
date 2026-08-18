"""Spawn-safe Stage-2 materialization and verification workers."""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .amazon import load_amazon_stage2_artifacts
from .artifacts import verify_materialized_family
from .config import load_stage2_config
from .materialize import materialize_family
from .planning import derive_seed, materialization_attempt_inputs
from .profile import load_reference_profile
from .reader import load_portable_cle


def _process_peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_heartbeat(
    path: Path | None,
    *,
    family_id: str,
    attempt_number: int,
    stage: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    if path is None:
        return
    _write_json(
        path,
        {
            "schema": "cle_evrptw_family_heartbeat_v1",
            "family_id": family_id,
            "attempt_number": int(attempt_number),
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "stage": stage,
            "details": dict(details or {}),
            "monotonic_seconds": time.monotonic(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


def classify_rejection(error: Exception) -> tuple[str, str]:
    message = str(error)
    prefix = message.split(":", 1)[0].strip()
    if prefix and prefix.replace("_", "").isalnum() and prefix.upper() == prefix:
        return prefix.lower(), message
    return type(error).__name__.lower(), message


def worker_error_is_fatal(error: Exception) -> bool:
    """Treat programming/runtime faults as fatal, never as seed rejections."""

    return isinstance(
        error,
        (
            AssertionError,
            AttributeError,
            ImportError,
            IndexError,
            KeyError,
            MemoryError,
            NameError,
            OSError,
            TypeError,
        ),
    )


def rejection_is_retryable(error: Exception) -> bool:
    """Honor explicit contract failures while retrying stochastic rejections."""

    return not worker_error_is_fatal(error) and bool(
        getattr(error, "retryable", True)
    )


def remaining_attempt_numbers(
    recorded_attempt_count: int,
    max_attempts_per_family: int,
    *,
    retry_closed: bool = False,
) -> range:
    """Return the unspent attempt numbers under the lifetime family cap."""

    if recorded_attempt_count < 0:
        raise ValueError("recorded_attempt_count must be non-negative")
    if max_attempts_per_family <= 0:
        raise ValueError("max_attempts_per_family must be positive")
    return range(0, 0) if retry_closed else range(recorded_attempt_count, max_attempts_per_family)


def materialize_family_chunk(task: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize one single-city chunk while reusing its routing topology."""

    heartbeat_path = (
        Path(str(task["heartbeat_path"])) if task.get("heartbeat_path") else None
    )
    heartbeat_family = str(task["families"][0]["family"]["family_id"])
    _write_heartbeat(
        heartbeat_path,
        family_id=heartbeat_family,
        attempt_number=int(task.get("process_attempt_number", 0)),
        stage="worker_start",
    )
    config = load_stage2_config(task["config_path"])
    profile = load_reference_profile(
        task["profile_path"],
        official=str(task["mode"]) == "official",
    )
    city = str(task["city_slug"])
    cle = load_portable_cle(task["cle_root"], city, mode=str(task["mode"]))
    output_root = Path(task["output_root"])
    customer_split_path = Path(task["customer_split_path"])
    community_adjacency_path = Path(task["community_adjacency_path"])
    amazon_artifacts = load_amazon_stage2_artifacts(
        task["amazon_artifact_root"],
        cohort_split_path=task["amazon_cohort_split_path"],
    )
    max_attempts = int(task["max_attempts_per_family"])
    final_materialized_root = Path(
        task.get("final_materialized_root", output_root / "materialized")
    )
    work_materialized_root = Path(
        task.get("materialized_output_root", final_materialized_root)
    )
    topology_cache = {}
    adjacency_cache = {}
    result: dict[str, Any] = {
        "chunk_id": str(task["chunk_id"]),
        "city_slug": city,
        "materialized": [],
        "rejected_attempts": [],
        "unresolved_family_ids": [],
    }

    for item in task["families"]:
        family = dict(item["family"])
        family_id = str(family["family_id"])
        views = pd.DataFrame(item["views"])
        family_dir = final_materialized_root / "families" / family_id
        if family_dir.is_dir():
            verification_started = time.perf_counter()
            verification = verify_materialized_family(family_dir)
            verification_seconds = time.perf_counter() - verification_started
            if not verification["passed"]:
                raise ValueError(f"Existing family {family_id} failed verification")
            result["materialized"].append(
                {
                    "family_id": family_id,
                    "status": "reused_verified",
                    "verification_seconds": verification_seconds,
                    "matrix_total_bytes": int(verification["matrix_total_bytes"]),
                    "worker_peak_rss_bytes_after": _process_peak_rss_bytes(),
                }
            )
            continue

        rejection_path = output_root / "rejections" / f"{family_id}.json"
        rejection_payload = (
            json.loads(rejection_path.read_text(encoding="utf-8"))
            if rejection_path.is_file()
            else {"schema": "cle_evrptw_family_rejection_ledger_v2", "attempts": []}
        )
        first_attempt_number = len(rejection_payload["attempts"])
        retry_closed = any(
            attempt.get("retryable") is False
            for attempt in rejection_payload["attempts"]
        )
        completed = False
        for attempt_number in remaining_attempt_numbers(
            first_attempt_number,
            max_attempts,
            retry_closed=retry_closed,
        ):
            progress_events: list[dict[str, Any]] = []
            _write_heartbeat(
                heartbeat_path,
                family_id=family_id,
                attempt_number=attempt_number,
                stage="attempt_start",
                details={"city_slug": city},
            )

            def progress(stage: str, details: Mapping[str, Any]) -> None:
                event = {"stage": stage, **dict(details)}
                progress_events.append(event)
                _write_heartbeat(
                    heartbeat_path,
                    family_id=family_id,
                    attempt_number=attempt_number,
                    stage=stage,
                    details={
                        "current_event": event,
                        "events": progress_events,
                    },
                )

            attempt_family, attempt_views = materialization_attempt_inputs(
                family,
                views,
                attempt_number=attempt_number,
            )
            started = time.perf_counter()
            try:
                manifest = materialize_family(
                    cle,
                    config=config,
                    profile=profile,
                    family=attempt_family,
                    views=attempt_views,
                    customer_split_path=customer_split_path,
                    community_adjacency_path=community_adjacency_path,
                    amazon_artifacts=amazon_artifacts,
                    output_root=work_materialized_root,
                    routing_topology_cache=topology_cache,
                    community_adjacency_cache=adjacency_cache,
                    code_provenance=task.get("code_provenance"),
                    progress_callback=progress,
                )
            except Exception as error:  # noqa: BLE001 - persist every failed attempt.
                if worker_error_is_fatal(error):
                    raise
                reason_code, reason_detail = classify_rejection(error)
                retryable = rejection_is_retryable(error)
                rejection = {
                    "family_id": family_id,
                    "city_slug": city,
                    "attempt_number": attempt_number,
                    "attempt_seed": int(attempt_family["materialization_attempt_seed"]),
                    "next_attempt_seed": (
                        derive_seed(
                            int(family["family_seed"]),
                            "materialization_attempt",
                            attempt_number + 1,
                        )
                        if retryable
                        else None
                    ),
                    "error_type": type(error).__name__,
                    "reason_code": reason_code,
                    "reason": reason_detail,
                    "retryable": retryable,
                    "retry_stopped_early": not retryable,
                    "roster_fingerprint": getattr(error, "roster_fingerprint", None),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                rejection_payload["family_id"] = family_id
                rejection_payload["attempts"].append(rejection)
                _write_json(rejection_path, rejection_payload)
                result["rejected_attempts"].append(rejection)
                if not retryable:
                    break
                continue
            elapsed = time.perf_counter() - started
            result["materialized"].append(
                {
                    "family_id": family_id,
                    "status": "materialized",
                    "materialization_attempt_number": attempt_number,
                    "materialization_attempt_seed": int(
                        attempt_family["materialization_attempt_seed"]
                    ),
                    "matrix_total_bytes": int(manifest["matrix_total_bytes"]),
                    "materialization_seconds": elapsed,
                    "stage_timings_seconds": manifest["stage_timings_seconds"],
                    "terminal_pair_throughput_per_second": (
                        int(manifest["terminal_count"]) ** 2 / elapsed
                    ),
                    "worker_peak_rss_bytes_after": _process_peak_rss_bytes(),
                }
            )
            completed = True
            break
        if not completed:
            result["unresolved_family_ids"].append(family_id)

    _write_heartbeat(
        heartbeat_path,
        family_id=heartbeat_family,
        attempt_number=max(0, max_attempts - 1),
        stage="worker_complete",
        details={"unresolved_count": len(result["unresolved_family_ids"])},
    )
    result["worker_peak_rss_bytes"] = _process_peak_rss_bytes()
    return result


def verify_family_path(family_dir: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    report = verify_materialized_family(family_dir)
    report["verification_seconds"] = time.perf_counter() - started
    report["worker_peak_rss_bytes"] = _process_peak_rss_bytes()
    return report
