"""One-attempt Stage-2 family worker launched in an independent session."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .artifacts import verify_materialized_family
from .parallel import materialize_family_chunk


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _record_heartbeat_event(
    heartbeat_path: Path,
    *,
    family_id: str,
    attempt_number: int,
    event: Mapping[str, Any],
) -> None:
    existing: dict[str, Any] = {}
    if heartbeat_path.is_file():
        existing = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    events = list(
        existing.get("events", existing.get("details", {}).get("events", []))
    )
    events.append(dict(event))
    _atomic_write_json(
        heartbeat_path,
        {
            **existing,
            "schema": "cle_evrptw_family_heartbeat_v2",
            "family_id": family_id,
            "attempt_number": int(attempt_number),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "stage": str(event["stage"]),
            "current_event": dict(event),
            "events": events,
        },
    )


def run_envelope(envelope_path: Path) -> dict[str, Any]:
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    task = dict(envelope["task"])
    family_id = str(envelope["family_id"])
    attempt_number = int(envelope["attempt_number"])
    output_root = Path(envelope["output_root"])
    staging_root = Path(envelope["staging_root"])
    final_materialized_root = output_root / "materialized"
    staged_materialized_root = staging_root / "materialized"
    result_path = Path(envelope["result_path"])
    heartbeat_path = Path(envelope["heartbeat_path"])

    if len(task["families"]) != 1:
        raise ValueError("A family process must contain exactly one family")
    if str(task["families"][0]["family"]["family_id"]) != family_id:
        raise ValueError("Envelope family ID does not match task family ID")
    task["max_attempts_per_family"] = attempt_number + 1
    task["process_attempt_number"] = attempt_number
    task["heartbeat_path"] = str(heartbeat_path)
    task["final_materialized_root"] = str(final_materialized_root)
    task["materialized_output_root"] = str(staged_materialized_root)
    staging_root.mkdir(parents=True, exist_ok=True)

    try:
        result = materialize_family_chunk(task)
        materialized = list(result.get("materialized", []))
        if materialized and (
            len(materialized) != 1 or result.get("unresolved_family_ids")
        ):
            raise ValueError("Single-attempt worker returned an inconsistent result")
        if materialized and materialized[0].get("status") != "reused_verified":
            staged_family = staged_materialized_root / "families" / family_id
            verification_wall_started = time.perf_counter()
            verification_cpu_started = time.process_time()
            _record_heartbeat_event(
                heartbeat_path,
                family_id=family_id,
                attempt_number=attempt_number,
                event={"stage": "verification", "status": "started"},
            )
            verification = verify_materialized_family(staged_family)
            peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            verification_profile = {
                "stage": "verification",
                "status": "completed",
                "wall_seconds": time.perf_counter() - verification_wall_started,
                "cpu_seconds": time.process_time() - verification_cpu_started,
                "peak_rss_bytes": (
                    peak_rss if sys.platform == "darwin" else peak_rss * 1024
                ),
            }
            _record_heartbeat_event(
                heartbeat_path,
                family_id=family_id,
                attempt_number=attempt_number,
                event=verification_profile,
            )
            if not verification["passed"]:
                raise ValueError(
                    f"Staged family {family_id} failed verification: {verification['errors']}"
                )
            final_family = final_materialized_root / "families" / family_id
            final_family.parent.mkdir(parents=True, exist_ok=True)
            if final_family.exists():
                raise FileExistsError(f"Refusing to overwrite family: {final_family}")
            os.replace(staged_family, final_family)
            materialized[0]["staging_verification"] = verification
            materialized[0]["verification_performance_profile"] = verification_profile
            materialized[0]["atomic_publish_from"] = str(staging_root)
        payload = {
            "schema": "cle_evrptw_family_process_result_v2",
            "runtime_contract_id": "family_process_timeout_and_abort_v2",
            "family_id": family_id,
            "attempt_number": attempt_number,
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "result": result,
        }
        _atomic_write_json(result_path, payload)
        return payload
    except BaseException as error:
        failure = {
            "schema": "cle_evrptw_family_process_result_v2",
            "runtime_contract_id": "family_process_timeout_and_abort_v2",
            "family_id": family_id,
            "attempt_number": attempt_number,
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "fatal_error": {
                "error_type": type(error).__name__,
                "reason": str(error),
                "traceback": traceback.format_exc(),
            },
        }
        _atomic_write_json(result_path, failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envelope", type=Path, required=True)
    args = parser.parse_args()
    run_envelope(args.envelope)


if __name__ == "__main__":
    main()
