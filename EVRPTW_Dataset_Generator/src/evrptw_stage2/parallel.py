"""Spawn-safe Stage-2 materialization and verification workers."""

from __future__ import annotations

import json
import resource
import sys
import time
from collections.abc import Mapping
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


def materialize_family_chunk(task: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize one single-city chunk while reusing its routing topology."""

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
    amazon_artifacts = load_amazon_stage2_artifacts(task["amazon_artifact_root"])
    max_attempts = int(task["max_attempts_per_family"])
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
        family_dir = output_root / "materialized" / "families" / family_id
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
            else {"schema": "cle_evrptw_family_rejection_ledger_v1", "attempts": []}
        )
        first_attempt_number = len(rejection_payload["attempts"])
        completed = False
        for offset in range(max_attempts):
            attempt_number = first_attempt_number + offset
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
                    output_root=output_root / "materialized",
                    routing_topology_cache=topology_cache,
                    community_adjacency_cache=adjacency_cache,
                )
            except Exception as error:  # noqa: BLE001 - persist every failed attempt.
                rejection = {
                    "family_id": family_id,
                    "city_slug": city,
                    "attempt_number": attempt_number,
                    "attempt_seed": int(attempt_family["materialization_attempt_seed"]),
                    "next_attempt_seed": derive_seed(
                        int(family["family_seed"]),
                        "materialization_attempt",
                        attempt_number + 1,
                    ),
                    "error_type": type(error).__name__,
                    "reason": str(error),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                rejection_payload["family_id"] = family_id
                rejection_payload["attempts"].append(rejection)
                _write_json(rejection_path, rejection_payload)
                result["rejected_attempts"].append(rejection)
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

    result["worker_peak_rss_bytes"] = _process_peak_rss_bytes()
    return result


def verify_family_path(family_dir: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    report = verify_materialized_family(family_dir)
    report["verification_seconds"] = time.perf_counter() - started
    report["worker_peak_rss_bytes"] = _process_peak_rss_bytes()
    return report
