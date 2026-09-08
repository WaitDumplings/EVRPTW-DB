#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from EVRPTW_Benchmark.Reinforcement_Learning.common.training_stream import (
    STREAM_INTEGRITY_MODE_PREVERIFIED,
    training_stream_contract_from_preverified_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PREVERIFIED_INTEGRITY_MODE = STREAM_INTEGRITY_MODE_PREVERIFIED


def _canonical_digest(payload: Mapping[str, Any], *, omitted: set[str]) -> str:
    canonical = {key: value for key, value in payload.items() if key not in omitted}
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return payload


def _repo_relative_stream(repo_root: Path, raw_path: Path) -> tuple[Path, str]:
    absolute = raw_path if raw_path.is_absolute() else repo_root / raw_path
    absolute = absolute.resolve()
    try:
        relative = absolute.relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"training stream must be inside the repository: {raw_path}") from error
    if absolute.suffix != ".parquet":
        raise ValueError(f"training stream must be a Parquet path: {raw_path}")
    return absolute, relative


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def refresh_preverified_streams(
    *,
    repo_root: Path,
    registry_path: Path,
    marker_path: Path,
    stream_paths: list[Path],
) -> dict[str, Any]:
    if not stream_paths:
        raise ValueError("at least one explicit --stream path is required")
    registry = _read_object(registry_path, label="training-stream registry")
    marker = _read_object(marker_path, label="artifact preparation marker")
    if registry.get("schema") != "drl_training_stream_registry_v1":
        raise ValueError("invalid training-stream registry schema")
    if marker.get("schema") != "drl_rq_artifact_preparation_v2":
        raise ValueError("invalid artifact preparation marker schema")
    if registry.get("runtime_budget_id") != marker.get("runtime_budget_id"):
        raise ValueError("registry and marker runtime budgets disagree")
    if registry.get("sha256") != _canonical_digest(registry, omitted={"sha256"}):
        raise ValueError("training-stream registry JSON is inconsistent")
    if marker.get("marker_sha256") != _canonical_digest(
        marker, omitted={"marker_sha256", "dataset_root"}
    ):
        raise ValueError("artifact preparation marker JSON is inconsistent")
    if registry.get("artifact_preparation_marker_sha256") != marker.get(
        "marker_sha256"
    ):
        raise ValueError("training-stream registry is not bound to this marker")

    marker_entries = {
        str(item.get("relative_path")): item
        for item in marker.get("training_stream_contracts", [])
        if isinstance(item, dict)
    }
    registered = registry.get("streams")
    if not isinstance(registered, dict):
        raise ValueError("training-stream registry streams must be an object")

    replacements: dict[str, dict[str, Any]] = {}
    for raw_path in stream_paths:
        stream_path, relative_path = _repo_relative_stream(repo_root, raw_path)
        if relative_path in replacements:
            raise ValueError(f"duplicate explicit training stream: {relative_path}")
        sidecar_path = stream_path.with_suffix(stream_path.suffix + ".manifest.json")
        if not stream_path.is_file() or not sidecar_path.is_file():
            raise ValueError(
                "preverified stream and sidecar must both exist: "
                f"{stream_path}, {sidecar_path}"
            )
        if relative_path not in marker_entries:
            raise ValueError(f"stream is absent from the preparation marker: {relative_path}")
        matching_registry_entries = [
            entry
            for entry in registered.values()
            if isinstance(entry, dict) and entry.get("path") == relative_path
        ]
        if not matching_registry_entries:
            raise ValueError(f"stream is absent from the frozen registry: {relative_path}")
        replacements[relative_path] = (
            training_stream_contract_from_preverified_manifest(sidecar_path)
        )

    for relative_path, snapshot in replacements.items():
        marker_entries[relative_path].update(
            sha256=snapshot["sha256"], snapshot=snapshot
        )
        for entry in registered.values():
            if isinstance(entry, dict) and entry.get("path") == relative_path:
                entry["snapshot"] = snapshot

    marker["file_hash_validation_performed"] = False
    marker["stream_integrity_mode"] = PREVERIFIED_INTEGRITY_MODE
    marker["selectively_refreshed_streams"] = sorted(replacements)
    marker["marker_sha256"] = _canonical_digest(
        marker, omitted={"marker_sha256", "dataset_root"}
    )
    registry["artifact_preparation_marker_sha256"] = marker["marker_sha256"]
    registry["sha256"] = _canonical_digest(registry, omitted={"sha256"})

    _atomic_write_json(marker_path, marker)
    _atomic_write_json(registry_path, registry)
    return {
        "updated_streams": sorted(replacements),
        "artifact_preparation_marker_sha256": marker["marker_sha256"],
        "training_stream_registry_sha256": registry["sha256"],
        "stream_integrity_mode": PREVERIFIED_INTEGRITY_MODE,
        "file_hash_validation_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh selected frozen stream snapshots from trusted sidecars "
            "without opening any Parquet content"
        )
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--stream", type=Path, action="append", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = refresh_preverified_streams(
        repo_root=args.repo_root.resolve(),
        registry_path=args.registry.resolve(),
        marker_path=args.marker.resolve(),
        stream_paths=args.stream,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
