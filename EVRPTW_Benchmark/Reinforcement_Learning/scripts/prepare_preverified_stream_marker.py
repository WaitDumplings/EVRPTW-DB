#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

PREVERIFIED_MODE = "reuse_preverified_snapshot_no_rehash"


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain one object: {path}")
    return payload


def read_jobs(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"preverified manifest is empty or invalid: {path}")
    return rows


def atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a local no-rehash marker from frozen manifest metadata."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    dataset = args.dataset_root.resolve()
    jobs = read_jobs(args.manifest.resolve())
    marker_paths = {str(row.get("artifact_preparation_marker_path", "")) for row in jobs}
    marker_ids = {str(row.get("artifact_preparation_marker_sha256", "")) for row in jobs}
    registry_paths = {str(row.get("training_stream_registry_path", "")) for row in jobs}
    if "" in marker_paths or len(marker_paths) != 1 or "" in marker_ids or len(marker_ids) != 1:
        raise RuntimeError("preverified jobs must share one frozen artifact marker")
    if "" in registry_paths or len(registry_paths) != 1:
        raise RuntimeError("preverified jobs must share one frozen stream registry")
    if any(
        row.get("stream_integrity_mode") != PREVERIFIED_MODE
        or row.get("file_hash_validation_performed") is not False
        for row in jobs
    ):
        raise RuntimeError("every selected job must explicitly require no-rehash reuse")

    registry = read_json(repo / next(iter(registry_paths)), "stream registry")
    if registry.get("artifact_preparation_marker_sha256") != next(iter(marker_ids)):
        raise RuntimeError("manifest and registry artifact-marker metadata disagree")

    entries_by_path: dict[str, dict[str, Any]] = {}
    registered = registry.get("streams", {})
    for row in jobs:
        relative = str(row["training_stream_path"])
        snapshot = row.get("training_stream_contract_snapshot")
        key = (
            f"{row['representation']}/{row['condition']}/{row['method']}/"
            f"{row['scale']}/seed_{int(row['seed'])}"
        )
        if not isinstance(snapshot, dict) or registered.get(key) != {
            "path": relative,
            "snapshot": snapshot,
        }:
            raise RuntimeError(f"job is not frozen in the registry: {row['job_id']}")
        stream_path = repo / relative
        sidecar_path = stream_path.with_suffix(stream_path.suffix + ".manifest.json")
        if not stream_path.is_file() or not sidecar_path.is_file():
            raise RuntimeError(f"preverified stream or sidecar is missing: {stream_path}")
        entry = {
            "relative_path": relative,
            "sha256": snapshot["sha256"],
            "snapshot": snapshot,
        }
        existing = entries_by_path.get(relative)
        if existing is not None and existing != entry:
            raise RuntimeError(
                f"conflicting preverified contracts for shared stream: {relative}"
            )
        # G/E jobs can use one ordered training-ID stream. The marker describes
        # artifacts, so emit that artifact once while checking every job above.
        entries_by_path[relative] = entry

    marker_path = (repo / next(iter(marker_paths))).resolve()
    if repo not in marker_path.parents:
        raise RuntimeError("artifact marker must remain inside the repository")
    atomic_write(
        marker_path,
        {
            "schema": "drl_rq_artifact_preparation_v2",
            "status": "passed",
            "file_hash_validation_performed": False,
            "stream_integrity_mode": PREVERIFIED_MODE,
            "marker_sha256": next(iter(marker_ids)),
            "dataset_root": str(dataset),
            "training_stream_contracts": list(entries_by_path.values()),
        },
    )
    print(f"Prepared no-rehash marker for {len(jobs)} job(s): {marker_path}")


if __name__ == "__main__":
    main()
