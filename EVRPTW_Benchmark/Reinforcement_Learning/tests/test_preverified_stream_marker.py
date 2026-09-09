from __future__ import annotations

import json
from pathlib import Path

import pytest

from EVRPTW_Benchmark.Reinforcement_Learning.scripts import prepare_preverified_stream_marker as MARKER


def _fixture(tmp_path: Path, monkeypatch, *, conflict: bool = False) -> Path:
    stream = tmp_path / "artifacts/stream.parquet"
    stream.parent.mkdir()
    stream.write_bytes(b"preverified-stream-placeholder")
    stream.with_suffix(".parquet.manifest.json").write_text("{}")
    marker = tmp_path / "artifacts/marker.json"
    marker.write_text('{"original": true}')
    snapshots = [{"sha256": "a" * 64, "sample_count": 10}] * 2
    if conflict:
        snapshots[1] = {"sha256": "b" * 64, "sample_count": 20}
    jobs = []
    registry = {"artifact_preparation_marker_sha256": "c" * 64, "streams": {}}
    for representation, snapshot in zip(("G", "E"), snapshots):
        row = {
            "job_id": representation,
            "representation": representation,
            "condition": "Full-support",
            "method": "terran",
            "scale": "Cus100",
            "seed": 1234,
            "artifact_preparation_marker_path": "artifacts/marker.json",
            "artifact_preparation_marker_sha256": "c" * 64,
            "training_stream_registry_path": "artifacts/registry.json",
            "training_stream_path": "artifacts/stream.parquet",
            "training_stream_contract_snapshot": snapshot,
            "stream_integrity_mode": MARKER.PREVERIFIED_MODE,
            "file_hash_validation_performed": False,
        }
        jobs.append(row)
        registry["streams"][f"{representation}/Full-support/terran/Cus100/seed_1234"] = {
            "path": row["training_stream_path"], "snapshot": snapshot,
        }
    (tmp_path / "artifacts/registry.json").write_text(json.dumps(registry))
    manifest = tmp_path / "jobs.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in jobs))
    monkeypatch.setattr("sys.argv", [
        "prepare", "--manifest", str(manifest), "--repo-root", str(tmp_path),
        "--dataset-root", str(tmp_path / "dataset"),
    ])
    return marker


def test_shared_g_and_e_stream_is_written_once(tmp_path: Path, monkeypatch) -> None:
    marker = _fixture(tmp_path, monkeypatch)
    MARKER.main()
    payload = json.loads(marker.read_text())
    assert len(payload["training_stream_contracts"]) == 1
    assert payload["training_stream_contracts"][0]["snapshot"] == {
        "sha256": "a" * 64, "sample_count": 10,
    }
    assert payload["marker_sha256"] == "c" * 64
    assert payload["file_hash_validation_performed"] is False


def test_shared_stream_conflict_does_not_replace_existing_marker(tmp_path: Path, monkeypatch) -> None:
    marker = _fixture(tmp_path, monkeypatch, conflict=True)
    before = marker.read_bytes()
    with pytest.raises(RuntimeError, match="conflicting preverified contracts"):
        MARKER.main()
    assert marker.read_bytes() == before
