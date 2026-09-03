from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from EVRPTW_Benchmark.Reinforcement_Learning.common import protocol_trainers
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_protocol import (
    parse_float_checkpoints,
    parse_int_checkpoints,
)


def test_checkpoint_schedule_parsers_are_sorted_and_unique() -> None:
    assert parse_int_checkpoints("500,100,500") == (100, 500)
    assert parse_float_checkpoints("24,6,12,6") == (6.0, 12.0, 24.0)


def test_registered_snapshot_crossing_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    saved_paths: list[Path] = []

    def fake_save(path: Path, **_kwargs) -> None:
        path.write_bytes(b"checkpoint")
        saved_paths.append(path)

    monkeypatch.setattr(protocol_trainers, "_save_checkpoint", fake_save)
    exposure_saved: set[int] = set()
    gpu_saved: set[float] = set()
    kwargs = {
        "output": tmp_path,
        "method": "AM-EVRPTW",
        "args": SimpleNamespace(protocol_id="test"),
        "data_pass": 0,
        "policy": object(),
        "baseline": object(),
        "optimizer": object(),
        "observed_exposure": 600,
        "observed_gpu_hours": 7.0,
        "exposure_checkpoints": (100, 500, 1000),
        "gpu_hour_checkpoints": (6.0, 12.0),
        "saved_exposure": exposure_saved,
        "saved_gpu_hours": gpu_saved,
    }
    protocol_trainers._save_registered_snapshots(**kwargs)
    protocol_trainers._save_registered_snapshots(**kwargs)
    assert {path.name for path in saved_paths} == {
        "checkpoint_customer_exposure_100.pt",
        "checkpoint_customer_exposure_500.pt",
        "checkpoint_gpu_hours_6.pt",
    }
    assert len(saved_paths) == 3
    events = [json.loads(line) for line in (tmp_path / "checkpoint_events.jsonl").read_text().splitlines()]
    assert len(events) == 3
    assert all(row["schema"] == "drl_training_checkpoint_event_v1" for row in events)
