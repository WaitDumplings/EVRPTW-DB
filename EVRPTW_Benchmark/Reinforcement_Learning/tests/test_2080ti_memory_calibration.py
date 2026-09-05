from __future__ import annotations

import json
from pathlib import Path

import pytest

from EVRPTW_Benchmark.Reinforcement_Learning.scripts import (
    build_2080ti_memory_calibration_manifest as builder,
)
from EVRPTW_Benchmark.Reinforcement_Learning.scripts import (
    run_2080ti_memory_calibration as runner,
)


def test_calibration_inventory_covers_each_2080ti_job_once() -> None:
    rows = builder._load_jobs()
    assert len(rows) == 16
    assert len({row["job_id"] for row in rows}) == 16
    counts = {
        wave: sum(builder._wave(row) == wave for row in rows)
        for wave in builder.WAVES
    }
    assert counts == {
        "cus50": 4,
        "cus100_g": 4,
        "cus100_e": 4,
        "cus100_support": 4,
    }


def test_calibration_job_preserves_formal_semantics_but_uses_two_epochs() -> None:
    source = next(row for row in builder._load_jobs() if row["method"] == "drl_ts")
    row = builder._calibration_job(source, batch=17, slot=2)
    assert row["calibration_original_job_id"] == source["job_id"]
    assert row["training_rollout_steps"] == source["training_rollout_steps"]
    assert row["validation_views"] == 500
    assert row["validation_candidate_count"] == 100
    assert row["training_epochs"] == 2
    assert row["soft_stage_end_epoch"] == 1
    assert row["target_environments"] == 2 * source["effective_batch_size"]


def test_terran_rejects_nondivisor_calibration_batch() -> None:
    source = next(row for row in builder._load_jobs() if row["method"] == "terran")
    with pytest.raises(ValueError, match="must divide"):
        builder._batch_for(source, {f"terran:{source['scale']}": 3})


def test_runner_rejects_truncated_validation_contract(tmp_path: Path) -> None:
    source = builder._load_jobs()[0]
    row = builder._calibration_job(source, batch=1, slot=0)
    row["validation_views"] = 100
    manifest = tmp_path / "jobs.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe calibration manifest"):
        runner._load_manifest(manifest)
