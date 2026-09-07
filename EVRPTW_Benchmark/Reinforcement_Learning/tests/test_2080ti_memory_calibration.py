from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from EVRPTW_Benchmark.Reinforcement_Learning.scripts import (
    build_2080ti_memory_calibration_manifest as builder,
)
from EVRPTW_Benchmark.Reinforcement_Learning.scripts import (
    run_2080ti_memory_calibration as runner,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_stream import (
    atomic_write_stream,
    load_training_stream_contract,
)


def test_historical_calibration_inventory_covers_each_2080ti_job_once() -> None:
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
    assert all(row["historical_only"] is True for row in rows)
    assert all(row["enabled"] is False for row in rows)
    assert all(
        row["calibration_source_status"] == "historical_frozen_nonlaunchable"
        for row in rows
    )
    assert all(row["nonlaunchable_reason"] for row in rows)

    # Historical memory evidence remains audit-only. After independent reward
    # calibration and explicit authorization, the active queues must cover the
    # same 16 scientific job IDs without inheriting executable history flags.
    active_root = builder.ROOT / "scripts" / "rq_v1"
    active = []
    for server in ("2080ti_4_1", "2080ti_4_2", "2080ti_3_1"):
        active.extend(
            json.loads(line)
            for line in (active_root / server / "jobs.jsonl").read_text().splitlines()
            if line.strip()
        )
    assert {row["job_id"] for row in active} == {row["job_id"] for row in rows}
    assert all(row.get("historical_only") is not True for row in active)
    assert all(row.get("enabled", True) is True for row in active)


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
    assert row["historical_only"] is True
    assert row["enabled"] is False


def test_terran_rejects_nondivisor_calibration_batch() -> None:
    source = next(row for row in builder._load_jobs() if row["method"] == "terran")
    with pytest.raises(ValueError, match="must divide"):
        builder._batch_for(source, {f"terran:{source['scale']}": 3})


def test_runner_rejects_truncated_validation_contract(tmp_path: Path) -> None:
    source = builder._load_jobs()[0]
    row = builder._calibration_job(source, batch=1, slot=0)
    row["historical_only"] = False
    row["enabled"] = True
    row["validation_views"] = 100
    manifest = tmp_path / "jobs.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe calibration manifest"):
        runner._load_manifest(manifest)


def test_runner_refuses_historical_inventory_as_executable_work(tmp_path: Path) -> None:
    source = builder._load_jobs()[0]
    row = builder._calibration_job(source, batch=1, slot=0)
    manifest = tmp_path / "historical_jobs.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="historical.*audit-only.*not executable"):
        runner._load_manifest(manifest)


def test_short_stream_writer_emits_and_rebinds_verified_contract(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    frame = pd.DataFrame(
        {
            "stream_position": range(5),
            "view_id": [f"iv_{index}" for index in range(5)],
            "family_id": [f"mf_{index}" for index in range(5)],
            "city_slug": ["city"] * 5,
            "day_type": ["weekday"] * 5,
            "scale_id": ["Cus50"] * 5,
            "customer_count": [50] * 5,
            "source_row_position": range(5),
        }
    )
    manifest = {
        "schema": "drl_training_id_stream_v3",
        "sampling": "test_prefix",
        "seed": 1234,
        "scale": "Cus50",
        "sample_count": 5,
        "customer_exposures": 250,
        "allowed_parent_family_count": 5,
        "pool_view_count": 5,
        "realized_unique_family_count": 5,
        "realized_unique_view_count": 5,
        "replacement": False,
        "reuse_policy": "test",
        "prefix_stable": True,
        "prefix_stability_scope": "test",
        "method_independent": True,
        "strata": [
            {
                "city_slug": "city",
                "day_type": "weekday",
                "pool_views": 5,
                "stream_draws": 5,
            }
        ],
        "source_index_sha256": "1" * 64,
        "allowed_family_ids_sha256": None,
        "file_hash_validation_performed": True,
    }
    atomic_write_stream(source, frame, manifest)
    job = {
        "training_stream_path": str(source),
        "effective_batch_size": 2,
    }
    runner._prepare_short_streams([job], tmp_path / "output")
    short = Path(job["training_stream_path"])
    assert short.is_file()
    assert short.with_suffix(short.suffix + ".manifest.json").is_file()
    contract = load_training_stream_contract(short)
    assert contract["sample_count"] == 4
    assert contract["sha256"] == job["training_stream_contract_sha256"]
    assert contract == job["training_stream_contract_snapshot"]
