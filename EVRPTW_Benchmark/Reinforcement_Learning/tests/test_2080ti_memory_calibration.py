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


def test_launchable_calibration_inventory_covers_each_2080ti_job_once() -> None:
    rows = builder._load_jobs()
    assert len(rows) == 16
    assert len({row["job_id"] for row in rows}) == 16
    assert {wave: sum(builder._wave(row) == wave for row in rows) for wave in builder.WAVES} == {
        "cus50": 4, "cus100_g": 4, "cus100_e": 4, "cus100_support": 4,
    }
    assert all(row.get("historical_only") is not True for row in rows)
    assert all(row.get("enabled", True) is True for row in rows)
    assert {row["validation_candidate_count"] for row in rows} == {50}
    assert {
        (row["method"], row["scale"]): row["effective_batch_size"]
        for row in rows if row["condition"] == "Full-support" and row["representation"] == "G"
    } == {
        ("am_evrptw", "Cus50"): 2304,
        ("evrptw_rl", "Cus50"): 336,
        ("drl_ts", "Cus50"): 144,
        ("terran", "Cus50"): 480,
        ("am_evrptw", "Cus100"): 800,
        ("evrptw_rl", "Cus100"): 96,
        ("drl_ts", "Cus100"): 40,
        ("terran", "Cus100"): 280,
    }


def test_calibration_job_preserves_formal_semantics_but_uses_two_epochs() -> None:
    source = next(row for row in builder._load_jobs() if row["method"] == "drl_ts")
    row = builder._calibration_job(source, batch=17, slot=2)
    assert row["calibration_original_job_id"] == source["job_id"]
    assert row["training_rollout_steps"] == source["training_rollout_steps"]
    assert row["validation_views"] == 500
    assert row["validation_candidate_count"] == 50
    assert row["training_epochs"] == 2
    assert row["soft_stage_end_epoch"] == 1
    assert row["target_environments"] == 34
    assert row["effective_batch_size"] == row["physical_batch_size"] == 17
    assert row["historical_only"] is False
    assert row["enabled"] is True


def test_calibration_batch_override_need_not_match_fairness_budget() -> None:
    source = next(row for row in builder._load_jobs() if row["method"] == "terran")
    assert builder._batch_for(source, {f"terran:{source['scale']}": 3}) == 3


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
