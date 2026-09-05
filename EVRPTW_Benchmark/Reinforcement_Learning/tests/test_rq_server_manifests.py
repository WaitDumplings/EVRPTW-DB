from __future__ import annotations

import json
from collections import Counter

from EVRPTW_Benchmark.Reinforcement_Learning.scripts.build_rq_server_manifests import (
    SERVERS,
    SCRIPT_ROOT,
    build,
    build_a6000_cus1000_priority_queue,
)


def test_four_server_queues_encode_exact_four_scale_one_seed_design() -> None:
    queues = build()
    assert set(queues) == set(SERVERS)
    rows = [row for queue in queues.values() for row in queue]
    formal = [row for row in rows if row["run_mode"] == "full"]
    assert len(formal) == 24
    assert rows == formal
    assert len({row["job_id"] for row in formal}) == 24
    assert {row["seed"] for row in rows} == {1234}
    assert Counter((row["representation"], row["condition"]) for row in formal) == {
        ("G", "Full-support"): 16,
        ("G", "Random-10%-support"): 2,
        ("G", "Coverage-10%-support"): 2,
        ("E", "Full-support"): 4,
    }
    assert all(row["formal_gate_file"] for row in formal)
    assert all(not row["training_stream_path"].startswith("/") for row in rows)
    assert all(
        row["scale"] in {"Cus50", "Cus100", "Cus500", "Cus1000"}
        for row in rows
    )


def test_shared_stream_is_method_independent_within_condition_scale_seed() -> None:
    rows = [
        row
        for queue in build().values()
        for row in queue
        if row["run_mode"] == "full"
    ]
    grouped: dict[tuple[str, str, str, int], set[str]] = {}
    for row in rows:
        key = (row["representation"], row["condition"], row["scale"], row["seed"])
        grouped.setdefault(key, set()).add(row["training_stream_path"])
    assert all(len(paths) == 1 for paths in grouped.values())


def test_scale_aware_hardware_assignment_is_strict() -> None:
    queues = build()
    assert all(any(row["run_mode"] == "full" for row in rows) for rows in queues.values())
    for server, rows in queues.items():
        allowed = (
            {"Cus50", "Cus100"}
            if server.startswith("2080ti_")
            else {"Cus500", "Cus1000"}
        )
        assert rows
        assert all(row["scale"] in allowed for row in rows)


def test_full_train_budget_has_exact_epoch_environment_and_exposure_semantics() -> None:
    expected = {
        "Cus50": (6_000, 1_024, 6_144_000, 307_200_000),
        "Cus100": (6_000, 256, 1_536_000, 153_600_000),
        "Cus500": (6_000, 64, 384_000, 192_000_000),
        "Cus1000": (6_000, 2, 12_000, 12_000_000),
    }
    formal = [
        row
        for queue in build().values()
        for row in queue
        if row["run_mode"] == "full"
    ]
    for row in formal:
        epochs, environments_per_epoch, total_environments, exposures = expected[
            row["scale"]
        ]
        assert (
            row["runtime_budget_id"]
            == "drl_rq_runtime_budget_v11_min5000_max6000_tailval50"
        )
        assert row["runtime_budget_id"] in row["training_stream_path"]
        assert row["training_epochs"] == epochs
        assert row["planned_optimizer_updates"] == epochs
        assert row["logical_environments_per_epoch"] == environments_per_epoch
        assert row["effective_batch_size"] == environments_per_epoch
        assert row["target_environments"] == total_environments
        assert row["customer_exposure_budget"] == exposures
        assert row["physical_batch_size"] <= row["effective_batch_size"]
        assert row["effective_batch_size"] % row["physical_batch_size"] == 0
        assert row["validation_every_epochs"] == 250
        assert row["validation_checkpoints"] == 40
        assert row["minimum_training_epochs"] == 5_000
        assert row["post_minimum_validation_every_epochs"] == 50
        assert row["validation_views"] == 500
        assert row["validation_decode_type"] == "sampling"
        assert row["validation_candidate_count"] == 100
        assert row["test_decode_type"] == "sampling"
        assert row["test_candidate_count"] == 100
        expected_trajectories = 100 if row["method"] == "terran" else 1
        if row["method"] == "am_evrptw" and row["scale"] in {
            "Cus500",
            "Cus1000",
        }:
            expected_trajectories = 100
        assert row["training_trajectory_count"] == expected_trajectories
        assert row["final_validation_views"] == 0
        assert row["planning_wall_time_hours"] is None
        assert row["early_stop_patience_validations"] == 10
        assert row["early_stop_start_epoch"] == 5_000
        assert row["soft_stage_end_epoch"] == (2_500 if row["method"] == "drl_ts" else None)
        assert row["primary_checkpoint"] == "best_overall.ckpt"
        assert row["minimum_budget_checkpoint"] == "best_within_5000.ckpt"
        assert row["extended_checkpoint"] == "best_overall.ckpt"
        assert row["validation_seed"] == row["seed"] + 910_000_000


def test_2080ti_jobs_use_only_measured_safe_sample100_batches() -> None:
    safe_batches = {
        "am_evrptw": {"Cus50": 1024, "Cus100": 256},
        "evrptw_rl": {"Cus50": 128, "Cus100": 64},
        "drl_ts": {"Cus50": 128, "Cus100": 32},
        "terran": {"Cus50": 128, "Cus100": 128},
    }
    queues = build()
    rows = [
        row
        for server, queue in queues.items()
        if server.startswith("2080ti_")
        for row in queue
    ]
    assert rows
    for row in rows:
        assert row["validation_decode_type"] == "sampling"
        assert row["validation_candidate_count"] == 100
        assert row["physical_batch_size"] == safe_batches[row["method"]][row["scale"]]


def test_a6000_jobs_use_calibrated_even_physical_batches() -> None:
    expected = {
        "am_evrptw": {"Cus500": 8, "Cus1000": 2},
        "evrptw_rl": {"Cus500": 16, "Cus1000": 2},
        "drl_ts": {"Cus500": 8, "Cus1000": 2},
        "terran": {"Cus500": 64, "Cus1000": 2},
    }
    rows = build()["a6000_2_1"]
    assert rows
    for row in rows:
        assert row["physical_batch_size"] == expected[row["method"]][row["scale"]]
        assert row["physical_batch_size"] % 2 == 0
        assert row["validation_views"] == 500


def test_a6000_cus1000_priority_queue_uses_approved_two_gpu_order() -> None:
    canonical = build()["a6000_2_1"]
    rows = build_a6000_cus1000_priority_queue()
    assert [
        (row["method"], row["global_slot"], row["queue_position"])
        for row in rows
    ] == [
        ("terran", 0, 0),
        ("drl_ts", 1, 0),
        ("evrptw_rl", 1, 1),
        ("am_evrptw", 1, 2),
    ]
    assert {row["scale"] for row in rows} == {"Cus1000"}
    assert {row["seed"] for row in rows} == {1234}
    assert {row["job_id"] for row in rows} == {
        row["job_id"] for row in canonical if row["scale"] == "Cus1000"
    }

    canonical_by_id = {row["job_id"]: row for row in canonical}
    for row in rows:
        scientific = {
            key: value
            for key, value in row.items()
            if key not in {"global_slot", "queue_position"}
        }
        canonical_scientific = {
            key: value
            for key, value in canonical_by_id[row["job_id"]].items()
            if key not in {"global_slot", "queue_position"}
        }
        assert scientific == canonical_scientific


def test_checked_in_a6000_cus1000_priority_manifest_matches_builder() -> None:
    manifest = SCRIPT_ROOT / "a6000_2_1" / "cus1000_jobs.jsonl"
    checked_in = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert checked_in == build_a6000_cus1000_priority_queue()


def test_checked_in_server_manifests_match_builder() -> None:
    for server, expected in build().items():
        manifest = SCRIPT_ROOT / server / "jobs.jsonl"
        checked_in = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert checked_in == expected


def test_artifact_preparation_uses_v11_manifest_exposure_budgets() -> None:
    script = (SCRIPT_ROOT / "prepare_artifacts.sh").read_text(encoding="utf-8")
    assert "drl_rq_runtime_budget_v11_min5000_max6000_tailval50" in script
    assert "drl_rq_runtime_budget_v10_min5000_max10000_tailval50" not in script
    for scale, exposure in {
        "Cus50": 307_200_000,
        "Cus100": 153_600_000,
        "Cus500": 192_000_000,
        "Cus1000": 12_000_000,
    }.items():
        assert f"[{scale}]={exposure}" in script
    assert '--customer-exposures "${FORMAL_EXPOSURE[Cus100]}"' in script
