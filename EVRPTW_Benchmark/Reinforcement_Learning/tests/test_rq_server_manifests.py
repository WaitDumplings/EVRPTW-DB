from __future__ import annotations

from collections import Counter

from EVRPTW_Benchmark.Reinforcement_Learning.scripts.build_rq_server_manifests import (
    SERVERS,
    build,
)


def test_four_server_queues_encode_exact_frozen_72_run_design() -> None:
    queues = build()
    assert set(queues) == set(SERVERS)
    rows = [row for queue in queues.values() for row in queue]
    formal = [row for row in rows if row["run_mode"] == "full"]
    pilots = [row for row in rows if row["run_mode"] == "pilot"]
    assert len(formal) == 72
    assert len(pilots) == 20
    assert len({row["job_id"] for row in formal}) == 72
    assert Counter((row["representation"], row["condition"]) for row in formal) == {
        ("G", "Full-support"): 48,
        ("G", "Random-10%-support"): 6,
        ("G", "Coverage-10%-support"): 6,
        ("E", "Full-support"): 12,
    }
    assert all(row["formal_gate_file"] for row in formal)
    assert all(not row["training_stream_path"].startswith("/") for row in rows)
    assert all(row["scale"] != "Cus2000" for row in formal)


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


def test_cus1000_formal_jobs_are_assigned_only_to_a6000() -> None:
    queues = build()
    for server, rows in queues.items():
        for row in rows:
            if row["run_mode"] == "full" and row["scale"] == "Cus1000":
                assert server == "a6000_2_1"


def test_cus1000_batch2_budget_has_exact_epoch_environment_and_exposure_semantics() -> None:
    expected = {
        "Cus50": (1_000, 200, 200_000, 10_000_000),
        "Cus100": (1_000, 50, 50_000, 5_000_000),
        "Cus500": (1_000, 4, 4_000, 2_000_000),
        "Cus1000": (1_000, 2, 2_000, 2_000_000),
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
        assert row["runtime_budget_id"] == "drl_rq_runtime_budget_v4_cus1000_b2_val100"
        assert row["runtime_budget_id"] in row["training_stream_path"]
        assert row["training_epochs"] == epochs
        assert row["planned_optimizer_updates"] == epochs
        assert row["logical_environments_per_epoch"] == environments_per_epoch
        assert row["effective_batch_size"] == environments_per_epoch
        assert row["target_environments"] == total_environments
        assert row["customer_exposure_budget"] == exposures
        assert row["physical_batch_size"] <= row["effective_batch_size"]
        assert row["effective_batch_size"] % row["physical_batch_size"] == 0
        assert row["validation_every_epochs"] == 50
        assert row["validation_checkpoints"] == 20
        assert row["validation_views"] == (
            100 if row["scale"] == "Cus1000" else 500
        )
        assert row["final_validation_views"] == (
            500 if row["scale"] == "Cus1000" else 0
        )
        assert row["planning_wall_time_hours"] == (
            48 if row["scale"] == "Cus1000" else None
        )
