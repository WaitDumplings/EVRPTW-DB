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
