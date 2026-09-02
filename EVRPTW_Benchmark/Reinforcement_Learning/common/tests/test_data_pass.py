from __future__ import annotations

from EVRPTW_Benchmark.Reinforcement_Learning.common.data_pass import (
    DataPassState,
    pass_batches,
    seeded_pass_order,
)


def test_each_pass_is_complete_unique_and_deterministic() -> None:
    rows = list(range(23))
    first = pass_batches(rows, seed=1234, data_pass=1, physical_batch_size=5)
    replay = pass_batches(rows, seed=1234, data_pass=1, physical_batch_size=5)
    second = pass_batches(rows, seed=1234, data_pass=2, physical_batch_size=5)
    flat = [item for batch in first for item in batch]
    assert first == replay
    assert flat != [item for batch in second for item in batch]
    assert sorted(flat) == rows
    assert len(flat) == len(set(flat)) == len(rows)


def test_atomic_state_resume_and_protocol_guard(tmp_path) -> None:
    path = tmp_path / "data_pass_state.json"
    state = DataPassState(
        protocol_id="v1",
        completed_data_passes=3,
        instances_seen=30,
        customer_exposures=3000,
        optimizer_steps=9,
        last_checkpoint="checkpoint_pass_0003.pt",
    )
    state.atomic_write(path)
    assert DataPassState.load(path, protocol_id="v1") == state
    try:
        DataPassState.load(path, protocol_id="v2")
    except ValueError as exc:
        assert "protocol mismatch" in str(exc)
    else:
        raise AssertionError("protocol mismatch must reject resume")


def test_pass_order_rejects_zero_based_pass() -> None:
    try:
        seeded_pass_order(4, seed=1, data_pass=0)
    except ValueError as exc:
        assert "one-based" in str(exc)
    else:
        raise AssertionError("zero data_pass must fail")
