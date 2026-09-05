from __future__ import annotations

from types import SimpleNamespace

from EVRPTW_Benchmark.Reinforcement_Learning.common.data_pass import (
    DataPassState,
    pass_batches,
    seeded_pass_order,
)

from EVRPTW_Benchmark.Reinforcement_Learning.common.training_protocol import (
    grouped_batches,
    require_registered_batches,
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


def test_microbatches_form_exact_logical_epochs_without_extra_rows() -> None:
    rows = list(range(1_000))
    physical = pass_batches(
        rows, seed=1234, data_pass=1, physical_batch_size=50
    )
    logical = list(
        grouped_batches(
            physical,
            effective_batch_size=100,
            max_batches=3 * 2,
        )
    )
    assert len(logical) == 3
    assert [sum(map(len, group)) for group in logical] == [100, 100, 100]
    selected = [row for group in logical for batch in group for row in batch]
    assert len(selected) == len(set(selected)) == 300


def test_remainder_microbatches_keep_exact_logical_boundaries() -> None:
    physical = [list(range(start, min(start + 6, 30))) for start in range(0, 30, 6)]
    logical = list(grouped_batches(physical, effective_batch_size=10))
    assert [sum(map(len, group)) for group in logical] == [10, 10, 10]
    assert [item for group in logical for batch in group for item in batch] == list(range(30))
    assert [[len(batch) for batch in group] for group in logical] == [
        [6, 4], [2, 6, 2], [4, 6]
    ]


def test_registered_batch_allows_safe_remainder_but_not_oversize() -> None:
    args = SimpleNamespace(physical_batch_size=6, effective_batch_size=10)
    assert require_registered_batches(args, legacy_batch=1) == (6, 10)
    args.physical_batch_size = 11
    try:
        require_registered_batches(args, legacy_batch=1)
    except ValueError as exc:
        assert "cannot exceed" in str(exc)
    else:
        raise AssertionError("oversize physical batch must fail")


def test_atomic_state_resume_and_protocol_guard(tmp_path) -> None:
    path = tmp_path / "data_pass_state.json"
    state = DataPassState(
        protocol_id="v1",
        completed_data_passes=3,
        instances_seen=30,
        customer_exposures=3000,
        optimizer_steps=9,
        environment_transitions=4567,
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


def test_old_state_without_transition_counter_remains_loadable(tmp_path) -> None:
    path = tmp_path / "data_pass_state.json"
    path.write_text(
        '{"protocol_id":"v1","completed_data_passes":2,"instances_seen":20,'
        '"customer_exposures":2000,"optimizer_steps":6,"last_checkpoint":"x.pt"}'
    )
    state = DataPassState.load(path, protocol_id="v1")
    assert state.environment_transitions == 0


def test_pass_order_rejects_zero_based_pass() -> None:
    try:
        seeded_pass_order(4, seed=1, data_pass=0)
    except ValueError as exc:
        assert "one-based" in str(exc)
    else:
        raise AssertionError("zero data_pass must fail")
