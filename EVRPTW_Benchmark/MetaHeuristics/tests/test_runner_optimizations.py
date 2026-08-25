from __future__ import annotations

import csv
import importlib.util
import json
import multiprocessing
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
META_ROOT = REPO_ROOT / "EVRPTW_Benchmark" / "MetaHeuristics"
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Dataset_Generator" / "src"))
sys.path.insert(0, str(META_ROOT))
sys.path.insert(0, str(META_ROOT / "ALNS_Solver"))

from evrptw_core.schema import EVRPTWInstance

import benchmark_common
from benchmark_common import (
    ALGORITHM_TIMING_SCOPE,
    IncumbentReplayCache,
    IncumbentEventRecorder,
    Stage2ViewTask,
    bounded_process_map,
    build_run_contract,
    build_input_tasks,
    certificate_singleton_routes,
    resolve_optional_iteration_budget,
    running_time_energy_matrix_kwh,
    running_time_matrix_s,
    read_stage2_tasks,
    stable_view_seed,
    stable_view_shard,
    validate_routes,
)
import benchmark_output
from benchmark_output import (
    IncrementalCsvStore,
    atomic_save_solution,
    error_snapshot_rows,
    save_result_artifacts,
    snapshot_rows,
)
from evrptw_core.schema import EVRPTWSolution
from instance_adapter import to_alns_tensor_instance


def make_multihop_certificate_instance() -> EVRPTWInstance:
    # Canonical order: depot, customer, CS-A, CS-B.  The only energy-feasible
    # return is customer -> CS-A -> CS-B -> depot.
    travel = np.asarray(
        [
            [0.0, 500.0, 900.0, 600.0],
            [900.0, 0.0, 400.0, 900.0],
            [900.0, 900.0, 0.0, 600.0],
            [600.0, 900.0, 900.0, 0.0],
        ],
        dtype=np.float32,
    )
    energy = np.asarray(
        [
            [0.0, 50.0, 150.0, 60.0],
            [150.0, 0.0, 40.0, 150.0],
            [150.0, 150.0, 0.0, 60.0],
            [60.0, 150.0, 150.0, 0.0],
        ],
        dtype=np.float32,
    )
    distance = (energy / np.float32(0.4)).astype(np.float32)
    return EVRPTWInstance.from_dict(
        {
            "instance_id": "multihop-certificate",
            "region_id": "test",
            "mother_board_id": "mf-test",
            "operating_day_id": "mf-test",
            "day_type": "weekday",
            "working_start_s": 0,
            "working_end_s": 100_000,
            "depot": [0.0, 0.0],
            "customers": [[1.0, 0.0]],
            "charging_stations": [[2.0, 0.0], [3.0, 0.0]],
            "distance_matrix_km": distance,
            "demands_cm3": [1.0],
            "package_counts": [1],
            "service_time_s": [0.0],
            "tw_s": [[0.0, 100_000.0]],
            "cs_time_to_depot_s": [3360.0, 600.0],
            "vehicle": {
                "battery_capacity_kwh": 100.0,
                "cargo_capacity_cm3": 100.0,
                "charging_power_derating_factor": 0.9,
            },
            "raw_travel_time_matrix_s": travel,
            "energy_matrix_kwh": energy,
            "running_time_shortest_matrix_s": travel,
            "running_time_path_energy_kwh": energy,
            "charging_power_kw": np.asarray([100.0, 100.0], dtype=np.float32),
            "charging_policy": {
                "policy": "full_charge_linear_derated_v2",
                "charging_power_derating_factor": 0.9,
            },
            "cs_activation": {"charging_power_kw": [100.0, 100.0]},
            "feasibility_certificate": {
                "inbound_full_state_terminal_index": np.asarray([0], dtype=np.int32),
                "first_post_customer_charger_terminal_index": np.asarray(
                    [2], dtype=np.int32
                ),
                "charging_visit_count": np.asarray([2], dtype=np.int16),
            },
        }
    )


def test_certificate_reconstructs_multihop_route_and_adapter_only_exposes_replayed() -> None:
    instance = make_multihop_certificate_instance()
    routes = certificate_singleton_routes(instance)
    assert routes == [[0, 1, 2, 3, 0]]
    audit = validate_routes(instance, routes)
    assert audit["passed"]
    assert audit["charging_visit_count"] == 2

    adapted = to_alns_tensor_instance(instance)
    assert adapted["certificate_singleton_routes"] == routes
    assert adapted["feasibility_certificate"] is instance.raw["feasibility_certificate"]

    # A stale visit count must disable the warm start rather than trusting raw
    # generator metadata.
    instance.raw["feasibility_certificate"]["charging_visit_count"] = np.asarray([1])
    assert certificate_singleton_routes(instance) is None


def test_route_replay_reuses_all_three_float32_matrices_without_value_change() -> None:
    instance = make_multihop_certificate_instance()
    travel = running_time_matrix_s(instance)
    energy = running_time_energy_matrix_kwh(instance)
    assert np.shares_memory(travel, instance.raw["running_time_shortest_matrix_s"])
    assert np.shares_memory(energy, instance.raw["running_time_path_energy_kwh"])
    assert np.shares_memory(np.asarray(instance.distance_matrix_km), instance.distance_matrix_km)
    assert travel.dtype == energy.dtype == instance.distance_matrix_km.dtype == np.float32

    routes = [[0, 1, 2, 3, 0]]
    float32_audit = validate_routes(instance, routes)
    promoted = EVRPTWInstance.from_dict(
        {
            **instance.raw,
            "distance_matrix_km": instance.distance_matrix_km.astype(np.float64),
            "raw_travel_time_matrix_s": travel.astype(np.float64),
            "energy_matrix_kwh": energy.astype(np.float64),
            "running_time_shortest_matrix_s": travel.astype(np.float64),
            "running_time_path_energy_kwh": energy.astype(np.float64),
        }
    )
    float64_audit = validate_routes(promoted, routes)
    assert float32_audit["violations"] == float64_audit["violations"]
    assert float32_audit["objective_distance_km"] == float64_audit[
        "objective_distance_km"
    ]
    assert float32_audit["total_charging_time_s"] == float64_audit[
        "total_charging_time_s"
    ]


def _stage2_tasks(count: int) -> list[Stage2ViewTask]:
    return [
        Stage2ViewTask(
            index_path="/tmp/view_index.parquet",
            family_dir=f"/tmp/family-{index}",
            view_id=f"view-{index:03d}",
            family_id=f"family-{index}",
            consumer_cohort_id="cohort",
            split_id="test",
            track_id="core",
            city_slug="city",
            scale_id="Cus50",
            customer_count=50,
            charging_station_count=5,
            row_position=index,
        )
        for index in range(count)
    ]


def test_view_seed_and_shards_are_order_server_and_slice_independent(monkeypatch) -> None:
    view_ids = [f"view-{index:03d}" for index in range(41)]
    reference = {view: stable_view_seed(2026, view) for view in view_ids}
    reversed_mapping = {
        view: stable_view_seed(2026, view) for view in reversed(view_ids)
    }
    assert reference == reversed_mapping
    assert reference != {view: stable_view_seed(2027, view) for view in view_ids}

    shard_sets = [
        {view for view in view_ids if stable_view_shard(view, 7) == shard}
        for shard in range(7)
    ]
    assert set().union(*shard_sets) == set(view_ids)
    assert sum(len(items) for items in shard_sets) == len(view_ids)
    assert all(shard_sets[i].isdisjoint(shard_sets[j]) for i in range(7) for j in range(i))

    tasks = _stage2_tasks(12)
    monkeypatch.setattr(benchmark_common, "read_stage2_tasks", lambda *a, **k: tasks)
    monkeypatch.setattr(benchmark_common, "missing_family_directories", lambda selected: [])
    selected = build_input_tasks(
        "/unused",
        start_index=2,
        end_index=11,
        shard_count=3,
        shard_index=1,
    )
    expected = [
        task.view_id
        for task in tasks[2:11]
        if stable_view_shard(task.view_id, 3) == 1
    ]
    assert [item["stage2_task"]["view_id"] for item in selected] == expected
    assert {
        item["stage2_task"]["view_id"]: stable_view_seed(
            2026, item["stage2_task"]["view_id"]
        )
        for item in selected
    } == {view: reference[view] for view in expected}


def _run_contract(
    *,
    time_limit_s: float = 30.0,
    base_seed: int = 2026,
    solver_parameter: int = 20,
    host_prefix: str = "/server-a/data",
    view_seed: int = 11,
    family_id: str = "mf-contract-a",
) -> tuple[str, str]:
    view_id = "view-contract-a"
    task = {
        "input_kind": "stage2",
        "stage2_task": {
            "index_path": (
                f"{host_prefix}/generation_plan/compatibility_cus50/"
                "test/test1/view_index.parquet"
            ),
            "family_dir": f"{host_prefix}/materialized/families/mf-contract-a",
            "view_id": view_id,
            "family_id": family_id,
            "consumer_cohort_id": "compatibility_cus50",
            "split_id": "test",
            "track_id": "core",
            "city_slug": "los-angeles",
            "scale_id": "Cus50",
            "customer_count": 50,
            "charging_station_count": 5,
            "terminal_count": 56,
            "family_cohort_id": "compatibility_cus50",
            "view_seed": view_seed,
            # Execution-only fields must not enter the run identity.
            "row_position": 123,
        },
        "seed": stable_view_seed(base_seed, view_id),
        "seed_scheme": benchmark_common.SEED_SCHEME,
        "time_limit_s": time_limit_s,
        "checkpoints_s": (min(30.0, time_limit_s), time_limit_s),
        "num_workers": 30,
        "shard_index": 4,
        "save_path": "/ignored/results",
    }
    return build_run_contract(
        task,
        algorithm_name="test_solver",
        algorithm_profile_id="test_profile_v1",
        base_seed=base_seed,
        solver_parameters={"eta_dist": solver_parameter},
    )


def test_run_contract_is_portable_and_excludes_execution_layout() -> None:
    first_fingerprint, first_json = _run_contract(host_prefix="/server-a/data")
    second_fingerprint, second_json = _run_contract(host_prefix="/mnt/server-b")
    assert first_fingerprint == second_fingerprint
    assert first_json == second_json
    parsed = json.loads(first_json)
    assert parsed["algorithm"] == {
        "name": "test_solver",
        "profile_id": "test_profile_v1",
    }
    assert parsed["data_identity"]["view_index_identity"].startswith(
        "generation_plan/"
    )
    assert "num_workers" not in first_json
    assert "shard_index" not in first_json
    assert "save_path" not in first_json
    assert len(first_fingerprint) == 64


def test_run_contract_changes_for_pilot_seed_or_solver_parameter() -> None:
    formal, _ = _run_contract(time_limit_s=7200.0)
    pilot, _ = _run_contract(time_limit_s=30.0)
    changed_seed, _ = _run_contract(time_limit_s=7200.0, base_seed=2027)
    changed_parameter, _ = _run_contract(
        time_limit_s=7200.0,
        solver_parameter=21,
    )
    changed_view_seed, _ = _run_contract(time_limit_s=7200.0, view_seed=99)
    changed_data, _ = _run_contract(
        time_limit_s=7200.0,
        family_id="mf-contract-b",
    )
    assert len(
        {
            formal,
            pilot,
            changed_seed,
            changed_parameter,
            changed_view_seed,
            changed_data,
        }
    ) == 6


def test_run_contract_versions_canonical_replay_semantics(monkeypatch) -> None:
    original, original_json = _run_contract()
    assert json.loads(original_json)["canonical_replay_profile_id"] == (
        "full_charge_derated_strict_route_v3"
    )
    monkeypatch.setattr(
        benchmark_common,
        "CANONICAL_REPLAY_PROFILE_ID",
        "full_charge_strict_route_v3",
    )
    changed, _ = _run_contract()
    assert changed != original


def test_run_contract_uses_ids_without_dataset_content_hashing() -> None:
    _, contract_json = _run_contract()
    identity = json.loads(contract_json)["data_identity"]
    assert identity["identity_mode"] == (
        "deterministic_stage2_ids_no_content_hash_v1"
    )
    assert identity["expected_family_schema"] == (
        "cle_evrptw_materialized_matrix_family_v3"
    )
    assert identity["expected_view_schema"] == "cle_evrptw_materialized_view_v4"
    assert identity["expected_generation_contract"] == "stage2_construct_valid_v3"
    assert identity["expected_matrix_storage"] == "parent_index_view"
    assert "sha256" not in contract_json.lower()
    assert "artifact_fingerprint" not in contract_json


def test_bounded_process_map_never_exceeds_configured_in_flight(monkeypatch) -> None:
    observed_pending: list[int] = []

    class FakeFuture:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def submit(self, function, task):
            return FakeFuture(function(task))

    def fake_wait(pending, return_when):  # noqa: ARG001
        observed_pending.append(len(pending))
        completed = {next(iter(pending))}
        return completed, set(pending) - completed

    monkeypatch.setattr(benchmark_common, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(benchmark_common, "wait", fake_wait)
    outputs = list(
        bounded_process_map(
            lambda task: task * 2,
            range(19),
            workers=4,
            max_in_flight=5,
        )
    )
    assert sorted(outputs) == [value * 2 for value in range(19)]
    assert max(observed_pending) == 5


SUMMARY_FIELDS = [
    "instance_id",
    "status",
    "value",
    "run_contract_fingerprint",
    "run_contract_json",
]
TRACE_FIELDS = ["instance_id", "checkpoint_s", "value"]


def _store(root: Path, *, resume: bool) -> IncrementalCsvStore:
    return IncrementalCsvStore(
        summary_path=root / "summary.csv",
        trace_path=root / "trace.csv",
        summary_fieldnames=SUMMARY_FIELDS,
        trace_fieldnames=TRACE_FIELDS,
        solver_key="test",
        resume=resume,
    )


def _result(
    instance_id: str,
    value: int,
    *,
    fingerprint: str | None = "test-contract-fingerprint",
) -> dict:
    summary = {
        "instance_id": instance_id,
        "status": "COMPLETED_WITH_INCUMBENT",
        "value": value,
    }
    if fingerprint is not None:
        summary["run_contract_fingerprint"] = fingerprint
        summary["run_contract_json"] = '{"schema":"test"}'
    return {
        "summary_row": summary,
        "time_rows": [
            {"instance_id": instance_id, "checkpoint_s": 60.0, "value": value}
        ],
    }


def _write_then_wait(root: str, ready) -> None:
    store = _store(Path(root), resume=False)
    store.record_result(_result("view-a", 1))
    ready.set()
    time.sleep(30.0)


def test_jsonl_journal_survives_actual_kill_resume_without_duplicates(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    process = context.Process(target=_write_then_wait, args=(str(tmp_path), ready))
    process.start()
    assert ready.wait(5.0)
    process.terminate()
    process.join(5.0)
    assert not process.is_alive()

    resumed = _store(tmp_path, resume=True)
    try:
        assert resumed.completed_instance_ids(
            {"COMPLETED_WITH_INCUMBENT"},
            {"view-a": "test-contract-fingerprint"},
        ) == {"view-a"}
        # Re-running the same view replaces its indexed row; it never creates
        # duplicate summary/checkpoint rows.
        resumed.record_result(_result("view-a", 2))
        assert len(resumed.summary_rows) == 1
        assert len(resumed.time_rows) == 1
        assert resumed.summary_rows[0]["value"] == 2
        resumed.flush_canonical()
    finally:
        resumed.close()

    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    with (tmp_path / "trace.csv").open(newline="", encoding="utf-8") as handle:
        trace = list(csv.DictReader(handle))
    assert len(summary) == len(trace) == 1
    assert summary[0]["value"] == trace[0]["value"] == "2"


def test_journal_repairs_partial_tail_before_append_and_second_resume(
    tmp_path: Path,
) -> None:
    first = _store(tmp_path, resume=False)
    first.record_result(_result("view-a", 1))
    first.flush_canonical()
    first.close()

    resumed = _store(tmp_path, resume=True)
    resumed.record_result(_result("view-b", 2))
    journal = resumed.journal_path
    resumed.close()
    with journal.open("a", encoding="utf-8") as handle:
        handle.write('{"record_type":"result"')

    with pytest.warns(RuntimeWarning, match="repairing incomplete runner journal tail"):
        recovered = _store(tmp_path, resume=True)
    try:
        assert {row["instance_id"] for row in recovered.summary_rows} == {
            "view-a",
            "view-b",
        }
        recovered.record_result(_result("view-c", 3))
    finally:
        recovered.close()

    # This is the failure mode the repair guards against: without truncating
    # the partial bytes, view-c would be appended to the malformed JSON token
    # and lost on this second restart.
    second_resume = _store(tmp_path, resume=True)
    try:
        assert {row["instance_id"] for row in second_resume.summary_rows} == {
            "view-a",
            "view-b",
            "view-c",
        }
        assert len(second_resume.summary_rows) == 3
        with second_resume.journal_path.open(encoding="utf-8") as handle:
            assert all(json.loads(line) for line in handle if line.strip())
    finally:
        second_resume.close()


def test_journal_rejects_malformed_record_before_later_nonempty_record(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, resume=False)
    store.record_result(_result("view-a", 1))
    journal = store.journal_path
    store.close()
    later_record = {
        "record_type": "result",
        "version": IncrementalCsvStore.VERSION,
        "instance_id": "view-b",
        "summary_row": _result("view-b", 2)["summary_row"],
        "time_rows": _result("view-b", 2)["time_rows"],
    }
    with journal.open("a", encoding="utf-8") as handle:
        handle.write('{"record_type":"broken"\n')
        handle.write(json.dumps(later_record) + "\n")

    with pytest.raises(ValueError, match="before the final line"):
        _store(tmp_path, resume=True)


def test_resume_skips_only_identical_terminal_run_contract(tmp_path: Path) -> None:
    pilot_fingerprint, _ = _run_contract(time_limit_s=30.0)
    formal_fingerprint, _ = _run_contract(time_limit_s=7200.0)
    changed_seed, _ = _run_contract(time_limit_s=30.0, base_seed=2027)
    changed_parameter, _ = _run_contract(
        time_limit_s=30.0,
        solver_parameter=21,
    )
    first = _store(tmp_path, resume=False)
    first.record_result(
        _result("view-contract-a", 1, fingerprint=pilot_fingerprint)
    )
    first.flush_canonical()
    first.close()

    resumed = _store(tmp_path, resume=True)
    try:
        terminal = {"COMPLETED_WITH_INCUMBENT", "UNFINISHED_NO_INCUMBENT"}
        assert resumed.has_completed_contract(
            "view-contract-a", pilot_fingerprint, terminal
        )
        assert not resumed.has_completed_contract(
            "view-contract-a", formal_fingerprint, terminal
        )
        assert not resumed.has_completed_contract(
            "view-contract-a", changed_seed, terminal
        )
        assert not resumed.has_completed_contract(
            "view-contract-a", changed_parameter, terminal
        )
    finally:
        resumed.close()


def test_legacy_terminal_csv_without_contract_is_rerun(tmp_path: Path) -> None:
    legacy = _store(tmp_path, resume=False)
    legacy.record_result(_result("view-contract-a", 1, fingerprint=None))
    legacy.flush_canonical()
    legacy.close()

    expected, _ = _run_contract()
    resumed = _store(tmp_path, resume=True)
    try:
        with pytest.warns(RuntimeWarning, match="legacy results without"):
            assert not resumed.has_completed_contract(
                "view-contract-a",
                expected,
                {"COMPLETED_WITH_INCUMBENT"},
            )
    finally:
        resumed.close()


def test_save_path_rejects_concurrent_runner_with_actionable_warning(tmp_path: Path) -> None:
    first = _store(tmp_path, resume=False)
    try:
        with pytest.warns(RuntimeWarning, match="unique --save_path"):
            with pytest.raises(RuntimeError, match="active benchmark runner"):
                _store(tmp_path, resume=True)
    finally:
        first.close()
    second = _store(tmp_path, resume=True)
    second.close()


def test_atomic_solution_save_failure_preserves_old_target_and_cleans_temp(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "solution.pkl"
    target.write_bytes(b"previous-complete-solution")
    solution = EVRPTWSolution(
        instance_id="view-a",
        solver_name="test",
        routes=[[0, 1, 0]],
        feasible=True,
    )

    def partial_then_fail(path, value):  # noqa: ARG001
        Path(path).write_bytes(b"partial")
        raise OSError("simulated serialization failure")

    monkeypatch.setattr(benchmark_output, "save_solution", partial_then_fail)
    with pytest.raises(OSError, match="simulated serialization failure"):
        atomic_save_solution(target, solution)
    assert target.read_bytes() == b"previous-complete-solution"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_checkpoint_solution_carries_complete_run_identity(
    tmp_path: Path, monkeypatch
) -> None:
    captured: list[tuple[Path, EVRPTWSolution]] = []
    monkeypatch.setattr(
        benchmark_output,
        "atomic_save_solution",
        lambda path, solution: captured.append((Path(path), solution)),
    )
    result = {
        "summary_row": {
            "instance_id": "view-a",
            "seed": 123,
            "seed_scheme": "blake2b_view_id_v1",
            "algorithm_profile_id": "profile-v2",
            "run_contract_fingerprint": "a" * 64,
            "run_contract_json": '{"schema":"contract-v1"}',
        },
        "solution": None,
        "time_rows": [{"checkpoint_solution_path": ""}],
        "snapshots": [
            {
                "has_incumbent": True,
                "routes": [[0, 1, 0]],
                "objective_distance_km": 2.0,
                "vehicle_count": 1,
                "elapsed_s": 60.0,
                "checkpoint_s": 60.0,
                "reached_checkpoint": True,
                "incumbent_event_time_s": 1.0,
                "source": "checkpoint_incumbent",
                "benchmark_status": "INCUMBENT_AVAILABLE",
            }
        ],
    }
    save_result_artifacts(
        result,
        solver_name="test",
        solutions_dir=tmp_path / "solutions",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    assert len(captured) == 1
    assert ("a" * 64) in captured[0][0].parts
    metadata = captured[0][1].metadata
    assert metadata["seed"] == 123
    assert metadata["seed_scheme"] == "blake2b_view_id_v1"
    assert metadata["algorithm_profile_id"] == "profile-v2"
    assert metadata["run_contract_fingerprint"] == "a" * 64


def test_changed_contracts_publish_artifacts_in_disjoint_namespaces(
    tmp_path: Path, monkeypatch
) -> None:
    captured_paths: list[Path] = []
    monkeypatch.setattr(
        benchmark_output,
        "atomic_save_solution",
        lambda path, solution: captured_paths.append(Path(path)),
    )

    def publish(fingerprint: str) -> None:
        solution = EVRPTWSolution(
            instance_id="view-a",
            solver_name="test",
            routes=[[0, 1, 0]],
            objective_distance_km=2.0,
            vehicle_count=1,
            feasible=True,
            metadata={"run_contract_fingerprint": fingerprint},
        )
        save_result_artifacts(
            {
                "summary_row": {
                    "instance_id": "view-a",
                    "run_contract_fingerprint": fingerprint,
                },
                "solution": solution.to_dict(),
                "time_rows": [],
                "snapshots": [],
            },
            solver_name="test",
            solutions_dir=tmp_path / "solutions",
            checkpoints_dir=tmp_path / "checkpoints",
        )

    publish("a" * 64)
    publish("b" * 64)
    assert len(captured_paths) == 2
    assert captured_paths[0] != captured_paths[1]
    assert ("a" * 64) in captured_paths[0].parts
    assert ("b" * 64) in captured_paths[1].parts


def test_trace_rows_carry_portable_run_identity_for_standalone_merging() -> None:
    provenance = {
        "solver_name": "test-solver",
        "algorithm_profile_id": "test-profile-v3",
        "seed": 123,
        "seed_scheme": "blake2b_view_id_v1",
        "run_contract_fingerprint": "a" * 64,
    }
    snapshots = [
        {
            "checkpoint_s": 60.0,
            "elapsed_s": 60.0,
            "reached_checkpoint": True,
            "status": "RUNNING",
            "benchmark_status": "INCUMBENT_AVAILABLE",
            "has_incumbent": True,
            "incumbent_event_time_s": 1.0,
            "objective_distance_km": 2.0,
            "vehicle_count": 1,
            "routes": [[0, 1, 0]],
            "route_sequence": [0, 1, 0],
            "source": "checkpoint_incumbent",
        }
    ]
    rows = snapshot_rows(
        "view-a",
        {"file": "view_index.parquet", "family_id": "mf-a"},
        snapshots,
        1.0,
        provenance=provenance,
    )
    errors = error_snapshot_rows(
        "view-a",
        {"file": "view_index.parquet", "family_id": "mf-a"},
        (60.0,),
        "ERROR",
        "failure",
        provenance=provenance,
    )
    for row in (*rows, *errors):
        for key, value in provenance.items():
            assert row[key] == value


def test_replay_cache_only_skips_exact_duplicate_routes(monkeypatch) -> None:
    instance = make_multihop_certificate_instance()
    calls = []
    real_validate = benchmark_common.validate_routes

    def counted_validate(current_instance, routes):
        calls.append([list(route) for route in routes])
        return real_validate(current_instance, routes)

    monkeypatch.setattr(benchmark_common, "validate_routes", counted_validate)
    cache = IncumbentReplayCache(instance)
    route = [[0, 1, 2, 3, 0]]
    first = cache.validate(route)
    second = cache.validate([list(route[0])])
    assert first == second
    assert len(calls) == 1
    assert (cache.hits, cache.misses) == (1, 1)

    # Any real route change is replayed, even if it is worse or infeasible.
    changed = cache.validate([[0, 1, 2, 0]])
    assert len(calls) == 2
    assert not changed["passed"]


def test_recorder_preserves_real_sub_nanometer_objective_improvement() -> None:
    recorder = IncumbentEventRecorder((60.0,), 60.0)
    recorder.observe(1.0, 100.0, [[0, 1, 0]])
    improved = 100.0 - 5e-10
    recorder.observe(2.0, improved, [[0, 2, 0]])
    assert recorder.best_event["objective_distance_km"] == improved
    assert recorder.best_event["routes"] == [[0, 2, 0]]


def _load_runner_with_fake_solver(
    monkeypatch,
    *,
    runner_path: Path,
    solver_class_name: str,
    fake_solver_class,
    adapter_module_name: str,
    adapter_function_name: str,
    adapter_function=None,
):
    fake_solver_module = types.ModuleType("solver")
    setattr(fake_solver_module, solver_class_name, fake_solver_class)
    fake_adapter_module = types.ModuleType(adapter_module_name)
    setattr(
        fake_adapter_module,
        adapter_function_name,
        adapter_function or (lambda instance: {"instance": instance}),
    )
    monkeypatch.setitem(sys.modules, "solver", fake_solver_module)
    monkeypatch.setitem(sys.modules, adapter_module_name, fake_adapter_module)
    name = f"runner_timing_test_{solver_class_name}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(name, runner_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeALNSSolver:
    constructor_delay_s = 0.03
    solve_budget_s = None

    def __init__(self, instance, seed, format):  # noqa: A002, ARG002
        time.sleep(type(self).constructor_delay_s)
        self.max_iters = 200
        self.cur_iter = 0
        self.terminated_by_time_limit = False

    def solve(self, *, delta_iters, time_limit_s, incumbent_callback):  # noqa: ARG002
        type(self).solve_budget_s = time_limit_s
        incumbent_callback(0.0, 0.0, [[0, 1, 2, 3, 0]])

    def get_run_metadata(self):
        return {
            "algorithm_profile_id": "fake-alns-profile",
            "initial_construction_strategy": "fake-construction",
            "singleton_source": "stage2_certificate_replayed",
            "initial_construction": {"elapsed_s": 0.01, "result_route_count": 1},
        }


class FakeVNSTSolver:
    constructor_delay_s = 0.03
    solve_budget_s = None

    def __init__(self, instance, **kwargs):  # noqa: ARG002
        time.sleep(type(self).constructor_delay_s)
        self.terminated_by_time_limit = False
        self.search_mode = kwargs["search_mode"]
        for key, value in kwargs.items():
            setattr(self, key, value)

    def solve(self, *, time_limit_s, incumbent_callback):
        type(self).solve_budget_s = time_limit_s
        incumbent_callback(0.0, 0.0, [[0, 1, 2, 3, 0]])

    def effective_fast_policy(self):
        return {
            "version": "fake-policy-v1",
            "move_candidate_limit": self.move_candidate_limit,
            "route_neighbor_limit": self.route_neighbor_limit,
            "position_neighbor_limit": self.position_neighbor_limit,
            "exchange_neighbor_limit": self.exchange_neighbor_limit,
            "station_candidate_limit": self.station_candidate_limit,
        }


def _common_runner_task(time_limit_s: float) -> dict:
    return {
        "input_kind": "stage2",
        "stage2_task": {
            "view_id": "multihop-certificate",
            "index_path": "/tmp/view_index.parquet",
            "family_id": "mf-test",
            "city_slug": "test",
            "split_id": "test",
            "track_id": "core",
            "scale_id": "Cus1",
        },
        "seed": 1,
        "time_limit_s": time_limit_s,
        "checkpoints_s": (time_limit_s,),
        "verbose": False,
        "save_traceback": True,
    }


def _patch_runner_input(monkeypatch, runner, instance):
    monkeypatch.setattr(
        runner,
        "load_input_task",
        lambda task: (
            instance,
            {
                "file": "/tmp/view_index.parquet",
                "family_id": "mf-test",
                "city_slug": "test",
                "split_id": "test",
                "track_id": "core",
                "scale_id": "Cus1",
            },
        ),
    )
    monkeypatch.setattr(
        runner,
        "validate_instance_structure",
        lambda instance: SimpleNamespace(success=True, errors=[]),
    )


def test_both_runner_clocks_include_constructor_and_pass_only_remaining_budget(monkeypatch) -> None:
    instance = make_multihop_certificate_instance()

    def slow_adapter(current_instance):
        time.sleep(0.02)
        return {"instance": current_instance}

    FakeALNSSolver.constructor_delay_s = 0.03
    alns = _load_runner_with_fake_solver(
        monkeypatch,
        runner_path=META_ROOT / "ALNS_Solver" / "run_alns.py",
        solver_class_name="ALNS_Solver",
        fake_solver_class=FakeALNSSolver,
        adapter_module_name="instance_adapter",
        adapter_function_name="to_alns_tensor_instance",
        adapter_function=slow_adapter,
    )
    _patch_runner_input(monkeypatch, alns, instance)
    alns_task = {
        **_common_runner_task(0.5),
        "max_iters": None,
        "delta_iters": 1,
    }
    alns_result = alns.solve_one(alns_task)["summary_row"]
    assert alns_result["runtime_s"] >= 0.045
    assert 0.0 < FakeALNSSolver.solve_budget_s < 0.46
    assert alns_result["timing_scope"] == ALGORITHM_TIMING_SCOPE
    assert alns_result["algorithm_profile_id"] == "fake-alns-profile"

    FakeVNSTSolver.constructor_delay_s = 0.03
    vns = _load_runner_with_fake_solver(
        monkeypatch,
        runner_path=META_ROOT / "VNS_TS_Solver" / "run_vns_ts.py",
        solver_class_name="VNSTSolver",
        fake_solver_class=FakeVNSTSolver,
        adapter_module_name="vnst_adapter",
        adapter_function_name="to_vnst_instance",
        adapter_function=slow_adapter,
    )
    _patch_runner_input(monkeypatch, vns, instance)
    vns_task = {
        **_common_runner_task(0.5),
        "predefine_route_number": 3,
        "eta_feas": 1,
        "eta_dist": 1,
        "eta_dist_requested": 1,
        "search_budget_mode": "iteration_limited",
        "tabu_iter": 1,
        "tabu_tenure": 3,
        "k_max": 2,
        "search_mode": "fast",
        "move_candidate_limit": 10,
        "route_neighbor_limit": 2,
        "position_neighbor_limit": 2,
        "exchange_neighbor_limit": 3,
        "station_candidate_limit": 2,
    }
    vns_result = vns.solve_one(vns_task)["summary_row"]
    assert vns_result["runtime_s"] >= 0.045
    assert 0.0 < FakeVNSTSolver.solve_budget_s < 0.46
    assert vns_result["timing_scope"] == ALGORITHM_TIMING_SCOPE
    assert vns_result["algorithm_profile_id"] == "vns_ts_stage2_adaptive_fast_v4"
    assert (
        vns_result["initial_construction_strategy"]
        == "certificate_singleton_best_fit_v1"
    )
    assert vns_result["fast_policy_version"] == "fake-policy-v1"
    assert vns_result["initial_route_count"] == 1


def test_constructor_timeout_is_clean_unfinished_result(monkeypatch) -> None:
    instance = make_multihop_certificate_instance()
    FakeALNSSolver.constructor_delay_s = 0.2
    runner = _load_runner_with_fake_solver(
        monkeypatch,
        runner_path=META_ROOT / "ALNS_Solver" / "run_alns.py",
        solver_class_name="ALNS_Solver",
        fake_solver_class=FakeALNSSolver,
        adapter_module_name="instance_adapter",
        adapter_function_name="to_alns_tensor_instance",
    )
    _patch_runner_input(monkeypatch, runner, instance)
    result = runner.solve_one(
        {
            **_common_runner_task(0.03),
            "max_iters": None,
            "delta_iters": 1,
        }
    )["summary_row"]
    assert result["status"] == "UNFINISHED_NO_INCUMBENT"
    assert result["terminated_by_time_limit"] is True
    assert result["has_incumbent"] is False
    assert 0.02 <= result["runtime_s"] < 0.15


def test_vns_default_budget_is_wall_clock_but_explicit_cap_remains_for_smoke() -> None:
    effective, mode = resolve_optional_iteration_budget(None)
    assert effective == 2_147_483_647
    assert mode == "wall_clock"
    assert resolve_optional_iteration_budget(20) == (20, "iteration_limited")
    with pytest.raises(ValueError):
        resolve_optional_iteration_budget(-1)


def _current_view_index_row() -> dict[str, object]:
    return {
        "view_id": "iv-current",
        "family_id": "mf-current",
        "family_cohort_id": "core/test/test1_new_seed",
        "consumer_cohort_id": "compatibility_cus50/test/test1_new_seed_same_cities",
        "split_id": "test",
        "track_id": "test1_new_seed",
        "city_slug": "new-york",
        "scale_id": "cus50",
        "customer_count": 50,
        "charging_station_count": 10,
        "terminal_count": 61,
        "view_seed": 123456789,
    }


def test_current_view_index_needs_no_removed_attribute_seed_columns(
    tmp_path: Path,
) -> None:
    index = (
        tmp_path
        / "generation_plan"
        / "compatibility_cus50"
        / "test"
        / "view_index.parquet"
    )
    index.parent.mkdir(parents=True)
    pd.DataFrame([_current_view_index_row()]).to_parquet(index, index=False)
    family = tmp_path / "materialized" / "families" / "mf-current"
    family.mkdir(parents=True)
    (family / "family_manifest.json").write_text("{}", encoding="utf-8")

    tasks = read_stage2_tasks(index)
    assert len(tasks) == 1
    assert tasks[0].view_id == "iv-current"
    assert tasks[0].terminal_count == 61
    assert tasks[0].view_seed == 123456789
