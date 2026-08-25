from __future__ import annotations

import importlib.util
import ast
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
META_ROOT = REPO_ROOT / "EVRPTW_Benchmark" / "MetaHeuristics"
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Dataset_Generator" / "src"))
sys.path.insert(0, str(META_ROOT))
sys.path.insert(0, str(META_ROOT / "ALNS_Solver"))
sys.path.insert(0, str(META_ROOT / "VNS_TS_Solver"))

from evrptw_core.schema import EVRPTWInstance

from benchmark_common import (
    IncumbentEventRecorder,
    charging_profile,
    parse_checkpoints,
    resolve_schedule,
    validate_routes,
)
from benchmark_output import TIME_TRACE_FIELDNAMES
from instance_adapter import to_alns_tensor_instance
from vnst_adapter import to_vnst_instance


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_instance() -> EVRPTWInstance:
    # Canonical order: depot, customer, 11 kW station, 100 kW station.
    distance = np.array(
        [
            [0.0, 1.0, 2.0, 2.0],
            [1.0, 0.0, 1.0, 1.0],
            [2.0, 1.0, 0.0, 2.0],
            [2.0, 1.0, 2.0, 0.0],
        ]
    )
    travel = np.zeros((4, 4), dtype=float)
    energy = np.zeros((4, 4), dtype=float)
    energy[0, 1] = 10.0
    energy[1, 2] = 10.0
    energy[1, 3] = 10.0
    return EVRPTWInstance.from_dict(
        {
            "instance_id": "iv_test",
            "region_id": "test-city",
            "mother_board_id": "mf_test",
            "operating_day_id": "mf_test",
            "day_type": "weekday",
            "working_start_s": 0,
            "working_end_s": 20_000,
            "depot": [0.0, 0.0],
            "customers": [[1.0, 0.0]],
            "charging_stations": [[2.0, 0.0], [3.0, 0.0]],
            "distance_matrix_km": distance,
            "demands_cm3": [1.0],
            "package_counts": [1],
            "service_time_s": [0.0],
            "tw_s": [[0.0, 20_000.0]],
            "cs_time_to_depot_s": [0.0, 0.0],
            "vehicle": {
                "battery_capacity_kwh": 100.0,
                "cargo_capacity_cm3": 100.0,
                "consumption_kwh_per_km": 99.0,
                "charging_power_derating_factor": 0.9,
            },
            "raw_travel_time_matrix_s": travel,
            "energy_matrix_kwh": energy,
            "running_time_shortest_matrix_s": travel,
            "running_time_path_energy_kwh": energy,
            "charging_power_kw": [11.0, 100.0],
            "charging_policy": {
                "policy": "full_charge_linear_derated_v2",
                "charging_power_derating_factor": 0.9,
            },
            "cs_activation": {"charging_power_kw": [11.0, 100.0]},
        }
    )


def test_adapters_preserve_stage2_time_energy_and_station_power() -> None:
    instance = make_instance()
    power, power_factor, source = charging_profile(instance)
    assert power.tolist() == [11.0, 100.0]
    assert power_factor == 0.9
    assert source == "charging_power_kw"

    alns = to_alns_tensor_instance(instance)
    np.testing.assert_allclose(alns["time_matrix_min"], 0.0)
    assert alns["energy_matrix_kwh"][0, 1] == 10.0
    assert alns["charging_power_kw"].tolist() == [11.0, 100.0]

    vnst = to_vnst_instance(instance)
    # VNS internal order is depot, stations, customers.
    assert vnst.energy_matrix[vnst.terminal_order.index(0), vnst.terminal_order.index(1)] == 10.0
    assert vnst.station_charging_power_kw == {2: 11.0, 3: 100.0}


def test_route_replay_uses_station_specific_full_charge_time() -> None:
    instance = make_instance()
    slow = validate_routes(instance, [[0, 1, 2, 0]])
    fast = validate_routes(instance, [[0, 1, 3, 0]])
    assert slow["passed"]
    assert fast["passed"]
    assert np.isclose(slow["total_charging_time_s"], 20.0 / (0.9 * 11.0) * 3600.0)
    assert np.isclose(fast["total_charging_time_s"], 20.0 / (0.9 * 100.0) * 3600.0)
    assert slow["total_charging_time_s"] > fast["total_charging_time_s"]


def test_both_solvers_use_station_specific_full_charge_time() -> None:
    instance = make_instance()
    alns_module = load_module(
        "alns_solver_stage2_test", META_ROOT / "ALNS_Solver" / "solver.py"
    )
    alns = alns_module.ALNS_Solver(
        to_alns_tensor_instance(instance), seed=1, format="tensor"
    )
    slow_sim = alns._simulate_route([0, 1, 2, 0])
    fast_sim = alns._simulate_route([0, 1, 3, 0])
    assert np.isclose(slow_sim["departure_times"][2], 20.0 * 60.0 / (0.9 * 11.0))
    assert np.isclose(fast_sim["departure_times"][2], 20.0 * 60.0 / (0.9 * 100.0))

    vnst_module = load_module(
        "vnst_solver_stage2_test", META_ROOT / "VNS_TS_Solver" / "solver.py"
    )
    adapted = to_vnst_instance(instance)
    vnst = vnst_module.VNSTSolver(adapted)
    assert np.isclose(vnst.charging_time(adapted.stations[0], 20.0), 20.0 / (0.9 * 11.0) * 3600.0)
    assert np.isclose(vnst.charging_time(adapted.stations[1], 20.0), 20.0 / (0.9 * 100.0) * 3600.0)


def test_canonical_replay_rejects_internal_depot_and_customerless_routes() -> None:
    instance = make_instance()
    internal_depot = validate_routes(instance, [[0, 1, 0, 2, 0]])
    customerless_extra_route = validate_routes(
        instance,
        [[0, 1, 0], [0, 2, 0]],
    )
    assert not internal_depot["passed"]
    assert any("internal depot" in item for item in internal_depot["violations"])
    assert not customerless_extra_route["passed"]
    assert any(
        "no customer" in item for item in customerless_extra_route["violations"]
    )


def test_checkpoint_snapshots_never_backfill_late_incumbent() -> None:
    recorder = IncumbentEventRecorder((60.0, 300.0, 900.0, 1200.0), 1200.0)
    recorder.observe(30.0, 100.0, [[0, 1, 0]])
    recorder.observe(200.0, 80.0, [[0, 2, 0]])
    snapshots = recorder.snapshots(
        runtime_s=1000.0,
        natural_completion=False,
        final_status="COMPLETED_WITH_INCUMBENT",
    )
    assert snapshots[0]["objective_distance_km"] == 100.0
    assert snapshots[0]["incumbent_event_time_s"] == 30.0
    assert snapshots[0]["benchmark_status"] == "INCUMBENT_AVAILABLE"
    assert snapshots[1]["objective_distance_km"] == 80.0
    assert snapshots[2]["objective_distance_km"] == 80.0
    assert not snapshots[3]["has_incumbent"]
    assert snapshots[3]["source"] == "checkpoint_not_reached"
    assert snapshots[3]["benchmark_status"] == "COMPLETED_WITH_INCUMBENT"


def test_default_schedule_follows_explicit_time_limit() -> None:
    checkpoints, limit = resolve_schedule(parse_checkpoints(""), None)
    assert checkpoints == (60.0, 300.0, 900.0, 3600.0, 7200.0)
    assert limit == 7200.0

    checkpoints, limit = resolve_schedule(parse_checkpoints(""), 60.0)
    assert checkpoints == (60.0,)
    assert limit == 60.0

    checkpoints, limit = resolve_schedule(parse_checkpoints(""), 400.0)
    assert checkpoints == (60.0, 300.0, 400.0)
    assert limit == 400.0

    checkpoints, limit = resolve_schedule(parse_checkpoints("300"), 60.0)
    assert checkpoints == (300.0,)
    assert limit == 300.0


def test_checkpoint_comparison_has_no_post_checkpoint_epsilon() -> None:
    recorder = IncumbentEventRecorder((60.0, 300.0), 300.0)
    recorder.observe(30.0, 100.0, [[0, 1, 0]])
    recorder.observe(60.0 + 5e-10, 80.0, [[0, 2, 0]])
    snapshots = recorder.snapshots(
        runtime_s=300.0,
        natural_completion=False,
        final_status="COMPLETED_WITH_INCUMBENT",
    )
    assert snapshots[0]["objective_distance_km"] == 100.0
    assert snapshots[0]["incumbent_event_time_s"] == 30.0
    assert snapshots[1]["objective_distance_km"] == 80.0


def test_early_only_checkpoint_is_not_marked_as_terminal_no_incumbent() -> None:
    recorder = IncumbentEventRecorder((60.0,), 7200.0)
    snapshots = recorder.snapshots(
        runtime_s=7200.0,
        natural_completion=False,
        final_status="UNFINISHED_NO_INCUMBENT",
    )
    assert snapshots[0]["reached_checkpoint"]
    assert snapshots[0]["status"] == "RUNNING"
    assert snapshots[0]["benchmark_status"] == "NO_INCUMBENT_YET"

    terminal = IncumbentEventRecorder((60.0, 7200.0), 7200.0).snapshots(
        runtime_s=7200.0,
        natural_completion=False,
        final_status="UNFINISHED_NO_INCUMBENT",
    )
    assert terminal[0]["benchmark_status"] == "NO_INCUMBENT_YET"
    assert terminal[1]["benchmark_status"] == "UNFINISHED_NO_INCUMBENT"


def test_natural_early_completion_only_forward_fills_future() -> None:
    recorder = IncumbentEventRecorder((60.0, 300.0, 900.0), 900.0)
    recorder.observe(200.0, 80.0, [[0, 1, 0]])
    snapshots = recorder.snapshots(
        runtime_s=250.0,
        natural_completion=True,
        final_status="COMPLETED_WITH_INCUMBENT",
    )
    assert not snapshots[0]["has_incumbent"]
    assert snapshots[1]["objective_distance_km"] == 80.0
    assert snapshots[1]["source"] == "final_after_early_stop"
    assert snapshots[2]["objective_distance_km"] == 80.0


def test_exact_compatible_status_fields_are_declared() -> None:
    assert {
        "benchmark_status",
        "solver_name",
        "algorithm_profile_id",
        "seed",
        "seed_scheme",
        "run_contract_fingerprint",
    } <= set(TIME_TRACE_FIELDNAMES)
    required = {
        "benchmark_status",
        "benchmark_completed",
        "has_incumbent",
        "run_contract_fingerprint",
        "run_contract_json",
    }
    for runner in (
        META_ROOT / "ALNS_Solver" / "run_alns.py",
        META_ROOT / "VNS_TS_Solver" / "run_vns_ts.py",
    ):
        tree = ast.parse(runner.read_text(encoding="utf-8"))
        fields = None
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(isinstance(target, ast.Name) and target.id == "SUMMARY_FIELDNAMES" for target in node.targets):
                fields = set(ast.literal_eval(node.value))
                break
        assert fields is not None
        assert required <= fields
