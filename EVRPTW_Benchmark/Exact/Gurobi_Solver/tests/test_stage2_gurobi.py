from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gurobipy import GurobiError

from evrptw_core.schema import EVRPTWInstance
from gurobi_solver import GurobiEVRPTWSolver, GurobiSolverConfig
from route_validator import validate_routes
from stage2_adapter import read_stage2_tasks


def _charging_fixture(*, power_kw: float = 100.0) -> EVRPTWInstance:
    # Terminal order: depot, customer, CS. The direct customer round trip uses
    # 120 kWh and is impossible. A route through the CS is feasible.
    distance = np.asarray(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    travel = np.asarray(
        [[0.0, 100.0, 100.0], [100.0, 0.0, 100.0], [100.0, 100.0, 0.0]],
        dtype=np.float32,
    )
    energy = np.asarray(
        [[0.0, 60.0, 40.0], [60.0, 0.0, 20.0], [40.0, 20.0, 0.0]],
        dtype=np.float32,
    )
    payload = {
        "instance_id": f"charge_fixture_{power_kw:g}",
        "region_id": "test",
        "mother_board_id": "test",
        "operating_day_id": "test",
        "day_type": "weekday",
        "working_start_s": 0,
        "working_end_s": 40_000,
        "depot": np.asarray([0.0, 0.0], dtype=np.float32),
        "customers": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "charging_stations": np.asarray([[2.0, 0.0]], dtype=np.float32),
        "distance_matrix_km": distance,
        "demands_cm3": np.asarray([1.0], dtype=np.float32),
        "package_counts": np.asarray([1], dtype=np.int32),
        "service_time_s": np.asarray([0.0], dtype=np.float32),
        "tw_s": np.asarray([[0.0, 40_000.0]], dtype=np.float32),
        "cs_time_to_depot_s": np.asarray([100.0], dtype=np.float32),
        "vehicle": {
            "battery_capacity_kwh": 100.0,
            "cargo_capacity_cm3": 10.0,
            "charging_power_derating_factor": 0.9,
        },
        "raw_travel_time_matrix_s": travel,
        "energy_matrix_kwh": energy,
        "charging_power_kw": np.asarray([power_kw], dtype=np.float32),
        "charging_policy": {"charging_power_derating_factor": 0.9},
        "running_time_shortest_matrix_s": travel,
        "running_time_path_energy_kwh": energy,
    }
    return EVRPTWInstance.from_dict(payload)


def test_route_replay_uses_arrival_soc_and_station_power() -> None:
    dc = validate_routes(_charging_fixture(power_kw=100.0), [[0, 2, 1, 0]])
    l2 = validate_routes(_charging_fixture(power_kw=11.0), [[0, 2, 1, 0]])

    assert dc["passed"] is True
    assert l2["passed"] is True
    # Depot -> CS consumes 40 kWh, so full charging restores exactly 40 kWh.
    assert dc["total_charging_time_s"] == 40.0 / (0.9 * 100.0) * 3600.0
    assert l2["total_charging_time_s"] == 40.0 / (0.9 * 11.0) * 3600.0


def test_gurobi_dynamic_charge_time_matches_independent_replay() -> None:
    instance = _charging_fixture(power_kw=100.0)
    solver = GurobiEVRPTWSolver(
        GurobiSolverConfig(
            time_limit_s=30.0,
            cs_copies=1,
            output_flag=0,
            threads=1,
            tie_break_vehicle_count=False,
        )
    )
    try:
        solution = solver.solve(instance)
    except GurobiError as exc:
        if "license" in str(exc).lower() or "hostid" in str(exc).lower():
            pytest.skip(f"Gurobi license unavailable on this host: {exc}")
        raise

    assert solution.feasible is True
    assert solution.metadata["benchmark_status"] == "COMPLETED_OPTIMAL"
    assert solution.metadata["benchmark_completed"] is True
    assert solution.metadata["gurobi_status_name"] == "OPTIMAL"
    assert solution.metadata["travel_time_matrix_source"] == (
        "running_time_shortest_matrix_s"
    )
    assert solution.metadata["energy_matrix_source"] == (
        "running_time_path_energy_kwh"
    )
    assert solution.metadata["charging_power_derating_factor"] == 0.9
    assert solution.metadata["charging_power_factor_source"] == (
        "charging_policy.charging_power_derating_factor"
    )
    replay = solution.metadata["route_validation"]
    assert replay["passed"] is True
    assert replay["charging_visit_count"] == 1
    assert 2 in solution.routes[0]
    charge_var = solver.model.getVarByName("charge_time[2]")
    assert math.isclose(
        float(charge_var.X),
        float(replay["total_charging_time_s"]),
        rel_tol=1e-9,
        abs_tol=1e-7,
    )
    checkpoints = solution.metadata["checkpoint_snapshots"]
    assert [row["checkpoint_s"] for row in checkpoints] == [
        60.0,
        300.0,
        900.0,
        3600.0,
        7200.0,
    ]
    assert all(row["routes"] == solution.routes for row in checkpoints)
    assert all(row["objective_distance_km"] == solution.objective_distance_km for row in checkpoints)
    assert all(row["route_validation_passed"] is True for row in checkpoints)


def test_direct_route_is_rejected_when_cumulative_energy_exceeds_battery() -> None:
    replay = validate_routes(_charging_fixture(), [[0, 1, 0]])
    assert replay["passed"] is False
    assert any("exceeds battery" in item for item in replay["violations"])


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
