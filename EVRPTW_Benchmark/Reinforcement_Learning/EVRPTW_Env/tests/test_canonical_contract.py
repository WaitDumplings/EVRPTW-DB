from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from evrptw_core.schema import EVRPTWInstance

from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import (
    EVRPTWVectorEnv,
    EVRPTWVectorEnvFast,
)


def canonical_instance(*, include_matrices: bool = True) -> EVRPTWInstance:
    distance = np.asarray(
        [
            [0.0, 100.0, 4.0],
            [8.0, 0.0, 3.0],
            [5.0, 6.0, 0.0],
        ],
        dtype=np.float32,
    )
    travel_time = np.asarray(
        [
            [0.0, 11.0, 17.0],
            [19.0, 0.0, 23.0],
            [29.0, 31.0, 0.0],
        ],
        dtype=np.float32,
    )
    energy = np.asarray(
        [
            [0.0, 2.0, 1.0],
            [2.5, 0.0, 1.5],
            [1.0, 1.5, 0.0],
        ],
        dtype=np.float32,
    )
    raw = {
        "charging_power_kw": np.asarray([20.0], dtype=np.float32),
        "charging_policy": {"charging_power_derating_factor": 0.5},
    }
    return EVRPTWInstance(
        instance_id="canonical_contract_test",
        region_id="test",
        mother_board_id="test",
        operating_day_id="test",
        day_type="weekday",
        working_start_s=0,
        working_end_s=100_000,
        depot=np.asarray([0.0, 0.0], dtype=np.float32),
        customers=np.asarray([[1.0, 0.0]], dtype=np.float32),
        charging_stations=np.asarray([[0.5, 0.0]], dtype=np.float32),
        distance_matrix_km=distance,
        demands_cm3=np.asarray([1.0], dtype=np.float32),
        package_counts=np.asarray([1], dtype=np.int32),
        service_time_s=np.asarray([7.0], dtype=np.float32),
        tw_s=np.asarray([[0.0, 100_000.0]], dtype=np.float32),
        cs_time_to_depot_s=np.asarray([29.0], dtype=np.float32),
        vehicle={
            "battery_capacity_kwh": 10.0,
            "cargo_capacity_cm3": 10.0,
            "specific_energy_consumption_kwh_per_km": 0.4,
        },
        shortest_time_matrix_s=travel_time if include_matrices else None,
        energy_matrix_kwh=energy if include_matrices else None,
        raw=raw,
    )


def test_canonical_env_uses_exported_time_and_energy_matrices() -> None:
    env = EVRPTWVectorEnv(canonical_instance(), n_traj=1)
    assert env.travel_time_source == "EVRPTWInstance.shortest_time_matrix_s"
    assert env.energy_source == "EVRPTWInstance.energy_matrix_kwh"
    assert env.travel_time_s[0, 1] == 11.0
    assert env.energy_kwh[0, 1] == 2.0
    assert env.travel_time_s[0, 1] != env.distance_km[0, 1] / (40.0 / 3600.0)


def test_station_charge_time_uses_arrival_soc_power_and_derating() -> None:
    env = EVRPTWVectorEnv(canonical_instance(), n_traj=1)
    station = env.station_start
    expected_s = 3600.0 * 1.0 / (20.0 * 0.5)
    assert env._charge_time_s(1.0, station) == pytest.approx(expected_s)

    obs, _ = env.reset(seed=7)
    assert bool(obs["action_mask"][0, station])
    _, _, _, _, _ = env.step(np.asarray([station], dtype=np.int64))
    assert env.current_time_s[0] == pytest.approx(17.0 + expected_s)
    assert env.battery_used_kwh[0] == 0.0


def test_fast_and_reference_masks_follow_same_canonical_contract() -> None:
    instance = canonical_instance()
    reference = EVRPTWVectorEnv(instance, n_traj=2)
    fast = EVRPTWVectorEnvFast(instance, n_traj=2, use_jit_mask=True)
    ref_obs, _ = reference.reset(seed=9)
    fast_obs, _ = fast.reset(seed=9)
    np.testing.assert_array_equal(ref_obs["action_mask"], fast_obs["action_mask"])
    np.testing.assert_allclose(
        ref_obs["charging_power"], fast_obs["charging_power"]
    )
    np.testing.assert_allclose(ref_obs["remaining_demand"], fast_obs["remaining_demand"])
    np.testing.assert_allclose(
        ref_obs["remaining_vehicle_ratio"],
        fast_obs["remaining_vehicle_ratio"],
    )


def test_canonical_mode_rejects_missing_matrices() -> None:
    with pytest.raises(ValueError, match="running_time_shortest_matrix_s"):
        EVRPTWVectorEnv(canonical_instance(include_matrices=False), n_traj=1)
