from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from evrptw_core.schema import EVRPTWInstance

from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.env import DRLTSHardConstraintEnv
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.soft_env import DRLTSSoftConstraintEnv
from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import ObjectiveConfig
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import (
    EVRPTWVectorEnv,
    EVRPTWVectorEnvFast,
)


def economic_instance() -> EVRPTWInstance:
    """Two customers and two stations with unconstraining canonical physics."""
    distance = np.asarray(
        [
            [0, 2, 3, 4, 5],
            [3, 0, 4, 2, 3],
            [2, 3, 0, 4, 2],
            [5, 3, 2, 0, 1],
            [4, 2, 3, 2, 0],
        ],
        dtype=np.float32,
    )
    return EVRPTWInstance(
        instance_id="economic_objective_test",
        region_id="test",
        mother_board_id="test",
        operating_day_id="test",
        day_type="weekday",
        working_start_s=0,
        working_end_s=100_000,
        depot=np.asarray([0, 0], dtype=np.float32),
        customers=np.asarray([[1, 0], [0, 1]], dtype=np.float32),
        charging_stations=np.asarray([[0.5, 0], [0, 0.5]], dtype=np.float32),
        distance_matrix_km=distance,
        demands_cm3=np.ones(2, dtype=np.float32),
        package_counts=np.ones(2, dtype=np.int32),
        service_time_s=np.ones(2, dtype=np.float32),
        tw_s=np.asarray([[0, 100_000], [0, 100_000]], dtype=np.float32),
        cs_time_to_depot_s=distance[3:, 0] * 10,
        vehicle={
            "battery_capacity_kwh": 100.0,
            "cargo_capacity_cm3": 10.0,
            # Deliberately not the economic consumption coefficient. Physical
            # SOC uses the exported matrix; economic electricity uses distance.
            "specific_energy_consumption_kwh_per_km": 0.9,
        },
        shortest_time_matrix_s=distance * 10,
        energy_matrix_kwh=distance * 0.7,
        raw={"charging_power_kw": np.asarray([20, 30], dtype=np.float32)},
    )


def cost_objective() -> ObjectiveConfig:
    return ObjectiveConfig(
        mode="energy_vehicle_cost", profile_id="rivian_energy_vehicle_cost_v1"
    )


def make_env(env_cls=EVRPTWVectorEnv, **kwargs):
    if issubclass(env_cls, EVRPTWVectorEnvFast):
        kwargs.setdefault("use_jit_mask", False)
    return env_cls(economic_instance(), **kwargs)


def assert_cost_ledger(env, info, distance, vehicles):
    objective = env.objective_config
    np.testing.assert_allclose(info["objective_distance_km"], distance)
    np.testing.assert_array_equal(info["vehicles_started"], vehicles)
    np.testing.assert_allclose(
        info["electricity_cost_usd"], np.asarray(distance) * objective.distance_unit_cost
    )
    np.testing.assert_allclose(
        info["vehicle_cost_usd"], np.asarray(vehicles) * objective.vehicle_unit_cost
    )
    expected = objective.value(np.asarray(distance), np.asarray(vehicles))
    np.testing.assert_allclose(info["objective_value"], expected)
    np.testing.assert_allclose(info["objective_cost_usd"], expected)
    assert info["objective_config"] == objective.to_dict()


@pytest.mark.parametrize(
    "env_cls", [EVRPTWVectorEnv, EVRPTWVectorEnvFast, DRLTSHardConstraintEnv, DRLTSSoftConstraintEnv]
)
@pytest.mark.parametrize("normalize_reward", [False, True])
def test_vehicle_fee_is_debited_on_departure_once_and_rewards_telescope(env_cls, normalize_reward):
    env = make_env(env_cls, objective_config=cost_objective(), normalize_reward=normalize_reward)
    obs, info = env.reset(seed=7)
    assert_cost_ledger(env, info, [0], [0])
    scale = env.reward_objective_scale if normalize_reward else 1.0
    reward_total = 0.0
    distance_total = 0.0
    previous = 0
    # A charging stop does not start another vehicle; returning home does not
    # charge again. The second departure does, before the route is complete.
    for action, vehicles, closed in [(1, 1, 0), (3, 1, 0), (0, 1, 1), (2, 2, 1), (0, 2, 2)]:
        assert obs["action_mask"][0, action]
        step_distance = float(env.distance_km[previous, action])
        departure = int(previous == 0 and action != 0)
        obs, reward, terminated, truncated, info = env.step([action])
        expected_delta = env.objective_config.value(step_distance, departure)
        np.testing.assert_allclose(reward, [-expected_delta / scale], rtol=1e-6)
        reward_total += float(reward[0])
        distance_total += step_distance
        assert_cost_ledger(env, info, [distance_total], [vehicles])
        np.testing.assert_array_equal(info["vehicle_count"], [closed])
        previous = action
    assert terminated[0] and not truncated[0]
    assert info["routes"] == [[[0, 1, 3, 0], [0, 2, 0]]]
    assert reward_total == pytest.approx(
        -env.objective_config.value(distance_total, 2) / scale, rel=1e-6
    )
    # Arbitrary finished padding, even a non-depot action, cannot debit again.
    _, reward, _, _, after_padding = env.step([1])
    np.testing.assert_array_equal(reward, [0])
    assert_cost_ledger(env, after_padding, [distance_total], [2])
    _, reset_info = env.reset(options={"n_traj": 2}, seed=8)
    assert_cost_ledger(env, reset_info, [0, 0], [0, 0])
    assert reset_info["routes"] == [[], []]


@pytest.mark.parametrize("env_cls", [EVRPTWVectorEnv, EVRPTWVectorEnvFast])
def test_station_departure_open_failure_keeps_fee_without_duplicate_charge(env_cls):
    env = make_env(env_cls, objective_config=cost_objective(), normalize_reward=False)
    obs, _ = env.reset(seed=11)
    total_distance = 0.0
    last = 0
    for action in [3]:
        assert obs["action_mask"][0, action]
        total_distance += env.distance_km[last, action]
        obs, _, _, _, info = env.step([action])
        assert_cost_ledger(env, info, [total_distance], [1])
        assert info["vehicle_count"][0] == 0
        last = action
    # An adjacent station is invalid: neither movement nor another fee.
    assert not obs["action_mask"][0, 4]
    _, reward, terminated, truncated, info = env.step([4])
    assert truncated[0] and not terminated[0]
    np.testing.assert_array_equal(reward, [env.invalid_action_penalty])
    assert_cost_ledger(env, info, [total_distance], [1])
    assert info["routes"] == [[[0, 3, 0]]]
    assert env.current_routes == [[0, 3]]  # Export does not mutate the ledger.
    _, reward, _, _, info = env.step([1])
    np.testing.assert_array_equal(reward, [0])
    assert_cost_ledger(env, info, [total_distance], [1])


@pytest.mark.parametrize("env_cls", [EVRPTWVectorEnv, EVRPTWVectorEnvFast])
def test_invalid_first_action_and_horizon_truncation_do_not_erase_or_add_fees(env_cls):
    env = make_env(env_cls, objective_config=cost_objective(), n_traj=2)
    env.reset(seed=13)
    env.max_steps = 1
    _, reward, terminated, truncated, info = env.step([-1, 1])
    np.testing.assert_array_equal(truncated, [True, True])
    np.testing.assert_array_equal(terminated, [False, False])
    assert reward[0] == env.invalid_action_penalty
    assert_cost_ledger(env, info, [0, 2], [0, 1])
    np.testing.assert_array_equal(info["vehicle_count"], [0, 0])
    assert info["routes"] == [[], [[0, 1, 0]]]


@pytest.mark.parametrize("use_jit_mask", [False, True])
@pytest.mark.parametrize("info_level", ["light", "full"])
def test_slow_fast_cost_ledgers_physics_masks_and_rewards_match(use_jit_mask, info_level):
    slow = make_env(objective_config=cost_objective(), n_traj=2)
    fast = make_env(
        EVRPTWVectorEnvFast, objective_config=cost_objective(), n_traj=2,
        use_jit_mask=use_jit_mask, info_level=info_level,
    )
    slow_obs, _ = slow.reset(seed=17)
    fast_obs, _ = fast.reset(seed=17)
    for actions in [[3, 1], [1, 3], [4, 0], [0, 2], [2, 0], [0, 1]]:
        np.testing.assert_array_equal(slow_obs["action_mask"], fast_obs["action_mask"])
        slow_obs, slow_reward, slow_done, slow_trunc, slow_info = slow.step(actions)
        fast_obs, fast_reward, fast_done, fast_trunc, fast_info = fast.step(actions)
        np.testing.assert_array_equal(slow_reward, fast_reward)
        np.testing.assert_array_equal(slow_done, fast_done)
        np.testing.assert_array_equal(slow_trunc, fast_trunc)
        for field in ["current_time_s", "battery_used_kwh", "load_cm3", "last", "visited"]:
            np.testing.assert_array_equal(getattr(slow, field), getattr(fast, field))
        for key in ["objective_distance_km", "objective_value", "objective_cost_usd",
                    "electricity_cost_usd", "vehicle_cost_usd", "vehicles_started", "vehicle_count"]:
            np.testing.assert_array_equal(slow_info[key], fast_info[key])
        if info_level == "full":
            assert slow_info["routes"] == fast_info["routes"]
        else:
            assert "routes" not in fast_info


@pytest.mark.parametrize("env_cls", [EVRPTWVectorEnv, EVRPTWVectorEnvFast])
def test_distance_default_retains_legacy_rewards_physics_and_route_exports(env_cls):
    legacy = make_env(env_cls)
    cost = make_env(env_cls, objective_config=cost_objective())
    legacy_obs, legacy_info = legacy.reset(seed=19)
    cost_obs, _ = cost.reset(seed=19)
    for action in [3, 1, 4, 0, 2, 0]:
        np.testing.assert_array_equal(legacy_obs["action_mask"], cost_obs["action_mask"])
        previous = int(legacy.last[0])
        legacy_obs, reward, _, _, legacy_info = legacy.step([action])
        cost_obs, _, _, _, _ = cost.step([action])
        expected = -legacy.distance_km[previous, action] / legacy.reward_distance_scale_km
        np.testing.assert_array_equal(reward, np.asarray([expected], dtype=np.float32))
        np.testing.assert_array_equal(legacy.objective_distance_km, cost.objective_distance_km)
        for field in ["current_time_s", "battery_used_kwh", "load_cm3", "last", "visited"]:
            np.testing.assert_array_equal(getattr(legacy, field), getattr(cost, field))
        if action == 3:
            assert legacy_info["routes"] == [[]]  # Historical charger-only omission.
        assert legacy_info["objective_cost_usd"] is None
        assert legacy_info["electricity_cost_usd"] is None
        assert legacy_info["vehicle_cost_usd"] is None
        np.testing.assert_array_equal(legacy_info["objective_value"], legacy.objective_distance_km)
    assert legacy_info["routes"] == [[[0, 3, 1, 4, 0], [0, 2, 0]]]


@pytest.mark.parametrize("scale_mode", [
    "max_edge", "single_customer_repair_sum", "single_customer_repair_mean", "single_customer_repair_median",
])
def test_cost_scale_uses_existing_distance_normalizer_mode_and_refreshes_on_instance_change(scale_mode):
    objective = cost_objective()
    env = make_env(objective_config=objective, reward_distance_scale_mode=scale_mode)
    expected = objective.reward_scale(env.reward_distance_scale_km, 2, scale_mode)
    assert env.reward_objective_scale == pytest.approx(expected)
    original = economic_instance()
    changed = replace(original, distance_matrix_km=original.distance_matrix_km * 2)
    _, info = env.reset(options={"instance": changed})
    assert env.reward_objective_scale == pytest.approx(
        objective.reward_scale(env.reward_distance_scale_km, 2, scale_mode)
    )
    assert info["reward_objective_scale"] == env.reward_objective_scale


def test_depot_noop_does_not_charge_or_create_a_route():
    env = make_env(objective_config=cost_objective())
    env.reset(seed=23)
    # A depot noop is valid only after all customers have been served. This
    # explicit state represents its pre-termination transition without moving.
    env.visited[:, env.customer_nodes] = True
    env.served_customers[:] = env.num_customers
    assert env._compute_action_mask()[0, 0]
    _, reward, terminated, _, info = env.step([0])
    np.testing.assert_array_equal(reward, [0])
    assert terminated[0]
    assert_cost_ledger(env, info, [0], [0])
    assert info["routes"] == [[]]


@pytest.mark.parametrize("env_cls", [EVRPTWVectorEnv, EVRPTWVectorEnvFast])
def test_closed_charger_only_trip_is_exported_only_in_cost_track(env_cls):
    original = economic_instance()
    indices = [0, 3, 4]
    # With no customers, both depot and station actions are valid before the
    # first termination. This exercises the existing legal mask unchanged.
    instance = replace(
        original,
        customers=np.empty((0, 2), dtype=np.float32),
        demands_cm3=np.empty(0, dtype=np.float32),
        package_counts=np.empty(0, dtype=np.int32),
        service_time_s=np.empty(0, dtype=np.float32),
        tw_s=np.empty((0, 2), dtype=np.float32),
        distance_matrix_km=original.distance_matrix_km[np.ix_(indices, indices)],
        shortest_time_matrix_s=original.shortest_time_matrix_s[np.ix_(indices, indices)],
        energy_matrix_kwh=original.energy_matrix_kwh[np.ix_(indices, indices)],
    )
    kwargs = {"use_jit_mask": False} if issubclass(env_cls, EVRPTWVectorEnvFast) else {}
    cost = env_cls(instance, objective_config=cost_objective(), **kwargs)
    legacy = env_cls(instance, **kwargs)
    cost_obs, _ = cost.reset(seed=29)
    legacy_obs, _ = legacy.reset(seed=29)
    for action in [1, 0]:
        assert cost_obs["action_mask"][0, action]
        np.testing.assert_array_equal(cost_obs["action_mask"], legacy_obs["action_mask"])
        cost_obs, _, _, _, cost_info = cost.step([action])
        legacy_obs, _, _, _, legacy_info = legacy.step([action])
    assert_cost_ledger(cost, cost_info, [9], [1])
    assert cost_info["routes"] == [[[0, 1, 0]]]
    assert legacy_info["routes"] == [[]]
    np.testing.assert_array_equal(cost_info["vehicle_count"], [0])
    np.testing.assert_array_equal(legacy_info["vehicle_count"], [0])


@pytest.mark.parametrize("env_cls", [EVRPTWVectorEnv, EVRPTWVectorEnvFast])
def test_cost_conversion_leaves_success_bonus_and_invalid_penalty_unscaled(env_cls):
    env = make_env(
        env_cls, objective_config=cost_objective(), reward_mode="distance_success",
        success_bonus=7.5, invalid_action_penalty=-4.25,
    )
    env.reset(seed=31)
    rewards = []
    for action in [1, 2, 0]:
        _, reward, _, _, info = env.step([action])
        rewards.append(float(reward[0]))
    assert sum(rewards) == pytest.approx(
        -float(info["objective_value"][0]) / env.reward_objective_scale + 7.5,
        abs=1e-6,
    )
    env.reset(seed=31)
    _, reward, _, _, info = env.step([-1])
    np.testing.assert_array_equal(reward, [-4.25])
    assert_cost_ledger(env, info, [0], [0])
