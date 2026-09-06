from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import (
    EVRPTWVectorEnv,
    EVRPTWVectorEnvFast,
)
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import env_fast as fast_module
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env.mask_jit import NUMBA_AVAILABLE
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env.tests.test_objective_cost import (
    cost_objective,
    economic_instance,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.action_constraints import (
    ACTION_CONSTRAINT_CONTRACT_ID,
    consecutive_cs_arcs,
)


VARIANTS = ["slow", "fast_numpy", "fast_jit"]


def make_env(variant, instance=None, **kwargs):
    instance = bridge_instance() if instance is None else instance
    if variant == "slow":
        return EVRPTWVectorEnv(instance, **kwargs)
    if variant == "fast_jit" and not NUMBA_AVAILABLE:
        pytest.skip("Numba is unavailable")
    return EVRPTWVectorEnvFast(instance, use_jit_mask=variant == "fast_jit", **kwargs)


def bridge_instance():
    """Station 3 cannot return directly but can serve 2 then charge at 4."""
    instance = economic_instance()
    energy = np.full((5, 5), 20.0, dtype=np.float32)
    np.fill_diagonal(energy, 0.0)
    for start, destination, value in [
        (0, 1, 1), (1, 0, 1), (0, 3, 2), (0, 4, 1), (4, 1, 1),
        (1, 3, 3), (1, 4, 1), (3, 0, 9), (3, 2, 2),
        (2, 0, 4), (2, 4, 2), (4, 0, 2), (3, 4, 1), (4, 3, 1),
    ]:
        energy[start, destination] = value
    travel = np.full((5, 5), 10.0, dtype=np.float32)
    np.fill_diagonal(travel, 0.0)
    return replace(
        instance, instance_id="no_consecutive_cs_bridge",
        energy_matrix_kwh=energy, shortest_time_matrix_s=travel,
        vehicle={**instance.vehicle, "battery_capacity_kwh": 5.0},
    )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("cost", [False, True])
def test_charging_customer_charging_remains_legal_but_adjacent_stations_do_not(variant, cost):
    env = make_env(variant, objective_config=cost_objective() if cost else None)
    obs, info = env.reset(seed=17)
    assert info["action_constraint_contract_id"] == ACTION_CONSTRAINT_CONTRACT_ID
    assert obs["action_mask"][0, 3]  # No new depot-to-station ban.
    assert not env._can_return_to_depot(3, 0.0, 0.0)
    assert env._station_customer_continuation_feasible(0, 3, 0.0)
    total_reward = 0.0
    for action in [1, 3, 2, 4, 0]:
        assert obs["action_mask"][0, action]
        obs, reward, terminated, truncated, info = env.step([action])
        total_reward += float(reward[0])
        assert not truncated[0]
        if action in [3, 4]:
            assert not obs["action_mask"][0, env.station_start:].any()
            assert env.battery_used_kwh[0] == 0.0
    assert terminated[0]
    assert info["routes"] == [[[0, 1, 3, 2, 4, 0]]]
    assert not consecutive_cs_arcs(env.instance, info["routes"][0])
    assert info["vehicles_started"][0] == 1
    assert total_reward == pytest.approx(-info["objective_value"][0] / env.reward_objective_scale, rel=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
def test_forced_cs_to_cs_is_invalid_without_movement_charge_or_extra_vehicle(variant):
    env = make_env(variant, objective_config=cost_objective())
    env.reset(seed=19)
    _, _, _, _, before = env.step([3])
    time_before = env.current_time_s.copy()
    battery_before = env.battery_used_kwh.copy()
    _, reward, terminated, truncated, after = env.step([4])
    assert truncated[0] and not terminated[0]
    assert reward[0] == env.invalid_action_penalty
    assert after["invalid_action"][0]
    for key in ["objective_distance_km", "objective_value", "vehicles_started"]:
        np.testing.assert_array_equal(before[key], after[key])
    np.testing.assert_array_equal(env.current_time_s, time_before)
    np.testing.assert_array_equal(env.battery_used_kwh, battery_before)
    assert env.current_routes == [[0, 3]]
    # Truncated padding cannot accidentally execute the rejected transition.
    _, reward, _, _, after_padding = env.step([4])
    np.testing.assert_array_equal(reward, [0])
    np.testing.assert_array_equal(before["objective_value"], after_padding["objective_value"])


@pytest.mark.parametrize("variant", VARIANTS)
def test_pure_multistation_return_is_not_used_as_customer_feasibility_proof(variant):
    instance = bridge_instance()
    energy = instance.energy_matrix_kwh.copy()
    energy[1, 0] = 9
    energy[1, 4] = 20
    instance = replace(instance, energy_matrix_kwh=energy)
    env = make_env(variant, instance)
    obs, _ = env.reset(seed=23)
    # Old stop-only Dijkstra found customer 1 -> CS3 -> CS4 -> depot.
    assert not env._can_return_to_depot(1, 0.0, 1.0)
    assert not obs["action_mask"][0, 1]
    assert env._shortest_stop_time(3, 0) is None
    for source, edges in env.stop_adj.items():
        assert all(not (env._is_station(source) and env._is_station(target)) for target, _ in edges)
    if isinstance(env, EVRPTWVectorEnvFast):
        assert np.isinf(env._stop_to_depot_time_s[3])
        assert env._stop_to_depot_time_s[4] == instance.shortest_time_matrix_s[4, 0]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("blocked_by", ["load", "energy", "time_window", "service_horizon", "return_horizon"])
def test_customer_bridge_obeys_dynamic_resource_constraints(variant, blocked_by):
    instance = bridge_instance()
    if blocked_by == "load":
        instance = replace(instance, vehicle={**instance.vehicle, "cargo_capacity_cm3": 1.0})
    elif blocked_by == "energy":
        energy = instance.energy_matrix_kwh.copy()
        energy[3, 2] = 6.0
        instance = replace(instance, energy_matrix_kwh=energy)
    elif blocked_by == "time_window":
        tw = instance.tw_s.copy()
        tw[1, 1] = 1.0
        instance = replace(instance, tw_s=tw)
    elif blocked_by == "service_horizon":
        service = instance.service_time_s.copy()
        service[1] = instance.working_end_s
        instance = replace(instance, service_time_s=service)
    else:
        instance = replace(instance, working_end_s=1000)
    env = make_env(variant, instance)
    env.reset(seed=29)
    obs, _, _, _, _ = env.step([1])
    assert not obs["action_mask"][0, 3]
    assert not env._station_action_feasible(0, 3)


@pytest.mark.parametrize("variant", VARIANTS)
def test_customer_bridge_cannot_use_previously_visited_station_or_customer(variant):
    env = make_env(variant)
    obs, _ = env.reset(seed=31)
    for action in [4, 1]:
        assert obs["action_mask"][0, action]
        obs, _, _, _, _ = env.step([action])
    assert env.cs_visited_current_route[0, 4]
    assert not obs["action_mask"][0, 3]  # Its only bridge needs CS4 again.
    assert not env._station_action_feasible(0, 3)
    assert not obs["action_mask"][0, 4]

    env.reset(seed=31)
    # Isolate customer availability from charging/arrival calculations.
    env.visited[0, 2] = True
    assert not env._station_customer_continuation_feasible(0, 3, 0.0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_customer_return_respects_available_stations_and_excluded_candidate(variant):
    env = make_env(variant)
    env.reset(seed=37)
    unavailable = np.zeros(env.num_nodes, dtype=bool)
    assert env._can_return_to_depot(2, 0.0, 2.0, unavailable_stations=unavailable)
    assert not env._can_return_to_depot(2, 0.0, 2.0, excluded_station=4)
    unavailable[4] = True
    assert not env._can_return_to_depot(2, 0.0, 2.0, unavailable_stations=unavailable)
    env.cs_visited_current_route[0, 4] = True
    # The normal customer mask, not only the station bridge, uses availability.
    env.last[0] = 3
    obs = env._make_observation()
    assert not obs["action_mask"][0, 2]


@pytest.mark.parametrize("variant", ["fast_numpy", "fast_jit"])
def test_direct_return_cache_refreshes_when_instance_changes(variant):
    env = make_env(variant)
    env.reset(seed=41)
    assert np.isinf(env._stop_to_depot_time_s[3])
    energy = env.instance.energy_matrix_kwh.copy()
    energy[3, 0] = 2.0
    replacement = replace(env.instance, energy_matrix_kwh=energy)
    _, info = env.reset(options={"instance": replacement})
    assert env._stop_to_depot_time_s[3] == replacement.shortest_time_matrix_s[3, 0]
    assert env._can_return_to_depot(3, 0.0, 0.0)
    assert info["action_constraint_contract_id"] == ACTION_CONSTRAINT_CONTRACT_ID


@pytest.mark.parametrize("info_level", ["light", "full"])
def test_action_constraint_contract_is_exported_in_all_info_levels(info_level):
    env = make_env("fast_numpy", info_level=info_level)
    _, info = env.reset(seed=43)
    assert info["action_constraint_contract_id"] == ACTION_CONSTRAINT_CONTRACT_ID
    _, _, _, _, info = env.step([1])
    assert info["action_constraint_contract_id"] == ACTION_CONSTRAINT_CONTRACT_ID


def test_jit_keeps_main_kernel_and_only_patches_missing_customer_bridge(monkeypatch):
    if not NUMBA_AVAILABLE:
        pytest.skip("Numba is unavailable")
    original_kernel = fast_module.compute_action_mask_jit
    kernel_station_flags = []

    def tracked_kernel(**kwargs):
        result = original_kernel(**kwargs)
        kernel_station_flags.append(bool(result[0, 3]))
        return result

    def reject_full_python_mask(_self):
        raise AssertionError("the full JIT mask must not fall back to Python")

    monkeypatch.setattr(fast_module, "compute_action_mask_jit", tracked_kernel)
    monkeypatch.setattr(EVRPTWVectorEnv, "_compute_action_mask", reject_full_python_mask)
    env = make_env("fast_jit")
    obs, _ = env.reset(seed=47)
    assert kernel_station_flags == [False]
    assert obs["action_mask"][0, 3]  # Selective dynamic bridge patch.

    # Ordinary stations with direct returns need no bridge exploration.
    direct_env = make_env("fast_jit", economic_instance())
    def reject_bridge(*_args):
        raise AssertionError("direct-return stations do not require lookahead")
    monkeypatch.setattr(direct_env, "_station_customer_continuation_feasible", reject_bridge)
    obs, _ = direct_env.reset(seed=47)
    assert obs["action_mask"][0, 3:].all()


def test_slow_numpy_jit_states_masks_and_rewards_match_through_customer_bridge():
    if not NUMBA_AVAILABLE:
        pytest.skip("Numba is unavailable")
    envs = [make_env(variant, objective_config=cost_objective(), n_traj=2) for variant in VARIANTS]
    observations = [env.reset(seed=53)[0] for env in envs]
    for actions in [[1, 3], [3, 2], [2, 4], [4, 0], [0, 1], [0, 0]]:
        for obs in observations[1:]:
            np.testing.assert_array_equal(observations[0]["action_mask"], obs["action_mask"])
        transitions = [env.step(actions) for env in envs]
        observations = [transition[0] for transition in transitions]
        reference = transitions[0]
        for env, transition in zip(envs[1:], transitions[1:]):
            for index in [1, 2, 3]:
                np.testing.assert_array_equal(reference[index], transition[index])
            for field in ["last", "current_time_s", "battery_used_kwh", "load_cm3", "visited", "cs_visited_current_route"]:
                np.testing.assert_array_equal(getattr(envs[0], field), getattr(env, field))
            for key in ["objective_distance_km", "objective_value", "vehicles_started"]:
                np.testing.assert_array_equal(reference[4][key], transition[4][key])
            assert reference[4]["routes"] == transition[4]["routes"]
