from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import _instance
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.soft_env import DRLTSSoftConstraintEnv
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.env import DRLTSHardConstraintEnv
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.model import DRLTSPolicy
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.rollout import rollout
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import EVRPTWVectorEnvFast


class LegacySoftEnv(DRLTSSoftConstraintEnv):
    """Frozen loop implementation used only as a differential-test oracle."""

    def _compute_action_mask(self):
        mask = np.zeros((self.n_traj, self.num_nodes), dtype=bool)
        for trajectory in range(self.n_traj):
            if self.terminated[trajectory] or self.truncated[trajectory]:
                mask[trajectory, 0] = True
                continue
            start = int(self.last[trajectory])
            if self.served_customers[trajectory] == self.num_customers:
                mask[trajectory, 0] = True
                continue
            if start != 0:
                mask[trajectory, 0] = True
            for customer in self.customer_nodes:
                node = int(customer)
                if not self.visited[trajectory, node]:
                    mask[trajectory, node] = True
            if self._is_customer(start):
                mask[trajectory, self.station_nodes] = ~self.cs_visited_current_route[
                    trajectory, self.station_nodes
                ]
        return mask

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.int64).reshape(self.n_traj)
        mask = self._compute_action_mask()
        for trajectory, destination in enumerate(action_arr):
            if self.terminated[trajectory] or self.truncated[trajectory]:
                continue
            if destination < 0 or destination >= self.num_nodes:
                continue
            if not mask[trajectory, destination]:
                continue
            capacity, time_window, energy = self._normalized_violations(
                trajectory, int(destination)
            )
            self.capacity_violation[trajectory] += capacity
            self.time_violation[trajectory] += time_window
            self.energy_violation[trajectory] += energy
        observation, reward, terminated, truncated, info = EVRPTWVectorEnvFast.step(
            self, action_arr
        )
        return observation, reward, terminated, truncated, self._with_violation_info(info)


def _assert_equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_equal(actual[key], value)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected


@pytest.mark.parametrize("n_traj", [1, 7, 100])
def test_vectorized_soft_mask_matches_loop_on_mixed_states(n_traj):
    env = DRLTSSoftConstraintEnv(_instance(), n_traj=n_traj)
    env.reset(seed=5)
    rng = np.random.default_rng(9107)
    for _ in range(30):
        env.terminated[:] = rng.random(n_traj) < 0.2
        env.truncated[:] = rng.random(n_traj) < 0.2
        env.last[:] = rng.integers(0, env.num_nodes, size=n_traj)
        env.served_customers[:] = rng.integers(0, env.num_customers + 1, size=n_traj)
        env.visited[:] = rng.random(env.visited.shape) < 0.5
        env.cs_visited_current_route[:] = rng.random(env.cs_visited_current_route.shape) < 0.5
        np.testing.assert_array_equal(
            env._compute_action_mask(), LegacySoftEnv._compute_action_mask(env)
        )
    assert not env.use_jit_mask  # Soft rules never enter the hard-feasibility kernel.


@pytest.mark.parametrize("seed", [3, 19, 77])
def test_cached_soft_mask_preserves_transitions_penalties_and_routes(seed):
    instance = replace(
        _instance(),
        vehicle={"battery_capacity_kwh": 0.25, "cargo_capacity_cm3": 0.5},
        tw_s=np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
    )
    optimized = DRLTSSoftConstraintEnv(instance, n_traj=7, info_level="full")
    legacy = LegacySoftEnv(instance, n_traj=7, info_level="full")
    actual, actual_info = optimized.reset(seed=seed)
    expected, expected_info = legacy.reset(seed=seed)
    _assert_equal(actual, expected)
    _assert_equal(actual_info, expected_info)
    rng = np.random.default_rng(seed)
    for step in range(20):
        actions = np.asarray(
            [rng.choice(np.flatnonzero(row)) for row in expected["action_mask"]]
        )
        if step == 0:
            actions[:] = 1  # Accumulate all three soft-resource violations.
        if step == 1:
            actions[0] = -1
            actions[1] = optimized.num_nodes
            actions[2] = 1  # Already served: a masked action is not penalized twice.
        actual_step = optimized.step(actions)
        expected_step = legacy.step(actions)
        for actual_part, expected_part in zip(actual_step, expected_step):
            _assert_equal(actual_part, expected_part)
        actual, expected = actual_step[0], expected_step[0]
        for field in [
            "last", "visited", "cs_visited_current_route", "load_cm3",
            "current_time_s", "battery_used_kwh", "served_customers",
            "capacity_violation", "time_violation", "energy_violation",
        ]:
            np.testing.assert_array_equal(getattr(optimized, field), getattr(legacy, field))
        assert optimized.get_routes() == legacy.get_routes()
        if np.all(expected_step[2] | expected_step[3]):
            break
    assert np.all(optimized.capacity_violation > 0)
    assert np.all(optimized.time_violation > 0)
    assert np.all(optimized.energy_violation > 0)


def test_soft_step_reuses_pre_action_mask_and_handles_missing_cache(monkeypatch):
    env = DRLTSSoftConstraintEnv(_instance(), n_traj=2, info_level="light")
    env.reset(seed=1)
    compute_mask = env._compute_action_mask
    calls = []

    def tracked_mask():
        calls.append(1)
        return compute_mask()

    monkeypatch.setattr(env, "_compute_action_mask", tracked_mask)
    env.step(np.asarray([1, 1]))
    assert len(calls) == 1  # Post-action mask only; old implementation did two.
    env._current_action_mask = None
    calls.clear()
    env.step(np.asarray([2, 2]))
    assert len(calls) == 2  # One pre-action fallback, one post-action observation.


@pytest.mark.parametrize("decode_type", ["greedy", "sampling"])
@pytest.mark.parametrize("soft", [False, True])
def test_cost_only_drl_ts_rollout_preserves_candidates_costs_and_rng(decode_type, soft):
    policy = DRLTSPolicy(
        embedding_dim=16, n_encode_layers=1, n_heads=4, nearest_neighbors=2
    ).eval()
    env_cls = DRLTSSoftConstraintEnv if soft else DRLTSHardConstraintEnv
    results = []
    random_states = []
    for compute_log_likelihood in [True, False]:
        torch.manual_seed(19)
        with torch.no_grad():
            results.append(
                rollout(
                    policy,
                    [env_cls(_instance(), n_traj=100, info_level="full")],
                    decode_type=decode_type,
                    max_steps=32,
                    seed=19,
                    soft_constraints=soft,
                    compute_log_likelihood=compute_log_likelihood,
                )
            )
        random_states.append(torch.get_rng_state().clone())
    full, cost_only = results
    for field in [
        "training_cost", "objective_distance_km", "feasible", "served_customers",
        "trajectory_steps", "rollout_budget_exhausted",
    ]:
        assert torch.equal(getattr(full, field), getattr(cost_only, field))
    _assert_equal(cost_only.infos[0], full.infos[0])
    assert torch.equal(random_states[0], random_states[1])
    assert full.environment_transitions == cost_only.environment_transitions
    assert torch.count_nonzero(cost_only.log_likelihood).item() == 0
