from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from EVRPTW_Benchmark.Exact.Gurobi_Solver.route_validator import validate_routes
from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.rollout import (
    stack_observations,
)
from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import (
    _instance,
)
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import EVRPTWVectorEnv
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.model import (
    EVRPTWRLPolicy,
)
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.rollout import (
    _normalized_travel_time,
    rollout,
)


def _policy() -> EVRPTWRLPolicy:
    return EVRPTWRLPolicy(embedding_dim=32, structure2vec_rounds=2)


def test_logits_respect_mask_and_recurrent_shape() -> None:
    envs = [EVRPTWVectorEnv(_instance(), n_traj=2) for _ in range(2)]
    observations = [env.reset(seed=index)[0] for index, env in enumerate(envs)]
    batch = stack_observations(observations)
    travel = np.stack(
        [env.travel_time_s / np.max(env.travel_time_s) for env in envs], axis=0
    )
    policy = _policy()
    logits, state = policy.logits(batch, travel, policy.initial_state(2, 2))
    assert logits.shape == (2, 2, 4)
    assert state.hidden.shape == (2, 2, 32)
    mask = torch.as_tensor(batch["action_mask"], dtype=torch.bool)
    assert torch.isneginf(logits[~mask]).all()
    assert torch.isfinite(logits[mask]).all()


def test_dynamic_remaining_demand_changes_after_service() -> None:
    env = EVRPTWVectorEnv(_instance(), n_traj=1)
    observation, _ = env.reset(seed=3)
    assert observation["remaining_demand"][0, 1] > 0.0
    observation, _, _, _, _ = env.step(np.asarray([1], dtype=np.int64))
    assert observation["remaining_demand"][0, 1] == 0.0


def test_rollout_reinforce_update_and_shared_verifier() -> None:
    policy = _policy()
    result = rollout(
        policy,
        [EVRPTWVectorEnv(_instance(), n_traj=4)],
        decode_type="sampling",
        max_steps=32,
        seed=29,
    )
    loss = (result.training_cost.detach() * result.log_likelihood).mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.parameters())

    greedy = rollout(
        policy,
        [EVRPTWVectorEnv(_instance(), n_traj=1)],
        decode_type="greedy",
        max_steps=32,
        seed=31,
    )
    routes = greedy.infos[0]["routes"][0]
    assert validate_routes(_instance(), routes)["passed"]


def test_evrptw_rl_rollout_reports_training_budget_exhaustion() -> None:
    result = rollout(
        _policy(),
        [EVRPTWVectorEnv(_instance(), n_traj=2)],
        decode_type="sampling",
        max_steps=1,
        seed=37,
    )
    assert result.trajectory_steps.tolist() == [[1, 1]]
    assert result.rollout_budget_exhausted.tolist() == [[True, True]]


def test_evrptw_rl_normalization_and_reward_match_adapter_contract() -> None:
    penalty = 17.0
    station_penalty = 0.3
    env = EVRPTWVectorEnv(_instance(), n_traj=2)
    normalized_time = _normalized_travel_time([env])
    assert normalized_time.min() == 0.0
    assert normalized_time.max() == 1.0

    result = rollout(
        _policy(),
        [env],
        decode_type="sampling",
        max_steps=1,
        seed=47,
        station_visit_penalty=station_penalty,
        incomplete_penalty=penalty,
    )
    objective = result.objective_distance_km.detach().cpu().numpy()[0]
    served = result.served_customers.detach().cpu().numpy()[0]
    station_visits = result.station_visits.detach().cpu().numpy()[0]
    feasible = result.feasible.detach().cpu().numpy()[0]
    incomplete_fraction = 1.0 - served / env.num_customers
    expected = (
        objective / env.reward_distance_scale_km
        + station_penalty * station_visits
        + (~feasible) * penalty * (1.0 + incomplete_fraction)
    )
    np.testing.assert_allclose(
        result.training_cost.detach().cpu().numpy()[0], expected
    )
