from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from EVRPTW_Benchmark.Exact.Gurobi_Solver.route_validator import validate_routes
from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.rollout import (
    stack_observations,
)
from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import (
    _dynamic_batches,
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
    np.testing.assert_allclose(
        normalized_time[0], env.travel_time_s / env.horizon_s
    )

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


@pytest.mark.parametrize("training", [False, True])
def test_static_cache_preserves_dynamic_logits_recurrence_and_gradients(
    training: bool,
) -> None:
    torch.manual_seed(131)
    policy = _policy().train(training)
    reference = deepcopy(policy)
    policy.load_state_dict(reference.state_dict(), strict=True)
    original_keys = tuple(policy.state_dict())
    batches = _dynamic_batches()
    travel = _normalized_travel_time([
        EVRPTWVectorEnv(_instance(), n_traj=2) for _ in range(2)
    ])
    with patch.object(
        policy, "_edge_message", wraps=policy._edge_message
    ) as cached_edges, patch.object(
        reference, "_edge_message", wraps=reference._edge_message
    ) as reference_edges, patch.object(
        policy, "_static_local_features", wraps=policy._static_local_features
    ) as cached_features, patch.object(
        reference, "_static_local_features", wraps=reference._static_local_features
    ) as reference_features:
        fixed = policy.encode_static(batches[0], travel)
        state = policy.initial_state(2, 2)
        reference_state = reference.initial_state(2, 2)
        losses, reference_losses = [], []
        for batch in batches:
            actual, state = policy.logits(batch, travel, state, fixed=fixed)
            expected, reference_state = reference.logits(batch, travel, reference_state)
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(state.hidden, reference_state.hidden)
            torch.testing.assert_close(state.cell, reference_state.cell)
            feasible = torch.as_tensor(batch["action_mask"])
            losses.append(actual[feasible].square().mean())
            reference_losses.append(expected[feasible].square().mean())
        assert cached_edges.call_count == cached_features.call_count == 1
        assert reference_edges.call_count == reference_features.call_count == len(batches)
    sum(losses).backward()
    sum(reference_losses).backward()
    for actual, expected in zip(policy.parameters(), reference.parameters()):
        assert (actual.grad is None) == (expected.grad is None)
        if actual.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-5, atol=3e-5)
    assert policy.edge_direction.grad is not None
    assert policy.edge_projection.weight.grad is not None
    assert tuple(policy.state_dict()) == original_keys


@pytest.mark.parametrize("decode_type", ["greedy", "sampling"])
@pytest.mark.parametrize("training", [False, True])
def test_static_cached_rollout_preserves_cost_routes_and_rng(
    decode_type: str, training: bool
) -> None:
    torch.manual_seed(137)
    policy = _policy().train(training)
    reference = deepcopy(policy)
    results, rng_states = [], []
    for active, cached in ((policy, True), (reference, False)):
        torch.manual_seed(139)
        results.append(rollout(
            active,
            [EVRPTWVectorEnv(_instance(), n_traj=3)],
            decode_type=decode_type,
            max_steps=32,
            seed=149,
            use_static_cache=cached,
        ))
        rng_states.append(torch.random.get_rng_state())
    actual, expected = results
    torch.testing.assert_close(actual.training_cost, expected.training_cost, rtol=0, atol=0)
    torch.testing.assert_close(actual.log_likelihood, expected.log_likelihood)
    torch.testing.assert_close(actual.trajectory_steps, expected.trajectory_steps)
    assert actual.infos[0]["routes"] == expected.infos[0]["routes"]
    assert actual.environment_transitions == expected.environment_transitions
    assert torch.equal(*rng_states)


@pytest.mark.parametrize("decode_type", ["greedy", "sampling"])
def test_cost_only_rollout_preserves_sampling_and_skips_unused_work(
    decode_type: str,
) -> None:
    torch.manual_seed(167)
    policy = _policy().eval()
    results, rng_states, distribution_counts = [], [], []
    for compute in (True, False):
        torch.manual_seed(173)
        with torch.no_grad(), patch.object(
            torch.distributions, "Categorical", wraps=torch.distributions.Categorical
        ) as distribution:
            results.append(rollout(
                policy,
                [EVRPTWVectorEnv(_instance(), n_traj=3)],
                decode_type=decode_type,
                max_steps=32,
                seed=179,
                compute_log_likelihood=compute,
            ))
            distribution_counts.append(distribution.call_count)
        rng_states.append(torch.random.get_rng_state())
    reference, cost_only = results
    torch.testing.assert_close(
        reference.training_cost, cost_only.training_cost, rtol=0, atol=0
    )
    assert reference.infos[0]["routes"] == cost_only.infos[0]["routes"]
    assert torch.equal(*rng_states)
    assert torch.count_nonzero(cost_only.log_likelihood) == 0
    if decode_type == "greedy":
        assert distribution_counts[0] > 0 and distribution_counts[1] == 0
    else:
        assert distribution_counts[0] == distribution_counts[1] > 0
