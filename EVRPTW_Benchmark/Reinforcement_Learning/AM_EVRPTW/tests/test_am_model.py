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

from evrptw_core.schema import EVRPTWInstance

from EVRPTW_Benchmark.Exact.Gurobi_Solver.route_validator import validate_routes
from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.model import (
    AMEVRPTWPolicy,
)
from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.rollout import (
    rollout,
    stack_observations,
)
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import EVRPTWVectorEnv


def _instance() -> EVRPTWInstance:
    distance = np.asarray(
        [[0, 2, 3, 1], [2, 0, 2, 1], [3, 2, 0, 1], [1, 1, 1, 0]],
        dtype=np.float32,
    )
    travel = distance * 60.0
    energy = distance * 0.5
    return EVRPTWInstance(
        instance_id="am_smoke",
        region_id="test",
        mother_board_id="test",
        operating_day_id="test",
        day_type="weekday",
        working_start_s=0,
        working_end_s=20_000,
        depot=np.asarray([0.0, 0.0], dtype=np.float32),
        customers=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        charging_stations=np.asarray([[0.5, 0.5]], dtype=np.float32),
        distance_matrix_km=distance,
        demands_cm3=np.asarray([1.0, 1.0], dtype=np.float32),
        package_counts=np.asarray([1, 1], dtype=np.int32),
        service_time_s=np.asarray([30.0, 30.0], dtype=np.float32),
        tw_s=np.asarray([[0, 20_000], [0, 20_000]], dtype=np.float32),
        cs_time_to_depot_s=np.asarray([60.0], dtype=np.float32),
        vehicle={"battery_capacity_kwh": 10.0, "cargo_capacity_cm3": 10.0},
        shortest_time_matrix_s=travel,
        energy_matrix_kwh=energy,
        raw={
            "charging_power_kw": np.asarray([20.0], dtype=np.float32),
            "charging_policy": {"charging_power_derating_factor": 0.9},
        },
    )


def test_am_logits_respect_external_action_mask() -> None:
    envs = [EVRPTWVectorEnv(_instance(), n_traj=2) for _ in range(2)]
    observations = [env.reset(seed=index)[0] for index, env in enumerate(envs)]
    batch = stack_observations(observations)
    policy = AMEVRPTWPolicy(
        embedding_dim=32,
        hidden_dim=32,
        n_encode_layers=1,
        n_heads=4,
    )
    fixed = policy.encode(batch)
    logits = policy.logits(batch, fixed)
    assert logits.shape == (2, 2, 4)
    infeasible = ~torch.as_tensor(batch["action_mask"], dtype=torch.bool)
    assert torch.isneginf(logits[infeasible]).all()
    assert torch.isfinite(logits[~infeasible]).all()


def test_am_sampling_log_probability_has_gradient() -> None:
    env = EVRPTWVectorEnv(_instance(), n_traj=1)
    observation, _ = env.reset(seed=11)
    batch = stack_observations([observation])
    policy = AMEVRPTWPolicy(
        embedding_dim=32,
        hidden_dim=32,
        n_encode_layers=1,
        n_heads=4,
    )
    logits = policy.logits(batch, policy.encode(batch))
    distribution = torch.distributions.Categorical(logits=logits)
    loss = -distribution.log_prob(distribution.sample()).mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.parameters())


def test_am_greedy_rollout_exports_a_finite_distance() -> None:
    env = EVRPTWVectorEnv(_instance(), n_traj=1)
    policy = AMEVRPTWPolicy(
        embedding_dim=32,
        hidden_dim=32,
        n_encode_layers=1,
        n_heads=4,
    )
    result = rollout(
        policy,
        [env],
        decode_type="greedy",
        max_steps=env.max_steps,
        seed=17,
        incomplete_penalty_km=10_000.0,
    )
    assert result.cost_km.shape == (1, 1)
    assert torch.isfinite(result.cost_km).all()
    assert result.log_likelihood.shape == (1, 1)
    assert validate_routes(_instance(), result.infos[0]["routes"][0])["passed"]


def test_am_rollout_reports_training_budget_exhaustion() -> None:
    policy = AMEVRPTWPolicy(
        embedding_dim=32,
        hidden_dim=32,
        n_encode_layers=1,
        n_heads=4,
    )
    result = rollout(
        policy,
        [EVRPTWVectorEnv(_instance(), n_traj=2)],
        decode_type="sampling",
        max_steps=1,
        seed=19,
        incomplete_penalty_km=10_000.0,
    )
    assert result.trajectory_steps.tolist() == [[1, 1]]
    assert result.rollout_budget_exhausted.tolist() == [[True, True]]


def test_am_reinforce_update_runs_through_complete_rollout() -> None:
    policy = AMEVRPTWPolicy(
        embedding_dim=32,
        hidden_dim=32,
        n_encode_layers=1,
        n_heads=4,
    )
    baseline = AMEVRPTWPolicy(
        embedding_dim=32,
        hidden_dim=32,
        n_encode_layers=1,
        n_heads=4,
    )
    baseline.load_state_dict(policy.state_dict())
    actor = rollout(
        policy,
        [EVRPTWVectorEnv(_instance(), n_traj=4)],
        decode_type="sampling",
        max_steps=32,
        seed=23,
        incomplete_penalty_km=10_000.0,
    )
    with torch.no_grad():
        reference = rollout(
            baseline,
            [EVRPTWVectorEnv(_instance(), n_traj=4)],
            decode_type="greedy",
            max_steps=32,
            seed=23,
            incomplete_penalty_km=10_000.0,
        )
    loss = ((actor.cost_km - reference.cost_km).detach() * actor.log_likelihood).mean()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    assert torch.isfinite(loss)


def test_am_training_cost_is_distance_plus_named_incomplete_guard() -> None:
    penalty = 123.0
    env = EVRPTWVectorEnv(_instance(), n_traj=2)
    result = rollout(
        AMEVRPTWPolicy(
            embedding_dim=32,
            hidden_dim=32,
            n_encode_layers=1,
            n_heads=4,
        ),
        [env],
        decode_type="sampling",
        max_steps=1,
        seed=43,
        incomplete_penalty_km=penalty,
    )
    objective = np.asarray(result.infos[0]["objective_distance_km"])
    served = result.served_customers.detach().cpu().numpy()[0]
    feasible = result.feasible.detach().cpu().numpy()[0]
    incomplete_fraction = 1.0 - served / env.num_customers
    expected = objective + (~feasible) * penalty * (1.0 + incomplete_fraction)
    np.testing.assert_allclose(result.cost_km.detach().cpu().numpy()[0], expected)


def _dynamic_batches() -> list[dict[str, np.ndarray]]:
    envs = [EVRPTWVectorEnv(_instance(), n_traj=2) for _ in range(2)]
    batches = [stack_observations([env.reset(seed=5)[0] for env in envs])]
    for actions in ([1, 2], [2, 1]):
        observations = [env.step(np.asarray(actions))[0] for env in envs]
        batches.append(stack_observations(observations))
    return batches


@pytest.mark.parametrize("training", [False, True])
def test_am_cached_glimpse_matches_logits_and_gradients(training: bool) -> None:
    torch.manual_seed(103)
    policy = AMEVRPTWPolicy(
        embedding_dim=32, hidden_dim=32, n_encode_layers=1, n_heads=4
    ).train(training)
    reference = deepcopy(policy)
    # Caches add no model parameters/buffers and old-style checkpoints still load.
    policy.load_state_dict(reference.state_dict(), strict=True)
    original_keys = tuple(policy.state_dict())
    batches = _dynamic_batches()
    with patch.object(
        policy, "_project_glimpse_memory", wraps=policy._project_glimpse_memory
    ) as project, patch.object(
        reference.glimpse, "forward", wraps=reference.glimpse.forward
    ) as uncached_attention:
        fixed = policy.encode(batches[0])
        uncached = reference.encode(batches[0], cache_decoder=False)
        losses, reference_losses = [], []
        for batch in batches:
            actual = policy.logits(batch, fixed)
            expected = reference.logits(batch, uncached)
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
            feasible = torch.as_tensor(batch["action_mask"])
            losses.append(actual[feasible].square().mean())
            reference_losses.append(expected[feasible].square().mean())
        assert project.call_count == 1
        assert uncached_attention.call_count == len(batches)
    sum(losses).backward()
    sum(reference_losses).backward()
    for actual, expected in zip(policy.parameters(), reference.parameters()):
        assert (actual.grad is None) == (expected.grad is None)
        if actual.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-5, atol=3e-5)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key])
    assert tuple(policy.state_dict()) == original_keys
    assert policy.glimpse.W_key.grad is not None
    assert policy.glimpse.W_val.grad is not None


@pytest.mark.parametrize("decode_type", ["greedy", "sampling"])
@pytest.mark.parametrize("training", [False, True])
def test_am_cached_rollout_preserves_cost_routes_and_rng(
    decode_type: str, training: bool
) -> None:
    torch.manual_seed(109)
    policy = AMEVRPTWPolicy(
        embedding_dim=32, hidden_dim=32, n_encode_layers=1, n_heads=4
    ).train(training)
    reference = deepcopy(policy)
    results, rng_states = [], []
    for active, cached in ((policy, True), (reference, False)):
        torch.manual_seed(113)
        results.append(rollout(
            active,
            [EVRPTWVectorEnv(_instance(), n_traj=3)],
            decode_type=decode_type,
            max_steps=32,
            seed=127,
            incomplete_penalty_km=10_000.0,
            use_static_cache=cached,
        ))
        rng_states.append(torch.random.get_rng_state())
    actual, expected = results
    torch.testing.assert_close(actual.cost_km, expected.cost_km, rtol=0, atol=0)
    torch.testing.assert_close(actual.log_likelihood, expected.log_likelihood)
    torch.testing.assert_close(actual.trajectory_steps, expected.trajectory_steps)
    assert actual.infos[0]["routes"] == expected.infos[0]["routes"]
    assert actual.environment_transitions == expected.environment_transitions
    assert torch.equal(*rng_states)


@pytest.mark.parametrize("decode_type", ["greedy", "sampling"])
def test_am_cost_only_rollout_preserves_sampling_and_skips_unused_work(
    decode_type: str,
) -> None:
    torch.manual_seed(151)
    policy = AMEVRPTWPolicy(
        embedding_dim=32, hidden_dim=32, n_encode_layers=1, n_heads=4
    ).eval()
    results, rng_states, distribution_counts = [], [], []
    for compute in (True, False):
        torch.manual_seed(157)
        with torch.no_grad(), patch.object(
            torch.distributions, "Categorical", wraps=torch.distributions.Categorical
        ) as distribution:
            results.append(rollout(
                policy,
                [EVRPTWVectorEnv(_instance(), n_traj=3)],
                decode_type=decode_type,
                max_steps=32,
                seed=163,
                incomplete_penalty_km=10_000.0,
                compute_log_likelihood=compute,
            ))
            distribution_counts.append(distribution.call_count)
        rng_states.append(torch.random.get_rng_state())
    reference, cost_only = results
    torch.testing.assert_close(reference.cost_km, cost_only.cost_km, rtol=0, atol=0)
    assert reference.infos[0]["routes"] == cost_only.infos[0]["routes"]
    assert torch.equal(*rng_states)
    assert torch.count_nonzero(cost_only.log_likelihood) == 0
    if decode_type == "greedy":
        assert distribution_counts[0] > 0 and distribution_counts[1] == 0
    else:
        assert distribution_counts[0] == distribution_counts[1] > 0
