from __future__ import annotations

import sys
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

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
    _instance,
)
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.env import (
    DRLTSHardConstraintEnv,
)
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.model import DRLTSPolicy
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.rollout import (
    bounded_soft_violation_component,
    normalized_edge_matrices,
    rollout,
)
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.soft_env import (
    DRLTSSoftConstraintEnv,
)
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.train import (
    _configure_soft_auxiliary,
)
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import EVRPTWVectorEnvFast


def _policy() -> DRLTSPolicy:
    return DRLTSPolicy(
        embedding_dim=32,
        n_encode_layers=1,
        n_heads=4,
        nearest_neighbors=2,
    )


def test_logits_respect_mask_and_recurrent_shape() -> None:
    envs = [EVRPTWVectorEnvFast(_instance(), n_traj=2) for _ in range(2)]
    observations = [env.reset(seed=index)[0] for index, env in enumerate(envs)]
    batch = stack_observations(observations)
    distance, travel_time, energy = normalized_edge_matrices(envs)
    policy = _policy()
    fixed = policy.encode(batch, distance, travel_time, energy)
    logits, state = policy.logits(batch, fixed, policy.initial_state(2, 2))
    mask = torch.as_tensor(batch["action_mask"], dtype=torch.bool)
    assert logits.shape == (2, 2, 4)
    assert state.hidden.shape == (2, 2, 32)
    assert torch.isneginf(logits[~mask]).all()
    assert torch.isfinite(logits[mask]).all()


def test_stage1_soft_mask_allows_and_penalizes_capacity_violation() -> None:
    constrained = replace(
        _instance(),
        vehicle={"battery_capacity_kwh": 10.0, "cargo_capacity_cm3": 0.5},
    )
    hard = EVRPTWVectorEnvFast(constrained, n_traj=1, use_jit_mask=False)
    soft = DRLTSSoftConstraintEnv(constrained, n_traj=1)
    hard_observation, _ = hard.reset(seed=3)
    soft_observation, _ = soft.reset(seed=3)
    assert not hard_observation["action_mask"][0, 1]
    assert soft_observation["action_mask"][0, 1]
    observation, _, _, _, info = soft.step(np.asarray([1], dtype=np.int64))
    assert info["capacity_violation_normalized"][0] > 0.0
    assert soft.observation_space.contains(observation)


def test_soft_violation_fixed_customer_normalization_is_bounded_and_not_dilutable() -> None:
    env = DRLTSSoftConstraintEnv(
        _instance(), n_traj=1, soft_violation_step_clip=1.0
    )
    env.reset(seed=3)
    # One extreme customer transition: raw audit values remain exact while the
    # only training-eligible contribution is clipped to one per component.
    env._record_normalized_violations(0, 1, 5.0, 2.0, 3.0)
    first = env._with_violation_info({})
    assert first["capacity_violation_raw_sum"].tolist() == [5.0]
    assert first["time_violation_raw_sum"].tolist() == [2.0]
    assert first["energy_violation_raw_sum"].tolist() == [3.0]
    assert first["capacity_violation_clipped_sum"].tolist() == [1.0]
    assert first["time_violation_clipped_sum"].tolist() == [1.0]
    assert first["energy_violation_clipped_sum"].tolist() == [1.0]

    fixed_n = np.asarray([float(env.num_customers)])
    before = bounded_soft_violation_component(
        first["time_violation_clipped_sum"], fixed_n, 1.0
    )
    for _ in range(100):
        # Extra zero-violation travel raises the audit count but cannot enter the
        # fixed denominator or reduce the component already incurred.
        env._record_normalized_violations(0, 0, 0.0, 0.0, 0.0)
    after_info = env._with_violation_info({})
    after = bounded_soft_violation_component(
        after_info["time_violation_clipped_sum"], fixed_n, 1.0
    )
    np.testing.assert_allclose(after, before)
    assert after_info["time_violation_applicable_transitions"].tolist() == [101]
    assert after_info["capacity_violation_applicable_transitions"].tolist() == [1]

    # The same violated-customer fraction has the same scale for small and
    # large instances, while the final clip provides a hard upper bound.
    np.testing.assert_allclose(
        bounded_soft_violation_component(
            np.asarray([1.0, 500.0]), np.asarray([2.0, 1000.0]), 1.0
        ),
        np.asarray([0.5, 0.5]),
    )
    assert bounded_soft_violation_component(
        np.asarray([10_000.0]), np.asarray([1000.0]), 1.0
    ).item() == 1.0


def test_signed_soft_profile_is_authoritative_over_legacy_cli_values() -> None:
    args = Namespace(
        method_auxiliary_profile=(
            REPO_ROOT
            / "EVRPTW_Benchmark/Reinforcement_Learning/configs/"
            "drl_ts_soft_auxiliary_v1.json"
        ),
        reward_contract="signed-task-contract.json",
        capacity_penalty=99.0,
        time_penalty=98.0,
        energy_penalty=97.0,
        soft_violation_contract_id="drl_ts_soft_auxiliary_v1",
        soft_violation_step_clip=7.0,
        soft_violation_component_clip=8.0,
        soft_violation_denominator="num_customers",
    )
    _configure_soft_auxiliary(args)
    assert args.capacity_penalty == 1.0
    assert args.time_penalty == 1.0
    assert args.energy_penalty == 1.0
    assert args.soft_violation_step_clip == 1.0
    assert args.soft_violation_component_clip == 1.0
    assert args.method_auxiliary_profile_id == "drl_ts_soft_auxiliary_v1"
    assert len(args.method_auxiliary_sha256) == 64


def test_formal_soft_profile_and_task_contract_are_required_together() -> None:
    profile = (
        REPO_ROOT
        / "EVRPTW_Benchmark/Reinforcement_Learning/configs/"
        "drl_ts_soft_auxiliary_v1.json"
    )
    common = {
        "capacity_penalty": 1.0,
        "time_penalty": 1.0,
        "energy_penalty": 1.0,
        "soft_violation_contract_id": "drl_ts_soft_auxiliary_v1",
        "soft_violation_step_clip": 1.0,
        "soft_violation_component_clip": 1.0,
        "soft_violation_denominator": "num_customers",
    }
    with pytest.raises(ValueError, match="requires --method-auxiliary-profile"):
        _configure_soft_auxiliary(
            Namespace(
                **common,
                method_auxiliary_profile=None,
                method_auxiliary_snapshot=None,
                reward_contract="signed-task-contract.json",
            )
        )
    with pytest.raises(ValueError, match="requires a reward contract"):
        _configure_soft_auxiliary(
            Namespace(
                **common,
                method_auxiliary_profile=profile,
                reward_contract=None,
            )
        )


def _three_customer_instance():
    base = _instance()
    distance = np.ones((5, 5), dtype=np.float32)
    np.fill_diagonal(distance, 0.0)
    return replace(
        base,
        customers=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        ),
        distance_matrix_km=distance,
        demands_cm3=np.ones(3, dtype=np.float32),
        package_counts=np.ones(3, dtype=np.int32),
        service_time_s=np.full(3, 30.0, dtype=np.float32),
        tw_s=np.asarray([[0, 20_000]] * 3, dtype=np.float32),
        shortest_time_matrix_s=distance * 60.0,
        energy_matrix_kwh=distance * 0.5,
    )


def test_station_mask_blocks_same_station_revisit_within_route() -> None:
    instance = _three_customer_instance()
    canonical = EVRPTWVectorEnvFast(instance, n_traj=1, use_jit_mask=False)
    canonical_observation, _ = canonical.reset(seed=5)
    station = canonical.station_start
    assert canonical_observation["action_mask"][0, station]

    hard = DRLTSHardConstraintEnv(instance, n_traj=1, use_jit_mask=False)
    observation, _ = hard.reset(seed=5)
    assert not observation["action_mask"][0, station]

    observation, _, _, _, _ = hard.step(np.asarray([1], dtype=np.int64))
    assert observation["action_mask"][0, station]

    observation, _, _, _, _ = hard.step(np.asarray([station], dtype=np.int64))
    assert not observation["action_mask"][0, station]

    observation, _, _, _, _ = hard.step(np.asarray([2], dtype=np.int64))
    assert not observation["action_mask"][0, station]


def test_rollout_has_gradient_and_hard_result_passes_verifier() -> None:
    policy = _policy()
    sampled = rollout(
        policy,
        [EVRPTWVectorEnvFast(_instance(), n_traj=4, use_jit_mask=False)],
        decode_type="sampling",
        max_steps=32,
        seed=17,
        soft_constraints=False,
    )
    loss = (sampled.training_cost.detach() * sampled.log_likelihood).mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.parameters())

    greedy = rollout(
        policy,
        [EVRPTWVectorEnvFast(_instance(), n_traj=1, use_jit_mask=False)],
        decode_type="greedy",
        max_steps=32,
        seed=19,
        soft_constraints=False,
    )
    assert validate_routes(_instance(), greedy.infos[0]["routes"][0])["passed"]


def test_drl_ts_rollout_reports_training_budget_exhaustion() -> None:
    result = rollout(
        _policy(),
        [EVRPTWVectorEnvFast(_instance(), n_traj=2, use_jit_mask=False)],
        decode_type="sampling",
        max_steps=1,
        seed=41,
        soft_constraints=False,
    )
    assert result.trajectory_steps.tolist() == [[1, 1]]
    assert result.rollout_budget_exhausted.tolist() == [[True, True]]


def test_drl_ts_scaling_is_a_frozen_implementation_contract() -> None:
    env = EVRPTWVectorEnvFast(_instance(), n_traj=1, use_jit_mask=False)
    distance, travel_time, energy = normalized_edge_matrices([env])
    np.testing.assert_allclose(
        distance[0], env.distance_km / env.reward_distance_scale_km
    )
    np.testing.assert_allclose(travel_time[0], env.travel_time_s / env.horizon_s)
    np.testing.assert_allclose(
        energy[0], env.energy_kwh / env.battery_capacity_kwh
    )


@pytest.mark.parametrize("soft", [False, True])
@pytest.mark.parametrize("stride", [1, 2])
def test_decoder_checkpoint_preserves_routes_gradients_rng_and_encoder_bn(soft, stride):
    from copy import deepcopy
    from unittest.mock import patch
    torch.manual_seed(401)
    reference = _policy().train()
    active = deepcopy(reference)
    active.activation_checkpoint_stride = stride
    assert deepcopy(active).activation_checkpoint_stride == stride
    env_class = DRLTSSoftConstraintEnv if soft else DRLTSHardConstraintEnv
    results, rngs = [], []
    for policy in (reference, active):
        torch.manual_seed(409)
        result = rollout(policy, [env_class(_instance(), n_traj=4)],
                         decode_type="sampling", max_steps=32, seed=419, soft_constraints=soft)
        (result.training_cost.detach() * result.log_likelihood).mean().backward()
        results.append(result)
        rngs.append(torch.random.get_rng_state())
    expected, actual = results
    torch.testing.assert_close(actual.training_cost, expected.training_cost, rtol=0, atol=0)
    torch.testing.assert_close(actual.log_likelihood, expected.log_likelihood, rtol=0, atol=0)
    assert actual.infos[0]["routes"] == expected.infos[0]["routes"]
    assert torch.equal(*rngs)
    for got, want in zip(active.parameters(), reference.parameters()):
        assert (got.grad is None) == (want.grad is None)
        if got.grad is not None:
            torch.testing.assert_close(got.grad, want.grad, rtol=2e-5, atol=2e-5)
    for got, want in zip(active.buffers(), reference.buffers()):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    with torch.no_grad(), patch("EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.rollout.checkpoint") as call:
        rollout(active, [env_class(_instance(), n_traj=2)], decode_type="greedy",
                max_steps=32, seed=421, soft_constraints=soft)
        call.assert_not_called()
