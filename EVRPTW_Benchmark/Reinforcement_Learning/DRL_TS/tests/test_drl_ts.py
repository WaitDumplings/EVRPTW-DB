from __future__ import annotations

import sys
from dataclasses import replace
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
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.env import (
    DRLTSHardConstraintEnv,
)
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.model import DRLTSPolicy
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.rollout import (
    normalized_edge_matrices,
    rollout,
)
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.soft_env import (
    DRLTSSoftConstraintEnv,
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


def test_paper_station_mask_blocks_full_battery_but_allows_revisit() -> None:
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
    assert observation["action_mask"][0, station]


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
    assert distance.min() == 0.0 and distance.max() == 1.0
    assert travel_time.min() == 0.0 and travel_time.max() == 1.0
    assert energy.min() == 0.0 and energy.max() == 1.0
