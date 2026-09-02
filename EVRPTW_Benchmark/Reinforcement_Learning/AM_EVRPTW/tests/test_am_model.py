from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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
