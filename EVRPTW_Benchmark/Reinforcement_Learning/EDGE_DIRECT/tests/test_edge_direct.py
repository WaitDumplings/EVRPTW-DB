from __future__ import annotations

import sys
from pathlib import Path

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
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.rollout import (
    normalized_edge_matrices,
)
from EVRPTW_Benchmark.Reinforcement_Learning.EDGE_DIRECT.model import (
    EdgeDirectHomogeneousPolicy,
)
from EVRPTW_Benchmark.Reinforcement_Learning.EDGE_DIRECT.rollout import rollout
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import EVRPTWVectorEnvFast


def _policy() -> EdgeDirectHomogeneousPolicy:
    return EdgeDirectHomogeneousPolicy(
        embedding_dim=32,
        n_encode_layers=1,
        n_heads=4,
    )


def test_time_window_graph_is_directed_and_logits_respect_mask() -> None:
    envs = [EVRPTWVectorEnvFast(_instance(), n_traj=2, use_jit_mask=False)]
    observations = [env.reset(seed=5)[0] for env in envs]
    batch = stack_observations(observations)
    _, travel_time, energy = normalized_edge_matrices(envs)
    time_window_travel = torch.as_tensor(
        envs[0].unwrapped.travel_time_s / envs[0].unwrapped.horizon_s,
        dtype=torch.float32,
    )[None]
    policy = _policy()
    fixed = policy.encode(
        batch,
        travel_time,
        energy,
        time_window_travel_time=time_window_travel,
    )
    adjacency = policy.time_window_adjacency(
        batch, time_window_travel
    )
    assert adjacency.shape == (1, 4, 4)
    assert torch.diagonal(adjacency, dim1=-2, dim2=-1).all()

    logits, vehicle_log_probability = policy.logits(batch, fixed)
    mask = torch.as_tensor(batch["action_mask"], dtype=torch.bool)
    assert logits.shape == (1, 2, 4)
    assert torch.isneginf(logits[~mask]).all()
    assert torch.isfinite(logits[mask]).all()
    assert torch.equal(vehicle_log_probability, torch.zeros_like(vehicle_log_probability))


def test_time_window_reachability_uses_directed_horizon_scaled_travel() -> None:
    policy = _policy()
    observation = {
        "time_window": torch.tensor(
            [[[0.0, 0.2], [0.1, 0.4]]], dtype=torch.float32
        )
    }
    travel = torch.tensor(
        [[[0.0, 0.1], [0.5, 0.0]]], dtype=torch.float32
    )
    adjacency = policy.time_window_adjacency(observation, travel)
    assert bool(adjacency[0, 0, 1])
    assert not bool(adjacency[0, 1, 0])


def test_rollout_has_gradient_and_greedy_route_passes_verifier() -> None:
    policy = _policy()
    sampled = rollout(
        policy,
        [EVRPTWVectorEnvFast(_instance(), n_traj=4, use_jit_mask=False)],
        decode_type="sampling",
        max_steps=32,
        seed=17,
        incomplete_penalty_km=10_000.0,
    )
    loss = (sampled.cost_km.detach() * sampled.log_likelihood).mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.parameters())

    greedy = rollout(
        policy,
        [EVRPTWVectorEnvFast(_instance(), n_traj=1, use_jit_mask=False)],
        decode_type="greedy",
        max_steps=32,
        seed=19,
        incomplete_penalty_km=10_000.0,
    )
    routes = greedy.infos[0]["routes"][0]
    verification = validate_routes(_instance(), routes)
    assert verification["passed"], verification["violations"]
