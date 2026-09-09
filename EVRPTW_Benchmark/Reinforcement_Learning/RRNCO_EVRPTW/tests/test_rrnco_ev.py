from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from EVRPTW_Benchmark.Exact.Gurobi_Solver.route_validator import validate_routes
from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.rollout import stack_observations
from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import _instance
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.rollout import normalized_edge_matrices
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import EVRPTWVectorEnvFast
from EVRPTW_Benchmark.Reinforcement_Learning.RRNCO_EVRPTW.model import RRNCOEVPolicy
from EVRPTW_Benchmark.Reinforcement_Learning.RRNCO_EVRPTW.rollout import rollout
from EVRPTW_Benchmark.Reinforcement_Learning.common.protocol_trainers import (
    paper_baseline_eval_due,
    paper_ema_baseline_due,
)


def _policy() -> RRNCOEVPolicy:
    return RRNCOEVPolicy(
        embedding_dim=32,
        n_encode_layers=1,
        n_heads=4,
        feedforward_hidden=64,
        distance_sample_size=3,
    )


def _encoded(policy: RRNCOEVPolicy):
    env = EVRPTWVectorEnvFast(_instance(), n_traj=2, use_jit_mask=False)
    observation, _ = env.reset(seed=7)
    batch = stack_observations([observation])
    matrices = normalized_edge_matrices([env])
    torch.manual_seed(101)
    return batch, matrices, policy.encode(batch, *matrices)


def test_rrnco_ev_uses_directed_d_t_e_and_respects_mask() -> None:
    policy = _policy().eval()
    batch, matrices, fixed = _encoded(policy)
    logits, state = policy.logits(batch, fixed)
    assert state is None
    assert logits.shape == (1, 2, 4)
    mask = torch.as_tensor(batch["action_mask"], dtype=torch.bool)
    assert torch.isneginf(logits[~mask]).all()
    assert torch.isfinite(logits[mask]).all()

    distance, duration, energy = (value.copy() for value in matrices)
    distance[:, 0, 1] += 0.75
    duration[:, 0, 1] += 0.5
    energy[:, 0, 1] += 0.25
    torch.manual_seed(101)
    changed = policy.encode(batch, distance, duration, energy)
    assert not torch.allclose(fixed.row_embeddings, changed.row_embeddings)
    assert not torch.allclose(fixed.col_embeddings, changed.col_embeddings)


def test_rrnco_ev_sampling_has_gradient_and_route_passes_verifier() -> None:
    policy = _policy()
    sampled = rollout(
        policy,
        [EVRPTWVectorEnvFast(_instance(), n_traj=4, use_jit_mask=False)],
        decode_type="sampling",
        max_steps=32,
        seed=17,
    )
    loss = (sampled.training_cost.detach() * sampled.log_likelihood).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in policy.parameters())

    policy.eval()
    with torch.no_grad():
        greedy = rollout(
            policy,
            [EVRPTWVectorEnvFast(_instance(), n_traj=1, use_jit_mask=False)],
            decode_type="greedy",
            max_steps=32,
            seed=19,
            compute_log_likelihood=False,
        )
    verification = validate_routes(_instance(), greedy.infos[0]["routes"][0])
    assert verification["passed"], verification["violations"]


def test_rrnco_ev_rejects_nonfinite_relation_matrix() -> None:
    policy = _policy()
    batch, matrices, _ = _encoded(policy)
    distance, duration, energy = matrices
    distance = distance.copy()
    distance[0, 0, 1] = np.inf
    with pytest.raises(ValueError, match="finite D/T/E"):
        policy.encode(batch, distance, duration, energy)


def test_rrnco_ev_uses_same_rollout_baseline_schedule_as_am() -> None:
    class Args:
        steps_per_epoch = 2500
        baseline_warmup_epochs = 1

    assert paper_ema_baseline_due("RRNCO-EV", 2499, Args())
    assert not paper_ema_baseline_due("RRNCO-EV", 2500, Args())
    assert paper_baseline_eval_due("RRNCO-EV", 2500, Args())
