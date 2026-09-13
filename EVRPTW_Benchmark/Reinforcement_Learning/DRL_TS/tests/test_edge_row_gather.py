"""Direct row indexing preserves the original decoder and repeated-index gradients."""
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import torch

from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.rollout import stack_observations
from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import _instance
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS import model as module
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.env import DRLTSHardConstraintEnv
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.rollout import normalized_edge_matrices


def _legacy_gather(edges, last):
    batch, nodes, _, embedding = edges.shape
    trajectories = last.shape[1]
    expanded = edges[:, None].expand(-1, trajectories, -1, -1, -1)
    index = last[:, :, None, None, None].expand(batch, trajectories, 1, nodes, embedding)
    return torch.gather(expanded, 2, index).squeeze(2)


def test_direct_edge_rows_match_expanded_gather_with_repeated_origins():
    torch.manual_seed(710)
    edges = torch.randn(2, 5, 5, 3, dtype=torch.float64, requires_grad=True)
    reference = edges.detach().clone().requires_grad_()
    last = torch.tensor([[0, 0, 3], [4, 2, 2]])
    actual = module._gather_edge_rows(edges, last)
    expected = _legacy_gather(reference, last)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    (actual * upstream).sum().backward()
    (expected * upstream).sum().backward()
    torch.testing.assert_close(edges.grad, reference.grad)
    torch.testing.assert_close(edges.grad[0, 0], upstream[0, 0] + upstream[0, 1])
    torch.testing.assert_close(edges.grad[1, 2], upstream[1, 1] + upstream[1, 2])
    assert torch.count_nonzero(edges.grad[0, 1]) == 0


def test_actual_decoder_logits_state_and_all_gradients_match_legacy_edge_gather():
    torch.manual_seed(713)
    reference = module.DRLTSPolicy(embedding_dim=16, n_encode_layers=1, n_heads=2, nearest_neighbors=2).train()
    direct = deepcopy(reference)
    envs = [DRLTSHardConstraintEnv(_instance(), n_traj=3, use_jit_mask=False) for _ in range(2)]
    batch = stack_observations([env.reset(seed=719 + i)[0] for i, env in enumerate(envs)])
    batch["last_node_idx"] = np.asarray([[0, 0, 1], [2, 2, 0]])
    batch["action_mask"] = np.ones_like(batch["action_mask"], dtype=bool)
    matrices = normalized_edge_matrices(envs)
    hidden = torch.randn(2, 3, 16)
    coefficients = torch.randn(2, 3, 4)
    values = []
    for policy, gather in ((reference, _legacy_gather), (direct, module._gather_edge_rows)):
        fixed = policy.encode(batch, *matrices)
        fixed.edge_embeddings.retain_grad()
        fixed.node_embeddings.retain_grad()
        state = policy.initial_state(2, 3)
        state.hidden.copy_(hidden)
        before_rng = torch.get_rng_state().clone()
        with patch.object(module, "_gather_edge_rows", gather):
            logits, next_state = policy.logits(batch, fixed, state)
            ((logits * coefficients).sum() + next_state.hidden.square().mean()).backward()
        torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)
        values.append((logits, next_state.hidden, fixed.edge_embeddings.grad, fixed.node_embeddings.grad))
    for actual, expected in zip(values[1], values[0]):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    for (name, actual), expected in zip(direct.named_parameters(), reference.parameters()):
        assert (actual.grad is None) == (expected.grad is None), name
        if actual.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-6, msg=name)
    for actual, expected in zip(direct.buffers(), reference.buffers()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
