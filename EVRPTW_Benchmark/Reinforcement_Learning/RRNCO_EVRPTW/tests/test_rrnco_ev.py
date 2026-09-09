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


@pytest.mark.parametrize("mode,ignored", [("node_only", (0, 1, 2)), ("distance", (1, 2)), ("distance_time", (2,))])
def test_graph_ablation_removes_every_disabled_matrix_path(mode, ignored) -> None:
    policy = RRNCOEVPolicy(
        embedding_dim=32, n_encode_layers=1, n_heads=4,
        feedforward_hidden=64, distance_sample_size=3,
        graph_mode=mode, aft_mode="stable", distance_sampling="nearest",
    ).eval()
    batch, matrices, fixed = _encoded(policy)
    changed = [value.copy() for value in matrices]
    for index in ignored:
        changed[index][:, 0, 1] += 2.0
        changed[index][:, 2, 3] += 3.0
    other = policy.encode(batch, *changed)
    torch.testing.assert_close(fixed.row_embeddings, other.row_embeddings, rtol=0, atol=0)
    torch.testing.assert_close(fixed.col_embeddings, other.col_embeddings, rtol=0, atol=0)
    torch.testing.assert_close(policy.logits(batch, fixed)[0], policy.logits(batch, other)[0], rtol=0, atol=0)


def test_checkpointed_chunked_bias_preserves_outputs_and_gradients() -> None:
    from EVRPTW_Benchmark.Reinforcement_Learning.RRNCO_EVRPTW.model import RelationBiasFusion

    torch.manual_seed(83)
    original = RelationBiasFusion(16).train()
    chunked = RelationBiasFusion(16, chunk_size=2, checkpoint_bias=True).train()
    chunked.load_state_dict(original.state_dict())
    inputs = (torch.rand(2, 5, 2), *(torch.rand(2, 5, 5) for _ in range(3)))
    reference = original(*inputs)
    actual = chunked(*inputs)
    torch.testing.assert_close(actual, reference)
    weights = torch.randn_like(reference)
    (reference * weights).sum().backward()
    (actual * weights).sum().backward()
    for left, right in zip(original.parameters(), chunked.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-4, atol=2e-6)


def test_stable_aft_matches_direct_exponential_formula_and_graph_gradient() -> None:
    from EVRPTW_Benchmark.Reinforcement_Learning.RRNCO_EVRPTW.model import AFTFull

    torch.manual_seed(87)
    aft = AFTFull(8, mode="stable").double()
    row, col = (torch.randn(2, 5, 8, dtype=torch.float64) for _ in range(2))
    bias = torch.randn(2, 5, 5, dtype=torch.float64, requires_grad=True)
    result = aft(row, col, bias)
    weights = torch.softmax(bias[..., None] + aft.key(col)[:, None], dim=2)
    reference = aft.project(torch.sigmoid(aft.query(row)) * (weights * aft.value(col)[:, None]).sum(2))
    torch.testing.assert_close(result, reference)
    result.square().sum().backward()
    assert torch.isfinite(bias.grad).all()
    assert bias.grad.abs().sum() > 0
    # A row-wise logit shift cannot change the attention distribution.
    torch.testing.assert_close(aft(row, col, bias + 1000), result)


def test_nearest_distance_embedding_is_deterministic_and_direction_specific() -> None:
    from EVRPTW_Benchmark.Reinforcement_Learning.RRNCO_EVRPTW.model import DirectedDistanceExpert

    expert = DirectedDistanceExpert(3, sample_size=3, sampling="nearest")
    with torch.no_grad():
        expert.row.weight.copy_(torch.eye(3))
        expert.col.weight.copy_(torch.eye(3))
        expert.row.bias.zero_()
        expert.col.bias.zero_()
    distance = torch.tensor([[[0.0, 1.0, 9.0], [4.0, 0.0, 3.0], [2.0, 8.0, 0.0]]])
    row, col = expert(distance)
    torch.testing.assert_close(row[0, 0], torch.tensor([1.0, 9.0, 9.0]))
    torch.testing.assert_close(col[0, 0], torch.tensor([2.0, 4.0, 4.0]))
    again = expert(distance)
    torch.testing.assert_close(row, again[0], rtol=0, atol=0)
    torch.testing.assert_close(col, again[1], rtol=0, atol=0)


@pytest.mark.parametrize("graph_mode", ["full", "node_only"])
def test_stable_rrnco_full_rollout_gradient_and_canonical_verifier(graph_mode) -> None:
    policy = RRNCOEVPolicy(
        embedding_dim=32, n_encode_layers=1, n_heads=4,
        feedforward_hidden=64, distance_sample_size=3,
        graph_mode=graph_mode, aft_mode="stable", distance_sampling="nearest",
        relation_chunk_size=2, checkpoint_bias=True, relation_temperature=5.0,
    )
    torch.manual_seed(17)
    result = rollout(
        policy, [EVRPTWVectorEnvFast(_instance(), n_traj=4, use_jit_mask=False)],
        decode_type="sampling", max_steps=32, seed=17,
    )
    (result.training_cost.detach() * result.log_likelihood).mean().backward()
    gradients = [value.grad for value in policy.parameters() if value.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
    if graph_mode == "full":
        assert policy.encoder[0].row.bias.distance[0].weight.grad.abs().sum() > 0
    for route in result.infos[0]["routes"]:
        verification = validate_routes(_instance(), route)
        assert verification["passed"], verification["violations"]


def test_stable_aft_handles_opposing_large_relation_and_key_logits() -> None:
    from EVRPTW_Benchmark.Reinforcement_Learning.RRNCO_EVRPTW.model import AFTFull

    # Separate exp(bias-max) * exp(key-max) underflows to zero for BOTH nodes.
    # Joint logits are [0, 0], whose exact weighted value is their average.
    bias = torch.tensor([[[0.0, -1000.0]]], requires_grad=True)
    key = torch.tensor([[[-1000.0], [0.0]]], requires_grad=True)
    value = torch.tensor([[[2.0], [6.0]]], requires_grad=True)
    result = AFTFull._stable_mix(bias, key, value)
    torch.testing.assert_close(result, torch.tensor([[[4.0]]]))
    result.sum().backward()
    assert all(torch.isfinite(item.grad).all() for item in (bias, key, value))
    assert bias.grad.abs().sum() > 0


def _checkpoint_for_metadata() -> dict:
    return {
        "method": "RRNCO-EV",
        "args": {
            "embedding_dim": 32, "n_encode_layers": 1, "n_heads": 4,
            "feedforward_hidden": 64, "distance_sample_size": 3,
            "tanh_clipping": 10.0, "scale": "Cus50", "seed": 1234,
            "validation_rollout_steps": 98, "training_rollout_steps": 65,
        },
    }


def _evaluation_cli():
    from argparse import Namespace
    return Namespace(
        checkpoint=Path("old.ckpt"), dataset_path=Path("val.parquet"),
        family_root=Path("families"), limit=500, candidates=100, seed=910001234,
    )


def test_checkpoint_export_restores_legacy_defaults_and_records_evaluation_contract() -> None:
    from EVRPTW_Benchmark.Reinforcement_Learning.RRNCO_EVRPTW.evaluate_checkpoint import (
        _checkpoint_args, _evaluation_metadata, _policy as load_policy,
    )

    payload = _checkpoint_for_metadata()
    saved = _checkpoint_args(payload)
    model = load_policy(payload["method"], saved)
    model.load_state_dict(_policy().state_dict(), strict=True)
    assert model.graph_mode == "full"
    assert model.encoder[0].row.aft.mode == "legacy"
    assert model.initial.distance_expert.sampling == "random"
    metadata = _evaluation_metadata(payload, saved, _evaluation_cli())
    assert metadata["graph_mode"] == "full"
    assert metadata["aft_mode"] == "legacy"
    assert metadata["reinforce_baseline"] == "paper"
    assert metadata["relation_temperature"] == pytest.approx(np.exp(5))
    assert metadata["checkpoint_bias"] is False
    assert metadata["scale"] == "Cus50"
    assert metadata["split"] == "validation"
    assert metadata["decode_type"] == "sampling"
    assert metadata["validation_rollout_steps"] == 98
    assert metadata["validation_candidates"] == 100
    assert metadata["validation_seed"] == 910001234
    assert metadata["training_seed"] == metadata["seed"] == 1234
    assert metadata["resolved_training_signature"] is None
    assert metadata["training_stream_contract_sha256"] is None
    assert "graph_mode" not in payload["args"]  # Loading must not mutate provenance.


def test_checkpoint_export_uses_signature_fallback_for_actual_restored_policy() -> None:
    from EVRPTW_Benchmark.Reinforcement_Learning.RRNCO_EVRPTW.evaluate_checkpoint import (
        _checkpoint_args, _evaluation_metadata, _policy as load_policy,
    )

    payload = _checkpoint_for_metadata()
    del payload["args"]["validation_rollout_steps"]
    payload["resolved_training_signature"] = {
        "sha256": "signature-digest", "validation_rollout_steps": 180,
        "training_trajectory_count": 5, "effective_batch_size": 24,
        "method_specific": {
            "graph_mode": "node_only", "aft_mode": "stable",
            "reinforce_baseline": "leave_one_out", "distance_sampling": "nearest",
            "relation_temperature": 5.0, "checkpoint_bias": True,
            "relation_chunk_size": 32,
        },
    }
    payload["training_stream_contract"] = {"sha256": "stream-digest"}
    payload["reward_contract"] = {"sha256": "reward-digest"}
    saved = _checkpoint_args(payload)
    model = load_policy(payload["method"], saved)
    metadata = _evaluation_metadata(payload, saved, _evaluation_cli())
    assert model.graph_mode == metadata["graph_mode"] == "node_only"
    assert model.encoder[0].row.aft.mode == metadata["aft_mode"] == "stable"
    assert model.initial.distance_expert.sampling == metadata["distance_sampling"] == "nearest"
    assert metadata["reinforce_baseline"] == "leave_one_out"
    assert metadata["validation_rollout_steps"] == 180
    assert metadata["training_trajectory_count"] == 5
    assert metadata["resolved_training_signature"] == payload["resolved_training_signature"]
    assert metadata["resolved_training_signature_sha256"] == "signature-digest"
    assert metadata["training_stream_contract_sha256"] == "stream-digest"
    assert metadata["reward_contract_sha256"] == "reward-digest"
    assert metadata["candidate_count"] == 100  # Actual CLI evaluation budget.


def test_checkpoint_restore_rejects_conflicting_graph_semantics() -> None:
    from EVRPTW_Benchmark.Reinforcement_Learning.RRNCO_EVRPTW.evaluate_checkpoint import _checkpoint_args

    payload = _checkpoint_for_metadata()
    payload["args"]["graph_mode"] = "full"
    payload["resolved_training_signature"] = {"method_specific": {"graph_mode": "node_only"}}
    with pytest.raises(ValueError, match="disagree on graph_mode"):
        _checkpoint_args(payload)
