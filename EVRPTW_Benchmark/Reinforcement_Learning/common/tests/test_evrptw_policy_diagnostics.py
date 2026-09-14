"""Numerical diagnostic regression: a finite saturated actor must fail."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "EVRPTW_Benchmark/Reinforcement_Learning/scripts"))
from diagnose_evrptw_policy import state_probe, training_pool
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.model import EVRPTWRLPolicy


def observation():
    n = 121
    rng = np.random.default_rng(41)
    coordinates = rng.uniform(0, 1, (1, n, 2)).astype(np.float32)
    mask = np.ones((1, 2, n), dtype=bool)
    mask[:, :, 0] = False
    demand = np.ones((1, 2, n), dtype=np.float32)
    demand[:, :, 0] = 0
    demand[:, :, 101:] = 0
    batch = {
        "depot_loc": coordinates[:, :1], "cus_loc": coordinates[:, 1:101],
        "rs_loc": coordinates[:, 101:],
        "time_window": np.broadcast_to(np.array([0, 1], dtype=np.float32), (1, n, 2)).copy(),
        "service_time": np.zeros((1, n), dtype=np.float32),
        "charging_time_ratio": np.ones((1, n), dtype=np.float32) * 0.1,
        "remaining_demand": demand, "action_mask": mask,
        "last_node_idx": np.zeros((1, 2), dtype=np.int64),
        "current_time": np.zeros((1, 2), dtype=np.float32),
        "remaining_battery": np.ones((1, 2), dtype=np.float32),
        "remaining_vehicle_ratio": np.ones((1, 2), dtype=np.float32),
    }
    travel = np.ones((1, n, n), dtype=np.float32) * 0.01
    travel[:, np.arange(n), np.arange(n)] = 0
    return batch, travel


def test_saturated_finite_policy_is_distinguished_from_effective_mean_policy():
    torch.set_num_threads(1)
    batch, travel = observation()
    probes = {}
    for aggregation in ("sum", "mean"):
        torch.manual_seed(1234)
        policy = EVRPTWRLPolicy(graph_aggregation=aggregation)
        before = {k: v.clone() for k, v in policy.state_dict().items()}
        rng_before = torch.random.get_rng_state().clone()
        probes[aggregation] = state_probe(
            policy, batch, travel, policy.initial_state(1, 2), np.ones((1, 2), bool),
        )
        assert all(torch.equal(v, before[k]) for k, v in policy.state_dict().items())
        assert torch.equal(torch.random.get_rng_state(), rng_before)
        assert all(parameter.grad is None for parameter in policy.parameters())
    assert probes["sum"]["finite_legal_logits"]
    assert probes["sum"]["legal_logit_span"]["p50"] < 1e-4
    assert probes["sum"]["encoder_gradient_norm"] < 1e-8
    assert probes["mean"]["legal_logit_span"]["p50"] > 1e-4
    assert probes["mean"]["encoder_gradient_norm"] > 1e-8


@pytest.mark.parametrize("split", ["val", "validation", "test", "testing"])
def test_diagnostic_refuses_evaluation_index_before_loading(split):
    with pytest.raises(ValueError, match="TRAIN index"):
        training_pool({"dataset_path": f"/unused/{split}/view_index.parquet"})


def test_forced_or_finished_actions_do_not_count_as_policy_discrimination():
    batch, travel = observation()
    batch["action_mask"][:] = False
    batch["action_mask"][:, :, 1] = True
    policy = EVRPTWRLPolicy(graph_aggregation="mean")
    report = state_probe(policy, batch, travel, policy.initial_state(1, 2), np.ones((1, 2), bool))
    assert report["decision_trajectories"] == 0
    assert report["legal_logit_span"]["count"] == 0
    assert report["encoder_gradient_norm"] == 0
