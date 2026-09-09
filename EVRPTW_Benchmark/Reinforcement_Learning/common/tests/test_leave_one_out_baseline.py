from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from EVRPTW_Benchmark.Reinforcement_Learning.common import protocol_trainers
from EVRPTW_Benchmark.Reinforcement_Learning.common.tests.test_reinforce_training_diagnostics import _TinyPool, _args


def test_leave_one_out_excludes_own_cost_and_detaches():
    costs = torch.tensor([[1., 3., 8.], [20., 30., 40.]], dtype=torch.float64, requires_grad=True)
    baseline = protocol_trainers.same_instance_leave_one_out(costs)
    torch.testing.assert_close(baseline, torch.tensor([[5.5, 4.5, 2.], [35., 30., 25.]], dtype=torch.float64))
    assert not baseline.requires_grad
    changed = costs.detach().clone()
    changed[0, 0] += 12
    updated = protocol_trainers.same_instance_leave_one_out(changed)
    assert updated[0, 0] == baseline[0, 0]
    torch.testing.assert_close(updated[1], baseline[1])
    torch.testing.assert_close(updated[0, 1:], baseline[0, 1:] + 6)


@pytest.mark.parametrize("shape", [(2,), (2, 1), (2, 3, 1)])
def test_leave_one_out_rejects_ambiguous_or_single_trajectory_cost(shape):
    with pytest.raises(ValueError, match="at least 2 trajectories"):
        protocol_trainers.same_instance_leave_one_out(torch.ones(shape))


def _train(tmp_path, *, method="RRNCO-EV", wrong_log_shape=False, trajectories=2):
    args = _args(tmp_path / "run", fixed=True, cost=False)
    args.reinforce_baseline = "leave_one_out"
    args.samples_per_instance = trajectories
    args.baseline_eval_size = 64  # LOO must suppress the paper baseline probe.
    policy = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(policy.weight)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    before = policy.weight.detach().clone()

    def actor(instances, *_args):
        count = len(instances)
        logits = torch.stack([policy.weight.sum(), -policy.weight.sum()])
        log_likelihood = logits.log_softmax(dim=0).expand(count, 2)
        if wrong_log_shape:
            log_likelihood = log_likelihood[:, :1]
        cost = torch.tensor([[1., 3.]]).expand(count, 2)
        return SimpleNamespace(cost=cost, objective=cost, feasible=torch.ones_like(cost, dtype=torch.bool),
            log_likelihood=log_likelihood, environment_transitions=cost.numel(),
            trajectory_steps=torch.ones_like(cost, dtype=torch.int64),
            rollout_budget_exhausted=torch.zeros_like(cost, dtype=torch.bool))

    protocol_trainers.train_reinforce_data_passes(
        method=method, args=args, pool=_TinyPool(), policy=policy, optimizer=optimizer,
        make_actor=actor,
        make_baseline=lambda *_args: pytest.fail("LOO must not evaluate a greedy baseline"),
        training_cost=lambda result: result.cost,
        objective_distance=lambda result: result.objective,
        feasible=lambda result: result.feasible,
        validation_solve=lambda *_args: {}, legacy_batch_size=1,
    )
    return args, before, policy.weight.detach().clone()


def test_real_trainer_loo_improves_low_cost_action_and_records_provenance(tmp_path):
    args, before, after = _train(tmp_path)
    assert after.item() > before.item()  # Raises the probability of the cost-1 action.
    rows = [json.loads(line) for line in (args.output_dir / "reward_diagnostics.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["baseline_kind"] == "same_instance_leave_one_out" for row in rows)
    terminal = json.loads((args.output_dir / "training_result.json").read_text())
    assert terminal["reinforce_baseline"] == "leave_one_out"
    assert terminal["baseline_eval_count"] == terminal["baseline_update_count"] == 0
    payload = torch.load(args.output_dir / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    assert payload["args"]["reinforce_baseline"] == "leave_one_out"


@pytest.mark.parametrize("method", ["AM-EVRPTW", "DRL-TS", "EVRPTW-RL"])
def test_leave_one_out_cannot_change_benchmark_recipes(tmp_path, method):
    with pytest.raises(ValueError, match="RRNCO-EV experiment only"):
        _train(tmp_path, method=method)


def test_loo_rejects_log_probability_broadcasting(tmp_path):
    with pytest.raises(ValueError, match="shapes must match"):
        _train(tmp_path, wrong_log_shape=True)


def test_loo_rejects_single_trajectory_before_training(tmp_path):
    with pytest.raises(ValueError, match="at least 2 trajectories per instance"):
        _train(tmp_path, trajectories=1)
