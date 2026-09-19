"""Size-stable message aggregation and frozen legacy checkpoint semantics."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.model import EVRPTWRLPolicy
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.train import configure_method_fields
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_protocol import (
    assert_checkpoint_training_signature, freeze_resolved_training_signature,
    resolved_training_signature_from_args,
)


def _observation(customers):
    generator = torch.Generator().manual_seed(37)
    stations = 20 if customers == 100 else 50
    nodes = 1 + customers + stations
    coordinates = torch.rand((1, nodes, 2), generator=generator)
    start = torch.rand((1, nodes), generator=generator) * 0.2
    observation = {
        "depot_loc": coordinates[:, :1], "cus_loc": coordinates[:, 1:1 + customers],
        "rs_loc": coordinates[:, 1 + customers:],
        "time_window": torch.stack((start, start + 0.7), dim=-1),
        "service_time": torch.full((1, nodes), 0.01),
        "charging_time_ratio": torch.full((1, nodes), 0.05),
        "remaining_demand": torch.rand((1, 2, nodes), generator=generator) * 0.1,
        "current_time": torch.zeros((1, 2)), "remaining_battery": torch.ones((1, 2)),
        "remaining_vehicle_ratio": torch.ones((1, 2)),
        "last_node_idx": torch.zeros((1, 2), dtype=torch.long),
        "action_mask": torch.ones((1, 2, nodes), dtype=torch.bool),
    }
    observation["action_mask"][:, :, 0] = False
    travel = torch.cdist(coordinates, coordinates) * 0.5
    return observation, travel


@pytest.mark.parametrize("customers", [100, 500])
def test_mean_messages_restore_legal_action_separation_and_encoder_gradient(customers):
    torch.manual_seed(1234)
    legacy = EVRPTWRLPolicy()
    stable = EVRPTWRLPolicy(graph_aggregation="mean")
    stable.load_state_dict(legacy.state_dict(), strict=True)
    observation, travel = _observation(customers)
    mask = observation["action_mask"]
    stats = {}
    for name, policy in (("sum", legacy), ("mean", stable)):
        projections = {}
        hooks = [layer.register_forward_hook(
            lambda _layer, _inputs, output, key=key: projections.__setitem__(key, output.detach())
        ) for key, layer in (("context", policy.context_projection), ("choice", policy.choice_projection))]
        logits, _ = policy.logits(observation, travel, policy.initial_state(1, 2))
        for hook in hooks:
            hook.remove()
        loss = -torch.log_softmax(logits, dim=-1)[:, :, 1].mean()
        loss.backward()
        gradients = [parameter.grad for parameter in policy.parameters() if parameter.grad is not None]
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        stats[name] = {
            "range": float((logits[mask].max() - logits[mask].min()).detach()),
            "encoder_grad": float(policy.neighbor_projection.weight.grad.norm()),
            "projection_max": max(float(value.abs().max()) for value in projections.values()),
        }
    assert stats["sum"]["projection_max"] > 1000
    assert stats["mean"]["projection_max"] < 10
    assert stats["mean"]["range"] > 0.01
    assert stats["mean"]["encoder_grad"] > 1e-4
    assert stats["mean"]["encoder_grad"] > stats["sum"]["encoder_grad"] * 100


def test_default_sum_preserves_explicit_legacy_outputs_and_state_keys():
    torch.manual_seed(19)
    legacy = EVRPTWRLPolicy(embedding_dim=16)
    explicit = EVRPTWRLPolicy(embedding_dim=16, graph_aggregation="sum")
    stable = EVRPTWRLPolicy(embedding_dim=16, graph_aggregation="mean")
    for policy in (explicit, stable):
        policy.load_state_dict(legacy.state_dict(), strict=True)
        assert list(policy.state_dict()) == list(legacy.state_dict())
    observation, travel = _observation(100)
    actual, actual_state = explicit.logits(observation, travel, explicit.initial_state(1, 2))
    expected, expected_state = legacy.logits(observation, travel, legacy.initial_state(1, 2))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_state.hidden, expected_state.hidden, rtol=0, atol=0)
    torch.testing.assert_close(actual_state.cell, expected_state.cell, rtol=0, atol=0)


@pytest.mark.parametrize("nodes", [1, 2, 121, 551])
def test_mean_edge_messages_exclude_self_edges_and_use_neighborhood_degree(nodes):
    policy = EVRPTWRLPolicy(embedding_dim=4, graph_aggregation="mean")
    with torch.no_grad():
        policy.edge_direction.fill_(1)
        policy.edge_projection.weight.copy_(torch.eye(4))
    travel = torch.full((1, nodes, nodes), 2.0)
    travel[:, torch.arange(nodes), torch.arange(nodes)] = 999
    actual = policy._edge_message(travel, 1, nodes)
    expected = torch.full((1, 1, nodes, 4), 2.0 if nodes > 1 else 0.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_aggregation_is_signed_without_changing_legacy_sum_signature():
    legacy = SimpleNamespace(activation_checkpoint_stride=0, structure2vec_rounds=3)
    before = resolved_training_signature_from_args(legacy)
    configure_method_fields(legacy)
    assert resolved_training_signature_from_args(legacy) == before
    explicit_sum = deepcopy(legacy)
    explicit_sum.graph_aggregation = "sum"
    configure_method_fields(explicit_sum)
    assert resolved_training_signature_from_args(explicit_sum) == before
    stable = deepcopy(legacy)
    stable.graph_aggregation = "mean"
    configure_method_fields(stable)
    signed = freeze_resolved_training_signature(stable)
    assert signed["sha256"] != before["sha256"]
    assert signed["method_specific"]["graph_aggregation"] == "mean"
    assert signed["method_specific"]["structure2vec_rounds"] == 3
    payload = {"resolved_training_signature": signed, "args": vars(stable)}
    assert_checkpoint_training_signature(payload, stable)
    with pytest.raises(ValueError, match="signature mismatch"):
        assert_checkpoint_training_signature(payload, explicit_sum)
    changed_rounds = deepcopy(legacy)
    changed_rounds.graph_aggregation = "mean"
    changed_rounds.structure2vec_rounds = 2
    configure_method_fields(changed_rounds)
    with pytest.raises(ValueError, match="signature mismatch"):
        assert_checkpoint_training_signature(payload, changed_rounds)


def test_mean_configuration_reaches_distributed_policy_and_signature(tmp_path):
    from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL import distributed_train as entry
    from EVRPTW_Benchmark.Reinforcement_Learning.common.distributed import DistributedContext
    from EVRPTW_Benchmark.Reinforcement_Learning.common.distributed_protocol import configure_distributed_contract
    cli = ["--dataset-path", str(tmp_path / "train.parquet"), "--output-dir", str(tmp_path / "run"),
           "--training-epochs", "6", "--physical-batch-size", "1", "--effective-batch-size", "2",
           "--batch-size", "1", "--device", "cpu", "--graph-aggregation", "mean"]
    args = entry.parse_args(cli)
    entry.prepare_method(args)
    policy = entry.build_policy(args)
    assert policy.graph_aggregation == "mean"
    configure_distributed_contract(args, DistributedContext(rank=0, world_size=2), method="EVRPTW-RL")
    signature = resolved_training_signature_from_args(args)
    assert signature["method_specific"]["architecture"] == "evrptw_rl_structure2vec_mean_v1"
    assert signature["method_specific"]["graph_aggregation"] == "mean"


def test_single_gpu_native_warmup_copies_actor_before_first_greedy_update(tmp_path, monkeypatch):
    from EVRPTW_Benchmark.Reinforcement_Learning.common import protocol_trainers as protocol
    from EVRPTW_Benchmark.Reinforcement_Learning.common.tests.test_reinforce_training_diagnostics import _args, _TinyPool
    args = _args(tmp_path / "warmup", fixed=True, cost=False)
    args.training_epochs = 6
    args.customer_exposure_budget = 6 * 2 * 50
    args.ema_warmup_steps = 2
    args.baseline_eval_interval = 2
    args.baseline_eval_size = 2
    args.baseline_alpha = 0.05
    args.validation_every_epochs = 1
    args.validation_checkpoints = 6
    policy = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    calls = {"actors": 0, "greedy_epochs": []}
    initial_weight = policy.weight.detach().clone()
    monkeypatch.setattr(protocol, "ttest_rel", lambda *_args, **_kwargs: SimpleNamespace(pvalue=1.0))

    def result(instances, *, actor):
        if actor:
            calls["actors"] += 1
            if calls["actors"] == 3:
                assert not torch.equal(policy.weight.detach(), initial_weight)
        count = len(instances)
        costs = torch.tensor([[2.0, 4.0]]).expand(count, -1)
        return SimpleNamespace(
            cost=costs, objective=costs, objective_value=costs,
            vehicles_started=torch.ones_like(costs), feasible=torch.ones_like(costs, dtype=torch.bool),
            log_likelihood=policy.weight.sum() * torch.tensor([[1.0, 2.0]]).expand_as(costs),
            environment_transitions=costs.numel(), trajectory_steps=torch.ones_like(costs, dtype=torch.int64),
            rollout_budget_exhausted=torch.zeros_like(costs, dtype=torch.bool),
        )

    def baseline(model, instances, _soft, _seed):
        epoch = (calls["actors"] + 1) // 2
        assert epoch > 2
        calls["greedy_epochs"].append(epoch)
        if calls["actors"] == 5:
            # The first greedy loss must compare against the actor copied after
            # optimizer update 2, rather than the random initial baseline.
            for actual, expected in zip(model.parameters(), policy.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return result(instances, actor=False)

    protocol.train_reinforce_data_passes(
        method="EVRPTW-RL", args=args, pool=_TinyPool(), policy=policy, optimizer=optimizer,
        make_actor=lambda instances, _soft, _seed: result(instances, actor=True),
        make_baseline=baseline, training_cost=lambda value: value.cost,
        objective_distance=lambda value: value.objective, feasible=lambda value: value.feasible,
        validation_solve=lambda *_args: {}, legacy_batch_size=1,
    )
    assert calls["greedy_epochs"][0] == 3
    events = [json.loads(line) for line in (args.output_dir / "baseline_history.jsonl").read_text().splitlines()]
    assert [row["optimizer_step"] for row in events] == [4, 6]
    native = SimpleNamespace(ema_warmup_steps=1000, baseline_eval_interval=100)
    assert not protocol.paper_baseline_eval_due("EVRPTW-RL", 1000, native)
    assert not protocol.paper_baseline_eval_due("EVRPTW-RL", 1001, native)
    assert protocol.paper_baseline_eval_due("EVRPTW-RL", 1100, native)
