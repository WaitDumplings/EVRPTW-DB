from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from EVRPTW_Benchmark.Reinforcement_Learning.common import protocol_trainers
from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import ObjectiveConfig
from EVRPTW_Benchmark.Reinforcement_Learning.common.reward_contract import (
    reward_contract_digest,
)


class _TinyPool:
    def __len__(self):
        return 3

    def first(self, *, limit):
        return list(range(min(limit, len(self))))

    def data_pass_batches(self, _data_pass, physical):
        for offset in range(0, len(self), physical):
            yield list(range(offset, min(offset + physical, len(self))))

    def stream_batches(self, _path, physical, *, start, stop, logical_batch_size):
        for offset in range(start, stop, physical):
            yield [index % len(self) for index in range(offset, min(offset + physical, stop))]


def _args(output, *, fixed, cost, ema=False):
    objective = ObjectiveConfig(
        mode="energy_vehicle_cost" if cost else "distance",
        profile_id="diagnostic-cost-test" if cost else "distance_v1",
    )
    args = SimpleNamespace(
        objective=objective.to_dict(), output_dir=output, protocol_id="diagnostic-test",
        training_epochs=2 if fixed else None, data_passes=None if fixed else 1,
        training_stream_path=output.parent / "stream.parquet" if fixed else None,
        customer_exposure_budget=200 if fixed else None,
        max_batches_per_pass=None, pilot_mode=True,
        physical_batch_size=1, effective_batch_size=2, scale="Cus50",
        validation_checkpoints=1, validation_limit=0, validation_dataset_path=None,
        validation_every_passes=1, final_validation_limit=0,
        resume=False, seed=1234, baseline_eval_size=0,
        exposure_checkpoints="", gpu_hour_checkpoints="", device="cpu",
        max_grad_norm=0.5, training_rollout_steps=80,
        steps_per_epoch=10, baseline_warmup_epochs=1 if ema else 0,
        ema_warmup_steps=10 if ema else 0, ema_decay=0.9, baseline_eval_interval=1000,
        reward_distance_scale_km=5.0,
        reward_distance_scale_mode="dataset_single_customer_repair_median",
        reward_distance_scale_metadata={"source": "training_pool", "count": 3},
    )
    if cost:
        payload = {
            "schema": "drl_reward_contract_v1",
            "contract_id": "diagnostic-reference-scale-v1",
            "objective": objective.to_dict(),
            "scales": {
                "Cus50": {
                    "objective_scale": float(objective.value(5.0, 1)),
                    "failure_base": 2.0,
                    "unserved_coefficient": 1.0,
                }
            },
        }
        payload["sha256"] = reward_contract_digest(payload)
        args.reward_contract_snapshot = payload
    return args


def _run(args, method):
    # Both random streams are inspected after training, including draws inside
    # synthetic rollouts; diagnostics must not consume either stream.
    torch.manual_seed(901)
    np.random.seed(902)
    policy = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1, momentum=0.9)
    objective = ObjectiveConfig(**args.objective)
    scale = float(objective.value(5.0, 1))
    calls = []

    def result(instances, *, actor):
        calls.append((actor, tuple(instances)))
        count = len(instances)
        offset = torch.tensor(instances, dtype=torch.float32)[:, None]
        cost = offset + torch.tensor([[2.0, 4.0]]) if actor else torch.ones(count, 1)
        distance = torch.ones_like(cost) * 5.0
        vehicles = torch.ones_like(cost)
        base = torch.ones_like(cost)
        return SimpleNamespace(
            cost=cost, objective=distance, objective_value=distance * objective.distance_unit_cost + vehicles * objective.vehicle_unit_cost,
            vehicles_started=vehicles, feasible=torch.ones_like(cost, dtype=torch.bool),
            log_likelihood=policy.weight.sum().expand_as(cost) + torch.rand_like(cost) * 0.01 + np.random.random() * 0.01,
            environment_transitions=cost.numel(),
            trajectory_steps=torch.ones_like(cost, dtype=torch.int64),
            rollout_budget_exhausted=torch.zeros_like(cost, dtype=torch.bool),
            training_cost_components={
                "base_objective": base,
                "base_distance_term": distance * objective.distance_unit_cost / scale,
                "base_vehicle_term": vehicles * objective.vehicle_unit_cost / scale,
                "incomplete_penalty": cost - base,
            },
            reward_objective_scale=torch.full((count, 1), scale, dtype=torch.float64),
        )

    protocol_trainers.train_reinforce_data_passes(
        method=method, args=args, pool=_TinyPool(), policy=policy, optimizer=optimizer,
        make_actor=lambda instances, _soft, _seed: result(instances, actor=True),
        make_baseline=lambda _policy, instances, _soft, _seed: result(instances, actor=False),
        training_cost=lambda value: value.cost,
        objective_distance=lambda value: value.objective,
        feasible=lambda value: value.feasible,
        validation_solve=lambda *_args: {}, legacy_batch_size=1,
        soft_stage_end_epoch=1 if args.training_epochs is not None and method == "DRL-TS" else None,
    )
    return (
        deepcopy(policy.state_dict()), deepcopy(optimizer.state_dict()),
        torch.get_rng_state().clone(), np.random.get_state(), calls,
    )


@pytest.mark.parametrize("method", ["AM-EVRPTW", "EVRPTW-RL", "DRL-TS"])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("cost", [False, True])
def test_diagnostics_preserve_updates_rng_and_log_actual_group_values(
    tmp_path, monkeypatch, method, fixed, cost,
):
    args = _args(tmp_path / "enabled", fixed=fixed, cost=cost)
    enabled = _run(args, method)
    rows = [json.loads(line) for line in (args.output_dir / "reward_diagnostics.jsonl").read_text().splitlines()]
    assert [row["optimizer_steps_total"] for row in rows] == [1, 2]
    assert [row["logical_epoch"] for row in rows] == [1, 2]
    assert [row["data_pass"] for row in rows] == [1, 1]
    assert rows[0]["session_id"] == rows[1]["session_id"]
    assert [row["distributions"]["actor_training_cost"]["count"] for row in rows] == [4, 4 if fixed else 2]
    assert rows[0]["sample_unit"] == "candidate_trajectory_including_failed_or_truncated"
    assert rows[0]["baseline_kind"] == "greedy_rollout"
    assert rows[0]["distributions"]["actor_training_cost"]["mean"] == 3.5
    assert rows[0]["distributions"]["baseline_training_cost"]["mean"] == 1.0
    assert rows[0]["distributions"]["baseline_training_cost"]["count"] == 4
    advantage = rows[0]["distributions"]["pre_loss_advantage"]
    assert advantage["mean"] == 2.5
    assert advantage["std"] == pytest.approx(np.std([1.0, 3.0, 2.0, 4.0]))
    assert rows[0]["gradients"]["pre_clip_norm"]["mean"] == pytest.approx(2.5)
    assert rows[0]["gradients"]["clipping_fraction"] == 1.0
    assert rows[0]["gradients"]["optimizer_updates"] == 1
    assert rows[0]["components"]["base_objective"]["mean"] == 1.0
    assert rows[0]["components"]["incomplete_penalty"]["mean"] == 2.5
    assert rows[0]["component_relationships"]["training_cost"] == ["base_objective", "incomplete_penalty"]
    norm = rows[0]["normalization"]
    assert norm["advantage_standardized"] is False
    assert norm["reward_distance_scale_metadata"] == args.reward_distance_scale_metadata
    assert norm["reward_objective_scale"]["count"] == 2
    objective = ObjectiveConfig(**args.objective)
    assert norm["reward_objective_scale"]["mean"] == pytest.approx(objective.value(5.0, 1))
    if cost:
        assert rows[0]["objective_unit"] == "USD"
        assert rows[0]["distributions"]["raw_electricity_cost_usd"]["mean"] == pytest.approx(5 * objective.distance_unit_cost)
        assert rows[0]["distributions"]["raw_vehicle_cost_usd"]["mean"] == pytest.approx(objective.vehicle_unit_cost)
    else:
        assert "raw_electricity_cost_usd" not in rows[0]["distributions"]
    if fixed and method == "DRL-TS":
        assert [row["training_stage"] for row in rows] == ["soft", "hard"]

    with monkeypatch.context() as patch:
        patch.setattr(protocol_trainers, "_collect_reinforce_diagnostics", lambda **_kwargs: None)
        patch.setattr(protocol_trainers, "_append_reinforce_diagnostics", lambda **_kwargs: None)
        disabled = _run(_args(tmp_path / "disabled", fixed=fixed, cost=cost), method)
    for name in enabled[0]:
        torch.testing.assert_close(enabled[0][name], disabled[0][name], rtol=0, atol=0)
    torch.testing.assert_close(
        enabled[1]["state"][0]["momentum_buffer"], disabled[1]["state"][0]["momentum_buffer"], rtol=0, atol=0,
    )
    assert enabled[1]["param_groups"] == disabled[1]["param_groups"]
    assert torch.equal(enabled[2], disabled[2])
    assert enabled[3][0] == disabled[3][0]
    np.testing.assert_array_equal(enabled[3][1], disabled[3][1])
    assert enabled[3][2:] == disabled[3][2:]
    assert enabled[4] == disabled[4]


@pytest.mark.parametrize("method", ["AM-EVRPTW", "EVRPTW-RL"])
def test_ema_diagnostics_record_actual_sequential_warmup_baseline(tmp_path, method):
    args = _args(tmp_path / method, fixed=False, cost=True, ema=True)
    result = _run(args, method)
    assert all(actor for actor, _instances in result[-1])
    rows = [json.loads(line) for line in (args.output_dir / "reward_diagnostics.jsonl").read_text().splitlines()]
    assert rows[0]["baseline_kind"] == "paper_ema"
    assert rows[0]["distributions"]["baseline_training_cost"]["mean"] == pytest.approx(3.05)
    assert rows[0]["distributions"]["pre_loss_advantage"]["mean"] == pytest.approx(0.45)
    assert rows[0]["gradients"]["pre_clip_norm"]["mean"] == pytest.approx(0.45)
    assert rows[0]["gradients"]["clipping_fraction"] == 0.0


def test_diagnostic_collector_owns_values_and_writes_strict_nonfinite_json(tmp_path):
    values = torch.tensor([1.0, float("nan"), float("inf")], requires_grad=True)
    groups = {}
    before = torch.get_rng_state().clone()
    protocol_trainers._collect_diagnostic_values(groups, "actor_training_cost", values)
    with torch.no_grad():
        values[0] = 99
    args = _args(tmp_path, fixed=False, cost=False)
    protocol_trainers._append_reinforce_diagnostics(
        output=tmp_path, method="AM-EVRPTW", args=args,
        objective_config=ObjectiveConfig(**args.objective),
        session={"session_id": "test"}, data_pass=1, logical_epoch=1,
        optimizer_steps=1, soft=False, baseline_kind="greedy_rollout",
        distributions=groups, components={}, scales={}, pre_clip_norm=float("nan"),
    )
    text = (tmp_path / "reward_diagnostics.jsonl").read_text()
    def reject_constant(value):
        raise AssertionError(f"nonstandard JSON constant: {value}")
    row = json.loads(text, parse_constant=reject_constant)
    assert row["distributions"]["actor_training_cost"]["mean"] == 1.0
    assert row["distributions"]["actor_training_cost"]["nonfinite_count"] == 2
    assert row["gradients"]["pre_clip_norm"]["mean"] is None
    assert row["gradients"]["clipping_fraction"] is None
    assert row["normalization"]["reward_objective_scale"]["count"] == 0
    assert torch.equal(before, torch.get_rng_state())
    assert values.grad is None


def test_warm_start_imports_policy_only_and_resets_training_state(tmp_path: Path) -> None:
    args = _args(tmp_path / "target", fixed=True, cost=True)
    args.scale = "Cus50"
    args.training_representation = "G"
    args.soft_stage_contract_snapshot = None
    protocol_trainers.reward_contract_from_args(
        args,
        objective=ObjectiveConfig(**args.objective),
        scale=args.scale,
    )
    source = torch.nn.Linear(1, 1, bias=False)
    source_baseline = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        source.weight.fill_(3.0)
        source_baseline.weight.fill_(9.0)
    saved_args = {
        "scale": args.scale,
        "training_representation": args.training_representation,
        "reward_contract_snapshot": args.reward_contract_snapshot,
        "reward_contract_scale": args.reward_contract_scale,
        "reward_contract_id": args.reward_contract_id,
        "reward_contract_sha256": args.reward_contract_sha256,
        "reward_objective_scale": args.reward_objective_scale,
        "reward_failure_base": args.reward_failure_base,
        "reward_unserved_coefficient": args.reward_unserved_coefficient,
    }
    checkpoint = tmp_path / "source.ckpt"
    torch.save(
        {
            "method": "AM-EVRPTW",
            "model": source.state_dict(),
            "baseline": source_baseline.state_dict(),
            "optimizer": {"sentinel": "must-not-load"},
            "logical_epoch": 4321,
            "objective_config": args.objective,
            "reward_contract": args.reward_contract_snapshot,
            "soft_stage_contract": None,
            "args": saved_args,
        },
        checkpoint,
    )
    policy = torch.nn.Linear(1, 1, bias=False)
    baseline = torch.nn.Linear(1, 1, bias=False)
    provenance = protocol_trainers._load_warm_start_checkpoint(
        checkpoint,
        method="AM-EVRPTW",
        policy=policy,
        baseline=baseline,
        objective_config=args.objective,
        contract_args=args,
    )
    assert policy.weight.item() == pytest.approx(3.0)
    assert baseline.weight.item() == pytest.approx(3.0)
    assert provenance["source_logical_epoch"] == 4321
    assert provenance["optimizer_reset"] is True
    assert provenance["epoch_reset"] is True
    assert provenance["validation_state_reset"] is True
    args.training_representation = "E"
    with pytest.raises(ValueError, match="training_representation mismatch"):
        protocol_trainers._load_warm_start_checkpoint(
            checkpoint,
            method="AM-EVRPTW",
            policy=policy,
            baseline=baseline,
            objective_config=args.objective,
            contract_args=args,
        )
