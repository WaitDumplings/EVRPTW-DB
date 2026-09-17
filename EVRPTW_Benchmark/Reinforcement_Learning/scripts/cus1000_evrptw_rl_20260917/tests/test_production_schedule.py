"""Regression for the production 2000-update schedule, without CUDA training.

The production test runs the real parser, method setup, reward/stream verification,
and trainer configuration with the four-rank contract. It stops before data loading
or collectives. A separate seven-update CPU fixture exercises actual stop decisions,
optimizer updates, and checkpoint writes; it is not a four-GPU/NCCL test.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL import distributed_train
from EVRPTW_Benchmark.Reinforcement_Learning.common import distributed_protocol as protocol
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_stream import (
    atomic_write_stream, build_training_stream, file_sha256, load_training_stream_contract,
)
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus1000_evrptw_rl_20260917 import common, launch


@pytest.fixture(scope="module")
def production_stream(tmp_path_factory):
    root = tmp_path_factory.mktemp("cus1000_schedule_stream")
    # Synthetic IDs only: no road matrices, route simulation, or CUDA allocations.
    index = pd.DataFrame({
        "view_id": [f"train-{i}" for i in range(5000)],
        "family_id": [f"family-{i}" for i in range(5000)],
        "split_id": "train", "track_id": "train", "city_slug": "fixture",
        "scale_id": "Cus1000", "customer_count": 1000, "day_type": "weekday",
    })
    index_path = root / "index.parquet"
    index.to_parquet(index_path, index=False)
    config = common.load_config()
    frame, manifest = build_training_stream(
        index, scale="Cus1000", seed=config["seed"], sample_count=config["sample_count"],
    )
    manifest["source_index_sha256"] = file_sha256(index_path)
    path = root / "stream.parquet"
    atomic_write_stream(path, frame, manifest)
    return {"path": str(path), "contract": load_training_stream_contract(path)}


def _actual_args(config, root, stream):
    command = launch.build_command(config, root / "road", root / "run", stream)
    assert command.count("--nproc_per_node=4") == 1
    args = distributed_train.parse_args(command[command.index(config["train_module"]) + 1:])
    distributed_train.prepare_method(args)
    return args


def _trainer_kwargs(args, *, pool=None):
    policy = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    def unused(*_args, **_kwargs):
        raise AssertionError("The configuration-only test must not reach a rollout")

    return dict(
        method="EVRPTW-RL", args=args, pool=pool, policy=policy, optimizer=optimizer,
        make_actor=unused, make_baseline=unused, training_cost=lambda actor: actor.cost,
        objective_distance=lambda actor: actor.objective, feasible=lambda actor: actor.feasible,
        validation_solve=unused, legacy_batch_size=12,
    )


def test_old_production_schedule_is_rejected_before_background_launch(tmp_path):
    config = json.loads(common.CONFIG.read_text())
    config.update(early_stop_patience_validations=5, early_stop_start_epoch=2000)
    path = tmp_path / "invalid_config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="early-stop start must precede maximum training epochs"):
        common.load_config(path)


class _ConfigurationPassed(RuntimeError):
    pass


@pytest.mark.parametrize("old_schedule", [True, False], ids=["old_rejected", "fixed_accepted"])
def test_actual_trainer_production_configuration(tmp_path, monkeypatch, production_stream, old_schedule):
    config = common.load_config()
    assert config["early_stop_patience_validations"] == config["early_stop_start_epoch"] == 0
    if old_schedule:
        # Reproduce the deployed failure through the actual trainer, deliberately
        # bypassing only the new launcher-side validation of the old combination.
        config.update(early_stop_patience_validations=5, early_stop_start_epoch=2000)
    args = _actual_args(config, tmp_path, production_stream)
    assert args.training_epochs == args.minimum_training_epochs == 2000
    assert args.validation_every_epochs == args.post_minimum_validation_every_epochs == 100
    assert args.validation_checkpoints == 20
    assert args.expected_world_size == 4
    assert args.batch_size == args.physical_batch_size == 12
    assert args.effective_batch_size == 48
    assert args.customer_exposure_budget == 96_000_000
    assert args.scale == "Cus1000" and args.graph_aggregation == "mean"
    assert args.samples_per_instance == args.validation_candidates == 30
    assert (args.training_rollout_steps, args.validation_rollout_steps) == (1800, 2700)
    assert args.ema_warmup_steps == 1000

    # No collectives occur before read_stream_view_ids. Keep the real four-rank
    # batch-contract calculations while replacing just the collective wrapper.
    context = SimpleNamespace(world_size=4, local_phase=lambda _name: nullcontext())
    monkeypatch.setattr(protocol.DistributedContext, "current", classmethod(lambda _cls: context))

    def after_configuration(_path, *, stop):
        assert stop == 96_000
        assert args.distributed_contract["world_size"] == 4
        assert args.distributed_contract["microbatches_per_rank"] == 1
        assert args.training_stream_contract_snapshot["sha256"] == production_stream["contract"]["sha256"]
        assert args.reward_contract_snapshot is not None
        raise _ConfigurationPassed("production schedule passed actual trainer checks")

    monkeypatch.setattr(protocol, "read_stream_view_ids", after_configuration)
    expected = ValueError if old_schedule else _ConfigurationPassed
    message = "early-stop start must precede maximum training epochs" if old_schedule else "passed actual trainer checks"
    with pytest.raises(expected, match=message):
        protocol.train_distributed_reinforce_data_passes(**_trainer_kwargs(args))
    assert not (tmp_path / "run/checkpoint_latest.pt").exists()


class _Pool:
    def __init__(self, size, prefix):
        self.tasks = [SimpleNamespace(view_id=f"{prefix}-{i}") for i in range(size)]
        self._task_by_view_id = {task.view_id: task for task in self.tasks}
        self.rng = np.random.default_rng(123)

    def __len__(self):
        return len(self.tasks)

    def instance(self, task):
        return SimpleNamespace(instance_id=task.view_id, index=int(task.view_id.split("-")[-1]))


def test_patience_zero_finishes_fixed_budget_despite_worsening_validation(tmp_path, monkeypatch, production_stream):
    config = deepcopy(common.load_config())
    config.update(training_epochs=7, minimum_training_epochs=7, validation_every_epochs=1,
                  validation_limit=1, customer_exposure_budget=7 * 48 * 1000)
    args = _actual_args(config, tmp_path, production_stream)
    # Run real CPU optimizer/checkpoint/stop logic on one rank, accumulating four
    # microbatches to preserve global batch 48. This is a short schedule fixture.
    args.device = "cpu"
    args.distributed_backend = "gloo"
    args.expected_world_size = 1
    assert args.early_stop_patience_validations == args.early_stop_start_epoch == 0
    assert protocol.DistributedContext.current().world_size == 1
    monkeypatch.setattr(protocol, "make_validation_pool", lambda *_a, **_kw: _Pool(1, "val"))
    kwargs = _trainer_kwargs(args, pool=_Pool(5000, "train"))
    policy = kwargs["policy"]
    validations = []

    def actor(instances, soft, _seed):
        assert not soft
        indices = torch.tensor([1 + instance.index % 7 for instance in instances], dtype=torch.float32)
        trajectories = torch.arange(args.samples_per_instance, dtype=torch.float32)[None, :]
        costs = indices[:, None] + trajectories * 0.01
        logp = policy.weight.sum() * (indices[:, None] + trajectories * 0.02)
        return SimpleNamespace(
            cost=costs, objective=costs, objective_value=costs,
            vehicles_started=torch.ones_like(costs), feasible=torch.ones_like(costs, dtype=torch.bool),
            log_likelihood=logp, environment_transitions=costs.numel(),
            trajectory_steps=torch.ones_like(costs, dtype=torch.int64),
            rollout_budget_exhausted=torch.zeros_like(costs, dtype=torch.bool),
        )

    def validation(instances, _solve, *, seed, objective_config, cuda_rng_devices):
        assert cuda_rng_devices == []
        value = 20.0 + len(validations)
        validations.append(value)
        rows = [dict(instance_id=instance.instance_id, verifier_passed=True,
                     objective_distance_km=value, objective_value=value) for instance in instances]
        return {
            "schema": "drl_validation_summary_v1", "instances": len(rows),
            "complete_and_feasible": len(rows), "complete_and_feasible_rate": 1.0,
            "verifier_summary_passed": True, "objective_mode": objective_config.mode,
            "objective_unit": objective_config.unit, "objective_config": objective_config.to_dict(),
            "mean_verified_distance_km": value, "mean_verified_objective": value,
            "mean_verified_cost_usd": value, "rows": rows,
        }

    monkeypatch.setattr(protocol, "verified_validation", validation)
    kwargs["make_actor"] = actor
    protocol.train_distributed_reinforce_data_passes(**kwargs)
    run = tmp_path / "run"
    result = json.loads((run / "training_result.json").read_text())
    history = [json.loads(line) for line in (run / "validation_history.jsonl").read_text().splitlines()]
    checkpoint = torch.load(run / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    assert validations == [20.0 + i for i in range(7)]
    assert [row["validation_checks_without_improvement"] for row in history] == list(range(7))
    assert all(not row["early_stop_due"] for row in history)
    assert result["status"] == "passed" and not result["early_stopped"]
    assert result["completed_training_epochs"] == result["optimizer_steps"] == 7
    assert result["completed_validation_checkpoints"] == 7
    assert result["instances_seen"] == 7 * 48
    assert result["customer_exposures"] == 7 * 48 * 1000
    assert checkpoint["logical_epoch"] == 7 and checkpoint["stream_cursor"] == 7 * 48
