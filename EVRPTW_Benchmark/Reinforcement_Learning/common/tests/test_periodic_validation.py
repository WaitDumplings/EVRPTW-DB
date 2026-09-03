from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from EVRPTW_Benchmark.Reinforcement_Learning.common import protocol_trainers


class _Pool:
    def __len__(self) -> int:
        return 100

    def first(self, *, limit: int):
        return [object() for _ in range(min(limit, 1))]

    def stream_batches(self, _path, physical: int, *, start: int, stop: int):
        for offset in range(start, stop, physical):
            yield [object() for _ in range(min(physical, stop - offset))]


class _ValidationPool:
    def first(self, *, limit: int):
        return [object() for _ in range(min(limit, 3))]


def test_fixed_epoch_validation_selects_best_and_records_every_interval(
    tmp_path, monkeypatch
) -> None:
    validation_calls = []

    def fake_validation(instances, _solve, *, seed):
        validation_calls.append((len(list(instances)), seed))
        score = 10.0 - len(validation_calls)
        count = validation_calls[-1][0]
        return {
            "schema": "drl_validation_summary_v1",
            "instances": count,
            "complete_and_feasible": count,
            "complete_and_feasible_rate": 1.0,
            "mean_verified_distance_km": score,
            "verifier_summary_passed": True,
            "rows": [],
        }

    monkeypatch.setattr(
        protocol_trainers, "make_validation_pool", lambda *_args, **_kwargs: _ValidationPool()
    )
    monkeypatch.setattr(protocol_trainers, "verified_validation", fake_validation)

    policy = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)

    def result(instances):
        count = len(instances)
        log_likelihood = policy.weight.sum().expand(count, 1)
        return SimpleNamespace(
            cost=torch.ones(count, 1),
            objective=torch.ones(count, 1),
            feasible=torch.ones(count, 1, dtype=torch.bool),
            log_likelihood=log_likelihood,
            environment_transitions=count,
            trajectory_steps=torch.ones(count, 1, dtype=torch.int64),
            rollout_budget_exhausted=torch.zeros(count, 1, dtype=torch.bool),
        )

    args = SimpleNamespace(
        training_epochs=4,
        data_passes=None,
        max_batches_per_pass=None,
        pilot_mode=False,
        validation_every_epochs=2,
        validation_checkpoints=2,
        physical_batch_size=1,
        effective_batch_size=2,
        training_stream_path=tmp_path / "stream.parquet",
        customer_exposure_budget=400,
        scale="Cus50",
        output_dir=tmp_path / "run",
        protocol_id="periodic-validation-test",
        resume=False,
        validation_limit=2,
        final_validation_limit=3,
        validation_every_passes=5,
        seed=1234,
        baseline_eval_size=0,
        exposure_checkpoints="",
        gpu_hour_checkpoints="",
        device="cpu",
        max_grad_norm=1.0,
        training_rollout_steps=80,
    )
    protocol_trainers.train_reinforce_data_passes(
        method="DRL-TS",
        args=args,
        pool=_Pool(),
        policy=policy,
        optimizer=optimizer,
        make_actor=lambda instances, _soft, _seed: result(instances),
        make_baseline=lambda _model, instances, _soft, _seed: result(instances),
        training_cost=lambda value: value.cost,
        objective_distance=lambda value: value.objective,
        feasible=lambda value: value.feasible,
        validation_solve=lambda *_args: {},
        legacy_batch_size=1,
    )

    output = args.output_dir
    history = [
        json.loads(line)
        for line in (output / "validation_history.jsonl").read_text().splitlines()
    ]
    assert [row["logical_epoch"] for row in history] == [2, 4]
    assert [count for count, _ in validation_calls] == [2, 2, 3]
    assert (output / "checkpoint_epoch_0002.pt").is_file()
    assert (output / "checkpoint_epoch_0004.pt").is_file()
    assert (output / "best.ckpt").read_bytes() == (
        output / "checkpoint_selected.pt"
    ).read_bytes()
    summary = json.loads((output / "validation_summary.json").read_text())
    assert summary["logical_epoch"] == 4
    final_audit = json.loads(
        (output / "validation_final_audit.json").read_text()
    )
    assert final_audit["schema"] == "drl_final_validation_audit_v1"
    assert final_audit["instances"] == 3
    assert final_audit["selection_logical_epoch"] == 4
    assert final_audit["selection_changed"] is False
