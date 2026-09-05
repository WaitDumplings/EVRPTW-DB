from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from EVRPTW_Benchmark.Reinforcement_Learning.common import protocol_trainers, training_protocol


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
        training_epochs=6,
        data_passes=None,
        max_batches_per_pass=None,
        pilot_mode=False,
        validation_every_epochs=2,
        minimum_training_epochs=4,
        post_minimum_validation_every_epochs=1,
        validation_checkpoints=4,
        physical_batch_size=1,
        effective_batch_size=2,
        training_stream_path=tmp_path / "stream.parquet",
        customer_exposure_budget=600,
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
        soft_stage_end_epoch=2,
    )

    output = args.output_dir
    history = [
        json.loads(line)
        for line in (output / "validation_history.jsonl").read_text().splitlines()
    ]
    assert [row["logical_epoch"] for row in history] == [2, 4, 5, 6]
    logical_history = [
        json.loads(line)
        for line in (output / "logical_epoch_history.jsonl").read_text().splitlines()
    ]
    assert [row["training_stage"] for row in logical_history] == [
        "soft", "soft", "hard", "hard", "hard", "hard"
    ]
    assert all(row["validation_wall_time_s"] >= 0 for row in history)
    assert [row["checkpoint_selected"] for row in history] == [True] * 4
    assert [row["best_within_minimum_selected"] for row in history] == [
        True,
        True,
        False,
        False,
    ]
    assert [count for count, _ in validation_calls] == [2, 2, 2, 2, 3]
    assert len({seed for _, seed in validation_calls[:-1]}) == 1
    assert (output / "checkpoint_epoch_0002.pt").is_file()
    assert (output / "checkpoint_epoch_0004.pt").is_file()
    assert (output / "best.ckpt").read_bytes() == (
        output / "checkpoint_selected.pt"
    ).read_bytes()
    assert (output / "best.ckpt").read_bytes() == (
        output / "best_overall.ckpt"
    ).read_bytes()
    summary = json.loads((output / "validation_summary.json").read_text())
    assert summary["logical_epoch"] == 6
    within_summary = json.loads(
        (output / "validation_summary_within_5000.json").read_text()
    )
    assert within_summary["logical_epoch"] == 4
    overall_summary = json.loads(
        (output / "validation_summary_overall.json").read_text()
    )
    assert overall_summary["logical_epoch"] == 6
    within_payload = torch.load(
        output / "best_within_5000.ckpt", map_location="cpu", weights_only=False
    )
    overall_payload = torch.load(
        output / "best_overall.ckpt", map_location="cpu", weights_only=False
    )
    selected_payload = torch.load(
        output / "checkpoint_selected.pt", map_location="cpu", weights_only=False
    )
    assert within_payload["logical_epoch"] == 4
    assert overall_payload["logical_epoch"] == 6
    assert selected_payload["logical_epoch"] == 6
    final_audit = json.loads(
        (output / "validation_final_audit.json").read_text()
    )
    assert final_audit["schema"] == "drl_final_validation_audit_v1"
    assert final_audit["instances"] == 3
    assert final_audit["selection_logical_epoch"] == 6
    assert final_audit["selection_changed"] is False


def test_fixed_epoch_early_stop_waits_until_after_start_epoch(tmp_path, monkeypatch) -> None:
    validation_calls = []

    def flat_validation(instances, _solve, *, seed):
        count = len(list(instances))
        validation_calls.append(seed)
        return {
            "schema": "drl_validation_summary_v1",
            "instances": count,
            "complete_and_feasible": count,
            "complete_and_feasible_rate": 1.0,
            "mean_verified_distance_km": 10.0,
            "verifier_summary_passed": True,
            "rows": [],
        }

    monkeypatch.setattr(
        protocol_trainers, "make_validation_pool", lambda *_args, **_kwargs: _ValidationPool()
    )
    monkeypatch.setattr(protocol_trainers, "verified_validation", flat_validation)
    policy = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)

    def result(instances):
        count = len(instances)
        return SimpleNamespace(
            cost=torch.ones(count, 1),
            objective=torch.ones(count, 1),
            feasible=torch.ones(count, 1, dtype=torch.bool),
            log_likelihood=policy.weight.sum().expand(count, 1),
            environment_transitions=count,
            trajectory_steps=torch.ones(count, 1, dtype=torch.int64),
            rollout_budget_exhausted=torch.zeros(count, 1, dtype=torch.bool),
        )

    args = SimpleNamespace(
        training_epochs=10,
        data_passes=None,
        max_batches_per_pass=None,
        pilot_mode=False,
        validation_every_epochs=2,
        minimum_training_epochs=4,
        post_minimum_validation_every_epochs=2,
        validation_checkpoints=5,
        early_stop_patience_validations=2,
        early_stop_start_epoch=4,
        physical_batch_size=1,
        effective_batch_size=2,
        training_stream_path=tmp_path / "stream.parquet",
        customer_exposure_budget=1_000,
        scale="Cus50",
        output_dir=tmp_path / "early-stop",
        protocol_id="early-stop-test",
        resume=False,
        validation_limit=2,
        final_validation_limit=0,
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
    terminal = json.loads((args.output_dir / "training_result.json").read_text())
    assert terminal["status"] == "early_stopped"
    assert terminal["requested_training_epochs"] == 10
    assert terminal["completed_training_epochs"] == 8
    assert terminal["completed_validation_checkpoints"] == 4
    assert terminal["early_stop_start_epoch"] == 4
    assert len(validation_calls) == 4
    assert (args.output_dir / "best_within_5000.ckpt").is_file()
    assert (args.output_dir / "best_overall.ckpt").is_file()


def test_completed_fixed_budget_can_resume_a_prefix_stable_extension(
    tmp_path, monkeypatch
) -> None:
    calls = []

    def fake_validation(instances, _solve, *, seed):
        count = len(list(instances))
        calls.append(seed)
        return {
            "schema": "drl_validation_summary_v1",
            "instances": count,
            "complete_and_feasible": count,
            "complete_and_feasible_rate": 1.0,
            "mean_verified_distance_km": 100.0 - len(calls),
            "verifier_summary_passed": True,
            "rows": [],
        }

    monkeypatch.setattr(
        protocol_trainers,
        "make_validation_pool",
        lambda *_args, **_kwargs: _ValidationPool(),
    )
    monkeypatch.setattr(protocol_trainers, "verified_validation", fake_validation)
    policy = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)

    def result(instances):
        count = len(instances)
        return SimpleNamespace(
            cost=torch.ones(count, 1),
            objective=torch.ones(count, 1),
            feasible=torch.ones(count, 1, dtype=torch.bool),
            log_likelihood=policy.weight.sum().expand(count, 1),
            environment_transitions=count,
            trajectory_steps=torch.ones(count, 1, dtype=torch.int64),
            rollout_budget_exhausted=torch.zeros(count, 1, dtype=torch.bool),
        )

    args = SimpleNamespace(
        training_epochs=2,
        data_passes=None,
        max_batches_per_pass=None,
        pilot_mode=False,
        validation_every_epochs=1,
        validation_checkpoints=2,
        early_stop_patience_validations=0,
        early_stop_start_epoch=0,
        physical_batch_size=1,
        effective_batch_size=2,
        training_stream_path=tmp_path / "stream.parquet",
        customer_exposure_budget=200,
        scale="Cus50",
        output_dir=tmp_path / "extension",
        protocol_id="fixed-extension-test",
        resume=False,
        validation_limit=2,
        final_validation_limit=0,
        validation_every_passes=5,
        seed=1234,
        baseline_eval_size=0,
        exposure_checkpoints="",
        gpu_hour_checkpoints="",
        device="cpu",
        max_grad_norm=1.0,
        training_rollout_steps=80,
    )

    def run():
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

    run()
    args.training_epochs = 4
    args.validation_checkpoints = 4
    args.customer_exposure_budget = 400
    args.resume = True
    run()

    history = [
        json.loads(line)
        for line in (args.output_dir / "validation_history.jsonl").read_text().splitlines()
    ]
    assert [row["logical_epoch"] for row in history] == [1, 2, 3, 4]
    state = json.loads((args.output_dir / "data_pass_state.json").read_text())
    assert state["completed_data_passes"] == 1
    assert state["instances_seen"] == 8
    assert state["optimizer_steps"] == 4


def test_two_phase_validation_schedule_has_exact_boundary_and_tail():
    epochs = training_protocol.validation_epochs(
        10_000,
        initial_interval=250,
        minimum_epochs=5_000,
        post_minimum_interval=50,
    )
    assert len(epochs) == 120
    assert epochs[:3] == (250, 500, 750)
    assert epochs[19:23] == (5_000, 5_050, 5_100, 5_150)
    assert epochs[-1] == 10_000


def test_verified_validation_disables_autograd(monkeypatch) -> None:
    instance = SimpleNamespace(instance_id="validation-instance")
    grad_states = []

    def solve(_instance, _seed):
        grad_states.append(torch.is_grad_enabled())
        return {"success": [True]}

    monkeypatch.setattr(
        training_protocol,
        "select_min_verified_distance",
        lambda _instance, _info: (
            0,
            [[0, 1, 0]],
            {"passed": True, "objective_distance_km": 7.5},
        ),
    )
    summary = training_protocol.verified_validation([instance], solve, seed=1234)

    assert grad_states == [False]
    assert summary["complete_and_feasible"] == 1
    assert summary["mean_verified_distance_km"] == 7.5
