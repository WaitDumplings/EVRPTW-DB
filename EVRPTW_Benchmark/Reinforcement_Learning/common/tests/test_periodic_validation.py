from __future__ import annotations

import json
from types import SimpleNamespace

import torch
import pytest

from EVRPTW_Benchmark.Reinforcement_Learning.common import protocol_trainers, training_protocol
from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import ObjectiveConfig
from EVRPTW_Benchmark.Reinforcement_Learning.common.reward_contract import (
    reward_contract_digest,
)


class _Pool:
    def __len__(self) -> int:
        return 100

    def first(self, *, limit: int):
        return [object() for _ in range(min(limit, 1))]

    def stream_batches(
        self, _path, physical: int, *, start: int, stop: int,
        logical_batch_size: int | None = None,
        training_stream_contract_sha256: str | None = None,
    ):
        for offset in range(start, stop, physical):
            yield [object() for _ in range(min(physical, stop - offset))]


class _ValidationPool:
    def first(self, *, limit: int):
        return [object() for _ in range(min(limit, 3))]


@pytest.mark.parametrize("cost_objective", [False, True])
@pytest.mark.parametrize("preverified_stream", [False, True])
def test_fixed_epoch_validation_selects_best_and_records_every_interval(
    tmp_path, monkeypatch, cost_objective, preverified_stream
) -> None:
    validation_calls = []
    objective = ObjectiveConfig(
        mode="energy_vehicle_cost" if cost_objective else "distance",
        profile_id="cost-test" if cost_objective else "distance_v1",
    )
    reward_contract_path = None
    if cost_objective:
        reward_contract = {
            "schema": "drl_reward_contract_v1",
            "contract_id": "periodic-validation-test",
            "objective": objective.to_dict(),
            "scales": {
                "Cus50": {
                    "objective_scale": 1.0,
                    "failure_base": 2.0,
                    "unserved_coefficient": 1.0,
                }
            },
        }
        reward_contract["sha256"] = reward_contract_digest(reward_contract)
        reward_contract_path = tmp_path / "reward_contract.json"
        reward_contract_path.write_text(json.dumps(reward_contract), encoding="utf-8")

    def fake_validation(instances, _solve, *, seed, objective_config=None):
        assert objective_config.to_dict() == objective.to_dict()
        validation_calls.append((len(list(instances)), seed))
        score = 10.0 - len(validation_calls)
        count = validation_calls[-1][0]
        return {
            "schema": "drl_validation_summary_v1",
            "instances": count,
            "complete_and_feasible": count,
            "complete_and_feasible_rate": 1.0,
            # Deliberately worsen distance while improving cost to prove that
            # cost-mode checkpoint selection follows the active objective.
            "mean_verified_distance_km": 20.0 - score if cost_objective else score,
            "mean_verified_objective": score,
            "objective_mode": objective.mode,
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
            objective_value=torch.full((count, 1), float(objective.value(1.0, 1))),
            vehicles_started=torch.ones(count, 1),
            feasible=torch.ones(count, 1, dtype=torch.bool),
            log_likelihood=log_likelihood,
            environment_transitions=count,
            trajectory_steps=torch.ones(count, 1, dtype=torch.int64),
            rollout_budget_exhausted=torch.zeros(count, 1, dtype=torch.bool),
        )

    args = SimpleNamespace(
        objective=objective.to_dict(),
        reward_contract=reward_contract_path,
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
    stream_job = None
    if preverified_stream:
        from EVRPTW_Benchmark.Reinforcement_Learning.common.training_stream import (
            STREAM_INTEGRITY_MODE_PREVERIFIED,
            training_stream_contract_digest,
        )

        snapshot = {
            "schema": "drl_training_stream_contract_v1",
            "sample_count": 12,
            "scale": "Cus50",
            "seed": args.seed,
        }
        snapshot["sha256"] = training_stream_contract_digest(snapshot)
        args.protocol_id = "drl_rq_protocol_frozen_v1"
        args.reuse_preverified_training_streams = True
        args.training_stream_contract_sha256 = snapshot["sha256"]
        args.training_stream_contract_snapshot_json = json.dumps(snapshot)
        stream_job = {
            "method": "drl_ts",
            "training_stream_path": str(args.training_stream_path),
            "training_stream_contract_snapshot": snapshot,
            "training_stream_contract_sha256": snapshot["sha256"],
            "stream_integrity_mode": STREAM_INTEGRITY_MODE_PREVERIFIED,
            "file_hash_validation_performed": False,
        }
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
    assert selected_payload["objective_config"] == objective.to_dict()
    final_audit = json.loads(
        (output / "validation_final_audit.json").read_text()
    )
    assert final_audit["schema"] == "drl_final_validation_audit_v1"
    assert final_audit["instances"] == 3
    assert final_audit["selection_logical_epoch"] == 6
    assert final_audit["selection_changed"] is False
    if stream_job is not None:
        from EVRPTW_Benchmark.Reinforcement_Learning.scripts.drl_job_runtime import (
            validate_completed_training_stream_contract,
        )

        terminal = json.loads((output / "training_result.json").read_text())
        # Exercise the actual launcher gate with artifacts written by training.
        # A checkpoint-only assertion missed the original terminal JSON omission.
        validate_completed_training_stream_contract(
            stream_job, {"repo": tmp_path}, terminal,
            output / "checkpoint_selected.pt", reuse_preverified=True,
        )


def test_fixed_epoch_early_stop_waits_until_after_start_epoch(tmp_path, monkeypatch) -> None:
    validation_calls = []

    def flat_validation(instances, _solve, *, seed, objective_config=None):
        assert objective_config.mode == "distance"
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

    def fake_validation(instances, _solve, *, seed, objective_config=None):
        assert objective_config.mode == "distance"
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
    diagnostics = [
        json.loads(line)
        for line in (args.output_dir / "reward_diagnostics.jsonl").read_text().splitlines()
    ]
    assert [row["logical_epoch"] for row in diagnostics] == [1, 2, 3, 4]
    assert [row["optimizer_steps_total"] for row in diagnostics] == [1, 2, 3, 4]
    assert diagnostics[0]["session_id"] == diagnostics[1]["session_id"]
    assert diagnostics[2]["session_id"] == diagnostics[3]["session_id"]
    assert diagnostics[0]["session_id"] != diagnostics[2]["session_id"]
    assert diagnostics[0]["resume_requested"] is False
    assert diagnostics[2]["resume_requested"] is True
    assert diagnostics[2]["session_start_optimizer_steps"] == 2
    assert diagnostics[2]["session_start_logical_epoch"] == 2
    assert diagnostics[2]["resume_checkpoint"] == str(args.output_dir / "checkpoint_latest.pt")


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
        "select_min_verified_objective",
        lambda _instance, _info, _objective: (
            0,
            [[0, 1, 0]],
            {"passed": True, "objective_distance_km": 7.5, "vehicles_started": 1},
        ),
    )
    summary = training_protocol.verified_validation([instance], solve, seed=1234)

    assert grad_states == [False]
    assert summary["complete_and_feasible"] == 1
    assert summary["mean_verified_distance_km"] == 7.5


def test_verified_validation_scopes_sampling_rng_and_restores_cpu_state(
    monkeypatch,
) -> None:
    instances = [
        SimpleNamespace(instance_id="validation-instance-0"),
        SimpleNamespace(instance_id="validation-instance-1"),
    ]
    monkeypatch.setattr(
        training_protocol,
        "select_min_verified_objective",
        lambda _instance, _info, _objective: (
            0,
            [[0, 1, 0]],
            {"passed": True, "objective_distance_km": 7.5, "vehicles_started": 1},
        ),
    )

    def sampled_draws(caller_seed: int):
        observed: list[tuple[int, torch.Tensor]] = []

        def solve(_instance, registered_seed):
            observed.append((registered_seed, torch.rand(4)))
            return {"success": [True]}

        torch.manual_seed(caller_seed)
        caller_state = torch.random.get_rng_state().clone()
        training_protocol.verified_validation(instances, solve, seed=12_345)
        assert torch.equal(torch.random.get_rng_state(), caller_state)
        return observed

    first = sampled_draws(111)
    second = sampled_draws(999)

    assert [seed for seed, _ in first] == [12_345, 12_346]
    assert [seed for seed, _ in second] == [12_345, 12_346]
    assert all(
        torch.equal(first_draw, second_draw)
        for (_, first_draw), (_, second_draw) in zip(first, second, strict=True)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_verified_validation_scopes_sampling_rng_and_restores_cuda_state(
    monkeypatch,
) -> None:
    instance = SimpleNamespace(instance_id="validation-instance")
    monkeypatch.setattr(
        training_protocol,
        "select_min_verified_objective",
        lambda _instance, _info, _objective: (
            0,
            [[0, 1, 0]],
            {"passed": True, "objective_distance_km": 7.5, "vehicles_started": 1},
        ),
    )

    def sampled_draw(caller_seed: int) -> torch.Tensor:
        observed: list[torch.Tensor] = []

        def solve(_instance, _registered_seed):
            observed.append(torch.rand(4, device="cuda:0").cpu())
            return {"success": [True]}

        torch.manual_seed(caller_seed)
        caller_cpu_state = torch.random.get_rng_state().clone()
        caller_cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
        training_protocol.verified_validation([instance], solve, seed=54_321)
        assert torch.equal(torch.random.get_rng_state(), caller_cpu_state)
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(
                torch.cuda.get_rng_state_all(), caller_cuda_states, strict=True
            )
        )
        return observed[0]

    assert torch.equal(sampled_draw(222), sampled_draw(888))
