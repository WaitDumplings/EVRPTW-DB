from __future__ import annotations

import hashlib
import importlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from EVRPTW_Benchmark.Reinforcement_Learning.common import protocol_trainers
from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import ObjectiveConfig
from EVRPTW_Benchmark.Reinforcement_Learning.common.reward_contract import reward_contract_digest
from EVRPTW_Benchmark.Reinforcement_Learning.common.tests.test_reinforce_training_diagnostics import (
    _TinyPool,
    _args,
)

METHODS = ("AM-EVRPTW", "EVRPTW-RL", "DRL-TS", "RRNCO-EV")


def _source(path, method="AM-EVRPTW", **saved_args):
    policy = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        policy.weight.fill_(3)
    baseline = deepcopy(policy)
    with torch.no_grad():
        baseline.weight.fill_(9)
    objective = ObjectiveConfig(mode="energy_vehicle_cost", profile_id="source-cost")
    payload = {
        "method": method,
        "model": policy.state_dict(),
        "baseline": baseline.state_dict(),
        "optimizer": {"sentinel": "must never be restored"},
        "objective_config": objective.to_dict(),
        "protocol_id": "diagnostic-test",
        "logical_epoch": 4300,
        "data_pass": 17,
        "best_validation_key": [1.0, 0.0],
        "best_within_minimum_key": [1.0, 0.0],
        "completed_validation_checks": 43,
        "validation_checks_without_improvement": 99,
        "baseline_eval_count": 70,
        "baseline_update_count": 50,
        "ema_cost": 12345,
        "soft_stage_contract": {
            "resolved_soft_stage_end_epoch": 2500,
        } if method == "DRL-TS" else None,
        "args": {"scale": "Cus50", "training_representation": "G", "seed": 1234, **saved_args},
    }
    torch.save(payload, path)
    return payload


def _target_args(output):
    args = _args(output, fixed=True, cost=True)
    args.objective["objective_distance_source"] = "running_time_path_distance_km"
    contract = args.reward_contract_snapshot
    contract["objective"] = deepcopy(args.objective)
    contract["sha256"] = reward_contract_digest(contract)
    args.warm_start_objective_transition = True
    args.training_representation = "G"
    args.soft_stage_end_epoch = 0
    args.soft_stage_contract_snapshot = None
    return args


def _load(path, args, *, method="AM-EVRPTW", policy=None):
    policy = policy if policy is not None else torch.nn.Linear(1, 1, bias=False)
    baseline = deepcopy(policy)
    provenance = protocol_trainers._load_warm_start_checkpoint(
        path, method=method, policy=policy, baseline=baseline,
        objective_config=args.objective, contract_args=args,
    )
    return policy, baseline, provenance


@pytest.mark.parametrize("module", ("AM_EVRPTW", "EVRPTW_RL", "DRL_TS", "RRNCO_EVRPTW"))
def test_all_reinforce_clis_expose_explicit_curriculum_transitions(module, monkeypatch):
    entry = importlib.import_module(f"EVRPTW_Benchmark.Reinforcement_Learning.{module}.train")
    monkeypatch.setattr(sys, "argv", [
        "train", "--dataset-path", "data", "--output-dir", "out",
        "--warm-start-checkpoint", "source.ckpt", "--warm-start-objective-transition",
        "--warm-start-scale-transition",
    ])
    args = entry.parse_args()
    assert args.warm_start_checkpoint == Path("source.ckpt")
    assert args.warm_start_objective_transition is True
    assert args.warm_start_scale_transition is True


def test_transition_is_explicit_and_cannot_weaken_resume(tmp_path):
    checkpoint = tmp_path / "source.ckpt"
    payload = _source(checkpoint)
    args = _target_args(tmp_path / "target")
    args.warm_start_objective_transition = False
    with pytest.raises(ValueError, match="checkpoint objective mismatch"):
        _load(checkpoint, args)
    args.warm_start_objective_transition = True
    objective_before = deepcopy(args.objective)
    contract_before = deepcopy(args.reward_contract_snapshot)
    policy, baseline, provenance = _load(checkpoint, args)
    assert policy.weight.item() == baseline.weight.item() == 3
    assert args.objective == objective_before
    assert args.reward_contract_snapshot == contract_before
    assert provenance["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert provenance["source_objective_config"] == payload["objective_config"]
    assert provenance["source_logical_epoch"] == 4300
    assert provenance["source_data_pass"] == 17
    assert provenance["objective_transition"] is True
    optimizer = torch.optim.AdamW(policy.parameters())
    with pytest.raises(ValueError, match="checkpoint objective mismatch"):
        protocol_trainers._load_checkpoint(
            checkpoint, policy=policy, baseline=baseline, optimizer=optimizer,
            protocol_id=args.protocol_id, objective_config=args.objective,
            reward_contract_args=args,
        )
    assert not optimizer.state


def test_transition_requires_checkpoint_and_is_mutually_exclusive_with_resume(tmp_path):
    args = _target_args(tmp_path / "target")
    with pytest.raises(ValueError, match="requires --warm-start-checkpoint"):
        protocol_trainers.prepare_training_objective(args)
    args.warm_start_checkpoint = tmp_path / "source.ckpt"
    _source(args.warm_start_checkpoint)
    args.resume = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        protocol_trainers.prepare_training_objective(args)


def test_scale_transition_requires_separate_opt_in_and_records_both_scales(tmp_path):
    checkpoint = tmp_path / "source.ckpt"
    _source(checkpoint, scale="Cus100")
    args = _target_args(tmp_path / "target")
    args.scale = "Cus500"
    with pytest.raises(ValueError, match="warm-start scale mismatch"):
        _load(checkpoint, args)
    args.warm_start_scale_transition = True
    policy, baseline, provenance = _load(checkpoint, args)
    assert policy.weight.item() == baseline.weight.item() == 3
    assert provenance["scale_transition"] is True
    assert provenance["source_scale"] == "Cus100"
    assert provenance["target_scale"] == "Cus500"
    assert provenance["source_logical_epoch"] == 4300
    assert provenance["target_objective_config"] == args.objective
    assert provenance["optimizer_reset"] and provenance["epoch_reset"]
    assert provenance["sample_counters_reset"] and provenance["baseline_history_reset"]
    assert provenance["validation_state_reset"] and provenance["early_stop_state_reset"]
    # Scale opt-in must never waive the independently protected objective contract.
    args.warm_start_objective_transition = False
    with pytest.raises(ValueError, match="checkpoint objective mismatch"):
        _load(checkpoint, args)


def test_scale_transition_requires_checkpoint_and_cannot_be_resume(tmp_path):
    args = _target_args(tmp_path / "target")
    args.warm_start_objective_transition = False
    args.warm_start_scale_transition = True
    with pytest.raises(ValueError, match="--warm-start-scale-transition requires"):
        protocol_trainers.prepare_training_objective(args)
    args.warm_start_checkpoint = tmp_path / "source.ckpt"
    _source(args.warm_start_checkpoint)
    args.resume = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        protocol_trainers.prepare_training_objective(args)


@pytest.mark.parametrize(("field", "value", "error"), [
    ("training_representation", "E", "training_representation mismatch"),
    ("seed", 5678, "seed mismatch"),
    ("tanh_clipping", 5.0, "architecture tanh_clipping mismatch"),
    ("scale", None, "recorded source and target scales"),
    ("scale", "Cus0", "invalid scale"),
])
def test_scale_transition_keeps_non_scale_contract_checks(tmp_path, field, value, error):
    checkpoint = tmp_path / "source.ckpt"
    _source(checkpoint, scale="Cus100", tanh_clipping=10.0)
    args = _target_args(tmp_path / "target")
    args.scale = "Cus500"
    args.warm_start_scale_transition = True
    setattr(args, field, value)
    with pytest.raises(ValueError, match=error):
        _load(checkpoint, args)


def test_scale_transition_rejects_unrecorded_source_scale_and_wrong_method(tmp_path):
    checkpoint = tmp_path / "source.ckpt"
    args = _target_args(tmp_path / "target")
    args.scale = "Cus500"
    args.warm_start_scale_transition = True
    _source(checkpoint, scale=None)
    with pytest.raises(ValueError, match="recorded source and target scales"):
        _load(checkpoint, args)
    _source(checkpoint, method="EVRPTW-RL", scale="Cus100")
    with pytest.raises(ValueError, match="warm-start method mismatch"):
        _load(checkpoint, args)


@pytest.mark.parametrize(("method", "field", "saved", "requested"), [
    ("AM-EVRPTW", "tanh_clipping", 10.0, 5.0),
    ("EVRPTW-RL", "graph_aggregation", "mean", "sum"),
    ("DRL-TS", "n_heads", 8, 4),
    ("RRNCO-EV", "aft_mode", "stable", "legacy"),
])
def test_transition_rejects_architecture_changes_even_with_matching_tensor_shapes(
    tmp_path, method, field, saved, requested,
):
    checkpoint = tmp_path / "source.ckpt"
    _source(checkpoint, method, **{field: saved})
    args = _target_args(tmp_path / "target")
    setattr(args, field, requested)
    with pytest.raises(ValueError, match=f"architecture {field} mismatch"):
        _load(checkpoint, args, method=method)
    setattr(args, field, saved)
    _load(checkpoint, args, method=method)
    with pytest.raises(RuntimeError, match="size mismatch"):
        _load(checkpoint, args, method=method, policy=torch.nn.Linear(2, 1, bias=False))


def test_drl_transition_requires_deliberate_stage_boundary(tmp_path):
    checkpoint = tmp_path / "source.ckpt"
    _source(checkpoint, "DRL-TS")
    args = _target_args(tmp_path / "target")
    args.soft_stage_end_epoch = None
    with pytest.raises(ValueError, match="explicit --soft-stage-end-epoch"):
        _load(checkpoint, args, method="DRL-TS")
    args.soft_stage_end_epoch = 0
    _, _, provenance = _load(checkpoint, args, method="DRL-TS")
    assert provenance["source_training_stage"] == "hard"
    assert provenance["source_soft_stage_contract"]["resolved_soft_stage_end_epoch"] == 2500


@pytest.mark.parametrize(("method", "scale"), [
    *[(method, "Cus50") for method in METHODS], ("AM-EVRPTW", "Cus500"),
])
def test_new_stage_runs_additional_epochs_with_fresh_state_and_target_objective(
    tmp_path, monkeypatch, method, scale,
):
    args = _target_args(tmp_path / "target")
    args.warm_start_checkpoint = tmp_path / "source.ckpt"
    _source(args.warm_start_checkpoint, method, scale="Cus100" if scale == "Cus500" else "Cus50")
    if scale == "Cus500":
        args.scale = scale
        args.warm_start_scale_transition = True
        args.reward_contract_snapshot["scales"][scale] = args.reward_contract_snapshot["scales"].pop("Cus50")
        args.reward_contract_snapshot["sha256"] = reward_contract_digest(args.reward_contract_snapshot)
        args.customer_exposure_budget = args.training_epochs * args.effective_batch_size * 500
    args.validation_limit = 3
    args.validation_every_epochs = 1
    args.validation_checkpoints = 2
    args.minimum_training_epochs = 1
    args.early_stop_patience_validations = 1
    args.early_stop_start_epoch = 1
    policy = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)
    calls = []
    validation_calls = []
    target_objective = ObjectiveConfig(**args.objective)

    def result(instances, soft, *, actor):
        if not calls:
            assert policy.weight.item() == 3
            assert not optimizer.state
        calls.append((actor, soft))
        count = len(instances)
        return SimpleNamespace(
            cost=torch.full((count, 1), 2.0 if actor else 1.0),
            objective=torch.ones(count, 1),
            objective_value=torch.full((count, 1), target_objective.value(1.0, 1)),
            vehicles_started=torch.ones(count, 1),
            feasible=torch.ones(count, 1, dtype=torch.bool),
            log_likelihood=policy.weight.sum().expand(count, 1),
            environment_transitions=count,
            trajectory_steps=torch.ones(count, 1, dtype=torch.int64),
            rollout_budget_exhausted=torch.zeros(count, 1, dtype=torch.bool),
        )

    def validation(instances, _solve, *, seed, objective_config):
        assert objective_config.to_dict() == args.objective
        validation_calls.append(seed)
        return {
            "schema": "drl_validation_summary_v1", "instances": 3,
            "complete_and_feasible": 3, "complete_and_feasible_rate": 1.0,
            "mean_verified_distance_km": 1.0,
            "mean_verified_objective": 10.0 - len(validation_calls),
            "objective_mode": "energy_vehicle_cost",
            "verifier_summary_passed": True, "rows": [],
        }

    monkeypatch.setattr(protocol_trainers, "make_validation_pool", lambda *_a, **_k: _TinyPool())
    monkeypatch.setattr(protocol_trainers, "verified_validation", validation)
    training_kwargs = dict(
        method=method, args=args, pool=_TinyPool(), policy=policy, optimizer=optimizer,
        make_actor=lambda instances, soft, seed: result(instances, soft, actor=True),
        make_baseline=lambda active, instances, soft, seed: result(instances, soft, actor=False),
        training_cost=lambda value: value.cost,
        objective_distance=lambda value: value.objective,
        feasible=lambda value: value.feasible,
        validation_solve=lambda *_a: {}, legacy_batch_size=1,
        soft_stage_end_epoch=0 if method == "DRL-TS" else None,
    )
    protocol_trainers.train_reinforce_data_passes(**training_kwargs)
    payload = torch.load(args.output_dir / "best.ckpt", map_location="cpu", weights_only=False)
    assert payload["logical_epoch"] == 2
    assert payload["completed_validation_checks"] == 2
    assert payload["validation_checks_without_improvement"] == 0
    assert payload["best_validation_key"] == [1.0, -8.0]
    assert payload["baseline_eval_count"] == payload["baseline_update_count"] == 0
    assert payload["ema_cost"] is None
    assert payload["objective_config"] == args.objective
    assert payload["reward_contract"] == args.reward_contract_snapshot
    assert payload["warm_start_provenance"]["source_logical_epoch"] == 4300
    assert len(validation_calls) == 2
    assert all(not soft for _, soft in calls)
    state = json.loads((args.output_dir / "data_pass_state.json").read_text())
    assert state["optimizer_steps"] == 2
    assert state["instances_seen"] == 4
    assert state["customer_exposures"] == 4 * int(scale.removeprefix("Cus"))
    assert payload["warm_start_provenance"]["target_scale"] == scale
    rows = [json.loads(line) for line in (args.output_dir / "reward_diagnostics.jsonl").read_text().splitlines()]
    assert [row["logical_epoch"] for row in rows] == [1, 2]
    assert all(row["session_start_logical_epoch"] == 0 for row in rows)
    assert all(row["session_start_optimizer_steps"] == 0 for row in rows)
    history = json.loads((args.output_dir / "train_history.jsonl").read_text().splitlines()[-1])
    assert history["training_stage"] == "hard"

    # A later same-objective resume keeps the original source identity without
    # reinitializing weights or needing the warm-start source file again.
    args.warm_start_checkpoint = None
    args.warm_start_objective_transition = False
    args.warm_start_scale_transition = False
    args.resume = True
    del args.warm_start_provenance
    protocol_trainers.train_reinforce_data_passes(**training_kwargs)
    resumed = json.loads((args.output_dir / "training_result.json").read_text())
    assert resumed["warm_start_provenance"] == payload["warm_start_provenance"]
    assert resumed["optimizer_steps"] == 2
    assert len(validation_calls) == 2
