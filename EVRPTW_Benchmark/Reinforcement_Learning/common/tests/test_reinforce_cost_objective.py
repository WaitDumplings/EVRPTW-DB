from __future__ import annotations

import importlib
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

from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import _instance
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.env import DRLTSHardConstraintEnv
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.soft_env import DRLTSSoftConstraintEnv
from EVRPTW_Benchmark.Reinforcement_Learning.common import protocol_entrypoints, protocol_trainers
from EVRPTW_Benchmark.Reinforcement_Learning.common.action_constraints import ACTION_CONSTRAINT_CONTRACT_ID
from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import ObjectiveConfig
from EVRPTW_Benchmark.Reinforcement_Learning.common.stage2_data import make_envs


def _objective(cost: bool = True) -> ObjectiveConfig:
    return ObjectiveConfig(
        mode="energy_vehicle_cost" if cost else "distance",
        profile_id="rivian_energy_vehicle_cost_v1" if cost else "distance_v1",
    )


class _ScriptedPolicy(torch.nn.Module):
    """Small differentiable policy keeps objective tests independent of models."""

    def __init__(self, method, actions):
        super().__init__()
        self.method, self.actions = method, actions
        self.device = torch.device("cpu")
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.index = 0

    def encode(self, *_args, **_kwargs):
        return None

    encode_static = encode
    initial_state = encode

    def logits(self, obs, *_args, **_kwargs):
        mask = torch.as_tensor(obs["action_mask"], dtype=torch.bool)
        preferred = torch.zeros(mask.shape[-1])
        preferred[self.actions[self.index]] = 1.0
        self.index += 1
        logits = ((4.0 + self.weight) * preferred).expand(mask.shape)
        logits = logits.masked_fill(~mask, -torch.inf)
        return logits if self.method == "AM_EVRPTW" else (logits, None)


@pytest.mark.parametrize("cost", [False, True])
@pytest.mark.parametrize("method,soft", [
    ("AM_EVRPTW", False), ("EVRPTW_RL", False),
    ("DRL_TS", False), ("DRL_TS", True),
])
@pytest.mark.parametrize("actions", [(1, 2, 0), (1, 0, 2, 0), (3, 1)])
def test_reinforce_rollouts_use_active_base_and_preserve_auxiliary_penalties(
    cost, method, soft, actions
):
    objective = _objective(cost)
    if method == "DRL_TS":
        env_cls = DRLTSSoftConstraintEnv if soft else DRLTSHardConstraintEnv
        env = env_cls(
            _instance(), n_traj=2, info_level="full", use_jit_mask=False,
            objective_config=objective, reward_distance_scale_km=5.0,
        )
    else:
        env = make_envs(
            [_instance()], n_traj=2, info_level="full", use_jit_mask=False,
            objective_config=objective, reward_distance_scale_km=5.0,
        )[0]
    # DRL-TS deliberately masks a station visit at full battery; retain that
    # published mask by placing its partial-route station visit after service.
    policy_actions = (1, 3) if method == "DRL_TS" and actions == (3, 1) else actions
    policy = _ScriptedPolicy(method, policy_actions)
    module = importlib.import_module(
        f"EVRPTW_Benchmark.Reinforcement_Learning.{method}.rollout"
    )
    kwargs = (
        {"incomplete_penalty_km": 123.0} if method == "AM_EVRPTW" else
        {"soft_constraints": soft, "capacity_penalty": 3.0,
         "time_penalty": 5.0, "energy_penalty": 7.0} if method == "DRL_TS" else {}
    )
    result = module.rollout(
        policy, [env], decode_type="greedy", max_steps=len(actions), seed=1, **kwargs
    )
    info = result.infos[0]
    distance = np.asarray(info["objective_distance_km"])
    vehicles = np.asarray(info["vehicles_started"])
    active_value = objective.value(distance, vehicles)
    incomplete = 1.0 - np.asarray(info["served_customers"]) / 2.0
    success = np.asarray(info["success"])
    expected = active_value / env.reward_objective_scale
    if method == "AM_EVRPTW":
        expected += (~success) * 123.0 * (1.0 + incomplete) * objective.distance_unit_cost / env.reward_objective_scale
        np.testing.assert_allclose(
            result.cost_km.detach().numpy()[0],
            distance + (~success) * 123.0 * (1.0 + incomplete),
        )
    elif method == "EVRPTW_RL":
        expected += 0.3 * result.station_visits.numpy()[0]
        expected += (~success) * 100.0 * (1.0 + incomplete)
    else:
        expected += (
            3.0 * result.capacity_violation.numpy()[0]
            + 5.0 * result.time_violation.numpy()[0]
            + 7.0 * result.energy_violation.numpy()[0]
            + (~success) * 100.0 * (1.0 + incomplete)
        )
    np.testing.assert_allclose(result.objective_distance_km.numpy()[0], distance)
    np.testing.assert_allclose(result.objective_value.numpy()[0], active_value, rtol=1e-6)
    np.testing.assert_allclose(result.vehicles_started.numpy()[0], vehicles)
    np.testing.assert_allclose(result.training_cost.numpy()[0], expected, rtol=1e-6)
    expected_vehicles = 2 if actions == (1, 0, 2, 0) else 1
    assert vehicles.tolist() == [expected_vehicles, expected_vehicles]
    (result.training_cost.detach() * result.log_likelihood).mean().backward()
    assert torch.isfinite(policy.weight.grad) and policy.weight.grad.abs() > 0


@pytest.mark.parametrize("runner,method", [
    (protocol_entrypoints.run_am, "AM_EVRPTW"),
    (protocol_entrypoints.run_evrptw_rl, "EVRPTW_RL"),
    (protocol_entrypoints.run_drl_ts, "DRL_TS"),
])
def test_formal_actor_baseline_and_validation_share_frozen_objective(
    monkeypatch, runner, method
):
    objective = _objective()
    seen = []
    module = importlib.import_module(
        f"EVRPTW_Benchmark.Reinforcement_Learning.{method}.rollout"
    )

    def fake_rollout(_policy, envs, **kwargs):
        infos = []
        for env in envs:
            assert env.objective_config.to_dict() == objective.to_dict()
            assert env.reward_objective_scale == pytest.approx(objective.value(5.0, 1))
            env.reset(seed=kwargs["seed"])
            for action in (1, 2, 0):
                _, _, _, _, info = env.step(np.full(env.n_traj, action))
            infos.append(info)
        seen.append(kwargs)
        return SimpleNamespace(infos=infos, runtime_s=0.0, training_cost=torch.tensor([[7.0]]))

    def inspect_callbacks(**callbacks):
        for soft in (False, True):
            actor = callbacks["make_actor"]([_instance()], soft, 1)
            callbacks["make_baseline"](object(), [_instance()], soft, 1)
            assert callbacks["training_cost"](actor).item() == 7.0
        info = callbacks["validation_solve"](object(), _instance(), 1)
        assert "routes" in info and info["objective_config"]["mode"] == objective.mode

    monkeypatch.setattr(module, "rollout", fake_rollout)
    monkeypatch.setattr(protocol_entrypoints, "train_reinforce_data_passes", inspect_callbacks)
    args = SimpleNamespace(
        objective=objective.to_dict(), training_rollout_steps=80,
        validation_decode_type="sampling", validation_candidates=2,
        samples_per_instance=1, incomplete_penalty_km=123.0,
        incomplete_penalty=100.0, station_visit_penalty=0.3,
        capacity_penalty=3.0, time_penalty=5.0, energy_penalty=7.0,
        batch_size=1, soft_stage_fraction=0.5,
    )
    pool = SimpleNamespace(reward_distance_scale_km=lambda _mode: 5.0, reward_scale_metadata={})
    runner(args, pool, object(), object())
    assert len(seen) == 5


def test_reinforce_checkpoint_freezes_objective_and_rejects_changed_resume(tmp_path):
    objective = _objective()
    policy = torch.nn.Linear(1, 1, bias=False)
    baseline = deepcopy(policy)
    optimizer = torch.optim.Adam(policy.parameters())
    args = SimpleNamespace(protocol_id="cost-test", objective=objective.to_dict())
    path = tmp_path / "checkpoint.pt"
    protocol_trainers._save_checkpoint(
        path, method="AM-EVRPTW", data_pass=0, policy=policy,
        baseline=baseline, optimizer=optimizer, args=args,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["objective_config"] == objective.to_dict()
    assert payload["action_constraint_contract_id"] == ACTION_CONSTRAINT_CONTRACT_ID
    assert payload["args"]["action_constraint_contract_id"] == ACTION_CONSTRAINT_CONTRACT_ID
    before = policy.weight.detach().clone()
    with pytest.raises(ValueError, match="objective mismatch"):
        protocol_trainers._load_checkpoint(
            path, policy=policy, baseline=baseline, optimizer=optimizer,
            protocol_id="cost-test", objective_config=_objective(False),
        )
    torch.testing.assert_close(policy.weight, before)
    protocol_trainers._load_checkpoint(
        path, policy=policy, baseline=baseline, optimizer=optimizer,
        protocol_id="cost-test", objective_config=objective,
    )


def test_soft_cost_rollout_retains_nonzero_capacity_penalty():
    from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.rollout import rollout

    instance = _instance()
    instance.vehicle["cargo_capacity_cm3"] = 1.5
    env = DRLTSSoftConstraintEnv(
        instance, n_traj=1, info_level="full", use_jit_mask=False,
        objective_config=_objective(), reward_distance_scale_km=5.0,
    )
    result = rollout(
        _ScriptedPolicy("DRL_TS", (1, 2, 0)), [env], decode_type="greedy",
        max_steps=3, seed=1, soft_constraints=True, capacity_penalty=3.0,
    )
    assert result.capacity_violation.item() > 0.0
    expected = (
        result.objective_value / env.reward_objective_scale
        + 3.0 * result.capacity_violation + result.time_violation + result.energy_violation
    )
    torch.testing.assert_close(result.training_cost, expected)


@pytest.mark.parametrize("method,soft", [
    ("AM_EVRPTW", False), ("EVRPTW_RL", False),
    ("DRL_TS", False), ("DRL_TS", True),
])
def test_real_policy_backpropagates_finite_cost_rollout(method, soft):
    torch.manual_seed(103)
    module = importlib.import_module(
        f"EVRPTW_Benchmark.Reinforcement_Learning.{method}.rollout"
    )
    models = importlib.import_module(
        f"EVRPTW_Benchmark.Reinforcement_Learning.{method}.model"
    )
    if method == "AM_EVRPTW":
        policy = models.AMEVRPTWPolicy(
            embedding_dim=16, hidden_dim=16, n_encode_layers=1, n_heads=4,
        )
        kwargs = {"incomplete_penalty_km": 123.0}
    elif method == "EVRPTW_RL":
        policy = models.EVRPTWRLPolicy(embedding_dim=16, structure2vec_rounds=2)
        kwargs = {}
    else:
        policy = models.DRLTSPolicy(embedding_dim=16, n_encode_layers=1, n_heads=4)
        kwargs = {"soft_constraints": soft}
    env_cls = DRLTSSoftConstraintEnv if soft else DRLTSHardConstraintEnv
    env = env_cls(
        _instance(), n_traj=4, info_level="full", use_jit_mask=False,
        objective_config=_objective(), reward_distance_scale_km=5.0,
    ) if method == "DRL_TS" else make_envs(
        [_instance()], n_traj=4, info_level="full", use_jit_mask=False,
        objective_config=_objective(), reward_distance_scale_km=5.0,
    )[0]
    result = module.rollout(
        policy, [env], decode_type="sampling", max_steps=env.max_steps,
        seed=103, **kwargs,
    )
    assert torch.isfinite(result.training_cost).all()
    loss = (result.training_cost.detach() * result.log_likelihood).mean()
    loss.backward()
    gradients = [parameter.grad for parameter in policy.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)


@pytest.mark.parametrize("cost", [False, True])
def test_fresh_training_guard_allows_launcher_logs_but_rejects_training_history(tmp_path, cost):
    args = SimpleNamespace(output_dir=tmp_path, objective=_objective(cost).to_dict(), resume=False)
    (tmp_path / "stdout.log").write_text("launching\n")
    (tmp_path / "provenance.json").write_text("{}")
    assert protocol_trainers.prepare_training_objective(args).is_cost == cost
    assert args.action_constraint_contract_id == ACTION_CONSTRAINT_CONTRACT_ID
    history = tmp_path / "validation_history.jsonl"
    history.write_text("old run\n")
    with pytest.raises(FileExistsError, match="new output directory"):
        protocol_trainers.prepare_training_objective(args)
    assert history.read_text() == "old run\n"


def test_standalone_resume_is_rejected_and_formal_cost_resume_is_checked(tmp_path):
    args = SimpleNamespace(output_dir=tmp_path, objective=_objective().to_dict(), resume=True)
    with pytest.raises(ValueError, match="standalone training does not implement"):
        protocol_trainers.prepare_training_objective(args)
    args.training_epochs = 2
    checkpoint = tmp_path / "checkpoint_latest.pt"
    torch.save({"objective_config": _objective(False).to_dict(), "action_constraint_contract_id": ACTION_CONSTRAINT_CONTRACT_ID}, checkpoint)
    with pytest.raises(ValueError, match="objective mismatch"):
        protocol_trainers.prepare_training_objective(args)
    torch.save({"objective_config": _objective().to_dict(), "action_constraint_contract_id": ACTION_CONSTRAINT_CONTRACT_ID}, checkpoint)
    assert protocol_trainers.prepare_training_objective(args).is_cost


@pytest.mark.parametrize("cost", [False, True])
@pytest.mark.parametrize("method", ["AM_EVRPTW", "EVRPTW_RL", "DRL_TS"])
def test_standalone_evaluation_uses_checkpoint_objective_and_action_contract(tmp_path, monkeypatch, method, cost):
    module = importlib.import_module(
        f"EVRPTW_Benchmark.Reinforcement_Learning.{method}.eval"
    )
    objective = _objective(cost)
    instance = _instance()
    args = SimpleNamespace(
        checkpoint=tmp_path / "checkpoint.pt", objective_config=None,
        decode_type="greedy", candidates=1, candidate_chunk_size=1, device="cpu",
        dataset_path=tmp_path / "dataset", family_root=None, scale="Cus2",
        split_ids="test", track_ids="test", city_slugs=None, seed=1,
        limit=1, output_dir=tmp_path / "evaluation", batch_size=1,
        incomplete_penalty_km=123.0,
    )
    policy = _ScriptedPolicy(method, (1, 2, 0))
    torch.save({
        "model": policy.state_dict(), "args": {}, "objective_config": objective.to_dict(),
        "action_constraint_contract_id": ACTION_CONSTRAINT_CONTRACT_ID,
    }, args.checkpoint)
    monkeypatch.setattr(module, "parse_args", lambda: args)
    policy_name = {"AM_EVRPTW": "AMEVRPTWPolicy", "EVRPTW_RL": "EVRPTWRLPolicy", "DRL_TS": "DRLTSPolicy"}[method]
    monkeypatch.setattr(module, policy_name, lambda **_kwargs: policy)
    monkeypatch.setattr(module, "Stage2TaskPool", lambda **_kwargs: SimpleNamespace(first=lambda **_kw: [instance]))
    module.main()
    row = json.loads((args.output_dir / "routes.jsonl").read_text())
    assert row["verifier_passed"] is True
    assert row["objective_mode"] == objective.mode
    assert row["objective_unit"] == objective.unit
    assert row["objective_value"] == pytest.approx(objective.value(row["objective_distance_km"], 1))
    assert row["vehicles_started"] == 1
    assert row["action_constraint_contract_id"] == ACTION_CONSTRAINT_CONTRACT_ID


@pytest.mark.parametrize("cost", [False, True])
@pytest.mark.parametrize("contract_fields", [
    {},
    {"action_constraint_contract_id": "drl_previous_rules_v0"},
    {"action_constraint_contract_id": ACTION_CONSTRAINT_CONTRACT_ID,
     "args": {"action_constraint_contract_id": "drl_previous_rules_v0"}},
    {"action_constraint_contract_id": ACTION_CONSTRAINT_CONTRACT_ID,
     "config": {"action_constraint_contract_id": "drl_previous_rules_v0"}},
])
def test_reinforce_resume_rejects_missing_old_or_conflicting_action_contract(
    tmp_path, monkeypatch, cost, contract_fields
):
    objective = _objective(cost)
    path = tmp_path / "checkpoint_latest.pt"
    torch.save({
        "protocol_id": "action-test", "objective_config": objective.to_dict(),
        **contract_fields,
    }, path)
    args = SimpleNamespace(
        output_dir=tmp_path, objective=objective.to_dict(), resume=True, training_epochs=2,
    )
    with pytest.raises(ValueError, match="action constraint contract mismatch"):
        protocol_trainers.prepare_training_objective(args)

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("incompatible action contract must fail before loading any model or optimizer")

    policy = torch.nn.Linear(1, 1)
    baseline = deepcopy(policy)
    optimizer = torch.optim.Adam(policy.parameters())
    for target in (policy, baseline, optimizer):
        monkeypatch.setattr(target, "load_state_dict", unexpected_load)
    with pytest.raises(ValueError, match="action constraint contract mismatch"):
        protocol_trainers._load_checkpoint(
            path, policy=policy, baseline=baseline, optimizer=optimizer,
            protocol_id="action-test", objective_config=objective,
        )


@pytest.mark.parametrize("cost", [False, True])
@pytest.mark.parametrize("method", ["AM_EVRPTW", "EVRPTW_RL", "DRL_TS"])
@pytest.mark.parametrize("contract", [None, "drl_previous_rules_v0"])
def test_evaluation_rejects_legacy_action_contract_before_model_or_data(
    tmp_path, monkeypatch, cost, method, contract
):
    module = importlib.import_module(
        f"EVRPTW_Benchmark.Reinforcement_Learning.{method}.eval"
    )
    checkpoint = tmp_path / "checkpoint.pt"
    payload = {"objective_config": _objective(cost).to_dict()}
    if contract is not None:
        payload["action_constraint_contract_id"] = contract
    torch.save(payload, checkpoint)
    args = SimpleNamespace(
        checkpoint=checkpoint, decode_type="greedy", candidates=1, device="cpu",
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)

    def unexpected_construction(*_args, **_kwargs):
        pytest.fail("incompatible action contract must fail before model or data construction")

    policy_name = {
        "AM_EVRPTW": "AMEVRPTWPolicy", "EVRPTW_RL": "EVRPTWRLPolicy", "DRL_TS": "DRLTSPolicy",
    }[method]
    monkeypatch.setattr(module, policy_name, unexpected_construction)
    monkeypatch.setattr(module, "Stage2TaskPool", unexpected_construction)
    with pytest.raises(ValueError, match="action constraint contract mismatch"):
        module.main()


@pytest.mark.parametrize("cost", [False, True])
@pytest.mark.parametrize("method", ["AM_EVRPTW", "EVRPTW_RL", "DRL_TS"])
def test_standalone_training_saves_action_contract_in_checkpoint_and_history(
    tmp_path, monkeypatch, cost, method
):
    module = importlib.import_module(
        f"EVRPTW_Benchmark.Reinforcement_Learning.{method}.train"
    )
    argv = [
        "train", "--dataset-path", str(tmp_path / "unused-dataset"),
        "--output-dir", str(tmp_path / "train"), "--device", "cpu",
        "--batch-size", "2", "--baseline-eval-size", "2", "--embedding-dim", "16",
    ]
    if method == "EVRPTW_RL":
        argv += ["--iterations", "1", "--structure2vec-rounds", "1"]
    else:
        argv += ["--epochs", "1", "--n-encode-layers", "1", "--n-heads", "4"]
        argv += ["--steps-per-epoch" if method == "AM_EVRPTW" else "--batches-per-epoch", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    args = module.parse_args()
    args.objective = _objective(cost).to_dict()
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "Stage2TaskPool", lambda **_kwargs: SimpleNamespace(
        first=lambda *, limit: [_instance() for _ in range(limit)],
        sample=lambda count: [_instance() for _ in range(count)],
    ))
    # This bounded synthetic test exercises real gradient/save paths, not the
    # significance test (identical tiny instances have zero paired variance).
    monkeypatch.setattr(module, "ttest_rel", lambda *_args, **_kwargs: SimpleNamespace(pvalue=1.0))
    module.main()
    payload = torch.load(args.output_dir / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    assert payload["action_constraint_contract_id"] == ACTION_CONSTRAINT_CONTRACT_ID
    assert payload["args"]["action_constraint_contract_id"] == ACTION_CONSTRAINT_CONTRACT_ID
    assert payload["objective_config"] == _objective(cost).to_dict()
    history = json.loads((args.output_dir / "train_history.jsonl").read_text())
    assert history["action_constraint_contract_id"] == ACTION_CONSTRAINT_CONTRACT_ID


@pytest.mark.parametrize("cost", [False, True])
@pytest.mark.parametrize("method", ["AM_EVRPTW", "EVRPTW_RL", "DRL_TS"])
def test_formal_training_main_rejects_old_action_contract_before_data_or_model(
    tmp_path, monkeypatch, cost, method
):
    module = importlib.import_module(
        f"EVRPTW_Benchmark.Reinforcement_Learning.{method}.train"
    )
    torch.save({"objective_config": _objective(cost).to_dict()}, tmp_path / "checkpoint_latest.pt")
    args = SimpleNamespace(
        output_dir=tmp_path, objective=_objective(cost).to_dict(), resume=True, training_epochs=2,
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)

    def unexpected_construction(*_args, **_kwargs):
        pytest.fail("old action contract must fail before training data or model construction")

    policy_name = {
        "AM_EVRPTW": "AMEVRPTWPolicy", "EVRPTW_RL": "EVRPTWRLPolicy", "DRL_TS": "DRLTSPolicy",
    }[method]
    monkeypatch.setattr(module, policy_name, unexpected_construction)
    monkeypatch.setattr(module, "Stage2TaskPool", unexpected_construction)
    with pytest.raises(ValueError, match="action constraint contract mismatch"):
        module.main()
