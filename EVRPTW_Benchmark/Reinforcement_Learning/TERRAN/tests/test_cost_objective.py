from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import _instance
from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import (
    ObjectiveConfig, objective_from_checkpoint, resolve_objective,
)
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import rollout as terran_rollout
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import trainer
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import train as train_entrypoint
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.protocol import _validation_summary
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.pbrs import PotentialRewardConfig


def _objective() -> ObjectiveConfig:
    return resolve_objective(
        "EVRPTW_Benchmark/Reinforcement_Learning/configs/rivian_energy_vehicle_cost_v1.json"
    )


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("pbrs_scale", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("actions, distance, vehicles", [
    ([1, 2, 0], 7.0, 1),
    ([1, 0, 2, 0], 10.0, 2),
    ([3, 1, 0, 2, 0], 10.0, 2),
])
def test_cost_pbrs_keeps_every_departure_fee_and_undiscounted_returns(
    normalize: bool, pbrs_scale: float, actions: list[int], distance: float, vehicles: int,
) -> None:
    objective = _objective()
    env = make_terran_env(
        instance=_instance(), n_traj=1, use_jit_mask=False,
        normalize_reward=normalize,
        reward_distance_scale_mode="single_customer_repair_sum",
        objective_config=objective,
        pbrs_config=PotentialRewardConfig(
            use_customer_pbrs=True, use_repair_distance_pbrs=True,
            customer_progress_budget=0.5, repair_progress_coef=0.5, gamma=1.0,
        ),
    )
    env.set_reward_scale(pbrs_scale)
    observation, _ = env.reset(seed=1234)
    rewards, dones, components = [], [], []
    for action in actions:
        assert observation["action_mask"][0, action]
        observation, reward, terminated, truncated, info = env.step(np.asarray([action]))
        rewards.append(torch.as_tensor(reward))
        dones.append(torch.as_tensor(terminated | truncated))
        components.append(info["reward_components"])
        np.testing.assert_allclose(info["reward_components"]["base_non_objective"], 0, atol=1e-6)

    scale = env.unwrapped.reward_objective_scale if normalize else 1.0
    assert info["success"].tolist() == [True]
    assert info["objective_distance_km"].tolist() == [distance]
    assert info["vehicles_started"].tolist() == [vehicles]
    assert info["objective_value"][0] == pytest.approx(objective.value(distance, vehicles))
    row = terran_rollout.select_best_trajectory(info, include_routes=False)
    assert row["objective_mode"] == objective.mode
    assert row["objective_unit"] == "USD"
    assert row["objective_profile_id"] == objective.profile_id
    for key, expected in {
        "base": -objective.value(distance, vehicles) / scale,
        "objective": -objective.value(distance, vehicles) / scale,
        "electricity_cost": -objective.distance_unit_cost * distance / scale,
        "vehicle_cost": -objective.vehicle_fixed_cost_usd * vehicles / scale,
        # This legacy diagnostic stays normalized km, not a renamed USD value.
        "distance": -distance / (env.unwrapped.reward_distance_scale_km if normalize else 1.0),
    }.items():
        assert sum(float(row[key][0]) for row in components) == pytest.approx(expected, abs=1e-5)
    returns = terran_rollout.compute_returns(torch.stack(rewards), torch.stack(dones), gamma=1.0)
    # The two strict potentials telescope to the fixed initial-state offset.
    assert returns[0, 0].item() == pytest.approx(-objective.value(distance, vehicles) / scale + pbrs_scale, abs=1e-5)
    _, reward, _, _, final_info = env.step(np.asarray([0]))
    assert reward.tolist() == [0.0]
    assert final_info["reward_components"]["vehicle_cost"].tolist() == [0.0]


@pytest.mark.parametrize("ending", ["invalid", "horizon"])
def test_charger_first_failure_retains_fee_separate_from_failure_penalty(ending: str) -> None:
    objective = _objective()
    env = make_terran_env(
        instance=_instance(), n_traj=1, use_jit_mask=False, normalize_reward=False,
        objective_config=objective, invalid_action_penalty=-1.0,
        rollout_horizon_steps=1 if ending == "horizon" else 10,
        pbrs_config=PotentialRewardConfig(use_customer_pbrs=True, use_terminal_heuristic=True),
    )
    env.reset(seed=1234)
    _, _, _, _, info = env.step(np.asarray([3]))
    component = info["reward_components"]
    assert component["base"][0] == pytest.approx(-objective.value(1.0, 1))
    assert component["vehicle_cost"][0] == pytest.approx(-objective.vehicle_fixed_cost_usd)
    assert component["base_non_objective"][0] == pytest.approx(0.0, abs=1e-6)
    if ending == "invalid":
        _, _, _, _, info = env.step(np.asarray([999]))
        assert info["reward_components"]["base_non_objective"].tolist() == [-1.0]
        assert info["reward_components"]["vehicle_cost"].tolist() == [0.0]
    assert not info["success"][0]
    assert info["vehicles_started"].tolist() == [1]
    assert info["objective_cost_usd"][0] == pytest.approx(objective.value(1.0, 1))


@pytest.mark.parametrize("pbrs_enabled", [False, True])
def test_rollout_diagnostics_separate_cost_from_auxiliary_rewards(monkeypatch, pbrs_enabled: bool) -> None:
    objective = _objective()
    env = make_terran_env(
        instance=_instance(), n_traj=1, use_jit_mask=False, normalize_reward=True,
        objective_config=objective, reward_distance_scale_mode="single_customer_repair_sum",
        pbrs_config=PotentialRewardConfig(use_customer_pbrs=True) if pbrs_enabled else None,
    )
    actions = iter([1, 0, 2, 0])

    def scripted_actions(*_args, **_kwargs):
        action = torch.tensor([[next(actions)]])
        zero = torch.zeros((1, 1))
        return action, zero, zero, zero, None

    monkeypatch.setattr(terran_rollout, "sample_actions", scripted_actions)
    batch = terran_rollout.collect_rollout(
        None, [env], rollout_steps=10, decode_mode="sample", device="cpu", seed=12,
        cache_static_embeddings=False, reward_discount_factor=1.0,
    )
    diagnostics = batch.reward_diagnostics
    scale = env.unwrapped.reward_objective_scale
    expected = -objective.value(10.0, 2) / scale
    assert diagnostics["objective_sum"] == pytest.approx(expected, abs=1e-6)
    assert diagnostics["objective_discounted_sum"] == pytest.approx(expected, abs=1e-6)
    assert diagnostics["vehicle_cost_sum"] == pytest.approx(-2 * objective.vehicle_fixed_cost_usd / scale)
    assert diagnostics["base_non_objective_sum"] == pytest.approx(0.0, abs=1e-6)
    assert diagnostics["base_sum"] == pytest.approx(diagnostics["electricity_cost_sum"] + diagnostics["vehicle_cost_sum"], abs=1e-6)
    assert diagnostics["shaped_sum"] == pytest.approx(diagnostics["base_sum"] + diagnostics["pbrs_total_sum"] + diagnostics["terminal_heuristic_sum"], abs=1e-6)
    returns = terran_rollout.compute_returns(batch.rewards, batch.dones, gamma=1.0)
    assert returns[0, 0, 0].item() == pytest.approx(diagnostics["shaped_sum"], abs=1e-6)


def test_trajectory_and_training_summary_choose_cost_not_shorter_distance() -> None:
    objective = _objective()
    distance, vehicles = np.asarray([20.0, 25.0, 1.0]), np.asarray([2, 1, 1])
    info = {
        "success": np.asarray([True, True, False]),
        "served_customers": np.asarray([2, 2, 1]),
        "objective_distance_km": distance,
        "vehicle_count": vehicles,
        "vehicles_started": vehicles,
        "objective_value": objective.value(distance, vehicles),
        "objective_cost_usd": objective.value(distance, vehicles),
        "electricity_cost_usd": objective.distance_unit_cost * distance,
        "vehicle_cost_usd": objective.vehicle_fixed_cost_usd * vehicles,
        "objective_mode": objective.mode,
        "objective_unit": objective.unit,
        "objective_config": objective.to_dict(),
    }
    row = terran_rollout.select_best_trajectory(info, include_routes=False)
    assert row["selected_traj_idx"] == 1
    assert row["objective_distance_km"] == 25.0
    assert row["objective_value"] == pytest.approx(objective.value(25.0, 1))
    summary = trainer.summarize_train_infos([info])
    assert summary["train_avg_best_objective_distance_km"] == 25.0
    assert summary["train_avg_best_objective"] == pytest.approx(objective.value(25.0, 1))


@pytest.mark.parametrize("environment_success", [True, False])
def test_online_validation_verifies_and_recomputes_cost_before_aggregating(
    tmp_path: Path, monkeypatch, environment_success: bool,
) -> None:
    objective = _objective()
    instance = _instance()
    instance.distance_matrix_km[1, 2] = 15.0
    info = {
        "success": np.asarray([environment_success, environment_success]),
        "served_customers": np.asarray([2, 2]),
        "objective_distance_km": np.asarray([10.0, 20.0]),
        "vehicles_started": np.asarray([2, 1]),
        "vehicle_count": np.asarray([2, 1]),
        # A stale local score must never outrank independently replayed routes.
        "objective_value": np.asarray([0.0, 1.0]),
        "objective_config": objective.to_dict(),
        "routes": [[[0, 1, 0], [0, 2, 0]], [[0, 1, 2, 0]]],
    }
    row = {
        "_final_info": info, "feasible": True, "selected_traj_idx": 0,
        "objective_distance_km": 10.0, "objective_value": 0.0,
        "vehicle_count": 2, "runtime_s": 0.01,
    }

    def rollout(_agent, envs, **kwargs):
        assert kwargs["return_final_info"] is True
        assert envs[0].unwrapped.objective_config.to_dict() == objective.to_dict()
        return [row]

    monkeypatch.setattr(trainer, "_eval_instance_batches", lambda *_args: [[instance]])
    monkeypatch.setattr(trainer, "rollout_eval_batch", rollout)
    agent = torch.nn.Linear(1, 1)
    result = trainer.evaluate_fixed_dataset(agent, {
        "objective": objective.to_dict(), "data": {}, "env": {"use_jit_mask": False},
        "evaluation": {"eval_path": str(tmp_path), "eval_n_traj": 2, "eval_require_independent_verifier": False},
    }, seed=1234, epoch=1, device="cpu")
    assert agent.training  # Evaluation restores the prior mode.
    assert result["eval_independent_verifier"] is True
    assert result["eval_complete_and_feasible"] == int(environment_success)
    assert row["selected_traj_idx"] == 1
    assert row["objective_distance_km"] == 20.0
    assert row["objective_value"] == pytest.approx(objective.value(20.0, 1))
    if environment_success:
        assert result["eval_avg_objective"] == pytest.approx(objective.value(20.0, 1))
        assert result["eval_avg_objective_distance_km"] == 20.0
        assert result["eval_avg_vehicle_cost_usd"] == objective.vehicle_fixed_cost_usd
    else:
        assert np.isnan(result["eval_avg_objective"])
        assert np.isnan(result["eval_avg_objective_distance_km"])
        assert result["eval_avg_objective_cost_usd"] is None


def test_protocol_csv_summary_averages_only_verified_cost_rows(tmp_path: Path) -> None:
    objective = _objective()
    path = tmp_path / "summary.csv"
    rows = [{
        "verifier_passed": passed, "objective_distance_km": distance,
        "vehicle_count": vehicles, **objective.fields(distance, vehicles),
    } for passed, distance, vehicles in [(True, 20, 1), (True, 10, 2), (False, 0, 0)]]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = _validation_summary(path, data_pass=1)
    assert summary["instances"] == 3 and summary["complete_and_feasible"] == 2
    assert summary["objective_unit"] == "USD"
    assert summary["mean_verified_objective"] == pytest.approx(objective.value(15, 1.5))
    assert summary["mean_verified_cost_usd"] == summary["mean_verified_objective"]
    assert summary["mean_verified_distance_km"] == 15.0
    assert summary["mean_verified_vehicle_count"] == 1.5


@pytest.mark.parametrize("changed", ["legacy", "electricity_price_usd_per_kwh", "consumption_kwh_per_km", "vehicle_fixed_cost_usd", "profile_id"])
def test_resume_rejects_different_objective_even_with_same_gamma_and_contract(changed: str) -> None:
    cfg = {
        "objective": _objective().to_dict(),
        "training": {"gamma": 1.0, "reward_contract_id": "terran_undiscounted_energy_vehicle_pbrs_v1"},
    }
    saved = deepcopy(cfg)
    if changed == "legacy":
        saved.pop("objective")
    elif changed == "profile_id":
        saved["objective"][changed] = "another-profile"
    else:
        saved["objective"][changed] *= 2
    with pytest.raises(ValueError, match="objective configuration mismatch"):
        trainer.validate_resume_reward_contract(cfg, {"config": saved})


def test_legacy_checkpoint_stays_distance_and_cannot_be_relabelled_cost() -> None:
    assert objective_from_checkpoint({"config": {}}).mode == "distance"
    with pytest.raises(ValueError, match="checkpoint objective mismatch"):
        objective_from_checkpoint({"config": {}}, _objective())


@pytest.mark.parametrize("malformed", ["mutable-path", "inconsistent-snapshots"])
def test_resume_rejects_mutable_or_inconsistent_checkpoint_objective(malformed: str) -> None:
    cfg = {
        "objective": _objective().to_dict(),
        "training": {"gamma": 1.0, "reward_contract_id": "terran_undiscounted_energy_vehicle_pbrs_v1"},
    }
    payload = {"config": deepcopy(cfg)}
    if malformed == "mutable-path":
        payload["config"]["objective"] = "EVRPTW_Benchmark/Reinforcement_Learning/configs/rivian_energy_vehicle_cost_v1.json"
    else:
        payload["objective_config"] = _objective().to_dict()
        payload["config"]["objective"]["vehicle_fixed_cost_usd"] *= 2
    with pytest.raises(ValueError, match="objective configuration mismatch"):
        trainer.validate_resume_reward_contract(cfg, payload)


@pytest.mark.parametrize("explicit_override", [False, True])
def test_cli_default_keeps_yaml_objective_and_explicit_override_is_honored(
    tmp_path: Path, monkeypatch, explicit_override: bool,
) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "stage2_cus100_terran.yaml"
    argv = ["terran-train", "--config", str(config_path), "--seed", "1234"]
    expected = _objective().to_dict()
    if explicit_override:
        expected["profile_id"] = "explicit-override-test"
        expected["vehicle_fixed_cost_usd"] = 1.0
        override_path = tmp_path / "objective.json"
        override_path.write_text(json.dumps(expected), encoding="utf-8")
        argv.extend(["--objective-config", str(override_path)])
    monkeypatch.setattr(sys, "argv", argv)
    captured = {}

    def train(_cfg, **kwargs):
        captured.update(kwargs["overrides"])
        return tmp_path / "unused.pt"

    monkeypatch.setattr(train_entrypoint, "train_from_config", train)
    monkeypatch.setattr(train_entrypoint, "finalize_protocol", lambda *_args: None)
    train_entrypoint.main()
    assert captured["objective"] == expected


def test_formal_yaml_resolves_shared_cost_profile_and_gamma_one() -> None:
    cfg = trainer.load_config(Path(__file__).resolve().parents[1] / "configs" / "stage2_cus100_terran.yaml")
    assert isinstance(cfg["objective"], str)
    assert resolve_objective(cfg["objective"]).to_dict() == _objective().to_dict()
    assert cfg["training"]["gamma"] == 1.0
    assert cfg["training"]["reward_contract_id"] == "terran_undiscounted_energy_vehicle_pbrs_v1"


def test_direct_cost_training_rejects_discounting_before_initialization(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cost objective requires undiscounted"):
        trainer.train_from_config({
            "output_dir": str(tmp_path / "must-not-exist"), "objective": _objective().to_dict(),
            "training": {"gamma": 0.999},
        }, seed=1234, device="cpu")
    assert not (tmp_path / "must-not-exist").exists()


def test_dataset_sum_scale_keeps_singleton_vehicle_reference_count(monkeypatch) -> None:
    objective = _objective()

    class Pool:
        def sample(self):
            return _instance()

        def reward_distance_scale_km(self, mode):
            assert mode == "single_customer_repair_sum"
            return 123.0

    monkeypatch.setattr(trainer, "FixedDatasetInstancePool", lambda **_kwargs: Pool())
    cfg = {
        "objective": objective.to_dict(),
        "data": {"train_dataset_path": "unused", "num_customers": 2, "num_charging_stations": 1},
        "training": {"num_envs_per_gpu": 1, "n_traj": 1, "rollout_steps": 10},
        "env": {"reward_distance_scale_mode": "dataset_single_customer_repair_sum", "use_jit_mask": False},
    }
    envs, _ = trainer.make_envs(cfg, seed=1234)
    assert cfg["env"]["reward_distance_scale_mode"] == "single_customer_repair_sum"
    assert envs[0].unwrapped.reward_objective_scale == pytest.approx(objective.value(123.0, 2))
