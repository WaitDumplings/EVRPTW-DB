from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import csv
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from EVRPTW_Benchmark.Reinforcement_Learning.common.action_constraints import (
    ACTION_CONSTRAINT_CONTRACT_ID,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import ObjectiveConfig
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import eval as legacy_eval
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import eval_stage2, trainer
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.protocol import _validation_summary


def _config(mode: str) -> dict:
    return {
        "action_constraint_contract_id": ACTION_CONSTRAINT_CONTRACT_ID,
        "objective": ObjectiveConfig(mode=mode).to_dict(),
        "training": {
            "gamma": 1.0,
            "reward_contract_id": (
                "terran_undiscounted_energy_vehicle_pbrs_v1"
                if mode == "energy_vehicle_cost" else "terran_undiscounted_distance_pbrs_v1"
            ),
        },
    }


def _payload(mode: str, variant: str) -> dict:
    payload = {"config": _config(mode)}
    if variant == "missing":
        payload["config"].pop("action_constraint_contract_id")
    elif variant == "wrong":
        payload["config"]["action_constraint_contract_id"] = "old_cs_chains_allowed"
    elif variant == "conflicting":
        payload["action_constraint_contract_id"] = ACTION_CONSTRAINT_CONTRACT_ID
        payload["config"]["action_constraint_contract_id"] = "old_cs_chains_allowed"
    elif variant != "current":
        raise AssertionError(variant)
    return payload


@pytest.mark.parametrize("mode", ["distance", "energy_vehicle_cost"])
@pytest.mark.parametrize("variant", ["missing", "wrong", "conflicting"])
def test_resume_rejects_old_action_contract_even_when_objective_and_gamma_match(mode, variant) -> None:
    with pytest.raises(ValueError, match="action constraint contract mismatch"):
        trainer.validate_resume_reward_contract(_config(mode), _payload(mode, variant))


@pytest.mark.parametrize("mode", ["distance", "energy_vehicle_cost"])
def test_resume_accepts_current_action_contract_without_altering_objective(mode) -> None:
    cfg = _config(mode)
    before = deepcopy(cfg)
    trainer.validate_resume_reward_contract(cfg, _payload(mode, "current"))
    assert cfg == before


@pytest.mark.parametrize("mode", ["distance", "energy_vehicle_cost"])
def test_training_rejects_unversioned_resume_before_model_or_output_creation(tmp_path, monkeypatch, mode) -> None:
    checkpoint = tmp_path / "legacy.pt"
    torch.save(_payload(mode, "missing"), checkpoint)
    cfg = _config(mode)
    output = tmp_path / "new-run"
    cfg.update(output_dir=str(output), protocol={"resume_checkpoint": str(checkpoint)})
    agent = Mock()
    monkeypatch.setattr(trainer, "Agent", agent)
    with pytest.raises(ValueError, match="action constraint contract mismatch"):
        trainer.train_from_config(cfg, seed=1234, device="cpu")
    agent.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("mode", ["distance", "energy_vehicle_cost"])
@pytest.mark.parametrize("entrypoint", [legacy_eval, eval_stage2], ids=["legacy-eval", "stage2-eval"])
@pytest.mark.parametrize("variant", ["missing", "wrong", "conflicting"])
def test_eval_rejects_old_action_contract_before_loading_policy(monkeypatch, mode, entrypoint, variant) -> None:
    monkeypatch.setattr(entrypoint, "parse_args", lambda: SimpleNamespace(
        checkpoint_path=Path("unused.pt"), checkpoint=Path("unused.pt"),
        objective_config=None, device="cpu", decode_mode="sample", candidates=1,
    ))
    monkeypatch.setattr(entrypoint.torch, "load", lambda *_args, **_kwargs: _payload(mode, variant))
    agent = Mock()
    monkeypatch.setattr(entrypoint, "Agent", agent)
    with pytest.raises(ValueError, match="action constraint contract mismatch"):
        entrypoint.main()
    agent.assert_not_called()


@pytest.mark.parametrize("entrypoint", [legacy_eval, eval_stage2], ids=["legacy-eval", "stage2-eval"])
def test_eval_accepts_current_contract_before_policy_construction(monkeypatch, entrypoint) -> None:
    class ReachedPolicy(Exception):
        pass

    monkeypatch.setattr(entrypoint, "parse_args", lambda: SimpleNamespace(
        checkpoint_path=Path("unused.pt"), checkpoint=Path("unused.pt"),
        objective_config=None, device="cpu", decode_mode="sample", candidates=1,
        solver_name=None,
    ))
    monkeypatch.setattr(entrypoint.torch, "load", lambda *_args, **_kwargs: _payload("energy_vehicle_cost", "current"))
    monkeypatch.setattr(entrypoint, "Agent", Mock(side_effect=ReachedPolicy))
    with pytest.raises(ReachedPolicy):
        entrypoint.main()


@pytest.mark.parametrize("mode", ["distance", "energy_vehicle_cost"])
def test_checkpoint_save_persists_independent_action_contract(tmp_path, mode) -> None:
    cfg = _config(mode)
    policy = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(policy.parameters())
    path = tmp_path / "checkpoints" / "current.pt"
    trainer.save_checkpoint(path, policy, optimizer, cfg, epoch=1, seed=1234)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["action_constraint_contract_id"] == ACTION_CONSTRAINT_CONTRACT_ID
    assert checkpoint["config"] == cfg
    trainer.validate_resume_reward_contract(cfg, checkpoint)


@pytest.mark.parametrize("contract_id", [None, "old_cs_chains_allowed"])
def test_training_does_not_relabel_explicit_incompatible_action_config(tmp_path, contract_id) -> None:
    cfg = _config("energy_vehicle_cost")
    cfg["action_constraint_contract_id"] = contract_id
    cfg["output_dir"] = str(tmp_path / "new-run")
    with pytest.raises(ValueError, match="action constraint contract mismatch"):
        trainer.train_from_config(cfg, seed=1234, device="cpu")
    assert not (tmp_path / "new-run").exists()


@pytest.mark.parametrize("contract_id", [None, "old_cs_chains_allowed"])
def test_validation_summary_does_not_relabel_old_evaluation_rows(tmp_path, contract_id) -> None:
    path = tmp_path / "summary.csv"
    row = {"verifier_passed": "true", "objective_distance_km": 1.0}
    if contract_id is not None:
        row["action_constraint_contract_id"] = contract_id
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="action constraint contract mismatch"):
        _validation_summary(path, data_pass=1)
