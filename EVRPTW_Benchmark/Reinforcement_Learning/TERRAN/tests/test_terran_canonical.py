from __future__ import annotations

from copy import deepcopy
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from EVRPTW_Benchmark.Exact.Gurobi_Solver.route_validator import validate_routes
from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import (
    _instance,
)
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import (
    make_terran_env,
)
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.models import Agent
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.pbrs import (
    PotentialRewardConfig,
)
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import protocol as terran_protocol
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import data_pool as terran_data_pool
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import trainer as terran_trainer
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.trainer import (
    pbrs_scale_for_epoch,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.data_pass import DataPassState
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_stream import (
    STREAM_CONTRACT_SCHEMA,
    training_stream_contract_digest,
)
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import (
    collect_rollout, compute_returns, rollout_eval_batch,
)


def test_terran_pool_revalidates_frozen_stream_before_reading(
    monkeypatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        terran_data_pool,
        "Stage2TaskPool",
        lambda **_kwargs: SimpleNamespace(tasks=[]),
    )

    def load_contract(_path):
        calls.append("load")
        return {"sha256": "actual"}

    def read_ids(_path):
        calls.append("read")
        return []

    monkeypatch.setattr(
        terran_data_pool, "load_training_stream_contract", load_contract
    )
    monkeypatch.setattr(terran_data_pool, "read_stream_view_ids", read_ids)
    with pytest.raises(
        ValueError, match="changed after protocol configuration"
    ):
        terran_data_pool.Stage2TERRANPool(
            dataset_path="unused.parquet",
            training_stream_path="stream.parquet",
            training_stream_contract_sha256="expected",
        )
    assert calls == ["load"]

    calls.clear()
    pool = terran_data_pool.Stage2TERRANPool(
        dataset_path="unused.parquet",
        training_stream_path="stream.parquet",
        training_stream_contract_sha256="actual",
    )
    assert pool._stream_view_ids == []
    assert calls == ["load", "read"]


def test_terran_pool_reuses_preverified_snapshot_but_still_reads_and_checks_ids(
    monkeypatch,
) -> None:
    task = SimpleNamespace(view_id="view-1")
    monkeypatch.setattr(
        terran_data_pool,
        "Stage2TaskPool",
        lambda **_kwargs: SimpleNamespace(tasks=[task]),
    )
    monkeypatch.setattr(
        terran_data_pool,
        "load_training_stream_contract",
        lambda *_args, **_kwargs: pytest.fail(
            "preverified TERRAN reuse must not rehash the stream"
        ),
    )
    monkeypatch.setattr(
        terran_data_pool,
        "read_stream_view_ids",
        lambda _path: ["view-1"],
    )
    snapshot = {
        "schema": STREAM_CONTRACT_SCHEMA,
        "sha256": "a" * 64,
        "sample_count": 1,
        "scale": "Cus100",
        "seed": 1234,
    }

    pool = terran_data_pool.Stage2TERRANPool(
        dataset_path="unused.parquet",
        scale="Cus100",
        seed=1234,
        training_stream_path="stream.parquet",
        training_stream_contract_sha256=snapshot["sha256"],
        training_stream_contract_snapshot=snapshot,
        stream_integrity_mode="reuse_preverified_snapshot_no_rehash",
    )
    assert pool._stream_view_ids == ["view-1"]

    monkeypatch.setattr(
        terran_data_pool,
        "read_stream_view_ids",
        lambda _path: ["outside-pool"],
    )
    with pytest.raises(ValueError, match="outside its pool"):
        terran_data_pool.Stage2TERRANPool(
            dataset_path="unused.parquet",
            scale="Cus100",
            seed=1234,
            training_stream_path="stream.parquet",
            training_stream_contract_sha256=snapshot["sha256"],
            training_stream_contract_snapshot=snapshot,
            stream_integrity_mode="reuse_preverified_snapshot_no_rehash",
        )


def test_undiscounted_complete_returns_sum_remaining_rewards() -> None:
    rewards = torch.tensor([[[2.0, -1.0]], [[-3.0, -2.0]], [[5.0, -4.0]]])
    dones = torch.zeros_like(rewards, dtype=torch.bool)
    dones[-1] = True

    returns = compute_returns(rewards, dones, gamma=1.0)

    torch.testing.assert_close(
        returns,
        torch.tensor([[[4.0, -7.0]], [[2.0, -6.0]], [[5.0, -4.0]]]),
    )


@pytest.mark.parametrize("gamma", [1.0, 0.999])
def test_returns_stop_at_each_trajectory_terminal_mask(gamma: float) -> None:
    # Nonzero later entries expose reward leakage across each trajectory's own
    # terminal boundary; the collector's valid mask excludes those later rows.
    rewards = torch.tensor(
        [[[1.0, 10.0]], [[2.0, 20.0]], [[100.0, 30.0]], [[200.0, 400.0]]]
    )
    dones = torch.tensor(
        [[[False, False]], [[True, False]], [[True, True]], [[True, True]]]
    )

    returns = compute_returns(rewards, dones, gamma=gamma)

    torch.testing.assert_close(
        returns[0],
        torch.tensor([[1.0 + gamma * 2.0, 10.0 + gamma * 20.0 + gamma**2 * 30.0]]),
    )
    assert returns[1, 0, 0].item() == 2.0
    assert returns[2, 0, 1].item() == 30.0


def test_default_training_and_pbrs_gamma_are_undiscounted() -> None:
    cfg = {"pbrs": {"use_customer_pbrs": True}}
    pbrs_config = terran_trainer.build_pbrs_config(cfg)

    assert terran_trainer.training_gamma(cfg) == 1.0
    assert PotentialRewardConfig().gamma == 1.0
    assert pbrs_config is not None and pbrs_config.gamma == 1.0


@pytest.mark.parametrize("gamma", [-0.01, 1.01, float("nan"), float("inf"), -float("inf")])
def test_invalid_gamma_is_rejected_by_training_and_pbrs(gamma: float) -> None:
    with pytest.raises(ValueError, match="gamma must be finite and in"):
        terran_trainer.training_gamma({"training": {"gamma": gamma}})
    with pytest.raises(ValueError, match="gamma must be finite and in"):
        PotentialRewardConfig(gamma=gamma)


@pytest.mark.parametrize(
    "bonus", [-0.01, float("nan"), float("inf"), -float("inf")]
)
def test_invalid_terminal_success_bonus_is_rejected(bonus: float) -> None:
    with pytest.raises(ValueError, match="terminal_success_bonus"):
        PotentialRewardConfig(terminal_success_bonus=bonus)
    with pytest.raises(ValueError, match="terminal_success_bonus"):
        terran_trainer.terminal_success_bonus(
            {"pbrs": {"terminal_success_bonus": bonus}}
        )


@pytest.mark.parametrize("gamma", [0.0, 0.999, 1.0])
@pytest.mark.parametrize("contract", [None, "terran_undiscounted_distance_pbrs_v1"])
def test_resume_accepts_matching_gamma_and_reward_contract(
    gamma: float, contract: str | None
) -> None:
    training = {"gamma": gamma}
    if contract is not None:
        training["reward_contract_id"] = contract

    terran_trainer.validate_resume_reward_contract(
        {"training": dict(training)}, {"config": {"training": dict(training)}}
    )


@pytest.mark.parametrize(
    ("saved_training", "message"),
    [
        ({}, "missing training.gamma"),
        ({"gamma": 0.999}, "gamma mismatch"),
        ({"gamma": 1.0}, "reward contract mismatch"),
        (
            {"gamma": 1.0, "reward_contract_id": "different-contract"},
            "reward contract mismatch",
        ),
    ],
)
def test_resume_rejects_missing_or_changed_reward_contract(
    saved_training: dict, message: str
) -> None:
    cfg = {
        "training": {
            "gamma": 1.0,
            "reward_contract_id": "terran_undiscounted_distance_pbrs_v1",
        }
    }
    with pytest.raises(ValueError, match=message):
        terran_trainer.validate_resume_reward_contract(
            cfg, {"config": {"training": saved_training}}
        )


@pytest.mark.parametrize(
    ("saved_optimizer", "saved_weight_decay", "state_weight_decay"),
    [
        (None, 0.01, 0.01),
        ("adam", 0.01, 0.01),
        ("adamw", None, 0.01),
        ("adamw", 0.0, 0.0),
        ("adamw", 0.01, 0.0),
    ],
)
def test_terran_resume_rejects_old_optimizer_contract(
    saved_optimizer, saved_weight_decay, state_weight_decay
) -> None:
    current_training = {
        "gamma": 1.0,
        "reward_contract_id": "terran_undiscounted_distance_pbrs_v1",
        "optimizer": "adamw",
        "weight_decay": 0.01,
    }
    saved_training = dict(current_training)
    if saved_optimizer is None:
        saved_training.pop("optimizer")
    else:
        saved_training["optimizer"] = saved_optimizer
    if saved_weight_decay is None:
        saved_training.pop("weight_decay")
    else:
        saved_training["weight_decay"] = saved_weight_decay
    payload = {
        "config": {"training": saved_training},
        "optimizer_state_dict": {
            "param_groups": [{"weight_decay": state_weight_decay}]
        },
    }
    with pytest.raises(ValueError, match="optimizer"):
        terran_trainer.validate_resume_reward_contract(
            {"training": current_training}, payload
        )


def test_terran_resume_accepts_exact_adamw_contract() -> None:
    training = {
        "gamma": 1.0,
        "reward_contract_id": "terran_undiscounted_distance_pbrs_v1",
        "optimizer": "adamw",
        "weight_decay": 0.01,
    }
    terran_trainer.validate_resume_reward_contract(
        {"training": dict(training)},
        {
            "config": {"training": dict(training)},
            "optimizer_state_dict": {
                "param_groups": [{"weight_decay": 0.01}]
            },
        },
    )


def _terran_scientific_signature_config() -> dict:
    return {
        "data": {
            "stage2_scale": "Cus2",
            "num_customers": 2,
            "num_charging_stations": 1,
            "stage2_training_representation": "G",
        },
        "training": {
            "epochs": 10,
            "num_envs_per_gpu": 4,
            "n_traj": 3,
            "rollout_steps": 8,
            "logical_microbatches_per_epoch": 2,
            "ppo_update_epochs": 3,
            "num_minibatches": 2,
            "gradient_accumulation_steps": 1,
            "ppo_step_chunk_size": 4,
            "gamma": 1.0,
            "optimizer": "adamw",
            "weight_decay": 0.01,
            "minimum_training_epochs": 5,
            "post_minimum_validation_every_epochs": 5,
            "validation_epochs": [5, 10],
            "early_stop_patience_validations": 2,
            "early_stop_start_epoch": 5,
        },
        "evaluation": {
            "eval_seed": 77,
            "eval_decode_mode": "sample",
            "eval_n_traj": 5,
            "eval_limit": 7,
            "eval_max_steps": 12,
            "eval_interval": 5,
            "eval_batch_size": 1,
        },
        "pbrs": {"terminal_success_bonus": 1.0},
        "protocol": {
            "protocol_id": "terran-signature-test",
            "physical_batch_size": 4,
            "effective_batch_size": 8,
            "logical_environments_per_epoch": 8,
            "training_rollout_steps": 8,
            "validation_rollout_steps": 12,
            "validation_every_epochs": 5,
            "minimum_training_epochs": 5,
            "post_minimum_validation_every_epochs": 5,
            "scheduled_validation_epochs": [5, 10],
            "validation_checkpoints": 2,
            "validation_seed": 77,
            "validation_decode_type": "sampling",
            "validation_candidates": 5,
            "early_stop_patience_validations": 2,
            "early_stop_start_epoch": 5,
            "final_validation_limit": 7,
        },
    }


def _signature_payload(cfg: dict) -> dict:
    return {
        "config": cfg,
        "seed": 1234,
        "optimizer_state_dict": {
            "param_groups": [{"weight_decay": 0.01}]
        },
    }


def test_terran_resolved_training_signature_freezes_method_specific_fields() -> None:
    cfg = _terran_scientific_signature_config()
    signature = terran_trainer._freeze_resolved_terran_training_signature(
        cfg, seed=1234
    )

    assert signature["schema"] == "drl_resolved_training_signature_v1"
    assert signature["method_specific"]["method"] == "TERRAN"
    assert signature["method_specific"]["task_reward"] == {
        "terminal_success_bonus": 1.0,
        "unit": "normalized_objective_cost",
    }
    assert signature["method_specific"]["training"] == {
        "epochs": 10,
        "num_envs_per_gpu": 4,
        "n_traj": 3,
        "rollout_steps": 8,
        "logical_microbatches_per_epoch": 2,
        "ppo_update_epochs": 3,
        "num_minibatches": 2,
        "gradient_accumulation_steps": 1,
        "ppo_step_chunk_size": 4,
        "clip_coef": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "learning_rate": 1e-4,
        "max_grad_norm": 1.0,
        "value_loss_type": "mse",
        "value_loss_beta": 1.0,
        "value_residual_scale": 1.0,
        "critic_backbone_grad_scale": 1.0,
        "gamma": 1.0,
    }
    assert signature["method_specific"]["evaluation"] == {
        "seed": 77,
        "decode_mode": "sample",
        "n_traj": 5,
        "limit": 7,
        "max_steps": 12,
        "batch_size": 1,
        "num_batches": None,
        "interval": 5,
    }
    assert cfg["protocol"]["resolved_training_signature"] == signature
    assert cfg["protocol"]["resolved_training_signature_sha256"] == signature["sha256"]


@pytest.mark.parametrize(
    "changed",
    [
        "n_traj",
        "rollout_steps",
        "num_envs",
        "ppo_update_epochs",
        "num_minibatches",
        "ppo_step_chunk_size",
        "terminal_success_bonus",
        "eval_seed",
        "eval_decode",
        "eval_n_traj",
        "eval_limit",
        "eval_max_steps",
        "eval_schedule",
        "validation_epochs",
        "early_stop",
    ],
)
def test_terran_resume_rejects_resolved_scientific_signature_drift(
    changed: str,
) -> None:
    saved = _terran_scientific_signature_config()
    terran_trainer._freeze_resolved_terran_training_signature(saved, seed=1234)
    current = _terran_scientific_signature_config()
    if changed == "n_traj":
        current["training"]["n_traj"] = 4
    elif changed == "rollout_steps":
        current["training"]["rollout_steps"] = 9
        current["protocol"]["training_rollout_steps"] = 9
    elif changed == "num_envs":
        current["training"]["num_envs_per_gpu"] = 5
        current["protocol"]["physical_batch_size"] = 5
        current["protocol"]["effective_batch_size"] = 10
        current["protocol"]["logical_environments_per_epoch"] = 10
    elif changed == "ppo_update_epochs":
        current["training"]["ppo_update_epochs"] = 4
    elif changed == "num_minibatches":
        current["training"]["num_minibatches"] = 3
    elif changed == "ppo_step_chunk_size":
        current["training"]["ppo_step_chunk_size"] = 2
    elif changed == "terminal_success_bonus":
        current["pbrs"]["terminal_success_bonus"] = 0.5
    elif changed == "eval_seed":
        current["evaluation"]["eval_seed"] = 78
        current["protocol"]["validation_seed"] = 78
    elif changed == "eval_decode":
        current["evaluation"]["eval_decode_mode"] = "greedy"
        current["protocol"]["validation_decode_type"] = "greedy"
    elif changed == "eval_n_traj":
        current["evaluation"]["eval_n_traj"] = 6
        current["protocol"]["validation_candidates"] = 6
    elif changed == "eval_limit":
        current["evaluation"]["eval_limit"] = 8
    elif changed == "eval_max_steps":
        current["evaluation"]["eval_max_steps"] = 13
        current["protocol"]["validation_rollout_steps"] = 13
    elif changed == "eval_schedule":
        current["evaluation"]["eval_interval"] = 2
        current["protocol"]["validation_every_epochs"] = 2
    elif changed == "validation_epochs":
        current["training"]["validation_epochs"] = [5, 9, 10]
        current["protocol"]["scheduled_validation_epochs"] = [5, 9, 10]
        current["protocol"]["validation_checkpoints"] = 3
    else:
        current["training"]["early_stop_start_epoch"] = 6
        current["protocol"]["early_stop_start_epoch"] = 6
    terran_trainer._freeze_resolved_terran_training_signature(
        current, seed=1234
    )

    with pytest.raises(ValueError, match="resolved training signature mismatch"):
        terran_trainer.validate_resume_reward_contract(
            current, _signature_payload(saved)
        )


def test_terran_resume_rejects_checkpoint_missing_resolved_signature() -> None:
    current = _terran_scientific_signature_config()
    terran_trainer._freeze_resolved_terran_training_signature(
        current, seed=1234
    )
    saved = _terran_scientific_signature_config()
    with pytest.raises(ValueError, match="signature is missing"):
        terran_trainer.validate_resume_reward_contract(
            current, _signature_payload(saved)
        )


def test_terran_resume_freezes_training_stream_snapshot_and_sha() -> None:
    stream_contract = {
        "schema": STREAM_CONTRACT_SCHEMA,
        "stream_schema": "drl_training_id_stream_v3",
        "content_digest_scheme": "test",
        "stream_content_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "sample_count": 80,
        "scale": "Cus2",
        "seed": 1234,
        "source_index_sha256": "c" * 64,
        "allowed_family_ids_sha256": None,
    }
    stream_contract["sha256"] = training_stream_contract_digest(
        stream_contract
    )
    saved = _terran_scientific_signature_config()
    saved["protocol"].update(
        {
            "training_stream_path": "/tmp/frozen-stream.parquet",
            "training_stream_contract_snapshot": deepcopy(stream_contract),
            "training_stream_contract_sha256": stream_contract["sha256"],
        }
    )
    terran_trainer._freeze_resolved_terran_training_signature(saved, seed=1234)
    current = deepcopy(saved)
    terran_trainer.validate_resume_reward_contract(
        current, _signature_payload(saved)
    )

    current = _terran_scientific_signature_config()
    current["protocol"].update(
        {
            "training_stream_path": "/tmp/frozen-stream.parquet",
            "training_stream_contract_snapshot": deepcopy(stream_contract),
            "training_stream_contract_sha256": stream_contract["sha256"],
        }
    )
    terran_trainer._freeze_resolved_terran_training_signature(
        current, seed=1234
    )
    saved_without_stream = _terran_scientific_signature_config()
    terran_trainer._freeze_resolved_terran_training_signature(
        saved_without_stream, seed=1234
    )
    with pytest.raises(ValueError, match="training-stream contract mismatch"):
        terran_trainer.validate_resume_reward_contract(
            current, _signature_payload(saved_without_stream)
        )


def test_fresh_output_allows_launcher_only_records(tmp_path: Path) -> None:
    for name, content in (
        ("provenance.json", '{"schema": "drl_job_provenance_v1"}'),
        ("stdout.log", "launching\n"),
        ("stderr.log", ""),
    ):
        (tmp_path / name).write_text(content, encoding="utf-8")
    (tmp_path / "checkpoints").mkdir()
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}

    terran_trainer.validate_fresh_training_output({"output_dir": str(tmp_path)})

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == before
    assert not list((tmp_path / "checkpoints").iterdir())


@pytest.mark.parametrize(
    "artifact",
    [
        "validation_history.jsonl",
        "checkpoints/checkpoint_0001.pt",
        "data_pass_state.json",
        "logs/train_log.csv",
    ],
)
def test_fresh_output_rejects_existing_training_evidence(
    tmp_path: Path, artifact: str
) -> None:
    path = tmp_path / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("prior training evidence", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already contains training evidence"):
        terran_trainer.validate_fresh_training_output({"output_dir": str(tmp_path)})

    assert path.read_text(encoding="utf-8") == "prior training evidence"


@pytest.mark.parametrize("mode", ["fresh_history", "resume_gamma_mismatch"])
def test_training_contract_rejection_precedes_model_and_environment_creation(
    tmp_path: Path, monkeypatch, mode: str
) -> None:
    agent_factory = Mock(side_effect=AssertionError("Agent must not be created"))
    env_factory = Mock(side_effect=AssertionError("Environments must not be created"))
    monkeypatch.setattr(terran_trainer, "Agent", agent_factory)
    monkeypatch.setattr(terran_trainer, "make_envs", env_factory)
    cfg = {
        "output_dir": str(tmp_path),
        "training": {"gamma": 1.0},
        "data": {"num_customers": 2, "num_charging_stations": 1},
    }
    if mode == "fresh_history":
        artifact = tmp_path / "validation_history.jsonl"
        artifact.write_text('{"logical_epoch": 1}\n', encoding="utf-8")
        expected_error = FileExistsError
        message = "already contains training evidence"
    else:
        artifact = tmp_path / "checkpoint_latest.pt"
        # No model/optimizer payload is necessary: validation must reject the
        # incompatible reward contract before trying to initialize/load either.
        torch.save({"config": {"training": {"gamma": 0.999}}}, artifact)
        cfg["protocol"] = {"resume_checkpoint": str(artifact)}
        expected_error = ValueError
        message = "gamma mismatch"
    before = artifact.read_bytes()

    with pytest.raises(expected_error, match=message):
        terran_trainer.train_from_config(cfg, seed=1234, device="cpu")

    agent_factory.assert_not_called()
    env_factory.assert_not_called()
    assert artifact.read_bytes() == before
    assert not (tmp_path / "logs").exists()


def test_canonical_terran_rollout_passes_shared_verifier() -> None:
    agent = Agent(
        embedding_dim=32,
        tanh_clipping=10.0,
        n_encode_layers=1,
        device="cpu",
    )
    env = make_terran_env(
        instance=_instance(),
        n_traj=4,
        charging_mode="station_power_full",
        matrix_mode="canonical",
        info_level="full",
        use_jit_mask=False,
    )
    rows = rollout_eval_batch(
        agent,
        [env],
        decode_mode="sample",
        max_steps=32,
        device="cpu",
        seed=41,
        include_routes=True,
    )
    routes = json.loads(rows[0]["routes_json"])
    assert validate_routes(_instance(), routes)["passed"]


def test_terran_rollout_reports_training_budget_exhaustion() -> None:
    agent = Agent(
        embedding_dim=32,
        tanh_clipping=10.0,
        n_encode_layers=1,
        device="cpu",
    )
    env = make_terran_env(
        instance=_instance(),
        n_traj=2,
        charging_mode="station_power_full",
        matrix_mode="canonical",
        info_level="full",
        use_jit_mask=False,
    )
    result = collect_rollout(
        agent, [env], rollout_steps=1, decode_mode="sample", device="cpu", seed=42
    )
    assert result.trajectory_steps.tolist() == [[1, 1]]
    assert result.rollout_budget_exhausted.tolist() == [[True, True]]


def test_terran_rollout_horizon_penalizes_remaining_customers() -> None:
    env = make_terran_env(
        instance=_instance(),
        n_traj=1,
        charging_mode="station_power_full",
        matrix_mode="canonical",
        info_level="full",
        use_jit_mask=False,
        rollout_horizon_steps=1,
        pbrs_config=PotentialRewardConfig(
            use_terminal_heuristic=True,
            failure_penalty=0.5,
        ),
    )
    env.reset(seed=43)
    _, _, terminated, truncated, info = env.step(np.asarray([1]))

    assert not terminated[0]
    assert truncated[0]
    assert info["rollout_budget_exhausted"].tolist() == [True]
    assert info["remaining_customers"].tolist() == [1]
    assert info["remaining_customer_fraction"].tolist() == [0.5]
    assert np.isclose(info["reward_components"]["terminal_heuristic"][0], -0.25)


def test_terminal_task_penalty_has_failure_floor_even_when_all_customers_served() -> None:
    env = make_terran_env(
        instance=_instance(),
        n_traj=1,
        charging_mode="station_power_full",
        matrix_mode="canonical",
        info_level="full",
        use_jit_mask=False,
        invalid_action_penalty=0.0,
        rollout_horizon_steps=2,
        pbrs_config=PotentialRewardConfig(
            use_terminal_task_penalty=True,
            failure_base=2.25,
            unserved_coefficient=1.0,
        ),
    )
    env.reset(seed=143)
    env.step(np.asarray([1]))
    _, _, terminated, truncated, info = env.step(np.asarray([2]))

    assert not terminated[0]
    assert truncated[0]
    assert not info["success"][0]
    assert info["served_customers"].tolist() == [2]
    assert info["failure_reason"].tolist() == [
        "rollout_budget_exhausted_not_returned"
    ]
    components = info["reward_components"]
    assert components["terminal_success_bonus"].tolist() == [0.0]
    assert components["terminal_failure_base"].tolist() == pytest.approx([-2.25])
    assert components["terminal_unserved"].tolist() == [0.0]
    assert components["terminal_task_total"].tolist() == pytest.approx([-2.25])
    np.testing.assert_allclose(
        components["terminal_task_total"],
        components["terminal_success_bonus"]
        + components["terminal_failure_base"]
        + components["terminal_unserved"],
    )
    np.testing.assert_allclose(
        components["shaped"],
        components["base"]
        + components["pbrs_customer"]
        + components["pbrs_repair_distance"]
        + components["pbrs_feasible_ratio"]
        + components["terminal_heuristic"]
        + components["terminal_task_total"],
    )

    # Padding an already finished trajectory must not charge the terminal cost
    # for a second time.
    _, _, _, _, info = env.step(np.asarray([0]))
    assert info["reward_components"]["terminal_task_total"].tolist() == [0.0]
    assert info["reward_components"]["terminal_success_bonus"].tolist() == [0.0]


def test_terminal_task_penalty_separates_floor_and_unserved_components() -> None:
    env = make_terran_env(
        instance=_instance(),
        n_traj=1,
        charging_mode="station_power_full",
        matrix_mode="canonical",
        info_level="full",
        use_jit_mask=False,
        invalid_action_penalty=0.0,
        rollout_horizon_steps=1,
        pbrs_config=PotentialRewardConfig(
            use_terminal_task_penalty=True,
            failure_base=2.25,
            unserved_coefficient=1.0,
        ),
    )
    env.reset(seed=144)
    _, _, _, truncated, info = env.step(np.asarray([1]))

    assert truncated[0]
    assert info["failure_reason"].tolist() == ["rollout_budget_exhausted"]
    components = info["reward_components"]
    assert components["base_non_objective"].tolist() == [0.0]
    assert components["terminal_success_bonus"].tolist() == [0.0]
    assert components["terminal_failure_base"].tolist() == pytest.approx([-2.25])
    assert components["terminal_unserved"].tolist() == pytest.approx([-0.5])
    assert components["terminal_task_total"].tolist() == pytest.approx([-2.75])
    np.testing.assert_allclose(
        components["terminal_task_total"],
        components["terminal_success_bonus"]
        + components["terminal_failure_base"]
        + components["terminal_unserved"],
    )
    np.testing.assert_allclose(
        components["shaped"],
        components["base"]
        + components["pbrs_customer"]
        + components["pbrs_repair_distance"]
        + components["pbrs_feasible_ratio"]
        + components["terminal_heuristic"]
        + components["terminal_task_total"],
    )


def test_terminal_task_success_bonus_is_once_only_and_not_pbrs_annealed() -> None:
    env = make_terran_env(
        instance=_instance(),
        n_traj=1,
        charging_mode="station_power_full",
        matrix_mode="canonical",
        info_level="full",
        use_jit_mask=False,
        invalid_action_penalty=0.0,
        rollout_horizon_steps=3,
        pbrs_config=PotentialRewardConfig(
            use_terminal_task_penalty=True,
            terminal_success_bonus=1.0,
            failure_base=2.25,
            unserved_coefficient=1.0,
        ),
    )
    # The completion reward is a task term and must not follow the auxiliary
    # PBRS annealing scale.
    env.set_reward_scale(0.0)
    env.reset(seed=145)
    env.step(np.asarray([1]))
    env.step(np.asarray([2]))
    _, _, terminated, truncated, info = env.step(np.asarray([0]))

    assert terminated[0]
    assert not truncated[0]
    assert info["success"].tolist() == [True]
    components = info["reward_components"]
    assert components["terminal_success_bonus"].tolist() == [1.0]
    assert components["terminal_failure_base"].tolist() == [0.0]
    assert components["terminal_unserved"].tolist() == [0.0]
    assert components["terminal_task_total"].tolist() == [1.0]
    np.testing.assert_allclose(
        components["terminal_task_total"],
        components["terminal_success_bonus"]
        + components["terminal_failure_base"]
        + components["terminal_unserved"],
    )

    # Padding an already completed trajectory cannot collect a second bonus.
    _, _, _, _, info = env.step(np.asarray([0]))
    assert info["reward_components"]["terminal_success_bonus"].tolist() == [0.0]
    assert info["reward_components"]["terminal_task_total"].tolist() == [0.0]


def test_terran_completion_at_rollout_horizon_gets_success_not_failure() -> None:
    env = make_terran_env(
        instance=_instance(),
        n_traj=1,
        charging_mode="station_power_full",
        matrix_mode="canonical",
        info_level="full",
        use_jit_mask=False,
        rollout_horizon_steps=3,
        pbrs_config=PotentialRewardConfig(
            use_terminal_heuristic=True,
            success_bonus=0.1,
            failure_penalty=0.5,
        ),
    )
    env.reset(seed=44)
    env.step(np.asarray([1]))
    env.step(np.asarray([2]))
    _, _, terminated, truncated, info = env.step(np.asarray([0]))

    assert terminated[0]
    assert not truncated[0]
    assert info["rollout_budget_exhausted"].tolist() == [False]
    assert info["remaining_customers"].tolist() == [0]
    assert np.isclose(info["reward_components"]["terminal_heuristic"][0], 0.1)


def test_terran_station_revisit_resets_only_after_depot() -> None:
    env = make_terran_env(
        instance=_instance(),
        n_traj=1,
        charging_mode="station_power_full",
        matrix_mode="canonical",
        info_level="full",
        use_jit_mask=False,
    )
    observation, _ = env.reset(seed=45)
    station = 3
    assert observation["action_mask"][0, station]

    observation, _, _, _, _ = env.step(np.asarray([station]))
    assert not observation["action_mask"][0, station]
    observation, _, _, _, _ = env.step(np.asarray([1]))
    assert not observation["action_mask"][0, station]
    observation, _, _, _, _ = env.step(np.asarray([0]))
    assert observation["action_mask"][0, station]


def test_pbrs_keeps_distance_as_named_base_reward() -> None:
    env = make_terran_env(
        instance=_instance(),
        n_traj=1,
        charging_mode="station_power_full",
        matrix_mode="canonical",
        info_level="full",
        use_jit_mask=False,
        pbrs_config=PotentialRewardConfig(
            use_customer_pbrs=True,
            use_repair_distance_pbrs=True,
        ),
    )
    observation, _ = env.reset(seed=43)
    customer = int(observation["action_mask"][0].nonzero()[0][0])
    _, reward, _, _, info = env.step([customer])
    components = info["reward_components"]
    assert components["base"][0] < 0.0
    assert components["distance"][0] < 0.0
    assert np.isclose(components["base_non_distance"][0], 0.0)
    assert np.isclose(
        components["base"][0],
        components["distance"][0] + components["base_non_distance"][0],
    )
    assert reward[0] == components["shaped"][0]
    assert env.unwrapped.objective_distance_km[0] > 0.0


def test_reward_components_separate_invalid_penalty_from_distance() -> None:
    env = make_terran_env(
        instance=_instance(),
        n_traj=1,
        charging_mode="station_power_full",
        matrix_mode="canonical",
        info_level="full",
        use_jit_mask=False,
        invalid_action_penalty=-1.0,
        pbrs_config=PotentialRewardConfig(use_customer_pbrs=True),
    )
    env.reset(seed=50)
    _, _, _, truncated, info = env.step(np.asarray([999]))
    components = info["reward_components"]

    assert truncated[0]
    assert components["distance"].tolist() == [0.0]
    assert components["base_non_distance"].tolist() == [-1.0]
    assert components["base"].tolist() == [-1.0]


@pytest.mark.parametrize("gamma", [1.0, 0.999])
@pytest.mark.parametrize("reward_scale", [1.0, 0.2])
@pytest.mark.parametrize("ending", ["success", "horizon_truncation", "invalid_truncation"])
def test_strict_pbrs_uses_zero_terminal_and_telescopes(
    gamma: float, reward_scale: float, ending: str
) -> None:
    actions = {
        "success": (1, 2, 0),
        "horizon_truncation": (1,),
        "invalid_truncation": (1, 999),
    }[ending]
    env = make_terran_env(
        instance=_instance(),
        n_traj=1,
        charging_mode="station_power_full",
        matrix_mode="canonical",
        info_level="full",
        use_jit_mask=False,
        rollout_horizon_steps=1 if ending == "horizon_truncation" else 3,
        pbrs_config=PotentialRewardConfig(
            use_customer_pbrs=True,
            use_repair_distance_pbrs=True,
            gamma=gamma,
            customer_progress_budget=0.5,
            repair_progress_coef=0.5,
        ),
    )
    env.set_reward_scale(reward_scale)
    env.reset(seed=47)
    pbrs_rewards = []
    for action in actions:
        _, _, terminated, truncated, info = env.step(np.asarray([action]))
        components = info["reward_components"]
        pbrs_rewards.append(
            float(
                components["pbrs_customer"][0]
                + components["pbrs_repair_distance"][0]
            )
        )

    assert bool(terminated[0]) == (ending == "success")
    assert bool(truncated[0]) == (ending != "success")
    assert bool(info["success"][0]) == (ending == "success")
    assert bool(info["rollout_budget_exhausted"][0]) == (ending == "horizon_truncation")
    # Phi(initial)=-1 across the two 0.5-budget potentials. Every ending,
    # including failure, has Phi(terminal)=0 and the same scaled offset.
    discounted = sum((gamma**step) * value for step, value in enumerate(pbrs_rewards))
    assert np.isclose(discounted, reward_scale, atol=1e-6)
    # Stepping an already-finished trajectory must not leak potential reward.
    _, _, _, _, info = env.step(np.asarray([0]))
    assert info["reward_components"]["pbrs_customer"].tolist() == [0.0]
    assert info["reward_components"]["pbrs_repair_distance"].tolist() == [0.0]


@pytest.mark.parametrize("gamma", [1.0, 0.999])
def test_terminal_objective_is_not_annealed_with_pbrs(gamma: float) -> None:
    env = make_terran_env(
        instance=_instance(),
        n_traj=1,
        charging_mode="station_power_full",
        matrix_mode="canonical",
        info_level="full",
        use_jit_mask=False,
        rollout_horizon_steps=1,
        pbrs_config=PotentialRewardConfig(
            use_customer_pbrs=True,
            use_repair_distance_pbrs=True,
            use_terminal_heuristic=True,
            gamma=gamma,
            failure_penalty=0.5,
        ),
    )
    env.set_reward_scale(0.2)
    env.reset(seed=48)
    _, _, _, _, info = env.step(np.asarray([1]))
    components = info["reward_components"]

    assert np.isclose(components["terminal_heuristic"][0], -0.25)
    assert np.isclose(
        components["pbrs_customer"][0]
        + components["pbrs_repair_distance"][0],
        0.2,
        atol=1e-6,
    )


def test_reward_diagnostics_match_active_rollout_rewards() -> None:
    agent = Agent(
        embedding_dim=32,
        tanh_clipping=10.0,
        n_encode_layers=1,
        device="cpu",
    )
    env = make_terran_env(
        instance=_instance(),
        n_traj=4,
        charging_mode="station_power_full",
        matrix_mode="canonical",
        info_level="full",
        use_jit_mask=False,
        pbrs_config=PotentialRewardConfig(
            use_customer_pbrs=True,
            use_repair_distance_pbrs=True,
            use_terminal_heuristic=True,
        ),
    )
    batch = collect_rollout(
        agent, [env], rollout_steps=16, decode_mode="sample", device="cpu", seed=49
    )
    diagnostics = batch.reward_diagnostics
    assert diagnostics["active_count"] == int(batch.valid.sum())
    assert np.isclose(
        diagnostics["shaped_sum"],
        float(batch.rewards[batch.valid].sum()),
        atol=1e-6,
    )
    assert np.isclose(
        diagnostics["pbrs_total_sum"],
        diagnostics["shaped_sum"]
        - diagnostics["base_sum"]
        - diagnostics["terminal_heuristic_sum"]
        - diagnostics["terminal_task_total_sum"],
        atol=1e-6,
    )
    assert np.isclose(
        diagnostics["terminal_task_total_sum"],
        diagnostics["terminal_success_bonus_sum"]
        + diagnostics["terminal_failure_base_sum"]
        + diagnostics["terminal_unserved_sum"],
        atol=1e-6,
    )
    assert np.isclose(
        diagnostics["base_sum"],
        diagnostics["distance_sum"] + diagnostics["base_non_distance_sum"],
        atol=1e-6,
    )
    assert np.isclose(
        diagnostics["shaping_total_sum"],
        diagnostics["shaped_sum"]
        - diagnostics["base_sum"]
        - diagnostics["terminal_task_total_sum"],
        atol=1e-6,
    )


def test_formal_pbrs_schedule_reaches_tail_scale_at_minimum_budget() -> None:
    cfg = {
        "pbrs": {
            "use_customer_pbrs": True,
            "annealing": {
                "enabled": True,
                "start_scale": 1.0,
                "end_scale": 0.2,
                "start_epoch": 1,
                "end_epoch": 5000,
                "schedule": "cosine",
            },
        }
    }
    values = [pbrs_scale_for_epoch(cfg, epoch, 10000) for epoch in (1, 300, 2500, 5000, 6000)]
    assert values[0] == 1.0
    assert values[-2:] == [0.2, 0.2]
    assert values == sorted(values, reverse=True)


def test_stage2_terran_config_uses_cus1000_reward_calibration() -> None:
    cfg = terran_trainer.load_config(
        REPO_ROOT
        / "EVRPTW_Benchmark/Reinforcement_Learning/TERRAN/configs/stage2_cus100_terran.yaml"
    )
    assert (
        cfg["env"]["reward_distance_scale_mode"]
        == "single_customer_repair_median"
    )
    assert cfg["env"]["invalid_action_penalty"] == 0.0
    assert cfg["training"]["gamma"] == 1.0
    assert (
        cfg["training"]["reward_contract_id"]
        == "drl_energy_vehicle_reference_scale_v3"
    )
    assert str(cfg["reward_contract"]).endswith(
        "configs/drl_reward_contract_energy_vehicle_v3.json"
    )
    cfg["data"].update(stage2_scale="Cus1000", num_customers=1000)
    terran_trainer._configure_reward_contract(cfg)
    assert cfg["normalization"]["reward_contract_scale"] == "Cus1000"
    expected = cfg["reward_contract"]["scales"]["Cus1000"]
    assert cfg["env"]["reward_objective_scale"] == expected["objective_scale"]
    assert cfg["pbrs"]["failure_base"] == expected["failure_base"]
    assert (
        cfg["pbrs"]["unserved_coefficient"]
        == expected["unserved_coefficient"]
    )
    assert cfg["pbrs"]["annealing"]["end_epoch"] == 5000


@pytest.mark.parametrize("gamma", [1.0, 0.999])
def test_terran_training_env_uses_registered_rollout_horizon(monkeypatch, gamma: float) -> None:
    class Pool:
        def sample(self):
            return _instance()

    captured: list[dict] = []

    monkeypatch.setattr(
        terran_trainer,
        "FixedDatasetInstancePool",
        lambda **_kwargs: Pool(),
    )
    monkeypatch.setattr(
        terran_trainer,
        "make_terran_env",
        lambda **kwargs: captured.append(kwargs) or object(),
    )
    cfg = {
        "data": {
            "train_dataset_path": "unused.jsonl",
            "num_customers": 2,
            "num_charging_stations": 1,
        },
        "training": {
            "num_envs_per_gpu": 2,
            "n_traj": 3,
            "rollout_steps": 140,
            "gamma": gamma,
        },
        "env": {"use_fast_env": False},
        "pbrs": {"use_terminal_heuristic": True, "failure_penalty": 0.5},
    }

    envs, _ = terran_trainer.make_envs(cfg, seed=46)

    assert len(envs) == 2
    assert [call["rollout_horizon_steps"] for call in captured] == [140, 140]
    assert all(call["pbrs_config"].use_terminal_heuristic for call in captured)
    assert all(call["pbrs_config"].failure_penalty == 0.5 for call in captured)
    assert all(call["pbrs_config"].gamma == gamma for call in captured)


def test_fixed_epoch_protocol_does_not_expand_to_a_full_data_pass(
    tmp_path: Path, monkeypatch
) -> None:
    class Pool:
        def __len__(self) -> int:
            return 5_000

    monkeypatch.setattr(terran_protocol, "Stage2TaskPool", lambda **_kwargs: Pool())
    args = SimpleNamespace(
        training_epochs=200,
        data_passes=None,
        stage2_dataset_path=Path("train.parquet"),
        stage2_family_root=Path("families"),
        stage2_scale="Cus100",
        stage2_split_ids="train",
        stage2_track_ids="train",
        output_dir=tmp_path,
        resume=False,
        protocol_id="fixed-epoch-test",
        num_envs_per_gpu=1,
        physical_batch_size=1,
        effective_batch_size=2,
        training_rollout_steps=140,
        seed=1234,
        max_batches_per_pass=None,
        validation_every_passes=5,
        validation_every_epochs=50,
        minimum_training_epochs=100,
        post_minimum_validation_every_epochs=10,
        validation_checkpoints=12,
        early_stop_patience_validations=3,
        early_stop_start_epoch=100,
        validation_dataset_path=Path("val.parquet"),
        validation_family_root=Path("families"),
        validation_limit=5,
        validation_decode_type="sampling",
        validation_candidates=100,
        validation_seed=77,
        training_representation="G",
        euclidean_manifest=None,
        pilot_mode=False,
    )
    configured, meta = terran_protocol.configure_protocol(args, {})
    assert configured["training"]["epochs"] == 200
    assert configured["training"]["checkpoint_interval"] == 50
    assert configured["protocol"]["validation_checkpoints"] == 12
    assert configured["training"]["validation_epochs"][:3] == [50, 100, 110]
    assert configured["training"]["validation_epochs"][-1] == 200
    assert configured["protocol"]["minimum_training_epochs"] == 100
    assert configured["protocol"]["post_minimum_validation_every_epochs"] == 10
    assert configured["protocol"]["budget_mode"] == "fixed_logical_epochs"
    assert configured["protocol"]["epochs_per_pass"] == 200
    assert configured["protocol"]["logical_environments_per_epoch"] == 2
    assert configured["protocol"]["validation_rollout_steps"] == 210
    assert configured["evaluation"]["eval_max_steps"] == 210
    assert configured["training"]["num_envs_per_gpu"] == 1
    assert configured["training"]["logical_microbatches_per_epoch"] == 2
    assert configured["protocol"]["views_per_pass"] == 5_000
    assert meta is not None and meta["physical_batch_size"] == 1
    assert meta["effective_batch_size"] == 2


def test_frozen_protocol_requires_explicit_terminal_success_bonus() -> None:
    args = SimpleNamespace(
        training_epochs=1,
        data_passes=None,
        protocol_id="drl_rq_protocol_frozen_v1",
        terminal_success_bonus=None,
    )

    with pytest.raises(ValueError, match="explicit --terminal-success-bonus"):
        terran_protocol.configure_protocol(args, {})


@pytest.mark.parametrize("gamma", [1.0, 0.999])
def test_terran_effective_batch_two_accumulates_before_optimizer_step(
    tmp_path: Path, monkeypatch, gamma: float
) -> None:
    class Pool:
        sample_count = 0

        def close(self, terminate: bool = False) -> None:
            del terminate

    pool = Pool()
    collect_calls = []
    return_gammas = []
    real_compute_returns = terran_trainer.compute_returns

    def capture_returns(rewards, dones, *, gamma):
        return_gammas.append(gamma)
        return real_compute_returns(rewards, dones, gamma=gamma)

    def fake_collect(_agent, _envs, **_kwargs):
        assert _kwargs["reward_discount_factor"] == gamma
        pool.sample_count += 1
        collect_calls.append(pool.sample_count)
        return SimpleNamespace(
            actions=torch.zeros((1, 1, 1), dtype=torch.long),
            rewards=torch.ones((1, 1, 1)),
            dones=torch.ones((1, 1, 1), dtype=torch.bool),
            values=torch.zeros((1, 1, 1)),
            valid=torch.ones((1, 1, 1), dtype=torch.bool),
            trajectory_steps=torch.ones((1, 1), dtype=torch.int64),
            rollout_budget_exhausted=torch.zeros((1, 1), dtype=torch.bool),
            final_infos=[
                {
                    "success": np.asarray([True]),
                    "objective_distance_km": np.asarray([1.0]),
                    "vehicle_count": np.asarray([1]),
                    "served_customers": np.asarray([1000]),
                }
            ],
            timings={},
        )

    def fake_loss(agent, *_args, **_kwargs):
        loss = agent.weight.sum()
        metric = loss.detach()
        return loss, metric, metric, metric

    monkeypatch.setattr(
        terran_trainer,
        "Agent",
        lambda **_kwargs: torch.nn.Linear(1, 1, bias=False),
    )
    monkeypatch.setattr(
        terran_trainer, "make_envs", lambda _cfg, _seed: ([object()], pool)
    )
    monkeypatch.setattr(terran_trainer, "collect_rollout", fake_collect)
    monkeypatch.setattr(terran_trainer, "compute_returns", capture_returns)
    monkeypatch.setattr(terran_trainer, "evaluate_policy_loss", fake_loss)

    cfg = {
        "run_name": "batch-two-test",
        "output_dir": str(tmp_path),
        "data": {"num_customers": 1000, "num_charging_stations": 200},
        "model": {},
        "env": {},
        "pbrs": {},
        "evaluation": {"eval_interval": 0},
        "training": {
            "gamma": gamma,
            "epochs": 1,
            "num_envs_per_gpu": 1,
            "logical_microbatches_per_epoch": 2,
            "rollout_steps": 1,
            "ppo_update_epochs": 1,
            "num_minibatches": 1,
            "gradient_accumulation_steps": 1,
            "checkpoint_interval": 1,
        },
    }
    checkpoint = terran_trainer.train_from_config(
        cfg, seed=1234, device="cpu"
    )

    assert collect_calls == [1, 2]
    assert return_gammas == [gamma, gamma]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["config"]["training"]["gamma"] == gamma
    optimizer_steps = [
        int(state["step"]) for state in payload["optimizer_state_dict"]["state"].values()
    ]
    assert optimizer_steps == [1]
    with (tmp_path / "logs" / "train_log.csv").open(newline="") as stream:
        row = list(csv.DictReader(stream))[-1]
    assert int(row["samples_seen"]) == 2
    assert int(row["effective_instances_per_optimizer_step"]) == 2


@pytest.mark.parametrize("cost_mode", [False, True])
def test_terran_online_selection_publishes_tail_best_as_formal_aliases(
    tmp_path: Path, monkeypatch, cost_mode: bool,
) -> None:
    class Pool:
        sample_count = 0

        def close(self, terminate: bool = False) -> None:
            del terminate

    pool = Pool()

    def fake_collect(_agent, _envs, **_kwargs):
        pool.sample_count += 1
        return SimpleNamespace(
            actions=torch.zeros((1, 1, 1), dtype=torch.long),
            rewards=torch.ones((1, 1, 1)),
            dones=torch.ones((1, 1, 1), dtype=torch.bool),
            values=torch.zeros((1, 1, 1)),
            valid=torch.ones((1, 1, 1), dtype=torch.bool),
            trajectory_steps=torch.ones((1, 1), dtype=torch.int64),
            rollout_budget_exhausted=torch.zeros((1, 1), dtype=torch.bool),
            final_infos=[
                {
                    "success": np.asarray([True]),
                    "objective_distance_km": np.asarray([1.0]),
                    "vehicle_count": np.asarray([1]),
                    "served_customers": np.asarray([1000]),
                }
            ],
            timings={},
        )

    def fake_loss(agent, *_args, **_kwargs):
        loss = agent.weight.sum()
        metric = loss.detach()
        return loss, metric, metric, metric

    def fake_evaluate(_agent, _cfg, *, seed, epoch, device):
        del seed, device
        return {
            "eval_avg_objective_distance_km": 10.0 - epoch,
            # Epoch 3 improves distance but worsens cost: cost selection must
            # retain epoch 2, unlike the legacy distance-only run.
            **({"eval_avg_objective": [100.0, 90.0, 95.0][epoch - 1]} if cost_mode else {}),
            "eval_feasible_rate": 1.0,
            "eval_num_instances": 2,
            "eval_complete_and_feasible": 2,
            "eval_independent_verifier": True,
            "eval_status": "ok",
        }

    monkeypatch.setattr(
        terran_trainer,
        "Agent",
        lambda **_kwargs: torch.nn.Linear(1, 1, bias=False),
    )
    monkeypatch.setattr(
        terran_trainer, "make_envs", lambda _cfg, _seed: ([object()], pool)
    )
    monkeypatch.setattr(terran_trainer, "collect_rollout", fake_collect)
    monkeypatch.setattr(terran_trainer, "evaluate_policy_loss", fake_loss)
    monkeypatch.setattr(
        terran_trainer, "evaluate_fixed_dataset", fake_evaluate
    )

    cfg = {
        "run_name": "overall-selection-test",
        "output_dir": str(tmp_path),
        "data": {"num_customers": 1000, "num_charging_stations": 200},
        "model": {},
        "env": {},
        "pbrs": {},
        "evaluation": {"eval_interval": 1},
        "training": {
            "epochs": 3,
            "num_envs_per_gpu": 1,
            "logical_microbatches_per_epoch": 1,
            "rollout_steps": 1,
            "ppo_update_epochs": 1,
            "num_minibatches": 1,
            "gradient_accumulation_steps": 1,
            "checkpoint_interval": 1,
            "minimum_training_epochs": 2,
            "validation_epochs": [1, 2, 3],
            "early_stop_patience_validations": 0,
            "early_stop_start_epoch": 2,
        },
    }
    if cost_mode:
        cfg["objective"] = {"mode": "energy_vehicle_cost", "profile_id": "cost-selection-test"}
    terran_trainer.train_from_config(cfg, seed=1234, device="cpu")

    selected = torch.load(
        tmp_path / "checkpoint_selected.pt", map_location="cpu", weights_only=False
    )
    best = torch.load(
        tmp_path / "best.ckpt", map_location="cpu", weights_only=False
    )
    overall = torch.load(
        tmp_path / "best_overall.ckpt", map_location="cpu", weights_only=False
    )
    within = torch.load(
        tmp_path / "best_within_5000.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert selected["epoch"] == best["epoch"] == overall["epoch"] == (2 if cost_mode else 3)
    assert selected["config"]["objective"]["mode"] == ("energy_vehicle_cost" if cost_mode else "distance")
    assert within["epoch"] == 2
    assert json.loads((tmp_path / "validation_summary.json").read_text())[
        "logical_epoch"
    ] == (2 if cost_mode else 3)
    assert json.loads(
        (tmp_path / "validation_summary_within_5000.json").read_text()
    )["logical_epoch"] == 2
    history = [
        json.loads(line)
        for line in (tmp_path / "validation_history.jsonl").read_text().splitlines()
    ]
    assert history[-1]["checkpoint_selected"] is (not cost_mode)
    assert history[-1]["best_within_minimum_selected"] is False


def test_terran_finalizer_runs_full_audit_without_reselecting(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "run"
    checkpoint_dir = output / "checkpoints"
    log_dir = output / "logs"
    checkpoint_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    final_checkpoint = checkpoint_dir / "checkpoint_final.pt"
    torch.save(
        {"label": "final", "config": {"reward_contract": None}},
        final_checkpoint,
    )
    torch.save(
        {"label": "stale-within", "config": {"reward_contract": None}},
        output / "best.ckpt",
    )
    torch.save(
        {"label": "within", "config": {"reward_contract": None}},
        output / "best_within_5000.ckpt",
    )
    torch.save(
        {"label": "overall", "config": {"reward_contract": None}},
        output / "best_overall.ckpt",
    )
    (output / "validation_summary.json").write_text(
        json.dumps(
            {
                "logical_epoch": 50,
                "complete_and_feasible_rate": 0.5,
                "mean_verified_distance_km": 10.0,
            }
        )
    )
    (output / "validation_summary_within_5000.json").write_text(
        json.dumps(
            {
                "logical_epoch": 50,
                "complete_and_feasible_rate": 0.5,
                "mean_verified_distance_km": 10.0,
            }
        )
    )
    (output / "validation_summary_overall.json").write_text(
        json.dumps(
            {
                "logical_epoch": 75,
                "complete_and_feasible_rate": 1.0,
                "mean_verified_distance_km": 8.0,
            }
        )
    )
    with (log_dir / "train_log.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "samples_seen",
                "environment_transitions_total",
                "optimizer_steps_total",
                "epoch_wall_time_s",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "samples_seen": 2,
                "environment_transitions_total": 10,
                "optimizer_steps_total": 3,
                "epoch_wall_time_s": 1.5,
            }
        )

    observed_commands = []

    def fake_run(command, check):
        assert check is True
        observed_commands.append(command)
        audit_dir = output / "validation" / "final_audit"
        audit_dir.mkdir(parents=True)
        with (audit_dir / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["verifier_passed", "objective_distance_km"],
            )
            writer.writeheader()
            writer.writerow(
                {"verifier_passed": "true", "objective_distance_km": 8.0}
            )
            writer.writerow(
                {"verifier_passed": "false", "objective_distance_km": 9.0}
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(terran_protocol.subprocess, "run", fake_run)
    args = SimpleNamespace(
        output_dir=output,
        training_epochs=1,
        data_passes=None,
        final_validation_limit=2,
        validation_dataset_path=Path("val.parquet"),
        validation_family_root=Path("families"),
        validation_limit=1,
        stage2_scale="Cus1000",
        seed=1234,
        device="cpu",
        training_representation="G",
        euclidean_manifest=None,
        max_batches_per_pass=None,
        protocol_id="final-audit-test",
        training_rollout_steps=1200,
        training_stream_path=Path("stream.parquet"),
        validation_every_epochs=50,
        validation_checkpoints=20,
        pilot_mode=False,
    )
    meta = {
        "views_per_pass": 5000,
        "epochs_per_pass": 1,
        "physical_batch_size": 1,
        "effective_batch_size": 2,
    }
    terran_protocol.finalize_protocol(args, final_checkpoint, meta)

    assert len(observed_commands) == 1
    assert observed_commands[0][observed_commands[0].index("--limit") + 1] == "2"
    assert observed_commands[0][observed_commands[0].index("--max-steps") + 1] == "1800"
    audit = json.loads((output / "validation_final_audit.json").read_text())
    assert audit["instances"] == 2
    assert audit["selection_logical_epoch"] == 75
    assert audit["selection_changed"] is False
    assert torch.load(
        output / "checkpoint_selected.pt", map_location="cpu", weights_only=False
    )["label"] == "overall"
    assert torch.load(
        output / "best.ckpt", map_location="cpu", weights_only=False
    )["label"] == "overall"
    assert torch.load(
        output / "best_within_5000.ckpt", map_location="cpu", weights_only=False
    )["label"] == "within"
    assert json.loads((output / "validation_summary.json").read_text())[
        "logical_epoch"
    ] == 75


def test_protocol_resume_carries_exact_optimizer_step_count(
    tmp_path: Path, monkeypatch
) -> None:
    state = DataPassState(
        protocol_id="optimizer-step-test",
        completed_data_passes=1,
        instances_seen=100,
        customer_exposures=10_000,
        optimizer_steps=7,
        environment_transitions=321,
        last_checkpoint="checkpoint_latest.pt",
    )
    state.atomic_write(tmp_path / "data_pass_state.json")
    (tmp_path / "checkpoint_latest.pt").write_bytes(b"checkpoint")

    class Pool:
        def __len__(self) -> int:
            return 100

    monkeypatch.setattr(terran_protocol, "Stage2TaskPool", lambda **_kwargs: Pool())
    args = SimpleNamespace(
        data_passes=2,
        stage2_dataset_path=Path("train.parquet"),
        stage2_family_root=Path("families"),
        stage2_scale="Cus100",
        stage2_split_ids="train",
        stage2_track_ids="train",
        output_dir=tmp_path,
        resume=True,
        protocol_id="optimizer-step-test",
        num_envs_per_gpu=10,
        physical_batch_size=10,
        effective_batch_size=10,
        training_rollout_steps=140,
        seed=1234,
        max_batches_per_pass=None,
        validation_every_passes=5,
    )
    configured, _ = terran_protocol.configure_protocol(args, {})
    assert configured["protocol"]["optimizer_steps"] == 7
    assert configured["protocol"]["completed_samples"] == 100
    assert configured["data"]["stage2_completed_samples"] == 100
    assert configured["training"]["rollout_steps"] == 140
    assert configured["protocol"]["training_rollout_steps"] == 140
    assert configured["protocol"]["validation_rollout_steps"] == 210
