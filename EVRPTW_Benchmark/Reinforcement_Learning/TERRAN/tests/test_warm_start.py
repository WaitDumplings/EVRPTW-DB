from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import protocol
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import train
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import trainer


def _semantic_config(*, scale: str = "Cus2", n_traj: int = 100) -> dict:
    cfg = {
        "data": {
            "stage2_scale": scale,
            "num_customers": int(scale.removeprefix("Cus")),
            "num_charging_stations": 1,
            "stage2_training_representation": "G",
        },
        "model": {
            "embedding_dim": 32,
            "tanh_clipping": 15.0,
            "n_encode_layers": 1,
            "use_graph_token": False,
            "use_dynamic_embedding": False,
        },
        "training": {
            "epochs": 10,
            "gamma": 1.0,
            "reward_contract_id": "warm-start-test-contract",
            "n_traj": n_traj,
            "rollout_steps": 8,
            "learning_rate": 5e-5,
        },
        "pbrs": {
            "use_customer_pbrs": True,
            "customer_progress_budget": 0.5,
            "annealing": {
                "enabled": True,
                "start_scale": 1.0,
                "end_scale": 0.2,
                "start_epoch": 1,
                "end_epoch": 10,
                "schedule": "cosine",
            },
        },
    }
    trainer._freeze_pbrs_reward_semantics(cfg)
    return cfg


def _attach_provenance(cfg: dict, checkpoint: Path, payload: dict) -> None:
    cfg["protocol"] = {
        "planned_training_epochs": cfg["training"]["epochs"] - payload["epoch"],
        "warm_start_checkpoint": str(checkpoint.resolve()),
        "warm_start_epoch_mode": "continue_global",
        "warm_start": {
            "schema": protocol.WARM_START_SCHEMA,
            "epoch_mode": "continue_global",
            "source_checkpoint_path": str(checkpoint.resolve()),
            "source_checkpoint_sha256": hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest(),
            "source_epoch": payload["epoch"],
            "source_seed": payload["seed"],
            "model_state_dict_loaded": True,
            "optimizer_state_dict_loaded": False,
            "optimizer_reset": True,
            "optimizer_name": "adamw",
        },
    }


def test_cli_exposes_weights_only_warm_start_and_rejects_resume_combination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = [
        "terran-train",
        "--config",
        "config.yaml",
        "--seed",
        "1234",
        "--warm-start-checkpoint",
        "source.pt",
        "--learning-rate",
        "0.00005",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = train.parse_args()
    assert args.warm_start_checkpoint == Path("source.pt")
    assert args.warm_start_epoch_mode == "reset"
    assert args.learning_rate == pytest.approx(5e-5)

    monkeypatch.setattr(sys, "argv", [*argv, "--resume"])
    with pytest.raises(SystemExit):
        train.parse_args()


@pytest.mark.parametrize("objective_transition", [False, True])
def test_protocol_counts_only_post_source_stream_and_global_validations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, objective_transition: bool,
) -> None:
    checkpoint = tmp_path / "source.pt"
    torch.save(
        {
            "epoch": 4200 if objective_transition else 1250,
            "seed": 1234,
            "model_state_dict": {"weight": torch.ones(1)},
        },
        checkpoint,
    )

    class Pool:
        def __len__(self) -> int:
            return 5_000

    monkeypatch.setattr(protocol, "Stage2TaskPool", lambda **_kwargs: Pool())
    monkeypatch.setattr(
        protocol,
        "read_stream_view_ids",
        lambda _path: ["view"] * (768_000 if objective_transition else 35_000),
    )
    monkeypatch.setattr(
        protocol, "training_stream_contract_from_args", lambda *_args, **_kwargs: None
    )
    args = SimpleNamespace(
        training_epochs=2_000 if objective_transition else 10_000,
        data_passes=None,
        stage2_dataset_path=Path("train.parquet"),
        stage2_family_root=Path("families"),
        stage2_scale="Cus100" if objective_transition else "Cus1000",
        stage2_split_ids="train",
        stage2_track_ids="train",
        output_dir=tmp_path / "fresh-output",
        resume=False,
        warm_start_checkpoint=checkpoint,
        warm_start_epoch_mode="reset" if objective_transition else "continue_global",
        warm_start_objective_transition=objective_transition,
        protocol_id="warm-start-test",
        terminal_success_bonus=None,
        num_envs_per_gpu=384 if objective_transition else 4,
        physical_batch_size=384 if objective_transition else 4,
        effective_batch_size=384 if objective_transition else 4,
        training_rollout_steps=1250,
        validation_rollout_steps=1875,
        seed=1234,
        max_batches_per_pass=None,
        validation_every_passes=5,
        validation_every_epochs=100 if objective_transition else 250,
        minimum_training_epochs=2000 if objective_transition else 5000,
        post_minimum_validation_every_epochs=50,
        validation_checkpoints=20 if objective_transition else 115,
        early_stop_patience_validations=0 if objective_transition else 10,
        early_stop_start_epoch=0 if objective_transition else 5000,
        validation_dataset_path=Path("val.parquet"),
        validation_family_root=Path("families"),
        validation_limit=500,
        validation_decode_type="sampling",
        validation_candidates=30 if objective_transition else 100,
        validation_seed=77,
        training_representation="G",
        euclidean_manifest=None,
        pilot_mode=False,
        training_stream_path=Path("stream.parquet"),
        training_stream_contract_sha256=None,
        customer_exposure_budget=76_800_000 if objective_transition else 35_000_000,
        final_validation_limit=0,
        exposure_checkpoints="",
        gpu_hour_checkpoints="",
    )

    configured, meta = protocol.configure_protocol(args, {})

    if objective_transition:
        assert configured["training"]["epochs"] == 2000
        assert configured["protocol"]["planned_training_epochs"] == 2000
        assert configured["protocol"]["warm_start_epoch_offset"] == 0
        assert configured["training"]["validation_epochs"] == list(range(100, 2001, 100))
        assert configured["evaluation"]["eval_limit"] == 500
        assert configured["evaluation"]["eval_n_traj"] == 30
        assert configured["protocol"]["completed_samples"] == 0
        assert configured["protocol"]["warm_start"]["critic_reset"] is True
        assert meta["planned_training_epochs"] == 2000
        return
    assert configured["training"]["epochs"] == 10_000
    assert configured["protocol"]["planned_training_epochs"] == 8750
    assert configured["protocol"]["epochs_per_pass"] == 8750
    assert configured["training"]["validation_epochs"][0] == 1500
    assert configured["training"]["validation_epochs"][-1] == 10_000
    assert len(configured["training"]["validation_epochs"]) == 115
    assert configured["protocol"]["warm_start"]["source_epoch"] == 1250
    assert configured["protocol"]["warm_start"]["optimizer_reset"] is True
    assert meta is not None and meta["planned_training_epochs"] == 8750


def test_warm_start_accepts_new_batch_and_lr_but_rejects_semantic_drift(
    tmp_path: Path,
) -> None:
    source_cfg = _semantic_config(n_traj=100)
    source_model = torch.nn.Linear(2, 2)
    payload = {
        "epoch": 4,
        "seed": 1234,
        "config": source_cfg,
        "model_state_dict": source_model.state_dict(),
        "optimizer_state_dict": {"state": {"must_not_be_loaded": True}},
    }
    checkpoint = tmp_path / "source.pt"
    torch.save(payload, checkpoint)

    current = _semantic_config(n_traj=50)
    current["training"]["rollout_steps"] = 7
    current["training"]["learning_rate"] = 1e-5
    _attach_provenance(current, checkpoint, payload)
    assert trainer.validate_warm_start_checkpoint(
        current, payload, checkpoint_path=checkpoint
    ) == 4

    changed_model = deepcopy(current)
    changed_model["model"]["embedding_dim"] = 64
    with pytest.raises(ValueError, match="model configuration mismatch"):
        trainer.validate_warm_start_checkpoint(
            changed_model, payload, checkpoint_path=checkpoint
        )

    changed_scale = deepcopy(current)
    changed_scale["data"]["stage2_scale"] = "Cus3"
    changed_scale["data"]["num_customers"] = 3
    with pytest.raises(ValueError, match="scale configuration mismatch"):
        trainer.validate_warm_start_checkpoint(
            changed_scale, payload, checkpoint_path=checkpoint
        )

    changed_pbrs = deepcopy(current)
    changed_pbrs["pbrs"]["customer_progress_budget"] = 0.7
    trainer._freeze_pbrs_reward_semantics(changed_pbrs)
    with pytest.raises(ValueError, match="PBRS shaping semantics mismatch"):
        trainer.validate_warm_start_checkpoint(
            changed_pbrs, payload, checkpoint_path=checkpoint
        )


def test_weights_only_initialization_resets_adamw_and_continues_global_epoch() -> None:
    source = torch.nn.Linear(2, 1)
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-4)
    source(torch.ones(1, 2)).sum().backward()
    source_optimizer.step()
    assert source_optimizer.state
    payload = {
        "epoch": 1250,
        "model_state_dict": source.state_dict(),
        "optimizer_state_dict": source_optimizer.state_dict(),
    }

    target = torch.nn.Linear(2, 1)
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=5e-5)
    start_epoch = trainer.apply_training_initialization(
        target,
        target_optimizer,
        warm_start_payload=payload,
        warm_start_mode="continue_global",
    )

    assert start_epoch == 1251
    assert target_optimizer.state == {}
    assert target_optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
    for target_parameter, source_parameter in zip(
        target.parameters(), source.parameters()
    ):
        torch.testing.assert_close(target_parameter, source_parameter)


def test_resume_accepts_only_legacy_continue_global_signature_enrichment() -> None:
    saved_cfg = _semantic_config()
    saved_cfg["training"].update(
        {
            "num_envs_per_gpu": 1,
            "logical_microbatches_per_epoch": 1,
            "validation_epochs": [5, 10],
            "minimum_training_epochs": 10,
            "post_minimum_validation_every_epochs": 5,
            "early_stop_patience_validations": 0,
            "early_stop_start_epoch": 0,
            "optimizer": "adamw",
            "weight_decay": 0.01,
        }
    )
    saved_cfg["evaluation"] = {
        "eval_seed": 77,
        "eval_decode_mode": "sample",
        "eval_n_traj": 100,
        "eval_interval": 5,
        "eval_limit": 2,
        "eval_max_steps": 12,
        "eval_batch_size": 1,
        "eval_num_batches": None,
    }
    legacy_provenance = {
        "schema": protocol.WARM_START_SCHEMA,
        "source_checkpoint_path": "/tmp/legacy-source.pt",
        "source_checkpoint_sha256": "a" * 64,
        "source_epoch": 4,
        "source_seed": 1234,
        "model_state_dict_loaded": True,
        "optimizer_state_dict_loaded": False,
        "optimizer_reset": True,
        "optimizer_name": "adamw",
    }
    saved_cfg["protocol"] = {
        "protocol_id": "legacy-warm-start-test",
        "physical_batch_size": 1,
        "effective_batch_size": 1,
        "logical_environments_per_epoch": 1,
        "training_rollout_steps": 8,
        "validation_rollout_steps": 12,
        "minimum_training_epochs": 10,
        "validation_every_epochs": 5,
        "post_minimum_validation_every_epochs": 5,
        "scheduled_validation_epochs": [5, 10],
        "validation_checkpoints": 2,
        "early_stop_patience_validations": 0,
        "early_stop_start_epoch": 0,
        "validation_seed": 77,
        "validation_candidates": 100,
        "validation_decode_type": "sampling",
        "training_stream_contract_sha256": None,
        "warm_start": legacy_provenance,
        "warm_start_checkpoint": "/tmp/legacy-source.pt",
        "planned_training_epochs": 6,
    }
    legacy_signature = trainer._resolved_terran_training_signature(
        saved_cfg, seed=1234
    )
    legacy_signature["method_specific"]["protocol"].pop(
        "warm_start_epoch_mode"
    )
    legacy_signature["sha256"] = trainer.resolved_training_signature_digest(
        legacy_signature
    )
    saved_cfg["protocol"]["resolved_training_signature"] = legacy_signature
    saved_cfg["protocol"]["resolved_training_signature_sha256"] = (
        legacy_signature["sha256"]
    )
    saved_cfg["protocol"]["resolved_training_method_fields"] = legacy_signature[
        "method_specific"
    ]
    payload = {"seed": 1234, "config": saved_cfg}

    current_cfg = deepcopy(saved_cfg)
    current_cfg["protocol"]["warm_start"] = {
        **legacy_provenance,
        "epoch_mode": "continue_global",
        "method": "TERRAN",
        "checkpoint": legacy_provenance["source_checkpoint_path"],
        "epoch_reset": False,
        "data_stream_cursor_reset": True,
        "validation_state_reset": True,
        "early_stop_state_reset": True,
        "source_baseline_evaluated": True,
    }
    current_cfg["protocol"]["warm_start_epoch_mode"] = "continue_global"
    current_cfg["protocol"]["warm_start_epoch_offset"] = 4
    current_cfg["protocol"]["warm_start_checkpoint"] = None
    for field in (
        "resolved_training_signature",
        "resolved_training_signature_sha256",
        "resolved_training_method_fields",
    ):
        current_cfg["protocol"].pop(field)
    trainer._freeze_resolved_terran_training_signature(current_cfg, seed=1234)

    trainer._validate_resume_training_signature(
        current_cfg, payload, current_seed=1234
    )

    changed_cfg = deepcopy(current_cfg)
    changed_cfg["training"]["n_traj"] = 50
    for field in (
        "resolved_training_signature",
        "resolved_training_signature_sha256",
        "resolved_training_method_fields",
    ):
        changed_cfg["protocol"].pop(field)
    trainer._freeze_resolved_terran_training_signature(changed_cfg, seed=1234)
    with pytest.raises(ValueError, match="resolved training signature mismatch"):
        trainer._validate_resume_training_signature(
            changed_cfg, payload, current_seed=1234
        )


@pytest.mark.parametrize("extra", [
    [], ["--warm-start-checkpoint", "source.pt", "--resume"],
    ["--warm-start-checkpoint", "source.pt", "--warm-start-epoch-mode", "continue_global"],
])
def test_objective_transition_cli_requires_fresh_reset(monkeypatch, extra) -> None:
    monkeypatch.setattr(sys, "argv", [
        "terran-train", "--config", "config.yaml", "--seed", "1234",
        "--warm-start-objective-transition", *extra,
    ])
    with pytest.raises(SystemExit):
        train.parse_args()


def _transition_fixture(tmp_path):
    source_cfg = _semantic_config()
    source_cfg["objective"] = {
        "mode": "energy_vehicle_cost", "profile_id": "source-cost",
    }
    source = trainer.Agent(**source_cfg["model"], device="cpu")
    payload = {
        "epoch": 4200, "seed": 1234, "config": source_cfg,
        "model_state_dict": source.state_dict(),
        "optimizer_state_dict": {"state": {"must_not_be_loaded": True}},
    }
    checkpoint = tmp_path / "source.ckpt"
    torch.save(payload, checkpoint)
    target_cfg = deepcopy(source_cfg)
    target_cfg["training"].update(epochs=2000, reward_contract_id="new-cost-contract")
    target_cfg["objective"] = {
        "mode": "energy_vehicle_cost", "profile_id": "target-time-path-cost",
        "objective_distance_source": "running_time_path_distance_km",
    }
    target_cfg["protocol"] = {
        "warm_start_checkpoint": str(checkpoint),
        "warm_start_epoch_mode": "reset", "warm_start_epoch_offset": 0,
        "warm_start": protocol._explicit_warm_start_provenance(
            checkpoint, epoch_mode="reset", objective_transition=True,
            target_objective=target_cfg["objective"],
        ),
    }
    return source, payload, checkpoint, target_cfg


def test_objective_transition_records_source_and_target_and_resets_state(tmp_path) -> None:
    source, payload, checkpoint, target_cfg = _transition_fixture(tmp_path)
    result = trainer.validate_warm_start_contract(
        target_cfg, payload, current_seed=1234, checkpoint_path=checkpoint,
    )
    assert result["source_epoch"] == 4200
    assert result["source_checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert result["target_objective"]["objective_distance_source"] == "running_time_path_distance_km"
    assert result["critic_reset"] is True
    assert result["model_state_dict_loaded"] is False
    target = trainer.Agent(**target_cfg["model"], device="cpu")
    original_critic = {key: value.clone() for key, value in target.critic.state_dict().items()}
    optimizer = torch.optim.AdamW(target.parameters(), lr=5e-5)
    start_epoch = trainer.apply_training_initialization(
        target, optimizer, warm_start_payload=payload, warm_start_mode="reset",
        warm_start_objective_transition=True,
    )
    assert start_epoch == 1
    assert optimizer.state == {}
    for key, value in target.backbone.state_dict().items():
        torch.testing.assert_close(value, source.backbone.state_dict()[key], rtol=0, atol=0)
    for key, value in target.critic.state_dict().items():
        torch.testing.assert_close(value, original_critic[key], rtol=0, atol=0)
    destination = tmp_path / "migrated.ckpt"
    trainer.save_checkpoint(destination, target, optimizer, target_cfg, 0, 1234)
    assert protocol._checkpoint_warm_start_provenance(destination) == target_cfg["protocol"]["warm_start"]
    assert protocol._inherited_warm_start_provenance(destination) == target_cfg["protocol"]["warm_start"]


@pytest.mark.parametrize("mutation,match", [
    ("representation", "scale configuration mismatch"),
    ("architecture", "model architecture mismatch"),
    ("target_objective", "target objective disagrees"),
    ("stable_cost", "requires the legacy actor"),
    ("unrequested", "reward contract mismatch"),
])
def test_objective_transition_retains_compatibility_checks(tmp_path, mutation, match) -> None:
    _, payload, checkpoint, target_cfg = _transition_fixture(tmp_path)
    if mutation == "representation":
        target_cfg["data"]["stage2_training_representation"] = "E"
    elif mutation == "architecture":
        target_cfg["model"]["embedding_dim"] = 64
    elif mutation == "target_objective":
        target_cfg["objective"]["profile_id"] = "undeclared-change"
    elif mutation == "stable_cost":
        target_cfg["training"]["algorithm"] = "stable_cost_v1"
    else:
        target_cfg["protocol"]["warm_start"] = protocol._explicit_warm_start_provenance(
            checkpoint, epoch_mode="reset",
        )
    with pytest.raises(ValueError, match=match):
        trainer.validate_warm_start_contract(
            target_cfg, payload, current_seed=1234, checkpoint_path=checkpoint,
        )


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "shape"])
def test_objective_transition_actor_weights_load_strictly(tmp_path, mutation) -> None:
    _, payload, _, target_cfg = _transition_fixture(tmp_path)
    state = dict(payload["model_state_dict"])
    key = "backbone.embedding.depot_embedding.weight"
    if mutation == "missing":
        del state[key]
    elif mutation == "unexpected":
        state["backbone.unexpected_parameter"] = torch.ones(1)
    else:
        state[key] = torch.ones(1)
    payload["model_state_dict"] = state
    target = trainer.Agent(**target_cfg["model"], device="cpu")
    optimizer = torch.optim.AdamW(target.parameters())
    with pytest.raises(RuntimeError):
        trainer.apply_training_initialization(
            target, optimizer, warm_start_payload=payload, warm_start_mode="reset",
            warm_start_objective_transition=True,
        )


@pytest.mark.parametrize("extra", [
    [], ["--warm-start-checkpoint", "source.pt", "--resume"],
    ["--warm-start-checkpoint", "source.pt", "--warm-start-epoch-mode", "continue_global"],
])
def test_scale_transition_cli_requires_fresh_reset(monkeypatch, extra) -> None:
    monkeypatch.setattr(sys, "argv", [
        "terran-train", "--config", "config.yaml", "--seed", "1234",
        "--warm-start-scale-transition", *extra,
    ])
    with pytest.raises(SystemExit):
        train.parse_args()


def _scale_transition_fixture(tmp_path):
    source, payload, checkpoint, cfg = _transition_fixture(tmp_path)
    cfg["data"].update(stage2_scale="Cus500", num_customers=500, num_charging_stations=50)
    cfg["training"]["epochs"] = 3000
    cfg["protocol"]["warm_start"] = protocol._explicit_warm_start_provenance(
        checkpoint, epoch_mode="reset", objective_transition=True,
        target_objective=cfg["objective"], scale_transition=True, target_scale="Cus500",
    )
    return source, payload, checkpoint, cfg


def test_scale_transition_loads_exact_actor_and_records_new_scale(tmp_path):
    source, payload, checkpoint, cfg = _scale_transition_fixture(tmp_path)
    proof = trainer.validate_warm_start_contract(cfg, payload, current_seed=1234)
    assert proof["source_scale"] == "Cus2"
    assert proof["target_scale"] == "Cus500"
    assert proof["source_epoch"] == 4200
    target = trainer.Agent(**cfg["model"], device="cpu")
    fresh_critic = deepcopy(target.critic.state_dict())
    optimizer = torch.optim.AdamW(target.parameters())
    assert trainer.apply_training_initialization(
        target, optimizer, warm_start_payload=payload, warm_start_mode="reset",
        warm_start_objective_transition=True, warm_start_scale_transition=True,
    ) == 1
    assert not optimizer.state
    for key, value in target.backbone.state_dict().items():
        torch.testing.assert_close(value, source.backbone.state_dict()[key], rtol=0, atol=0)
    for key, value in target.critic.state_dict().items():
        torch.testing.assert_close(value, fresh_critic[key], rtol=0, atol=0)
    destination = tmp_path / "cus500.ckpt"
    trainer.save_checkpoint(destination, target, optimizer, cfg, 0, 1234)
    assert protocol._checkpoint_warm_start_provenance(destination) == cfg["protocol"]["warm_start"]
    assert protocol._inherited_warm_start_provenance(destination) == cfg["protocol"]["warm_start"]


@pytest.mark.parametrize("mutation,match", [
    ("implicit", "scale configuration mismatch"),
    ("domain", "representation mismatch"),
    ("seed", "seed mismatch"),
    ("architecture", "architecture mismatch"),
    ("provenance", "scale transition disagrees"),
    ("reward", "reward contract mismatch"),
])
def test_scale_transition_preserves_other_compatibility_checks(tmp_path, mutation, match):
    _, payload, checkpoint, cfg = _scale_transition_fixture(tmp_path)
    seed = 1234
    if mutation == "implicit":
        cfg["protocol"]["warm_start"] = protocol._explicit_warm_start_provenance(
            checkpoint, epoch_mode="reset", objective_transition=True,
            target_objective=cfg["objective"],
        )
    elif mutation == "domain":
        cfg["data"]["stage2_training_representation"] = "E"
    elif mutation == "seed":
        seed = 99
    elif mutation == "architecture":
        cfg["model"]["embedding_dim"] = 64
    elif mutation == "provenance":
        cfg["protocol"]["warm_start"]["target_scale"] = "Cus1000"
    else:
        cfg["protocol"]["warm_start"]["objective_transition"] = False
    with pytest.raises(ValueError, match=match):
        trainer.validate_warm_start_contract(cfg, payload, current_seed=seed)
