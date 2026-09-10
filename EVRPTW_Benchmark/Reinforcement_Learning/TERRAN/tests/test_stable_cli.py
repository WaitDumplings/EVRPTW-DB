from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import stable_cli


def _arguments(tmp_path: Path, customers: int = 500) -> argparse.Namespace:
    root = tmp_path / "dataset"
    (root / "materialized/families").mkdir(parents=True, exist_ok=True)
    for split, track in [("train", "train"), ("val", "validation")]:
        path = root / f"generation_plan/core/{split}/view_index.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"view_id": [f"{split}-a", f"{split}-b"],
                      "customer_count": [customers] * 2,
                      "charging_station_count": [20 if customers <= 100 else 50] * 2,
                      "split_id": [split] * 2, "track_id": [track] * 2}).to_parquet(path)
    return argparse.Namespace(scale=f"Cus{customers}", config=None, dataset_root=str(root),
                              output_dir=str(tmp_path / "run"), seed=1234, gpu=None,
                              epochs=2, physical_batch_size=None, effective_batch_size=None,
                              n_traj=None, rollout_steps=None, ppo_step_chunk_size=None,
                              warm_start_checkpoint=None, resume=None)


@pytest.mark.parametrize("customers,microbatches", [(100, 2), (500, 2), (1000, 4)])
def test_prepare_uses_real_scale_and_independent_sampling(tmp_path, customers, microbatches):
    args = _arguments(tmp_path, customers)
    path = stable_cli.prepare(args)
    cfg = yaml.safe_load(path.read_text())
    provenance = json.loads((path.parent / "provenance.json").read_text())
    assert cfg["training"]["logical_microbatches_per_epoch"] == microbatches
    assert cfg["training"]["effective_batch_size"] == (128 if customers == 100 else 256)
    assert cfg["data"]["num_charging_stations"] == (20 if customers == 100 else 50)
    assert cfg["stable_cost"]["popart_min_std"] == 1.0
    assert cfg["data"]["training_index_sha256"] == provenance["train_index"]["sha256"]
    assert not cfg.get("reward_contract")
    assert not cfg["data"].get("stage2_training_stream_path")
    assert provenance["sampling"]["uses_old_registered_stream"] is False
    with pytest.raises(FileExistsError):
        stable_cli.prepare(args)


def test_invalid_batch_is_rejected_before_creating_output(tmp_path):
    args = _arguments(tmp_path)
    args.effective_batch_size = 129
    with pytest.raises(ValueError, match="integer multiple"):
        stable_cli.prepare(args)
    assert not Path(args.output_dir).exists()


def test_batch_resume_prepare_checks_signature_and_records_explicit_exception(tmp_path):
    from dataclasses import asdict
    import torch
    from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.stable_trainer import (
        SCHEMA, StableState, config_signature, resolve_stable_config,
    )

    args = _arguments(tmp_path)
    cfg, _ = stable_cli.resolve_config(args)
    cfg = resolve_stable_config(cfg)
    # A resolved config from an actor-warmstarted run must become a full resume.
    cfg["training"]["actor_warm_start"] = str(tmp_path / "old-actor.ckpt")
    cfg["data"]["stage2_completed_samples"] = 31
    source_config = tmp_path / "old-resolved.yaml"
    source_config.write_text(yaml.safe_dump(cfg))
    source_checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"schema": SCHEMA, "seed": 1234, "config": cfg,
                "training_signature": config_signature(cfg),
                "stable_state": asdict(StableState(epoch=1, sample_count=128))}, source_checkpoint)
    args.config, args.resume = str(source_config), str(source_checkpoint)
    args.physical_batch_size = cfg["training"]["num_envs_per_gpu"] * 2
    args.effective_batch_size = cfg["training"]["effective_batch_size"] * 2
    with pytest.raises(ValueError, match="explicit allow_batch_resize_resume"):
        stable_cli.prepare(args)
    assert not Path(args.output_dir).exists()
    args.allow_batch_resize_resume = True
    path = stable_cli.prepare(args)
    resumed = yaml.safe_load(path.read_text())
    assert resumed["training"]["allow_batch_resize_resume"] is True
    assert "actor_warm_start" not in resumed["training"]
    provenance = json.loads((path.parent / "provenance.json").read_text())
    assert provenance["resume"]["source_checkpoint"] == str(source_checkpoint)
    assert provenance["resume"]["source_sample_count"] == 128
    assert provenance["resume"]["new_batch_geometry"]["effective_batch_size"] == args.effective_batch_size


def test_batch_resize_flag_without_resume_rejected_before_output(tmp_path):
    args = _arguments(tmp_path)
    args.allow_batch_resize_resume = True
    with pytest.raises(ValueError, match="requires --resume"):
        stable_cli.prepare(args)
    assert not Path(args.output_dir).exists()


def test_unlisted_scale_uses_bucket_profile_when_dataset_supports_it(tmp_path):
    args = _arguments(tmp_path, customers=50)
    path = stable_cli.prepare(args)
    cfg = yaml.safe_load(path.read_text())
    provenance = json.loads((path.parent / "provenance.json").read_text())
    assert cfg["data"]["stage2_scale"] == "Cus50"
    assert cfg["data"]["num_customers"] == 50
    assert cfg["data"]["num_charging_stations"] == 20
    assert cfg["training"]["num_envs_per_gpu"] == 64
    assert cfg["training"]["rollout_steps"] == 95
    assert cfg["evaluation"]["eval_max_steps"] == 95
    assert provenance["profile_path"].endswith("cus100.yaml")
    assert provenance["uses_bucket_profile"] is True
    assert stable_cli.parse_scale("cus50") == "Cus50"


def test_scale_must_exist_in_selected_dataset(tmp_path):
    args = _arguments(tmp_path, customers=100)
    args.scale = "Cus50"
    with pytest.raises(ValueError, match="available customer counts: \\[100\\]"):
        stable_cli.prepare(args)
    assert not Path(args.output_dir).exists()


def test_modified_validation_index_blocks_prepared_run(tmp_path):
    args = _arguments(tmp_path)
    path = stable_cli.prepare(args)
    val_path = Path(args.dataset_root) / "generation_plan/core/val/view_index.parquet"
    val_path.write_bytes(val_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="evaluation index changed"):
        stable_cli.run(path, device="cpu", gpu=None)
    assert not (path.parent / "run_state.json").exists()


def test_launch_prepared_config_preserves_selected_gpu_and_new_run(tmp_path, monkeypatch):
    args = _arguments(tmp_path)
    path = stable_cli.prepare(args)
    calls = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return argparse.Namespace(pid=987654321)

    monkeypatch.setattr(stable_cli.subprocess, "Popen", fake_popen)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "3")
    monkeypatch.setattr(sys, "argv", ["stable_cli", "launch", "--resolved-config", str(path), "--gpu", "1"])
    stable_cli.main()
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:4] == [sys.executable, "-u", "-m", stable_cli.MODULE]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
    assert kwargs["env"]["MKL_NUM_THREADS"] == "3"
    assert kwargs["start_new_session"] is True
    assert json.loads((path.parent / "launch_process.json").read_text())["pid"] == 987654321
    assert stable_cli.status(path.parent)["process_matches_run"] is False
    with pytest.raises(FileExistsError):
        stable_cli.main()
    assert len(calls) == 1


@pytest.mark.parametrize("customers,physical,microbatches", [(500, 32, 2), (1000, 8, 8)])
def test_four_gpu_profiles_keep_global_batch_and_trajectory_count(tmp_path, customers, physical, microbatches):
    args = _arguments(tmp_path, customers)
    args.machine_profile = "4x2080ti"
    args.gpus = (0, 1, 2, 3)
    path = stable_cli.prepare(args)
    cfg = yaml.safe_load(path.read_text())
    t = cfg["training"]
    assert t["distributed_world_size"] == 4
    assert t["num_envs_per_gpu"] == physical
    assert t["logical_microbatches_per_epoch"] == microbatches
    assert physical * microbatches * 4 == t["effective_batch_size"] == 256
    assert t["n_traj"] == 16
    assert cfg["evaluation"]["eval_n_traj"] == 100
    assert cfg["model"]["critic_mode"] == "stable_cost_v1"
    metadata = json.loads((path.parent / "provenance.json").read_text())
    assert metadata["world_size"] == 4
    assert metadata["gpus"] == [0, 1, 2, 3]


def test_distributed_launch_uses_torchrun_and_clears_parent_rank_environment(tmp_path, monkeypatch):
    args = _arguments(tmp_path)
    args.world_size = 4
    args.physical_batch_size = 32
    path = stable_cli.prepare(args)
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("MASTER_PORT", "12345")
    command, environment = stable_cli.launch_command(path, device="cuda", gpu=None, gpus=(3, 2, 1, 0))
    assert command[:4] == [sys.executable, "-u", "-m", "torch.distributed.run"]
    assert "--nproc_per_node=4" in command
    assert "--standalone" in command
    assert "--max_restarts=0" in command
    assert environment["CUDA_VISIBLE_DEVICES"] == "3,2,1,0"
    assert all(key not in environment for key in ("RANK", "WORLD_SIZE", "MASTER_PORT"))
    with pytest.raises(ValueError, match="match the prepared world size"):
        stable_cli.launch_command(path, device="cuda", gpu=None, gpus=(0, 1))
    with pytest.raises(ValueError, match="selects one worker"):
        stable_cli.launch_command(path, device="cuda", gpu=0, gpus=None)


def test_preparation_rejects_nondivisible_global_batch_and_gpu_count_mismatch(tmp_path):
    args = _arguments(tmp_path)
    args.world_size = 3
    args.physical_batch_size = 32
    with pytest.raises(ValueError, match="physical batch times world size"):
        stable_cli.prepare(args)
    args.gpus = (0, 1, 2, 3)
    with pytest.raises(ValueError, match="number of --gpus"):
        stable_cli.prepare(args)
    assert not Path(args.output_dir).exists()


def test_direct_run_rejects_missing_distributed_workers_before_creating_state(tmp_path, monkeypatch):
    args = _arguments(tmp_path)
    args.world_size = 4
    args.physical_batch_size = 32
    path = stable_cli.prepare(args)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    with pytest.raises(ValueError, match="requires 4 workers, got 1"):
        stable_cli.run(path, device="cpu", gpu=None)
    assert not (path.parent / "run_state.json").exists()


@pytest.mark.parametrize("value", ["", "0,", "-1,0", "0,0", "a,b"])
def test_gpu_list_rejects_invalid_or_duplicate_ids(value):
    with pytest.raises(argparse.ArgumentTypeError):
        stable_cli.parse_gpus(value)
