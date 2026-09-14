"""CPU coverage of exact stage continuation, durability and fail-closed guards."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil

import numpy as np
import pytest
import torch

from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.distributed_train import parse_args, prepare_method
from EVRPTW_Benchmark.Reinforcement_Learning.common.distributed import DistributedContext
from EVRPTW_Benchmark.Reinforcement_Learning.common.distributed_protocol import configure_distributed_contract, restore_rank_rng
from EVRPTW_Benchmark.Reinforcement_Learning.common.protocol_trainers import _load_checkpoint, paper_ema_baseline_due, paper_baseline_eval_due
from EVRPTW_Benchmark.Reinforcement_Learning.common.reward_contract import reward_contract_from_args
from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import objective_from_args
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_protocol import freeze_resolved_training_signature, assert_checkpoint_training_signature
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_stream import training_stream_contract_digest
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913 import stage2_transition as transition

CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def source(tmp_path, monkeypatch):
    run = tmp_path / "source"
    run.mkdir()
    args = parse_args([
        "--dataset-path", str(tmp_path / "train.parquet"), "--output-dir", str(run),
        "--scale", "Cus500", "--training-epochs", "10000", "--minimum-training-epochs", "5000",
        "--training-rollout-steps", "1700", "--validation-rollout-steps", "2550",
        "--samples-per-instance", "30", "--batch-size", "24", "--physical-batch-size", "24",
        "--effective-batch-size", "48", "--customer-exposure-budget", "240000000",
        "--expected-world-size", "2", "--distributed-backend", "nccl",
        "--activation-checkpoint-stride", "1", "--training-stream-path", str(tmp_path / "stream.parquet"),
        "--validation-every-epochs", "100", "--validation-checkpoints", "100",
        "--validation-decode-type", "sampling", "--validation-candidates", "30",
        "--objective-config", str(CONFIGS / "rivian_energy_vehicle_cost_v2.json"),
        "--reward-contract", str(CONFIGS / "drl_reward_contract_energy_vehicle_v3.json"),
        "--method-auxiliary-profile", str(CONFIGS / "evrptw_rl_station_auxiliary_v1.json"),
        "--device", "cuda:0",
    ])
    prepare_method(args)
    configure_distributed_contract(args, DistributedContext(rank=0, world_size=2), method="EVRPTW-RL")
    objective = objective_from_args(args)
    reward_contract_from_args(args, objective=objective, scale=args.scale)
    contract = {"schema": "drl_training_stream_contract_v1", "sample_count": 480000,
                "scale": "Cus500", "seed": 1234, "stream_content_sha256": "a" * 64}
    contract["sha256"] = training_stream_contract_digest(contract)
    args.training_stream_contract_snapshot = contract
    args.training_stream_contract_sha256 = contract["sha256"]
    signature = freeze_resolved_training_signature(args)
    # Production is pinned to the real source signature; synthetic fixtures
    # pin their own signed paths, without adding a production override option.
    monkeypatch.setattr(transition, "EXPECTED_SOURCE_SIGNATURE_SHA256", signature["sha256"])
    summaries = [{"logical_epoch": epoch, "instances": 500, "decode_type": "sampling", "candidate_count": 30,
                  "complete_and_feasible_rate": 1.0, "mean_verified_objective": 20.0 + epoch,
                  "objective_mode": "energy_vehicle_cost"} for epoch in (100, 200, 300)]
    actor = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    actor(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    rngs = [{"python": random.Random(rank + 7).getstate(), "numpy": np.random.RandomState(rank + 11).get_state(),
             "torch_cpu": torch.Generator().manual_seed(rank + 21).get_state(),
             "torch_cuda": torch.tensor([rank, 2, 3], dtype=torch.uint8),
             "pool": np.random.default_rng(rank + 31).bit_generator.state} for rank in (0, 1)]
    for epoch in (100, 200, 300):
        opt = deepcopy(optimizer.state_dict())
        for value in opt["state"].values():
            value["step"] = torch.tensor(float(epoch))
        model = deepcopy(actor.state_dict())
        model["weight"].add_(epoch)
        payload = {
            "method": "EVRPTW-RL", "data_pass": 0, "model": model,
            "baseline": deepcopy(actor.state_dict()), "optimizer": opt, "args": deepcopy(vars(args)),
            "protocol_id": args.protocol_id, "objective_config": objective.to_dict(),
            "reward_contract": deepcopy(args.reward_contract_snapshot),
            "method_auxiliary_profile": deepcopy(args.method_auxiliary_snapshot),
            "training_stream_contract": deepcopy(contract), "soft_stage_contract": None,
            "resolved_training_signature": deepcopy(signature), "distributed_contract": deepcopy(args.distributed_contract),
            "logical_epoch": epoch, "stream_cursor": epoch * 48, "rank_rng_states": deepcopy(rngs),
            "baseline_probe_view_ids": [f"train-{index}" for index in range(64)], "ema_cost": 5.5,
            "data_pass_state": {"protocol_id": args.protocol_id, "completed_data_passes": 0,
                                "instances_seen": epoch * 48, "customer_exposures": epoch * 48 * 500,
                                "optimizer_steps": epoch, "environment_transitions": epoch * 12000,
                                "last_checkpoint": str(run / "checkpoint_latest.pt")},
            "best_validation_summary": summaries[0], "best_within_minimum_summary": summaries[0],
            "best_validation_key": [1.0, -120.0], "best_within_minimum_key": [1.0, -120.0],
            "validation_checks_without_improvement": 0, "completed_validation_checks": epoch // 100,
            "baseline_eval_count": 0, "baseline_update_count": 0, "early_stopped": False,
            "pilot_partial_pass": True, "total_wall_time_s": epoch * 22.2, "total_gpu_hours": epoch / 30,
            "warm_start_provenance": None,
        }
        torch.save(payload, run / f"checkpoint_epoch_{epoch:04d}.pt")
    shutil.copy2(run / "checkpoint_epoch_0300.pt", run / "checkpoint_latest.pt")
    (run / "data_pass_state.json").write_text(json.dumps(payload["data_pass_state"]))
    _write_jsonl(run / "logical_epoch_history.jsonl", [
        {"logical_epoch": epoch, "baseline_kind": "paper_ema", "baseline_warmup_synchronized": False,
         "optimizer_steps_total": epoch, "global_stream_cursor": epoch * 48} for epoch in range(1, 302)])
    _write_jsonl(run / "reward_diagnostics.jsonl", [{"logical_epoch": epoch} for epoch in range(1, 302)])
    _write_jsonl(run / "sampled_view_ids.jsonl", [
        {"logical_epoch": epoch, "start_cursor": (epoch - 1) * 48, "end_cursor": epoch * 48,
         "view_ids": [f"sample-{epoch}-{index}" for index in range(48)]} for epoch in range(1, 302)])
    _write_jsonl(run / "validation_history.jsonl", summaries)
    return run


def test_exact_resume_preserves_optimizer_rng_selection_and_handoff(source, tmp_path):
    destination = tmp_path / "destination"
    before = {p.name: transition.sha256_file(p) for p in source.iterdir()}
    report = transition.prepare_stage2_run(source, destination, 300, before["checkpoint_epoch_0300.pt"])
    original = torch.load(source / "checkpoint_epoch_0300.pt", weights_only=False)
    saved = torch.load(destination / "checkpoint_latest.pt", weights_only=False)
    for key in original.keys() - {"baseline", "args", "resolved_training_signature", "data_pass_state"}:
        assert transition.state_equal(original[key], saved[key]), key
    assert transition.state_equal(saved["model"], saved["baseline"])
    assert not transition.state_equal(original["baseline"], saved["baseline"])
    assert saved["data_pass_state"] == {**original["data_pass_state"], "last_checkpoint": str(destination / "checkpoint_latest.pt")}
    args = Namespace(**saved["args"])
    assert args.ema_warmup_steps == 300 and args.resume and args.warm_start_checkpoint is None
    assert args.resolved_training_method_fields["rollout_baseline_warmup_optimizer_updates"] == 300
    assert not paper_ema_baseline_due("EVRPTW-RL", 300, args)
    assert not paper_baseline_eval_due("EVRPTW-RL", 300, args)
    assert paper_baseline_eval_due("EVRPTW-RL", 400, args)
    transition.assert_stage2_launch_args(saved, args)
    # Exercise the actual strict checkpoint loader, including all signed
    # objective, auxiliary, reward and stream contracts plus AdamW moments.
    model, baseline = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loaded = _load_checkpoint(destination / "checkpoint_latest.pt", policy=model, baseline=baseline,
                              optimizer=optimizer, protocol_id=args.protocol_id,
                              objective_config=objective_from_args(args), optimizer_name="adamw",
                              optimizer_weight_decay=args.weight_decay, reward_contract_args=args)
    assert transition.state_equal(optimizer.state_dict(), original["optimizer"])
    assert transition.state_equal(model.state_dict(), baseline.state_dict())
    pool = Namespace(rng=np.random.default_rng())
    restore_rank_rng(loaded["rank_rng_states"][1], pool, "cpu")
    draws = (random.random(), np.random.random(), torch.rand(3), pool.rng.random())
    restore_rank_rng(original["rank_rng_states"][1], pool, "cpu")
    assert transition.state_equal(draws, (random.random(), np.random.random(), torch.rand(3), pool.rng.random()))
    for name in ("checkpoint_epoch_0100.pt", "best.ckpt", "best_overall.ckpt", "checkpoint_selected.pt", "best_within_5000.ckpt"):
        selected = torch.load(destination / name, weights_only=False)
        assert_checkpoint_training_signature(selected, args)
        historical = torch.load(source / "checkpoint_epoch_0100.pt", weights_only=False)
        assert transition.state_equal(selected["baseline"], historical["baseline"])
        assert selected["stage2_transition_provenance"]["historical_checkpoint_compatible_prefix"]
    assert report["optimizer_steps"] == 300 and report["stream_cursor"] == 14400
    assert len((destination / "logical_epoch_history.jsonl").read_text().splitlines()) == 300
    assert len((destination / "source_checkpoint_archive/logical_epoch_history.jsonl").read_text().splitlines()) == 301
    assert before == {p.name: transition.sha256_file(p) for p in source.iterdir()}
    # Idempotence also holds after later training has changed mutable files.
    (destination / "checkpoint_latest.pt").write_bytes(b"later checkpoint")
    assert transition.prepare_stage2_run(source, destination) == report
    assert (destination / "checkpoint_latest.pt").read_bytes() == b"later checkpoint"


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p.update(stream_cursor=14448), "cursor"),
    (lambda p: p["data_pass_state"].update(optimizer_steps=299), "embedded progress"),
    (lambda p: p.update(completed_validation_checks=2), "validation count"),
    (lambda p: p.update(rank_rng_states=p["rank_rng_states"][:1]), "RNG"),
    (lambda p: p["optimizer"]["state"][0].update(step=torch.tensor(299.0)), "optimizer step"),
    (lambda p: p["args"].update(ema_warmup_steps=999), "cutoff"),
    (lambda p: p["args"].update(samples_per_instance=29), "signature"),
    (lambda p: p["distributed_contract"].update(gradient_reduction="mean"), "distributed contract"),
    (lambda p: p.update(baseline_eval_count=1), "baseline evaluations"),
])
def test_malformed_boundary_fails_without_publishing(source, tmp_path, mutation, match):
    path = source / "checkpoint_epoch_0300.pt"
    payload = torch.load(path, weights_only=False)
    mutation(payload)
    torch.save(payload, path)
    shutil.copy2(path, source / "checkpoint_latest.pt")
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match=match):
        transition.prepare_stage2_run(source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".destination.stage2-*"))


def test_missing_validation_or_selected_checkpoint_fails_closed(source, tmp_path):
    history = source / "validation_history.jsonl"
    history.write_text("\n".join(history.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="validation history"):
        transition.prepare_stage2_run(source, tmp_path / "destination")
    (source / "checkpoint_epoch_0100.pt").unlink()
    with pytest.raises(ValueError, match="historical committed checkpoint missing"):
        transition.prepare_stage2_run(source, tmp_path / "destination")


def test_checksum_boundary_existing_directory_and_archive_guards(source, tmp_path):
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="SHA256"):
        transition.prepare_stage2_run(source, destination, expected_source_checkpoint_sha="0" * 64)
    with pytest.raises(ValueError, match="only the audited"):
        transition.prepare_stage2_run(source, destination, boundary_epoch=200)
    with pytest.raises(ValueError, match="must be separate"):
        transition.prepare_stage2_run(source, source)
    destination.mkdir()
    with pytest.raises(ValueError, match="without a completed"):
        transition.prepare_stage2_run(source, destination)
    destination.rmdir()
    transition.prepare_stage2_run(source, destination)
    (destination / "source_checkpoint_archive/data_pass_state.json").write_text("corrupt")
    with pytest.raises(ValueError, match="archived source artifact SHA256"):
        transition.prepare_stage2_run(source, destination)


def test_incomplete_latest_publication_and_ahead_sidecar_rejected(source, tmp_path):
    shutil.copy2(source / "checkpoint_epoch_0200.pt", source / "checkpoint_latest.pt")
    with pytest.raises(ValueError, match="latest checkpoint"):
        transition.prepare_stage2_run(source, tmp_path / "destination")
    shutil.copy2(source / "checkpoint_epoch_0300.pt", source / "checkpoint_latest.pt")
    path = source / "data_pass_state.json"
    sidecar = json.loads(path.read_text())
    sidecar["optimizer_steps"] = 301
    path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="sidecar is ahead"):
        transition.prepare_stage2_run(source, tmp_path / "destination")
