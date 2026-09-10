"""CPU/Gloo checks for the same frozen logical PPO batch on 1/2/4 ranks."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import stable_trainer
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.distributed import DistributedContext
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.stable_trainer import (
    StableState, config_signature, optimize_rollouts, resolve_stable_config,
    validate_resume_checkpoint,
)
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.tests.test_stable_cost_training import (
    _agent, _batch, _cfg, _optimizers, _refresh_old_predictions, _resume_payload,
    _slice_instances,
)


def _initialize(rank, world_size, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world_size,
                            init_method=f"file://{directory}/group", timeout=timedelta(seconds=90))


def _assert_nested_equal(left, right, *, atol=2e-6, rtol=3e-4):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key], atol=atol, rtol=rtol)
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b, atol=atol, rtol=rtol)
    elif isinstance(left, float):
        assert left == pytest.approx(right, abs=atol, rel=rtol)
    else:
        assert left == right


def _optimization_worker(rank, world_size, directory):
    _initialize(rank, world_size, directory)
    try:
        payload = torch.load(Path(directory) / "input.pt", weights_only=False)
        agent = _agent()
        agent.load_state_dict(payload["model"])
        optimizers = _optimizers(agent)
        for optimizer, saved in zip(optimizers, payload["optimizers"]):
            optimizer.load_state_dict(saved)
        state = StableState(**payload["state"])
        metrics = optimize_rollouts(agent, *optimizers, [payload["records"][rank]], payload["cfg"], state)
        result = {"model": agent.state_dict(), "optimizers": [o.state_dict() for o in optimizers],
                  "state": asdict(state), "metrics": metrics}
        torch.save(result, Path(directory) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size,backtrack", [(2, False), (4, True)])
def test_distributed_update_matches_single_logical_batch_with_unequal_lengths(tmp_path, world_size, backtrack):
    torch.set_num_threads(1)
    agent, cfg = _agent(), _cfg()
    optimizers = _optimizers(agent)
    state = StableState(phase="cost", lambda_usd=200)
    # Start from a real Adam/PopArt history, so reduction and output-coordinate
    # changes must preserve resumed optimizer state on every rank.
    optimize_rollouts(agent, *optimizers, [_batch(agent)], cfg, state)
    base = _refresh_old_predictions(agent, _batch(agent))
    records = []
    for rank in range(world_size):
        record = _slice_instances(base, rank % 2, rank % 2 + 1)
        valid = record.valid.clone()
        if rank % 2:
            valid[-1] = False  # Different rank-level active state counts.
        records.append(replace(record, valid=valid,
                               cost_returns=record.cost_returns * (1 + rank * 0.2) + rank * 15))
    if backtrack:
        cfg["stable_cost"].update(target_kl=1e-8, actor_kl_backtracks=2)
        optimizers[0].param_groups[0]["lr"] = 0.02
    payload = {"model": deepcopy(agent.state_dict()),
               "optimizers": [deepcopy(o.state_dict()) for o in optimizers],
               "state": asdict(state), "records": records, "cfg": cfg}
    torch.save(payload, tmp_path / "input.pt")
    metrics = optimize_rollouts(agent, *optimizers, records, cfg, state)
    expected = {"model": agent.state_dict(), "optimizers": [o.state_dict() for o in optimizers],
                "state": asdict(state), "metrics": metrics}
    if backtrack:
        assert metrics["actor_backtracks"] > 0
    else:
        assert metrics["ppo_updates"] > 0
    mp.spawn(_optimization_worker, args=(world_size, str(tmp_path)), nprocs=world_size, join=True)
    ranks = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=False) for rank in range(world_size)]
    for result in ranks:
        _assert_nested_equal(result, expected)
        # All synchronized replicas and Adam states agree exactly with each
        # other, beyond the floating-point tolerance against a single reduction.
        _assert_nested_equal(result, ranks[0], atol=0, rtol=0)


def _error_worker(rank, world_size, directory):
    _initialize(rank, world_size, directory)
    context = DistributedContext.current()
    messages = []
    try:
        parameters = [torch.nn.Parameter(torch.zeros(2)) for _ in range(3)]
        parameters[rank].grad = torch.full((2,), rank + 1.0)
        context.sum_gradients(parameters)
        torch.testing.assert_close(parameters[0].grad, torch.ones(2))
        torch.testing.assert_close(parameters[1].grad, torch.full((2,), 2.0))
        assert parameters[2].grad is None
        from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.models.stable_cost_critic import PopArtHead
        head = PopArtHead(2)
        targets = torch.tensor([10.0, 20.0]) if rank == 0 else torch.empty(0)
        head.update_stats(targets, distributed=context)
        assert head.mean.item() == 15.0
        assert head.std.item() == 5.0
        assert head.sample_count.item() == 2
        with pytest.raises(RuntimeError, match="rank 1"):
            head.update_stats(torch.tensor([float("nan") if rank else 1.0]), distributed=context)
        with pytest.raises(RuntimeError, match="rank 1") as error:
            with context.local_phase("synthetic failure"):
                if rank == 1:
                    raise ValueError("bad local rollout")
        messages.append(str(error.value))
        with pytest.raises(RuntimeError, match="rank-zero operation failed") as error:
            context.main_call(lambda: (_ for _ in ()).throw(ValueError("bad validation")))
        messages.append(str(error.value))
        (Path(directory) / f"error-{rank}.json").write_text(json.dumps(messages))
    finally:
        dist.destroy_process_group()


def test_rank_local_and_rank_zero_errors_propagate_to_all_workers(tmp_path):
    mp.spawn(_error_worker, args=(2, str(tmp_path)), nprocs=2, join=True)
    messages = [json.loads((tmp_path / f"error-{rank}.json").read_text()) for rank in range(2)]
    assert messages[0] == messages[1]


def _ownership_worker(rank, world_size, directory):
    _initialize(rank, world_size, directory)
    from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import trainer
    try:
        payload = torch.load(Path(directory) / "source.pt", weights_only=False)
        cfg = deepcopy(payload["config"])
        cfg["training"].update(distributed_world_size=world_size, num_envs_per_gpu=1,
                                effective_batch_size=world_size, resume_checkpoint=str(Path(directory) / "source.pt"),
                                allow_batch_resize_resume=True)
        cfg["output_dir"] = str(Path(directory) / "run")
        seen = {"checkpoint_calls": 0, "evaluation_calls": 0, "runtime_rank": None}
        created = []
        real_agent = stable_trainer.Agent
        real_checkpoint = stable_trainer._atomic_checkpoint

        def agent_factory(*args, **kwargs):
            result = real_agent(*args, **kwargs)
            created.append(result)
            return result

        def fake_envs(actual, seed):
            assert actual["data"]["stage2_completed_samples"] == 113
            seen["runtime_rank"] = actual["data"]["stage2_sampling_rank"]
            return [], SimpleNamespace(global_committed_cursor=113 + world_size,
                                        drain_sampled_view_ids=lambda: [f"rank-{rank}"], close=lambda: None)

        def fake_collect(agent, *args, **kwargs):
            return _slice_instances(_batch(agent), rank % 2, rank % 2 + 1)

        def checkpoint(path, data):
            assert rank == 0
            seen["checkpoint_calls"] += 1
            real_checkpoint(path, data)

        def evaluate(*args, **kwargs):
            assert rank == 0
            seen["evaluation_calls"] += 1
            return {"complete_and_feasible_rate": 1.0, "mean_verified_objective_cost_usd": 1234.0}

        stable_trainer.Agent = agent_factory
        stable_trainer.collect_rollout = fake_collect
        stable_trainer._atomic_checkpoint = checkpoint
        trainer.make_envs = fake_envs
        trainer.evaluate_fixed_dataset = evaluate
        trainer.validation_summary_from_eval_row = lambda row, **kwargs: row
        stable_trainer.train_stable_cost(cfg, seed=1234, device="cpu")
        torch.save({"seen": seen, "model": created[0].state_dict()}, Path(directory) / f"ownership-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_single_gpu_resume_multi_rank_checkpoint_validation_and_global_state(tmp_path):
    agent, cfg = _agent(), _cfg()
    cfg["evaluation"] = {"eval_interval": 1}
    optimizers = _optimizers(agent)
    state = StableState(phase="cost", lambda_usd=200, epoch=1, sample_count=113,
                        transitions=100, best_feasible_rate=-1)
    optimize_rollouts(agent, *optimizers, [_batch(agent)], cfg, state)
    payload = {**_resume_payload(cfg), "stable_state": asdict(state),
               "model_state_dict": agent.state_dict(), "optimizer_state_dict": optimizers[0].state_dict(),
               "critic_optimizer_state_dict": optimizers[1].state_dict()}
    torch.save(payload, tmp_path / "source.pt")
    mp.spawn(_ownership_worker, args=(2, str(tmp_path)), nprocs=2, join=True)
    results = [torch.load(tmp_path / f"ownership-{rank}.pt", weights_only=False) for rank in range(2)]
    assert results[0]["seen"]["evaluation_calls"] == 1
    assert results[0]["seen"]["checkpoint_calls"] >= 2
    assert results[1]["seen"]["evaluation_calls"] == results[1]["seen"]["checkpoint_calls"] == 0
    assert [r["seen"]["runtime_rank"] for r in results] == [0, 1]
    _assert_nested_equal(results[0]["model"], results[1]["model"], atol=0, rtol=0)
    run = tmp_path / "run"
    saved = torch.load(run / "checkpoint_latest.pt", weights_only=False)
    assert saved["stable_state"]["epoch"] == 2
    assert saved["stable_state"]["sample_count"] == 115
    assert saved["stable_state"]["best_cost"] == 1234
    assert saved["stable_state"]["transitions"] > 100
    assert saved["resume"]["optimizer_reset"] is False
    assert saved["resume"]["old_batch_geometry"]["distributed_world_size"] == 1
    assert saved["resume"]["new_batch_geometry"]["distributed_world_size"] == 2
    assert saved["training_signature"] == config_signature(saved["config"])
    assert not any(key.startswith("stage2_sampling_") for key in saved["config"]["data"])
    assert len((run / "metrics.jsonl").read_text().splitlines()) == 1
    assert len((run / "validation_history.jsonl").read_text().splitlines()) == 1
    ids = json.loads((run / "sampled_view_ids.jsonl").read_text())
    assert ids == {"epoch": 2, "start_cursor": 113, "end_cursor": 115, "view_ids": ["rank-0", "rank-1"]}


def test_world_size_is_explicit_and_part_of_effective_batch_and_resume_geometry():
    original = _cfg()
    assert "distributed_world_size" not in original["training"]
    resized = deepcopy(original)
    resized["training"].update(distributed_world_size=4, effective_batch_size=8)
    resolved = resolve_stable_config(resized)
    assert resolved["training"]["effective_batch_size"] == 8
    with pytest.raises(ValueError, match="explicit allow_batch_resize_resume"):
        validate_resume_checkpoint(_resume_payload(original), resolved, seed=1234, source="source.pt")
    resolved["training"].update(allow_batch_resize_resume=True, resume_checkpoint="source.pt")
    resume = validate_resume_checkpoint(_resume_payload(original), resolved, seed=1234, source="source.pt")
    assert resume["changed_batch_fields"]["distributed_world_size"] == {"old": 1, "new": 4}
    assert resume["optimizer_reset"] is False
    bad = deepcopy(resized)
    bad["training"]["effective_batch_size"] = 2
    with pytest.raises(ValueError, match="world size"):
        resolve_stable_config(bad)
    with pytest.raises(ValueError, match="initialized process group"):
        stable_trainer.train_stable_cost(resolved, seed=1234, device="cpu")


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "4"])
def test_invalid_world_size_is_rejected(value):
    cfg = _cfg()
    cfg["training"]["distributed_world_size"] = value
    with pytest.raises(ValueError, match="positive integer"):
        resolve_stable_config(cfg)


def test_explicit_dataset_relocation_keeps_original_provenance_and_training_semantics():
    old = _cfg()
    old["data"].update(stage2_dataset_path="/old/train.json", stage2_family_root="/old/data",
                        training_index_sha256="matching-frozen-index-hash")
    old["evaluation"] = {"validation_index_sha256": "matching-validation-index"}
    new = deepcopy(old)
    new["data"].update(stage2_dataset_path="/new/train.json", stage2_family_root="/new/data")
    new["training"].update(resume_checkpoint="source.pt", allow_dataset_relocation_resume=True)
    payload = _resume_payload(old)
    result = validate_resume_checkpoint(payload, new, seed=1234, source="source.pt")
    assert set(result["relocated_dataset_paths"]) == {"stage2_dataset_path", "stage2_family_root"}
    assert result["optimizer_reset"] is False
    assert not result["changed_batch_fields"]
    changed = deepcopy(new)
    changed["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="permits only"):
        validate_resume_checkpoint(payload, changed, seed=1234, source="source.pt")
    changed = deepcopy(new)
    changed["data"]["training_index_sha256"] = "different"
    with pytest.raises(ValueError, match="matching nonempty"):
        validate_resume_checkpoint(payload, changed, seed=1234, source="source.pt")
    changed = deepcopy(new)
    changed["evaluation"]["validation_index_sha256"] = "different-validation-index"
    with pytest.raises(ValueError, match="matching validation_index_sha256"):
        validate_resume_checkpoint(payload, changed, seed=1234, source="source.pt")
    payload["config"]["data"]["stage2_family_root"] = "/tampered"
    with pytest.raises(ValueError, match="original configuration signature"):
        validate_resume_checkpoint(payload, new, seed=1234, source="source.pt")
