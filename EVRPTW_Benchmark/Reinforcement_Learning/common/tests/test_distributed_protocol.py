from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist

from EVRPTW_Benchmark.Reinforcement_Learning.common import distributed_protocol as protocol
from EVRPTW_Benchmark.Reinforcement_Learning.common.distributed import DistributedContext
from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import ObjectiveConfig
from EVRPTW_Benchmark.Reinforcement_Learning.common.tests.test_distributed_helpers import (
    init_gloo,
    run_gloo_workers,
)


@pytest.mark.parametrize("world_size", [1, 2, 3])
def test_stream_shards_partition_each_global_epoch_without_overlap(world_size):
    physical = 2
    effective = physical * world_size * 2
    view_ids = [f"view-{i}" for i in range(3 * effective)]
    for epoch in range(1, 4):
        shards = [protocol.shard_stream_epoch(
            view_ids, logical_epoch=epoch, physical_batch_size=physical,
            effective_batch_size=effective, rank=rank, world_size=world_size,
        ) for rank in range(world_size)]
        assert all(len(shard) == 2 for shard in shards)
        assert all(len(batch) == physical for shard in shards for batch in shard)
        restored = [view_id for micro in range(2) for rank in range(world_size)
                    for view_id in shards[rank][micro]]
        assert restored == view_ids[(epoch - 1) * effective:epoch * effective]
        assert len(set(restored)) == effective


@pytest.mark.parametrize("changes", [
    {"logical_epoch": 0}, {"logical_epoch": 3}, {"rank": 2},
    {"world_size": 0}, {"physical_batch_size": 0}, {"effective_batch_size": 7},
])
def test_stream_shards_reject_invalid_or_partial_global_batches(changes):
    options = dict(logical_epoch=1, physical_batch_size=2,
                   effective_batch_size=8, rank=0, world_size=2)
    options.update(changes)
    with pytest.raises(ValueError):
        protocol.shard_stream_epoch([str(i) for i in range(16)], **options)


def _summary(rows):
    passed = [row for row in rows if row["verifier_passed"]]
    return {
        "schema": "drl_validation_summary_v1", "instances": len(rows),
        "complete_and_feasible": len(passed),
        "complete_and_feasible_rate": len(passed) / max(1, len(rows)),
        "verifier_summary_passed": bool(rows) and len(passed) == len(rows),
        "objective_mode": "distance", "objective_unit": "km",
        "objective_config": ObjectiveConfig(mode="distance").to_dict(),
        "mean_verified_distance_km": float(np.mean([r["objective_distance_km"] for r in passed])) if passed else None,
        "mean_verified_objective": float(np.mean([r["objective_distance_km"] for r in passed])) if passed else None,
        "mean_verified_cost_usd": None,
        "rows": rows,
    }


def test_validation_merge_weights_successful_instances_and_handles_empty_rank():
    rows = [dict(instance_id=f"v{i}", verifier_passed=passed, objective_distance_km=value)
            for i, (passed, value) in enumerate([(True, 2), (False, 999), (True, 6), (True, 10), (False, 999)])]
    merged = protocol.merge_validation_summaries([
        _summary(rows[:2]), _summary(rows[2:]), _summary([]),
    ])
    assert merged["instances"] == 5
    assert merged["complete_and_feasible"] == 3
    assert merged["complete_and_feasible_rate"] == pytest.approx(0.6)
    assert merged["mean_verified_distance_km"] == pytest.approx(6.0)
    assert merged["mean_verified_objective"] == pytest.approx(6.0)
    assert merged["mean_verified_cost_usd"] is None
    assert not merged["verifier_summary_passed"]
    assert merged["rows"] == rows
    failed = protocol.merge_validation_summaries([_summary(rows[1:2]), _summary([])])
    assert failed["mean_verified_objective"] is None


def test_validation_merge_rejects_mixed_objectives_and_missing_success_statistic():
    row = dict(instance_id="v0", verifier_passed=True, objective_distance_km=2)
    first = _summary([row])
    second = _summary([row])
    second["objective_mode"] = "energy_vehicle_cost"
    with pytest.raises(ValueError, match="objective_mode"):
        protocol.merge_validation_summaries([first, second])
    second = _summary([row])
    second["mean_verified_objective"] = None
    with pytest.raises(ValueError, match="incomplete successful-shard statistic"):
        protocol.merge_validation_summaries([first, second])


def test_distributed_contract_explicitly_distinguishes_local_and_global_batch():
    args = SimpleNamespace(batch_size=2, physical_batch_size=2, effective_batch_size=12)
    contract = protocol.configure_distributed_contract(args, DistributedContext(rank=0, world_size=3))
    assert contract["physical_batch_size_per_rank"] == 2
    assert contract["effective_batch_size_global"] == 12
    assert contract["microbatches_per_rank"] == 2
    args.effective_batch_size = 10
    with pytest.raises(ValueError, match="divisible"):
        protocol.configure_distributed_contract(args, DistributedContext(rank=0, world_size=3))


def test_rank_rng_roundtrip_restores_python_numpy_torch_and_pool():
    pool = SimpleNamespace(rng=np.random.default_rng(17))
    random.seed(19)
    np.random.seed(23)
    torch.manual_seed(29)
    state = protocol.capture_rank_rng(pool, "cpu")

    def draw():
        return (random.random(), float(np.random.random()),
                float(torch.rand(())), float(pool.rng.random()))

    expected = draw()
    for _ in range(3):
        draw()
    protocol.restore_rank_rng(state, pool, "cpu")
    assert draw() == expected


class _Pool:
    def __init__(self, *, validation=False, seed=123):
        self.tasks = [SimpleNamespace(view_id=f"{'val' if validation else 'train'}-{index}")
                      for index in range(5 if validation else 24)]
        self._task_by_view_id = {task.view_id: task for task in self.tasks}
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.tasks)

    def instance(self, task):
        return SimpleNamespace(instance_id=task.view_id, index=int(task.view_id.split("-")[-1]))


def _arguments(output, common_root, *, resume=False):
    return SimpleNamespace(
        objective=ObjectiveConfig(mode="distance").to_dict(), reward_contract=None,
        training_epochs=6, data_passes=None, max_batches_per_pass=None, pilot_mode=False,
        validation_every_epochs=1, minimum_training_epochs=2,
        post_minimum_validation_every_epochs=1, validation_checkpoints=6,
        early_stop_patience_validations=2, early_stop_start_epoch=2,
        physical_batch_size=1, effective_batch_size=4, batch_size=1,
        training_stream_path=Path(common_root) / "stream.parquet",
        customer_exposure_budget=6 * 4 * 50, scale="Cus50",
        output_dir=Path(output), protocol_id="distributed-protocol-test", resume=resume,
        validation_limit=5, final_validation_limit=0, validation_every_passes=1,
        validation_decode_type="sampling", validation_candidates=2,
        validation_seed=90210, seed=1234, baseline_eval_size=0,
        steps_per_epoch=1, baseline_warmup_epochs=100, baseline_alpha=0.05,
        ema_decay=0.5, samples_per_instance=2, exposure_checkpoints="",
        gpu_hour_checkpoints="", device="cpu", max_grad_norm=0.4,
        training_rollout_steps=10, validation_rollout_steps=15,
    )


def _protocol_worker(rank, world_size, rendezvous, output, failure_phase, baseline_probe):
    ctx = init_gloo(rank, world_size, rendezvous)
    output = Path(output)
    try:
        protocol.read_stream_view_ids = lambda _path, *, stop: [f"train-{i}" for i in range(stop)]
        protocol.make_validation_pool = lambda *_args, **_kwargs: _Pool(validation=True)

        def train(run_name, *, resume=False, interrupt=False):
            random.seed(100 + rank)
            np.random.seed(200 + rank)
            torch.manual_seed(300 + rank)
            pool = _Pool(seed=400 + rank)
            policy = torch.nn.Linear(1, 1, bias=False)
            optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)
            state = {"epoch": 0, "validation": [], "baseline_probes": []}

            def actor(instances, _soft, _seed):
                epoch = instances[0].index // 4 + 1
                state["epoch"] = epoch
                if interrupt and failure_phase == "actor" and epoch == 3 and rank == 1:
                    raise RuntimeError("intentional epoch 3 interruption")
                indices = torch.tensor([instance.index + 1 for instance in instances], dtype=torch.float32)
                noise = random.random() + np.random.random() + pool.rng.random() + float(torch.rand(()))
                costs = indices[:, None] + torch.tensor([0.0, 0.5])[None, :]
                value = policy.weight.sum() * (indices[:, None] + noise * 0.01)
                # Different ranks perform different local graph work before synchronization.
                for _ in range(rank * 3 + epoch):
                    value = value + 0.0
                return SimpleNamespace(
                    cost=costs, objective=costs, objective_value=costs,
                    vehicles_started=torch.ones_like(costs), feasible=torch.ones_like(costs, dtype=torch.bool),
                    log_likelihood=value.expand(-1, 2), environment_transitions=len(instances) * (rank + 1) * 2,
                    trajectory_steps=torch.full_like(costs, rank + 1, dtype=torch.int64),
                    rollout_budget_exhausted=torch.zeros_like(costs, dtype=torch.bool),
                )

            def validation(instances, solve, *, seed, objective_config=None, cuda_rng_devices=None):
                if interrupt and failure_phase == "validation" and state["epoch"] == 3 and rank == 1:
                    raise RuntimeError("intentional epoch 3 interruption")
                assert cuda_rng_devices == [], "CPU worker must not snapshot unrelated GPUs"
                rows = []
                for offset, instance in enumerate(instances):
                    solve(instance, seed + offset)
                    rows.append(dict(instance_id=instance.instance_id, verifier_passed=True,
                                     objective_distance_km=20.0 - min(state["epoch"], 2) + instance.index))
                return _summary(rows)

            def validation_solve(_model, instance, seed):
                state["validation"].append((state["epoch"], instance.index, seed))
                assert seed == 90210 + instance.index
                return {}

            def baseline(model, instances, _soft, seed):
                if seed < 10_000_000:
                    assert rank == 0, "Only rank zero executes paired baseline probes"
                    state["baseline_probes"].append((state["epoch"], instance_ids(instances)))
                # Give the probe a deterministic improvement to exercise replacement.
                offset = 0.0 if model is policy else 10.0
                costs = torch.tensor([[instance.index + offset, instance.index + offset + 0.5]
                                      for instance in instances])
                return SimpleNamespace(cost=costs)

            def instance_ids(instances):
                return [instance.instance_id for instance in instances]

            protocol.verified_validation = validation
            args = _arguments(output / run_name, output, resume=resume)
            if baseline_probe:
                args.steps_per_epoch = 2
                args.baseline_warmup_epochs = 1
                args.baseline_eval_size = 3
                protocol.ttest_rel = lambda *_args, **_kwargs: SimpleNamespace(pvalue=0.001)
            kwargs = dict(method="AM-EVRPTW", args=args, pool=pool, policy=policy,
                          optimizer=optimizer, make_actor=actor,
                          make_baseline=baseline,
                          training_cost=lambda value: value.cost,
                          objective_distance=lambda value: value.objective,
                          feasible=lambda value: value.feasible,
                          validation_solve=validation_solve, legacy_batch_size=1)
            if interrupt:
                with pytest.raises(RuntimeError, match="intentional epoch 3 interruption"):
                    protocol.train_distributed_reinforce_data_passes(**kwargs)
            else:
                protocol.train_distributed_reinforce_data_passes(**kwargs)
            # Peers may finish a shard before another fails. Preserve only
            # the checkpoint-committed validation cohort across a restart.
            return [entry for entry in state["validation"] if not interrupt or entry[0] <= 2]

        full_validation = train("uninterrupted")
        interrupted_validation = train("resumed", interrupt=True)
        if failure_phase == "validation":
            def mutate_state(*, ahead):
                root = output / "resumed"
                path = root / "data_pass_state.json"
                value = json.loads(path.read_text())
                value.update(optimizer_steps=3 if ahead else 1,
                             instances_seen=12 if ahead else 4,
                             customer_exposures=600 if ahead else 200,
                             environment_transitions=36 if ahead else 12)
                path.write_text(json.dumps(value))
                if not ahead:
                    (root / "best.ckpt").unlink()
                    (root / "validation_summary.json").unlink()
                    with (root / "sampled_view_ids.jsonl").open("a") as stream:
                        stream.write('{"logical_epoch":')
            ctx.main_call(lambda: mutate_state(ahead=True))
            with pytest.raises(RuntimeError, match="ahead of the last durable checkpoint"):
                train("resumed", resume=True)
            # A lagging sidecar/selected alias is a recoverable checkpoint-write window.
            ctx.main_call(lambda: mutate_state(ahead=False))
        interrupted_validation += train("resumed", resume=True)
        assert interrupted_validation == full_validation
        for run_name in ("uninterrupted", "resumed"):
            checkpoint = torch.load(output / run_name / "checkpoint_latest.pt", weights_only=False)
            assert checkpoint["logical_epoch"] == 4
            assert checkpoint["stream_cursor"] == 16
            assert checkpoint["early_stopped"]
            assert len(checkpoint["rank_rng_states"]) == world_size
            # Every global wave has two environments and two trajectories each.
            ema_sample_count = 8 if baseline_probe else 16
            means = [sum((index + 1.25) for index in range(begin, begin + 2)) / 2
                     for begin in range(0, ema_sample_count, 2)]
            expected_ema = means[0]
            for mean in means[1:]:
                expected_ema = 0.5 * expected_ema + 0.5 * mean
            assert checkpoint["ema_cost"] == pytest.approx(expected_ema)
            if baseline_probe:
                assert checkpoint["baseline_eval_count"] == 2
                assert checkpoint["baseline_update_count"] == 2
                assert len(checkpoint["baseline_probe_view_ids"]) == 3
                assert all(view_id in {f"train-{i}" for i in range(24)}
                           for view_id in checkpoint["baseline_probe_view_ids"])
                for key in checkpoint["model"]:
                    torch.testing.assert_close(checkpoint["model"][key], checkpoint["baseline"][key])
        reference = torch.load(output / "uninterrupted" / "checkpoint_latest.pt", weights_only=False)
        resumed = torch.load(output / "resumed" / "checkpoint_latest.pt", weights_only=False)
        for name in reference["model"]:
            torch.testing.assert_close(reference["model"][name], resumed["model"][name], rtol=0, atol=0)
        assert reference["optimizer"]["param_groups"] == resumed["optimizer"]["param_groups"]
        for parameter_id, optimizer_state in reference["optimizer"]["state"].items():
            for key, value in optimizer_state.items():
                torch.testing.assert_close(value, resumed["optimizer"]["state"][parameter_id][key], rtol=0, atol=0)
        assert reference["baseline_probe_view_ids"] == resumed["baseline_probe_view_ids"]
        # A terminal early-stop resume must not consume additional training samples.
        assert train("resumed", resume=True) == []
        Path(output, f"protocol_rank_{rank}.json").write_text(json.dumps({"validation": full_validation}))
        ctx.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("failure_phase,baseline_probe", [("actor", False), ("validation", False), ("validation", True)])
def test_two_rank_protocol_global_ema_validation_early_stop_and_exact_resume(tmp_path, failure_phase, baseline_probe):
    run_gloo_workers(_protocol_worker, tmp_path, extra=(failure_phase, baseline_probe))
    for run_name in ("uninterrupted", "resumed"):
        root = tmp_path / run_name
        history = [json.loads(line) for line in (root / "logical_epoch_history.jsonl").read_text().splitlines()]
        assert [row["logical_epoch"] for row in history] == [1, 2, 3, 4]
        assert all(row["instances_seen"] == 4 for row in history)
        assert all(row["physical_microbatches"] == 4 for row in history)
        assert all(row["mean_environment_feasible_rate"] == 1.0 for row in history)
        assert all(row["environment_transitions"] == 12 for row in history)
        assert [row["baseline_kind"] for row in history] == (
            ["paper_ema", "paper_ema", "greedy_rollout", "greedy_rollout"]
            if baseline_probe else ["paper_ema"] * 4
        )
        if baseline_probe:
            baseline_rows = [json.loads(line) for line in (root / "baseline_history.jsonl").read_text().splitlines()]
            assert [row["optimizer_step"] for row in baseline_rows] == [2, 4]
            assert all(row["baseline_updated"] for row in baseline_rows)
        sampled = [json.loads(line) for line in (root / "sampled_view_ids.jsonl").read_text().splitlines()]
        assert [view for row in sampled for view in row["view_ids"]] == [f"train-{i}" for i in range(16)]
        validation = [json.loads(line) for line in (root / "validation_history.jsonl").read_text().splitlines()]
        assert [row["logical_epoch"] for row in validation] == [1, 2, 3, 4]
        assert all(row["instances"] == row["complete_and_feasible"] == 5 for row in validation)
        assert all([v["instance_id"] for v in row["rows"]] == [f"val-{i}" for i in range(5)] for row in validation)
        result = json.loads((root / "training_result.json").read_text())
        assert result["completed_training_epochs"] == 4
        assert result["early_stopped"] and result["instances_seen"] == 16
        best = torch.load(root / "best.ckpt", weights_only=False)
        assert best["logical_epoch"] == 2
    if failure_phase == "validation":
        backups = list((tmp_path / "resumed").glob("resume_uncommitted_history_*"))
        assert backups
        assert any((backup / "logical_epoch_history.jsonl").is_file() for backup in backups)
    assert len(list(tmp_path.glob("protocol_rank_*.json"))) == 2


def test_explicit_validation_rng_devices_seed_only_owned_gpu(monkeypatch):
    from EVRPTW_Benchmark.Reinforcement_Learning.common import training_protocol

    calls = []

    class GeneratorSpy:
        def __init__(self, index):
            self.index = index

        def manual_seed(self, seed):
            calls.append((self.index, seed))

    @contextmanager
    def fork_rng(*, devices, enabled):
        assert devices == [1] and enabled
        previous = torch.get_rng_state()
        try:
            yield
        finally:
            torch.set_rng_state(previous)

    def seed_all_is_forbidden(_seed):
        raise AssertionError("Explicit rank RNG scope must not seed every visible GPU")

    monkeypatch.setattr(torch.random, "fork_rng", fork_rng)
    monkeypatch.setattr(torch, "manual_seed", seed_all_is_forbidden)
    monkeypatch.setattr(torch.cuda, "default_generators", tuple(GeneratorSpy(i) for i in range(3)))
    monkeypatch.setattr(training_protocol, "select_min_verified_objective", lambda *_args: (
        0, [[0, 1, 0]], {"passed": True, "objective_distance_km": 2.0, "vehicles_started": 1},
    ))
    before = torch.get_rng_state().clone()

    def solve(_instance, seed):
        assert seed == 123
        torch.rand(4)
        return {"success": [True]}

    result = training_protocol.verified_validation(
        [SimpleNamespace(instance_id="v0")], solve, seed=123,
        objective_config=ObjectiveConfig(mode="distance"), cuda_rng_devices=[1],
    )
    assert result["complete_and_feasible"] == 1
    assert calls == [(1, 123)]
    torch.testing.assert_close(torch.get_rng_state(), before)


@pytest.mark.parametrize("existing_alias", [False, True])
def test_atomic_checkpoint_copy_preserves_destination_on_partial_copy_failure(tmp_path, monkeypatch, existing_alias):
    source = tmp_path / "epoch.pt"
    destination = tmp_path / "checkpoint_latest.pt"
    source.write_bytes(b"new complete checkpoint")
    if existing_alias:
        destination.write_bytes(b"previous complete checkpoint")

    def partial_copy(_source, temporary):
        Path(temporary).write_bytes(b"partially copied checkpoint")
        raise OSError("simulated disk failure during copy")

    monkeypatch.setattr(protocol.shutil, "copy2", partial_copy)
    with pytest.raises(OSError, match="simulated disk failure"):
        protocol.atomic_copy(source, destination)
    if existing_alias:
        assert destination.read_bytes() == b"previous complete checkpoint"
    else:
        assert not destination.exists()
    assert source.read_bytes() == b"new complete checkpoint"
    assert not list(tmp_path.glob(".*.copy.tmp"))


def test_atomic_checkpoint_copy_publishes_complete_loadable_replacement(tmp_path):
    source = tmp_path / "epoch.pt"
    destination = tmp_path / "checkpoint_latest.pt"
    torch.save({"epoch": 20, "weight": torch.tensor([1.0, 2.0])}, source)
    destination.write_bytes(b"previous checkpoint bytes")
    protocol.atomic_copy(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    loaded = torch.load(destination, weights_only=False)
    assert loaded["epoch"] == 20
    torch.testing.assert_close(loaded["weight"], torch.tensor([1.0, 2.0]))
    assert not list(tmp_path.glob(".*.copy.tmp"))
