"""Real CPU/Gloo PPO synchronization and deterministic data-shard regression."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist

from EVRPTW_Benchmark.Reinforcement_Learning.common.data_pass import seeded_pass_order
from EVRPTW_Benchmark.Reinforcement_Learning.common.distributed import DistributedContext
from EVRPTW_Benchmark.Reinforcement_Learning.common.tests.test_distributed_helpers import init_gloo, run_gloo_workers
from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import _instance
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import data_pool, trainer
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.models import Agent
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import collect_rollout, compute_returns
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.train_distributed import (
    configure_topology, normalize_global_advantages, synchronous_ppo_update, train_distributed,
)


class FakePool:
    def __init__(self, **kwargs):
        self.tasks = [SimpleNamespace(view_id=f"i{i}") for i in range(7)]
    def __len__(self):
        return len(self.tasks)
    def instance(self, task):
        return replace(_instance(), instance_id=task.view_id)
    def first(self, limit=None):
        return [self.instance(task) for task in self.tasks[:limit]]


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_shards_reconstitute_shared_global_cycle_and_resume(monkeypatch, world_size):
    monkeypatch.setattr(data_pool, "Stage2TaskPool", FakePool)
    physical = 2
    pools = [data_pool.Stage2TERRANPool("unused", seed=55, distributed_rank=rank,
             distributed_world_size=world_size, physical_batch_size=physical) for rank in range(world_size)]
    observed = []
    for _ in range(3):
        for pool in pools:
            observed.extend(pool.sample().instance_id for _ in range(physical))
    expected = []
    for position in range(len(observed)):
        cycle, offset = divmod(position, 7)
        expected.append(f"i{seeded_pass_order(7, 55, cycle + 1)[offset]}")
    assert observed == expected
    resumed = [data_pool.Stage2TERRANPool("unused", seed=55, distributed_rank=rank,
               distributed_world_size=world_size, physical_batch_size=physical,
               completed_samples=3 * physical * world_size) for rank in range(world_size)]
    for original, restored in zip(pools, resumed):
        assert [original.sample().instance_id for _ in range(4)] == [restored.sample().instance_id for _ in range(4)]


def _ppo_worker(rank, world_size, rendezvous, output):
    context = init_gloo(rank, world_size, rendezvous)
    try:
        torch.manual_seed(17 + rank)
        agent = Agent(embedding_dim=16, n_encode_layers=1, device="cpu").train()
        context.broadcast_model(agent)
        env = make_terran_env(instance=_instance(), n_traj=3, use_jit_mask=False, rollout_horizon_steps=5 + rank)
        torch.manual_seed(73 + rank)
        batch = collect_rollout(agent, [env], 5 + rank, "sample", "cpu", seed=99 + rank)
        # Explicitly unequal valid populations exercise global loss weighting.
        if rank == 1:
            batch.valid[-1].zero_()
        returns = compute_returns(batch.rewards, batch.dones, gamma=1)
        records = [(batch, returns, returns - batch.values)]
        all_records = [record for part in context.gather_objects(records) for record in part]
        all_values = torch.cat([adv[b.valid].double() for b, _, adv in all_records])
        normalized, stats = normalize_global_advantages(records, context)
        assert stats["count"] == all_values.numel()
        assert stats["mean"] == pytest.approx(float(all_values.mean()))
        assert stats["std"] == pytest.approx(float(all_values.std(unbiased=False)))
        reference = deepcopy(agent)
        reference_records = [(b, ret, (adv - stats["mean"]) / (stats["std"] + 1e-8)) for b, ret, adv in all_records]
        cfg = {"training": {"ppo_update_epochs": 2, "num_minibatches": 1, "ppo_step_chunk_size": 2,
                "gradient_accumulation_steps": 1, "max_grad_norm": .2, "value_loss_type": "smooth_l1",
                "value_loss_beta": 1., "value_residual_scale": 1., "vf_coef": .1, "ent_coef": .01}}
        optimizer = torch.optim.SGD(agent.parameters(), lr=.01)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
        actual = synchronous_ppo_update(agent, optimizer, normalized, cfg, "cpu", context, epoch_seed=37)
        expected = synchronous_ppo_update(reference, ref_optimizer, reference_records, cfg, "cpu", DistributedContext(), epoch_seed=37)
        assert actual["optimizer_steps"] == expected["optimizer_steps"] == 2
        for parameter, expected_parameter in zip(agent.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter, expected_parameter, rtol=2e-5, atol=2e-6)
        states = context.gather_objects({name: value.detach().clone() for name, value in agent.state_dict().items()})
        for name in states[0]:
            for state in states[1:]:
                torch.testing.assert_close(state[name], states[0][name], rtol=0, atol=0)
        Path(output, f"ppo_rank_{rank}.json").write_text(json.dumps(stats))
    finally:
        dist.destroy_process_group()


def test_two_real_gloo_workers_match_serial_legacy_ppo(tmp_path):
    run_gloo_workers(_ppo_worker, tmp_path)
    assert len(list(tmp_path.glob("ppo_rank_*.json"))) == 2


def _train_worker(rank, world_size, rendezvous, output):
    context = init_gloo(rank, world_size, rendezvous)
    data_pool.Stage2TaskPool = FakePool
    trainer.Stage2TaskPool = FakePool
    try:
        cfg = {"output_dir": str(Path(output, "training")), "data": {"stage2_dataset_path": "unused",
               "stage2_scale": "Cus2", "num_customers": 2, "num_charging_stations": 1,
               "stage2_record_sample_ids": True}, "model": {"embedding_dim": 16, "n_encode_layers": 1},
               "training": {"epochs": 2, "num_envs_per_gpu": 1, "n_traj": 2, "rollout_steps": 6,
                "ppo_step_chunk_size": 2, "ppo_update_epochs": 1, "num_minibatches": 1, "gamma": 1,
                "checkpoint_interval": 1, "learning_rate": .0001}, "env": {"use_jit_mask": False},
               "evaluation": {"eval_path": output, "eval_scale": "Cus2", "eval_n_traj": 3,
                 "eval_limit": 2, "eval_interval": 1, "eval_max_steps": 12, "eval_require_independent_verifier": True}}
        path = train_distributed(cfg, seed=18, device="cpu", context=context)
        assert path.is_file()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_real_distributed_training_writes_single_checkpoint_and_global_counts(tmp_path, world_size):
    run_gloo_workers(_train_worker, tmp_path, world_size=world_size)
    output = tmp_path / "training"
    rows = [json.loads(line) for line in (output / "logical_epoch_history.jsonl").read_text().splitlines()]
    assert [r["logical_epoch"] for r in rows] == [1, 2]
    assert [r["samples_seen"] for r in rows] == [world_size, 2 * world_size]
    assert rows[-1]["optimizer_steps_total"] == 2
    payload = torch.load(output / "checkpoints/checkpoint_final.pt", weights_only=False)
    assert payload["epoch"] == 2
    assert payload["config"]["distributed_contract"]["world_size"] == world_size
    assert (output / "best.ckpt").is_file()
    validations = [json.loads(line) for line in (output / "validation_history.jsonl").read_text().splitlines()]
    assert [row["logical_epoch"] for row in validations] == [1, 2]
    assert all(row["instances"] == 2 for row in validations)
    samples = [json.loads(line) for line in (output / "sampled_view_ids.jsonl").read_text().splitlines()]
    assert all(len({view for shard in row["rank_shards"] for view in shard}) == world_size for row in samples)


def test_distributed_topology_rejects_nondivisible_effective_batch_and_variants():
    cfg = {"data": {"stage2_dataset_path": "unused"}, "training": {"num_envs_per_gpu": 2},
           "protocol": {"effective_batch_size": 6}}
    with pytest.raises(ValueError, match="divisible"):
        configure_topology(deepcopy(cfg), DistributedContext(0, 2))
    cfg["protocol"]["effective_batch_size"] = 8
    assert configure_topology(deepcopy(cfg), DistributedContext(0, 2))["microbatches_per_rank"] == 2
    cfg["training"]["algorithm"] = "stable_cost_v1"
    with pytest.raises(ValueError, match="legacy PPO"):
        configure_topology(cfg, DistributedContext(0, 2))
