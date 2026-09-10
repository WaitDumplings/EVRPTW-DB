from __future__ import annotations

from types import SimpleNamespace

import pytest

from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import data_pool


def _fake_pool(monkeypatch, *, size=17, instances=False):
    class FakeTaskPool:
        def __init__(self, **kwargs):
            self.tasks = [SimpleNamespace(view_id=f"view-{index}") for index in range(size)]
            self.loaded_ids = []

        def __len__(self):
            return len(self.tasks)

        def instance(self, task):
            self.loaded_ids.append(task.view_id)
            if instances:
                from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import _instance
                return _instance()
            return task.view_id

    monkeypatch.setattr(data_pool, "Stage2TaskPool", FakeTaskPool)


def _pool(**kwargs):
    return data_pool.Stage2TERRANPool(
        dataset_path="unused-frozen-index", seed=41, record_sample_ids=True, **kwargs,
    )


@pytest.mark.parametrize("world,batch,size,offset", [
    (4, 3, 53, 0), (4, 3, 17, 113), (2, 7, 17, 16),
    (4, 8, 5, 3), (4, 2, 1, 0), (1, 4, 17, 3),
])
def test_rank_batches_reconstruct_exact_global_sequence_without_extra_loads(
    monkeypatch, world, batch, size, offset,
):
    _fake_pool(monkeypatch, size=size)
    workers = [_pool(sampling_rank=rank, sampling_world_size=world,
                     sampling_batch_size=batch, completed_samples=offset)
               for rank in range(world)]
    reference = _pool(completed_samples=offset)
    actual = []
    for microbatch in range(5):
        local_blocks = [[pool.sample() for _ in range(batch)] for pool in workers]
        actual.extend(view_id for block in local_blocks for view_id in block)
        assert {pool.global_committed_cursor for pool in workers} == {
            offset + (microbatch + 1) * world * batch,
        }
        # No rank materializes or records another rank's skipped task tensors.
        for pool in workers:
            assert len(pool.pool.loaded_ids) == (microbatch + 1) * batch
    assert actual == [reference.sample() for _ in actual]
    for pool in workers:
        assert pool.drain_sampled_view_ids() == pool.pool.loaded_ids
        assert pool.drain_sampled_view_ids() == []
    if offset == 0:
        for start in range(0, len(actual) - size + 1, size):
            assert len(set(actual[start:start + size])) == size


def test_resume_with_different_world_and_batch_preserves_next_global_sample(monkeypatch):
    _fake_pool(monkeypatch)
    first_workers = [_pool(sampling_rank=rank, sampling_world_size=2, sampling_batch_size=3)
                     for rank in range(2)]
    actual = [pool.sample() for pool in first_workers for _ in range(3)]
    cursor = first_workers[0].global_committed_cursor
    assert cursor == 6
    resumed = [_pool(sampling_rank=rank, sampling_world_size=4, sampling_batch_size=2,
                     completed_samples=cursor) for rank in range(4)]
    for _ in range(4):
        actual.extend(pool.sample() for pool in resumed for _ in range(2))
    assert {pool.global_committed_cursor for pool in resumed} == {38}
    reference = _pool()
    assert actual == [reference.sample() for _ in actual]
    single_gpu = _pool(completed_samples=resumed[0].global_committed_cursor)
    assert [single_gpu.sample() for _ in range(11)] == [reference.sample() for _ in range(11)]


def test_partial_rank_batch_cannot_be_checkpointed(monkeypatch):
    _fake_pool(monkeypatch)
    worker = _pool(sampling_rank=2, sampling_world_size=4, sampling_batch_size=3,
                   completed_samples=113)
    assert worker.global_committed_cursor == 113
    worker.sample()
    assert worker.sample_count == 113
    with pytest.raises(RuntimeError, match="inside a physical batch"):
        _ = worker.global_committed_cursor
    worker.sample()
    worker.sample()
    assert worker.global_committed_cursor == 125


def test_single_gpu_cursor_still_advances_per_sample(monkeypatch):
    _fake_pool(monkeypatch)
    pool = _pool(sampling_batch_size=8)
    pool.sample()
    assert pool.sample_count == pool.global_committed_cursor == 1


@pytest.mark.parametrize("kwargs,message", [
    ({"sampling_world_size": 0}, "positive"),
    ({"sampling_batch_size": 0}, "positive"),
    ({"sampling_rank": -1}, "rank"),
    ({"sampling_rank": 4, "sampling_world_size": 4}, "rank"),
    ({"sampling_world_size": 2.5}, "integer"),
    ({"sampling_batch_size": True}, "integer"),
    ({"sampling_world_size": 4, "training_stream_path": "old-stream"}, "shuffle cycle"),
])
def test_invalid_distributed_sampling_rejected_before_dataset_loading(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _pool(**kwargs)


def test_env_bootstrap_and_first_reset_do_not_repeat_rank_batch(monkeypatch):
    from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env

    _fake_pool(monkeypatch, instances=True)
    world, batch = 4, 2
    workers = [_pool(sampling_rank=rank, sampling_world_size=world, sampling_batch_size=batch)
               for rank in range(world)]
    environments = [[make_terran_env(instance_sampler=pool.sample, n_traj=1,
                                    use_jit_mask=False) for _ in range(batch)]
                    for pool in workers]
    try:
        assert {pool.global_committed_cursor for pool in workers} == {world * batch}
        for envs in environments:
            for env in envs:
                env.reset()
        assert {pool.global_committed_cursor for pool in workers} == {world * batch}
        first = [view_id for pool in workers for view_id in pool.drain_sampled_view_ids()]
        for envs in environments:
            for env in envs:
                env.reset()
        assert {pool.global_committed_cursor for pool in workers} == {2 * world * batch}
        second = [view_id for pool in workers for view_id in pool.drain_sampled_view_ids()]
        reference = _pool()
        for _ in range(2 * world * batch):
            reference.sample()
        assert first + second == reference.drain_sampled_view_ids()
        assert all(len(pool.pool.loaded_ids) == 2 * batch for pool in workers)
    finally:
        for envs in environments:
            for env in envs:
                env.close()


@pytest.mark.parametrize("world", [1, 4])
@pytest.mark.parametrize("offset", [0, 13])
def test_real_make_envs_forwards_rank_sampler_before_bootstrap(tmp_path, monkeypatch, world, offset):
    import pandas as pd

    from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import _instance
    from EVRPTW_Benchmark.Reinforcement_Learning.common import stage2_data
    from EVRPTW_Benchmark.Reinforcement_Learning.common.data_pass import seeded_pass_order
    from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.stable_trainer import resolve_stable_config
    from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.trainer import make_envs

    size, batch = 17, 2
    index_path = tmp_path / "view_index.parquet"
    pd.DataFrame([dict(
        view_id=f"view-{index}", family_id="family", consumer_cohort_id="train",
        split_id="train", track_id="train", city_slug="test", scale_id="Cus2",
        customer_count=2, charging_station_count=1, family_cohort_id="family",
        terminal_count=4, view_seed=index,
    ) for index in range(size)]).to_parquet(index_path)
    loaded_ids = []

    def load_instance(task):
        loaded_ids.append(task.view_id)
        return _instance()

    # Use the real parquet reader, both pool classes and real environment
    # construction. Only materialized matrix IO is replaced by a small instance.
    monkeypatch.setattr(stage2_data, "load_stage2_instance", load_instance)
    workers, environments = [], []
    try:
        for rank in range(world):
            cfg = resolve_stable_config({
                "output_dir": str(tmp_path / "unused-run"),
                "objective": "EVRPTW_Benchmark/Reinforcement_Learning/configs/rivian_energy_vehicle_cost_v2.json",
                "training": {"algorithm": "stable_cost_v1", "num_envs_per_gpu": batch,
                             "distributed_world_size": world, "n_traj": 2, "rollout_steps": 4},
                "data": {"stage2_dataset_path": str(index_path), "stage2_family_root": str(tmp_path),
                         "stage2_scale": "Cus2", "stage2_cache_size": 0,
                         "stage2_completed_samples": offset, "stage2_record_sample_ids": True},
                "env": {"use_jit_mask": False},
            })
            if world > 1:
                cfg["data"].update(stage2_sampling_rank=rank, stage2_sampling_world_size=world,
                                   stage2_sampling_batch_size=batch)
            envs, pool = make_envs(cfg, seed=41)
            workers.append(pool)
            environments.append(envs)
            assert isinstance(pool, data_pool.Stage2TERRANPool)
            assert isinstance(pool.pool, stage2_data.Stage2TaskPool)
            assert (pool.sampling_rank, pool.sampling_world_size, pool.sampling_batch_size) == (
                rank, world, batch,
            )
            assert pool.global_committed_cursor == offset + world * batch
        assert len(loaded_ids) == world * batch
        first = [view_id for pool in workers for view_id in pool.drain_sampled_view_ids()]
        for envs in environments:
            for env in envs:
                env.reset()
        assert len(loaded_ids) == world * batch  # Reuse constructor bootstrap.
        for envs in environments:
            for env in envs:
                env.reset()
        second = [view_id for pool in workers for view_id in pool.drain_sampled_view_ids()]
        assert len(loaded_ids) == 2 * world * batch
        assert {pool.global_committed_cursor for pool in workers} == {offset + 2 * world * batch}
        expected = [f"view-{seeded_pass_order(size, 41, position // size + 1)[position % size]}"
                    for position in range(offset, offset + 2 * world * batch)]
        assert first + second == expected
    finally:
        for envs in environments:
            for env in envs:
                env.close()


@pytest.mark.parametrize("data", [{}, {"train_dataset_path": "unused-fixed-instances"}])
def test_make_envs_rejects_non_stage2_distributed_pool(data):
    from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.trainer import make_envs

    with pytest.raises(ValueError, match="requires a Stage-2 dataset pool"):
        make_envs({"data": data, "training": {"distributed_world_size": 4}}, seed=41)


def test_make_envs_rejects_rank_batch_different_from_environment_count():
    from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.trainer import make_envs

    cfg = {"data": {"stage2_dataset_path": "unused", "stage2_sampling_world_size": 4,
                    "stage2_sampling_batch_size": 3},
           "training": {"distributed_world_size": 4, "num_envs_per_gpu": 2}}
    with pytest.raises(ValueError, match="sampling batch must equal num_envs_per_gpu"):
        make_envs(cfg, seed=41)
