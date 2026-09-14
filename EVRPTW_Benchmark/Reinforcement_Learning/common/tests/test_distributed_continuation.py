"""A declared 48-to-72 transition preserves consumed rows and resume state."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist

from EVRPTW_Benchmark.Reinforcement_Learning.common import distributed_protocol as protocol
from EVRPTW_Benchmark.Reinforcement_Learning.common.data_pass import DataPassState
from EVRPTW_Benchmark.Reinforcement_Learning.common.distributed import DistributedContext
from EVRPTW_Benchmark.Reinforcement_Learning.common.tests.test_distributed_dual_protocol import _assert_tree_equal
from EVRPTW_Benchmark.Reinforcement_Learning.common.tests.test_distributed_helpers import init_gloo, run_gloo_workers
from EVRPTW_Benchmark.Reinforcement_Learning.common.tests.test_distributed_protocol import _arguments, _Pool, _summary
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_protocol import resolved_training_signature_from_args


def _continuation_args(**changes):
    values = dict(batch_size=24, physical_batch_size=24, effective_batch_size=72,
                  stream_continuation_epoch=300, stream_continuation_cursor=14400,
                  stream_continuation_source_batch=48)
    values.update(changes)
    return SimpleNamespace(**values)


def test_actual_three_gpu_cursor_budget_and_disjoint_first_shards():
    args = _continuation_args()
    assert [protocol.stream_cursor_after_epoch(args, epoch, 72)
            for epoch in (0, 299, 300, 301, 10000)] == [0, 14352, 14400, 14472, 712800]
    assert protocol.stream_cursor_after_epoch(args, 10000, 72) * 500 == 356400000
    views = [str(index) for index in range(14472)]
    shards = [protocol.shard_stream_epoch(
        views, logical_epoch=301, physical_batch_size=24, effective_batch_size=72,
        rank=rank, world_size=3, start_cursor=14400) for rank in range(3)]
    assert [view for rank_batches in shards for batch in rank_batches for view in batch] == views[14400:14472]
    assert protocol.stream_cursor_after_epoch(SimpleNamespace(), 301, 72) == 21672


@pytest.mark.parametrize('changes', [
    {'stream_continuation_epoch': None}, {'stream_continuation_cursor': None},
    {'stream_continuation_source_batch': None}, {'stream_continuation_epoch': 0},
    {'stream_continuation_source_batch': 0}, {'stream_continuation_cursor': 21600},
])
def test_incomplete_or_inconsistent_continuation_is_rejected(changes):
    with pytest.raises(ValueError, match='stream continuation'):
        protocol.configure_distributed_contract(_continuation_args(**changes), DistributedContext(world_size=3))


def test_cli_continuation_is_signed_and_unset_contract_remains_identical(tmp_path):
    from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL import distributed_train as adapter

    cli = ['--dataset-path', str(tmp_path / 'train.parquet'), '--output-dir', str(tmp_path / 'run'),
           '--training-epochs', '10000', '--physical-batch-size', '24', '--batch-size', '24',
           '--effective-batch-size', '72', '--device', 'cpu']
    args = adapter.parse_args(cli)
    adapter.prepare_method(args)
    original = protocol.configure_distributed_contract(args, DistributedContext(world_size=3), method='EVRPTW-RL')
    assert 'stream_continuation' not in original
    signature = resolved_training_signature_from_args(args)
    for field in ('stream_continuation_epoch', 'stream_continuation_cursor', 'stream_continuation_source_batch'):
        delattr(args, field)
    assert protocol.configure_distributed_contract(args, DistributedContext(world_size=3), method='EVRPTW-RL') == original
    assert resolved_training_signature_from_args(args) == signature
    resumed = adapter.parse_args(cli + ['--resume', '--stream-continuation-epoch', '300',
        '--stream-continuation-cursor', '14400', '--stream-continuation-source-batch', '48'])
    adapter.prepare_method(resumed)
    contract = protocol.configure_distributed_contract(resumed, DistributedContext(world_size=3), method='EVRPTW-RL')
    actual = resolved_training_signature_from_args(resumed)
    assert actual['sha256'] != signature['sha256']
    assert actual['distributed_training']['stream_continuation'] == contract['stream_continuation']



@pytest.mark.parametrize("case,expected", [
    ("fresh", "requires an explicit --resume"),
    ("budget", "customer-exposure budget"),
    ("before_boundary", "precedes the stream continuation boundary"),
    ("wrong_cursor", "checkpoint epoch and global stream cursor disagree"),
])
def test_continuation_runtime_rejects_fresh_or_inconsistent_resume(tmp_path, monkeypatch, case, expected):
    args = _arguments(tmp_path / "run", tmp_path, resume=case != "fresh")
    vars(args).update(vars(_continuation_args()))
    args.training_epochs = 303
    args.minimum_training_epochs = 300
    args.validation_every_epochs = 100
    args.validation_checkpoints = 6
    args.early_stop_patience_validations = 0
    args.early_stop_start_epoch = 0
    args.customer_exposure_budget = (303 * 72 if case == "budget" else 14616) * 50
    args.ema_warmup_steps = 0
    args.baseline_eval_interval = 100000
    contract = protocol.configure_distributed_contract(args, DistributedContext(), method="EVRPTW-RL")
    objective = protocol.prepare_training_objective(args)
    monkeypatch.setattr(protocol, "prepare_training_objective", lambda _args: objective)
    args.output_dir.mkdir()
    (args.output_dir / "checkpoint_latest.pt").touch()
    monkeypatch.setattr(protocol, "_load_checkpoint", lambda *_args, **_kwargs: {
        "distributed_contract": contract, "logical_epoch": 299 if case == "before_boundary" else 300,
        "stream_cursor": 21600})
    policy = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match=expected):
        protocol.train_distributed_reinforce_data_passes(
            method="EVRPTW-RL", args=args, pool=_Pool(), policy=policy,
            optimizer=torch.optim.AdamW(policy.parameters()), make_actor=None, make_baseline=None,
            training_cost=None, objective_distance=None, feasible=None, validation_solve=None,
            legacy_batch_size=24)


class _ContinuationPool(_Pool):
    def __init__(self, seed):
        super().__init__(seed=seed)
        self.tasks = [SimpleNamespace(view_id=f'train-{i}') for i in range(14616)]
        self._task_by_view_id = {task.view_id: task for task in self.tasks}


def _continuation_worker(rank, world_size, rendezvous, output):
    context = init_gloo(rank, world_size, rendezvous)
    output = Path(output)
    try:
        def read_stream(_path, *, stop):
            assert stop == 14616, 'Stream extent must include the original 48-wide prefix'
            return [f'train-{i}' for i in range(stop)]

        protocol.read_stream_view_ids = read_stream
        protocol.make_validation_pool = lambda *_args, **_kwargs: _Pool(validation=True)
        protocol.verified_validation = lambda instances, *_args, **_kwargs: _summary([
            dict(instance_id=item.instance_id, verifier_passed=True, objective_distance_km=20.)
            for item in instances])

        def arguments(run_name):
            args = _arguments(output / run_name, output, resume=True)
            vars(args).update(vars(_continuation_args()))
            args.training_epochs = 303
            args.minimum_training_epochs = 300
            args.validation_every_epochs = 100
            args.post_minimum_validation_every_epochs = 1
            args.validation_checkpoints = 6
            args.early_stop_patience_validations = 0
            args.early_stop_start_epoch = 0
            args.customer_exposure_budget = 14616 * 50
            args.samples_per_instance = 30
            args.max_grad_norm = 1e9
            args.ema_warmup_steps = 0
            args.baseline_eval_interval = 100000
            return args

        # Produce genuine nonempty AdamW moments and optimizer step 300, then
        # represent the already-migrated source checkpoint with three RNG slots.
        torch.manual_seed(713)
        source_policy = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        source_optimizer = torch.optim.AdamW(source_policy.parameters(), lr=.001)
        for _ in range(300):
            source_optimizer.zero_grad(set_to_none=True)
            source_policy.weight.sum().mul(.1).backward()
            source_optimizer.step()
        random.seed(721 + rank)
        np.random.seed(731 + rank)
        torch.manual_seed(739 + rank)
        source_pool = _ContinuationPool(seed=743 + rank)
        rng_states = context.gather_objects(protocol.capture_rank_rng(source_pool, 'cpu'))
        source_model = deepcopy(source_policy.state_dict())
        source_adam = deepcopy(source_optimizer.state_dict())

        def prepare(run_name):
            args = arguments(run_name)
            args.output_dir.mkdir(parents=True)
            contract = protocol.configure_distributed_contract(args, context, method='EVRPTW-RL')
            state = DataPassState(protocol_id=args.protocol_id, optimizer_steps=300,
                                  instances_seen=14400, customer_exposures=720000,
                                  environment_transitions=123456)
            protocol._save_checkpoint(
                args.output_dir / 'checkpoint_latest.pt', method='EVRPTW-RL', data_pass=0,
                policy=source_policy, baseline=deepcopy(source_policy), optimizer=source_optimizer,
                args=args, extra=dict(logical_epoch=300, stream_cursor=14400,
                    distributed_contract=contract, data_pass_state=asdict(state),
                    rank_rng_states=rng_states, baseline_probe_view_ids=[],
                    completed_validation_checks=3, total_wall_time_s=1800., total_gpu_hours=1.,
                    stage2_transition_provenance={'source_checkpoint_sha256': 'a' * 64,
                                                  'source_world_size': 2, 'target_world_size': 3}))
            state.atomic_write(args.output_dir / 'data_pass_state.json')

        for run_name in ('full', 'resumed'):
            context.main_call(lambda: prepare(run_name))

        original_restore = protocol.restore_rank_rng
        restored = []

        def restore_and_verify(saved, pool, device):
            original_restore(saved, pool, device)
            _assert_tree_equal(protocol.capture_rank_rng(pool, device), saved)
            restored.append(deepcopy(saved))

        protocol.restore_rank_rng = restore_and_verify

        def train(run_name, *, interrupt=False, expected_start=300):
            # Deliberately different initial state proves the checkpoint reloads it.
            random.seed(801 + rank)
            np.random.seed(811 + rank)
            torch.manual_seed(821 + rank)
            pool = _ContinuationPool(seed=823 + rank)
            policy = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
            optimizer = torch.optim.AdamW(policy.parameters(), lr=.9)
            args = arguments(run_name)
            local_grad_terms = []
            observed_ids = []
            expected_payload = torch.load(args.output_dir / 'checkpoint_latest.pt', weights_only=False)
            first = True

            def actor(instances, _soft, _seed):
                nonlocal first
                if first:
                    _assert_tree_equal(policy.state_dict(), expected_payload['model'])
                    _assert_tree_equal(optimizer.state_dict(), expected_payload['optimizer'])
                    assert int(next(iter(optimizer.state.values()))['step']) == expected_start
                    first = False
                epoch = 301 + (instances[0].index - 14400) // 72
                if interrupt and epoch == 302 and rank == 1:
                    raise RuntimeError('intentional three-rank continuation interruption')
                assert len(instances) == 24
                observed_ids.extend(item.index for item in instances)
                indices = torch.tensor([item.index for item in instances], dtype=torch.float64)
                noise = random.random() + float(np.random.random()) + float(pool.rng.random()) + float(torch.rand(()))
                costs = indices[:, None] / 10000 + torch.arange(30, dtype=torch.float64)[None, :] / 100
                features = indices[:, None] / 20000 + torch.arange(30, dtype=torch.float64)[None, :] / 70 + noise / 100
                local_grad_terms.append(float((costs * features).sum()))
                return SimpleNamespace(cost=costs, objective=costs, objective_value=costs,
                    log_likelihood=policy.weight.sum() * features,
                    feasible=torch.ones_like(costs, dtype=torch.bool), vehicles_started=torch.ones_like(costs),
                    trajectory_steps=torch.ones_like(costs, dtype=torch.int64),
                    environment_transitions=24 * 30,
                    rollout_budget_exhausted=torch.zeros_like(costs, dtype=torch.bool))

            actual_step = optimizer.step
            updates = []

            def checked_step(*step_args, **step_kwargs):
                # A serial mean of all 72 environments and their 30 trajectories
                # is the correct gradient; neither source batch 48 nor per-rank
                # batch 24 may be used as the global denominator.
                terms = context.gather_objects(local_grad_terms.pop())
                expected_gradient = sum(terms) / (72 * 30)
                assert float(policy.weight.grad) == pytest.approx(expected_gradient, rel=1e-12)
                updates.append(expected_gradient)
                return actual_step(*step_args, **step_kwargs)

            optimizer.step = checked_step
            kwargs = dict(method='EVRPTW-RL', args=args, pool=pool, policy=policy,
                optimizer=optimizer, make_actor=actor,
                make_baseline=lambda _model, instances, *_args: SimpleNamespace(
                    cost=torch.zeros((len(instances), 30), dtype=torch.float64)),
                training_cost=lambda item: item.cost, objective_distance=lambda item: item.objective,
                feasible=lambda item: item.feasible, validation_solve=lambda *_args: {}, legacy_batch_size=24)
            if interrupt:
                with pytest.raises(RuntimeError, match='intentional three-rank continuation interruption'):
                    protocol.train_distributed_reinforce_data_passes(**kwargs)
            else:
                protocol.train_distributed_reinforce_data_passes(**kwargs)
            return updates, observed_ids

        full_updates, full_ids = train('full')
        first_updates, _ = train('resumed', interrupt=True)
        next_updates, _ = train('resumed', expected_start=301)
        assert first_updates + next_updates == full_updates
        assert len(restored) == 3
        _assert_tree_equal(restored[0], rng_states[rank])
        _assert_tree_equal(restored[1], rng_states[rank])
        full = torch.load(output / 'full' / 'checkpoint_latest.pt', weights_only=False)
        resumed = torch.load(output / 'resumed' / 'checkpoint_latest.pt', weights_only=False)
        for key in ('model', 'optimizer', 'baseline', 'rank_rng_states'):
            _assert_tree_equal(full[key], resumed[key])
        assert full['stage2_transition_provenance'] == resumed['stage2_transition_provenance'] == {
            'source_checkpoint_sha256': 'a' * 64, 'source_world_size': 2, 'target_world_size': 3}
        assert full['logical_epoch'] == 303 and full['stream_cursor'] == 14616
        assert full['data_pass_state']['instances_seen'] == 14616
        assert full['data_pass_state']['customer_exposures'] == 730800
        assert full['data_pass_state']['environment_transitions'] == 123456 + 3 * 72 * 30
        assert full['total_gpu_hours'] == pytest.approx(1. + (full['total_wall_time_s'] - 1800.) * 3 / 3600)
        assert int(next(iter(full['optimizer']['state'].values()))['step']) == 303
        _assert_tree_equal(full['baseline'], source_model)
        assert int(next(iter(source_adam['state'].values()))['step']) == 300
        Path(output, f'continuation_rank_{rank}.json').write_text(json.dumps({'ids': full_ids}))
        context.barrier()
    finally:
        dist.destroy_process_group()


def test_three_rank_48_to_72_continuation_exact_resume_rng_and_global_gradient(tmp_path):
    run_gloo_workers(_continuation_worker, tmp_path, world_size=3)
    for run_name in ('full', 'resumed'):
        output = tmp_path / run_name
        sampled = [json.loads(line) for line in (output / 'sampled_view_ids.jsonl').read_text().splitlines()]
        assert [row['logical_epoch'] for row in sampled] == [301, 302, 303]
        assert [(row['start_cursor'], row['end_cursor']) for row in sampled] == [
            (14400, 14472), (14472, 14544), (14544, 14616)]
        assert [view for row in sampled for view in row['view_ids']] == [f'train-{i}' for i in range(14400, 14616)]
        history = [json.loads(line) for line in (output / 'logical_epoch_history.jsonl').read_text().splitlines()]
        assert [row['global_stream_cursor'] for row in history] == [14472, 14544, 14616]
        assert all(row['instances_seen'] == 72 for row in history)
        result = json.loads((output / 'training_result.json').read_text())
        assert result['optimizer_steps'] == 303 and result['instances_seen'] == 14616
        assert result['customer_exposures'] == 730800
    all_ids = [index for path in tmp_path.glob('continuation_rank_*.json') for index in json.loads(path.read_text())['ids']]
    assert sorted(all_ids) == list(range(14400, 14616))
