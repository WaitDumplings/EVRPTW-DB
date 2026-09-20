"""Cross-scale deployment protections; no trainer or CUDA workload is launched."""
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from script_curriculum.cus500_curr import launch


def objective():
    return json.loads((launch.REPO / 'EVRPTW_Benchmark/Reinforcement_Learning/scripts/'
                       'ablation_final/configs/objective_dtime.json').read_text())['objective']


def payload():
    return dict(method='AM-EVRPTW', model={'weight': torch.ones(2)}, logical_epoch=1300,
                protocol_id='curriculum_stage1_cus100_dtime_v1', objective_config=objective(),
                args=dict(scale='Cus100', training_representation='G', seed=1234,
                          embedding_dim=128, n_encode_layers=3, n_heads=8, tanh_clipping=10.0))


def test_default_is_one_shared_model_on_two_gpus_with_additional_budget():
    args = launch.parse_args([])
    job = launch.curriculum_job(args)
    assert args.gpus == [0, 1]
    assert job['scale'] == 'Cus500'
    assert job['world_size'] == 2
    assert job['training_epochs'] == job['minimum_training_epochs'] == 3000
    assert job['physical_batch_size'] == 12
    assert job['effective_batch_size'] == 24
    assert job['customer_exposure_budget'] == 36_000_000
    assert job['validation_checkpoints'] == 30
    assert job['validation_views'] == 500
    assert job['validation_candidate_count'] == job['training_trajectory_count'] == 30
    assert (job['training_rollout_steps'], job['validation_rollout_steps']) == (1700, 2550)
    assert job['warm_start_scale_transition'] is True
    assert job['early_stop_patience_validations'] == 0
    command = launch.shared.build_command(job, Path('/data/road'), Path('/data/new'), Path('/data/source.pt'))
    assert command.count('torch.distributed.run') == 1
    assert '--nproc_per_node=2' in command
    assert 'EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.distributed_train' in command
    assert '--warm-start-objective-transition' in command
    # The shared command builder adds the dedicated opt-in only for cross-scale jobs.
    assert '--warm-start-scale-transition' in command
    assert '--resume' not in command
    from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.distributed_train import parse_args
    native = parse_args(command[command.index('--module') + 2:])
    assert native.expected_world_size == 2
    assert native.scale == 'Cus500'
    assert native.training_representation == 'G'
    assert native.physical_batch_size == 12 and native.effective_batch_size == 24
    assert native.training_epochs == native.minimum_training_epochs == 3000
    assert native.warm_start_scale_transition is True
    assert native.warm_start_objective_transition is True
    assert native.steps_per_epoch * native.baseline_warmup_epochs == 2500
    assert native.learning_rate == 1e-4 and native.weight_decay == 0.01


@pytest.mark.parametrize('gpus', ['0', '0,0', '0,1,2', '-1,2', 'GPU-a,1', '0,'])
def test_invalid_topology_rejected(gpus):
    with pytest.raises(SystemExit):
        launch.parse_args(['--gpus', gpus])


def test_source_override_is_validated_without_claiming_default_validation_score(tmp_path):
    path = tmp_path / 'custom.pt'
    torch.save(payload(), path)
    args = launch.parse_args(['--source-checkpoint', str(path)])
    resolved, metadata = launch.source_checkpoint(args)
    assert resolved == path
    assert metadata['logical_epoch'] == 1300
    assert metadata['sha256'] == launch.shared.digest(path)
    assert metadata['default_frozen_source'] is False
    assert 'validation' not in metadata
    assert metadata['source_selection'] == 'explicit_checkpoint_override'


@pytest.mark.parametrize('field,value', [
    ('method', 'EVRPTW-RL'), ('protocol_id', 'old_road_cost'), ('logical_epoch', 0),
    ('args.training_representation', 'E'), ('args.scale', 'Cus500'), ('args.n_heads', 4),
    ('objective_config.objective_distance_source', 'shortest_path_distance_km'),
    ('objective_config.vehicle_fixed_cost_usd', 0),
])
def test_override_rejects_wrong_policy_domain_stage_architecture_or_cost(tmp_path, field, value):
    row = deepcopy(payload())
    keys = field.split('.')
    target = row
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path = tmp_path / 'wrong.pt'
    torch.save(row, path)
    with pytest.raises(ValueError):
        launch.source_checkpoint(launch.parse_args(['--source-checkpoint', str(path)]))


def test_default_source_has_frozen_hash_and_no_latest_fallback(tmp_path, monkeypatch):
    frozen = json.loads((launch.HERE / 'source_checkpoint.json').read_text())
    here = tmp_path / 'metadata'
    here.mkdir()
    path = tmp_path / frozen['relative_path']
    path.parent.mkdir(parents=True)
    torch.save(payload(), path)
    frozen['sha256'] = launch.shared.digest(path)
    (here / 'source_checkpoint.json').write_text(json.dumps(frozen))
    monkeypatch.setattr(launch, 'HERE', here)
    args = launch.parse_args(['--stage1-root', str(tmp_path)])
    assert launch.source_checkpoint(args)[1]['validation'] == frozen['validation']
    torch.save(dict(payload(), logical_epoch=2000), path)
    with pytest.raises(ValueError, match='frozen source hash'):
        launch.source_checkpoint(args)
    path.unlink()
    torch.save(payload(), path.parent / 'checkpoint_latest.pt')
    with pytest.raises(FileNotFoundError):
        launch.source_checkpoint(args)


def write_data(root, job, *, overlap=False):
    for split, count, filename in [('train', 10000, job['train_index']), ('val', 500, job['validation_index'])]:
        frame = pd.DataFrame(dict(view_id=[f'{split}-{i}' for i in range(count)],
                                  family_id=[f'{"train" if overlap else split}-family-{i}' for i in range(count)],
                                  customer_count=500, split_id=split, track_id='train' if split == 'train' else 'validation'))
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)


def test_data_audit_uses_complete_cohorts_and_rejects_parent_overlap(tmp_path):
    job = launch.curriculum_job(launch.parse_args([]))
    write_data(tmp_path, job)
    assert launch.inspect_data(tmp_path, job)['train_views'] == 10000
    assert launch.inspect_data(tmp_path, job)['validation_views'] == 500
    write_data(tmp_path, job, overlap=True)
    with pytest.raises(ValueError, match='parent families overlap'):
        launch.inspect_data(tmp_path, job)


def test_dry_run_does_not_query_gpu_or_create_training_output(tmp_path, monkeypatch, capsys):
    job = launch.curriculum_job(launch.parse_args([]))
    write_data(tmp_path / 'data', job)
    checkpoint = tmp_path / 'source.pt'
    torch.save(payload(), checkpoint)
    def unexpected_call(*args, **kwargs):
        raise AssertionError('Dry run must not query GPU or launch training')
    monkeypatch.setattr(launch.shared, 'gpu_inventory', unexpected_call)
    monkeypatch.setattr(launch.shared, 'run_training', unexpected_call)
    assert launch.main(['--source-checkpoint', str(checkpoint), '--road-root', str(tmp_path / 'data'),
                        '--output-root', str(tmp_path / 'output'), '--dry-run']) == 0
    assert not (tmp_path / 'output').exists()
    assert 'torch.distributed.run' in capsys.readouterr().out
