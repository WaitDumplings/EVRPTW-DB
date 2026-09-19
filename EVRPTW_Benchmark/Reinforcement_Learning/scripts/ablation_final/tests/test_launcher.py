"""CPU-only contract checks; no dataset, CUDA allocation, or child training."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

SPEC = importlib.util.spec_from_file_location(
    'ablation_final_launcher', Path(__file__).resolve().parents[1] / 'launch.py'
)
LAUNCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCH)


def argument(command, name):
    return command[command.index(name) + 1]


@pytest.mark.parametrize('scale,available,expected', [
    (100, 1, 1), (100, 4, 1), (500, 2, 2), (500, 4, 2),
    (1000, 3, 3), (1000, 4, 4), (1000, 8, 4),
])
def test_gpu_topology_matches_deployment_scale(scale, available, expected):
    assert LAUNCH.gpu_count(scale, available) == expected


@pytest.mark.parametrize('scale,available', [(100, 0), (500, 1), (1000, 2), (2000, 4)])
def test_gpu_topology_rejects_unsupported_allocation(scale, available):
    with pytest.raises(ValueError):
        LAUNCH.gpu_count(scale, available)


@pytest.mark.parametrize('method', LAUNCH.METHODS)
@pytest.mark.parametrize('scale,world_size', [(100, 1), (500, 2), (1000, 3), (1000, 4)])
def test_command_preserves_global_batch_objective_and_fixed_validation(tmp_path, method, scale, world_size):
    job = LAUNCH.make_job(method, scale, world_size, 300)
    root, run = tmp_path / 'data', tmp_path / 'run'
    command = LAUNCH.command_for(job, root, run)
    physical = LAUNCH.BATCHES[scale][method]
    assert job['physical_batch_size'] == physical
    assert job['effective_batch_size'] == physical * world_size
    assert job['customer_exposure_budget'] == 300 * physical * world_size * scale
    assert argument(command, '--training-epochs') == '300'
    assert argument(command, '--minimum-training-epochs') == '300'
    assert argument(command, '--physical-batch-size') == str(physical)
    assert argument(command, '--effective-batch-size') == str(physical * world_size)
    assert argument(command, '--validation-every-epochs') == '100'
    assert argument(command, '--validation-checkpoints') == '3'
    assert argument(command, '--validation-limit') == '500'
    assert argument(command, '--validation-candidates') == '30'
    assert argument(command, '--early-stop-patience-validations') == '0'
    assert argument(command, '--objective-distance-source') == 'running_time_path_distance_km'
    assert argument(command, '--output-dir') == str(run)
    objective = json.loads(Path(argument(command, '--objective-config')).read_text())
    assert objective['objective']['objective_distance_source'] == 'running_time_path_distance_km'
    assert '--resume' not in command
    assert '--warm-start-checkpoint' not in command
    if world_size == 1:
        assert command[:3] == [sys.executable, '-m', job['train_module']]
        assert job['train_module'].endswith('.train')
        assert '--expected-world-size' not in command
    else:
        assert command[:3] == [sys.executable, '-m', 'torch.distributed.run']
        assert f'--nproc_per_node={world_size}' in command
        assert argument(command, '--module') == job['train_module']
        assert argument(command, '--expected-world-size') == str(world_size)
        suffix = 'train_distributed' if method == 'terran' else 'distributed_train'
        assert job['train_module'].endswith('.' + suffix)
    trajectories_flag = '--n-traj' if method == 'terran' else '--samples-per-instance'
    assert argument(command, trajectories_flag) == '30'
    if method == 'evrptw_rl':
        assert argument(command, '--graph-aggregation') == 'mean'
    if method == 'drl_ts':
        # The 300-epoch diagnostic is explicitly still within its soft stage.
        assert int(argument(command, '--soft-stage-end-epoch')) == 300
        assert job['planned_full_run_soft_stage_end_epoch'] == 2500


def test_batch_override_is_per_gpu_and_model_jobs_do_not_share_mutable_arguments(tmp_path):
    job = LAUNCH.make_job('rrnco', 500, 2, 300, batch=7)
    assert job['physical_batch_size'] == 7
    assert job['effective_batch_size'] == 14
    job['extra_args'].append('--test-mutation')
    next_job = LAUNCH.make_job('rrnco', 500, 2, 300, batch=7)
    assert '--test-mutation' not in next_job['extra_args']
    assert next_job['extra_args'].count('--objective-distance-source') == 1


@pytest.mark.parametrize('epochs,interval', [(0, 100), (300, 0), (299, 100), (300, -1)])
def test_incomplete_validation_budget_is_rejected(epochs, interval):
    with pytest.raises(ValueError):
        LAUNCH.make_job('am_evrptw', 100, 1, epochs, validation_every=interval)


def test_zero_physical_batch_is_rejected():
    with pytest.raises(ValueError):
        LAUNCH.make_job('am_evrptw', 100, 1, 300, batch=0)


def test_inventory_respects_visible_indices_and_uuids(monkeypatch):
    fake = ('0, GPU-first, NVIDIA GeForce RTX 2080 Ti, 11264, 100\n'
            '1, GPU-second, NVIDIA GeForce RTX 2080 Ti, 11264, 200\n'
            '2, GPU-third, NVIDIA GeForce RTX 2080 Ti, 11264, 300\n')
    monkeypatch.setattr(LAUNCH.subprocess, 'check_output', lambda *a, **k: fake)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1,GPU-third')
    assert [g['index'] for g in LAUNCH.gpu_inventory()] == [1, 2]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    assert LAUNCH.gpu_inventory() == []
