"""Cross-scale deployment protections; no CUDA workloads are launched here."""
import importlib
import json
import os
from pathlib import Path
import subprocess

import pandas as pd
import pytest

from script_curriculum.cus500_curr import launch


@pytest.mark.parametrize('method,batch', [
    ('am_evrptw', 12), ('evrptw_rl', 24), ('drl_ts', 2), ('terran', 16), ('rrnco', 22),
])
def test_five_native_distributed_commands_preserve_curriculum_contract(method, batch):
    args = launch.parse_args(['--method', method, '--gpus', '1,2'])
    job = launch.curriculum_job(args)
    assert args.gpus == [1, 2]
    assert args.checkpoint_root == Path('/data/cus100_ckpt')
    assert job['scale'] == 'Cus500' and job['world_size'] == 2
    assert job['training_epochs'] == job['minimum_training_epochs'] == 3000
    assert job['physical_batch_size'] == batch and job['effective_batch_size'] == batch * 2
    assert job['customer_exposure_budget'] == 3000 * batch * 2 * 500
    assert job['validation_checkpoints'] == 30 and job['validation_views'] == 500
    assert job['validation_candidate_count'] == job['training_trajectory_count'] == 30
    assert job['warm_start_scale_transition'] is True
    assert job['early_stop_patience_validations'] == 0
    command = launch.shared.build_command(job, Path('/data/road'), Path('/data/new'), Path('/data/source.ckpt'))
    assert '--nproc_per_node=2' in command and command.count('torch.distributed.run') == 1
    assert '--warm-start-checkpoint' in command and '--warm-start-scale-transition' in command
    assert '--warm-start-objective-transition' in command and '--resume' not in command
    module = importlib.import_module(job['train_module'])
    native_args = command[command.index('--module') + 2:]
    # TERRAN reuses its shared train.parse_args parser with distributed arguments.
    if method == 'terran':
        native = module.parse_distributed_args(module.parse_args, native_args)
        assert native.warm_start_epoch_mode == 'reset'
        assert native.num_charging_stations == 50
    else:
        native = module.parse_args(native_args)
    assert native.expected_world_size == 2
    scale = native.stage2_scale if method == 'terran' else native.scale
    assert scale == 'Cus500' and native.training_representation == 'G'
    assert native.physical_batch_size == batch and native.effective_batch_size == batch * 2
    assert native.training_epochs == native.minimum_training_epochs == 3000
    assert native.warm_start_scale_transition is True and native.warm_start_objective_transition is True
    if method == 'evrptw_rl':
        assert native.graph_aggregation == 'mean'
        assert (native.training_rollout_steps, native.validation_rollout_steps) == (600, 700)
    if method == 'drl_ts':
        assert native.soft_stage_end_epoch == 0
    if method == 'rrnco':
        assert native.graph_mode == 'full'


@pytest.mark.parametrize('gpus', ['0', '0,0', '0,1,2', '-1,2', 'GPU-a,1', '0,'])
def test_invalid_topology_rejected(gpus):
    with pytest.raises(SystemExit):
        launch.parse_args(['--gpus', gpus])


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
    monkeypatch.setattr(launch, 'source_checkpoint', lambda args: (tmp_path / 'am.ckpt', {'sha256': 'different'}))
    def unexpected_call(*args, **kwargs):
        raise AssertionError('Dry run must not query GPU or launch training')
    monkeypatch.setattr(launch.shared, 'gpu_inventory', unexpected_call)
    monkeypatch.setattr(launch.shared, 'run_training', unexpected_call)
    assert launch.main(['--road-root', str(tmp_path / 'data'), '--output-root', str(tmp_path / 'output'), '--dry-run']) == 0
    assert not (tmp_path / 'output').exists()
    assert 'torch.distributed.run' in capsys.readouterr().out


def test_memory_profile_does_not_claim_a_different_source_was_tested():
    job = launch.curriculum_job(launch.parse_args([]))
    launch.apply_batch_evidence(job, {'sha256': 'unknown-source'})
    assert job['calibration_status'] == 'prior_Cus500_recipe_reused_new_source_not_gpu_profiled'
    frozen = json.loads((launch.HERE / 'source_checkpoint.json').read_text())
    launch.apply_batch_evidence(job, {'sha256': frozen['sha256']})
    assert job['calibration_status'] == 'curriculum_exact_source_two_gpu_smoke_passed'


@pytest.mark.parametrize('name,method', [
    ('am', 'am_evrptw'), ('evrptw_rl', 'evrptw_rl'), ('drl_ts', 'drl_ts'),
    ('terran', 'terran'), ('rrnco', 'rrnco'),
])
def test_shell_passes_method_and_positional_gpu_pair_without_running_training(tmp_path, name, method):
    fake_python = tmp_path / 'python'
    fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    fake_python.chmod(0o755)
    shell = launch.HERE / f'{name}_cus100_to_500.sh'
    env = dict(os.environ, CURRICULUM_PYTHON=str(fake_python))
    result = subprocess.run([str(shell), '1', '2', '--dry-run'], env=env, text=True, capture_output=True, check=True)
    assert result.stdout.splitlines()[1:] == ['--method', method, '--gpus', '1,2', '--dry-run']
    rejected = subprocess.run([str(shell), '1', '1'], env=env, text=True, capture_output=True)
    assert rejected.returncode == 2 and 'physical GPU' in rejected.stderr


def test_background_supervisor_starts_in_its_own_session(tmp_path):
    import shutil
    import sys
    import time
    here = tmp_path / 'repo/script_curriculum/cus500_curr'
    here.mkdir(parents=True)
    shutil.copy2(launch.HERE / '_launch.sh', here / '_launch.sh')
    (here / 'launch.py').write_text('''import json, os, sys
from pathlib import Path
if '--dry-run' not in sys.argv:
    Path(__file__).with_name('session.json').write_text(json.dumps(dict(pid=os.getpid(), session=os.getsid(0), args=sys.argv[1:])))
''')
    env = dict(os.environ, CURRICULUM_PYTHON=sys.executable,
               CURRICULUM_CUS500_OUTPUT_ROOT=str(tmp_path / 'outputs'))
    submitted = subprocess.run([str(here / '_launch.sh'), 'am_evrptw', '--gpus', '1,2'],
                               env=env, text=True, capture_output=True, check=True)
    assert 'Submitted am_evrptw' in submitted.stdout
    proof = here / 'session.json'
    deadline = time.monotonic() + 5
    while not proof.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    data = json.loads(proof.read_text())
    assert data['session'] == data['pid']
    assert data['session'] != os.getsid(0)
    assert data['args'] == ['--method', 'am_evrptw', '--gpus', '1,2']
    pid_file = next((tmp_path / 'outputs/launchers/am_evrptw').glob('*/launcher.pid'))
    assert int(pid_file.read_text()) == data['pid']
