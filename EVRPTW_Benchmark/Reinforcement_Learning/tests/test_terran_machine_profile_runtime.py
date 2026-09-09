from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from EVRPTW_Benchmark.Reinforcement_Learning.scripts import drl_job_runtime as runtime


def _job():
    rows = (runtime.ROOT / 'scripts/rq_v1/2080ti_4_1/jobs.jsonl').read_text().splitlines()
    return next(json.loads(row) for row in rows if json.loads(row)['method'] == 'terran')


def test_profile_controls_command_signature_and_provenance(tmp_path):
    job = copy.deepcopy(_job())
    job.pop('warm_start_source_commit', None)
    profile = yaml.safe_load(runtime.TERRAN_CONFIG.read_text())
    profile['training'].update(vf_coef=0.07, critic_backbone_grad_scale=0.03, ppo_step_chunk_size=16)
    path = tmp_path / 'profiles' / 'small.yaml'
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(profile))
    job.update(terran_config_path='profiles/small.yaml', terran_config_sha256=runtime.file_sha256(path))
    job.pop('ppo_step_chunk_size', None)
    context = {'repo': tmp_path, 'dataset': tmp_path / 'dataset'}
    command = runtime.training_command(job, context, tmp_path / 'run', False)
    assert Path(command[command.index('--config') + 1]) == path
    signature = runtime.expected_resolved_training_signature(job, context)
    controls = signature['method_specific']['training']
    assert controls['vf_coef'] == 0.07
    assert controls['critic_backbone_grad_scale'] == 0.03
    assert controls['ppo_step_chunk_size'] == 16
    contract = runtime.training_contract(job)
    assert contract['terran_config_sha256'] == runtime.file_sha256(path)
    # Changing a profile after manifest preparation must fail before any launch.
    path.write_text(path.read_text() + '\n# changed\n')
    with pytest.raises(RuntimeError, match='profile hash mismatch'):
        runtime.training_command(job, context, tmp_path / 'run', False)


@pytest.mark.parametrize('relative', ['/tmp/profile.yaml', '../profile.yaml'])
def test_profile_rejects_paths_outside_repository(tmp_path, relative):
    with pytest.raises(ValueError, match='relative to the repository'):
        runtime.terran_config_path({'terran_config_path': relative, 'terran_config_sha256': 'a' * 64}, tmp_path)


def test_profile_requires_hash_and_preserves_legacy_default(tmp_path):
    assert runtime.terran_config_path({}) == runtime.TERRAN_CONFIG
    with pytest.raises(ValueError, match='requires terran_config_sha256'):
        runtime.terran_config_path({'terran_config_path': 'profile.yaml'}, tmp_path)
    with pytest.raises(ValueError, match='requires terran_config_path'):
        runtime.terran_config_path({'terran_config_sha256': 'a' * 64}, tmp_path)
