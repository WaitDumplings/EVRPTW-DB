"""CPU boundary tests: all process launches, signals and GPU access are forbidden."""
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import socket

import pytest
import torch

from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913 import watch_stage2 as identities
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913 import watch_stop_epoch as base
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913 import watch_terran_stop5100 as watch
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913.common import write_json


@pytest.fixture(autouse=True)
def prohibit_real_operations(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('TERRAN stop tests must not signal, launch or initialize GPUs')

    for name in ('kill', 'killpg', 'pidfd_open'):
        monkeypatch.setattr(os, name, forbidden, raising=False)
    monkeypatch.setattr(signal, 'pidfd_send_signal', forbidden, raising=False)
    monkeypatch.setattr(identities, 'pidfd_open', forbidden)
    monkeypatch.setattr(identities, 'pidfd_signal', forbidden)
    monkeypatch.setattr(watch, 'send_owned', forbidden)
    monkeypatch.setattr(base, 'send_owned', forbidden)
    monkeypatch.setattr(base.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    run = tmp_path / 'TR05'
    run.mkdir()
    (run / 'checkpoints').mkdir()
    (run / 'logs').mkdir()
    directory = tmp_path / 'stop'
    directory.mkdir()
    source = run / 'launch_record.json'
    write_json(source, {'fixture': 'immutable launch record'})
    known = {
        'launcher': {'pid': 101, 'start_ticks': 1010, 'uid': os.getuid(),
                     'ppid': 1, 'cwd': str(tmp_path), 'argv': ['python', 'launch.py']},
        'trainer': {'pid': 102, 'start_ticks': 1020, 'uid': os.getuid(),
                    'ppid': 101, 'cwd': str(tmp_path), 'argv': ['python', 'TERRAN.train']},
    }
    current = {row['pid']: deepcopy(row) for row in known.values()}
    for module in (watch, base, identities):
        monkeypatch.setattr(module, 'process_identity', current.get, raising=False)
    monkeypatch.setattr(watch, 'source_snapshot', lambda: {'source_sha256': 'frozen-code'})
    request = {'host': socket.gethostname(), 'source_run': str(run),
               'source_request': str(source), 'source_request_sha256': base.digest(source),
               'source_processes': {'trainer': known['trainer']},
               'source_launcher': known['launcher'], 'target_epoch': 5100,
               'physical_gpus': [3], 'poll_seconds': 1,
               'watcher_source_sha256': 'frozen-code'}
    (run / 'logs/train_log.csv').write_text('epoch,epoch_wall_time_s,train_feasible_rate\n5100,90.0,1.0\n')
    (run / 'reward_diagnostics.jsonl').write_text('{"logical_epoch": 5100}\n')
    write_json(run / 'data_pass_state.json', {'instances_seen': 5000 * 384,
                                           'optimizer_steps': 5000 * 12})
    return watch.Watcher(request, directory), run, current


def publish_checkpoint(run, epoch=5100):
    payload = {
        'epoch': epoch, 'seed': 1234,
        'config': {
            'data': {'stage2_training_representation': 'E', 'stage2_scale': 'Cus100',
                     'num_customers': 100},
            'protocol': {'protocol_id': 'cus100_terran_synthetic_road_20260911_v1',
                         'effective_batch_size': 384,
                         'resolved_training_signature_sha256': '8875e4622c87664234903e2127da159d6d1f716a19abe0221ccd6d719f7b036f'},
            'training': {'n_traj': 30, 'num_envs_per_gpu': 384,
                         'ppo_update_epochs': 3, 'num_minibatches': 4},
        },
        'model_state_dict': {'weight': torch.tensor([1.0, 2.0])},
        'optimizer_state_dict': {'state': {0: {'step': torch.tensor(epoch * 12),
                                              'exp_avg': torch.tensor([0.1, 0.2])}},
                                 'param_groups': [{'params': [0], 'lr': 0.0001}]},
    }
    save_payload(run, payload, epoch)
    state = {'protocol_id': 'cus100_terran_synthetic_road_20260911_v1',
             'instances_seen': epoch * 384, 'customer_exposures': epoch * 384 * 100,
             'optimizer_steps': epoch * 12, 'completed_data_passes': 0,
             'environment_transitions': 123,
             'last_checkpoint': str(run / 'checkpoint_latest.pt')}
    write_json(run / 'data_pass_state.json', state)
    val = {'logical_epoch': epoch, 'instances': 500, 'complete_and_feasible': 500,
           'mean_verified_cost_usd': 6400.0}
    (run / 'validation_history.jsonl').write_text(json.dumps(val) + '\n')
    for name in ('best.ckpt', 'best_overall.ckpt', 'best_within_5000.ckpt', 'checkpoint_selected.pt'):
        (run / name).write_bytes(b'preserved selection')
    for name in ('validation_summary.json', 'validation_summary_overall.json',
                 'validation_summary_within_5000.json'):
        write_json(run / name, val)
    return payload


def save_payload(run, payload, epoch=5100):
    torch.save(payload, run / 'checkpoints' / f'checkpoint_epoch_{epoch:04d}.pt')
    # TERRAN saves each file independently; valid content need not have equal byte hashes.
    torch.save(payload, run / 'checkpoint_latest.pt')


def test_boundary_accepts_identical_content_with_distinct_serialized_hashes(scenario):
    _, run, _ = scenario
    publish_checkpoint(run)
    assert base.digest(run / 'checkpoint_latest.pt') != base.digest(run / 'checkpoints/checkpoint_epoch_5100.pt')
    checked = watch.validate_checkpoint(run, 5100)
    assert checked['logical_epoch'] == 5100
    assert checked['customer_exposures'] == 195840000


@pytest.mark.parametrize('publication', ['validation_only', 'checkpoint_before_state'])
def test_waits_for_final_sidecar_before_checkpoint_load(scenario, monkeypatch, publication):
    watcher, run, _ = scenario
    publish_checkpoint(run)
    write_json(run / 'data_pass_state.json', {'instances_seen': 5000 * 384,
                                           'optimizer_steps': 5000 * 12})
    if publication == 'validation_only':
        (run / 'checkpoints/checkpoint_epoch_5100.pt').unlink()
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('Incomplete boundary loaded'))
    watcher.step()
    assert watcher.state['status'] == 'waiting_epoch'
    assert not (watcher.directory / 'checkpoint_backup').exists()


@pytest.mark.parametrize('field,value', [('instances_seen', 5101 * 384),
                                         ('customer_exposures', 5100 * 384 * 100 + 1),
                                         ('optimizer_steps', 5100 * 12 - 1)])
def test_wrong_committed_counters_cannot_trigger_stop(scenario, field, value):
    watcher, run, _ = scenario
    publish_checkpoint(run)
    state = base.read_json(run / 'data_pass_state.json')
    state[field] = value
    write_json(run / 'data_pass_state.json', state)
    with pytest.raises(RuntimeError):
        watcher.step()
    assert not (watcher.directory / 'checkpoint_backup').exists()


@pytest.mark.parametrize('field', ['epoch', 'model_state_dict', 'optimizer_state_dict'])
def test_rejects_latest_checkpoint_content_mismatch(scenario, field):
    watcher, run, _ = scenario
    payload = publish_checkpoint(run)
    if field == 'epoch':
        payload[field] = 5000
    elif field == 'model_state_dict':
        payload[field]['weight'][0] = 9
    else:
        payload[field]['state'][0]['exp_avg'][0] = 9
    torch.save(payload, run / 'checkpoint_latest.pt')
    with pytest.raises(RuntimeError):
        watcher.step()


def test_reused_trainer_pid_cannot_be_signalled(scenario):
    role = 'trainer'
    watcher, run, current = scenario
    publish_checkpoint(run)
    current[watcher.request['source_processes'][role]['pid']]['start_ticks'] += 1
    with pytest.raises(RuntimeError, match='Reused or changed'):
        watcher.step()


def test_only_trainer_signalled_after_backups_verified(scenario, monkeypatch):
    watcher, run, current = scenario
    publish_checkpoint(run)
    signals = []

    def send(saved, sig):
        backup = watcher.directory / 'checkpoint_backup'
        manifest = base.read_json(backup / 'manifest.json')
        assert len(manifest['files']) >= 9
        for row in manifest['files']:
            assert base.digest(backup / row['name']) == row['sha256']
        assert saved == watcher.request['source_processes']['trainer']
        assert sig == signal.SIGTERM
        signals.append(saved['pid'])
        del current[saved['pid']]
        return True

    monkeypatch.setattr(watch, 'send_owned', send)
    watcher.step()
    assert signals == [102]
    assert list(current) == [101]  # The shared launcher is never signalled.
    assert watcher.state['status'] == 'stopped_at_checkpoint'
    assert base.read_json(run / 'user_stop_after_epoch5100.json')['reason'] == 'user_requested_stop_after_epoch5100'


def test_missing_best_backup_prevents_signal(scenario):
    watcher, run, _ = scenario
    publish_checkpoint(run)
    (run / 'best.ckpt').unlink()
    with pytest.raises(FileNotFoundError):
        watcher.step()
    assert not (run / 'user_stop_after_epoch5100.json').exists()
