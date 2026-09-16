"""CPU-only stop-boundary checks; real signals, launches and GPUs are prohibited."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import signal
import socket

import pytest
import torch

from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913 import launch
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913 import watch_stage2 as identities
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913 import watch_stop_epoch as watch
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913.common import write_json


@pytest.fixture(autouse=True)
def prohibit_real_operations(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Stop tests must never signal, launch, or initialize physical GPUs')

    for name in ('kill', 'killpg', 'pidfd_open'):
        monkeypatch.setattr(os, name, forbidden, raising=False)
    monkeypatch.setattr(signal, 'pidfd_send_signal', forbidden, raising=False)
    monkeypatch.setattr(identities, 'pidfd_open', forbidden)
    monkeypatch.setattr(identities, 'pidfd_signal', forbidden)
    monkeypatch.setattr(watch, 'send_owned', forbidden)
    monkeypatch.setattr(watch.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    for name in ('gpu_inventory', 'gpu_processes', 'lock_gpus', 'preflight'):
        monkeypatch.setattr(launch, name, forbidden)


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    run.mkdir()
    directory = tmp_path / 'stop'
    directory.mkdir()
    source_request = tmp_path / 'launch_request.json'
    write_json(source_request, {'fixture': 'immutable launch request'})
    known = {
        role: {'pid': pid, 'start_ticks': pid * 10, 'uid': os.getuid(),
               'ppid': 101 if role == 'torchrun' else 102,
               'cwd': str(tmp_path), 'argv': ['python', watch.TRAIN_MODULE]}
        for role, pid in [('launcher', 101), ('torchrun', 102),
                          ('rank0', 103), ('rank1', 104), ('rank2', 105)]
    }
    current = {saved['pid']: deepcopy(saved) for saved in known.values()}
    monkeypatch.setattr(watch, 'process_identity', current.get)
    monkeypatch.setattr(identities, 'process_identity', current.get)
    monkeypatch.setattr(watch, 'source_snapshot', lambda: {'source_sha256': 'frozen-code'})
    request = {'host': socket.gethostname(), 'source_run': str(run),
               'source_request': str(source_request),
               'source_request_sha256': watch.digest(source_request),
               'source_processes': known, 'target_epoch': 900,
               'physical_gpus': [0, 1, 2], 'poll_seconds': 1,
               'watcher_source_sha256': 'frozen-code'}
    write_json(run / 'progress.json', {'logical_epoch': 900, 'phase': 'validation'})
    write_json(run / 'data_pass_state.json', {'optimizer_steps': 800})
    return watch.Watcher(request, directory), run, current


def publish_checkpoint(run, *, epoch=900):
    cursor = 14400 + (epoch - 300) * 72
    state = {'optimizer_steps': epoch, 'instances_seen': cursor,
             'customer_exposures': cursor * 500,
             'protocol_id': 'cus500_evrptw_rl_after_am_20260913_v1',
             'completed_data_passes': 0, 'environment_transitions': 123,
             'last_checkpoint': str(run / 'checkpoint_latest.pt')}
    payload = {'method': 'EVRPTW-RL', 'logical_epoch': epoch,
               'stream_cursor': cursor, 'completed_validation_checks': epoch // 100,
               'distributed_contract': {'world_size': 3}, 'rank_rng_states': [{}, {}, {}],
               'args': {'ema_warmup_steps': 300, 'effective_batch_size': 72, 'scale': 'Cus500'},
               'model': {'weight': torch.tensor([1.0])},
               'optimizer': {'state': {0: {'step': torch.tensor(900)}}},
               'data_pass_state': state}
    save_payload(run, payload, epoch)
    write_json(run / 'data_pass_state.json', state)
    validation = {'logical_epoch': epoch, 'instances': 500,
                  'complete_and_feasible': 499, 'mean_verified_cost_usd': 123.5}
    (run / 'validation_history.jsonl').write_text(json.dumps(validation) + '\n')
    for name in ('best.ckpt', 'best_overall.ckpt', 'best_within_5000.ckpt'):
        (run / name).write_bytes(b'preserved selected checkpoint')
    for name in ('validation_summary.json', 'validation_summary_overall.json',
                 'validation_summary_within_5000.json', 'stage2_transition.json'):
        write_json(run / name, {'preserved': True})
    return payload


def save_payload(run, payload, epoch=900):
    artifact = run / f'checkpoint_epoch_{epoch:04d}.pt'
    torch.save(payload, artifact)
    shutil.copy2(artifact, run / 'checkpoint_latest.pt')


@pytest.mark.parametrize('phase', ['training_row', 'validation_row', 'checkpoint_before_sidecar'])
def test_waits_until_final_sidecar_publication(scenario, monkeypatch, phase):
    watcher, run, _ = scenario
    (run / 'logical_epoch_history.jsonl').write_text('{"logical_epoch": 900}\n')
    if phase != 'training_row':
        publish_checkpoint(run)
        write_json(run / 'data_pass_state.json', {'optimizer_steps': 800})
        if phase == 'validation_row':
            (run / 'checkpoint_epoch_0900.pt').unlink()
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('Incomplete publication must not be loaded'))
    watcher.step()
    assert watcher.state['status'] == 'waiting_epoch'
    assert not (watcher.directory / 'checkpoint_backup').exists()


def test_committed_checkpoint_has_continuation_cursor_and_exposure(scenario):
    _, run, _ = scenario
    publish_checkpoint(run)
    checked = watch.validate_checkpoint(run, 900)
    assert checked['logical_epoch'] == 900
    assert checked['stream_cursor'] == 57600
    assert checked['customer_exposures'] == 28800000
    assert checked['validation_instances'] == 500
    assert checked['checkpoint_sha256'] == watch.digest(run / 'checkpoint_epoch_0900.pt')


@pytest.mark.parametrize('key,value', [
    ('logical_epoch', 899), ('stream_cursor', 64800), ('completed_validation_checks', 8),
    ('distributed_contract', {'world_size': 2}), ('rank_rng_states', [{}, {}]),
    ('model', {}), ('optimizer', {'state': {}}),
])
def test_rejects_wrong_checkpoint_training_state(scenario, key, value):
    watcher, run, _ = scenario
    payload = publish_checkpoint(run)
    payload[key] = value
    save_payload(run, payload)
    with pytest.raises(RuntimeError, match='Checkpoint method, epoch, topology or training state differs'):
        watcher.step()
    assert not (watcher.directory / 'checkpoint_backup').exists()


@pytest.mark.parametrize('where', ['sidecar', 'payload', 'both'])
def test_rejects_inconsistent_or_wrong_committed_cursor(scenario, where):
    watcher, run, _ = scenario
    payload = publish_checkpoint(run)
    if where in ('payload', 'both'):
        payload['data_pass_state']['instances_seen'] = 64800
        save_payload(run, payload)
    if where in ('sidecar', 'both'):
        state = watch.read_json(run / 'data_pass_state.json')
        state['instances_seen'] = 64800
        write_json(run / 'data_pass_state.json', state)
    with pytest.raises(RuntimeError, match='embedded state and committed sidecar disagree'):
        watcher.step()


@pytest.mark.parametrize('problem', ['later_sidecar', 'wrong_validation', 'partial_validation', 'latest_mismatch'])
def test_publication_mismatches_fail_without_signal(scenario, problem):
    watcher, run, _ = scenario
    publish_checkpoint(run)
    if problem == 'later_sidecar':
        write_json(run / 'data_pass_state.json', {'optimizer_steps': 1000})
    elif problem == 'latest_mismatch':
        (run / 'checkpoint_latest.pt').write_bytes(b'previous checkpoint')
    else:
        validation = {'logical_epoch': 800 if problem == 'wrong_validation' else 900,
                      'instances': 499 if problem == 'partial_validation' else 500}
        (run / 'validation_history.jsonl').write_text(json.dumps(validation) + '\n')
    with pytest.raises(RuntimeError):
        watcher.step()
    assert not (watcher.directory / 'checkpoint_backup').exists()


@pytest.mark.parametrize('role', ['launcher', 'torchrun', 'rank0', 'rank1', 'rank2'])
def test_reused_pid_prevents_any_signal(scenario, role):
    watcher, run, current = scenario
    publish_checkpoint(run)
    current[watcher.request['source_processes'][role]['pid']]['start_ticks'] += 1
    with pytest.raises(RuntimeError, match='Reused or changed'):
        watcher.step()


def test_signal_helper_rejects_pid_reuse_without_opening_pidfd(scenario):
    watcher, _, current = scenario
    saved = watcher.request['source_processes']['torchrun']
    current[saved['pid']]['start_ticks'] += 1
    with pytest.raises(RuntimeError, match='reused'):
        identities.send_owned(saved, signal.SIGTERM)


def test_missing_rank_prevents_any_signal(scenario):
    watcher, run, current = scenario
    publish_checkpoint(run)
    del current[105]
    with pytest.raises(RuntimeError, match='Source stopped before'):
        watcher.step()


def test_backup_is_complete_and_verified_before_only_torchrun_signal(scenario, monkeypatch):
    watcher, run, current = scenario
    publish_checkpoint(run)
    events = []

    def send(saved, sig):
        manifest = watch.read_json(watcher.directory / 'checkpoint_backup/manifest.json')
        assert len(manifest['files']) == 9
        for row in manifest['files']:
            assert watch.digest(watcher.directory / 'checkpoint_backup' / row['name']) == row['sha256']
        assert saved == watcher.request['source_processes']['torchrun']
        assert sig == signal.SIGTERM
        events.append(('signal', saved['pid']))
        for pid in (102, 103, 104, 105):
            del current[pid]
        return True

    def finish_launcher(seconds):
        assert watcher.state['remaining_roles'] == ['launcher']
        events.append(('wait', seconds))
        current.clear()

    monkeypatch.setattr(watch, 'send_owned', send)
    monkeypatch.setattr(watch.time, 'sleep', finish_launcher)
    watcher.step()
    assert events == [('signal', 102), ('wait', 2)]
    assert watcher.state['status'] == 'stopped_at_checkpoint'
    assert watch.read_json(run / 'user_stop_after_epoch900.json')['reason'] == 'user_requested_stop_after_epoch900'
    assert len(watcher.state['stopped_processes']) == 5


@pytest.mark.parametrize('failure', ['missing_artifact', 'corrupted_backup'])
def test_backup_failure_never_signals(scenario, monkeypatch, failure):
    watcher, run, _ = scenario
    publish_checkpoint(run)
    if failure == 'missing_artifact':
        (run / 'best.ckpt').unlink()
    else:
        copy = watch.shutil.copy2

        def corrupt(source, target):
            result = copy(source, target)
            Path(target).write_bytes(b'corrupt backup')
            return result

        monkeypatch.setattr(watch.shutil, 'copy2', corrupt)
    with pytest.raises((FileNotFoundError, RuntimeError)):
        watcher.step()
    assert not (run / 'user_stop_after_epoch900.json').exists()


def test_shutdown_timeout_does_not_signal_more_processes(scenario, monkeypatch):
    watcher, _, _ = scenario
    sent = []
    times = iter([0, 0, 181])
    monkeypatch.setattr(watch.time, 'monotonic', lambda: next(times))
    monkeypatch.setattr(watch.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(watch, 'send_owned', lambda saved, sig: sent.append((saved['pid'], sig)))
    with pytest.raises(RuntimeError, match='did not fully exit'):
        watcher.stop_source()
    assert sent == [(102, signal.SIGTERM)]


def test_duplicate_directory_lock_rejects_second_owner(tmp_path):
    directory = tmp_path / 'stop'
    first = watch.lock(directory)
    try:
        with pytest.raises(RuntimeError, match='already owns'):
            watch.lock(directory)
    finally:
        os.close(first)


def test_registered_request_is_not_overwritten_or_relaunched(tmp_path):
    directory = tmp_path / 'stop'
    directory.mkdir()
    path = directory / 'request.json'
    write_json(path, {'registered': True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match='already registered'):
        watch.main(['--output-dir', str(directory)])
    assert path.read_bytes() == before


@pytest.mark.parametrize('change', ['request', 'code', 'past_boundary'])
def test_changed_registration_or_missed_boundary_never_signals(scenario, monkeypatch, change):
    watcher, run, _ = scenario
    publish_checkpoint(run)
    if change == 'request':
        write_json(watcher.request['source_request'], {'changed': True})
    elif change == 'code':
        monkeypatch.setattr(watch, 'source_snapshot', lambda: {'source_sha256': 'changed'})
    else:
        write_json(run / 'progress.json', {'logical_epoch': 902})
    with pytest.raises(RuntimeError):
        watcher.step()
    assert not (watcher.directory / 'checkpoint_backup').exists()
