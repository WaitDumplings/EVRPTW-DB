"""CPU-only safety checks for the epoch-300 handoff; never touch live training."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913 import watch_stage2 as watch
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913.common import load_config, write_json


@pytest.fixture(autouse=True)
def prohibit_real_operations(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('CPU watcher tests must never signal, launch, or inspect physical GPUs')

    monkeypatch.setattr(os, 'kill', forbidden)
    monkeypatch.setattr(os, 'killpg', forbidden)
    monkeypatch.setattr(os, 'pidfd_open', forbidden, raising=False)
    monkeypatch.setattr(signal, 'pidfd_send_signal', forbidden, raising=False)
    monkeypatch.setattr(watch, 'pidfd_open', forbidden)
    monkeypatch.setattr(watch, 'pidfd_signal', forbidden)
    monkeypatch.setattr(watch.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(watch.launch, 'gpu_inventory', forbidden)
    monkeypatch.setattr(watch.launch, 'gpu_processes', forbidden)
    monkeypatch.setattr(watch.launch, 'lock_gpus', forbidden)
    monkeypatch.setattr(watch.launch, 'preflight', forbidden)
    monkeypatch.setattr(watch.launch, 'environment_report', forbidden)
    monkeypatch.setattr(watch.launch, 'inspect_data', forbidden)
    monkeypatch.setattr(watch, 'validate_stage2_resume', forbidden)


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    write_json(source / 'progress.json', {'logical_epoch': 300, 'phase': 'validation'})
    write_json(source / 'data_pass_state.json', {'optimizer_steps': 200})
    config = load_config()
    config['extra_args'][config['extra_args'].index('--ema-warmup-steps') + 1] = '300'
    identities = {
        role: {'pid': pid, 'start_ticks': pid * 10, 'uid': os.getuid(), 'ppid': 100,
               'cwd': str(tmp_path), 'argv': ['python', watch.TRAIN_MODULE]}
        for role, pid in [('launcher', 101), ('torchrun', 102), ('rank0', 103), ('rank1', 104)]
    }
    current = {value['pid']: deepcopy(value) for value in identities.values()}
    monkeypatch.setattr(watch, 'process_identity', current.get)
    request = {'source_run': str(source), 'output_root': str(tmp_path / 'output'),
               'boundary_epoch': 300, 'poll_seconds': 1, 'config': config,
               'source_processes': identities}
    watcher = watch.Watcher(request)
    return watcher, source, current


def validation(source, epoch):
    (source / 'validation_history.jsonl').write_text(json.dumps({'logical_epoch': epoch}) + '\n')


def test_stage_config_changes_only_ema_and_preserves_input():
    original = load_config()
    saved = deepcopy(original)
    updated = watch.stage_config(original, 300)
    expected = deepcopy(original)
    expected['extra_args'][expected['extra_args'].index('--ema-warmup-steps') + 1] = '300'
    assert updated == expected
    assert original == saved
    assert updated['extra_args'] is not original['extra_args']


@pytest.mark.parametrize('boundary', [0, 250, 1000])
def test_stage_config_rejects_unsafe_boundary(boundary):
    with pytest.raises(ValueError, match='Boundary'):
        watch.stage_config(load_config(), boundary)


@pytest.mark.parametrize('phase', ['training_row', 'validation_row', 'checkpoint_before_sidecar'])
def test_boundary_waits_for_complete_checkpoint_publication(scenario, phase):
    watcher, source, _ = scenario
    (source / 'logical_epoch_history.jsonl').write_text(json.dumps({'logical_epoch': 300}) + '\n')
    if phase != 'training_row':
        validation(source, 300)
    if phase == 'checkpoint_before_sidecar':
        (source / 'checkpoint_epoch_0300.pt').write_bytes(b'checkpoint')
    assert watcher.boundary_ready() is False
    assert watcher.state['status'] == 'waiting_epoch'


def test_boundary_requires_matching_validation(scenario):
    watcher, source, _ = scenario
    (source / 'checkpoint_epoch_0300.pt').write_bytes(b'checkpoint')
    write_json(source / 'data_pass_state.json', {'optimizer_steps': 300})
    validation(source, 200)
    with pytest.raises(RuntimeError, match='matching completed validation'):
        watcher.boundary_ready()
    validation(source, 300)
    assert watcher.boundary_ready() is True


def test_skipped_boundary_refuses_later_epoch(scenario):
    watcher, source, _ = scenario
    write_json(source / 'progress.json', {'logical_epoch': 302})
    validation(source, 300)
    with pytest.raises(RuntimeError, match='skipped'):
        watcher.boundary_ready()


@pytest.mark.parametrize('change', ['disappeared', 'reused'])
def test_lost_source_ownership_never_stops_or_launches(scenario, monkeypatch, change):
    watcher, _, current = scenario
    monkeypatch.setattr(watcher, 'guard', lambda **kwargs: None)
    if change == 'disappeared':
        current.pop(103)
    else:
        current[103]['start_ticks'] += 1
    monkeypatch.setattr(watcher, 'stop_source', lambda: pytest.fail('Ownership loss must not stop any process'))
    monkeypatch.setattr(watcher, 'start_stage2', lambda *_: pytest.fail('Ownership loss must not launch'))
    with pytest.raises(RuntimeError, match='exited|identity changed'):
        watcher.step()


@pytest.mark.parametrize('change', ['disappeared', 'reused'])
def test_send_owned_refuses_missing_or_reused_pid(scenario, change):
    watcher, _, current = scenario
    saved = watcher.request['source_processes']['torchrun']
    if change == 'disappeared':
        current.pop(saved['pid'])
        assert watch.send_owned(saved, signal.SIGTERM) is False
    else:
        current[saved['pid']]['start_ticks'] += 1
        with pytest.raises(RuntimeError, match='reused'):
            watch.send_owned(saved, signal.SIGTERM)


def test_send_owned_uses_verified_pidfd_and_closes_it(scenario, monkeypatch):
    watcher, _, _ = scenario
    saved = watcher.request['source_processes']['torchrun']
    events = []
    monkeypatch.setattr(watch, 'process_identity', lambda pid: events.append(('verify', pid)) or deepcopy(saved))
    monkeypatch.setattr(watch, 'pidfd_open', lambda pid: events.append(('open', pid)) or 987)
    monkeypatch.setattr(watch, 'pidfd_signal', lambda fd, sig: events.append(('signal', fd, sig)))
    monkeypatch.setattr(os, 'close', lambda fd: events.append(('close', fd)))
    assert watch.send_owned(saved, signal.SIGTERM) is True
    assert events == [('verify', 102), ('open', 102), ('verify', 102),
                      ('signal', 987, signal.SIGTERM), ('close', 987)]


def test_send_owned_rechecks_identity_after_opening_pidfd(scenario, monkeypatch):
    watcher, _, current = scenario
    saved = watcher.request['source_processes']['torchrun']
    closed = []

    def open_pidfd(pid):
        current[pid]['start_ticks'] += 1
        return 987

    monkeypatch.setattr(watch, 'pidfd_open', open_pidfd)
    monkeypatch.setattr(os, 'close', closed.append)
    with pytest.raises(RuntimeError, match='identity changed while opening pidfd'):
        watch.send_owned(saved, signal.SIGTERM)
    assert closed == [987]


@pytest.mark.parametrize('failure', [None, 'prepare', 'resume', 'environment', 'data', 'stop'])
def test_step_prepares_before_stop_and_stops_before_launch(scenario, monkeypatch, failure):
    watcher, source, _ = scenario
    events = []
    transition = {'validated': True, 'epoch': 300}
    data = {'root': str(source.parent / 'dataset')}
    watcher.request['source_launch'] = {'preflight': {'data': data}}
    fake_module = ModuleType(watch.__package__ + '.stage2_transition')

    def prepare(source_path, target_path, epoch):
        events.append('prepare_and_validate')
        assert source_path == source
        assert target_path == Path(watcher.state['stage2_run'])
        assert epoch == 300
        if failure == 'prepare':
            raise RuntimeError('Invalid checkpoint fork')
        return transition

    def stop():
        events.append('stop')
        assert watcher.state['transition'] == transition
        if failure == 'stop':
            raise RuntimeError('Owned source is still alive')

    def start(value):
        assert value == transition
        events.append('launch')

    def validate_resume(request, run):
        events.append('validate_resume')
        assert request is watcher.request
        assert run == Path(watcher.state['stage2_run'])
        if failure == 'resume':
            raise RuntimeError('Resume checkpoint validation failed')
        return {'status': 'passed'}

    def environment(config):
        events.append('environment')
        if failure == 'environment':
            raise RuntimeError('Environment validation failed')
        return {'status': 'passed'}

    def inspect_data(root, config):
        events.append('data')
        return {'root': 'changed'} if failure == 'data' else data

    monkeypatch.setattr(watch, 'validate_stage2_resume', validate_resume)
    monkeypatch.setattr(watch.launch, 'environment_report', environment)
    monkeypatch.setattr(watch.launch, 'inspect_data', inspect_data)
    fake_module.prepare_stage2_run = prepare
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    monkeypatch.setattr(watcher, 'guard', lambda **kwargs: events.append('guard_sources' if kwargs else 'guard'))
    monkeypatch.setattr(watcher, 'boundary_ready', lambda: True)
    monkeypatch.setattr(watcher, 'stop_source', stop)
    monkeypatch.setattr(watcher, 'start_stage2', start)
    monkeypatch.setattr(watch.launch, 'gpu_inventory', lambda: [])
    monkeypatch.setattr(watch.launch, 'gpu_processes', lambda: [])
    monkeypatch.setattr(watch.launch, 'validate_gpus', lambda *args: events.append('verify_gpus_free'))
    if failure:
        with pytest.raises(RuntimeError):
            watcher.step()
    else:
        watcher.step()
    expected = ['guard', 'guard_sources', 'prepare_and_validate']
    if failure != 'prepare':
        expected += ['validate_resume']
    if failure not in ('prepare', 'resume'):
        expected += ['environment']
    if failure not in ('prepare', 'resume', 'environment'):
        expected += ['data']
    if failure is None or failure == 'stop':
        expected += ['guard_sources', 'stop']
    if failure is None:
        expected += ['verify_gpus_free', 'launch']
    assert events == expected


def test_existing_watcher_request_rejects_duplicate_before_registration(tmp_path, monkeypatch):
    path = tmp_path / 'watcher' / 'request.json'
    write_json(path, {'existing': True})
    maker = Mock(side_effect=AssertionError('Duplicate registration must not bind live processes'))
    monkeypatch.setattr(watch, 'make_request', maker)
    with pytest.raises(FileExistsError, match='already registered'):
        watch.main(['--output-root', str(tmp_path)])
    maker.assert_not_called()
    assert watch.read_json(path) == {'existing': True}
    # The rejection also releases the watcher lock.
    fd, _ = watch.lock_watcher(tmp_path)
    os.close(fd)


def test_existing_stage2_launch_request_rejects_duplicate(scenario, tmp_path, monkeypatch):
    watcher, _, _ = scenario
    directory = tmp_path / 'launcher'
    directory.mkdir()
    write_json(directory / 'launch_request.json', {'existing': True})
    gpus = [{'index': 0, 'uuid': 'GPU-0'}, {'index': 1, 'uuid': 'GPU-1'}]
    data = {'root': str(tmp_path / 'dataset')}
    watcher.request['source_launch'] = {'preflight': {'data': data, 'gpus': gpus}}
    monkeypatch.setattr(watch.launch, 'lock_output', lambda *args: (os.open(directory / 'lock', os.O_CREAT | os.O_RDWR, 0o600), directory))
    monkeypatch.setattr(watch.launch, 'preflight', lambda *args, **kwargs: {'data': data, 'gpus': gpus})
    monkeypatch.setattr(watch.launch, 'lock_gpus', lambda *args: [])
    with pytest.raises(FileExistsError, match='already attempted'):
        watcher.start_stage2({})
    assert watch.read_json(directory / 'launch_request.json') == {'existing': True}


def test_stop_signals_only_bound_torchrun_and_waits_for_all_exits(scenario, monkeypatch):
    watcher, _, _ = scenario
    polls = iter([['launcher', 'torchrun', 'rank0', 'rank1'], ['launcher', 'rank1'], []])
    events = []
    monkeypatch.setattr(watcher, 'live_source', lambda: next(polls))
    monkeypatch.setattr(watch, 'send_owned', lambda saved, sig: events.append(('signal', saved, sig)) or True)
    monkeypatch.setattr(watch.time, 'sleep', lambda duration: events.append(('wait', duration)))
    watcher.stop_source()
    assert events == [('signal', watcher.request['source_processes']['torchrun'], signal.SIGTERM), ('wait', 2)]
    receipt = watch.read_json(watcher.directory / 'source_stop.json')
    assert receipt['epoch'] == 300
    assert receipt['processes'] == watcher.request['source_processes']


def test_stop_timeout_fails_without_further_signals(scenario, monkeypatch):
    watcher, _, _ = scenario
    clock = iter([0, 181])
    sent = []
    monkeypatch.setattr(watch.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(watch, 'send_owned', lambda saved, sig: sent.append((saved, sig)) or True)
    with pytest.raises(RuntimeError, match='did not exit'):
        watcher.stop_source()
    assert sent == [(watcher.request['source_processes']['torchrun'], signal.SIGTERM)]
    assert not (watcher.directory / 'source_stop.json').exists()
