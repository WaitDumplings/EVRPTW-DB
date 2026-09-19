"""The opt-in stop helper must never signal unverified or reused PIDs."""
import importlib.util
import json
import os
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/cus100_20260911/2080ti_4_2/stop_old_evrptw_rl.py'
spec = importlib.util.spec_from_file_location('stop_old_evr_tests', SCRIPT)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def identity(run, pid=99):
    return {'pid': pid, 'uid': os.getuid(), 'start_ticks': 123,
            'argv': ['/python', '-m', helper.MODULE, '--output-dir', str(run)]}


@pytest.mark.parametrize('change', ['owner', 'module', 'output', 'python_c'])
def test_identity_rejects_other_process(tmp_path, change):
    run = tmp_path / 'runs/TR17'
    current = identity(run)
    if change == 'owner':
        current['uid'] += 1
    elif change == 'module':
        current['argv'][2] = 'another.model'
    elif change == 'output':
        current['argv'][-1] = str(tmp_path / 'other_run')
    else:
        current['argv'][1:1] = ['-c', 'print(1)']
    with pytest.raises(RuntimeError):
        helper.validate_identity(current, run)


def test_inspect_refuses_missing_checkpoint_before_any_signal(tmp_path, monkeypatch):
    run = tmp_path / 'runs/TR17'
    run.mkdir(parents=True)
    (run / 'launch_record.json').write_text(json.dumps({'experiment_id': 'TR17', 'output_dir': str(run), 'pid': 99}))
    monkeypatch.setattr(helper, 'process', lambda pid: identity(run))
    with pytest.raises(RuntimeError, match='saved best.ckpt'):
        helper.inspect(tmp_path)


def test_reused_pid_prevents_any_signal(tmp_path, monkeypatch):
    run = tmp_path / 'runs/TR17'
    saved = identity(run)
    changed = dict(saved, start_ticks=124)
    monkeypatch.setattr(helper, 'process', lambda pid: changed)
    signals = []
    monkeypatch.setattr(helper.os, 'kill', lambda *args: signals.append(args))
    with pytest.raises(RuntimeError, match='PID was reused'):
        helper.stop_records(tmp_path, [{'experiment_id': 'TR17', 'run': str(run), 'process': saved}], 0)
    assert not signals


def test_backup_exists_before_term_and_only_verified_pid_is_targeted(tmp_path, monkeypatch):
    run = tmp_path / 'runs/TR17'
    run.mkdir(parents=True)
    for name in ('best.ckpt', 'checkpoint_latest.pt'):
        (run / name).write_bytes(b'checkpoint-snapshot')
    saved = identity(run)
    running = True
    monkeypatch.setattr(helper, 'process', lambda pid: saved if running else None)
    monkeypatch.delattr(helper.os, 'pidfd_open', raising=False)
    sent = []
    def signal_verified(pid, signal):
        nonlocal running
        archive = next((tmp_path / 'repair_backups').iterdir())
        assert (archive / 'TR17/best.ckpt').read_bytes() == b'checkpoint-snapshot'
        assert (archive / 'TR17/checkpoint_latest.pt').read_bytes() == b'checkpoint-snapshot'
        assert (archive / 'stop_request.json').is_file()
        sent.append((pid, signal))
        running = False
    monkeypatch.setattr(helper.os, 'kill', signal_verified)
    result = helper.stop_records(tmp_path, [{'experiment_id': 'TR17', 'run': str(run), 'process': saved}], 0)
    assert sent == [(99, helper.signal.SIGTERM)]
    assert result['status'] == 'stopped' and not result['still_alive']
    assert (run / 'best.ckpt').read_bytes() == b'checkpoint-snapshot'


def test_timeout_does_not_escalate_to_kill(tmp_path, monkeypatch):
    run = tmp_path / 'runs/TR18'
    run.mkdir(parents=True)
    for name in ('best.ckpt', 'checkpoint_latest.pt'):
        (run / name).write_bytes(b'checkpoint')
    saved = identity(run)
    monkeypatch.setattr(helper, 'process', lambda pid: saved)
    monkeypatch.delattr(helper.os, 'pidfd_open', raising=False)
    sent = []
    monkeypatch.setattr(helper.os, 'kill', lambda *args: sent.append(args))
    result = helper.stop_records(tmp_path, [{'experiment_id': 'TR18', 'run': str(run), 'process': saved}], 0)
    assert result['status'] == 'waiting_for_exit'
    assert sent == [(99, helper.signal.SIGTERM)]


@pytest.mark.parametrize('exits_before_term', [False, True])
def test_pidfd_targets_verified_process_and_closes_descriptor(tmp_path, monkeypatch, exits_before_term):
    run = tmp_path / 'runs/TR18'
    run.mkdir(parents=True)
    for name in ('best.ckpt', 'checkpoint_latest.pt'):
        (run / name).write_bytes(b'checkpoint')
    saved = identity(run)
    running = True
    monkeypatch.setattr(helper, 'process', lambda pid: saved if running else None)
    opened, sent, closed = [], [], []
    def open_verified(pid):
        opened.append(pid)
        return 77
    def send_verified(descriptor, signum):
        nonlocal running
        assert next((tmp_path / 'repair_backups').iterdir()).joinpath('stop_request.json').is_file()
        sent.append((descriptor, signum))
        running = False
        if exits_before_term:
            raise ProcessLookupError()
    monkeypatch.setattr(helper.os, 'pidfd_open', open_verified, raising=False)
    monkeypatch.setattr(helper.signal, 'pidfd_send_signal', send_verified, raising=False)
    monkeypatch.setattr(helper.os, 'close', lambda fd: closed.append(fd))
    monkeypatch.setattr(helper.os, 'kill', lambda *args: pytest.fail('pidfd available: must not use PID-only signal'))
    result = helper.stop_records(tmp_path, [{'experiment_id': 'TR18', 'run': str(run), 'process': saved}], 0)
    assert opened == [99] and closed == [77]
    assert sent == [(77, helper.signal.SIGTERM)]
    assert result['status'] == 'stopped'
