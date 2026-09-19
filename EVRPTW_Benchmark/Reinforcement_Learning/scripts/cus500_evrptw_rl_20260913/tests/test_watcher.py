from __future__ import annotations

import copy
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913 import watch_after_am as watch
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913.common import CONFIG, REPO, digest, load_config, write_json


def gpu(index):
    return {'index': index, 'uuid': f'GPU-{index}', 'name': 'NVIDIA GeForce RTX 2080 Ti',
            'memory.used': 10, 'memory.total': 11264}


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    config_path = tmp_path / 'template.json'
    config_path.write_bytes(CONFIG.read_bytes())
    config = load_config(config_path)
    root = tmp_path / 'am'
    run = root / 'runs' / watch.AM_RUN
    launcher = root / watch.AM_LAUNCHER
    run.mkdir(parents=True)
    launcher.mkdir(parents=True)
    amconfig = {**config, 'run_id': watch.AM_RUN}
    road = tmp_path / 'road'
    upstream = {'preflight': {'host': socket.gethostname(), 'config': amconfig,
        'output_dir': str(run), 'gpus': [gpu(0), gpu(1)],
        'data': {'root': str(road), 'objective_config_sha256': digest(REPO / config['objective_config']),
                 'reward_contract_file_sha256': digest(REPO / config['reward_contract']),
                 'train_index_sha256': config['expected_train_index_sha256'],
                 'validation_index_sha256': config['expected_validation_index_sha256']}},
        'source': {'source_sha256': 'am-source'},
        'command': ['python', '--module', 'EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.distributed_train']}
    write_json(launcher / 'launch_request.json', upstream)
    amstatus = {'status': 'running', 'returncode': None, 'host': socket.gethostname(),
                'output_dir': str(run), 'launcher_pid': 101, 'pid': 102}
    write_json(launcher / 'status.json', amstatus)
    write_json(run / 'progress.json', {'logical_epoch': 100, 'workers': [{'rank': 0, 'pid': 103}, {'rank': 1, 'pid': 104}]})
    identities = {pid: {'pid': pid, 'start_ticks': pid * 10,
                  'command': 'cus500_am_multigpu_20260912/launch.py' if pid == 101 else 'AM_EVRPTW.distributed_train'}
                  for pid in (101, 102, 103, 104)}
    original_identity = watch.process_identity
    monkeypatch.setattr(watch, 'process_identity', lambda pid: identities.get(pid) if pid in range(100, 300) else original_identity(pid))
    monkeypatch.setattr(watch, 'source_snapshot', lambda: {'source_sha256': 'source'})
    monkeypatch.setattr(watch, 'gpu_inventory', lambda: [gpu(0), gpu(1), gpu(2)])
    monkeypatch.setattr(watch, 'gpu_processes', lambda: [])
    request = watch.make_request(config_path, root, tmp_path / 'output', road, .02)
    calls = {'calibrate': 0, 'start': 0}

    def completed(**changes):
        amstatus.update(status='completed', returncode=0)
        write_json(launcher / 'status.json', amstatus)
        result = {'status': 'passed', 'method': 'AM-EVRPTW', 'completed_training_epochs': 5500, **changes}
        write_json(run / 'training_result.json', result)
        (run / 'best.ckpt').write_bytes(b'best')
        (run / 'checkpoint_latest.pt').write_bytes(b'latest')
        identities.clear()

    def calibrate(cfg, road_root, output_root, *, expected_source_sha256):
        calls['calibrate'] += 1
        assert expected_source_sha256 == 'source'
        assert road_root == road
        final = copy.deepcopy(cfg)
        final.update(physical_batch_size=2, calibration_status='passed_fake_cpu_test')
        path = output_root / 'calibrated_config.json'
        write_json(path, final)
        return path

    def start(command, *, progress=True, failed=False):
        calls['start'] += 1
        assert command[command.index('--gpus') + 1] == '0,1'
        assert command[command.index('--batch-size') + 1] == '2'
        assert command[command.index('--accumulation-steps') + 1] == '1'
        assert command[command.index('--instance-cache-size') + 1] == '256'
        final = load_config(command[command.index('--config') + 1])
        output = Path(request['output_root'])
        down = output / 'launchers' / final['server'] / final['model']
        target = output / 'runs' / final['run_id']
        target.mkdir(parents=True, exist_ok=True)
        write_json(down / 'launch_request.json', {'source': {'source_sha256': 'source'},
                   'preflight': {'host': request['host'], 'output_dir': str(target), 'config': final}})
        write_json(down / 'status.json', {'status': 'failed' if failed else 'running', 'launcher_pid': 201, 'pid': 202})
        identities[201] = {'pid': 201, 'start_ticks': 2010, 'command': 'cus500_evrptw_rl_20260913/launch.py'}
        if progress:
            write_json(target / 'progress.json', {'status': 'running', 'logical_epoch': 2, 'completed_logical_epoch': 1})

    return {'request': request, 'calls': calls, 'completed': completed, 'calibrate': calibrate, 'start': start,
            'identities': identities, 'run': run, 'launcher': launcher, 'amstatus': amstatus,
            'upstream': upstream, 'config_path': config_path}


def watcher(s):
    return watch.Watcher(s['request'], calibrator=s['calibrate'], starter=s['start'])


def test_running_am_waits_without_gpu_or_launch(scenario, monkeypatch):
    monkeypatch.setattr(watch, 'gpu_inventory', lambda: pytest.fail('No GPU check before AM success'))
    w = watcher(scenario)
    assert w.step() == 'waiting_am'
    assert scenario['calls'] == {'calibrate': 0, 'start': 0}
    assert watch.read_json(w.path)['am_live_pids'] == [101, 102, 103, 104]


def test_failed_am_never_triggers(scenario):
    scenario['amstatus']['status'] = 'failed'
    write_json(scenario['launcher'] / 'status.json', scenario['amstatus'])
    w = watcher(scenario)
    assert w.run() == 1
    assert w.state['status'] == 'failed'
    assert scenario['calls'] == {'calibrate': 0, 'start': 0}


def test_missing_am_launcher_without_completion_fails(scenario):
    scenario['identities'].pop(101)
    w = watcher(scenario)
    assert w.run() == 1
    assert 'disappeared' in w.state['error']
    assert scenario['calls']['start'] == 0


@pytest.mark.parametrize('changes', [dict(status='pilot_partial'), dict(method='EVRPTW-RL'),
                                     dict(completed_training_epochs=4999), dict(completed_training_epochs=10001)])
def test_am_requires_formal_budget_and_method(scenario, changes):
    scenario['completed'](**changes)
    w = watcher(scenario)
    assert w.run() == 1
    assert scenario['calls']['calibrate'] == 0


def test_am_completed_waits_for_all_workers_to_exit(scenario):
    saved = scenario['identities'][103]
    scenario['completed']()
    scenario['identities'][103] = saved
    w = watcher(scenario)
    assert w.step() == 'waiting_am_exit'
    assert scenario['calls']['start'] == 0
    scenario['identities'].clear()
    assert w.step() == 'handed_off'


def test_pid_reuse_does_not_keep_old_am_alive(scenario):
    w = watcher(scenario)
    assert w.step() == 'waiting_am'
    scenario['completed']()
    scenario['identities'][101] = {'pid': 101, 'start_ticks': 9999, 'command': 'unrelated-service'}
    assert w.step() == 'handed_off'


def test_gpu_occupied_waits_without_stopping_process(scenario, monkeypatch):
    scenario['completed']()
    monkeypatch.setattr(watch, 'gpu_processes', lambda: [{'gpu_uuid': 'GPU-1', 'pid': 777, 'used_memory_mib': 100,
                                                        'executable': '/bin/python'}])
    w = watcher(scenario)
    assert w.step() == 'waiting_gpus'
    assert scenario['calls'] == {'calibrate': 0, 'start': 0}
    monkeypatch.setattr(watch, 'gpu_processes', lambda: [])
    assert w.step() == 'handed_off'
    assert scenario['calls'] == {'calibrate': 1, 'start': 1}


def test_source_change_refuses_calibration(scenario, monkeypatch):
    scenario['completed']()
    monkeypatch.setattr(watch, 'source_snapshot', lambda: {'source_sha256': 'changed'})
    w = watcher(scenario)
    assert w.run() == 1
    assert 'source changed' in w.state['error']
    assert scenario['calls'] == {'calibrate': 0, 'start': 0}


def test_am_request_change_refuses_even_while_waiting(scenario):
    path = scenario['launcher'] / 'launch_request.json'
    path.write_text(path.read_text() + '\n')
    w = watcher(scenario)
    assert w.run() == 1
    assert 'AM launch request changed' in w.state['error']


def test_completed_am_launches_once_then_restart_observes(scenario):
    scenario['completed']()
    w = watcher(scenario)
    assert w.run() == 0
    assert scenario['calls'] == {'calibrate': 1, 'start': 1}
    # Simulate daemon death before it wrote its handoff state.
    w.path.unlink()
    again = watcher(scenario)
    assert again.run() == 0
    assert scenario['calls'] == {'calibrate': 1, 'start': 1}


def test_launch_waits_for_actual_optimizer_progress(scenario):
    scenario['completed']()
    w = watch.Watcher(scenario['request'], calibrator=scenario['calibrate'],
                      starter=lambda command: scenario['start'](command, progress=False))
    assert w.step() == 'confirming_evrptw_rl_start'
    assert w.step() == 'confirming_evrptw_rl_start'
    assert scenario['calls']['start'] == 1
    _, run = w.downstream_paths()
    write_json(run / 'progress.json', {'logical_epoch': 1, 'completed_logical_epoch': 0, 'phase': 'training'})
    assert w.step() == 'confirming_evrptw_rl_start'
    write_json(run / 'progress.json', {'logical_epoch': 1, 'completed_logical_epoch': 0, 'phase': 'validation'})
    assert w.step() == 'handed_off'


def test_failed_downstream_is_not_automatically_resumed(scenario):
    scenario['completed']()
    w = watch.Watcher(scenario['request'], calibrator=scenario['calibrate'],
                      starter=lambda command: scenario['start'](command, failed=True))
    assert w.run() == 1
    assert 'will not silently resume' in w.state['error']
    w.path.unlink()
    again = watcher(scenario)
    assert again.run() == 1
    assert scenario['calls']['start'] == 1


def test_heartbeat_continues_during_calibration(scenario):
    scenario['completed']()
    observed = []
    def calibrate(*args, **kwargs):
        observed.append(watch.read_json(Path(scenario['request']['output_root']) / 'watcher/status.json')['heartbeat'])
        time.sleep(.08)
        observed.append(watch.read_json(Path(scenario['request']['output_root']) / 'watcher/status.json')['heartbeat'])
        return scenario['calibrate'](*args, **kwargs)
    w = watch.Watcher(scenario['request'], calibrator=calibrate, starter=scenario['start'])
    assert w.run() == 0
    assert observed[0] != observed[1]


def test_watcher_lock_prevents_duplicate_daemon(tmp_path):
    fd, _ = watch.lock_watcher(tmp_path)
    try:
        with pytest.raises(RuntimeError, match='already owns'):
            watch.lock_watcher(tmp_path)
    finally:
        os.close(fd)
    fd, _ = watch.lock_watcher(tmp_path)
    os.close(fd)


def test_scope_registration_rejects_wrong_gpu_pair(scenario):
    scenario['upstream']['preflight']['config']['gpus'] = [1, 2]
    write_json(scenario['launcher'] / 'launch_request.json', scenario['upstream'])
    with pytest.raises(ValueError, match='gpus'):
        watch.make_request(scenario['config_path'], Path(scenario['request']['am_root']),
                           Path(scenario['request']['output_root']), None, 30)


def test_real_cpu_child_advances_before_handoff(scenario):
    scenario['completed']()
    children = []
    def starter(command):
        scenario['start'](command, progress=False)
        final = load_config(command[command.index('--config') + 1])
        output = Path(scenario['request']['output_root'])
        down = output / 'launchers' / final['server'] / final['model']
        target = output / 'runs' / final['run_id']
        code = """import json,os,sys,time
from pathlib import Path
status,progress,ready=map(Path,sys.argv[1:4])
status.write_text(json.dumps({'status':'running','launcher_pid':os.getpid(),'pid':os.getpid()}))
ready.write_text('ready')
time.sleep(.1)
progress.write_text(json.dumps({'status':'running','logical_epoch':2,'completed_logical_epoch':1}))
time.sleep(.3)
"""
        ready = output / 'ready'
        child = subprocess.Popen([sys.executable, '-c', code, str(down / 'status.json'), str(target / 'progress.json'),
                                  str(ready), 'cus500_evrptw_rl_20260913/launch.py'])
        children.append(child)
        deadline = time.monotonic() + 5
        while not ready.is_file() and time.monotonic() < deadline:
            time.sleep(.005)
        assert ready.is_file()
    w = watch.Watcher(scenario['request'], calibrator=scenario['calibrate'], starter=starter)
    try:
        assert w.run() == 0
        assert w.state['status'] == 'handed_off'
        assert w.state['downstream_progress']['completed_logical_epoch'] == 1
        assert scenario['calls']['start'] == 1
    finally:
        for child in children:
            assert child.wait(timeout=5) == 0


def test_background_start_inherits_lock_and_publishes_failure_status(scenario, tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    scenario['amstatus']['status'] = 'failed'
    write_json(scenario['launcher'] / 'status.json', scenario['amstatus'])
    bootstrap = tmp_path / 'bootstrap'
    bootstrap.mkdir()
    (bootstrap / 'watch_after_am.py').write_text(
        'import sys,time\n'
        f'sys.path.insert(0, {str(REPO)!r})\n'
        'time.sleep(.2)\n'
        'from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913 import watch_after_am\n'
        'raise SystemExit(watch_after_am.main())\n'
    )
    monkeypatch.setattr(watch, 'HERE', bootstrap)
    output = Path(scenario['request']['output_root'])
    args = SimpleNamespace(output_root=output, config=scenario['config_path'],
                           am_root=Path(scenario['request']['am_root']), road_root=None,
                           poll_seconds=.02, foreground=False)
    assert watch.start(args) == 0
    report = json.loads(capsys.readouterr().out)
    with pytest.raises(RuntimeError, match='already owns'):
        watch.lock_watcher(output)
    deadline = time.monotonic() + 5
    state = None
    while time.monotonic() < deadline:
        state = watch.read_json(output / 'watcher/status.json', optional=True)
        if state and state['status'] == 'failed':
            break
        time.sleep(.01)
    assert state and state['status'] == 'failed'
    assert 'AM failed' in state['error']
    # The child really ran the public worker entry with an inherited descriptor.
    try:
        _, exitstatus = os.waitpid(report['watcher_pid'], 0)
        assert os.waitstatus_to_exitcode(exitstatus) == 1
    except ChildProcessError:
        pass  # subprocess cleanup may already have reaped this completed child.
    fd, _ = watch.lock_watcher(output)
    os.close(fd)
    assert scenario['calls'] == {'calibrate': 0, 'start': 0}
