#!/usr/bin/env python3
"""Wait for the specified AM experiment, calibrate, and launch EVRPTW-RL once."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

if __package__:
    from .common import CONFIG, HERE, OUTPUT, REPO, digest, load_config, resolve_road_root, source_snapshot, timestamp, write_json
    from .launch import gpu_inventory, gpu_processes, last_jsonl, validate_gpus
else:
    from common import CONFIG, HERE, OUTPUT, REPO, digest, load_config, resolve_road_root, source_snapshot, timestamp, write_json
    from launch import gpu_inventory, gpu_processes, last_jsonl, validate_gpus

AM_ROOT = Path('/data/Maojie/ICLR/cus500-am-multigpu/EVRPTW_Benchmark/results/cus500_am_dual_2080ti_4_1_20260913')
AM_RUN = 'am_road_cus500_seed1234_2gpu'
AM_LAUNCHER = Path('launchers/2080ti_3_1')
TERMINAL = {'handed_off', 'failed'}


def read_json(path, *, optional=False):
    path = Path(path)
    if optional and not path.is_file():
        return None
    return json.loads(path.read_text())


def process_identity(pid):
    """Return Linux start ticks; an exited/zombie or inaccessible PID is not live."""
    try:
        pid = int(pid)
        if pid <= 0:
            return None
        base = Path('/proc') / str(pid)
        fields = (base / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': int(fields[19]),
                'command': (base / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')}
    except (OSError, ValueError, IndexError, TypeError):
        return None


def lock_watcher(output):
    directory = Path(output) / 'watcher'
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / '.watch.lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError('An after-AM watcher already owns this output directory') from None
    return fd, directory


def check_upstream_request(am_root, config, road_root=None):
    am_root = Path(am_root).resolve()
    path = am_root / AM_LAUNCHER / 'launch_request.json'
    launch = read_json(path)
    audit = launch['preflight']
    old = audit['config']
    if audit['host'] != socket.gethostname():
        raise ValueError('AM was launched on a different host')
    expected = {'gpus': [0, 1], 'world_size': 2, 'scale': 'Cus500', 'training_rollout_steps': 1700,
                'validation_rollout_steps': 2550, 'samples_per_instance': 30,
                'validation_candidates': 30, 'validation_limit': 500, 'seed': 1234,
                'minimum_training_epochs': 5000, 'training_epochs': 10000}
    for key, value in expected.items():
        if old.get(key) != value or config.get(key) != value:
            raise ValueError(f'AM/EVRPTW-RL scope mismatch: {key}')
    if old.get('run_id') != AM_RUN or Path(audit['output_dir']).resolve() != am_root / 'runs' / AM_RUN:
        raise ValueError('Unexpected AM run identity/output')
    if 'EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.distributed_train' not in launch['command']:
        raise ValueError('Upstream command is not the AM distributed trainer')
    road = Path(audit['data']['root']).resolve()
    if road_root is not None and Path(road_root).resolve() != road:
        raise ValueError('EVRPTW-RL must use the same frozen Road root as AM')
    for key, file_key in (('objective_config_sha256', 'objective_config'),
                          ('reward_contract_file_sha256', 'reward_contract')):
        if audit['data'][key] != digest(REPO / config[file_key]):
            raise ValueError(f'AM/EVRPTW-RL objective or reward differs: {key}')
    for key in ('train_index_sha256', 'validation_index_sha256'):
        if audit['data'][key] != config['expected_' + key]:
            raise ValueError(f'AM/EVRPTW-RL frozen data differs: {key}')
    return {'am_root': str(am_root), 'am_request': str(path), 'am_request_sha256': digest(path),
            'am_source_sha256': launch['source']['source_sha256'], 'road_root': str(road),
            'am_gpu_uuids': {str(gpu['index']): gpu['uuid'] for gpu in audit['gpus']}}


def make_request(config_path, am_root, output, road_root, poll_seconds):
    config = load_config(config_path, gpus='0,1')
    upstream = check_upstream_request(am_root, config, road_root)
    return {'schema': 'cus500_evrptw_rl_after_am_request_v1', 'created_at': timestamp(),
            'host': socket.gethostname(), 'config_path': str(Path(config_path).resolve()),
            'config_file_sha256': digest(config_path), 'config': config,
            'output_root': str(Path(output).resolve()), 'poll_seconds': float(poll_seconds),
            'source_sha256': source_snapshot()['source_sha256'], **upstream}


def formal_result(run, method, minimum=5000, maximum=10000):
    result = read_json(Path(run) / 'training_result.json')
    epoch = result.get('completed_training_epochs')
    if (result.get('status') not in {'passed', 'early_stopped'} or result.get('method') != method
            or not isinstance(epoch, int) or isinstance(epoch, bool) or not minimum <= epoch <= maximum):
        raise RuntimeError(f'{method} has no successful formal training result in the required epoch budget')
    for name in ('best.ckpt', 'checkpoint_latest.pt'):
        path = Path(run) / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f'{method} completion is missing a nonempty {name}')
    return result


def final_config_compatible(template, final):
    adjustable = {'physical_batch_size', 'effective_batch_size', 'gradient_accumulation_steps',
                  'sample_count', 'customer_exposure_budget', 'calibration_status'}
    for key, value in template.items():
        if key not in adjustable and final.get(key) != value:
            raise ValueError(f'Calibrated config changed the frozen experiment: {key}')
    if not str(final.get('calibration_status', '')).startswith('passed'):
        raise ValueError('Formal launch requires a passed calibration status')


class Watcher:
    def __init__(self, request, *, calibrator=None, starter=None):
        self.request = request
        self.output = Path(request['output_root'])
        self.directory = self.output / 'watcher'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'status.json'
        self.state = read_json(self.path, optional=True) or {
            'schema': 'cus500_evrptw_rl_after_am_status_v1', 'status': 'waiting_am',
            'created_at': timestamp(), 'am_processes': {}, 'launch_attempted': False}
        self.state.update(host=socket.gethostname(), watcher_pid=os.getpid(),
                          am_root=request['am_root'], output_root=str(self.output))
        self.mutex = threading.RLock()
        self.calibrator = calibrator
        self.starter = starter
        self.update()

    def update(self, **values):
        with self.mutex:
            self.state.update(values, heartbeat=timestamp())
            write_json(self.path, self.state)

    def guard(self, *, source=False):
        r = self.request
        if socket.gethostname() != r['host']:
            raise RuntimeError('Watcher host changed')
        if digest(r['am_request']) != r['am_request_sha256']:
            raise RuntimeError('AM launch request changed after watcher registration')
        if digest(r['config_path']) != r['config_file_sha256']:
            raise RuntimeError('EVRPTW-RL template config changed after watcher registration')
        if source and source_snapshot()['source_sha256'] != r['source_sha256']:
            raise RuntimeError('EVRPTW-RL source changed after watcher registration')

    def remember_am_processes(self, status, progress):
        observations = [('launcher', status.get('launcher_pid')), ('torchrun', status.get('pid'))]
        observations.extend((f"rank{item.get('rank')}", item.get('pid')) for item in (progress or {}).get('workers', []))
        known = dict(self.state.get('am_processes', {}))
        launcher_alive = False
        for role, pid in observations:
            identity = process_identity(pid)
            if identity is None:
                continue
            token = ('cus500_am_multigpu_20260912/launch.py' if role == 'launcher'
                     else 'AM_EVRPTW.distributed_train')
            if token not in identity['command']:
                continue
            known[f"{identity['pid']}:{identity['start_ticks']}"] = {k: identity[k] for k in ('pid', 'start_ticks')}
            if role == 'launcher':
                launcher_alive = True
        self.update(am_processes=known)
        live = []
        for saved in known.values():
            now = process_identity(saved['pid'])
            if now is not None and now['start_ticks'] == saved['start_ticks']:
                live.append(saved['pid'])
        return launcher_alive, live

    def am_completed(self):
        root = Path(self.request['am_root'])
        status = read_json(root / AM_LAUNCHER / 'status.json', optional=True)
        if status is None:
            raise RuntimeError('AM launcher status is missing; refusing to infer completion from empty GPUs')
        if status.get('host') != self.request['host']:
            raise RuntimeError('AM status host differs from the registered host')
        run = root / 'runs' / AM_RUN
        if Path(status.get('output_dir', '')).resolve() != run:
            raise RuntimeError('AM status points to a different run')
        progress = read_json(run / 'progress.json', optional=True)
        launcher_alive, live = self.remember_am_processes(status, progress)
        if status.get('status') == 'completed':
            if status.get('returncode') != 0:
                raise RuntimeError('AM completed status has a nonzero return code')
            result = formal_result(run, 'AM-EVRPTW')
            if live:
                self.update(status='waiting_am_exit', am_live_pids=live)
                return False
            self.update(am_completed_training_epochs=result['completed_training_epochs'], am_live_pids=[])
            return True
        if status.get('status') == 'failed':
            raise RuntimeError('AM failed; EVRPTW-RL will not start automatically')
        if status.get('status') not in {'starting', 'running'} or not launcher_alive:
            raise RuntimeError('AM launcher disappeared without a successful completion record')
        self.update(status='waiting_am', am_progress=progress, am_live_pids=live)
        return False

    def gpus_ready(self):
        inventory = gpu_inventory()
        selected = {str(row['index']): row for row in inventory if row['index'] in (0, 1)}
        for index, expected in self.request['am_gpu_uuids'].items():
            if index not in selected or selected[index]['uuid'] != expected:
                raise RuntimeError('Physical GPU identity changed since the AM launch')
        try:
            validate_gpus(self.request['config'], inventory, gpu_processes())
        except RuntimeError as error:
            self.update(status='waiting_gpus', waiting_reason=str(error))
            return False
        return True

    def downstream_paths(self):
        cfg = self.request['config']
        launcher = self.output / 'launchers' / cfg['server'] / cfg['model']
        return launcher, self.output / 'runs' / cfg['run_id']

    def observe_downstream(self):
        launcher, run = self.downstream_paths()
        launch = read_json(launcher / 'launch_request.json', optional=True)
        status = read_json(launcher / 'status.json', optional=True)
        if launch is None:
            if status is not None or (run.exists() and any(run.iterdir())):
                raise RuntimeError('EVRPTW-RL output exists without its launch request; no automatic overwrite/resume')
            if self.state.get('launch_attempted'):
                raise RuntimeError('A prior EVRPTW-RL launch attempt left no request; inspect before retrying')
            return False
        if launch['source']['source_sha256'] != self.request['source_sha256']:
            raise RuntimeError('Existing EVRPTW-RL run was launched from different source')
        if launch['preflight']['host'] != self.request['host'] or Path(launch['preflight']['output_dir']).resolve() != run:
            raise RuntimeError('Existing EVRPTW-RL launch has a different host or output')
        final_config_compatible(self.request['config'], launch['preflight']['config'])
        self.update(status='confirming_evrptw_rl_start', launch_attempted=True,
                    downstream_status_file=str(launcher / 'status.json'))
        if status is None:
            started = float(self.state.get('launch_attempt_time', time.time()))
            self.update(launch_attempt_time=started)
            if time.time() - started > 600:
                raise RuntimeError('EVRPTW-RL launcher produced no status within ten minutes')
            return True
        if status.get('status') == 'failed':
            raise RuntimeError('EVRPTW-RL launch/training failed; watcher will not silently resume it')
        if status.get('status') == 'completed':
            if status.get('returncode') != 0:
                raise RuntimeError('EVRPTW-RL completion has a nonzero return code')
            formal_result(run, 'EVRPTW-RL')
        elif status.get('status') in {'running', 'starting'}:
            identity = process_identity(status.get('launcher_pid'))
            if identity is None or 'cus500_evrptw_rl_20260913/launch.py' not in identity['command']:
                raise RuntimeError('EVRPTW-RL launcher disappeared before handoff')
        else:
            raise RuntimeError(f"Unexpected EVRPTW-RL status: {status.get('status')}")
        progress = read_json(run / 'progress.json', optional=True) or {}
        history = last_jsonl(run / 'logical_epoch_history.jsonl') or {}
        validation = last_jsonl(run / 'validation_history.jsonl') or {}
        completed = int(progress.get('completed_logical_epoch', 0))
        progressed = (completed >= 1 or int(history.get('logical_epoch', 0)) >= 1
                      or int(validation.get('logical_epoch', 0)) >= 1
                      or (progress.get('phase') == 'validation' and int(progress.get('logical_epoch', 0)) >= 1)
                      or status.get('status') == 'completed')
        if progressed:
            self.update(status='handed_off', handed_off_at=timestamp(), downstream_progress=progress,
                        downstream_latest_epoch=history.get('logical_epoch', completed),
                        downstream_launcher_pid=status.get('launcher_pid'), downstream_torchrun_pid=status.get('pid'))
        return True

    def step(self):
        if self.state['status'] in TERMINAL:
            return self.state['status']
        self.guard()
        if self.observe_downstream():
            return self.state['status']
        if not self.am_completed() or not self.gpus_ready():
            return self.state['status']
        self.guard(source=True)
        path = self.state.get('calibrated_config_path')
        if path:
            if digest(path) != self.state['calibrated_config_sha256']:
                raise RuntimeError('Previously calibrated config changed')
        else:
            self.update(status='calibrating', calibration_started_at=timestamp())
            calibrator = self.calibrator
            if calibrator is None:
                if __package__:
                    from .autocalibrate import calibrate
                else:
                    from autocalibrate import calibrate
                calibrator = calibrate
            path = Path(calibrator(self.request['config'], Path(self.request['road_root']), self.output,
                                   expected_source_sha256=self.request['source_sha256'])).resolve()
            self.update(calibrated_config_path=str(path), calibrated_config_sha256=digest(path))
        final = load_config(path, gpus='0,1')
        final_config_compatible(self.request['config'], final)
        self.guard(source=True)
        if not self.gpus_ready():
            return self.state['status']
        self.update(status='launching_evrptw_rl', launch_attempted=True, launch_attempt_time=time.time())
        command = [sys.executable, str(HERE / 'launch.py'), '--config', str(path), '--gpus', '0,1',
                   '--batch-size', str(final['physical_batch_size']), '--accumulation-steps', str(final['gradient_accumulation_steps']),
                   '--instance-cache-size', str(final['instance_cache_size']), '--road-root', self.request['road_root'],
                   '--output-root', str(self.output)]
        self.update(launch_command=command)
        if self.starter is not None:
            self.starter(command)
        else:
            with (self.directory / 'launch.log').open('ab') as log:
                subprocess.run(command, cwd=REPO, stdout=log, stderr=log, check=True)
        self.observe_downstream()
        return self.state['status']

    def run(self):
        stop = threading.Event()
        def heartbeat():
            while not stop.wait(self.request['poll_seconds']):
                self.update()
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            while self.state['status'] not in TERMINAL:
                self.step()
                if self.state['status'] not in TERMINAL:
                    time.sleep(self.request['poll_seconds'])
            return 0 if self.state['status'] == 'handed_off' else 1
        except Exception as error:
            self.update(status='failed', error=f'{type(error).__name__}: {error}', failed_at=timestamp())
            return 1
        finally:
            stop.set()
            thread.join(timeout=1)


def worker(request_path, lock_fd):
    os.fstat(lock_fd)
    try:
        return Watcher(read_json(request_path)).run()
    finally:
        os.close(lock_fd)


def start(args):
    output = args.output_root.expanduser().resolve()
    fd, directory = lock_watcher(output)
    handed_off = False
    try:
        candidate = make_request(args.config, args.am_root, output, args.road_root, args.poll_seconds)
        path = directory / 'request.json'
        if path.is_file():
            request = read_json(path)
            for key in ('host', 'config', 'config_file_sha256', 'output_root', 'am_root', 'am_request_sha256', 'source_sha256', 'road_root'):
                if request[key] != candidate[key]:
                    raise ValueError(f'Watcher restart differs from its original request: {key}')
        else:
            write_json(path, candidate)
        if args.foreground:
            handed_off = True
            return worker(path, fd)
        with (directory / 'watcher.log').open('ab') as log:
            child = subprocess.Popen([sys.executable, str(HERE / 'watch_after_am.py'), '--mode', 'worker',
                                      '--request', str(path), '--lock-fd', str(fd)], cwd=REPO,
                                     stdout=log, stderr=log, start_new_session=True, pass_fds=(fd,))
        print(json.dumps({'watcher_pid': child.pid, 'status_file': str(directory / 'status.json'),
                          'am_root': str(args.am_root), 'poll_seconds': args.poll_seconds}, indent=2), flush=True)
        return 0
    finally:
        if not handed_off:
            os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('start', 'worker', 'status'), default='start')
    parser.add_argument('--am-root', type=Path, default=AM_ROOT)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--road-root', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path(os.environ.get('CUS500_OUTPUT_ROOT', OUTPUT)))
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--foreground', action='store_true')
    parser.add_argument('--request', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.poll_seconds <= 60:
        parser.error('--poll-seconds must be in (0, 60]')
    if args.mode == 'status':
        path = args.output_root.expanduser().resolve() / 'watcher/status.json'
        print(json.dumps(read_json(path, optional=True) or {'status': 'not_started', 'status_file': str(path)}, indent=2))
        return 0
    if args.mode == 'worker':
        if args.request is None or args.lock_fd is None:
            parser.error('worker requires an inherited watcher lock and request')
        return worker(args.request, args.lock_fd)
    return start(args)


if __name__ == '__main__':
    raise SystemExit(main())
