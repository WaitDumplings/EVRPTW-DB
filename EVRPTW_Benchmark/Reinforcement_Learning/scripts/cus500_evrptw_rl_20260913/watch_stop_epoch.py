#!/usr/bin/env python3
"""Stop the owned three-GPU Road500 run after a durable validation checkpoint."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

if __package__:
    from .common import HERE, REPO, digest, source_snapshot, timestamp, write_json
    from .watch_stage2 import TRAIN_MODULE, option, process_identity, same_process, send_owned
    from .launch import last_jsonl
else:
    from common import HERE, REPO, digest, source_snapshot, timestamp, write_json
    from watch_stage2 import TRAIN_MODULE, option, process_identity, same_process, send_owned
    from launch import last_jsonl

SOURCE_ROOT = Path('/data/Maojie/ICLR/cus500-evrptw-stage2-3gpu/EVRPTW_Benchmark/results/cus500_evrptw_rl_stage2_3gpu_20260914')
DEFAULT_SOURCE = SOURCE_ROOT / 'launchers/local_stage2_gpu012/evrptw_rl/launch_request.json'
DEFAULT_OUTPUT = SOURCE_ROOT / 'stop_after_epoch900'


def read_json(path):
    return json.loads(Path(path).read_text())


def bind_source(request_path, target_epoch):
    request_path = Path(request_path).resolve()
    launch = read_json(request_path)
    audit = launch['preflight']
    config = audit['config']
    if audit['host'] != socket.gethostname():
        raise RuntimeError('Source request belongs to another host')
    if (config['train_module'] != TRAIN_MODULE or config['scale'] != 'Cus500'
            or config['gpus'] != [0, 1, 2] or config['world_size'] != 3
            or config['effective_batch_size'] != 72):
        raise ValueError('Expected the three-GPU EVRPTW-RL Road500 continuation')
    if target_epoch != 900:
        raise ValueError('This deployment is restricted to the requested epoch 900')
    run = Path(audit['output_dir']).resolve()
    status = read_json(request_path.parent / 'status.json')
    progress = read_json(run / 'progress.json')
    if status['status'] != 'running' or Path(status['output_dir']).resolve() != run:
        raise RuntimeError('Source launcher is not running the requested experiment')
    if int(progress['logical_epoch']) > target_epoch + 1:
        raise RuntimeError('Run already passed epoch 900; refusing an implicit rollback')
    roles = [('launcher', status['launcher_pid']), ('torchrun', status['pid'])]
    roles += [(f"rank{row['rank']}", row['pid']) for row in progress['workers']]
    if sorted(role for role, _ in roles) != ['launcher', 'rank0', 'rank1', 'rank2', 'torchrun']:
        raise RuntimeError('Expected one launcher, torchrun, and three ranks')
    known = {}
    for role, pid in roles:
        identity = process_identity(pid)
        if identity is None or identity['uid'] != os.getuid():
            raise RuntimeError(f'Cannot bind owned {role}')
        if role == 'launcher':
            if not any(arg.endswith('/cus500_evrptw_rl_20260913/launch.py') for arg in identity['argv']):
                raise RuntimeError('Source launcher executable differs')
            if Path(option(identity['argv'], '--request')).resolve() != request_path:
                raise RuntimeError('Source launcher request differs')
        elif TRAIN_MODULE not in identity['argv'] or Path(option(identity['argv'], '--output-dir')).resolve() != run:
            raise RuntimeError(f'{role} is not the requested run')
        known[role] = identity
    if (known['torchrun']['argv'] != launch['command']
            or known['torchrun']['ppid'] != known['launcher']['pid']
            or any(known[role]['ppid'] != known['torchrun']['pid'] for role in ('rank0', 'rank1', 'rank2'))
            or len({p['cwd'] for p in known.values()}) != 1):
        raise RuntimeError('Source command, ancestry or working directories differ')
    return {'schema': 'evrptw_rl_stop_epoch_request_v1', 'host': socket.gethostname(),
            'created_at': timestamp(), 'source_request': str(request_path),
            'source_request_sha256': digest(request_path), 'source_run': str(run),
            'source_processes': known, 'source_config': config,
            'target_epoch': target_epoch, 'physical_gpus': config['gpus'],
            'gpu_uuids': {str(gpu['index']): gpu['uuid'] for gpu in audit['gpus']},
            'watcher_source_sha256': source_snapshot()['source_sha256']}


def validate_checkpoint(run, epoch):
    """A validation log row alone is insufficient: require the final sidecar."""
    run = Path(run)
    path = run / f'checkpoint_epoch_{epoch:04d}.pt'
    state = read_json(run / 'data_pass_state.json')
    if not path.is_file() or state.get('optimizer_steps', -1) < epoch:
        return None
    if state['optimizer_steps'] != epoch:
        raise RuntimeError('Committed state is ahead of requested stopping epoch')
    val = last_jsonl(run / 'validation_history.jsonl') or {}
    if val.get('logical_epoch') != epoch or val.get('instances') != 500:
        raise RuntimeError('Checkpoint has no matching completed 500-instance validation')
    latest = run / 'checkpoint_latest.pt'
    sha = digest(path)
    if not latest.is_file() or digest(latest) != sha:
        raise RuntimeError('Epoch artifact and latest checkpoint differ')
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False)
    args = payload['args']
    cursor = 14400 + (epoch - 300) * 72
    if (payload.get('method') != 'EVRPTW-RL' or payload.get('logical_epoch') != epoch
            or payload.get('stream_cursor') != cursor
            or payload.get('completed_validation_checks') != epoch // 100
            or payload.get('distributed_contract', {}).get('world_size') != 3
            or len(payload.get('rank_rng_states', [])) != 3
            or args.get('ema_warmup_steps') != 300 or args.get('effective_batch_size') != 72
            or args.get('scale') != 'Cus500'
            or not payload.get('model') or not payload.get('optimizer', {}).get('state')):
        raise RuntimeError('Checkpoint method, epoch, topology or training state differs')
    saved = payload['data_pass_state']
    if (saved != state or saved.get('optimizer_steps') != epoch
            or saved.get('instances_seen') != cursor or saved.get('customer_exposures') != cursor * 500):
        raise RuntimeError('Checkpoint embedded state and committed sidecar disagree')
    return {'checkpoint': str(path), 'checkpoint_sha256': sha, 'logical_epoch': epoch,
            'stream_cursor': cursor, 'customer_exposures': cursor * 500,
            'validation_cost_usd': val.get('mean_verified_cost_usd'),
            'validation_feasible': val.get('complete_and_feasible'), 'validation_instances': val['instances']}


def archive_boundary(run, directory, checked):
    """Preserve the committed checkpoint and selections before signalling."""
    run, destination = Path(run), Path(directory) / 'checkpoint_backup'
    destination.mkdir(exist_ok=False)
    names = [Path(checked['checkpoint']).name, 'data_pass_state.json',
             'best.ckpt', 'best_overall.ckpt', 'best_within_5000.ckpt',
             'validation_summary.json', 'validation_summary_overall.json',
             'validation_summary_within_5000.json', 'stage2_transition.json']
    records = []
    for name in names:
        source = run / name
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f'Required regular boundary artifact missing: {source}')
        before = digest(source)
        target = destination / name
        shutil.copy2(source, target)
        if before != digest(target) or before != digest(source):
            raise RuntimeError(f'Artifact changed during backup: {name}')
        with target.open('rb') as handle:
            os.fsync(handle.fileno())
        records.append({'name': name, 'sha256': before, 'bytes': target.stat().st_size})
    write_json(destination / 'manifest.json', {'time': timestamp(), 'source_run': str(run),
                                              'boundary': checked, 'files': records})
    fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return str(destination)


class Watcher:
    def __init__(self, request, directory):
        self.request = request
        self.directory = Path(directory)
        self.state = {'schema': 'evrptw_rl_stop_epoch_status_v1', 'status': 'waiting_epoch',
                      'watcher_pid': os.getpid(), 'source_run': request['source_run'],
                      'target_epoch': request['target_epoch'], 'physical_gpus': request['physical_gpus']}
        self.update()

    def update(self, **values):
        self.state.update(values, heartbeat=timestamp())
        write_json(self.directory / 'status.json', self.state)

    def guard(self):
        if (self.request['host'] != socket.gethostname()
                or digest(self.request['source_request']) != self.request['source_request_sha256']):
            raise RuntimeError('Registered host or source launch request changed')

    def live_source(self):
        result = []
        for role, saved in self.request['source_processes'].items():
            current = process_identity(saved['pid'])
            if current is None:
                continue
            if not same_process(saved, current):
                raise RuntimeError(f'Reused or changed {role} PID; refusing to signal')
            result.append(role)
        return result

    def stop_source(self):
        if len(self.live_source()) != 5:
            raise RuntimeError('A source process exited before the planned stop')
        self.update(status='stopping')
        send_owned(self.request['source_processes']['torchrun'], signal.SIGTERM)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            live = self.live_source()
            self.update(remaining_roles=live)
            if not live:
                return
            time.sleep(2)
        raise RuntimeError('Source did not fully exit within 180 seconds; inspect registered PIDs')

    def step(self):
        self.guard()
        if len(self.live_source()) != 5:
            raise RuntimeError('Source stopped before the planned boundary; no signal sent')
        progress = read_json(Path(self.request['source_run']) / 'progress.json')
        self.update(source_progress=progress)
        if progress.get('logical_epoch', 0) > self.request['target_epoch'] + 1:
            raise RuntimeError('Source passed the stopping boundary without watcher handoff')
        checked = validate_checkpoint(self.request['source_run'], self.request['target_epoch'])
        if checked is None:
            return
        if source_snapshot()['source_sha256'] != self.request['watcher_source_sha256']:
            raise RuntimeError('Watcher code changed since registration')
        self.update(status='backing_up', boundary=checked)
        backup = archive_boundary(self.request['source_run'], self.directory, checked)
        if digest(checked['checkpoint']) != checked['checkpoint_sha256']:
            raise RuntimeError('Boundary checkpoint changed after backup')
        self.update(checkpoint_backup=backup)
        self.guard()
        self.stop_source()
        self.update(status='stopped_at_checkpoint', stopped_at=timestamp(),
                    reason='user_requested_stop_after_epoch900', stopped_processes=self.request['source_processes'])
        write_json(Path(self.request['source_run']) / 'user_stop_after_epoch900.json', self.state)

    def run(self):
        try:
            while self.state['status'] != 'stopped_at_checkpoint':
                self.step()
                if self.state['status'] != 'stopped_at_checkpoint':
                    time.sleep(self.request['poll_seconds'])
            return 0
        except Exception as error:
            self.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise


def lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / '.watch.lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError('A stop watcher already owns this directory') from None
    return fd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('start', 'worker', 'status'), default='start')
    parser.add_argument('--source-request', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--target-epoch', type=int, default=900)
    parser.add_argument('--poll-seconds', type=float, default=15)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    directory = args.output_dir.resolve()
    if args.mode == 'status':
        print(json.dumps(read_json(directory / 'status.json'), indent=2))
        return 0
    if args.mode == 'worker':
        if args.lock_fd is None:
            parser.error('Internal worker requires inherited lock')
        os.fstat(args.lock_fd)
        try:
            return Watcher(read_json(directory / 'request.json'), directory).run()
        finally:
            os.close(args.lock_fd)
    if not 1 <= args.poll_seconds <= 60:
        parser.error('Polling interval must be between 1 and 60 seconds')
    fd = lock(directory)
    try:
        path = directory / 'request.json'
        if path.exists():
            raise FileExistsError('Stop watcher already registered; inspect its status')
        request = bind_source(args.source_request, args.target_epoch)
        request.update(output_dir=str(directory), poll_seconds=args.poll_seconds)
        write_json(path, request)
        with (directory / 'watcher.log').open('ab') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--mode', 'worker',
                                      '--output-dir', str(directory), '--lock-fd', str(fd)],
                                     cwd=REPO, stdout=log, stderr=log, start_new_session=True, pass_fds=(fd,))
        print(json.dumps({'watcher_pid': child.pid, 'target_epoch': args.target_epoch,
                          'status_file': str(directory / 'status.json')}, indent=2))
        return 0
    finally:
        os.close(fd)


if __name__ == '__main__':
    raise SystemExit(main())
