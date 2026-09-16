#!/usr/bin/env python3
"""Stop the local TR05 TERRAN run after its epoch-5100 validation checkpoint."""
from __future__ import annotations

import argparse
import csv
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
    from . import watch_stop_epoch as base
else:
    import watch_stop_epoch as base

read_json, digest, write_json = base.read_json, base.digest, base.write_json
process_identity, same_process, send_owned = base.process_identity, base.same_process, base.send_owned
source_snapshot, timestamp = base.source_snapshot, base.timestamp
DEFAULT_RUN = Path('/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/results/cus100_20260911/runs/TR05')
MODULE = 'EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.train'
SIGNATURE = '8875e4622c87664234903e2127da159d6d1f716a19abe0221ccd6d719f7b036f'
PROTOCOL = 'cus100_terran_synthetic_road_20260911_v1'


def train_progress(run):
    with (Path(run) / 'logs/train_log.csv').open() as handle:
        rows = [row for row in csv.DictReader(handle)
                if row.get('epoch', '').isdigit() and None not in row.values()]
    if not rows:
        raise RuntimeError('No complete TERRAN training log row')
    row = rows[-1]
    return {'completed_logical_epoch': int(row['epoch']),
            'epoch_wall_time_s': float(row['epoch_wall_time_s']),
            'train_feasible_rate': float(row['train_feasible_rate'])}


def bind_source(run, target_epoch=5100):
    run = Path(run).resolve()
    path = run / 'launch_record.json'
    launch = read_json(path)
    job = launch['job']
    if (target_epoch != 5100 or launch['experiment_id'] != 'TR05'
            or launch['hostname'] != socket.gethostname() or launch['status'] != 'running'
            or Path(launch['output_dir']).resolve() != run
            or job['train_module'] != MODULE or job['source_kind'] != 'terran_synthetic'
            or job['scale'] != 'Cus100' or job['effective_batch_size'] != 384
            or job['training_representation'] != 'E' or launch['gpu']['index'] != 3):
        raise RuntimeError('Expected the running local GPU3 TR05 Euclidean Cus100 experiment')
    trainer = process_identity(launch['pid'])
    if (trainer is None or trainer['uid'] != os.getuid()
            or trainer['argv'] != [arg for arg in launch['command'] if arg] or MODULE not in trainer['argv']
            or Path(base.option(trainer['argv'], '--output-dir')).resolve() != run):
        raise RuntimeError('TERRAN process identity differs from its launch record')
    parent = process_identity(trainer['ppid'])
    if (parent is None or parent['uid'] != os.getuid() or parent['cwd'] != trainer['cwd']
            or not any(arg.endswith('/cus100_20260911/launch.py') for arg in parent['argv'])
            or Path(base.option(parent['argv'], '--output-root')).resolve() != run.parent.parent):
        raise RuntimeError('Unexpected source launcher')
    if train_progress(run)['completed_logical_epoch'] > target_epoch:
        raise RuntimeError('Run already passed requested stopping epoch')
    return {'schema': 'terran_stop_epoch_request_v1', 'host': socket.gethostname(),
            'created_at': timestamp(), 'source_request': str(path),
            'source_request_sha256': digest(path), 'source_run': str(run),
            'source_processes': {'trainer': trainer}, 'source_launcher': parent,
            'target_epoch': target_epoch, 'physical_gpus': [3], 'gpu_uuid': launch['gpu']['uuid'],
            'watcher_source_sha256': source_snapshot()['source_sha256']}


def same_payload(left, right):
    import torch
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_payload(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same_payload(a, b) for a, b in zip(left, right))
    return left == right


def validate_checkpoint(run, epoch=5100):
    run = Path(run)
    state = read_json(run / 'data_pass_state.json')
    if state.get('instances_seen', -1) < epoch * 384:
        return None
    if (state.get('instances_seen') != epoch * 384
            or state.get('customer_exposures') != epoch * 38400
            or state.get('optimizer_steps') != epoch * 12
            or state.get('protocol_id') != PROTOCOL
            or Path(state.get('last_checkpoint', '')).resolve() != run / 'checkpoint_latest.pt'):
        raise RuntimeError('TERRAN committed accounting differs from requested boundary')
    val = base.last_jsonl(run / 'validation_history.jsonl') or {}
    if val.get('logical_epoch') != epoch or val.get('instances') != 500:
        raise RuntimeError('Missing completed 500-instance validation at stopping epoch')
    import torch
    paths = [run / 'checkpoints' / f'checkpoint_epoch_{epoch:04d}.pt', run / 'checkpoint_latest.pt']
    payloads = [torch.load(path, map_location='cpu', weights_only=False) for path in paths]
    for payload in payloads:
        cfg = payload['config']
        if (payload.get('epoch') != epoch or payload.get('seed') != 1234
                or cfg.get('protocol', {}).get('resolved_training_signature_sha256') != SIGNATURE
                or cfg.get('data', {}).get('stage2_scale') != 'Cus100'
                or cfg.get('data', {}).get('stage2_training_representation') != 'E'
                or not payload.get('model_state_dict')
                or not payload.get('optimizer_state_dict', {}).get('state')):
            raise RuntimeError('TERRAN checkpoint epoch, signature or training state differs')
    if not same_payload(*payloads):
        raise RuntimeError('Epoch checkpoint and latest contain different training state')
    return {'logical_epoch': epoch, 'instances_seen': state['instances_seen'],
            'optimizer_steps': state['optimizer_steps'], 'customer_exposures': state['customer_exposures'],
            'checkpoint_files': {str(p.relative_to(run)): digest(p) for p in paths},
            'validation_cost_usd': val.get('mean_verified_cost_usd'),
            'validation_feasible': val.get('complete_and_feasible'), 'validation_instances': val['instances']}


def archive_boundary(run, directory, checked):
    run, destination = Path(run), Path(directory) / 'checkpoint_backup'
    destination.mkdir(exist_ok=False)
    names = list(checked['checkpoint_files']) + ['data_pass_state.json', 'best.ckpt',
        'best_overall.ckpt', 'best_within_5000.ckpt', 'checkpoint_selected.pt',
        'validation_summary.json', 'validation_summary_overall.json',
        'validation_summary_within_5000.json', 'validation_history.jsonl', 'launch_record.json']
    records = []
    for name in names:
        source, target = run / name, destination / name
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f'Required regular artifact missing: {source}')
        before = digest(source)
        if name in checked['checkpoint_files'] and before != checked['checkpoint_files'][name]:
            raise RuntimeError('Checkpoint changed before backup')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if before != digest(target) or before != digest(source):
            raise RuntimeError(f'Artifact changed during backup: {name}')
        with target.open('rb') as handle:
            os.fsync(handle.fileno())
        records.append({'name': name, 'sha256': before, 'bytes': target.stat().st_size})
    write_json(destination / 'manifest.json', {'time': timestamp(), 'boundary': checked, 'files': records})
    with (destination / 'manifest.json').open('rb') as handle:
        os.fsync(handle.fileno())
    for directory in (destination / 'checkpoints', destination):
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return str(destination)


class Watcher(base.Watcher):
    def __init__(self, request, directory):
        self.request, self.directory = request, Path(directory)
        self.state = {'schema': 'terran_stop_epoch_status_v1', 'status': 'waiting_epoch',
                      'watcher_pid': os.getpid(), 'source_run': request['source_run'],
                      'target_epoch': request['target_epoch'], 'physical_gpus': [3]}
        self.update()

    def stop_source(self):
        if self.live_source() != ['trainer']:
            raise RuntimeError('Trainer exited before planned stop')
        self.update(status='stopping')
        send_owned(self.request['source_processes']['trainer'], signal.SIGTERM)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if not self.live_source():
                return
            time.sleep(1)
        raise RuntimeError('TERRAN did not exit within 60 seconds')

    def step(self):
        self.guard()
        if self.live_source() != ['trainer']:
            raise RuntimeError('Trainer stopped before the planned boundary')
        progress = train_progress(self.request['source_run'])
        self.update(source_progress=progress)
        if progress['completed_logical_epoch'] > self.request['target_epoch']:
            raise RuntimeError('Source passed stopping boundary; inspect before rollback')
        checked = validate_checkpoint(self.request['source_run'], self.request['target_epoch'])
        if checked is None:
            return
        if source_snapshot()['source_sha256'] != self.request['watcher_source_sha256']:
            raise RuntimeError('Watcher code changed since registration')
        self.update(status='backing_up', boundary=checked)
        backup = archive_boundary(self.request['source_run'], self.directory, checked)
        self.update(checkpoint_backup=backup)
        self.guard()
        self.stop_source()
        self.update(status='stopped_at_checkpoint', stopped_at=timestamp(),
                    reason='user_requested_stop_after_epoch5100')
        write_json(Path(self.request['source_run']) / 'user_stop_after_epoch5100.json', self.state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('start', 'worker', 'status'), default='start')
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    directory = (args.output_dir or args.run / 'stop_after_epoch5100').resolve()
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
    fd = base.lock(directory)
    try:
        path = directory / 'request.json'
        if path.exists():
            raise FileExistsError('Stop watcher already registered; inspect status')
        request = bind_source(args.run)
        request.update(output_dir=str(directory), poll_seconds=10)
        write_json(path, request)
        with (directory / 'watcher.log').open('ab') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--mode', 'worker',
                                      '--output-dir', str(directory), '--lock-fd', str(fd)],
                                     cwd=base.REPO, stdout=log, stderr=log, start_new_session=True, pass_fds=(fd,))
        print(json.dumps({'watcher_pid': child.pid, 'target_epoch': 5100,
                          'status_file': str(directory / 'status.json')}, indent=2))
        return 0
    finally:
        os.close(fd)


if __name__ == '__main__':
    raise SystemExit(main())
