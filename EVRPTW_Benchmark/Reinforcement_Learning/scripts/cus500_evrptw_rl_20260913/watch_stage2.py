#!/usr/bin/env python3
"""Fork the owned Road500 run into its greedy-baseline phase at a saved epoch."""
from __future__ import annotations

import argparse
from copy import deepcopy
import ctypes
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

if __package__:
    from .common import HERE, REPO, digest, source_snapshot, timestamp, write_json
    from . import launch
else:
    from common import HERE, REPO, digest, source_snapshot, timestamp, write_json
    import launch

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TRAIN_MODULE = 'EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.distributed_train'
DEFAULT_SOURCE = Path('/data/Maojie/ICLR/cus500-evrptw-rl-after-am/EVRPTW_Benchmark/results/cus500_evrptw_rl_20260913/launchers/local_after_am_gpu01/evrptw_rl/launch_request.json')
DEFAULT_OUTPUT = REPO / 'EVRPTW_Benchmark/results/cus500_evrptw_rl_stage2_at300_20260914'
TERMINAL = {'handed_off', 'failed'}


def _pidfd_libc():
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, 'pidfd_open') or not hasattr(libc, 'pidfd_send_signal'):
        raise RuntimeError('Linux pidfd support is required for safe process signalling')
    libc.pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
    libc.pidfd_open.restype = ctypes.c_int
    libc.pidfd_send_signal.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    libc.pidfd_send_signal.restype = ctypes.c_int
    return libc


def pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    fd = _pidfd_libc().pidfd_open(pid, 0)
    if fd < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return fd


def pidfd_signal(fd, sig):
    if hasattr(signal, 'pidfd_send_signal'):
        return signal.pidfd_send_signal(fd, sig)
    if _pidfd_libc().pidfd_send_signal(fd, sig, None, 0) < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def read_json(path):
    return json.loads(Path(path).read_text())


def process_identity(pid):
    """Bind signals to UID, exact argv/cwd and Linux PID start time."""
    try:
        base = Path('/proc') / str(int(pid))
        fields = (base / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': int(pid), 'start_ticks': int(fields[19]),
                'uid': base.stat().st_uid, 'ppid': int(fields[1]),
                'cwd': str((base / 'cwd').resolve(strict=True)),
                'argv': [p.decode() for p in (base / 'cmdline').read_bytes().split(b'\0') if p]}
    except (OSError, ValueError, TypeError, IndexError):
        return None


def same_process(saved, current):
    return current is not None and all(saved[k] == current[k] for k in
                                       ('pid', 'start_ticks', 'uid', 'cwd', 'argv'))


def option(argv, flag):
    if argv.count(flag) != 1:
        raise ValueError(f'Expected exactly one {flag}')
    return argv[argv.index(flag) + 1]


def stage_config(old, boundary):
    config = deepcopy(old)
    if (config.get('model') != 'evrptw_rl' or config.get('scale') != 'Cus500'
            or config.get('gpus') != [0, 1] or config.get('world_size') != 2
            or config.get('train_module') != TRAIN_MODULE):
        raise ValueError('Watcher requires the two-GPU 0/1 EVRPTW-RL Road500 run')
    extra = config['extra_args']
    previous = int(option(extra, '--ema-warmup-steps'))
    if not 0 < boundary < previous or boundary % config['validation_every_epochs']:
        raise ValueError('Boundary must precede the old warmup and fall on a validation checkpoint')
    extra[extra.index('--ema-warmup-steps') + 1] = str(boundary)
    return config


def bind_processes(upstream, status):
    run = Path(upstream['preflight']['output_dir']).resolve()
    progress = read_json(run / 'progress.json')
    if status.get('status') != 'running' or Path(status['output_dir']).resolve() != run:
        raise RuntimeError('Source launcher is not running this experiment')
    roles = [('launcher', status['launcher_pid']), ('torchrun', status['pid'])]
    roles += [(f"rank{row['rank']}", row['pid']) for row in progress['workers']]
    if sorted(role for role, _ in roles) != ['launcher', 'rank0', 'rank1', 'torchrun']:
        raise RuntimeError('Expected exactly one launcher, torchrun and two ranks')
    known = {}
    for role, pid in roles:
        identity = process_identity(pid)
        if identity is None or identity['uid'] != os.getuid():
            raise RuntimeError(f'Cannot bind owned source {role}')
        if role == 'launcher':
            if not any(a.endswith('/cus500_evrptw_rl_20260913/launch.py') for a in identity['argv']):
                raise RuntimeError('Source launcher executable differs')
            if Path(option(identity['argv'], '--request')).resolve() != Path(status['request']).resolve():
                raise RuntimeError('Source launcher request differs')
        else:
            if TRAIN_MODULE not in identity['argv'] or Path(option(identity['argv'], '--output-dir')).resolve() != run:
                raise RuntimeError(f'Source {role} is not the requested EVRPTW-RL run')
        known[role] = identity
    if known['torchrun']['argv'] != upstream['command']:
        raise RuntimeError('Live torchrun command differs from the saved request')
    if (known['torchrun']['ppid'] != known['launcher']['pid']
            or any(known[role]['ppid'] != known['torchrun']['pid'] for role in ('rank0', 'rank1'))
            or len({p['cwd'] for p in known.values()}) != 1):
        raise RuntimeError('Source process ancestry/cwd differs')
    return known


def lock_watcher(output):
    directory = Path(output) / 'watcher'
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / '.lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError('This stage2 watcher is already running') from None
    return fd, directory


def make_request(source_request, output, boundary, poll_seconds):
    source_request = Path(source_request).resolve()
    upstream = read_json(source_request)
    if upstream['preflight']['host'] != socket.gethostname():
        raise RuntimeError('Source experiment belongs to another host')
    output = Path(output).resolve()
    source_run = Path(upstream['preflight']['output_dir']).resolve()
    if output == source_run or output in source_run.parents or source_run in output.parents:
        raise ValueError('Stage2 must use a separate output root')
    config = stage_config(upstream['preflight']['config'], boundary)
    identities = bind_processes(upstream, read_json(source_request.parent / 'status.json'))
    progress = read_json(source_run / 'progress.json')
    if int(progress['logical_epoch']) > boundary:
        raise ValueError('Source has already passed the requested boundary; no automatic rollback')
    # Probe kernel support without signalling any process. Some Conda builds
    # omit Python pidfd APIs, so the helpers support the libc implementation.
    probe = pidfd_open(os.getpid())
    os.close(probe)
    original_source = source_snapshot(identities['torchrun']['cwd'])
    if original_source['source_sha256'] != upstream['source']['source_sha256']:
        raise RuntimeError('Live source differs from its original launch snapshot')
    return {'schema': 'evrptw_rl_stage2_watcher_request_v1', 'created_at': timestamp(),
            'host': socket.gethostname(), 'source_request': str(source_request),
            'source_request_sha256': digest(source_request), 'source_run': str(source_run),
            'source_repo': identities['torchrun']['cwd'],
            'source_code_sha256': original_source['source_sha256'],
            'source_processes': identities, 'source_launch': upstream,
            'output_root': str(output), 'config': config, 'boundary_epoch': boundary,
            'poll_seconds': poll_seconds, 'stage2_source': source_snapshot(),
            'python': sys.executable}


def send_owned(saved, sig):
    """A pidfd prevents signalling a reused PID between verification and kill."""
    current = process_identity(saved['pid'])
    if current is None:
        return False
    if not same_process(saved, current):
        raise RuntimeError(f"PID {saved['pid']} was reused or its command changed; refusing signal")
    try:
        fd = pidfd_open(saved['pid'])
    except ProcessLookupError:
        return False
    try:
        if not same_process(saved, process_identity(saved['pid'])):
            raise RuntimeError('Source process identity changed while opening pidfd')
        pidfd_signal(fd, sig)
        return True
    finally:
        os.close(fd)


def validate_stage2_resume(request, run):
    """Exercise the exact CLI/signature/model/optimizer resume contract on CPU."""
    import torch
    from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL import distributed_train as entry
    from EVRPTW_Benchmark.Reinforcement_Learning.common.distributed import DistributedContext
    from EVRPTW_Benchmark.Reinforcement_Learning.common.distributed_protocol import configure_distributed_contract
    from EVRPTW_Benchmark.Reinforcement_Learning.common.protocol_trainers import prepare_training_objective, _load_checkpoint
    from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913.stage2_transition import assert_stage2_launch_args
    root = request['source_launch']['preflight']['data']['root']
    command = launch.build_command(request['config'], root, run, request['source_launch']['stream'],
                                   python=request['python'], resume=True)
    args = entry.parse_args(command[command.index(TRAIN_MODULE) + 1:])
    args.device = 'cpu'
    entry.prepare_method(args)
    contract = configure_distributed_contract(args, DistributedContext(rank=0, world_size=2), method='EVRPTW-RL')
    objective = prepare_training_objective(args)
    policy = entry.build_policy(args)
    baseline = deepcopy(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    payload = _load_checkpoint(Path(run) / 'checkpoint_latest.pt', policy=policy, baseline=baseline,
                               optimizer=optimizer, protocol_id=args.protocol_id, objective_config=objective,
                               optimizer_name=args.optimizer, optimizer_weight_decay=args.weight_decay,
                               reward_contract_args=args)
    assert_stage2_launch_args(payload, args)
    if payload['distributed_contract'] != contract:
        raise ValueError('Stage2 distributed topology/batch contract differs')
    return {'status': 'passed', 'logical_epoch': payload['logical_epoch'],
            'stream_cursor': payload['stream_cursor'], 'ema_warmup_steps': args.ema_warmup_steps,
            'check': 'strict_actual_cli_checkpoint_and_optimizer_resume_on_cpu'}


class Watcher:
    def __init__(self, request):
        self.request = request
        self.output = Path(request['output_root'])
        self.directory = self.output / 'watcher'
        self.state_path = self.directory / 'status.json'
        self.state = {'schema': 'evrptw_rl_stage2_watcher_status_v1', 'status': 'waiting_epoch',
                      'watcher_pid': os.getpid(), 'host': socket.gethostname(),
                      'boundary_epoch': request['boundary_epoch'],
                      'resume_epoch': request['boundary_epoch'] + 1,
                      'source_run': request['source_run'], 'physical_gpus': [0, 1],
                      'stage2_run': str(self.output / 'runs' / request['config']['run_id'])}
        self.update()

    def update(self, **values):
        self.state.update(values, heartbeat=timestamp())
        write_json(self.state_path, self.state)

    def guard(self, *, sources=False):
        r = self.request
        if socket.gethostname() != r['host'] or digest(r['source_request']) != r['source_request_sha256']:
            raise RuntimeError('Source host/request changed after registration')
        if sources:
            if source_snapshot()['source_sha256'] != r['stage2_source']['source_sha256']:
                raise RuntimeError('Stage2 source changed after registration')
            if source_snapshot(r['source_repo'])['source_sha256'] != r['source_code_sha256']:
                raise RuntimeError('Live training source changed after registration')
            stream = r['source_launch']['stream']
            if digest(stream['path']) != stream['file_sha256']:
                raise RuntimeError('Frozen training stream changed')

    def live_source(self):
        live = []
        for role, saved in self.request['source_processes'].items():
            current = process_identity(saved['pid'])
            if current is None:
                continue
            if not same_process(saved, current):
                raise RuntimeError(f'Source {role} PID identity changed')
            live.append(role)
        return live

    def boundary_ready(self):
        r = self.request
        run = Path(r['source_run'])
        progress = read_json(run / 'progress.json')
        self.update(source_progress=progress)
        live = self.live_source()
        if len(live) != 4:
            raise RuntimeError(f'Source process exited before the planned handoff: {live}')
        epoch = r['boundary_epoch']
        checkpoint = run / f'checkpoint_epoch_{epoch:04d}.pt'
        # An epoch history row alone is written BEFORE validation. Wait for both
        # the epoch artifact AND latest sidecar, published after validation.
        state = read_json(run / 'data_pass_state.json')
        latest = launch.last_jsonl(run / 'validation_history.jsonl') or {}
        if not checkpoint.is_file() or int(state.get('optimizer_steps', -1)) < epoch:
            if int(progress.get('logical_epoch', 0)) > epoch + 1:
                raise RuntimeError('Boundary checkpoint was skipped; refusing a later checkpoint')
            return False
        if int(latest.get('logical_epoch', -1)) != epoch:
            raise RuntimeError('Boundary checkpoint has no matching completed validation')
        return True

    def stop_source(self):
        self.update(status='stopping_source')
        known = self.request['source_processes']
        if len(self.live_source()) != 4:
            raise RuntimeError('Source ownership changed before planned stop')
        # Torchrun handles SIGTERM by shutting down its two registered workers.
        # The original launcher reaps torchrun and releases its GPU flocks.
        send_owned(known['torchrun'], signal.SIGTERM)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            live = self.live_source()
            self.update(stopping_roles=live)
            if not live:
                write_json(self.directory / 'source_stop.json', {
                    'time': timestamp(), 'reason': 'user_requested_stage2_boundary',
                    'epoch': self.request['boundary_epoch'], 'processes': known})
                return
            time.sleep(2)
        # No broad kill or unverified replacement PID, even if shutdown stalls.
        raise RuntimeError('Owned source did not exit within 180 seconds; stage2 not launched')

    def start_stage2(self, transition):
        r = self.request
        config = r['config']
        root = Path(r['source_launch']['preflight']['data']['root'])
        lock_fd, directory = launch.lock_output(self.output, config)
        fds = [lock_fd]
        try:
            audit = launch.preflight(config, root, self.output, resume=True)
            expected = {row['index']: row['uuid'] for row in r['source_launch']['preflight']['gpus']}
            if {row['index']: row['uuid'] for row in audit['gpus']} != expected:
                raise RuntimeError('Physical GPU UUIDs changed')
            if audit['data'] != r['source_launch']['preflight']['data']:
                raise RuntimeError('Source dataset/objective/reward contract changed')
            fds.extend(launch.lock_gpus(audit['gpus']))
            request_path = directory / 'launch_request.json'
            if request_path.exists():
                raise FileExistsError('Stage2 launch was already attempted; refusing duplicate launch')
            command = launch.build_command(config, root, audit['output_dir'], r['source_launch']['stream'],
                                           python=r['python'], resume=True)
            request = {'schema': 'cus500_multigpu_stage2_launch_request_v1', 'time': timestamp(),
                       'preflight': audit, 'stream': r['source_launch']['stream'],
                       'source': r['stage2_source'], 'command': command,
                       'environment': r['source_launch']['environment'], 'stage2_transition': transition}
            write_json(request_path, request)
            with (directory / 'launcher.log').open('ab') as log:
                child = subprocess.Popen([r['python'], str(HERE / 'launch.py'), '--mode', 'worker',
                    '--request', str(request_path), '--lock-fds', ','.join(map(str, fds))],
                    cwd=REPO, stdout=log, stderr=log, start_new_session=True, pass_fds=tuple(fds))
            self.update(status='confirming_stage2', stage2_launcher_pid=child.pid,
                        stage2_status_file=str(directory / 'status.json'), launched_at=timestamp(),
                        launch_monotonic=time.monotonic())
        finally:
            for fd in fds:
                os.close(fd)

    def confirm_stage2(self):
        path = Path(self.state['stage2_status_file'])
        if path.is_file():
            status = read_json(path)
            if status.get('status') == 'failed':
                raise RuntimeError(f"Stage2 launcher failed; inspect {path}")
            history = launch.last_jsonl(Path(self.state['stage2_run']) / 'logical_epoch_history.jsonl') or {}
            if int(history.get('logical_epoch', 0)) > self.request['boundary_epoch']:
                if history.get('baseline_kind') != 'greedy_rollout':
                    raise RuntimeError('Resumed epoch did not use greedy rollout baseline')
                self.update(status='handed_off', first_stage2_epoch=history['logical_epoch'],
                            stage2_torchrun_pid=status.get('pid'), completed_at=timestamp())
                return
        if process_identity(self.state['stage2_launcher_pid']) is None:
            raise RuntimeError('Stage2 launcher exited before the first greedy-baseline epoch')
        if time.monotonic() - self.state['launch_monotonic'] > 7200:
            raise RuntimeError('Stage2 has not completed its first epoch within two hours; inspect training logs')

    def step(self):
        if self.state['status'] in TERMINAL:
            return
        self.guard()
        if self.state['status'] == 'confirming_stage2':
            self.confirm_stage2()
            return
        if self.state['status'] != 'waiting_epoch':
            raise RuntimeError('Unexpected watcher phase')
        if not self.boundary_ready():
            return
        self.guard(sources=True)
        self.update(status='preparing_transition')
        from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913.stage2_transition import prepare_stage2_run
        # Prepare and validate the complete fork BEFORE stopping live training.
        transition = prepare_stage2_run(Path(self.request['source_run']), Path(self.state['stage2_run']),
                                        self.request['boundary_epoch'])
        self.update(transition=transition)
        resume_check = validate_stage2_resume(self.request, Path(self.state['stage2_run']))
        self.update(resume_check=resume_check)
        environment = launch.environment_report(self.request['config'])
        data = launch.inspect_data(self.request['source_launch']['preflight']['data']['root'], self.request['config'])
        if data != self.request['source_launch']['preflight']['data']:
            raise RuntimeError('Source dataset/objective changed before the planned stop')
        self.update(stage2_environment=environment)
        self.guard(sources=True)
        self.stop_source()
        self.update(status='waiting_gpu_release')
        # GPU accounting may lag process exit briefly. Never stop new occupants.
        deadline = time.monotonic() + 180
        while True:
            try:
                launch.validate_gpus(self.request['config'], launch.gpu_inventory(), launch.gpu_processes())
                break
            except RuntimeError as error:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'GPUs did not become available after source exit: {error}') from error
                self.update(waiting_reason=str(error))
                time.sleep(5)
        self.start_stage2(transition)

    def run(self):
        try:
            while self.state['status'] not in TERMINAL:
                self.step()
                self.update()
                if self.state['status'] not in TERMINAL:
                    time.sleep(self.request['poll_seconds'])
            return 0
        except Exception as error:
            self.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('start', 'worker', 'status'), default='start')
    parser.add_argument('--source-request', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--boundary-epoch', type=int, default=300)
    parser.add_argument('--poll-seconds', type=float, default=20)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    output = args.output_root.resolve()
    directory = output / 'watcher'
    if args.mode == 'status':
        print(json.dumps(read_json(directory / 'status.json'), indent=2))
        return 0
    if args.mode == 'worker':
        if args.lock_fd is None:
            parser.error('Internal worker requires inherited lock fd')
        os.fstat(args.lock_fd)
        try:
            return Watcher(read_json(directory / 'request.json')).run()
        finally:
            os.close(args.lock_fd)
    if not 1 <= args.poll_seconds <= 60:
        parser.error('Poll interval must be between 1 and 60 seconds')
    fd, directory = lock_watcher(output)
    try:
        if (directory / 'request.json').exists():
            raise FileExistsError('Watcher already registered here; inspect status before creating another request')
        request = make_request(args.source_request, output, args.boundary_epoch, args.poll_seconds)
        write_json(directory / 'request.json', request)
        write_json(output / 'stage2_config.json', request['config'])
        with (directory / 'watcher.log').open('ab') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--mode', 'worker',
                                      '--output-root', str(output), '--lock-fd', str(fd)],
                                     cwd=REPO, stdout=log, stderr=log, start_new_session=True, pass_fds=(fd,))
        print(json.dumps({'watcher_pid': child.pid, 'boundary_epoch': args.boundary_epoch,
                          'resume_epoch': args.boundary_epoch + 1, 'gpus': [0, 1],
                          'status_file': str(directory / 'status.json')}, indent=2))
        return 0
    finally:
        os.close(fd)


if __name__ == '__main__':
    raise SystemExit(main())
