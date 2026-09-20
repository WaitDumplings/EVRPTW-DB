#!/usr/bin/env python3
"""Stage 1: single-GPU, Cus100 policy continuation under the D_time cost contract."""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
import csv
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.ablation_final.launch import (
    METHODS, NAMES, command_for, compute_busy_uuids, dataset_root,
    digest, gpu_inventory, make_job, now, read_rows, write_json,
)

SYNTHETIC = 'TERRAN_synthetic100_feasible4_20260911'
DEFAULT_BATCHES = {
    'G': {'am_evrptw': 208, 'evrptw_rl': 240, 'drl_ts': 44, 'terran': 384, 'rrnco': 72},
    'E': {'am_evrptw': 108, 'evrptw_rl': 200, 'drl_ts': 44, 'terran': 384, 'rrnco': 50},
}


# Keep historical measurements separate from later requested batch defaults.
PROFILED_BATCHES = {domain: dict(batches) for domain, batches in DEFAULT_BATCHES.items()}
PROFILED_BATCHES['G']['rrnco'] = 82


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('method', choices=METHODS)
    p.add_argument('domain', type=lambda s: s.rstrip(',').upper(), choices=('G', 'E'))
    p.add_argument('gpu', type=lambda s: int(s.rstrip(',')), help='Physical nvidia-smi GPU index')
    p.add_argument('--checkpoint-root', type=Path, default=os.environ.get('CURRICULUM_CKPT_ROOT', '/data/best_ckpt'))
    p.add_argument('--road-root', default=os.environ.get('CURRICULUM_ROAD_ROOT', os.environ.get('CUS100_ROAD_ROOT', os.environ.get('EVRPTW_DATASET_ROOT'))))
    p.add_argument('--synthetic-root', default=os.environ.get('CURRICULUM_SYNTHETIC_ROOT', os.environ.get('CUS100_SYNTHETIC_ROOT')))
    p.add_argument('--output-root', type=Path, default=os.environ.get('CURRICULUM_OUTPUT_ROOT', str(REPO / 'EVRPTW_Benchmark/results/curriculum_stage1')))
    p.add_argument('--run-dir', type=Path, help='Explicit NEW directory, e.g. for a smoke test')
    p.add_argument('--batch-size', type=int, help='Per-GPU instances; defaults are measured Cus100 settings')
    p.add_argument('--epochs', type=int, default=2000, help='ADDITIONAL epochs, with reset stage counters')
    p.add_argument('--validation-every', type=int, default=100)
    p.add_argument('--validation-limit', type=int, default=500)
    p.add_argument('--dry-run', action='store_true', help='Check checkpoint hash and dataset location; print plan without training')
    a = p.parse_args(argv)
    if a.gpu < 0 or not 1 <= a.validation_limit <= 500:
        p.error('GPU index must be nonnegative; validation limit must be 1..500')
    return a


def checkpoint_record(method, domain, root):
    records = json.loads((HERE / 'checkpoints.json').read_text())['checkpoints']
    row = next(r for r in records if r['method'] == method and r['source_domain'] == domain)
    path = root.expanduser().resolve() / row['relative_path']
    if not path.is_file():
        raise FileNotFoundError(f'Missing archived checkpoint: {path}')
    if digest(path) != row['sha256']:
        raise ValueError(f'Checkpoint differs from the audited archive: {path}')
    return path, row


def resolve_data(domain, road_root=None, synthetic_root=None):
    if domain == 'G':
        return dataset_root(road_root)
    candidates = [Path(synthetic_root).expanduser()] if synthetic_root else [
        REPO / 'EVRPTW_Dataset' / SYNTHETIC,
        REPO.parent / 'EVRPTW-DB/EVRPTW_Dataset' / SYNTHETIC,
    ]
    for path in candidates:
        if (path / 'corpus_manifest.json').is_file():
            return path.resolve()
    raise FileNotFoundError('Frozen Euclidean corpus missing; set CURRICULUM_SYNTHETIC_ROOT')


def curriculum_job(a):
    batch = DEFAULT_BATCHES[a.domain][a.method] if a.batch_size is None else a.batch_size
    job = make_job(a.method, 100, 1, a.epochs, a.validation_every, a.validation_limit, batch)
    calibration_status = 'curriculum_full_batch_cuda_smoke_passed'
    if batch != PROFILED_BATCHES[a.domain][a.method]:
        calibration_status = ('default_batch_not_gpu_profiled' if a.batch_size is None
                              else 'explicit_batch_override_not_profiled')
    job.update(protocol_id='curriculum_stage1_cus100_dtime_v1', schema='curriculum_stage1_cus100_dtime_v1',
               training_representation=a.domain, run_id=f'{a.method}_{a.domain}_Cus100_stage1_seed1234',
               calibration_status=calibration_status)
    if a.domain == 'E':
        job.update(source_kind='terran_synthetic', train_index='train/view_index.parquet',
                   validation_index='val/view_index.parquet',
                   reward_contract_config_path='script_curriculum/configs/reward_euclidean_dtime.json')
    if a.method == 'drl_ts':
        # Both archived Cus100 checkpoints have already completed their soft stage.
        job['soft_stage_end_epoch'] = 0
        job.pop('planned_full_run_soft_stage_end_epoch', None)
    return job


def prepare_data(root, run, job):
    import pandas as pd
    from EVRPTW_Benchmark.Reinforcement_Learning.common.training_stream import (
        atomic_write_stream, build_training_stream, load_training_stream_contract,
    )
    audit = {}
    if job['source_kind'] == 'terran_synthetic':
        from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus100_20260911.data_contract import inspect_synthetic
        checked = inspect_synthetic(root, verify_payloads=False)
        audit['synthetic_manifest_sha256'] = checked['manifest_sha256']
        audit['synthetic_payload_hashes_rechecked'] = False
    train_path, val_path = root / job['train_index'], root / job['validation_index']
    index, validation = pd.read_parquet(train_path), pd.read_parquet(val_path)
    scale = int(job.get('scale', 'Cus100').removeprefix('Cus'))
    expected_train = {100: 50000, 500: 10000, 1000: 5000}[scale]
    train = index[(index.customer_count == scale) & (index.split_id == 'train') & (index.track_id == 'train')]
    val = validation[(validation.customer_count == scale) & (validation.split_id == 'val')]
    if len(train) != expected_train or len(val) != 500:
        raise ValueError(f'Expected full Cus{scale} corpus {expected_train} train/500 val; got {len(train)}/{len(val)}')
    if train.view_id.duplicated().any() or val.view_id.duplicated().any():
        raise ValueError('Duplicate dataset view IDs')
    if set(train.view_id) & set(val.view_id) or set(train.family_id) & set(val.family_id):
        raise ValueError('Train/val views or parent families overlap')
    count = job['training_epochs'] * job['effective_batch_size']
    frame, metadata = build_training_stream(index, scale=f'Cus{scale}', seed=job['seed'], sample_count=count)
    metadata.update(source_index=str(train_path), source_index_sha256=digest(train_path),
                    source_kind=job['source_kind'], allowed_family_ids_source=None, allowed_family_ids_sha256=None)
    path = run / 'artifacts/training_stream.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_stream(path, frame, metadata)
    contract = load_training_stream_contract(path)
    job.update(training_stream_path=str(path), training_stream_contract_sha256=contract['sha256'],
               train_index_sha256=digest(train_path), validation_index_sha256=digest(val_path))
    audit.update(train_views=len(train), validation_views=len(val), test_data_read=False,
                 train_val_parent_overlap=0, additional_instance_exposures=count,
                 stream_policy='fresh_stage_seed1234_full_pool_shuffled_cycles')
    return audit


def build_command(job, data, run, checkpoint):
    command = command_for(job, data, run)
    command += ['--warm-start-checkpoint', str(checkpoint), '--warm-start-objective-transition']
    if job.get('warm_start_scale_transition'):
        command += ['--warm-start-scale-transition']
    if job['method'] == 'terran':
        command += ['--warm-start-epoch-mode', 'reset']
    return command


def export_validation(run):
    fields = ['logical_epoch', 'instances', 'complete_and_feasible', 'mean_verified_cost_usd', 'best_overall_selected']
    rows = read_rows(run / 'validation_history.jsonl')
    temporary = run / 'validation_summary.csv.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(run / 'validation_summary.csv')
    return rows


def source_file_hashes(job):
    files = {
        'script_curriculum/launch.py', 'script_curriculum/checkpoints.json',
        'EVRPTW_Benchmark/Reinforcement_Learning/common/protocol_trainers.py',
        'EVRPTW_Benchmark/Reinforcement_Learning/common/protocol_entrypoints.py',
        'EVRPTW_Benchmark/Reinforcement_Learning/common/training_protocol.py',
        'EVRPTW_Benchmark/Reinforcement_Learning/TERRAN/train.py',
        'EVRPTW_Benchmark/Reinforcement_Learning/TERRAN/trainer.py',
        'EVRPTW_Benchmark/Reinforcement_Learning/TERRAN/protocol.py',
    }
    files.update(job.get('launcher_source_paths', []))
    for field in ('objective_config_path', 'reward_contract_config_path',
                  'terran_config_path', 'method_auxiliary_profile_path'):
        if job.get(field):
            files.add(job[field])
    return {str(path): digest(REPO / path) for path in sorted(files)}


def terminate_trainer(child):
    """Reap this launcher's process group if supervision fails."""
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        return child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return child.wait(timeout=10)


def run_training(a, job, data, checkpoint, source, gpu):
    # Keep the lock FD alive in both supervisor and trainer to survive a lost shell.
    selected_gpus = [gpu] if isinstance(gpu, dict) else list(gpu)
    if len(selected_gpus) != job['world_size']:
        raise ValueError('Selected GPU count does not match the training world size')
    if len({g['uuid'] for g in selected_gpus}) != len(selected_gpus):
        raise ValueError('Each distributed rank must use a distinct GPU')
    gpu_indices = [g['index'] for g in selected_gpus]
    gpu_label = gpu_indices[0] if len(gpu_indices) == 1 else gpu_indices
    locks = Path(f'/tmp/evrptw-ablation-gpu-locks-{os.getuid()}')
    locks.mkdir(exist_ok=True)
    with ExitStack() as stack:
        held_locks = []
        for selected in sorted(selected_gpus, key=lambda value: value['uuid']):
            lock = stack.enter_context((locks / f"{selected['uuid']}.lock").open('a+'))
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"GPU {selected['index']} is already reserved by another launcher") from exc
            held_locks.append(lock)
        busy = compute_busy_uuids()
        for selected in selected_gpus:
            if selected['uuid'] in busy:
                raise RuntimeError(f"GPU {selected['index']} already has a compute job")
        stamp = time.strftime('%Y%m%dT%H%M%S', time.gmtime())
        run = (a.run_dir or a.output_root / f"{job['run_id']}_{stamp}_{os.getpid()}").expanduser().resolve()
        run.mkdir(parents=True, exist_ok=False)
        status = dict(status='preparing', launcher_pid=os.getpid(), started_at=now(),
                      gpu=gpu_label, logical_epoch=0)
        write_json(run / 'status.json', status)
        print(f"{NAMES[a.method]} {a.domain}: GPU {gpu_label}, batch={job['physical_batch_size']}, +{a.epochs} epochs", flush=True)
        print(f"Output: {run}\nLog: {run / 'training.log'}", flush=True)
        interrupted = []
        child = None
        previous = {}
        error = None
        rc = 1
        try:
            audit = prepare_data(data, run, job)
            command = build_command(job, data, run, checkpoint)
            request = dict(schema=job['schema'], created_at=now(), host=socket.gethostname(), job=job,
                           checkpoint=str(checkpoint), source=source, dataset_root=str(data), data_audit=audit,
                           gpu=gpu, command=command, output_dir=str(run), source_revision=subprocess.check_output(
                               ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                           source_hashes=source_file_hashes(job))
            write_json(run / 'request.json', request)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in selected_gpus), PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
            def stop(sig, _frame):
                interrupted.append(sig)
                if child is not None and child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
            with (run / 'training.log').open('w') as log:
                child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, pass_fds=tuple(lock.fileno() for lock in held_locks))
                status.update(status='running', pid=child.pid)
                write_json(run / 'status.json', status)
                if interrupted:
                    stop(interrupted[-1], None)
                previous_epoch = -1
                while child.poll() is None:
                    history = read_rows(run / 'logical_epoch_history.jsonl') or read_rows(run / 'reward_diagnostics.jsonl')
                    epoch = history[-1].get('logical_epoch', 0) if history else 0
                    status.update(updated_at=now(), logical_epoch=epoch)
                    write_json(run / 'status.json', status)
                    export_validation(run)
                    if epoch != previous_epoch and (epoch % 100 == 0 or a.epochs < 100):
                        print(f"{NAMES[a.method]} {a.domain}: epoch {epoch}/{a.epochs}", flush=True)
                        previous_epoch = epoch
                    time.sleep(5)
                rc = child.wait()
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            if child is not None:
                rc = terminate_trainer(child) or 1
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        completed = None
        try:
            export_validation(run)
            history = read_rows(run / 'logical_epoch_history.jsonl') or read_rows(run / 'reward_diagnostics.jsonl')
            if history:
                status['logical_epoch'] = int(history[-1].get('logical_epoch', 0))
            result_path = run / 'training_result.json'
            result = json.loads(result_path.read_text()) if result_path.is_file() else {}
            if result.get('completed_training_epochs') is not None:
                completed = int(result['completed_training_epochs'])
            if completed is not None:
                status['logical_epoch'] = completed
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            error = error or f'{type(exc).__name__}: {exc}'
        complete = rc == 0 and completed == a.epochs and not interrupted and error is None
        status.update(status='stopped' if interrupted else ('completed' if complete else 'failed'),
                      returncode=rc, finished_at=now(), completed_training_epochs=completed)
        if error is not None:
            status['error'] = error
        write_json(run / 'status.json', status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        if not complete and (run / 'training.log').is_file():
            with (run / 'training.log').open(errors='replace') as log:
                print(''.join(deque(log, maxlen=35)), file=sys.stderr, end='')
        if interrupted:
            return 128 + interrupted[-1]
        return 0 if complete else (rc if rc > 0 else 1)


def main(argv=None):
    a = parse_args(argv)
    job = curriculum_job(a)
    checkpoint, source = checkpoint_record(a.method, a.domain, a.checkpoint_root)
    data = resolve_data(a.domain, a.road_root, a.synthetic_root)
    if a.dry_run:
        command = build_command(job, data, Path('<new-output-dir>'), checkpoint)
        print(json.dumps(dict(job=job, checkpoint=str(checkpoint), source=source, dataset_root=str(data)), indent=2))
        print(shlex.join(command))
        return 0
    gpus = {g['index']: g for g in gpu_inventory()}
    if a.gpu not in gpus:
        raise ValueError(f'Physical GPU {a.gpu} is unavailable or excluded by CUDA_VISIBLE_DEVICES')
    return run_training(a, job, data, checkpoint, source, gpus[a.gpu])


if __name__ == '__main__':
    raise SystemExit(main())
