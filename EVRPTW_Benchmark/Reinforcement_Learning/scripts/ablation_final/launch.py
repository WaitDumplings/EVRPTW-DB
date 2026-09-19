#!/usr/bin/env python3
"""Fresh Road training for the ablation branch; single/dual/3-or-4 GPU policies."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
PREFIX = 'EVRPTW_Benchmark.Reinforcement_Learning.'
METHODS = ('terran', 'evrptw_rl', 'am_evrptw', 'rrnco', 'drl_ts')
NAMES = {'am_evrptw': 'AM-EVRPTW', 'evrptw_rl': 'EVRPTW-RL', 'drl_ts': 'DRL-TS', 'terran': 'TERRAN', 'rrnco': 'RRNCO'}
MODULES = {'am_evrptw': 'AM_EVRPTW', 'evrptw_rl': 'EVRPTW_RL', 'drl_ts': 'DRL_TS', 'terran': 'TERRAN', 'rrnco': 'RRNCO_EVRPTW'}
BATCHES = {100: {'am_evrptw': 108, 'evrptw_rl': 200, 'drl_ts': 24, 'terran': 384, 'rrnco': 50},
           500: {'am_evrptw': 4, 'evrptw_rl': 24, 'drl_ts': 2, 'terran': 16, 'rrnco': 22},
           1000: {'am_evrptw': 1, 'evrptw_rl': 12, 'drl_ts': 1, 'terran': 4, 'rrnco': 4}}
TRAIN_INDEX = 'generation_plan/core/train/view_index.parquet'
VAL_INDEX = 'generation_plan/core/val/view_index.parquet'
RELEASE = 'us_11city_full_clean_v7_bbde5db_20260823'
STOP = False


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def gpu_inventory():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used', '--format=csv,noheader,nounits'], text=True)
    rows = []
    for values in csv.reader(raw.splitlines()):
        idx, uuid, name, total, used = [v.strip() for v in values]
        rows.append(dict(index=int(idx), uuid=uuid, name=name, total_mib=int(total), used_mib=int(used)))
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        selected = {v.strip() for v in visible.split(',') if v.strip()}
        rows = [r for r in rows if str(r['index']) in selected or r['uuid'] in selected]
    return rows


def gpu_count(scale, visible_count):
    if scale == 100:
        needed = 1
    elif scale == 500:
        needed = 2
    elif scale == 1000:
        needed = 4 if visible_count >= 4 else 3
    else:
        raise ValueError('Only Cus100/500/1000 are configured')
    if visible_count < needed:
        raise ValueError(f'Cus{scale} requires {needed} GPUs, found {visible_count}')
    return needed


def compute_busy_uuids():
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    busy = set()
    for fields in csv.reader(raw.splitlines()):
        if len(fields) != 2:
            continue
        uuid, pid = [v.strip() for v in fields]
        try:
            executable = Path(f'/proc/{int(pid)}/exe').resolve().name
        except (OSError, ValueError):
            executable = ''
        if executable == 'gnome-remote-desktop-daemon':
            continue  # Desktop C+G service, not a training job; leave it running.
        busy.add(uuid)
    return busy


def dataset_root(requested):
    if requested:
        roots = [Path(requested).expanduser().resolve()]
    else:
        roots = [REPO / 'EVRPTW_Dataset/Instances_v2' / RELEASE,
                 REPO / 'EVRPTW_Dataset/Instances_v2/us_11city',
                 REPO.parent / 'EVRPTW-DB/EVRPTW_Dataset/Instances_v2' / RELEASE]
    for root in roots:
        if (root / TRAIN_INDEX).is_file() and (root / VAL_INDEX).is_file():
            return root.resolve()
    raise FileNotFoundError('Road train/val unavailable; pass --road-root or ABLATION_ROAD_ROOT')


def make_job(method, scale, world_size, epochs, validation_every=100, validation_limit=500, batch=None):
    if epochs < 1 or validation_every < 1 or epochs % validation_every:
        raise ValueError('Epoch budget must be a positive multiple of validation interval')
    original = [json.loads(line) for line in (HERE.parent / 'cus100_20260911/cus100_seed1234_jobs.jsonl').read_text().splitlines() if line.strip()]
    base = next(j for j in original if j['method'] == method and j['representation'] == 'G')
    fields = ['method', 'optimizer_name', 'optimizer_weight_decay', 'method_auxiliary_profile_path', 'extra_args', 'num_minibatches', 'ppo_step_chunk_size', 'terran_terminal_success_bonus']
    job = {k: copy.deepcopy(base[k]) for k in fields if k in base}
    batch = BATCHES[scale][method] if batch is None else batch
    if batch < 1:
        raise ValueError('Physical batch must be positive')
    steps = {100: 240, 500: 1700, 1000: 1250}[scale]
    validation_steps = (3 * steps + 1) // 2
    if method == 'evrptw_rl' and scale == 500:
        steps, validation_steps = 600, 700
    job.update(schema='ablation_road_dtime_v1', protocol_id='ablation_road_dtime_v1',
               method=method, scale=f'Cus{scale}', seed=1234, source_kind='stage2_road',
               train_index=TRAIN_INDEX, validation_index=VAL_INDEX, training_representation='G',
               physical_batch_size=batch, effective_batch_size=batch * world_size, world_size=world_size,
               training_epochs=epochs, minimum_training_epochs=epochs,
               early_stop_start_epoch=0, early_stop_patience_validations=0,
               training_rollout_steps=steps, validation_rollout_steps=validation_steps,
               validation_every_epochs=validation_every, post_minimum_validation_every_epochs=validation_every,
               validation_checkpoints=epochs // validation_every, validation_views=validation_limit,
               validation_seed=910001234, validation_decode_type='sampling',
               training_trajectory_count=30, validation_candidate_count=30, final_validation_views=0,
               objective_distance_source='running_time_path_distance_km',
               objective_config_path=str((HERE / 'configs/objective_dtime.json').relative_to(REPO)),
               reward_contract_config_path=str((HERE / 'configs/reward_dtime.json').relative_to(REPO)),
               customer_exposure_budget=epochs * batch * world_size * scale,
               run_id=f'{method}_road_cus{scale}_seed1234', warm_start_source_commit='',
               calibration_status='prior_Cus100_batch_reused_pending_Dtime_pilot' if scale == 100 else 'initial_multigpu_profile_not_yet_benchmarked')
    suffix = 'train' if world_size == 1 else ('train_distributed' if method == 'terran' else 'distributed_train')
    job['train_module'] = PREFIX + MODULES[method] + '.' + suffix
    job['extra_args'].extend(['--objective-distance-source', 'running_time_path_distance_km'])
    if method == 'evrptw_rl':
        job['extra_args'].extend(['--graph-aggregation', 'mean', '--learning-rate', '0.001', '--ema-warmup-steps', '1000'])
    if method == 'evrptw_rl' and scale == 500:
        job['extra_args'].extend(['--validation-rollout-policy', 'explicit'])
    if method == 'drl_ts':
        job['soft_stage_end_epoch'] = min(2500, epochs)
        job['planned_full_run_soft_stage_end_epoch'] = 2500
    if method == 'terran':
        config = HERE / 'configs/terran.yaml'
        job['terran_config_path'] = str(config.relative_to(REPO))
        job['terran_config_sha256'] = digest(config)
        job['ppo_step_chunk_size'] = {100: 64, 500: 16, 1000: 8}[scale]
        job['num_charging_stations'] = 20 if scale == 100 else 50
        job['extra_args'].extend(['--num-charging-stations', str(job['num_charging_stations'])])
    elif scale > 100 and method in {'evrptw_rl', 'drl_ts', 'rrnco'}:
        if '--activation-checkpoint-stride' not in job['extra_args']:
            job['extra_args'].extend(['--activation-checkpoint-stride', '1'])
    return job


def prepare_streams(root, output, jobs):
    import pandas as pd
    from EVRPTW_Benchmark.Reinforcement_Learning.common.training_stream import atomic_write_stream, build_training_stream, load_training_stream_contract
    index, val_index = pd.read_parquet(root / TRAIN_INDEX), pd.read_parquet(root / VAL_INDEX)
    scale = int(jobs[0]['scale'][3:])
    train = index[(index.customer_count == scale) & (index.split_id == 'train')]
    val = val_index[(val_index.customer_count == scale) & (val_index.split_id == 'val')]
    expected = {100: 50000, 500: 10000, 1000: 5000}[scale]
    if len(train) != expected or len(val) != 500:
        raise ValueError(f'Expected {expected} train/500 val, got {len(train)}/{len(val)}')
    if set(train.view_id) & set(val.view_id) or set(train.family_id) & set(val.family_id):
        raise ValueError('Train/validation IDs or parent families overlap')
    index_hash = digest(root / TRAIN_INDEX)
    for job in jobs:
        count = job['training_epochs'] * job['effective_batch_size']
        path = output / 'artifacts' / f"stream_Cus{scale}_seed{job['seed']}_n{count}.parquet"
        if not path.exists():
            frame, metadata = build_training_stream(index, scale=job['scale'], seed=job['seed'], sample_count=count)
            metadata.update(source_index=str(root / TRAIN_INDEX), source_index_sha256=index_hash,
                            source_kind='stage2_road', allowed_family_ids_source=None, allowed_family_ids_sha256=None)
            atomic_write_stream(path, frame, metadata)
        contract = load_training_stream_contract(path)
        if any(contract[k] != value for k, value in {'source_index_sha256': index_hash, 'sample_count': count, 'scale': job['scale'], 'seed': job['seed']}.items()):
            raise ValueError('Existing stream contract mismatch')
        job.update(training_stream_path=str(path), training_stream_contract_sha256=contract['sha256'],
                   train_index_sha256=index_hash, validation_index_sha256=digest(root / VAL_INDEX))
    return dict(train_views=len(train), validation_views=len(val), train_val_parent_overlap=0, test_data_read=False)


def command_for(job, root, run):
    from EVRPTW_Benchmark.Reinforcement_Learning.scripts.drl_job_runtime import training_command
    command = training_command(job, {'repo': REPO, 'dataset': root}, run, resume=False)
    command[0] = sys.executable
    command.extend(job['extra_args'])
    if job['world_size'] > 1:
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                   f"--nproc_per_node={job['world_size']}", '--max_restarts=0', '--module', *command[2:]]
        command.extend(['--expected-world-size', str(job['world_size'])])
    return command


def read_rows(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def summarize(output, records):
    rows = []
    for record in records:
        run = Path(record['output_dir'])
        for v in read_rows(run / 'validation_history.jsonl'):
            rows.append({'solver': NAMES[record['job']['method']], 'scale': record['job']['scale'],
                         'logical_epoch': v.get('logical_epoch'), 'instances': v.get('instances'),
                         'complete_and_feasible': v.get('complete_and_feasible'),
                         'cost_usd': v.get('mean_verified_cost_usd'), 'validation_record_json': json.dumps(v)})
    path = output / 'validation_summary.csv'
    fields = ['solver', 'scale', 'logical_epoch', 'instances', 'complete_and_feasible', 'cost_usd', 'validation_record_json']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', type=int, choices=(100, 500, 1000), default=100)
    parser.add_argument('--models', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--validation-every', type=int, default=100)
    parser.add_argument('--validation-limit', type=int, default=500)
    parser.add_argument('--batch-size', type=int, help='Explicit per-GPU override, applied to all selected methods')
    parser.add_argument('--gpus', default='auto', help='Physical GPU indices; default all visible GPUs')
    parser.add_argument('--road-root', default=os.environ.get('ABLATION_ROAD_ROOT') or os.environ.get('CUS100_ROAD_ROOT'))
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    gpus = gpu_inventory()
    if args.gpus != 'auto':
        indices = [int(v) for v in args.gpus.split(',')]
        if len(set(indices)) != len(indices) or not set(indices).issubset({g['index'] for g in gpus}):
            raise ValueError('GPU list is duplicated or not visible')
        gpus = [next(g for g in gpus if g['index'] == idx) for idx in indices]
    world_size = gpu_count(args.scale, len(gpus))
    if not 1 <= args.validation_limit <= 500:
        parser.error('--validation-limit must be between 1 and 500')
    if len(set(args.models)) != len(args.models):
        parser.error('--models must not contain duplicate methods')
    if len(gpus) < world_size:
        raise ValueError(f'Cus{args.scale} requires {world_size} selected GPUs')
    root = dataset_root(args.road_root)
    output = (args.output_root or REPO / 'EVRPTW_Benchmark/results' /
              f"ablation_road_cus{args.scale}_{args.epochs}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs = [make_job(m, args.scale, world_size, args.epochs, args.validation_every, args.validation_limit, args.batch_size) for m in args.models]
    data_audit = prepare_streams(root, output, jobs)
    records = [dict(job=j, output_dir=str(output / 'runs' / j['run_id']), status='queued') for j in jobs]
    for record in records:
        record['command'] = command_for(record['job'], root, Path(record['output_dir']))
    from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_dual_20260913.common import source_snapshot
    state = dict(schema='ablation_launch_v1', started_at=now(), host=socket.gethostname(), pid=os.getpid(),
                 source=source_snapshot(REPO), dataset=str(root), data_audit=data_audit,
                 selected_gpus=gpus, world_size_per_model=world_size, jobs=records,
                 python=sys.executable, status='dry_run' if args.dry_run else 'running')
    write_json(output / 'status.json', state)
    print(json.dumps({'output': str(output), 'world_size_per_model': world_size,
                      'jobs': [j['run_id'] for j in jobs], 'dry_run': args.dry_run}), flush=True)
    if args.dry_run:
        return 0
    lock_dir = Path('/tmp') / f'evrptw-ablation-gpu-locks-{os.getuid()}'
    lock_dir.mkdir(exist_ok=True)
    lock_handles = []
    running = []
    def stop(signum, frame):
        global STOP
        STOP = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    try:
        busy = compute_busy_uuids()
        if any(g['uuid'] in busy for g in gpus):
            raise RuntimeError('Selected GPU has an existing compute process; no task was stopped')
        for gpu in gpus:
            handle = (lock_dir / (gpu['uuid'] + '.lock')).open('a+')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_handles.append(handle)
        while any(r['status'] in ('queued', 'running') for r in records):
            if STOP:
                for child, record, handles in running:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                for record in records:
                    if record['status'] == 'queued':
                        record['status'] = 'cancelled'
            used = {idx for _, r, _ in running for idx in r['gpus']}
            available = [g for g in gpus if g['index'] not in used]
            for record in records:
                if STOP or record['status'] != 'queued' or len(available) < world_size:
                    continue
                selected, available = available[:world_size], available[world_size:]
                run = Path(record['output_dir'])
                run.mkdir(parents=True, exist_ok=False)
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in selected),
                           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                           NUMBA_NUM_THREADS='2', PYTHONUNBUFFERED='1')
                logs = [(run / name).open('w') for name in ('stdout.log', 'stderr.log')]
                child = subprocess.Popen(record['command'], cwd=REPO, env=env,
                                         stdout=logs[0], stderr=logs[1], start_new_session=True)
                record.update(status='running', pid=child.pid, started_at=now(), gpus=[g['index'] for g in selected])
                write_json(run / 'launch_record.json', record)
                running.append((child, record, logs))
            for child, record, logs in running[:]:
                rc = child.poll()
                if rc is None:
                    continue
                for handle in logs:
                    handle.close()
                result_path = Path(record['output_dir']) / 'training_result.json'
                result = json.loads(result_path.read_text()) if result_path.exists() else {}
                validations = read_rows(Path(record['output_dir']) / 'validation_history.jsonl')
                expected = set(range(args.validation_every, args.epochs + 1, args.validation_every))
                actual = {v.get('logical_epoch') for v in validations}
                completed = (result.get('status') == 'passed'
                             and result.get('completed_training_epochs') == args.epochs
                             and actual == expected and len(validations) == len(expected)
                             and all(v.get('instances') == args.validation_limit for v in validations))
                record.update(status='completed' if rc == 0 and completed else 'failed', returncode=rc,
                              finished_at=now(), training_result=result,
                              validated_epochs=sorted(e for e in actual if e is not None))
                running.remove((child, record, logs))
            state['updated_at'] = now()
            write_json(output / 'status.json', state)
            summarize(output, records)
            if running:
                time.sleep(10)
        state.update(status='completed' if all(r['status'] == 'completed' for r in records) else 'incomplete', finished_at=now())
        write_json(output / 'status.json', state)
        summarize(output, records)
        return 0 if state['status'] == 'completed' else 1
    except BaseException as exc:
        # Own children must not continue unmonitored after a scheduler failure.
        for child, record, logs in running:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for child, record, logs in running:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            for handle in logs:
                handle.close()
            record.update(status='stopped_after_launcher_error', returncode=child.returncode, finished_at=now())
        state.update(status='failed', error=repr(exc), updated_at=now())
        write_json(output / 'status.json', state)
        raise
    finally:
        for handle in lock_handles:
            handle.close()


if __name__ == '__main__':
    raise SystemExit(main())
