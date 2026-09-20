#!/usr/bin/env python3
"""Road Cus100 -> Cus500 AM curriculum, one model synchronously trained on two GPUs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from script_curriculum import launch as shared


def parse_gpus(value):
    try:
        indices = [int(item.strip()) for item in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('GPUs must be two physical indices, e.g. 0,1') from exc
    if len(indices) != 2 or len(set(indices)) != 2 or min(indices) < 0:
        raise argparse.ArgumentTypeError('Select exactly two distinct nonnegative physical GPU indices')
    return indices


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=parse_gpus, default='0,1')
    parser.add_argument('--stage1-root', type=Path,
                        default=os.environ.get('CURRICULUM_STAGE1_ROOT', '/data/curriculum_stage1'))
    parser.add_argument('--source-checkpoint', type=Path,
                        help='Explicit alternative AM G Cus100 stage-1 checkpoint; validated and hashed')
    parser.add_argument('--road-root', default=os.environ.get('CURRICULUM_ROAD_ROOT',
                        os.environ.get('CUS500_ROAD_ROOT', os.environ.get('CUS100_ROAD_ROOT',
                        os.environ.get('EVRPTW_DATASET_ROOT')))))
    parser.add_argument('--output-root', type=Path,
                        default=os.environ.get('CURRICULUM_CUS500_OUTPUT_ROOT', '/data/curriculum_stage2_cus500'))
    parser.add_argument('--run-dir', type=Path, help='Explicit NEW output directory')
    parser.add_argument('--batch-size', type=int, default=12, help='Instances per GPU; global batch is twice this')
    parser.add_argument('--epochs', type=int, default=3000, help='Additional updates in the new stage')
    parser.add_argument('--validation-every', type=int, default=100)
    parser.add_argument('--validation-limit', type=int, default=500)
    parser.add_argument('--dry-run', action='store_true', help='Validate source/data and print command without GPU use')
    args = parser.parse_args(argv)
    if not 1 <= args.validation_limit <= 500:
        parser.error('Validation limit must be 1..500')
    args.method, args.domain = 'am_evrptw', 'G'
    return args


def curriculum_job(args):
    job = shared.make_job('am_evrptw', 500, 2, args.epochs, args.validation_every,
                          args.validation_limit, args.batch_size)
    job.update(schema='curriculum_stage2_road_cus500_v1',
               protocol_id='curriculum_stage2_road_cus500_v1',
               run_id='am_evrptw_G_Cus500_stage2_seed1234',
               training_representation='G', warm_start_scale_transition=True,
               launcher_source_paths=[
                   'script_curriculum/cus500_curr/launch.py',
                   'script_curriculum/cus500_curr/source_checkpoint.json',
                   'script_curriculum/cus500_curr/am.sh',
                   'EVRPTW_Benchmark/Reinforcement_Learning/AM_EVRPTW/distributed_train.py',
                   'EVRPTW_Benchmark/Reinforcement_Learning/common/distributed_protocol.py',
                   'EVRPTW_Benchmark/Reinforcement_Learning/common/distributed.py',
               ],
               calibration_status=f'curriculum_two_gpu_batch{args.batch_size}_smoke_passed'
                                  if args.batch_size in (4, 12) else 'explicit_batch_override_not_profiled')
    return job


def source_checkpoint(args):
    """Validate a trusted local checkpoint on CPU and pin its actual file identity."""
    import torch
    frozen = json.loads((HERE / 'source_checkpoint.json').read_text())
    explicit = args.source_checkpoint is not None
    path = (args.source_checkpoint if explicit else args.stage1_root / frozen['relative_path'])
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Stage-1 AM Road checkpoint unavailable: {path}; '
                                'set CURRICULUM_STAGE1_ROOT or --source-checkpoint')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        sha256 = digest.hexdigest()
        if not explicit and sha256 != frozen['sha256']:
            raise ValueError('Default stage-1 checkpoint differs from the frozen source hash')
        stream.seek(0)
        payload = torch.load(stream, map_location='cpu', weights_only=False)
    if payload.get('method') != 'AM-EVRPTW' or not isinstance(payload.get('model'), dict):
        raise ValueError('Source must contain AM-EVRPTW policy weights')
    saved = payload.get('args') or {}
    if not isinstance(saved, dict):
        saved = vars(saved)
    for key, expected in {'scale': 'Cus100', 'training_representation': 'G', 'seed': 1234,
                          **frozen['architecture']}.items():
        if saved.get(key) != expected:
            raise ValueError(f'Source {key} mismatch: {saved.get(key)!r} != {expected!r}')
    if payload.get('protocol_id') != frozen['source_protocol_id']:
        raise ValueError('Source must be a D_time curriculum stage-1 run, not the historical archive')
    expected_objective = json.loads((REPO / 'EVRPTW_Benchmark/Reinforcement_Learning/scripts/'
                                     'ablation_final/configs/objective_dtime.json').read_text())['objective']
    objective = payload.get('objective_config') or {}
    for key, value in expected_objective.items():
        if objective.get(key) != value:
            raise ValueError(f'Source objective {key} mismatch')
    epoch = int(payload.get('logical_epoch', 0))
    if epoch < 1 or (not explicit and epoch != frozen['logical_epoch']):
        raise ValueError('Source logical epoch is missing or disagrees with the frozen source')
    record = dict(method='AM-EVRPTW', source_domain='G', source_scale='Cus100',
                  source_protocol_id=payload['protocol_id'], checkpoint=str(path), sha256=sha256,
                  logical_epoch=epoch, architecture=frozen['architecture'], objective_config=objective,
                  source_selection='explicit_checkpoint_override' if explicit else frozen['selection'],
                  default_frozen_source=not explicit,
                  parent_warm_start_provenance={key: (payload.get('warm_start_provenance') or {}).get(key)
                      for key in ('checkpoint', 'checkpoint_sha256', 'source_logical_epoch')})
    if not explicit:
        record.update(validation=frozen['validation'],
                      source_run_completed_training_epochs=frozen['source_run_completed_training_epochs'])
    return path, record


def inspect_data(root, job):
    """Audit the published train/val lists, without preparing a stream or touching test data."""
    import pandas as pd
    train = pd.read_parquet(root / job['train_index'])
    val = pd.read_parquet(root / job['validation_index'])
    train = train[(train.customer_count == 500) & (train.split_id == 'train') & (train.track_id == 'train')]
    val = val[(val.customer_count == 500) & (val.split_id == 'val')]
    if len(train) != 10000 or len(val) != 500:
        raise ValueError(f'Expected full Cus500 corpus 10000 train/500 val; got {len(train)}/{len(val)}')
    if train.view_id.duplicated().any() or val.view_id.duplicated().any():
        raise ValueError('Duplicate dataset view IDs')
    if set(train.view_id) & set(val.view_id) or set(train.family_id) & set(val.family_id):
        raise ValueError('Train/val views or parent families overlap')
    return dict(train_views=len(train), validation_views=len(val),
                train_index_sha256=shared.digest(root / job['train_index']),
                validation_index_sha256=shared.digest(root / job['validation_index']),
                test_data_read=False)


def main(argv=None):
    args = parse_args(argv)
    job = curriculum_job(args)
    checkpoint, source = source_checkpoint(args)
    if args.source_checkpoint is not None:
        job['calibration_status'] = 'explicit_source_checkpoint_override_not_profiled'
    data = shared.resolve_data('G', args.road_root)
    audit = inspect_data(data, job)
    if args.dry_run:
        command = shared.build_command(job, data, Path('<new-output-dir>'), checkpoint)
        print(json.dumps(dict(job=job, checkpoint=str(checkpoint), source=source,
                              dataset_root=str(data), data_audit=audit, physical_gpus=args.gpus), indent=2))
        print(shlex.join(command))
        return 0
    available = {gpu['index']: gpu for gpu in shared.gpu_inventory()}
    unavailable = [index for index in args.gpus if index not in available]
    if unavailable:
        raise ValueError(f'Physical GPUs unavailable or excluded by CUDA_VISIBLE_DEVICES: {unavailable}')
    return shared.run_training(args, job, data, checkpoint, source, [available[index] for index in args.gpus])


if __name__ == '__main__':
    raise SystemExit(main())
