#!/usr/bin/env python3
"""Road Cus100 -> Cus500 curriculum, one policy synchronously trained on two or four GPUs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from script_curriculum import launch as shared
from script_curriculum.cus500_curr.checkpoint_source import load_source

DEFAULT_BATCHES = {
    'am_evrptw': 12, 'evrptw_rl': 24, 'drl_ts': 2, 'terran': 16, 'rrnco': 22,
}


def parse_gpus(value):
    try:
        indices = [int(item.strip()) for item in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('GPUs must be two or four physical indices, e.g. 0,1 or 0,1,2,3') from exc
    if len(indices) not in (2, 4) or len(set(indices)) != len(indices) or min(indices) < 0:
        raise argparse.ArgumentTypeError('Select two or four distinct nonnegative physical GPU indices')
    return indices


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=shared.METHODS, default='am_evrptw')
    parser.add_argument('--gpus', type=parse_gpus, default='0,1')
    parser.add_argument('--checkpoint-root', type=Path,
                        default=os.environ.get('CURRICULUM_CUS100_CKPT_ROOT', '/data/cus100_ckpt'))
    parser.add_argument('--source-checkpoint', type=Path,
                        help='Explicit alternative Road Cus100 stage-1 checkpoint; validated and hashed')
    parser.add_argument('--road-root', default=os.environ.get('CURRICULUM_ROAD_ROOT',
                        os.environ.get('CUS500_ROAD_ROOT', os.environ.get('CUS100_ROAD_ROOT',
                        os.environ.get('EVRPTW_DATASET_ROOT')))))
    parser.add_argument('--output-root', type=Path,
                        default=os.environ.get('CURRICULUM_CUS500_OUTPUT_ROOT', '/data/curriculum_stage2_cus500'))
    parser.add_argument('--run-dir', type=Path, help='Explicit NEW output directory')
    parser.add_argument('--batch-size', type=int, help='Per-GPU instances; default is model-specific')
    parser.add_argument('--epochs', type=int, default=3000, help='Additional updates in the new stage')
    parser.add_argument('--validation-every', type=int, default=100)
    parser.add_argument('--validation-limit', type=int, default=500)
    parser.add_argument('--dry-run', action='store_true', help='Validate source/data and print command without GPU use')
    args = parser.parse_args(argv)
    if not 1 <= args.validation_limit <= 500:
        parser.error('Validation limit must be 1..500')
    args.domain = 'G'
    return args


def curriculum_job(args):
    batch = DEFAULT_BATCHES[args.method] if args.batch_size is None else args.batch_size
    job = shared.make_job(args.method, 500, len(args.gpus), args.epochs, args.validation_every,
                          args.validation_limit, batch)
    job.update(schema='curriculum_stage2_road_cus500_v1',
               protocol_id='curriculum_stage2_road_cus500_v1',
               run_id=f'{args.method}_G_Cus500_stage2_seed1234',
               training_representation='G', warm_start_scale_transition=True,
               launcher_source_paths=[
                   'script_curriculum/cus500_curr/launch.py',
                   'script_curriculum/cus500_curr/checkpoint_source.py',
                   'script_curriculum/cus500_curr/batch_profiles.json',
                   'script_curriculum/cus500_curr/_launch.sh',
                   'EVRPTW_Benchmark/Reinforcement_Learning/common/distributed_protocol.py',
                   'EVRPTW_Benchmark/Reinforcement_Learning/common/distributed.py',
               ],
               calibration_status='prior_Cus500_recipe_reused_new_source_not_gpu_profiled'
                                  if batch == DEFAULT_BATCHES[args.method]
                                  else 'explicit_batch_override_not_profiled')
    module_file = job['train_module'].replace('.', '/') + '.py'
    job['launcher_source_paths'].append(module_file)
    if args.method == 'drl_ts':
        # Stage-1 Road policy already completed its soft stage; continue hard.
        job['soft_stage_end_epoch'] = 0
        job.pop('planned_full_run_soft_stage_end_epoch', None)
    return job


def source_checkpoint(args):
    return load_source(args.method, args.checkpoint_root, args.source_checkpoint)


def apply_batch_evidence(job, source):
    evidence = json.loads((HERE / 'batch_profiles.json').read_text())
    profile = evidence['models'][job['method']]
    if job['world_size'] == 4:
        profile = evidence.get('four_gpu_models', {}).get(job['method'], profile)
    job['batch_evidence'] = profile
    if (job['world_size'] == profile.get('world_size', 2)
            and job['physical_batch_size'] == profile['batch_per_gpu']
            and profile.get('checkpoint_sha256') == source['sha256']):
        job['calibration_status'] = ('curriculum_exact_source_two_gpu_smoke_passed'
                                     if job['world_size'] == 2
                                     else 'curriculum_exact_source_four_gpu_smoke_passed')


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
    apply_batch_evidence(job, source)
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
