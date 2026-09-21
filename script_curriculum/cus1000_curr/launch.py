#!/usr/bin/env python3
"""Road AM Cus500 -> Cus1000 curriculum on four GPUs, weights-only warm start."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO))
from script_curriculum import launch as shared
from script_curriculum.cus500_curr.checkpoint_source import ARCHITECTURES, _expect, _validate_weights


def parse_gpus(value):
    try: ids=[int(x) for x in value.split(',')]
    except ValueError as exc: raise argparse.ArgumentTypeError('Use four indices: 0,1,2,3') from exc
    if len(ids)!=4 or len(set(ids))!=4 or min(ids)<0:
        raise argparse.ArgumentTypeError('Select exactly four different physical GPU indices')
    return ids


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',type=parse_gpus,default='0,1,2,3')
    p.add_argument('--source-checkpoint',type=Path,default=os.environ.get('CURRICULUM_CUS500_AM_CKPT','/data/cus500_ckpt/am.ckpt'))
    p.add_argument('--road-root',default=os.environ.get('CURRICULUM_ROAD_ROOT',os.environ.get('CUS100_ROAD_ROOT',os.environ.get('EVRPTW_DATASET_ROOT'))))
    p.add_argument('--output-root',type=Path,default=os.environ.get('CURRICULUM_CUS1000_OUTPUT_ROOT','/data/curriculum_stage3_cus1000'))
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--batch-size',type=int,default=3,help='Instances per GPU; global batch is four times this')
    p.add_argument('--epochs',type=int,default=2000,help='New stage optimizer updates, not source checkpoint epoch')
    p.add_argument('--validation-every',type=int,default=100)
    p.add_argument('--validation-limit',type=int,default=500)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    if not 1<=a.validation_limit<=500: p.error('validation-limit must be 1..500')
    a.method='am_evrptw';a.domain='G'
    return a


def source_checkpoint(path):
    import torch
    from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import objective_from_checkpoint
    from EVRPTW_Benchmark.Reinforcement_Learning.common.training_protocol import assert_checkpoint_training_signature
    from types import SimpleNamespace
    path=path.expanduser().resolve()
    payload=torch.load(path,map_location='cpu',weights_only=False)
    _expect(payload,dict(method='AM-EVRPTW',protocol_id='curriculum_stage2_road_cus500_v1'),'')
    saved=payload['args']
    _expect(saved,dict(scale='Cus500',training_representation='G',seed=1234),'args.')
    _expect(saved,ARCHITECTURES['am_evrptw'],'architecture.')
    assert_checkpoint_training_signature(payload,SimpleNamespace(**saved))
    objective=objective_from_checkpoint(payload).to_dict()
    expected=json.loads((REPO/'EVRPTW_Benchmark/Reinforcement_Learning/scripts/ablation_final/configs/objective_dtime.json').read_text())['objective']
    _expect(objective,expected,'objective.')
    epoch=payload.get('logical_epoch')
    if not isinstance(epoch,int) or isinstance(epoch,bool) or epoch<1:
        raise ValueError('Source must record a positive selected epoch')
    checked=_validate_weights('am_evrptw',payload.get('model'))
    source=dict(checkpoint=str(path),sha256=shared.digest(path),method='AM-EVRPTW',
        source_scale='Cus500',source_domain='G',logical_epoch=epoch,
        source_protocol_id=payload['protocol_id'],objective_config=objective,
        weight_compatibility=checked,parent_warm_start_provenance=payload.get('warm_start_provenance'),
        continuation_policy='weights_only_new_optimizer_baseline_and_epoch_counter')
    return path,source


def curriculum_job(a):
    job=shared.make_job('am_evrptw',1000,4,a.epochs,a.validation_every,a.validation_limit,a.batch_size)
    job.update(schema='curriculum_stage3_road_cus1000_v1',protocol_id='curriculum_stage3_road_cus1000_v1',
        run_id='am_evrptw_G_Cus1000_stage3_seed1234',warm_start_scale_transition=True,
        calibration_status='batch_override_requires_exact_source_gpu_smoke',
        launcher_source_paths=['script_curriculum/cus1000_curr/launch.py',
            'script_curriculum/cus1000_curr/am_cus500_to_1000.sh',
            'script_curriculum/cus500_curr/checkpoint_source.py',
            'EVRPTW_Benchmark/Reinforcement_Learning/common/distributed_protocol.py',
            'EVRPTW_Benchmark/Reinforcement_Learning/AM_EVRPTW/distributed_train.py'])
    profile=HERE/'batch_profile.json'
    if profile.exists():
        evidence=json.loads(profile.read_text())
        if evidence.get('batch_per_gpu')==a.batch_size and evidence.get('source_sha256')==shared.digest(a.source_checkpoint):
            job.update(calibration_status='exact_source_four_gpu_smoke_passed',batch_evidence=evidence)
    return job


def main(argv=None):
    args=parse_args(argv)
    checkpoint,source=source_checkpoint(args.source_checkpoint)
    job=curriculum_job(args)
    data=shared.resolve_data('G',args.road_root)
    if args.dry_run:
        import pandas as pd
        train=pd.read_parquet(data/job['train_index']);val=pd.read_parquet(data/job['validation_index'])
        train=train[(train.customer_count==1000)&(train.split_id=='train')&(train.track_id=='train')]
        val=val[(val.customer_count==1000)&(val.split_id=='val')]
        if len(train)!=5000 or len(val)!=500 or set(train.family_id)&set(val.family_id):
            raise ValueError('Expected disjoint Road Cus1000 5000 train/500 val')
        print(json.dumps(dict(job=job,source=source,data_root=str(data)),indent=2))
        print(shlex.join(shared.build_command(job,data,Path('<new-output-dir>'),checkpoint)))
        return 0
    available={r['index']:r for r in shared.gpu_inventory()}
    if any(i not in available for i in args.gpus):raise ValueError('Requested GPUs not available')
    return shared.run_training(args,job,data,checkpoint,source,[available[i] for i in args.gpus])


if __name__=='__main__':raise SystemExit(main())
