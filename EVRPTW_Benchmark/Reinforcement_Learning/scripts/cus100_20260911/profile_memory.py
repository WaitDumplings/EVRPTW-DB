#!/usr/bin/env python3
"""Disposable native-trainer probes; never select these checkpoints for research."""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
RL = 'EVRPTW_Benchmark.Reinforcement_Learning.'
METHODS = {'am_evrptw': 'AM_EVRPTW.train', 'drl_ts': 'DRL_TS.train', 'evrptw_rl': 'EVRPTW_RL.train', 'rrnco': 'RRNCO_EVRPTW.train', 'terran': 'TERRAN.train'}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=True) + '\n')


def worker(command, output):
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    original = torch.optim.AdamW.step
    worker_started = time.monotonic()
    index = 0
    records = []
    def checked_step(self, *args, **kwargs):
        nonlocal index
        params = [p for g in self.param_groups for p in g['params'] if p.requires_grad]
        before = [p.detach().clone() for p in params]
        answer = original(self, *args, **kwargs)
        delta = sum(float((p.detach()-q).square().sum().item()) for p, q in zip(params, before)) ** .5
        finite = all(bool(torch.isfinite(p).all().item()) for p in params)
        index += 1
        row = {'optimizer_step': index, 'elapsed_since_worker_start_s': time.monotonic()-worker_started, 'parameter_delta_l2': delta, 'parameters_finite': finite,
               'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        records.append(row)
        write(output / 'optimizer_update_probe.json', records)
        if not finite or not delta > 0:
            raise RuntimeError('Nonfinite parameters or zero native optimizer update')
        return answer
    torch.optim.AdamW.step = checked_step
    sys.argv = [command[2]] + command[3:]
    runpy.run_module(command[2], run_name='__main__')


def command_for(args, output):
    cfg = ROOT / 'EVRPTW_Benchmark/Reinforcement_Learning/configs'
    dataset = Path(args.dataset_root)
    train = args.train_index or str(dataset / 'generation_plan/core/train/view_index.parquet')
    val = args.validation_index or str(dataset / 'generation_plan/core/val/view_index.parquet')
    families = args.family_root or str(dataset / 'materialized/families')
    cmd = [sys.executable, '-m', RL + METHODS[args.method], '--dataset-path', train,
           '--family-root', families, '--scale', 'Cus100', '--split-ids', 'train', '--track-ids', 'train',
           '--seed', '1234', '--device', 'cuda', '--training-epochs', str(args.updates),
           '--minimum-training-epochs', str(args.updates), '--physical-batch-size', str(args.batch),
           '--effective-batch-size', str(args.batch), '--samples-per-instance', str(args.n_traj),
           '--training-rollout-steps', str(args.steps), '--validation-rollout-steps', str((3 * args.steps + 1)//2),
           '--validation-dataset-path', val, '--validation-family-root', families,
           '--validation-limit', str(args.val_count), '--validation-decode-type', 'sampling',
           '--validation-candidates', str(args.n_traj), '--validation-seed', '910001234',
           '--validation-every-epochs', str(args.updates), '--validation-checkpoints', '1',
           '--protocol-id', 'cus100_20260911_disposable_memory_probe_v1', '--pilot-mode',
           '--objective-config', str(cfg / 'rivian_energy_vehicle_cost_v2.json'),
           '--reward-contract', args.reward_contract or str(cfg / 'drl_reward_contract_energy_vehicle_v3.json'),
           '--training-representation', args.representation, '--optimizer', 'adamw', '--weight-decay', '0.01',
           '--output-dir', str(output)]
    if args.method == 'drl_ts':
        cmd += ['--method-auxiliary-profile', str(cfg / 'drl_ts_soft_auxiliary_v1.json'), '--soft-stage-end-epoch', str(args.soft_end)]
    if args.method == 'evrptw_rl':
        cmd += ['--method-auxiliary-profile', str(cfg / 'evrptw_rl_station_auxiliary_v1.json')]
    if args.method == 'rrnco':
        cmd += ['--aft-mode', 'stable', '--distance-sampling', 'nearest', '--relation-temperature', '5',
                '--relation-chunk-size', '32', '--checkpoint-bias', '--reinforce-baseline', 'leave_one_out',
                '--learning-rate', '0.0001', '--graph-mode', 'full']
    if args.ema_warmup_steps is not None:
        cmd += ['--ema-warmup-steps', str(args.ema_warmup_steps)]
    if args.activation_checkpoint_stride:
        cmd += ['--activation-checkpoint-stride', str(args.activation_checkpoint_stride)]
    if args.euclidean_manifest:
        cmd += ['--euclidean-manifest', args.euclidean_manifest]
    if args.representation == 'E' and args.euclidean_manifest is None:
        # Synthetic indexes carry their own canonical artifact locations.
        # Omit unused Road family-root defaults from provenance.
        for option in ('--family-root', '--validation-family-root'):
            if option in cmd:
                index = cmd.index(option)
                del cmd[index:index+2]
    extra = json.loads(args.extra_args_json)
    if not isinstance(extra, list) or not all(isinstance(x, str) for x in extra):
        raise ValueError('--extra-args-json must encode a list of strings')
    cmd += extra
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--updates', type=int, default=2)
    p.add_argument('--n-traj', type=int, default=30)
    p.add_argument('--steps', type=int, default=240)
    p.add_argument('--val-count', type=int, default=4)
    p.add_argument('--soft-end', type=int, default=1)
    p.add_argument('--activation-checkpoint-stride', type=int, default=0)
    p.add_argument('--ema-warmup-steps', type=int)
    p.add_argument('--extra-args-json', default='[]')
    p.add_argument('--representation', choices=('E','G'), default='G')
    p.add_argument('--dataset-root', default=str(ROOT / 'EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823'))
    p.add_argument('--train-index')
    p.add_argument('--validation-index')
    p.add_argument('--family-root')
    p.add_argument('--euclidean-manifest')
    p.add_argument('--reward-contract')
    p.add_argument('--tag', default='')
    p.add_argument('--output-root', default=str(ROOT / 'EVRPTW_Benchmark/results/cus100_20260911/profiling'))
    p.add_argument('--worker-command')
    p.add_argument('--worker-output')
    args = p.parse_args()
    if args.worker_command:
        worker(json.loads(args.worker_command), Path(args.worker_output)); return
    tag = args.tag or time.strftime('%Y%m%dT%H%M%S')
    output = Path(args.output_root) / f'{args.method}_{args.representation}_b{args.batch}_t{args.n_traj}_h{args.steps}_{tag}'
    output.mkdir(parents=True, exist_ok=False)
    cmd = command_for(args, output)
    write(output / 'command.json', cmd)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    selfcmd = [sys.executable, str(Path(__file__).resolve()), '--method', args.method, '--gpu', str(args.gpu), '--batch', str(args.batch), '--worker-command', json.dumps(cmd), '--worker-output', str(output)]
    started = time.monotonic()
    peak_mib = 0
    peak_util = 0
    gpu_samples = []
    with (output / 'stdout.log').open('w') as stdout, (output / 'stderr.log').open('w') as stderr:
        child = subprocess.Popen(selfcmd, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
        write(output / 'pid.json', {'pid': child.pid, 'gpu': args.gpu})
        while child.poll() is None:
            result = subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
            for line in result.stdout.splitlines():
                values = [x.strip() for x in line.split(',')]
                if len(values)==2 and values[0]==str(child.pid):
                    try: peak_mib=max(peak_mib,int(values[1]))
                    except ValueError: pass
            utilization = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
            for line in utilization.stdout.splitlines():
                values = [x.strip() for x in line.split(',')]
                if len(values) == 3 and values[0] == str(args.gpu):
                    try:
                        percent, device_mib = int(values[1]), int(values[2])
                        peak_util = max(peak_util, percent)
                        gpu_samples.append({'elapsed_s': time.monotonic()-started, 'utilization_percent': percent, 'device_memory_mib': device_mib})
                    except ValueError: pass
            time.sleep(.2)
        status = child.wait()
    histories=[]
    history_path = output / 'logical_epoch_history.jsonl'
    if history_path.exists(): histories=[json.loads(x) for x in history_path.read_text().splitlines() if x]
    terran_rows = []
    csv_path = output / 'logs/train_log.csv'
    if not histories and csv_path.exists():
        with csv_path.open() as handle:
            terran_rows = list(csv.DictReader(handle))
        histories = [{
            'logical_epoch': int(row['epoch']),
            'epoch_wall_time_s': float(row['epoch_wall_time_s']) - float(row.get('eval_wall_time_s') or 0),
            'epoch_wall_including_validation_s': float(row['epoch_wall_time_s']),
            'mean_loss': float(row['policy_loss']),
            'value_loss': float(row['value_loss']),
            'mean_environment_feasible_rate': float(row['train_feasible_rate']),
            'rollout_budget_exhausted_rate': float(row['rollout_budget_exhausted_rate']),
        } for row in terran_rows]
    summary={'method':args.method,'representation':args.representation,'physical_batch_size':args.batch,'effective_batch_size':args.batch,
             'n_traj':args.n_traj,'steps':args.steps,'returncode':status,'wall_s':time.monotonic()-started,
             'gpu':args.gpu,'peak_gpu_utilization_percent':peak_util,
             'mean_gpu_utilization_percent':sum(x['utilization_percent'] for x in gpu_samples)/len(gpu_samples) if gpu_samples else None,
             'peak_process_mib':peak_mib,'peak_process_gib':peak_mib/1024,'updates_recorded':len(histories),
             'epoch_wall_s':[x.get('epoch_wall_time_s') for x in histories],
             'losses_finite':all(math.isfinite(float(x.get('mean_loss', 0))) and math.isfinite(float(x.get('value_loss', 0))) for x in histories) if histories else None,
             'training_success_rate':[x.get('mean_environment_feasible_rate') for x in histories],
             'rollout_budget_exhausted_rate':[x.get('rollout_budget_exhausted_rate') for x in histories],
             'note':'Disposable probe: no formal checkpoint reuse; short success is not convergence or long-run memory stability.'}
    validation_path = output / 'validation_summary.json'
    validation_history_path = output / 'validation_history.jsonl'
    validation = None
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
    elif validation_history_path.exists():
        validation = json.loads(validation_history_path.read_text().splitlines()[-1])
    if validation is not None:
        summary['validation'] = {key: validation.get(key) for key in ('instances', 'candidate_count', 'complete_and_feasible', 'verifier_summary_passed', 'validation_wall_time_s', 'mean_verified_cost_usd', 'mean_verified_distance_km', 'mean_verified_vehicle_count')}
        verified_rows = [x for x in validation.get('rows', []) if x.get('verifier_passed')]
        summary['verified_cost_decomposition_max_abs_error_usd'] = max((abs(float(x['objective_cost_usd']) - (0.151750972762646 * float(x['objective_distance_km']) + 413.6331536717643 * float(x['vehicle_count']))) for x in verified_rows), default=None)
    optimizer_probe = output / 'optimizer_update_probe.json'
    if optimizer_probe.exists():
        update_rows = json.loads(optimizer_probe.read_text())
        summary['optimizer_updates_recorded'] = len(update_rows)
        summary['optimizer_parameter_updates_finite_nonzero'] = all(x['parameters_finite'] and x['parameter_delta_l2'] > 0 for x in update_rows)
        summary['torch_peak_allocated_gib'] = max((x['cuda_peak_allocated_bytes'] for x in update_rows), default=0) / 2**30
        summary['torch_peak_reserved_gib'] = max((x['cuda_peak_reserved_bytes'] for x in update_rows), default=0) / 2**30
    if terran_rows:
        summary['epoch_wall_including_validation_s'] = [x['epoch_wall_including_validation_s'] for x in histories]
        summary['epoch_time_definition'] = 'TERRAN CSV epoch_wall_time_s minus eval_wall_time_s; PPO multiple optimizer steps remain separately counted'
    measured_times=[float(x['epoch_wall_time_s']) for x in histories if x.get('epoch_wall_time_s')]
    summary['training_instances_per_s']=args.batch*len(measured_times)/sum(measured_times) if measured_times else None
    write(output/'gpu_utilization_samples.json',gpu_samples)
    write(output/'probe_summary.json',summary)
    print(json.dumps({'output':str(output),**summary},indent=2), flush=True)
    if status:
        raise SystemExit(status)

if __name__=='__main__': main()
