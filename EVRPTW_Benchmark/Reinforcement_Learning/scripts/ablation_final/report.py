#!/usr/bin/env python3
"""Read-only training report; derived files go exclusively under OUTPUT/analysis.

Example:
  python report.py --output-root /data/ablation_final_road_cus100_300_20260919 --watch

No Torch/CUDA imports. Cost is conditional on each configuration's own verified
feasible validation instances; this reporter does not construct an intersection.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import time

NAMES = {'am_evrptw': 'AM-EVRPTW', 'evrptw_rl': 'EVRPTW-RL', 'drl_ts': 'DRL-TS',
         'terran': 'TERRAN', 'rrnco': 'RRNCO'}
ORDER = {name: i for i, name in enumerate(NAMES)}
TARGET_EPOCHS = (100, 200, 300)
TERMINAL = {'completed', 'failed', 'cancelled', 'stopped', 'stopped_after_launcher_error', 'interrupted'}
LAUNCH_TERMINAL = {'completed', 'failed', 'incomplete', 'cancelled', 'dry_run'}
BASE_FIELDS = ['run_id', 'solver', 'scale', 'run_status', 'recorded_run_status',
               'planned_epochs', 'last_training_epoch', 'physical_batch_per_gpu',
               'world_size', 'effective_instance_batch', 'training_trajectories',
               'current_instance_exposure', 'current_exposure_evidence',
               'current_customer_exposure', 'current_train_feasible_rate',
               'output_dir', 'notes']
EPOCH_FIELDS = BASE_FIELDS + ['logical_epoch', 'validation_status', 'instances',
                            'complete_and_feasible', 'fr_percent', 'cost_usd',
                            'instance_exposure_at_epoch', 'exposure_evidence',
                            'validation_source']
FINAL_FIELDS = BASE_FIELDS + ['result_300_status', 'instances', 'complete_and_feasible',
                            'fr_percent', 'cost_usd', 'instance_exposure_at_epoch',
                            'exposure_evidence', 'validation_source']


def integer(value):
    try:
        result = float(value)
        return int(result) if math.isfinite(result) and result.is_integer() else None
    except (TypeError, ValueError, OverflowError):
        return None


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def read_json(path, warnings):
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError('expected JSON object')
        return value
    except (OSError, ValueError) as exc:
        warnings.append(f'{path}: {exc}')
        return {}


def read_jsonl(path, warnings):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('expected JSON object')
            rows.append(row)
        except ValueError as exc:
            # A partially written final line is expected while training; all
            # skipped content remains visible in warnings rather than vanishing.
            warnings.append(f'{path}:{index}: skipped unreadable record ({exc})')
    return rows


def training_rows(run, method, warnings):
    path = run / ('logs/train_log.csv' if method == 'terran' else 'logical_epoch_history.jsonl')
    if path.suffix == '.csv':
        if not path.exists():
            raw = []
        else:
            # Ignore only the unfinished final CSV line; source is never edited.
            text = path.read_text()
            if text and not text.endswith('\n'):
                warnings.append(f'{path}: incomplete trailing CSV line ignored')
                text = text.rsplit('\n', 1)[0] + '\n' if '\n' in text else ''
            raw = list(csv.DictReader(io.StringIO(text)))
    else:
        raw = read_jsonl(path, warnings)
    rows = {}
    for row in raw:
        epoch = integer(row.get('logical_epoch', row.get('epoch')))
        if epoch is None or epoch < 1:
            warnings.append(f'{path}: invalid training epoch ignored')
            continue
        if epoch in rows:
            warnings.append(f'{path}: duplicate training epoch {epoch}; last record shown')
        rows[epoch] = row
    return rows


def exposure(training, epoch, method, effective_batch):
    """Distinguish per-epoch REINFORCE counters from cumulative TERRAN counts."""
    if epoch == 0:
        return 0, 'no_completed_training_epoch'
    if method == 'terran':
        value = integer(training.get(epoch, {}).get('samples_seen'))
        if value is not None:
            return value, 'observed_cumulative_samples_seen'
    else:
        values = [integer(training.get(e, {}).get('instances_seen')) for e in range(1, epoch + 1)]
        if all(value is not None for value in values):
            return sum(values), 'sum_of_observed_epoch_instances'
    if effective_batch is not None:
        return epoch * effective_batch, 'configured_global_batch_times_epoch_estimate'
    return None, 'unavailable'


def stage_note(job):
    method = job.get('method')
    if method == 'am_evrptw':
        extras = job.get('extra_args', [])
        def option(name, default):
            try:
                return int(extras[extras.index(name) + 1])
            except (ValueError, IndexError):
                return default
        steps = option('--steps-per-epoch', 2500)
        epochs = option('--baseline-warmup-epochs', 1)
        warmup = steps * epochs
        return (f'AM-EVRPTW EMA baseline warmup {steps} x {epochs} = {warmup} optimizer updates; '
                'the default 300-epoch pilot remains in EMA warmup, before greedy-rollout baseline')
    if method == 'drl_ts':
        boundary = job.get('soft_stage_end_epoch', 'unknown')
        planned = job.get('planned_full_run_soft_stage_end_epoch', boundary)
        return f'DRL-TS pilot soft stage through epoch {boundary}; full-run boundary {planned}; validation uses hard constraints'
    if method == 'evrptw_rl':
        extras = job.get('extra_args', [])
        try:
            warmup = int(extras[extras.index('--ema-warmup-steps') + 1])
        except (ValueError, IndexError):
            warmup = 1000
        return f'EVRPTW-RL EMA baseline warmup {warmup} updates; 300-epoch pilot does not reach greedy-rollout baseline'
    return ''


def validation_values(records, epoch, expected_instances, run, warnings):
    selected = [r for r in records if integer(r.get('logical_epoch')) == epoch]
    empty = dict(logical_epoch=epoch, validation_status='missing', instances=None,
                 complete_and_feasible=None, fr_percent=None, cost_usd=None,
                 validation_source=str(run / 'validation_history.jsonl'))
    if not selected:
        return empty
    if len(selected) != 1:
        warnings.append(f'{run}: duplicate validation epoch {epoch}; quality not selected arbitrarily')
        return {**empty, 'validation_status': 'duplicate_records'}
    record = selected[0]
    total, passed = integer(record.get('instances')), integer(record.get('complete_and_feasible'))
    if total is None or total <= 0 or passed is None or not 0 <= passed <= total:
        return {**empty, 'validation_status': 'invalid_counts'}
    if total != expected_instances:
        return {**empty, 'validation_status': 'cohort_count_mismatch', 'instances': total,
                'complete_and_feasible': passed}
    cost = number(record.get('mean_verified_cost_usd')) if passed else None
    status = 'observed' if passed == 0 or cost is not None else 'missing_verified_cost'
    return {**empty, 'validation_status': status, 'instances': total,
            'complete_and_feasible': passed, 'fr_percent': 100.0 * passed / total, 'cost_usd': cost}


def collect(output):
    warnings = []
    state = read_json(output / 'status.json', warnings)
    if not isinstance(state.get('jobs'), list) or not state['jobs']:
        raise ValueError(f'No readable nonempty jobs list in {output / "status.json"}: {warnings}')
    epoch_rows, final_rows = [], []
    records = sorted(state['jobs'], key=lambda r: (ORDER.get(r['job']['method'], 999), r['job']['run_id']))
    for record in records:
        job = record['job']
        run = Path(record['output_dir'])
        if not run.is_absolute():
            run = output / run
        method = job['method']
        training = training_rows(run, method, warnings)
        validations = read_jsonl(run / 'validation_history.jsonl', warnings)
        result = read_json(run / 'training_result.json', warnings) or record.get('training_result', {})
        last = max(training, default=0)
        # A terminal result can establish completion even if a buffered train log
        # is absent; we then explicitly label batch-derived exposure as estimated.
        terminal_epoch = integer(result.get('completed_training_epochs'))
        if result.get('status') == 'passed' and terminal_epoch is not None:
            last = max(last, terminal_epoch)
        effective = integer(job.get('effective_batch_size'))
        sampled, evidence = exposure(training, last, method, effective)
        status = record.get('status', 'unknown')
        recorded_status = status
        if state.get('status') in {'failed', 'incomplete', 'cancelled'} and status not in TERMINAL:
            status = 'interrupted_by_launcher' if status == 'running' else 'not_run_launcher_terminated'
        scale = integer(str(job.get('scale', '')).removeprefix('Cus'))
        last_training = training.get(max(training, default=0), {})
        train_fr = number(last_training.get('train_feasible_rate' if method == 'terran' else 'mean_environment_feasible_rate'))
        base = dict(run_id=job['run_id'], solver=NAMES.get(method, method), scale=job.get('scale'),
                    run_status=status, recorded_run_status=recorded_status,
                    planned_epochs=integer(job.get('training_epochs')), last_training_epoch=last,
                    physical_batch_per_gpu=integer(job.get('physical_batch_size')),
                    world_size=integer(job.get('world_size')), effective_instance_batch=effective,
                    training_trajectories=integer(job.get('training_trajectory_count')),
                    current_instance_exposure=sampled, current_exposure_evidence=evidence,
                    current_customer_exposure=sampled * scale if sampled is not None and scale else None,
                    current_train_feasible_rate=train_fr, output_dir=str(run), notes=stage_note(job))
        observed_epochs = {integer(v.get('logical_epoch')) for v in validations}
        epochs = sorted(set(TARGET_EPOCHS) | {e for e in observed_epochs if e is not None})
        by_epoch = {}
        for epoch in epochs:
            values = validation_values(validations, epoch, integer(job.get('validation_views')), run, warnings)
            seen, source = exposure(training, epoch, method, effective) if values['validation_status'] == 'observed' else (None, 'no_valid_validation_record')
            row = {**base, **values, 'instance_exposure_at_epoch': seen, 'exposure_evidence': source}
            epoch_rows.append(row)
            by_epoch[epoch] = row
        final = {k: v for k, v in by_epoch[300].items() if k != 'logical_epoch' and k != 'validation_status'}
        target_budget = base['planned_epochs'] == 300 and integer(job.get('validation_every_epochs')) == 100
        full_validation = (observed_epochs == set(TARGET_EPOCHS) and len(validations) == 3
                           and all(by_epoch[e]['validation_status'] == 'observed' for e in TARGET_EPOCHS))
        if not target_budget:
            final['result_300_status'] = 'not_a_300_epoch_pilot'
            # Never let a profile/smoke run masquerade as an epoch-300 result.
            for field in ('instances', 'complete_and_feasible', 'fr_percent', 'cost_usd', 'instance_exposure_at_epoch'):
                final[field] = None
        elif (status == 'completed' and result.get('status') == 'passed' and terminal_epoch == 300
              and full_validation and record.get('returncode', 0) == 0):
            final['result_300_status'] = 'completed_300'
        elif status in TERMINAL or status in {'interrupted_by_launcher', 'not_run_launcher_terminated'}:
            final['result_300_status'] = 'failed' if status != 'completed' else 'incomplete_completion_evidence'
        elif by_epoch[300]['validation_status'] == 'observed':
            final['result_300_status'] = 'validation_300_available_run_not_completed'
        else:
            final['result_300_status'] = 'queued' if status == 'queued' else 'pending_300'
        final_rows.append(final)
    done = state.get('status') in LAUNCH_TERMINAL or all(r.get('status') in TERMINAL for r in records)
    complete = all(r['result_300_status'] == 'completed_300' for r in final_rows)
    return state, epoch_rows, final_rows, warnings, done, complete


def atomic_text(path, text):
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(text)
    temporary.replace(path)


def write_csv(path, fields, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='raise')
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def plots(analysis, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(NAMES.values(), ['#3478bf', '#e68a2e', '#3c9b62', '#be5c80', '#8054aa']))
    for field, title, ylabel, filename in [
        ('cost_usd', 'Road validation cost', 'Verified feasible mean cost (USD)', 'cost_curve.png'),
        ('fr_percent', 'Road validation feasibility', 'Best-of-30 verified FR (%)', 'feasibility_curve.png'),
    ]:
        fig, ax = plt.subplots(figsize=(8.2, 4.7), layout='constrained')
        drawn = False
        for solver in NAMES.values():
            values = sorted((r for r in rows if r['solver'] == solver and r['logical_epoch'] in TARGET_EPOCHS
                             and r['planned_epochs'] == 300 and r['validation_status'] == 'observed'
                             and r[field] is not None), key=lambda r: r['logical_epoch'])
            if values:
                ax.plot([r['logical_epoch'] for r in values], [r[field] for r in values],
                        marker='o', label=solver, color=colors[solver], linewidth=1.8)
                drawn = True
        ax.set(title=title, xlabel='Logical epoch', ylabel=ylabel, xlim=(80, 320), xticks=TARGET_EPOCHS)
        ax.grid(alpha=0.25)
        if field == 'fr_percent':
            ax.set_ylim(-2, 102)
        if drawn:
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No completed validation checkpoint yet', ha='center', va='center', transform=ax.transAxes)
        temporary = analysis / (filename + f'.tmp.{os.getpid()}')
        fig.savefig(temporary, format='png', dpi=180)
        plt.close(fig)
        temporary.replace(analysis / filename)


def display(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def markdown(state, epochs, finals, warnings, done, complete):
    updated = datetime.now(timezone.utc).isoformat()
    message = '五模型 300 epoch 记录已全部验证完成。' if complete else (
        '调度已结束，存在失败、缺失或非 300 epoch 记录；不能视为全部完成。' if done else
        '训练仍在进行；下表只展示已写出的验证记录，缺项不是 0。')
    if complete and len(finals) != 5:
        message = f'本目录 {len(finals)} 个模型的 300 epoch 记录已全部验证完成。'
    lines = ['# Road Cus100 300 epoch 试跑汇总', '', f'更新时间：{updated}', '', message, '',
             f"调度状态：`{state.get('status', 'unknown')}`；源码：`{state.get('source', {}).get('git_head', 'unknown')}`。", '',
             '成本为 D_time 电费＋车辆固定费，单位 USD。Cost 仅对**各模型自身独立验证可行的实例**取均值，',
             '**没有计算五模型共同可行交集**；FR 分母为该次完整验证 cohort。验证采用每实例 best-of-30，',
             '训练轨迹可行率与验证 FR 是不同指标。CSV 保留原始浮点精度；下表仅作显示舍入。', '',
             '| 模型 | 状态 / 300结果 | 当前/最后 epoch | 每卡 batch × 卡数 = 全局 batch | 已完成实例曝光 | epoch100 cost / FR | epoch200 cost / FR | epoch300 cost / FR |',
             '|---|---|---:|---|---:|---|---|---|']
    by_key = {(r['run_id'], r['logical_epoch']): r for r in epochs}
    for row in finals:
        cells = []
        for epoch in TARGET_EPOCHS:
            item = by_key[(row['run_id'], epoch)]
            cells.append(f"{display(item['cost_usd'])} / {display(item['fr_percent'])}%" if
                         item['validation_status'] == 'observed' and row['planned_epochs'] == 300 else
                         f"— ({item['validation_status']})")
        batch = f"{row['physical_batch_per_gpu']} × {row['world_size']} = {row['effective_instance_batch']}"
        count = str(row['current_instance_exposure']) if row['current_instance_exposure'] is not None else '—'
        if row['current_exposure_evidence'].endswith('_estimate'):
            count += '（估计）'
        lines.append(f"| {row['solver']} | {row['run_status']} / {row['result_300_status']} | {row['last_training_epoch']} | {batch} | {count} | " + ' | '.join(cells) + ' |')
    lines += ['', '曝光次数指实例出现次数，**不乘训练轨迹数**。REINFORCE 按逐 epoch `instances_seen` 求和；',
              'TERRAN 读取 `logs/train_log.csv` 的累计 `samples_seen`。日志缺项时的 batch×epoch 估计单独标注，',
              '不同模型 batch 不同，因此相同 epoch 不等于相同训练曝光或计算预算。', '',
              '- DRL-TS：本轮为 soft 阶段试跑；不能将这 300 epoch 当作完整两阶段训练结论。',
              '- AM-EVRPTW：默认 EMA warmup 为 steps_per_epoch=2500 × baseline_warmup_epochs=1，即 2500 次更新；本轮 300 epoch 全在 EMA 阶段。',
              '- EVRPTW-RL：本轮尚未达到 1000 次更新的 EMA→greedy-rollout baseline 切换，不能据此判断后续阶段性能。',
              '- 本报告严格使用此目录 `status.json` 指定的运行；不搜索或并入早期 1 epoch profile/其他历史结果。', '']
    for row in finals:
        if row['notes']:
            lines.append(f"- {row['solver']} 配置记录：{row['notes']}")
    lines += ['', '文件：`summary_by_epoch.csv`（含缺失 epoch 行）、`summary_at_300.csv`（每模型一行，含状态）、',
              '`cost_curve.png`、`feasibility_curve.png`。曲线只绘制本轮已验证的 100/200/300 epoch 观测点；不填补缺失值。', '',
              '原始训练与验证文件均未改写。`summary_at_300.csv` 的 `completed_300` 需要正常退出、',
              '终止记录确认 300 epoch，以及三个各自完整且不重复的验证记录同时成立。', '']
    if warnings:
        lines += ['## 读取提示', ''] + [f'- {warning}' for warning in warnings] + ['']
    return '\n'.join(lines)


def generate(output):
    output = Path(output).resolve()
    state, epoch_rows, final_rows, warnings, done, complete = collect(output)
    analysis = output / 'analysis'
    analysis.mkdir(parents=True, exist_ok=True)
    write_csv(analysis / 'summary_by_epoch.csv', EPOCH_FIELDS, epoch_rows)
    write_csv(analysis / 'summary_at_300.csv', FINAL_FIELDS, final_rows)
    plots(analysis, epoch_rows)
    atomic_text(analysis / 'RESULTS_SUMMARY.md', markdown(state, epoch_rows, final_rows, warnings, done, complete))
    return dict(output=str(analysis), launcher_status=state.get('status'), terminal=done,
                completed_300=complete, models=len(final_rows),
                valid_checkpoints=sum(r['validation_status'] == 'observed' for r in epoch_rows),
                warnings=warnings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--watch', action='store_true', help='Refresh every 30 seconds until the launcher/tasks are terminal')
    parser.add_argument('--interval', type=float, default=30.0, help='Watch refresh seconds (default 30)')
    args = parser.parse_args()
    if not 1 <= args.interval <= 60:
        parser.error('--interval must be between 1 and 60 seconds')
    while True:
        try:
            result = generate(args.output_root)
            print(json.dumps(result, allow_nan=False), flush=True)
        except (OSError, ValueError) as exc:
            if not args.watch:
                raise
            print(json.dumps({'report_error': str(exc), 'retry_seconds': args.interval}), flush=True)
            time.sleep(args.interval)
            continue
        if not args.watch:
            return 0
        if result['terminal']:
            return 0 if result['completed_300'] else 1
        time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
