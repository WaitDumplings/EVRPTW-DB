#!/usr/bin/env python3
"""Audit BKS routes and continue ALNS for a per-instance wall-clock budget."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

# One compute thread per instance, independent of the host's BLAS defaults.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'
REPO = Path(__file__).resolve().parents[4]
META = REPO / 'EVRPTW_Benchmark/MetaHeuristics'
for directory in (REPO / 'EVRPTW_Core', REPO / 'EVRPTW_Dataset_Generator/src', META, META / 'ALNS_Solver'):
    sys.path.insert(0, str(directory))

# The BKS directory is beside the repository under ICLR, independent of cwd.
DEFAULT_BKS = REPO.parent / 'bks_cus1000_T1_three_models_20260923' / 'BKS_Cus1000_T1.jsonl'
DEFAULT_CHECKPOINTS = (900, 1800, 2700, 3600, 4500, 5400, 6300, 7200)
INDEX_RELATIVE = Path('generation_plan/core/test/test1_new_seed/view_index.parquet')
DATASET_NAME = 'us_11city_full_clean_v7_bbde5db_20260823'
OBJECTIVE_PATH = REPO / 'EVRPTW_Benchmark/Reinforcement_Learning/configs/rivian_energy_vehicle_cost_v2.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read(path):
    return json.loads(Path(path).read_text())


def select_part(records, part):
    if len(records) != 500 or len({r['instance_id'] for r in records}) != 500:
        raise ValueError('Expected exactly 500 distinct BKS instances')
    # The BKS JSONL order defines the halves, never index/parquet order.
    return records if part == 'all' else records[:250] if part == 'upper' else records[250:]


def resolve_index(explicit, data_root):
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if p.is_dir():
            p = p / INDEX_RELATIVE
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    # Accept both the restored release name and the original source name.
    # Use script-derived repository/ICLR locations, never the caller's cwd.
    names = ('us_11city', DATASET_NAME)
    roots = (Path(data_root).expanduser().resolve(), REPO, REPO.parent,
             REPO.parent / 'EVRPTW-DB')
    bases = []
    for root in roots:
        bases.append(root)
        for name in names:
            bases.extend((root / name, root / 'EVRPTW_Dataset/Instances_v2' / name,
                          root / 'Instances_v2' / name))
    # Match the shared benchmark launcher restore layouts.
    for ancestor in REPO.parents[:3]:
        runtime_dataset = ancestor / 'evrptw_runtime/EVRPTW_Dataset'
        bases.append(runtime_dataset)
        for name in names:
            bases.append(runtime_dataset / 'Instances_v2' / name)
    candidates = list(dict.fromkeys(base / INDEX_RELATIVE for base in bases))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = '\n'.join(f'  {candidate}' for candidate in candidates)
    raise FileNotFoundError(
        'Cannot locate the Cus1000 T1 dataset index. BKS routes alone do not include '
        'the instance matrices. Set EVRPTW_DATASET_ROOT or --dataset-path to the '
        'dataset root (containing generation_plan and materialized), or pass the '
        'T1 view_index.parquet file with --dataset-path. Checked:\n' + checked)



def build_tasks(args):
    from benchmark_common import build_input_tasks, stable_view_seed
    from evrptw_core.objective import load_objective
    records = [json.loads(line) for line in args.bks.read_text().splitlines() if line.strip()]
    selected = select_part(records, args.part)
    index = resolve_index(args.dataset_path, args.data_root)
    inputs = build_input_tasks(index, family_root=args.family_root, scales={'Cus1000'})
    by_id = {v['stage2_task']['view_id']: v for v in inputs}
    if set(by_id) != {r['instance_id'] for r in records}:
        raise ValueError('Cus1000 T1 index IDs must exactly match all 500 BKS IDs')
    objective = load_objective(OBJECTIVE_PATH)
    hashes = {str(p.relative_to(REPO)): digest(p) for p in [Path(__file__),
              META/'ALNS_Solver/solver.py', META/'ALNS_Solver/instance_adapter.py',
              META/'benchmark_common.py', REPO/'EVRPTW_Core/evrptw_core/objective.py']}
    contract = dict(schema='alns_bks_refine_v1', bks_sha256=digest(args.bks),
                    index_sha256=digest(index), source_hashes=hashes, part=args.part,
                    instance_ids=[r['instance_id'] for r in selected], seed=args.seed,
                    time_limit_s=args.time_limit_s, checkpoints_s=args.checkpoints_s,
                    objective=objective.to_dict(), validation_only=args.validate_only,
                    objective_atol_usd=args.objective_atol_usd,
                    timing_scope='continuous_ALNS_solve_including_postprocess_and_incumbent_replay_excluding_input_audit_and_constructor')
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    tasks = []
    for row in selected:
        task = dict(by_id[row['instance_id']])
        ref = task['stage2_task']
        if not (row['family_id'] == ref['family_id'] and row['city'] == ref['city_slug']
                and row['track_id'] == ref['track_id'] == 'test1_new_seed'
                and ref['split_id'] == 'test' and ref['customer_count'] == 1000
                and ref['charging_station_count'] == 50 and row['test_scale'] == 1000
                and row['split'] == 'T1' and row['objective_unit'] == 'USD'):
            raise ValueError(f'BKS/index metadata mismatch: {row["instance_id"]}')
        task.update(bks=row, objective_config=objective.to_dict(),
                    seed=stable_view_seed(args.seed, row['instance_id']),
                    output_dir=str(args.output / 'instances' / row['instance_id']),
                    time_limit_s=args.time_limit_s, checkpoints_s=args.checkpoints_s,
                    validation_only=args.validate_only, objective_atol_usd=args.objective_atol_usd,
                    contract_fingerprint=fingerprint)
        tasks.append(task)
    if args.limit is not None:
        tasks = tasks[:args.limit]
    contract['executed_instance_ids'] = [t['bks']['instance_id'] for t in tasks]
    contract['fingerprint'] = fingerprint
    return tasks, contract, index


def audit_initial(task):
    from benchmark_common import load_input_task, validate_routes
    from evrptw_core.objective import ObjectiveConfig
    from evrptw_core.validation import validate_instance_structure
    from instance_adapter import to_alns_tensor_instance
    from solver import ALNS_Solver
    instance, _ = load_input_task(task)
    check = validate_instance_structure(instance)
    if not check.success:
        raise ValueError(f'Invalid instance: {check.errors}')
    bks = task['bks']
    routes = bks['solution']
    if not isinstance(routes, list) or not routes or any(
            not isinstance(route, list) or any(type(n) is not int for n in route) for route in routes):
        raise ValueError('BKS solution must be nonempty nested integer lists')
    objective = ObjectiveConfig(**task['objective_config'])
    audit = validate_routes(instance, routes)
    solver = ALNS_Solver(to_alns_tensor_instance(instance), seed=task['seed'], format='tensor',
                         distance_unit_cost=objective.distance_unit_cost,
                         vehicle_fixed_cost=objective.vehicle_unit_cost)
    feasible = solver.is_solution_feasible(routes)
    cost = solver.objective_value(routes)
    fields = objective.fields(audit['objective_distance_km'], len(routes))
    supplied = float(bks['objective'])
    errors = [abs(cost-supplied), abs(fields['objective_value']-supplied), abs(cost-fields['objective_value'])]
    passed = (audit['passed'] and feasible and len(routes) == bks['vehicle_count']
              and math.isfinite(supplied) and all(math.isfinite(e) and e <= task['objective_atol_usd'] for e in errors))
    report = dict(instance_id=instance.instance_id, passed=bool(passed),
                  canonical_feasible=bool(audit['passed']), alns_feasible=bool(feasible),
                  supplied_objective_usd=supplied, alns_objective_usd=cost,
                  canonical_objective_usd=fields['objective_value'], max_absolute_error_usd=max(errors),
                  objective_atol_usd=task['objective_atol_usd'], violations=audit['violations'],
                  objective_distance_km=audit['objective_distance_km'],
                  supplied_distance_km=bks['distance_km'], vehicle_count=len(routes),
                  objective_config=objective.to_dict(), objective_distance_source='running_time_path_distance_km')
    return instance, solver, objective, fields, report


class Timeline:
    """Thread-safe verified improvements; no later event enters an earlier snapshot."""
    def __init__(self, task, fields):
        from benchmark_common import IncumbentEventRecorder
        self.task = task
        self.directory = Path(task['output_dir'])
        self.recorder = IncumbentEventRecorder(task['checkpoints_s'], task['time_limit_s'])
        self.lock = threading.RLock()
        self.recorder.observe(0, fields['objective_value'], task['bks']['solution'],
                              objective_distance_km=task['bks']['distance_km'], objective_fields=fields)
        self.written = set()

    def observe(self, elapsed, routes, distance, fields):
        with self.lock:
            self.recorder.observe(elapsed, fields['objective_value'], routes,
                                  objective_distance_km=distance, objective_fields=fields)

    def write_due(self, elapsed, final=False):
        with self.lock:
            for snapshot in self.recorder.snapshots(runtime_s=elapsed, natural_completion=False,
                    final_status='COMPLETED' if final else 'RUNNING'):
                cutoff = snapshot['checkpoint_s']
                if not snapshot['reached_checkpoint'] or cutoff in self.written:
                    continue
                value = dict(instance_id=self.task['bks']['instance_id'], **snapshot,
                             checkpoint_minutes=cutoff/60, feasible=True,
                             initial_objective_usd=self.task['bks']['objective'],
                             improvement_usd=self.task['bks']['objective']-snapshot['objective_value'],
                             contract_fingerprint=self.task['contract_fingerprint'], written_at=now())
                atomic_json(self.directory / f'best_at_{cutoff:g}s.json', value)
                self.written.add(cutoff)


def worker(task_path):
    from benchmark_common import IncumbentReplayCache, SolverTimeLimit, hard_time_limit, TIME_BUDGET_ITERATION_CEILING
    task = read(task_path)
    directory = Path(task['output_dir'])
    base = dict(instance_id=task['bks']['instance_id'], pid=os.getpid(),
                contract_fingerprint=task['contract_fingerprint'])
    started = time.perf_counter()
    stop = threading.Event()
    ticker = None
    try:
        atomic_json(directory/'status.json', dict(**base, status='validating', updated_at=now()))
        instance, solver, objective, fields, audit = audit_initial(task)
        atomic_json(directory/'initial_validation.json', audit)
        if not audit['passed']:
            raise ValueError(f'BKS fails feasibility or objective agreement: {audit}')
        if task['validation_only']:
            result = dict(**base, status='validated', initial_validation=audit, total_runtime_s=time.perf_counter()-started)
        else:
            timeline = Timeline(task, fields)
            replay = IncumbentReplayCache(instance)
            limit = task['time_limit_s']
            solver.max_iters = TIME_BUDGET_ITERATION_CEILING
            errors = []
            search_start = time.perf_counter()
            def elapsed():
                return time.perf_counter()-search_start
            def tick():
                try:
                    while not stop.wait(0.2):
                        timeline.write_due(elapsed())
                except Exception:
                    errors.append(traceback.format_exc())
            ticker = threading.Thread(target=tick, daemon=True)
            ticker.start()
            atomic_json(directory/'status.json', dict(**base, status='optimizing',
                         search_started_at=now(), time_limit_s=limit, updated_at=now()))
            def observe(_solver_elapsed, _cost, routes):
                checked = replay.validate(routes)
                if not checked['passed']:
                    raise ValueError(f'ALNS reported an infeasible incumbent: {checked["violations"]}')
                candidate_fields = objective.fields(checked['objective_distance_km'], len(routes))
                if abs(candidate_fields['objective_value']-_cost) > task['objective_atol_usd']:
                    raise ValueError('ALNS incumbent objective differs from replay')
                timeline.observe(elapsed(), routes, checked['objective_distance_km'], candidate_fields)
            try:
                with hard_time_limit(max(0.001, limit-elapsed())):
                    solver.solve(initial_routes=task['bks']['solution'], time_limit_s=max(0.001,limit-elapsed()),
                                 incumbent_callback=observe)
            except SolverTimeLimit:
                pass
            search_runtime = elapsed()
            stop.set()
            ticker.join()
            if errors:
                raise RuntimeError(errors[0])
            if search_runtime < limit:
                raise RuntimeError(f'ALNS ended early after {search_runtime:.3f}s; no full-budget result')
            timeline.write_due(search_runtime, final=True)
            best = timeline.recorder.best_event
            final_audit = replay.validate(best['routes'])
            if not final_audit['passed'] or best['objective_value'] > task['bks']['objective']+task['objective_atol_usd']:
                raise RuntimeError('Final incumbent invalid or worse than input BKS')
            result = dict(**base, status='completed', initial_validation=audit,
                          search_runtime_s=search_runtime, time_limit_s=limit,
                          total_runtime_s=time.perf_counter()-started, iterations=solver.cur_iter,
                          objective=best['objective_value'], solution=best['routes'], feasible=True,
                          initial_objective_usd=task['bks']['objective'],
                          improvement_usd=task['bks']['objective']-best['objective_value'],
                          checkpoint_count=len(timeline.written), seed=task['seed'])
        atomic_json(directory/'result.json', result)
        atomic_json(directory/'status.json', dict(**base, status=result['status'], updated_at=now()))
        return 0
    except Exception:
        stop.set()
        if ticker:
            ticker.join()
        error = traceback.format_exc()
        atomic_json(directory/'status.json', dict(**base, status='failed', error=error, updated_at=now()))
        print(error, file=sys.stderr, flush=True)
        return 1


def run(args):
    tasks, contract, index = build_tasks(args)
    if args.dry_run:
        print(json.dumps(dict(contract=contract, dataset_index=str(index), workers=args.workers,
                             output=str(args.output), instance_count=len(tasks)), indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = args.output/'run.json'
        if existing.exists():
            if not args.resume or read(existing)['contract'] != contract:
                raise ValueError('Output exists: use --resume only with the identical run contract, or a new --output')
        atomic_json(existing, dict(contract=contract, dataset_index=str(index), bks_path=str(args.bks),
                    workers=args.workers, updated_at=now(), driver_pid=os.getpid()))
        pending = []
        results = []
        failures = []
        for task in tasks:
            directory = Path(task['output_dir'])
            directory.mkdir(parents=True, exist_ok=True)
            result_path = directory/'result.json'
            if args.resume and result_path.exists():
                result = read(result_path)
                if result['contract_fingerprint'] == contract['fingerprint'] and result['status'] in ('completed','validated'):
                    results.append(result)
                    continue
            # Retry incomplete work from its BKS for the entire budget. Preserve old evidence.
            if args.resume and (directory/'task.json').exists():
                backup = directory/'previous_attempts'/str(time.time_ns())
                backup.mkdir(parents=True)
                for p in list(directory.iterdir()):
                    if p.is_file():
                        p.rename(backup/p.name)
            atomic_json(directory/'task.json',task)
            pending.append(directory/'task.json')
        active = {}
        def status(phase):
            atomic_json(args.output/'status.json', dict(status=phase, total=len(tasks),
                        completed=len(results), failed=len(failures), queued=len(pending),
                        active=[dict(pid=p.pid, instance_id=path.parent.name) for p,(path,_) in active.items()],
                        updated_at=now(), driver_pid=os.getpid()))
        try:
            while pending or active:
                while pending and len(active)<args.workers:
                    path=pending.pop(0)
                    log=(path.parent/'worker.log').open('w')
                    child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker-task',str(path)],
                                           stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                           pass_fds=(lock.fileno(),))
                    active[child]=(path,log)
                for child,(path,log) in list(active.items()):
                    code=child.poll()
                    if code is None:
                        continue
                    log.close()
                    del active[child]
                    if code==0:
                        result=read(path.parent/'result.json');results.append(result)
                    else:
                        failures.append(dict(instance_id=path.parent.name,returncode=code))
                    print(f'{len(results)+len(failures)}/{len(tasks)} {path.parent.name} exit={code}',flush=True)
                status('running')
                if active:
                    time.sleep(0.5)
        finally:
            for child,(_,log) in active.items():
                child.terminate()
            for child,(_,log) in active.items():
                child.wait();log.close()
        reports=[v['initial_validation'] for v in results]
        report=dict(instances=len(tasks),passed=len(results),failed=len(failures),failures=failures,
                    max_absolute_error_usd=max((v['max_absolute_error_usd'] for v in reports),default=None),
                    mean_initial_objective_usd=sum(v['supplied_objective_usd'] for v in reports)/len(reports) if reports else None,
                    all_initial_feasible_and_cost_matched=len(results)==len(tasks), contract=contract)
        atomic_json(args.output/'validation_report.json',report)
        with (args.output/'summary.csv').open('w') as stream:
            names=['instance_id','status','initial_objective_usd','objective','improvement_usd','search_runtime_s','iterations','checkpoint_count']
            writer=csv.DictWriter(stream,fieldnames=names,extrasaction='ignore');writer.writeheader()
            writer.writerows(sorted(results,key=lambda r:r['instance_id']))
        status('failed' if failures else 'completed')
        return 1 if failures else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker-task',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--bks',type=Path,default=DEFAULT_BKS)
    p.add_argument('--data-root',type=Path,default=Path('/data'))
    p.add_argument('--dataset-path',default=os.environ.get('EVRPTW_DATASET_ROOT'))
    p.add_argument('--family-root',default=os.environ.get('EVRPTW_FAMILY_ROOT'))
    p.add_argument('--output',type=Path)
    p.add_argument('--part',choices=['upper','lower','all'],default='all')
    p.add_argument('--workers',type=int,default=30)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--time-limit-s',type=float,default=7200)
    p.add_argument('--checkpoints-s',default=','.join(map(str,DEFAULT_CHECKPOINTS)))
    p.add_argument('--objective-atol-usd',type=float,default=1e-6)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--limit',type=int,help='Short local smoke test only; default uses all selected instances')
    a=p.parse_args()
    if a.worker_task:
        return worker(a.worker_task)
    a.checkpoints_s=sorted(set(float(v) for v in a.checkpoints_s.split(',')))
    if not math.isfinite(a.time_limit_s) or a.time_limit_s<=0 or not a.checkpoints_s or any(not math.isfinite(t) or t<=0 or t>a.time_limit_s for t in a.checkpoints_s) or a.checkpoints_s[-1]!=a.time_limit_s:
        p.error('Checkpoints must be positive, finite, and end at the time limit')
    if a.workers<1 or (a.limit is not None and a.limit<1) or not math.isfinite(a.objective_atol_usd) or a.objective_atol_usd<0:
        p.error('Invalid workers, limit or tolerance')
    a.output=(a.output or Path('/data')/f'alns_bks_cus1000_T1_{a.part}_2h').resolve()
    return run(a)


if __name__=='__main__':
    raise SystemExit(main())
