#!/usr/bin/env python3
"""Training-only synthetic reward normalization; deterministic construction, no search."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time

from .launch import REPO, RUN_ROOT, sha256, timestamp, write_json

NAMESPACE = 'cus100_terran_synthetic_reward_calibration_v1'
DEFAULT_ROOT = REPO / 'EVRPTW_Dataset/TERRAN_synthetic100_feasible4_20260911'
OBJECTIVE_PATH = REPO / 'EVRPTW_Benchmark/Reinforcement_Learning/configs/rivian_energy_vehicle_cost_v2.json'


def _worker(item):
    from EVRPTW_Benchmark.Reinforcement_Learning.scripts.calibrate_training_reference_cost import (
        ALNS_Solver, to_alns_tensor_instance, validate_meta_routes, validate_objective_routes,
        resolve_objective, route_dispatch_count, route_sha256,
    )
    from EVRPTW_Benchmark.Reinforcement_Learning.common.terran_synthetic import load_synthetic_instance
    from EVRPTW_Benchmark.MetaHeuristics.benchmark_common import Stage2ViewTask
    task = Stage2ViewTask.from_dict(item['task'])
    if task.split_id != 'train' or task.track_id != 'train':
        raise ValueError('Calibration may consume training instances only')
    started = time.monotonic()
    instance = load_synthetic_instance(task)
    objective = resolve_objective(item['objective'])
    witness = instance.raw['synthetic_singleton_witness_routes']
    for verifier in (validate_meta_routes, validate_objective_routes):
        audit = verifier(instance, witness)
        if not audit['passed']:
            raise ValueError(f"Invalid frozen synthetic witness for {task.view_id}: {audit['violations']}")
    adapted = to_alns_tensor_instance(instance)
    adapted['certificate_singleton_routes'] = [[int(node) for node in route] for route in witness]
    solver = ALNS_Solver(adapted, seed=0, format='tensor',
                         distance_unit_cost=objective.distance_unit_cost,
                         vehicle_fixed_cost=objective.vehicle_unit_cost)
    routes = solver.construct_deterministic_reference_solution()
    meta = validate_meta_routes(instance, routes)
    verified = validate_objective_routes(instance, routes)
    if not meta['passed'] or not verified['passed']:
        raise ValueError(f"Reference replay failed for {task.view_id}: {meta['violations']}, {verified['violations']}")
    distance = float(verified['objective_distance_km'])
    if not math.isclose(distance, float(meta['objective_distance_km']), rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError('Independent verifier distances disagree')
    vehicles = route_dispatch_count(routes)
    if vehicles != len(routes):
        raise ValueError('Reference routes have unexpected depot dispatches')
    construction = dict(solver.initial_construction_stats)
    if (construction.get('wall_clock_cutoff_enabled') or not construction.get('deterministic')
            or construction.get('termination_basis') != 'candidate_limits_only'
            or construction.get('budget_exhausted')):
        raise ValueError('Reference construction did not obey its deterministic candidate budget')
    construction['implementation_certificate_label'] = construction.pop('singleton_source', None)
    construction['singleton_source'] = 'synthetic_witness_canonical_replayed'
    return {'view_id': task.view_id, 'family_id': task.family_id, 'cohort_position': item['position'],
            'source_split': 'train', 'source_track': 'train', 'source_kind': 'terran_synthetic',
            'representation': 'E', 'scale_label': 'Cus100', 'routes': routes, 'route_sha256': route_sha256(routes),
            'objective_distance_km': distance, 'vehicles_started': vehicles, **objective.fields(distance, vehicles),
            'route_validation_passed': True, 'witness_dual_validation_passed': True,
            'constructor': construction, 'stochastic_search_iterations': 0,
            'elapsed_s': time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--count', type=int, default=500)
    args = parser.parse_args()
    if args.workers < 1 or args.count < 1:
        raise ValueError('workers and count must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite calibration results: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    from EVRPTW_Benchmark.Reinforcement_Learning.scripts.calibrate_training_reference_cost import summarize_scale_rows, load_objective
    from EVRPTW_Benchmark.Reinforcement_Learning.common.reward_contract import REWARD_CONTRACT_SCHEMA, RewardContract, reward_contract_digest
    from EVRPTW_Benchmark.Reinforcement_Learning.common.terran_synthetic import read_synthetic_tasks
    corpus_path = args.dataset_root / 'corpus_manifest.json'
    corpus = json.loads(corpus_path.read_text())
    if not args.pilot and (corpus.get('complete') is not True or corpus.get('formal_training_authorized') is not True):
        raise ValueError('Formal calibration requires a complete, authorized, frozen synthetic corpus')
    index = args.dataset_root / ('train_smoke' if args.pilot else 'train') / 'view_index.parquet'
    tasks = read_synthetic_tasks(index)
    if not args.pilot and (len(tasks) != 50000 or args.count != 500):
        raise ValueError('Formal calibration must select 500 instances from the frozen 50000-view training pool')
    if args.count > len(tasks) or any(task.split_id != 'train' or task.track_id != 'train' for task in tasks):
        raise ValueError('Invalid training-only calibration cohort')
    namespace = f'{NAMESPACE}:training_seed={args.seed}'
    rank = lambda task: hashlib.sha256(f'{namespace}:view={task.view_id}'.encode()).hexdigest()
    selected = sorted(tasks, key=lambda task: (rank(task), task.view_id))[:args.count]
    objective = load_objective(OBJECTIVE_PATH)
    cohort = {'schema': 'cus100_synthetic_reward_cohort_v1', 'selection': 'uniform_sha256_rank_without_replacement',
              'namespace': namespace, 'training_seed': args.seed, 'pool_count': len(tasks), 'selected_count': len(selected),
              'source_split': 'train', 'source_track': 'train', 'index': str(index), 'index_sha256': sha256(index),
              'view_ids': [task.view_id for task in selected], 'view_ranks': [rank(task) for task in selected],
              'pilot': args.pilot, 'validation_data_read': False, 'test_data_read': False}
    write_json(args.output / 'cohort.json', cohort)
    items = [{'task': task.to_dict(), 'position': position, 'objective': objective.to_dict()} for position, task in enumerate(selected)]
    started = time.monotonic()
    rows = []
    with (args.output / 'per_view.jsonl').open('w') as output:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for row in executor.map(_worker, items, chunksize=1):
                rows.append(row)
                output.write(json.dumps(row, sort_keys=True) + '\n')
                output.flush()
                progress = {'time': timestamp(), 'completed': len(rows), 'total': len(items),
                            'elapsed_s': time.monotonic() - started, 'last_view': row['view_id']}
                write_json(args.output / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    terms, statistics = summarize_scale_rows(rows)
    report = {'schema': 'cus100_synthetic_reward_calibration_v1', 'pilot': args.pilot, 'time': timestamp(),
              'source_kind': 'terran_synthetic', 'representation': 'E', 'source_split': 'train', 'source_track': 'train',
              'terms': terms, 'statistics': statistics, 'elapsed_s': time.monotonic() - started,
              'workers': args.workers, 'mean_reference_seconds': sum(row['elapsed_s'] for row in rows) / len(rows),
              'cohort_sha256': sha256(args.output / 'cohort.json'), 'per_view_sha256': sha256(args.output / 'per_view.jsonl'),
              'corpus_manifest_sha256': sha256(corpus_path), 'train_index_sha256': sha256(index),
              'objective_config': objective.to_dict(), 'objective_config_sha256': sha256(OBJECTIVE_PATH),
              'constructor_profile': 'alns_singleton_best_fit_cost_v2_synthetic_witness_deterministic_v1',
              'constructor_stochastic_search_iterations': 0, 'constructor_wall_clock_cutoff_enabled': False,
              'normalization': 'divide the complete monetary objective by one positive source-level median reference cost',
              'failure_base': 'q99_linear(reference_cost / scale) + 1; reference cohort calibration, not a universal feasibility guarantee',
              'validation_data_read': False, 'test_data_read': False}
    write_json(args.output / 'calibration_report.json', report)
    if not args.pilot:
        payload = {'schema': REWARD_CONTRACT_SCHEMA,
                   'contract_id': 'drl_terran_synthetic100_reference_scale_20260911_v1',
                   'objective': objective.to_dict(), 'scales': {'Cus100': terms},
                   'calibration': {**report, 'cohort': cohort}}
        payload['sha256'] = reward_contract_digest(payload)
        RewardContract.from_payload(payload)
        write_json(args.output / 'reward_contract.json', payload)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
