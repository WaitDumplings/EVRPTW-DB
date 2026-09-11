#!/usr/bin/env python3
"""Audit frozen synthetic fields, RNG replay and independent route accounting."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import numpy as np

from EVRPTW_Benchmark.Reinforcement_Learning.common.terran_synthetic import (
    _read_index, _record, derive_seed, read_synthetic_tasks, load_synthetic_instance,
    LENGTH_KM_PER_UNIT, TIME_S_PER_UNIT, ENERGY_KWH_PER_UNIT, CANONICAL_CARGO_CM3,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.stage2_data import Stage2TaskPool
from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import load_objective
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import EVRPTWVectorEnvFast
from EVRPTW_Benchmark.Exact.Gurobi_Solver.route_validator import validate_routes
from EVRPTW_Benchmark.MetaHeuristics.benchmark_common import validate_routes as validate_meta_routes


def run(args):
    root = Path(args.corpus).resolve()
    out = Path(args.output).resolve()
    manifest = json.loads((root / 'corpus_manifest.json').read_text())
    source = root / 'provenance/instance_generator.py'
    spec = importlib.util.spec_from_file_location('frozen_synthetic_rng_audit', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = {'corpus': str(root), 'source_kind': 'terran_synthetic', 'test_data_accessed': False,
              'upstream_commit': manifest['upstream_commit'], 'distribution_mode': manifest['distribution_mode'],
              'splits': {}, 'environment_replays': []}
    objectives = Path(__file__).resolve().parents[1] / 'configs/rivian_energy_vehicle_cost_v2.json'
    objective = load_objective(objectives)
    for split in ['train', 'val']:
        index_name = split if (root / split / 'view_index.parquet').exists() else 'train_smoke'
        pool = Stage2TaskPool(root / index_name / 'view_index.parquet', scale='Cus100', split_ids=split,
                              track_ids='validation' if split == 'val' else 'train', representation='E')
        assert pool.source_kind == 'terran_synthetic' and pool._euclidean is None
        tasks = pool.tasks
        derived_seed = derive_seed(1234, split)
        module.set_seed(derived_seed)
        generator = module.Solomon_EVRPTW_Generation(str(root / 'provenance/effective_config_100c.json'))
        _, rows, _ = _read_index(tasks[0].index_path)
        selected_types = {}
        values = {'distance_max_km': [], 'battery_kwh': [], 'horizon_s': [], 'customer_service_s': [],
                  'demand_fraction': [], 'capacity_K_lower_bound': []}
        checked = min(int(args.prefix_count), len(tasks))
        for position, task in enumerate(tasks[:checked]):
            raw_regenerated = generator._generate_instances()
            row = rows[task.row_position]
            raw_hash = hashlib.sha256(pickle.dumps(raw_regenerated, protocol=5)).hexdigest()
            assert raw_hash == row['raw_sha256'], f'full-RNG raw replay differs at {task.view_id}'
            raw = _record(root, row, 'raw')
            x = pool.instance(task)
            xy = np.vstack([raw['depot_loc'], raw['cus_loc'], raw['rs_loc']]) * 100.0
            native_distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
            np.testing.assert_allclose(x.distance_matrix_km, native_distance * LENGTH_KM_PER_UNIT, rtol=1e-7, atol=1e-5)
            np.testing.assert_allclose(x.raw_travel_time_matrix_s, native_distance / raw['velocity_base'] * TIME_S_PER_UNIT, rtol=1e-7, atol=1e-3)
            np.testing.assert_allclose(x.energy_matrix_kwh, native_distance / raw['velocity_base'] * raw['energy_consumption'] * ENERGY_KWH_PER_UNIT, rtol=1e-7, atol=1e-5)
            np.testing.assert_allclose(x.demands_cm3 / CANONICAL_CARGO_CM3, np.asarray(raw['demand'])[21:], rtol=2e-7, atol=1e-8)
            np.testing.assert_allclose(x.tw_s, np.asarray(raw['time_window'])[21:] * raw['max_time'] * TIME_S_PER_UNIT, rtol=1e-7, atol=1e-2)
            np.testing.assert_allclose(x.service_time_s, raw['service_time'] * raw['max_time'] * TIME_S_PER_UNIT, rtol=1e-7, atol=1e-4)
            native_to_canonical = x.metadata['upstream_to_canonical_node']
            canonical_to_native = x.metadata['canonical_to_upstream_node']
            np.testing.assert_array_equal(native_to_canonical[canonical_to_native], np.arange(121))
            values['distance_max_km'].append(float(np.max(x.distance_matrix_km)))
            values['battery_kwh'].append(float(x.vehicle['battery_capacity_kwh']))
            values['horizon_s'].append(x.working_end_s)
            values['customer_service_s'].extend(float(v) for v in x.service_time_s)
            values['demand_fraction'].extend(float(v) for v in x.demands_cm3 / CANONICAL_CARGO_CM3)
            values['capacity_K_lower_bound'].append(int(np.ceil(np.sum(x.demands_cm3.astype(float)) / CANONICAL_CARGO_CM3)))
            if split == 'train' and selected_types.get(x.metadata['upstream_type'], 0) < int(args.replays_per_type):
                selected_types[x.metadata['upstream_type']] = selected_types.get(x.metadata['upstream_type'], 0) + 1
                routes = x.raw['synthetic_singleton_witness_routes']
                verifier = validate_routes(x, routes)
                meta_verifier = validate_meta_routes(x, routes)
                assert verifier['passed'] and meta_verifier['passed']
                env = EVRPTWVectorEnvFast(instance=x, n_traj=1, objective_config=objective, info_level='full', use_jit_mask=True)
                obs, _ = env.reset(seed=1234)
                for route in routes:
                    for node in route[1:]:
                        assert obs['action_mask'][0, node], (task.view_id, 'witness not executable under canonical action mask', route, node)
                        obs, _, terminated, truncated, _ = env.step([node])
                        assert not truncated[0], (task.view_id, env.failure_reason[0])
                assert terminated[0]
                D = float(env.objective_distance_km[0]); K = int(env.vehicles_started[0]); C = float(objective.value(D, K))
                np.testing.assert_allclose(D, verifier['objective_distance_km'], rtol=0, atol=1e-8)
                np.testing.assert_allclose(D, meta_verifier['objective_distance_km'], rtol=0, atol=1e-8)
                assert K == len(routes) == 100
                report['environment_replays'].append({'instance_id':task.view_id,'type':x.metadata['upstream_type'],
                                                      'environment_and_both_verifiers_agree':True,'customers':100,
                                                      'routes':K,'distance_km':D,'cost_usd':C,'route_steps':int(env.step_count)})
        report['splits'][split] = {'available_instances':len(tasks),'prefix_checked':checked,'derived_seed':derived_seed,
                                  'full_rng_raw_hash_replay_passed':True,'physical_units_roundtrip_passed':True,
                                  'canonical_node_reordering_passed':True,
                                  'ranges':{name:{'min':min(v),'median':float(np.median(v)),'max':max(v)} for name,v in values.items()}}
    report['passed'] = True
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--prefix-count',type=int,default=128)
    parser.add_argument('--replays-per-type',type=int,default=3)
    run(parser.parse_args())


if __name__=='__main__':
    main()
