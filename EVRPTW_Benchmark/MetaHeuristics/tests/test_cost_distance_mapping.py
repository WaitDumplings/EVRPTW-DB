from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
for path in (REPO / 'EVRPTW_Core', REPO / 'EVRPTW_Dataset_Generator' / 'src', REPO / 'EVRPTW_Benchmark' / 'MetaHeuristics'):
    sys.path.insert(0, str(path))

from evrptw_core.objective import ObjectiveConfig, load_objective, select_objective_distance
from evrptw_core.schema import EVRPTWInstance
import benchmark_common


def cost_profile():
    return load_objective(REPO / 'EVRPTW_Benchmark/Reinforcement_Learning/configs/rivian_energy_vehicle_cost_v2.json')


def road_instance():
    shortest = np.asarray([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float32)
    fastest = np.asarray([[0, 3, 4], [3, 0, 5], [4, 5, 0]], dtype=np.float64)
    return EVRPTWInstance.from_dict({
        'instance_id': 'different-paths', 'working_start_s': 0, 'working_end_s': 10000,
        'depot': [0, 0], 'customers': [[1, 0], [2, 0]], 'charging_stations': [],
        'distance_matrix_km': shortest, 'running_time_path_distance_km': fastest,
        'demands_cm3': [1, 1], 'package_counts': [1, 1], 'service_time_s': [0, 0],
        'tw_s': [[0, 10000], [0, 10000]], 'cs_time_to_depot_s': [],
        'vehicle': {'battery_capacity_kwh': 100, 'cargo_capacity_cm3': 100},
        'running_time_shortest_matrix_s': fastest * 60,
        'running_time_path_energy_kwh': fastest * (100 / 257),
        'charging_power_kw': [],
        'metadata': {'metric_contract': {'objective': 'distance_matrix_km', 'travel_time': 'running_time_shortest_matrix_s'}},
    })


def test_cost_uses_fastest_path_for_solver_and_independent_replay():
    original = road_instance()
    objective = cost_profile()
    selected = select_objective_distance(original, objective)
    replay = benchmark_common.validate_routes(selected, [[0, 1, 2, 0]])
    assert replay['passed']
    assert replay['objective_distance_km'] == 12
    assert replay['objective_distance_source'] == 'running_time_path_distance_km'
    assert objective.fields(replay['objective_distance_km'], 1)['objective_cost_usd'] == pytest.approx(
        objective.vehicle_unit_cost + objective.distance_unit_cost * 12
    )
    assert benchmark_common.validate_routes(original, [[0, 1, 2, 0]])['objective_distance_km'] == 3
    assert selected.distance_matrix_km.dtype == np.float64
    assert selected.raw['distance_matrix_km'] is selected.distance_matrix_km
    assert selected.raw['running_time_path_energy_kwh'] is original.raw['running_time_path_energy_kwh']
    assert selected.raw['running_time_shortest_matrix_s'] is original.raw['running_time_shortest_matrix_s']
    assert selected.metadata['metric_contract']['objective'] == 'running_time_path_distance_km'
    assert original.metadata['metric_contract']['objective'] == 'distance_matrix_km'
    assert 'shortest_distance_matrix_km' not in original.raw


def test_cost_mapping_is_idempotent_and_explicit_distance_recovers_original():
    original = road_instance()
    selected = select_objective_distance(original, cost_profile())
    repeated = select_objective_distance(selected, cost_profile())
    restored = select_objective_distance(repeated, ObjectiveConfig())
    np.testing.assert_array_equal(repeated.distance_matrix_km, original.raw['running_time_path_distance_km'])
    np.testing.assert_array_equal(restored.distance_matrix_km, original.distance_matrix_km)
    assert restored.metadata['objective_distance_source'] == 'distance_matrix_km'
    assert benchmark_common.validate_routes(restored, [[0, 1, 2, 0]])['objective_distance_km'] == 3


def test_cost_missing_fastest_path_fails_instead_of_pricing_shortest_path():
    instance = road_instance()
    instance.raw.pop('running_time_path_distance_km')
    with pytest.raises(ValueError, match='refusing a shortest-distance fallback'):
        select_objective_distance(instance, cost_profile())
    assert select_objective_distance(instance, ObjectiveConfig()).distance_matrix_km[0, 1] == 1


@pytest.mark.parametrize('bad', [np.ones((2, 2)), np.full((3, 3), np.nan), np.full((3, 3), -1), np.full((3, 3), 1j)])
def test_cost_mapping_rejects_invalid_fastest_distance_matrix(bad):
    instance = road_instance()
    instance.raw['running_time_path_distance_km'] = bad
    with pytest.raises(ValueError):
        select_objective_distance(instance, cost_profile())


def task_config():
    ref = benchmark_common.Stage2ViewTask(
        index_path='/fixture/generation_plan/core/test/view_index.parquet', family_dir='/fixture/materialized/families/family',
        view_id='different-paths', family_id='family', consumer_cohort_id='cohort', split_id='test',
        track_id='test1', city_slug='city', scale_id='Cus2', customer_count=2, charging_station_count=0,
        row_position=0, terminal_count=3,
    )
    return {'input_kind': 'stage2', 'stage2_task': ref.to_dict(), 'objective_config': cost_profile().to_dict(),
            'time_limit_s': 60, 'checkpoints_s': [60], 'seed_scheme': 'test', 'seed': 1}


def test_shared_loader_maps_cost_before_adapter_and_replay_cache(monkeypatch):
    monkeypatch.setattr(benchmark_common, 'load_stage2_instance', lambda ref: road_instance())
    instance, info = benchmark_common.load_input_task(task_config())
    audit = benchmark_common.IncumbentReplayCache(instance).validate([[0, 1, 2, 0]])
    assert info['scale_id'] == 'Cus2'
    assert audit['objective_distance_km'] == 12
    assert audit['objective_distance_source'] == 'running_time_path_distance_km'


def test_run_contract_distinguishes_cost_path_from_legacy_path():
    task = task_config()
    kwargs = dict(algorithm_name='fixture', algorithm_profile_id='fixture', base_seed=1, solver_parameters={})
    new_fingerprint, new_json = benchmark_common.build_run_contract(task, **kwargs)
    legacy = {**task, 'objective_config': ObjectiveConfig().to_dict()}
    old_fingerprint, old_json = benchmark_common.build_run_contract(legacy, **kwargs)
    assert new_fingerprint != old_fingerprint
    assert json.loads(new_json)['schema'] == 'evrptw_meta_run_contract_v4'
    assert json.loads(new_json)['objective_distance_source'] == 'running_time_path_distance_km'
    assert json.loads(old_json)['objective_distance_source'] == 'distance_matrix_km'
