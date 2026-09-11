"""Synthetic XY must pass unchanged through both public training pools."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import pickle

import numpy as np
import pytest

from EVRPTW_Benchmark.Reinforcement_Learning.common import stage2_data
from EVRPTW_Benchmark.Reinforcement_Learning.common.terran_synthetic import convert_instance, _write_index
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_protocol import make_validation_pool
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.data_pool import Stage2TERRANPool


@pytest.fixture
def synthetic_corpus(tmp_path):
    raw = {
        'depot_loc': np.array([[0., 0.]]),
        'rs_loc': np.column_stack([np.linspace(.01, .2, 20), np.zeros(20)]),
        'cus_loc': np.column_stack([np.linspace(.001, .1, 100), np.full(100, .02)]),
        'time_window': np.tile([0., 1.], (121, 1)),
        'demand': np.r_[np.zeros(21), np.full(100, .01)],
        'service_time': .01, 'max_time': 1000., 'velocity_base': 1.,
        'energy_consumption': 1., 'charging_rate': 2.,
        'battery_capacity': 100., 'demand_capacity': 200., 'types': 'R2',
    }
    result = {}
    for split in ('train', 'val'):
        value = convert_instance(raw, instance_id=f'synthetic-{split}', split=split,
                                 generator_seed=1, sequence_position=0,
                                 time_window_probability_actual=3,
                                 config={'general': {'pos_scale': 100}})
        path = tmp_path / f'{split}.pkl'
        blob = pickle.dumps(value, protocol=5)
        path.write_bytes(blob)
        row = {'view_id': value['instance_id'], 'family_id': value['instance_id'],
               'split_id': split, 'track_id': 'validation' if split == 'val' else 'train',
               'view_seed': 1, 'canonical_relative_path': path.name,
               'canonical_offset': 0, 'canonical_length': len(blob),
               'canonical_sha256': hashlib.sha256(blob).hexdigest()}
        _write_index(tmp_path, split, [row], complete=True)
        result[split] = (tmp_path / split / 'view_index.parquet', value)
    return result


def test_synthetic_xy_bypasses_road_and_haversine(synthetic_corpus, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Synthetic XY must never enter a Road loader or Haversine converter')
    monkeypatch.setattr(stage2_data, 'load_stage2_instance', forbidden)
    monkeypatch.setattr(stage2_data, 'euclidean_instance', forbidden)
    path, expected = synthetic_corpus['train']
    pool = stage2_data.Stage2TaskPool(path, scale='Cus100', split_ids='train', representation='E')
    instance = next(pool.first())
    assert pool.source_kind == 'terran_synthetic'
    np.testing.assert_array_equal(instance.distance_matrix_km, expected['distance_matrix_km'])
    np.testing.assert_array_equal(instance.energy_matrix_kwh, expected['energy_matrix_kwh'])
    np.testing.assert_array_equal(instance.raw_travel_time_matrix_s, expected['raw_travel_time_matrix_s'])
    assert instance.distance_matrix_km[0, 1] == pytest.approx(np.linalg.norm(expected['customers'][0]))
    terran = Stage2TERRANPool(path, scale='Cus100', representation='E')
    assert terran.sample().instance_id == instance.instance_id


def test_synthetic_validation_uses_its_independent_split(synthetic_corpus):
    path, expected = synthetic_corpus['val']
    args = SimpleNamespace(validation_dataset_path=path, validation_family_root=None,
                           training_representation='E', euclidean_manifest=None)
    pool = make_validation_pool(args, scale='Cus100', seed=1234)
    assert len(pool) == 1
    assert next(pool.first()).instance_id == expected['instance_id']


@pytest.mark.parametrize('representation,calibration', [('G', None), ('E', 'old_road_calibration.json')])
def test_synthetic_rejects_wrong_representation(synthetic_corpus, representation, calibration):
    with pytest.raises(ValueError):
        stage2_data.Stage2TaskPool(synthetic_corpus['train'][0], scale='Cus100',
                                 representation=representation, euclidean_manifest=calibration)
