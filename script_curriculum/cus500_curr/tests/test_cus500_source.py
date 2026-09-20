"""CPU-only checks for portable, user-designated stage-1 source checkpoints."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from script_curriculum.cus500_curr import checkpoint_source as source


@pytest.fixture(scope='module')
def valid_payloads():
    objective = json.loads((source.REPO / 'EVRPTW_Benchmark/Reinforcement_Learning/scripts/'
                             'ablation_final/configs/objective_dtime.json').read_text())['objective']
    result = {}
    signature = dict(protocol_id=source.SOURCE_PROTOCOL, scale='Cus100',
                     training_representation='G', seed=1234)
    for method in source.METHOD_FILES:
        state = source._native_policy(method).state_dict()
        architecture = deepcopy(source.ARCHITECTURES[method])
        if method == 'terran':
            payload = dict(epoch=350, seed=1234, model_state_dict=state,
                           config=dict(model=architecture, objective=deepcopy(objective), training={},
                                       data=dict(stage2_scale='Cus100', stage2_training_representation='G',
                                                 num_customers=100, num_charging_stations=20),
                                       protocol=dict(protocol_id=source.SOURCE_PROTOCOL,
                                                     resolved_training_signature=deepcopy(signature))))
        else:
            payload = dict(method=source.CHECKPOINT_METHODS[method], logical_epoch=350,
                           model=state, objective_config=deepcopy(objective),
                           protocol_id=source.SOURCE_PROTOCOL, resolved_training_signature=deepcopy(signature),
                           args=dict(scale='Cus100', training_representation='G', seed=1234, **architecture))
            if method == 'drl_ts':
                payload['soft_stage_contract'] = dict(resolved_soft_stage_end_epoch=0)
        result[method] = payload
    return result


@pytest.mark.parametrize('method', source.METHOD_FILES)
def test_all_five_standardized_sources_validate_without_frozen_hash_or_endpoint(tmp_path, valid_payloads, method):
    path = tmp_path / source.METHOD_FILES[method]
    payload = deepcopy(valid_payloads[method])
    torch.save(payload, path)
    rng = torch.random.get_rng_state().clone()
    actual, record = source.load_source(method, tmp_path)
    assert actual == path
    assert record['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record['logical_epoch'] == 350
    assert record['source_training_stage'] == 'hard'
    assert record['weight_compatibility']['strict_native_policy_load'] is True
    assert record['weight_compatibility']['device'] == 'cpu'
    assert record['source_domain'] == 'G' and record['source_scale'] == 'Cus100'
    assert 'source_run_completed_training_epochs' not in record
    assert 'validation' not in record
    assert torch.equal(rng, torch.random.get_rng_state())
    # A different, legitimately selected remote checkpoint remains usable; no
    # requirement that its selected epoch or terminal budget equal this host's.
    payload['epoch' if method == 'terran' else 'logical_epoch'] = 2700
    torch.save(payload, path)
    _, updated = source.load_source(method, tmp_path)
    assert updated['logical_epoch'] == 2700 and updated['sha256'] != record['sha256']


def test_explicit_override_and_missing_paths_never_fall_back(tmp_path, valid_payloads):
    override = tmp_path / 'selected_elsewhere.ckpt'
    torch.save(valid_payloads['am_evrptw'], override)
    actual, record = source.load_source('am', tmp_path / 'missing-root', override)
    assert actual == override and record['source_path_override'] is True
    with pytest.raises(FileNotFoundError):
        source.load_source('am_evrptw', tmp_path, tmp_path / 'missing.ckpt')
    with pytest.raises(FileNotFoundError):
        source.load_source('am_evrptw', tmp_path)
    with pytest.raises(ValueError, match='Unsupported'):
        source.load_source('unknown', tmp_path)


@pytest.mark.parametrize(('field', 'value', 'error'), [
    ('scale', 'Cus500', 'scale mismatch'),
    ('training_representation', 'E', 'training_representation mismatch'),
    ('seed', 1, 'seed mismatch'),
    ('n_heads', 4, 'architecture.n_heads mismatch'),
])
def test_semantic_source_checks_reject_compatible_tensor_shapes(tmp_path, valid_payloads, field, value, error):
    payload = deepcopy(valid_payloads['am_evrptw'])
    payload['args'][field] = value
    path = tmp_path / 'am.ckpt'
    torch.save(payload, path)
    with pytest.raises(ValueError, match=error):
        source.load_source('am_evrptw', tmp_path)


@pytest.mark.parametrize(('field', 'value', 'error'), [
    ('method', 'EVRPTW-RL', 'method mismatch'),
    ('protocol_id', 'historical_archive', 'protocol mismatch'),
    ('logical_epoch', 0, 'positive selected logical epoch'),
    ('logical_epoch', None, 'positive selected logical epoch'),
])
def test_old_wrong_or_unselected_sources_rejected(tmp_path, valid_payloads, field, value, error):
    payload = deepcopy(valid_payloads['am_evrptw'])
    payload[field] = value
    torch.save(payload, tmp_path / 'am.ckpt')
    with pytest.raises(ValueError, match=error):
        source.load_source('am_evrptw', tmp_path)


def test_objective_and_redundant_signature_disagreement_rejected(tmp_path, valid_payloads):
    payload = deepcopy(valid_payloads['am_evrptw'])
    payload['objective_config']['objective_distance_source'] = 'distance_matrix_km'
    torch.save(payload, tmp_path / 'am.ckpt')
    with pytest.raises(ValueError, match='objective.objective_distance_source'):
        source.load_source('am_evrptw', tmp_path)
    payload = deepcopy(valid_payloads['am_evrptw'])
    payload['resolved_training_signature']['training_representation'] = 'E'
    torch.save(payload, tmp_path / 'am.ckpt')
    with pytest.raises(ValueError, match='signature.training_representation'):
        source.load_source('am_evrptw', tmp_path)


@pytest.mark.parametrize(('method', 'field', 'value'), [
    ('evrptw_rl', 'graph_aggregation', 'sum'),
    ('rrnco', 'graph_mode', 'node_only'),
    ('rrnco', 'aft_mode', 'legacy'),
])
def test_method_variants_must_match_curriculum_actor(tmp_path, valid_payloads, method, field, value):
    payload = deepcopy(valid_payloads[method])
    payload['args'][field] = value
    torch.save(payload, tmp_path / source.METHOD_FILES[method])
    with pytest.raises(ValueError, match=f'architecture.{field}'):
        source.load_source(method, tmp_path)


@pytest.mark.parametrize('bad_boundary', [None, 400, -1])
def test_drl_must_record_and_reach_hard_stage(tmp_path, valid_payloads, bad_boundary):
    payload = deepcopy(valid_payloads['drl_ts'])
    payload['soft_stage_contract']['resolved_soft_stage_end_epoch'] = bad_boundary
    torch.save(payload, tmp_path / 'drl_ts.ckpt')
    with pytest.raises(ValueError, match='Source DRL-TS'):
        source.load_source('drl_ts', tmp_path)


@pytest.mark.parametrize(('section', 'field', 'value', 'error'), [
    ('training', 'algorithm', 'stable_cost_v1', 'legacy PPO'),
    ('model', 'use_dynamic_embedding', True, 'architecture.use_dynamic_embedding'),
    ('data', 'num_charging_stations', 50, 'data.num_charging_stations'),
    ('data', 'stage2_training_representation', 'E', 'data.stage2_training_representation'),
])
def test_terran_legacy_actor_and_road_cus100_required(tmp_path, valid_payloads, section, field, value, error):
    payload = deepcopy(valid_payloads['terran'])
    payload['config'][section][field] = value
    torch.save(payload, tmp_path / 'terran.ckpt')
    with pytest.raises(ValueError, match=error):
        source.load_source('terran', tmp_path)


def test_missing_or_invalid_model_tensor_rejected(tmp_path, valid_payloads):
    payload = deepcopy(valid_payloads['am_evrptw'])
    first = next(iter(payload['model']))
    payload['model'].pop(first)
    torch.save(payload, tmp_path / 'am.ckpt')
    with pytest.raises(ValueError, match='policy weights are incompatible'):
        source.load_source('am_evrptw', tmp_path)
    payload = deepcopy(valid_payloads['am_evrptw'])
    payload['model'][first].fill_(float('nan'))
    torch.save(payload, tmp_path / 'am.ckpt')
    with pytest.raises(ValueError, match='nonfinite'):
        source.load_source('am_evrptw', tmp_path)


def test_record_only_metadata_actually_present(tmp_path, valid_payloads):
    payload = deepcopy(valid_payloads['am_evrptw'])
    payload['warm_start_provenance'] = dict(checkpoint='historical.ckpt', source_logical_epoch=5500)
    payload['best_validation_key'] = [1.0, -468.9]
    torch.save(payload, tmp_path / 'am.ckpt')
    _, record = source.load_source('am_evrptw', tmp_path)
    assert record['parent_warm_start_provenance'] == payload['warm_start_provenance']
    assert record['checkpoint_best_validation_key'] == payload['best_validation_key']
    assert 'validation' not in record
    assert 'source_run_completed_training_epochs' not in record
