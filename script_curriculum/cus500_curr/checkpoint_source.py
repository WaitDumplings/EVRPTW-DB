"""Validate user-supplied Road Cus100 stage-1 checkpoints on CPU.

Only trusted local PyTorch checkpoint files should be supplied: loading their
saved configuration uses torch.load(weights_only=False). File names identify the
requested method; actual content, rather than an old server path or fixed hash,
establishes compatibility. Source selection remains the user's responsibility.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import torch

from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import objective_from_checkpoint

REPO = Path(__file__).resolve().parents[2]
SOURCE_PROTOCOL = 'curriculum_stage1_cus100_dtime_v1'
METHOD_FILES = {
    'am_evrptw': 'am.ckpt', 'evrptw_rl': 'evrptw_rl.ckpt',
    'drl_ts': 'drl_ts.ckpt', 'terran': 'terran.ckpt', 'rrnco': 'rrnco.ckpt',
}
CHECKPOINT_METHODS = {
    'am_evrptw': 'AM-EVRPTW', 'evrptw_rl': 'EVRPTW-RL',
    'drl_ts': 'DRL-TS', 'terran': 'TERRAN', 'rrnco': 'RRNCO-EV',
}
ARCHITECTURES = {
    'am_evrptw': dict(embedding_dim=128, n_encode_layers=3, n_heads=8, tanh_clipping=10.0),
    'evrptw_rl': dict(embedding_dim=128, structure2vec_rounds=3, graph_aggregation='mean'),
    'drl_ts': dict(embedding_dim=128, n_encode_layers=2, n_heads=8,
                   nearest_neighbors=10, tanh_clipping=10.0),
    'terran': dict(embedding_dim=256, n_encode_layers=3, tanh_clipping=15.0,
                   use_graph_token=False, use_dynamic_embedding=False),
    'rrnco': dict(embedding_dim=128, n_encode_layers=6, n_heads=8, feedforward_hidden=512,
                  distance_sample_size=25, tanh_clipping=10.0, graph_mode='full',
                  aft_mode='stable', distance_sampling='nearest', relation_temperature=5.0),
}
_POLICY_CLASSES = {
    'am_evrptw': ('AM_EVRPTW.model', 'AMEVRPTWPolicy'),
    'evrptw_rl': ('EVRPTW_RL.model', 'EVRPTWRLPolicy'),
    'drl_ts': ('DRL_TS.model', 'DRLTSPolicy'),
    'terran': ('TERRAN.models', 'Agent'),
    'rrnco': ('RRNCO_EVRPTW.model', 'RRNCOEVPolicy'),
}


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f'Source {name} must be a recorded mapping')
    return dict(value)


def _expect(saved: Mapping[str, Any], required: Mapping[str, Any], prefix: str) -> None:
    for field, expected in required.items():
        if saved.get(field) != expected:
            raise ValueError(f'Source {prefix}{field} mismatch: {saved.get(field)!r} != {expected!r}')


def _native_policy(method: str) -> torch.nn.Module:
    module, name = _POLICY_CLASSES[method]
    cls = getattr(importlib.import_module(f'EVRPTW_Benchmark.Reinforcement_Learning.{module}'), name)
    kwargs = dict(ARCHITECTURES[method])
    if method == 'terran':
        kwargs['device'] = 'cpu'
    # Preflight must not consume the training RNG stream or initialize CUDA.
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        return cls(**kwargs)


def _validate_weights(method: str, state: Any) -> dict[str, Any]:
    state = _mapping(state, 'policy weights')
    policy = _native_policy(method)
    try:
        policy.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f'Source {CHECKPOINT_METHODS[method]} policy weights are incompatible: {exc}') from exc
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f'Source model tensor {name} is not a tensor')
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f'Source model tensor {name} contains nonfinite values')
    return dict(strict_native_policy_load=True, device='cpu',
                state_tensor_count=len(state), parameter_count=sum(p.numel() for p in policy.parameters()))


def load_source(method: str, checkpoint_root: Path, source_checkpoint: Path | None = None):
    """Return ``(resolved_path, metadata)`` for a compatible stage-1 source.

    Accepts internal method names, with ``am`` also accepted as an alias. The
    source's selected epoch need not equal its planned or terminal budget. This
    function never reads adjacent training histories to guess an endpoint or a
    validation score, and does not require a previously frozen checkpoint hash.
    """
    method = 'am_evrptw' if method == 'am' else method
    if method not in METHOD_FILES:
        raise ValueError(f'Unsupported curriculum method: {method!r}')
    path = (Path(source_checkpoint) if source_checkpoint is not None
            else Path(checkpoint_root) / METHOD_FILES[method]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Stage-1 Road Cus100 checkpoint unavailable: {path}; '
                                'set --checkpoint-root or --source-checkpoint')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        stream.seek(0)
        payload = _mapping(torch.load(stream, map_location='cpu', weights_only=False), 'checkpoint')

    if method == 'terran':
        if payload.get('method') not in (None, 'TERRAN'):
            raise ValueError('Source method mismatch: expected TERRAN')
        cfg = _mapping(payload.get('config'), 'TERRAN config')
        architecture = _mapping(cfg.get('model'), 'TERRAN model config')
        data = _mapping(cfg.get('data'), 'TERRAN data config')
        protocol = _mapping(cfg.get('protocol'), 'TERRAN protocol')
        _expect(data, dict(stage2_scale='Cus100', stage2_training_representation='G',
                           num_customers=100, num_charging_stations=20), 'data.')
        _expect(payload, {'seed': 1234}, '')
        training = _mapping(cfg.get('training'), 'TERRAN training config')
        if training.get('algorithm') not in (None, 'legacy_ppo'):
            raise ValueError('Source TERRAN algorithm must use the legacy PPO actor')
        source_protocol = protocol.get('protocol_id')
        epoch_value = payload.get('epoch')
        source_signature = protocol.get('resolved_training_signature')
        state = payload.get('model_state_dict')
        parent = protocol.get('warm_start')
        stage = 'hard'
    else:
        _expect(payload, {'method': CHECKPOINT_METHODS[method]}, '')
        saved = payload.get('args')
        if not isinstance(saved, Mapping) and hasattr(saved, '__dict__'):
            saved = vars(saved)
        saved = _mapping(saved, 'args')
        _expect(saved, dict(scale='Cus100', training_representation='G', seed=1234), '')
        architecture = saved
        source_protocol = payload.get('protocol_id')
        epoch_value = payload.get('logical_epoch')
        source_signature = payload.get('resolved_training_signature')
        state = payload.get('model')
        parent = payload.get('warm_start_provenance')
        stage = 'hard'
        if method == 'drl_ts':
            contract = _mapping(payload.get('soft_stage_contract'), 'DRL-TS soft-stage contract')
            boundary = contract.get('resolved_soft_stage_end_epoch')
            if not isinstance(boundary, int) or isinstance(boundary, bool) or boundary < 0:
                raise ValueError('Source DRL-TS must record an absolute soft-stage epoch boundary')
            if not isinstance(epoch_value, int) or epoch_value <= boundary:
                raise ValueError('Source DRL-TS checkpoint must have reached the hard stage')
    if source_protocol != SOURCE_PROTOCOL:
        raise ValueError(f'Source protocol mismatch: {source_protocol!r} != {SOURCE_PROTOCOL!r}')
    if not isinstance(epoch_value, int) or isinstance(epoch_value, bool) or epoch_value < 1:
        raise ValueError('Source checkpoint must record a positive selected logical epoch')
    if source_signature is not None:
        _expect(_mapping(source_signature, 'training signature'),
                dict(protocol_id=SOURCE_PROTOCOL, scale='Cus100', training_representation='G', seed=1234),
                'signature.')
    _expect(architecture, ARCHITECTURES[method], 'architecture.')
    objective = objective_from_checkpoint(payload).to_dict()
    expected_objective = json.loads((REPO / 'EVRPTW_Benchmark/Reinforcement_Learning/scripts/'
                                     'ablation_final/configs/objective_dtime.json').read_text())['objective']
    _expect(objective, expected_objective, 'objective.')
    weight_check = _validate_weights(method, state)
    record = dict(method=method, checkpoint_method=CHECKPOINT_METHODS[method],
                  source_domain='G', source_scale='Cus100', source_seed=1234,
                  source_protocol_id=source_protocol, checkpoint=str(path), sha256=digest.hexdigest(),
                  logical_epoch=epoch_value, source_training_stage=stage,
                  architecture=dict(ARCHITECTURES[method]), objective_config=objective,
                  source_selection='user_supplied_stage1_checkpoint',
                  source_path_override=source_checkpoint is not None,
                  weight_compatibility=weight_check)
    if isinstance(parent, Mapping):
        record['parent_warm_start_provenance'] = dict(parent)
    if payload.get('best_validation_key') is not None:
        record['checkpoint_best_validation_key'] = payload['best_validation_key']
    if method == 'drl_ts':
        record['source_soft_stage_contract'] = dict(payload['soft_stage_contract'])
    return path, record
