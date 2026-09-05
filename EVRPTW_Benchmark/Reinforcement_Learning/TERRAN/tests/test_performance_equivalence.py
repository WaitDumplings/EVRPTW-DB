from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from EVRPTW_Benchmark.Exact.Gurobi_Solver.route_validator import validate_routes
from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import _instance
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import rollout as rollout_module
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.models import Agent
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.models.attention_model_wrapper import (
    MODEL_OBSERVATION_KEYS,
    STATIC_OBSERVATION_KEYS,
    stateWrapper,
)
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.pbrs import PotentialRewardConfig
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import (
    collect_rollout,
    compute_returns,
    rollout_eval_batch,
    sample_actions,
    sample_eval_actions,
    stack_observations,
    stack_policy_observations,
)
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.trainer import (
    _slice_obs_by_env,
    evaluate_policy_loss,
    masked_mean,
)


def _agent(training=False):
    torch.manual_seed(173)
    agent = Agent(embedding_dim=32, n_encode_layers=1, device="cpu")
    agent.train(training)
    return agent


def _envs(pbrs=False):
    return [
        make_terran_env(
            instance=_instance(), n_traj=4, charging_mode="station_power_full",
            matrix_mode="canonical", info_level="full", use_jit_mask=False,
            pbrs_config=PotentialRewardConfig(
                use_customer_pbrs=True, use_terminal_heuristic=True,
            ) if pbrs else None,
        ) for _ in range(2)
    ]


def _assert_infos_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], np.ndarray):
            np.testing.assert_array_equal(left[key], right[key])
        elif isinstance(left[key], dict):
            _assert_infos_equal(left[key], right[key])
        else:
            assert left[key] == right[key]


@pytest.mark.parametrize("training", [False, True])
def test_compact_inputs_preserve_logits_and_rng(training):
    agent = _agent(training)
    observations = [env.reset(seed=18)[0] for env in _envs()]
    full = stack_observations(observations)
    compact, static = stack_policy_observations(observations)
    assert set(compact) == MODEL_OBSERVATION_KEYS - {"instance_mask"}
    assert "remaining_demand" in full and "remaining_demand" not in compact
    state = stateWrapper(full, "cpu")
    assert "remaining_demand" not in state.states
    device_inputs = {**compact, **{key: torch.as_tensor(value) for key, value in static.items()}}
    torch.manual_seed(19)
    reference = agent.backbone(full)
    reference_rng = torch.get_rng_state().clone()
    torch.manual_seed(19)
    actual = agent.backbone(device_inputs)
    assert torch.equal(torch.get_rng_state(), reference_rng)
    for expected, observed in zip(reference, actual):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_training_encoder_cache_preserves_rollout_and_storage():
    agent = _agent(training=True)
    encoder_calls = []
    hook = agent.backbone.encoder.register_forward_hook(lambda *_: encoder_calls.append(1))
    torch.manual_seed(23)
    reference = collect_rollout(
        agent,
        _envs(pbrs=True),
        12,
        "sample",
        "cpu",
        seed=24,
        compact_observations=False,
        cache_static_embeddings=False,
    )
    reference_rng = torch.get_rng_state().clone()
    reference_calls = len(encoder_calls)
    encoder_calls.clear()
    torch.manual_seed(23)
    compact = collect_rollout(agent, _envs(pbrs=True), 12, "sample", "cpu", seed=24)
    hook.remove()
    assert reference_calls == len(reference.observations)
    assert len(encoder_calls) == 1
    assert torch.equal(torch.get_rng_state(), reference_rng)
    for name in ("actions", "old_logprobs", "rewards", "dones", "values", "valid", "entropies", "trajectory_steps", "rollout_budget_exhausted"):
        torch.testing.assert_close(getattr(compact, name), getattr(reference, name), rtol=0, atol=0)
    for actual, expected in zip(compact.final_infos, reference.final_infos):
        _assert_infos_equal(actual, expected)
    assert len(compact.observations) > 1
    for key in STATIC_OBSERVATION_KEYS:
        if key in compact.observations[0]:
            assert compact.observations[0][key] is compact.observations[1][key]
    for actual, expected in zip(compact.observations, reference.observations):
        for key, value in actual.items():
            np.testing.assert_array_equal(value, expected[key])
    assert not np.shares_memory(compact.observations[0]["action_mask"], compact.observations[1]["action_mask"])


def _legacy_policy_loss(agent, batch, returns, advantages, indices, start, end):
    """Pre-optimization PPO loss, intentionally retaining full observation slices."""
    state = agent.backbone.encode(_slice_obs_by_env(batch.observations[0], indices))
    policies, values, entropies = [], [], []
    for step in range(start, end):
        obs = _slice_obs_by_env(batch.observations[step], indices)
        _, logprob, entropy, value, _ = agent.get_action_and_value_cached(
            obs, action=batch.actions[step, indices].long(), state=state,
        )
        ratio = torch.exp(logprob - batch.old_logprobs[step, indices])
        advantage = advantages[step, indices]
        valid = batch.valid[step, indices]
        policies.append(-masked_mean(torch.minimum(ratio * advantage, torch.clamp(ratio, 0.8, 1.2) * advantage), valid))
        values.append(masked_mean(F.mse_loss(value.squeeze(-1), returns[step, indices], reduction="none"), valid))
        entropies.append(masked_mean(entropy, valid))
    return torch.stack(policies).mean() + 0.5 * torch.stack(values).mean() - 0.01 * torch.stack(entropies).mean()


@pytest.mark.parametrize("step_start", [0, 1])
def test_ppo_compact_transport_preserves_loss_and_gradients(step_start):
    agent = _agent(training=True)
    batch = collect_rollout(agent, _envs(), 8, "sample", "cpu", seed=29, compact_observations=False)
    returns = compute_returns(batch.rewards, batch.dones, 0.99)
    advantages = returns - batch.values
    indices = np.asarray([1, 0])
    step_end = min(len(batch.observations), step_start + 2)
    torch.manual_seed(31)
    reference = _legacy_policy_loss(agent, batch, returns, advantages, indices, step_start, step_end)
    reference.backward()
    reference_rng = torch.get_rng_state().clone()
    reference_gradients = [None if p.grad is None else p.grad.clone() for p in agent.parameters()]
    agent.zero_grad(set_to_none=True)
    torch.manual_seed(31)
    actual, *_ = evaluate_policy_loss(agent, batch, returns, advantages, {"training": {}}, "cpu", env_indices=indices, step_start=step_start, step_end=step_end)
    actual.backward()
    assert torch.equal(torch.get_rng_state(), reference_rng)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for parameter, expected in zip(agent.parameters(), reference_gradients):
        if expected is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)


def test_ppo_logprob_ratio_is_one_before_parameter_update():
    agent = _agent(training=True)
    batch = collect_rollout(agent, _envs(), 8, "sample", "cpu", seed=33)
    indices = np.arange(batch.actions.size(1), dtype=np.int64)
    cached_state = agent.backbone.encode(
        _slice_obs_by_env(batch.observations[0], indices)
    )
    valid_old = []
    valid_new = []
    for step, obs in enumerate(batch.observations):
        _, new_logprob, _, _, _ = agent.get_action_and_value_cached(
            _slice_obs_by_env(obs, indices),
            action=batch.actions[step, indices].long(),
            state=cached_state,
        )
        valid = batch.valid[step, indices]
        valid_old.append(batch.old_logprobs[step, indices][valid])
        valid_new.append(new_logprob[valid])
    old_logprob = torch.cat(valid_old)
    new_logprob = torch.cat(valid_new)
    torch.testing.assert_close(new_logprob, old_logprob, rtol=0, atol=1e-7)
    torch.testing.assert_close(
        torch.exp(new_logprob - old_logprob),
        torch.ones_like(old_logprob),
        rtol=0,
        atol=1e-7,
    )


def test_terran_encoder_has_no_active_dropout():
    agent = _agent(training=True)
    dropouts = [module for module in agent.modules() if isinstance(module, torch.nn.Dropout)]
    assert dropouts
    assert all(module.p == 0.0 for module in dropouts)


@pytest.mark.parametrize("decode_mode", ["sample", "greedy"])
@pytest.mark.parametrize("max_steps", [1, 32])
def test_cached_eval_preserves_routes_verifier_rng_and_exports_once(decode_mode, max_steps):
    agent = _agent()
    encoder_calls, critic_calls = [], []
    encoder_hook = agent.backbone.encoder.register_forward_hook(lambda *_: encoder_calls.append(1))
    critic_hook = agent.critic.register_forward_hook(lambda *_: critic_calls.append(1))
    torch.manual_seed(37)
    reference = rollout_eval_batch(agent, _envs(), decode_mode, max_steps, "cpu", seed=38, include_routes=True, return_final_info=True, cache_static_embeddings=False, compact_observations=False, final_routes_only=False)
    reference_rng = torch.get_rng_state().clone()
    reference_calls = len(encoder_calls)
    encoder_calls.clear()
    envs = _envs()
    exports = []
    for env in envs:
        original = env.unwrapped.get_routes
        def count_export(original=original):
            exports.append(1)
            return original()
        env.unwrapped.get_routes = count_export
    torch.manual_seed(37)
    actual = rollout_eval_batch(agent, envs, decode_mode, max_steps, "cpu", seed=38, include_routes=True, return_final_info=True)
    encoder_hook.remove()
    critic_hook.remove()
    assert len(encoder_calls) == 1
    assert reference_calls >= len(encoder_calls)
    if max_steps > 1:
        assert reference_calls > 1
    assert len(exports) == len(envs)
    assert not critic_calls
    assert all(env.unwrapped.info_level == "full" for env in envs)
    assert torch.equal(torch.get_rng_state(), reference_rng)
    for expected, observed in zip(reference, actual):
        for row in (expected, observed):
            row.pop("runtime_s")
            row.pop("batch_runtime_s")
        _assert_infos_equal(observed, expected)
        expected_verification = validate_routes(_instance(), json.loads(expected["routes_json"]))
        actual_verification = validate_routes(_instance(), json.loads(observed["routes_json"]))
        assert actual_verification == expected_verification


@pytest.mark.parametrize("decode_mode", ["greedy", "sample"])
def test_action_only_matches_reference_and_cached_logits(decode_mode):
    agent = _agent()
    obs = stack_observations([env.reset()[0] for env in _envs()])
    with torch.no_grad():
        expected_logits, expected_glimpse = agent.backbone(obs)
        cache = agent.backbone.encode(obs)
        logits, glimpse = agent.backbone.decode(obs, cache)
        torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
        torch.testing.assert_close(glimpse, expected_glimpse, rtol=0, atol=0)
        torch.manual_seed(43)
        expected = sample_actions(agent, obs, decode_mode, "cpu")[0]
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(43)
        actual = sample_eval_actions(agent, obs, decode_mode, cache)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    agent.train()
    with pytest.raises(ValueError, match="agent.eval"):
        sample_eval_actions(agent, obs, decode_mode, cache)


def test_eval_wrapper_does_not_cache_in_training_mode_and_restores_info(monkeypatch):
    agent = _agent(training=True)
    calls = []
    hook = agent.backbone.encoder.register_forward_hook(lambda *_: calls.append(1))
    rollout_eval_batch(agent, _envs(), "greedy", 3, "cpu", seed=47)
    hook.remove()
    assert len(calls) == 3
    assert agent.training
    envs = _envs()
    def fail(*args, **kwargs):
        raise RuntimeError("inference failed")
    monkeypatch.setattr(rollout_module, "sample_eval_actions", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        rollout_eval_batch(agent, envs, "greedy", 3, "cpu")
    assert all(env.unwrapped.info_level == "full" for env in envs)


def test_eval_runtime_includes_static_encoding_and_final_route_export(monkeypatch):
    agent = _agent()
    clock = [100.0]
    work = []
    original_encode = agent.backbone.encode
    original_finalize = rollout_module.finalize_route_infos

    def timed_encode(*args, **kwargs):
        work.append("encode")
        clock[0] += 2.0
        return original_encode(*args, **kwargs)

    def timed_finalize(*args, **kwargs):
        work.append("routes")
        clock[0] += 3.0
        return original_finalize(*args, **kwargs)

    monkeypatch.setattr(rollout_module.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(agent.backbone, "encode", timed_encode)
    monkeypatch.setattr(rollout_module, "finalize_route_infos", timed_finalize)
    rows = rollout_eval_batch(
        agent, _envs(), "greedy", 1, "cpu", seed=53,
        include_routes=True, return_final_info=True,
    )
    assert work == ["encode", "routes"]
    assert len(rows) == 2
    assert all(row["batch_runtime_s"] == 5.0 for row in rows)
    assert all(row["runtime_s"] == 2.5 for row in rows)
