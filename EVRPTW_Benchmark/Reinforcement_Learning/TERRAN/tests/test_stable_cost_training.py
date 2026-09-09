from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields, replace
from types import SimpleNamespace
import json

import pytest
import torch

from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import _instance
from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import resolve_objective
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.models import Agent
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import collect_rollout
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import stable_trainer
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import data_pool
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.stable_trainer import (
    SCHEMA,
    StableState,
    _atomic_checkpoint,
    checked_actor_step,
    config_signature,
    loss_chunk,
    make_advantages,
    optimize_rollouts,
    resolve_stable_config,
    rollout_policy_kl,
    trajectory_state_weights,
    validate_resume_checkpoint,
)


@pytest.fixture(scope="module", autouse=True)
def _few_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _cfg():
    return resolve_stable_config({
        "output_dir": "/tmp/stable-cost-unit-test-unused",
        "objective": "EVRPTW_Benchmark/Reinforcement_Learning/configs/rivian_energy_vehicle_cost_v2.json",
        "model": {"embedding_dim": 16, "n_encode_layers": 1},
        "training": {
            "algorithm": "stable_cost_v1", "epochs": 2,
            "num_envs_per_gpu": 2, "n_traj": 3, "rollout_steps": 4,
            "ppo_step_chunk_size": 2, "ppo_update_epochs": 2,
            "learning_rate": 1e-4, "critic_learning_rate": 1e-4,
            "weight_decay": 0.0,
        },
        "stable_cost": {"target_kl": 0.5},
    })


def _agent():
    torch.manual_seed(105)
    return Agent(embedding_dim=16, n_encode_layers=1, critic_mode="stable_cost_v1").train()


def _optimizers(agent, *, sgd=False):
    actor = [parameter for name, parameter in agent.named_parameters() if not name.startswith("critic.")]
    optimizer = torch.optim.SGD if sgd else torch.optim.AdamW
    return (
        optimizer(actor, lr=1e-4, weight_decay=0),
        optimizer(agent.critic.parameters(), lr=1e-4, weight_decay=0),
    )


class _ForcedActions:
    """Evaluate the actual model on feasible paths with unequal stop times."""

    critic_mode = "stable_cost_v1"

    def __init__(self, agent):
        self.agent = agent
        self.backbone = agent.backbone
        self.step = 0
        self.actions = [
            [[1, 1, 3], [3, 1, 1]],
            [[2, 0, 1], [1, 2, 0]],
            [[0, 2, 0], [0, 0, 2]],
            [[0, 0, 2], [2, 0, 0]],
        ]

    def get_action_and_value_cached(self, obs, **kwargs):
        action = torch.tensor(self.actions[self.step], dtype=torch.long)
        self.step += 1
        return self.agent.get_action_and_value_cached(obs, action=action, **kwargs)


def _batch(agent):
    envs = [make_terran_env(
        instance=_instance(), n_traj=3, use_jit_mask=False,
        training_mode="stable_cost_v1", rollout_horizon_steps=4,
        objective_config=resolve_objective(_cfg()["objective"]),
    ) for _ in range(2)]
    try:
        batch = collect_rollout(_ForcedActions(agent), envs, 4, "sample", "cpu", storage_device="cpu")
    finally:
        for env in envs:
            env.close()
    assert batch.valid.sum(dim=0).tolist() == [[3, 4, 4], [4, 3, 4]]
    assert torch.isfinite(batch.old_logprobs[batch.valid]).all()
    assert batch.terminal_failure.tolist() == [[False, False, True], [True, False, False]]
    return batch


def _slice_instances(batch, start, end):
    changes = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, torch.Tensor):
            changes[field.name] = value[:, start:end] if value.ndim == 3 else value[start:end]
    changes["observations"] = [{key: value[start:end] for key, value in obs.items()}
                               for obs in batch.observations]
    changes["final_infos"] = batch.final_infos[start:end]
    return replace(batch, **changes)


def _gradients(agent):
    return {name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in agent.named_parameters()}


def _compare_gradients(left, right):
    assert left.keys() == right.keys()
    for name in left:
        assert (left[name] is None) == (right[name] is None), name
        if left[name] is not None:
            torch.testing.assert_close(left[name], right[name], rtol=3e-4, atol=3e-6, msg=name)


def test_warmup_and_cost_advantages_use_frozen_baselines_and_original_units():
    batch = _batch(_agent())
    warmup = StableState(phase="feasibility", lambda_usd=99999)
    actual = make_advantages(batch, warmup)
    score = batch.terminal_failure.float() + batch.unserved_fraction
    expected = -(score - (score.sum(-1, keepdim=True) - score) / 2)
    torch.testing.assert_close(actual, torch.where(batch.valid, expected.unsqueeze(0), 0))
    altered = replace(batch, cost_returns=batch.cost_returns * 10000)
    torch.testing.assert_close(make_advantages(altered, warmup), actual)
    cost = StableState(phase="cost", lambda_usd=200)
    actual = make_advantages(batch, cost)
    expected = -(batch.cost_returns - batch.old_cost_values) - 200 * (
        batch.failure_returns - batch.old_failure_values)
    torch.testing.assert_close(actual, torch.where(batch.valid, expected, 0))
    # This is a monetary advantage, not a count-normalized or PopArt-coordinate one.
    assert actual[batch.valid].abs().max() > 100


def test_popart_and_critic_weight_each_trajectory_equally():
    agent, cfg = _agent(), _cfg()
    batch = _batch(agent)
    weights = trajectory_state_weights(batch.valid)
    torch.testing.assert_close(weights.sum(0), torch.ones(2, 3))
    agent.critic.popart.update_stats(batch.cost_returns, mask=batch.valid, weights=weights)
    expected_mean = (batch.cost_returns.double() * weights).sum() / 6
    assert agent.critic.popart.mean.item() == pytest.approx(expected_mean.item(), rel=1e-6)
    advantage = make_advantages(batch, StableState(phase="cost", lambda_usd=100))
    result = loss_chunk(agent, batch, advantage, cfg, 0, 4, trajectory_denominator=6, time_unit=4)
    assert result[0].dtype == torch.float32
    assert agent.critic.popart.std.dtype == torch.float64
    # Independent direct computation validates trajectory averaging and dtype.
    expected = torch.zeros((), dtype=torch.float32)
    with torch.no_grad():
        cached = agent.backbone.encode(batch.observations[0])
        for step, observation in enumerate(batch.observations):
            values = agent.get_critic_outputs_cached(observation, cached)
            active = batch.valid[step]
            normalized_target = agent.critic.popart.normalize(batch.cost_returns[step][active])
            cost_loss = (values["cost_normalized"][active] - normalized_target).square()
            failure_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                values["failure_logits"][active], batch.failure_returns[step][active], reduction="none")
            expected += ((cost_loss + failure_loss) * weights[step][active]).sum() / 6
    torch.testing.assert_close(result[2], expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("phase", ["feasibility", "cost"])
def test_microbatch_and_time_chunk_gradients_equal_full_logical_batch(phase):
    agent, cfg = _agent(), _cfg()
    batch = _batch(agent)
    agent.critic.popart.update_stats(batch.cost_returns, mask=batch.valid,
                                   weights=trajectory_state_weights(batch.valid))
    split_agent = deepcopy(agent)
    state = StableState(phase=phase, lambda_usd=300)
    advantages = make_advantages(batch, state) / 100
    full = loss_chunk(agent, batch, advantages, cfg, 0, 4, trajectory_denominator=6, time_unit=4)
    full[0].backward()
    summed_loss = 0.0
    for index in range(2):
        microbatch = _slice_instances(batch, index, index + 1)
        for start in range(4):
            part = loss_chunk(split_agent, microbatch, advantages[:, index:index+1], cfg,
                              start, start + 1, trajectory_denominator=6, time_unit=4)
            if part is not None:
                summed_loss += float(part[0].detach())
                part[0].backward()
    assert summed_loss == pytest.approx(float(full[0].detach()), abs=3e-6, rel=3e-6)
    _compare_gradients(_gradients(agent), _gradients(split_agent))


def test_optimizer_results_do_not_depend_on_env_microbatch_partition():
    full_agent, cfg = _agent(), _cfg()
    batch = _batch(full_agent)
    split_agent = deepcopy(full_agent)
    full_state = StableState(phase="cost", lambda_usd=200)
    split_state = deepcopy(full_state)
    full_optimizers, split_optimizers = _optimizers(full_agent), _optimizers(split_agent)
    full_metrics = optimize_rollouts(full_agent, *full_optimizers, [batch], cfg, full_state)
    split_metrics = optimize_rollouts(split_agent, *split_optimizers,
                                     [_slice_instances(batch, 0, 1), _slice_instances(batch, 1, 2)],
                                     cfg, split_state)
    assert asdict(full_state) == pytest.approx(asdict(split_state))
    for key, value in full_agent.state_dict().items():
        torch.testing.assert_close(value, split_agent.state_dict()[key], rtol=2e-4, atol=2e-6, msg=key)
    assert full_metrics["actor_scale_used"] == pytest.approx(split_metrics["actor_scale_used"])
    assert full_metrics["ppo_updates"] > 0
    assert full_metrics["approx_kl"] <= cfg["stable_cost"]["target_kl"]
    assert full_metrics["approx_kl"] == pytest.approx(rollout_policy_kl(full_agent, [batch]))


def test_actor_update_is_unchanged_by_large_critic_error_with_independent_clips(monkeypatch):
    agent, cfg = _agent(), _cfg()
    batch = _batch(agent)
    perturbed = deepcopy(agent)
    with torch.no_grad():
        perturbed.critic.popart.linear.bias.add_(1e6)
    cfg["training"]["ppo_update_epochs"] = 1
    cfg["stable_cost"]["critic_grad_norm"] = 1e-5
    first_state = StableState(phase="cost", lambda_usd=200)
    second_state = deepcopy(first_state)
    actor_ids = {id(p) for name, p in agent.named_parameters() if not name.startswith("critic.")}
    critic_ids = {id(p) for p in agent.critic.parameters()}
    calls = []
    clip = torch.nn.utils.clip_grad_norm_

    def audited_clip(parameters, max_norm, **kwargs):
        parameters = list(parameters)
        calls.append({id(p) for p in parameters})
        return clip(parameters, max_norm, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", audited_clip)
    optimize_rollouts(agent, *_optimizers(agent, sgd=True), [batch], cfg, first_state)
    assert calls[:2] == [actor_ids, critic_ids]
    assert actor_ids.isdisjoint(critic_ids)
    metrics = optimize_rollouts(perturbed, *_optimizers(perturbed, sgd=True), [batch], cfg, second_state)
    assert metrics["critic_grad_norm"] > cfg["stable_cost"]["critic_grad_norm"]
    for (name, parameter), (other_name, other) in zip(agent.named_parameters(), perturbed.named_parameters()):
        assert name == other_name
        if not name.startswith("critic."):
            torch.testing.assert_close(parameter, other, rtol=0, atol=0, msg=name)


def _refresh_old_predictions(agent, batch):
    logprobs, costs, logits = [], [], []
    with torch.no_grad():
        cached = agent.backbone.encode(batch.observations[0])
        for step, observation in enumerate(batch.observations):
            result = agent.get_action_and_value_cached(observation, action=batch.actions[step],
                                                       state=cached, return_critic_outputs=True)
            logprobs.append(result[1])
            costs.append(result[-1]["cost_value"])
            logits.append(result[-1]["failure_logits"])
    costs, logits = torch.stack(costs), torch.stack(logits)
    return replace(batch, old_logprobs=torch.stack(logprobs), values=costs, old_cost_values=costs,
                   old_failure_logits=logits, old_failure_values=logits.sigmoid())


def test_checkpoint_restores_both_adam_optimizers_popart_and_next_update(tmp_path):
    agent, cfg = _agent(), _cfg()
    batch = _batch(agent)
    optimizers = _optimizers(agent)
    state = StableState(phase="cost", lambda_usd=200)
    optimize_rollouts(agent, *optimizers, [batch], cfg, state)
    state.epoch = 1
    checkpoint = tmp_path / "checkpoint_latest.pt"
    _atomic_checkpoint(checkpoint, {
        "schema": SCHEMA, "training_signature": config_signature(cfg),
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizers[0].state_dict(),
        "critic_optimizer_state_dict": optimizers[1].state_dict(),
        "stable_state": asdict(state),
    })
    payload = torch.load(checkpoint, weights_only=False)
    restored = _agent()
    restored.load_state_dict(payload["model_state_dict"])
    resumed_optimizers = _optimizers(restored)
    resumed_optimizers[0].load_state_dict(payload["optimizer_state_dict"])
    resumed_optimizers[1].load_state_dict(payload["critic_optimizer_state_dict"])
    resumed_state = StableState(**payload["stable_state"])
    assert resumed_state == state
    assert payload["training_signature"] == config_signature(cfg)
    for expected, actual in zip(optimizers, resumed_optimizers):
        assert len(actual.state) == len(expected.state) > 0
        for old_group, new_group in zip(expected.param_groups, actual.param_groups):
            for old_parameter, new_parameter in zip(old_group["params"], new_group["params"]):
                for key, value in expected.state[old_parameter].items():
                    torch.testing.assert_close(value, actual.state[new_parameter][key])
    next_batch = _refresh_old_predictions(agent, batch)
    optimize_rollouts(agent, *optimizers, [next_batch], cfg, state)
    optimize_rollouts(restored, *resumed_optimizers, [next_batch], cfg, resumed_state)
    for key, value in agent.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0, msg=key)
    assert resumed_state == state


def _resume_payload(cfg):
    return {"schema": SCHEMA, "training_signature": config_signature(cfg),
            "config": deepcopy(cfg), "seed": 1234,
            "stable_state": asdict(StableState(epoch=2, sample_count=113))}


def _resized_config(cfg):
    resized = deepcopy(cfg)
    resized["training"].update(num_envs_per_gpu=4, effective_batch_size=12,
                                logical_microbatches_per_epoch=3, ppo_step_chunk_size=1,
                                resume_checkpoint="/tmp/source.ckpt")
    return resized


def test_resume_batch_change_is_strict_by_default_and_requires_explicit_flag():
    cfg = _cfg()
    payload = _resume_payload(cfg)
    resized = _resized_config(cfg)
    with pytest.raises(ValueError, match="explicit allow_batch_resize_resume"):
        validate_resume_checkpoint(payload, resized, seed=1234, source="/tmp/source.ckpt")
    resized["training"]["allow_batch_resize_resume"] = True
    metadata = validate_resume_checkpoint(payload, resized, seed=1234, source="/tmp/source.ckpt")
    assert set(metadata["changed_batch_fields"]) == set(stable_trainer.BATCH_RESUME_FIELDS)
    assert metadata["old_batch_geometry"]["effective_batch_size"] == 2
    assert metadata["new_batch_geometry"]["effective_batch_size"] == 12
    assert metadata["source_sample_count"] == 113
    assert metadata["optimizer_reset"] is False


@pytest.mark.parametrize("section,key,value", [
    ("training", "n_traj", 16), ("training", "learning_rate", 0.01),
    ("training", "critic_learning_rate", 0.01), ("training", "ppo_update_epochs", 3),
    ("training", "rollout_steps", 100), ("training", "epochs", 3),
    ("model", "embedding_dim", 32), ("stable_cost", "target_kl", 0.01),
    ("env", "normalize_reward", True), ("objective", "vehicle_unit_cost", 9999),
    ("data", "stage2_scale", "Cus1000"),
])
def test_batch_resume_flag_never_bypasses_other_training_changes(section, key, value):
    cfg = _cfg()
    resized = _resized_config(cfg)
    resized["training"]["allow_batch_resize_resume"] = True
    resized[section][key] = value
    with pytest.raises(ValueError, match="permits only changes"):
        validate_resume_checkpoint(_resume_payload(cfg), resized, seed=1234, source="source.ckpt")


def test_resume_validates_checkpoint_original_signature_before_batch_exception():
    cfg = _cfg()
    payload = _resume_payload(cfg)
    resized = _resized_config(cfg)
    resized["training"]["allow_batch_resize_resume"] = True
    # Even tampering with an allowed field must fail original provenance checks.
    payload["config"]["training"]["ppo_step_chunk_size"] = 3
    with pytest.raises(ValueError, match="original configuration signature is invalid"):
        validate_resume_checkpoint(payload, resized, seed=1234, source="source.ckpt")


def test_batch_resume_requires_checkpoint_and_preserves_seed_validation():
    cfg = _cfg()
    cfg["training"]["allow_batch_resize_resume"] = True
    with pytest.raises(ValueError, match="requires a resume checkpoint"):
        resolve_stable_config(cfg)
    cfg["training"]["resume_checkpoint"] = "source.ckpt"
    with pytest.raises(ValueError, match="seed mismatch"):
        validate_resume_checkpoint(_resume_payload(cfg), cfg, seed=17, source="source.ckpt")


def test_batch_resize_training_resume_preserves_all_state_and_sampler_cursor(tmp_path, monkeypatch):
    from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import trainer

    agent, cfg = _agent(), _cfg()
    optimizers = _optimizers(agent)
    state = StableState(phase="cost", lambda_usd=200, epoch=2, sample_count=113,
                        success_streak=3, transitions=591, best_feasible_rate=1.0, best_cost=321)
    optimize_rollouts(agent, *optimizers, [_batch(agent)], cfg, state)
    # Deliberately distinguish the old pool creation cursor from current progress.
    cfg["data"]["stage2_completed_samples"] = 17
    payload = {**_resume_payload(cfg), "stable_state": asdict(state),
               "model_state_dict": agent.state_dict(),
               "optimizer_state_dict": optimizers[0].state_dict(),
               "critic_optimizer_state_dict": optimizers[1].state_dict(),
               "initialization": {"path": "legacy-actor.ckpt", "optimizer_reset": True}}
    source = tmp_path / "source.ckpt"
    _atomic_checkpoint(source, payload)
    resized = _resized_config(cfg)
    resized["training"].update(allow_batch_resize_resume=True, resume_checkpoint=str(source))
    resized["output_dir"] = str(tmp_path / "resumed")
    observed = {}

    def make_empty_envs(actual_cfg, seed):
        observed.update(cfg=deepcopy(actual_cfg), seed=seed)
        return [], SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(trainer, "make_envs", make_empty_envs)
    # An already completed schedule takes the real restore/save path, with no
    # gradient update that could obscure whether any state was reset.
    latest = stable_trainer.train_stable_cost(resized, seed=1234, device="cpu")
    restored = torch.load(latest, weights_only=False)
    assert restored["stable_state"] == asdict(state)
    assert observed["cfg"]["data"]["stage2_completed_samples"] == 113
    assert observed["seed"] == 1234
    assert restored["initialization"] == payload["initialization"]
    for name, value in agent.state_dict().items():
        torch.testing.assert_close(restored["model_state_dict"][name], value, rtol=0, atol=0)
    for optimizer, field in zip(optimizers, ["optimizer_state_dict", "critic_optimizer_state_dict"]):
        comparison = deepcopy(optimizer)
        comparison.load_state_dict(restored[field])
        _assert_optimizer_state_equal(comparison, optimizer.state_dict())
    contract = json.loads((latest.parent / "training_contract.json").read_text())
    assert restored["resume"] == contract["resume"]
    assert restored["resume"]["source_checkpoint"] == str(source)
    assert restored["resume"]["source_sample_count"] == 113
    assert restored["resume"]["new_batch_geometry"]["effective_batch_size"] == 12
    assert restored["training_signature"] == config_signature(restored["config"])
    # Saved cursor rewriting and the one-time exception flag do not break a
    # subsequent normal, strict resume of the new batch geometry.
    strict = deepcopy(restored["config"])
    strict["training"].pop("allow_batch_resize_resume")
    assert not validate_resume_checkpoint(restored, strict, seed=1234, source=latest)["changed_batch_fields"]


def _assert_optimizer_state_equal(actual, expected, *, expected_lr=None):
    actual = actual.state_dict()
    expected = deepcopy(expected)
    if expected_lr is not None:
        for group in expected["param_groups"]:
            group["lr"] = expected_lr
    assert actual["param_groups"] == expected["param_groups"]
    assert actual["state"].keys() == expected["state"].keys()
    for parameter, values in expected["state"].items():
        assert actual["state"][parameter].keys() == values.keys()
        for key, value in values.items():
            torch.testing.assert_close(actual["state"][parameter][key], value, rtol=0, atol=0)


def _actor_with_adam_history():
    agent = _agent()
    actor_optimizer, _ = _optimizers(agent)
    for group in actor_optimizer.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.full_like(parameter, 0.125)
    actor_optimizer.step()
    for group in actor_optimizer.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.full_like(parameter, -0.25)
    return agent, actor_optimizer


def test_rejected_actor_proposals_restore_parameters_and_adam_moments(monkeypatch):
    agent, optimizer = _actor_with_adam_history()
    cfg = _cfg()
    cfg["stable_cost"]["actor_kl_backtracks"] = 2
    original_parameters = deepcopy(agent.state_dict())
    original_optimizer = deepcopy(optimizer.state_dict())
    original_gradients = _gradients(agent)
    estimates = iter([float("inf"), 0.8, 0.7, 0.0])
    monkeypatch.setattr(stable_trainer, "rollout_policy_kl", lambda *_: next(estimates))
    accepted, kl, backtracks = checked_actor_step(agent, optimizer, [], cfg)
    assert (accepted, kl, backtracks) == (False, 0.0, 3)
    for name, parameter in agent.state_dict().items():
        torch.testing.assert_close(parameter, original_parameters[name], rtol=0, atol=0, msg=name)
    # The persistent smaller learning rate is the only optimizer change.
    _assert_optimizer_state_equal(optimizer, original_optimizer, expected_lr=1e-4 / 8)
    _compare_gradients(_gradients(agent), original_gradients)


def test_accepted_backtrack_matches_one_adam_step_at_reduced_learning_rate(monkeypatch):
    agent, optimizer = _actor_with_adam_history()
    expected_agent = deepcopy(agent)
    expected_optimizer, _ = _optimizers(expected_agent)
    expected_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    for expected_group, group in zip(expected_optimizer.param_groups, optimizer.param_groups):
        expected_group["lr"] *= 0.5
        for expected, parameter in zip(expected_group["params"], group["params"]):
            expected.grad = parameter.grad.clone()
    expected_optimizer.step()
    estimates = iter([0.75, 0.125])
    monkeypatch.setattr(stable_trainer, "rollout_policy_kl", lambda *_: next(estimates))
    accepted, kl, backtracks = checked_actor_step(agent, optimizer, [], _cfg())
    assert (accepted, kl, backtracks) == (True, 0.125, 1)
    for name, parameter in agent.state_dict().items():
        torch.testing.assert_close(parameter, expected_agent.state_dict()[name], rtol=0, atol=0, msg=name)
    _assert_optimizer_state_equal(optimizer, expected_optimizer.state_dict())


@pytest.mark.parametrize("offset", [3, 7, 14])
def test_stage2_sample_cursor_resume_matches_uninterrupted_order(monkeypatch, offset):
    class FakeStage2TaskPool:
        def __init__(self, **kwargs):
            self.tasks = [SimpleNamespace(view_id=f"view-{index}") for index in range(7)]

        def __len__(self):
            return len(self.tasks)

        def instance(self, task):
            return task.view_id

    monkeypatch.setattr(data_pool, "Stage2TaskPool", FakeStage2TaskPool)
    kwargs = dict(dataset_path="unused-frozen-index", seed=41)
    uninterrupted = data_pool.Stage2TERRANPool(**kwargs)
    reference = [uninterrupted.sample() for _ in range(28)]
    assert uninterrupted.drain_sampled_view_ids() == []  # Legacy default adds no ID buffer.
    resumed = data_pool.Stage2TERRANPool(**kwargs, completed_samples=offset, record_sample_ids=True)
    actual = [resumed.sample() for _ in range(28 - offset)]
    assert actual == reference[offset:]
    assert resumed.sample_count == 28
    assert resumed.drain_sampled_view_ids() == actual
    assert resumed.drain_sampled_view_ids() == []
    # Existing whole-pass checkpoints retain exactly the same next sample order.
    if offset % 7 == 0:
        legacy = data_pool.Stage2TERRANPool(**kwargs, completed_data_passes=offset // 7)
        assert [legacy.sample() for _ in range(28 - offset)] == reference[offset:]
