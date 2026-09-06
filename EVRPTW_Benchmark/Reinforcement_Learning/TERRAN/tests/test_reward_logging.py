from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import _instance
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_diagnostics import summarize_values
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import trainer
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.models import Agent
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.pbrs import PotentialRewardConfig
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import BoundedBaseRewardStats, collect_rollout, compute_returns


def test_bounded_base_statistics_are_masked_full_moments_and_chunk_invariant() -> None:
    values = np.arange(10000, dtype=np.float64) / 7
    values[[4, 7]] = [np.nan, np.inf]
    mask = np.arange(values.size) % 3 != 0
    before = values.copy()
    whole, chunked = BoundedBaseRewardStats(31), BoundedBaseRewardStats(31)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    whole.update(values, mask)
    for offset in range(0, values.size, 127):
        chunked.update(values[offset:offset + 127], mask[offset:offset + 127])
        assert chunked.samples.size <= 31
    summary = chunked.summary()
    expected = summarize_values(values, mask)
    for key in ("count", "finite_count", "nonfinite_count", "mean", "std", "min", "max"):
        assert summary[key] == pytest.approx(expected[key])
    for key in ("p05", "p50", "p95"):
        assert summary[key] == whole.summary()[key]
    assert summary["quantile_sample_count"] == 31
    assert summary["quantiles_approximate"] is True
    assert "splitmix64_bottom_k" in summary["quantile_method"]
    np.testing.assert_equal(values, before)
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert random.getstate() == python_state


def test_multipart_tensor_summary_matches_global_finite_distribution() -> None:
    first = torch.tensor([1.0, 999.0, float("nan"), 3.0])
    second = torch.tensor([5.0, float("inf"), 7.0, 9.0])
    mask = torch.tensor([True, False, True, True])
    actual = trainer._summarize_tensor_parts([(first, mask), (second, None)], max_quantile_samples=3)
    expected = summarize_values(torch.cat((first[mask], second)), max_quantile_samples=3)
    for key in ("count", "finite_count", "nonfinite_count", "mean", "std", "min", "p05", "p50", "p95", "max", "quantile_sample_count"):
        assert actual[key] == pytest.approx(expected[key])


def _manual_record():
    rewards = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 1e9]]])
    valid = torch.tensor([[[True, True]], [[True, True]], [[True, False]]])
    dones = torch.tensor([[[False, False]], [[False, True]], [[True, True]]])
    returns = compute_returns(rewards, dones, gamma=1.0)
    raw = returns - torch.tensor([[[0.5, 1.0]], [[1.0, 2.0]], [[3.0, -1e9]]])
    flattened = raw[valid]
    normalized = (raw - flattened.mean()) / (flattened.std(unbiased=False) + 1e-8)
    return SimpleNamespace(rewards=rewards, valid=valid), returns, flattened, normalized


def test_epoch_log_distinguishes_steps_initial_returns_and_gradients_without_mutation() -> None:
    batch, returns, raw, normalized = _manual_record()
    before = batch.rewards.clone()
    rng = torch.get_rng_state().clone()
    base = BoundedBaseRewardStats()
    base.update(np.asarray([0.5, 1, 2, 3, 4, -1e9]), np.asarray([1, 1, 1, 1, 1, 0], dtype=bool))
    result = trainer.build_epoch_reward_diagnostics(
        cfg={"training": {"gamma": 1.0, "max_grad_norm": 1.0}, "normalization": {"seed": 1234, "sampled_view_ids": ["a", "b"]}},
        epoch=7, session_id="test", start_epoch=6, resume_from="saved.pt",
        rollout_records=[(batch, returns, normalized)], raw_advantages=raw,
        base_reward_stats=base, reward_components={"base_sum": 10.5, "base_abs_sum": 10.5, "shaped_sum": 15.0},
        normalization_records=[{"normalize_reward": True, "objective_scale": 100.0, "distance_scale_km": 50.0}],
        preclip_norms=[torch.tensor(0.5), torch.tensor(2.0), torch.tensor(float("nan"))],
        optimizer_steps_total=20, pbrs_scale=0.25,
    )
    distributions = result["distributions"]
    assert distributions["base_reward_per_active_step"]["mean"] == 2.1
    assert distributions["shaped_reward_per_active_step"]["mean"] == 3.0
    assert distributions["shaped_return_to_go"]["count"] == 5
    assert distributions["shaped_return_to_go"]["mean"] == 6.4
    for key in ("shaped_initial_return_per_trajectory", "shaped_total_per_trajectory"):
        assert distributions[key]["count"] == 2
        assert distributions[key]["mean"] == 7.5
    assert distributions["advantage_after_normalization"]["mean"] == pytest.approx(0.0, abs=1e-7)
    assert distributions["advantage_after_normalization"]["std"] == pytest.approx(1.0)
    assert result["gradients"]["preclip_global_norm"]["nonfinite_count"] == 1
    assert result["gradients"]["clip_fraction"] == 0.5
    assert result["gradients"]["clip_fraction_finite_optimizer_steps"] == 2
    assert result["normalization"]["training_pool_metadata"]["sampled_view_ids"] == ["a", "b"]
    assert result["components"]["base"]["active_step_mean"] == 2.1
    assert result["components"]["base"]["mean_sum_per_trajectory"] == 5.25
    assert result["resume_from"] == "saved.pt" and result["start_epoch"] == 6
    json.dumps(result, allow_nan=False)
    torch.testing.assert_close(batch.rewards, before, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)


def test_rollout_instrumentation_leaves_actions_rewards_and_rng_identical() -> None:
    policy = Agent(embedding_dim=32, n_encode_layers=1, device="cpu")

    def env():
        return make_terran_env(instance=_instance(), n_traj=4, use_jit_mask=False,
            pbrs_config=PotentialRewardConfig(use_customer_pbrs=True, use_repair_distance_pbrs=True))

    torch.manual_seed(777)
    reference = collect_rollout(policy, [env()], 12, "sample", "cpu", seed=313)
    reference_rng = torch.get_rng_state().clone()
    stats = BoundedBaseRewardStats()
    torch.manual_seed(777)
    observed = collect_rollout(policy, [env()], 12, "sample", "cpu", seed=313, base_reward_stats=stats)
    for name in ("actions", "rewards", "dones", "values", "valid", "old_logprobs"):
        torch.testing.assert_close(getattr(reference, name), getattr(observed, name), rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), reference_rng)
    summary = stats.summary()
    assert summary["count"] == int(observed.valid.sum())
    assert summary["mean"] == pytest.approx(observed.reward_diagnostics["base_sum"] / summary["count"])


@pytest.mark.parametrize("active_count", [0, 1])
def test_empty_and_single_active_advantage_diagnostics_remain_defined(active_count) -> None:
    rewards = torch.tensor([[[2.0, 1e9]]])
    valid = torch.tensor([[[bool(active_count), False]]])
    batch = SimpleNamespace(rewards=rewards, valid=valid)
    result = trainer.build_epoch_reward_diagnostics(
        cfg={"training": {"gamma": 1.0}}, epoch=1, session_id="empty-single", start_epoch=1, resume_from=None,
        rollout_records=[(batch, rewards, rewards)], raw_advantages=rewards[valid],
        base_reward_stats=BoundedBaseRewardStats(), reward_components={}, normalization_records=[],
        preclip_norms=[], optimizer_steps_total=0, pbrs_scale=0,
    )
    assert result["normalization"]["advantage_normalization_applied"] is False
    stats = result["distributions"]["advantage_after_normalization"]
    assert stats["count"] == active_count
    assert stats["mean"] == (2.0 if active_count else None)
    assert stats["std"] == (0.0 if active_count else None)
    assert result["distributions"]["shaped_initial_return_per_trajectory"]["count"] == active_count
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("microbatches, expected_steps", [(1, 4), (2, 2)])
def test_both_ppo_paths_log_existing_clip_norm_once_and_keep_updates_identical(
    tmp_path: Path, monkeypatch, microbatches: int, expected_steps: int,
) -> None:
    real_builder = trainer.build_epoch_reward_diagnostics
    real_clip = torch.nn.utils.clip_grad_norm_
    captured_norms = []

    def capture_clip(parameters, max_norm):
        parameters = list(parameters)
        manual = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach()) for p in parameters if p.grad is not None]))
        returned = real_clip(parameters, max_norm)
        torch.testing.assert_close(returned, manual)
        captured_norms.append(float(returned))
        return returned

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture_clip)
    monkeypatch.setattr(trainer, "Agent", lambda **_kwargs: torch.nn.Linear(1, 1, bias=False))
    payloads, rng_states, logs = [], [], []
    for logging_enabled in (False, True):
        pool = SimpleNamespace(sample_count=0)
        env = SimpleNamespace(normalize_reward=True, reward_objective_scale=10.0, reward_distance_scale_km=10.0)
        monkeypatch.setattr(trainer, "make_envs", lambda _cfg, _seed: ([env, env], pool))

        def collect(_agent, _envs, **kwargs):
            pool.sample_count += 2
            rewards = torch.rand(2, 2, 2)
            valid = torch.tensor([[[True, True], [True, True]], [[True, False], [True, False]]])
            kwargs["base_reward_stats"].update(rewards.numpy(), valid.numpy())
            return SimpleNamespace(
                actions=torch.zeros((2, 2, 2), dtype=torch.long), rewards=rewards,
                dones=torch.ones((2, 2, 2), dtype=torch.bool), values=torch.zeros_like(rewards), valid=valid,
                trajectory_steps=valid.sum(dim=0), rollout_budget_exhausted=torch.zeros((2, 2), dtype=torch.bool),
                final_infos=[{"success": np.ones(2, dtype=bool), "objective_distance_km": np.ones(2), "vehicle_count": np.ones(2), "served_customers": np.full(2, 2)}] * 2,
                timings={}, reward_diagnostics={"active_count": int(valid.sum()), "base_sum": float(rewards[valid].sum()), "shaped_sum": float(rewards[valid].sum())},
            )

        def loss(agent, _batch, _returns, advantages, *_args, **_kwargs):
            total = (agent.weight.square().sum() + agent.weight.sum() * advantages.square().mean())
            return total, total.detach(), total.detach(), total.detach()

        monkeypatch.setattr(trainer, "collect_rollout", collect)
        monkeypatch.setattr(trainer, "evaluate_policy_loss", loss)
        monkeypatch.setattr(trainer, "build_epoch_reward_diagnostics", real_builder if logging_enabled else lambda **_kwargs: {"schema": "disabled-test"})
        output = tmp_path / str(logging_enabled)
        cfg = {"output_dir": str(output), "data": {"num_customers": 2, "num_charging_stations": 1},
            "model": {}, "env": {}, "pbrs": {}, "evaluation": {"eval_interval": 0},
            "training": {"epochs": 1, "gamma": 1.0, "num_envs_per_gpu": 2, "n_traj": 2,
                "logical_microbatches_per_epoch": microbatches, "rollout_steps": 2, "ppo_update_epochs": 2,
                "num_minibatches": 2, "gradient_accumulation_steps": 1, "max_grad_norm": 0.1, "checkpoint_interval": 1}}
        offset = len(captured_norms)
        checkpoint = trainer.train_from_config(cfg, seed=1234, device="cpu")
        assert len(captured_norms) - offset == expected_steps
        payloads.append(torch.load(checkpoint, map_location="cpu", weights_only=False))
        rng_states.append(torch.get_rng_state().clone())
        logs.append(json.loads((output / "reward_diagnostics.jsonl").read_text()))
    for name, value in payloads[0]["model_state_dict"].items():
        torch.testing.assert_close(value, payloads[1]["model_state_dict"][name], rtol=0, atol=0)
    for parameter_id, state in payloads[0]["optimizer_state_dict"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, payloads[1]["optimizer_state_dict"]["state"][parameter_id][key], rtol=0, atol=0)
    assert torch.equal(*rng_states)
    assert logs[1]["gradients"]["preclip_global_norm"]["count"] == expected_steps
    assert logs[1]["gradients"]["preclip_global_norm"]["mean"] == pytest.approx(np.mean(captured_norms[-expected_steps:]))
    assert logs[1]["schema"] == "drl_reward_diagnostics_v1"
    sidecar = output / "reward_diagnostics.jsonl"
    previous_log = sidecar.read_text()
    cfg["training"]["epochs"] = 2
    cfg["protocol"] = {"resume_checkpoint": str(checkpoint)}
    trainer.train_from_config(cfg, seed=1234, device="cpu")
    assert sidecar.read_text().startswith(previous_log)
    sessions = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert [row["epoch"] for row in sessions] == [1, 2]
    assert sessions[1]["session_id"] != sessions[0]["session_id"]
    assert sessions[1]["start_epoch"] == 2
    assert sessions[1]["resume_from"] == str(checkpoint)
