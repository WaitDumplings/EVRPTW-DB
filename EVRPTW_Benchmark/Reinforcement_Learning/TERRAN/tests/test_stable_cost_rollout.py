from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import _instance
from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import resolve_objective
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.env_factory import make_terran_env
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.pbrs import PotentialRewardConfig
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.rollout import (
    BoundedBaseRewardStats,
    collect_rollout,
)


def _env(**kwargs):
    options = dict(
        instance=_instance(), n_traj=1, use_jit_mask=False,
        training_mode="stable_cost_v1",
        objective_config=resolve_objective(
            "EVRPTW_Benchmark/Reinforcement_Learning/configs/rivian_energy_vehicle_cost_v2.json"
        ),
    )
    options.update(kwargs)
    return make_terran_env(**options)


@pytest.mark.parametrize("fast", [False, True])
def test_raw_usd_includes_dispatch_station_and_return_without_extra_rewards(fast):
    env = _env(
        use_fast_env=fast, normalize_reward=True, reward_objective_scale=7.0,
        success_bonus=50.0, invalid_action_penalty=-10.0,
    )
    obs, _ = env.reset()
    assert not obs["dispatch_paid"].any()
    total = 0.0
    for destination in (3, 1, 0, 2, 0):
        obs, reward, terminated, truncated, info = env.step([destination])
        total += float(reward[0])
        assert bool(obs["dispatch_paid"][0]) == (destination != 0)
    assert terminated[0] and not truncated[0]
    assert info["vehicles_started"][0] == 2
    assert info["objective_distance_km"][0] == pytest.approx(10.0)
    assert total == pytest.approx(-info["objective_value"][0], abs=1e-4)
    assert not info["failure_terminal"][0]
    _, reward, _, _, padded = env.step([1])
    assert reward[0] == 0.0
    assert padded["objective_value"][0] == info["objective_value"][0]
    assert not padded["failure_terminal"].any()


@pytest.mark.parametrize("fast", [False, True])
def test_unserved_is_visited_ledger_not_positive_demand_or_action_mask(fast):
    instance = deepcopy(_instance())
    instance.demands_cm3[0] = 0.0
    instance.tw_s[1] = [0.0, 1.0]  # Customer 2 is still unserved, but unreachable.
    env = _env(instance=instance, use_fast_env=fast)
    obs, _ = env.reset()
    np.testing.assert_array_equal(obs["customer_unserved"], [[True, True]])
    assert obs["remaining_demand"][0, 1] == 0.0
    assert not obs["action_mask"][0, 2]
    obs, _, _, _, _ = env.step([1])
    np.testing.assert_array_equal(obs["customer_unserved"], [[False, True]])


def test_horizon_failure_is_once_and_padding_cannot_advance_environment():
    env = _env(rollout_horizon_steps=1)
    obs, _ = env.reset()
    np.testing.assert_array_equal(obs["episode_step_budget"], [1.0])
    np.testing.assert_array_equal(obs["remaining_step_budget"], [1.0])
    obs, reward, terminated, truncated, info = env.step([1])
    assert truncated[0] and not terminated[0]
    assert info["failure_terminal"][0]
    assert info["unserved_fraction"][0] == pytest.approx(0.5)
    assert obs["remaining_step_budget"][0] == 0.0
    assert reward[0] == pytest.approx(-info["objective_value"][0], abs=1e-4)
    _, reward, _, _, again = env.step([2])
    assert not again["failure_terminal"].any()
    assert reward[0] == 0.0
    assert again["served_customers"][0] == 1


def test_stable_mode_rejects_distance_objective_and_pbrs():
    with pytest.raises(ValueError, match="economic cost"):
        _env(objective_config=None)
    with pytest.raises(ValueError, match="does not permit PBRS"):
        _env(pbrs_config=PotentialRewardConfig(use_customer_pbrs=True))


class _ScriptedBackbone:
    def encode(self, obs):
        return {"demand": obs["demand"]}


class _ScriptedStableAgent:
    critic_mode = "stable_cost_v1"
    backbone = _ScriptedBackbone()

    def __init__(self, actions):
        self.actions = actions
        self.step = 0

    def get_action_and_value_cached(self, obs, **kwargs):
        assert kwargs["return_critic_outputs"]
        actions = torch.tensor([self.actions[self.step]], dtype=torch.long)
        self.step += 1
        cost = torch.full_like(actions, float(self.step), dtype=torch.float32)
        logits = torch.full_like(cost, -0.25)
        zero = torch.zeros_like(cost)
        outputs = {"cost_value": cost, "cost_normalized": cost, "failure_logits": logits}
        return actions, zero, zero, cost.unsqueeze(-1), kwargs["state"], outputs


def test_complete_mc_labels_old_values_and_compact_cpu_state_are_aligned():
    env = _env(n_traj=2, rollout_horizon_steps=4)
    agent = _ScriptedStableAgent([[1, 1], [2, 0], [0, 2], [0, 1]])
    batch = collect_rollout(agent, [env], 4, "sample", "cpu", storage_device="cpu")
    assert batch.valid[:, 0, 0].tolist() == [True, True, True, False]
    assert batch.valid[:, 0, 1].tolist() == [True, True, True, True]
    assert batch.terminal_failure.tolist() == [[False, True]]
    # A failed all-served episode is not mistaken for success without a depot return.
    assert batch.unserved_fraction.tolist() == [[0.0, 0.0]]
    assert batch.failure_returns[:, 0, 1].tolist() == [1.0] * 4
    assert batch.failure_returns[:, 0, 0].tolist() == [0.0] * 4
    assert batch.cost_returns[3, 0, 0] == 0.0
    torch.testing.assert_close(batch.old_cost_values, batch.values)
    torch.testing.assert_close(batch.old_failure_values, batch.old_failure_logits.sigmoid())
    np.testing.assert_allclose(
        batch.cost_returns[0, 0].numpy(), batch.final_infos[0]["objective_value"], atol=1e-4,
    )
    for snapshot in batch.observations:
        assert snapshot["customer_unserved"].dtype == np.bool_
        assert snapshot["customer_unserved"].shape == (1, 2, 2)
        assert "remaining_demand" not in snapshot
        assert snapshot["demand"] is batch.observations[0]["demand"]
    assert batch.observations[0]["customer_unserved"].all()
    assert not batch.observations[2]["customer_unserved"][0, 0].any()
    assert batch.old_cost_values.device.type == "cpu"


def test_collector_limit_is_visible_and_is_a_complete_failure_target():
    env = _env()  # No explicit environment rollout wrapper.
    batch = collect_rollout(_ScriptedStableAgent([[1]]), [env], 1, "sample", "cpu")
    assert batch.observations[0]["episode_step_budget"][0, 0] == 1.0
    assert batch.observations[0]["remaining_step_budget"][0, 0] == 1.0
    assert batch.dones[-1].all()
    assert batch.terminal_failure.all()
    assert batch.unserved_fraction.item() == pytest.approx(0.5)
    assert batch.failure_returns.item() == 1.0
    assert batch.final_infos[0]["failure_reason"][0] == "rollout_budget_exhausted"


def test_stable_rollout_rejects_discounted_cost_target():
    with pytest.raises(ValueError, match="gamma=1"):
        collect_rollout(_ScriptedStableAgent([[1]]), [_env()], 1, "sample", "cpu", reward_discount_factor=0.99)


def test_dead_initial_state_counts_failure_without_fabricating_an_active_action():
    instance = deepcopy(_instance())
    instance.vehicle["battery_capacity_kwh"] = 0.01
    batch = collect_rollout(
        _ScriptedStableAgent([[0]]), [_env(instance=instance)], 5, "sample", "cpu",
    )
    assert not batch.valid.any()
    assert batch.trajectory_steps.item() == 0
    assert batch.terminal_failure.item()
    assert batch.unserved_fraction.item() == 1.0
    assert not batch.cost_returns.any()
    assert not batch.failure_returns.any()  # Padding is not a critic sample.
    assert batch.final_infos[0]["failure_reason"][0] == "no_feasible_action"


def test_actual_stable_agent_replays_compact_cpu_rollout_without_value_drift():
    from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.models import Agent

    torch.manual_seed(81)
    agent = Agent(embedding_dim=32, n_encode_layers=1, critic_mode="stable_cost_v1", device="cpu")
    agent.eval()
    batch = collect_rollout(agent, [_env(n_traj=2, rollout_horizon_steps=5)], 5, "sample", "cpu", storage_device="cpu")
    with torch.no_grad():
        cached = agent.backbone.encode(batch.observations[0])
        for step, obs in enumerate(batch.observations):
            _, logprob, _, _, _, outputs = agent.get_action_and_value_cached(
                obs, action=batch.actions[step], state=cached, return_critic_outputs=True,
            )
            torch.testing.assert_close(logprob, batch.old_logprobs[step])
            torch.testing.assert_close(outputs["cost_value"], batch.old_cost_values[step])
            torch.testing.assert_close(outputs["failure_logits"], batch.old_failure_logits[step])


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("with_base_stats", [False, True])
def test_disabling_reward_diagnostics_preserves_seeded_rollout(fast, with_base_stats):
    from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.models import Agent

    torch.manual_seed(81)
    agent = Agent(embedding_dim=32, n_encode_layers=1, critic_mode="stable_cost_v1", device="cpu")
    agent.eval()
    batches, accumulators = [], []
    for enabled in (True, False):
        torch.manual_seed(919)
        env = _env(n_traj=2, rollout_horizon_steps=5, use_fast_env=fast)
        stats = BoundedBaseRewardStats() if with_base_stats else None
        try:
            batches.append(collect_rollout(
                agent, [env], 5, "sample", "cpu", seed=919,
                storage_device="cpu", collect_reward_diagnostics=enabled,
                base_reward_stats=stats,
            ))
            accumulators.append(stats)
        finally:
            env.close()
    enabled, disabled = batches
    assert enabled.reward_diagnostics["active_count"] > 0
    assert disabled.reward_diagnostics == {}
    for name, expected in vars(enabled).items():
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(getattr(disabled, name), expected, rtol=0, atol=0)
    assert len(enabled.observations) == len(disabled.observations)
    for expected, actual in zip(enabled.observations, disabled.observations):
        assert actual.keys() == expected.keys()
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
    for key in ("objective_value", "success", "served_customers", "failure_terminal", "failure_reason"):
        np.testing.assert_array_equal(disabled.final_infos[0][key], enabled.final_infos[0][key])
    if with_base_stats:
        assert accumulators[0].count > 0
        assert accumulators[0].summary() == accumulators[1].summary()
