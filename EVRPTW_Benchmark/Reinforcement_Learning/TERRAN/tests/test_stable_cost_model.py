from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.models import Agent
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.models.attention_model_wrapper import (
    MODEL_OBSERVATION_KEYS,
    stateWrapper,
)
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.models.stable_cost_critic import (
    PopArtHead,
    remaining_task_summary,
)


@pytest.fixture(scope="module", autouse=True)
def _few_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _observation(n: int, *, trajectories: int = 2, stations: int = 2):
    gen = torch.Generator().manual_seed(73 + n)
    nodes = 1 + n + stations
    demand = torch.zeros(1, nodes)
    demand[:, 1 : 1 + n] = 0.1
    unserved = torch.ones(1, trajectories, n, dtype=torch.bool)
    unserved[:, 0] = False  # Finished serving still needs return/charging context.
    return {
        "cus_loc": torch.rand(1, n, 2, generator=gen),
        "depot_loc": torch.zeros(1, 1, 2),
        "rs_loc": torch.rand(1, stations, 2, generator=gen),
        "demand": demand,
        "time_window": torch.stack((torch.zeros(1, nodes), torch.ones(1, nodes)), -1),
        "charging_time_ratio": torch.zeros(1, nodes),
        "battery_capacity": torch.ones(1, 1),
        "loading_capacity": torch.ones(1, 1),
        "action_mask": torch.ones(1, trajectories, nodes, dtype=torch.bool),
        "last_node_idx": torch.ones(1, trajectories, dtype=torch.long),
        "current_load": torch.full((1, trajectories), 0.4),
        "current_battery": torch.full((1, trajectories), 0.3),
        "current_time": torch.full((1, trajectories), 0.6),
        "customer_unserved": unserved,
        "dispatch_paid": torch.ones(1, trajectories, dtype=torch.bool),
        "remaining_step_budget": torch.full((1, trajectories), float(n)),
        "episode_step_budget": torch.full((1, trajectories), float(2 * n)),
    }


def test_one_stable_model_accepts_different_customer_counts():
    agent = Agent(embedding_dim=16, n_encode_layers=1, critic_mode="stable_cost_v1").eval()
    keys = {key: tuple(value.shape) for key, value in agent.state_dict().items()}
    for n in (5, 1000):
        obs = _observation(n)
        with torch.no_grad():
            action, logp, entropy, value, cached, outputs = agent.get_action_and_value_cached(
                obs, decode_mode="greedy", return_critic_outputs=True,
            )
        assert action.shape == logp.shape == entropy.shape == (1, 2)
        assert value.shape == (1, 2, 1)
        assert cached[0].shape == (1, n + 3, 16)
        for tensor in outputs.values():
            assert tensor.shape == (1, 2)
            assert torch.isfinite(tensor).all()
        assert {key: tuple(value.shape) for key, value in agent.state_dict().items()} == keys


def test_unserved_pool_uses_customer_order_and_not_action_mask():
    obs = _observation(3, trajectories=2)
    obs["customer_unserved"][:] = torch.tensor([[[False, True, False], [False, False, False]]])
    # Customer 2 is temporarily infeasible, but remains part of the task.
    obs["action_mask"][:, :, 2] = False
    wrapped = stateWrapper(obs, "cpu", stable_cost=True)
    encoded = torch.tensor([[[100.0], [200.0], [300.0], [1.0], [2.0], [3.0]]])
    pooled, scalars = remaining_task_summary(encoded, wrapped)
    torch.testing.assert_close(pooled, torch.tensor([[[2.0], [0.0]]]))
    assert torch.isfinite(scalars).all()
    assert scalars[0, 0, 0] > 0 and scalars[0, 1, 0] == 0
    # No remaining customers must not force the predicted return-leg cost to 0.
    agent = Agent(embedding_dim=16, n_encode_layers=1, critic_mode="stable_cost_v1")
    with torch.no_grad():
        agent.critic.popart.linear.bias.fill_(3.0)
    assert agent.get_value(obs)[0, 1, 0].item() == pytest.approx(3.0)


def test_critic_backward_does_not_reach_actor_encoder_or_pointer():
    agent = Agent(embedding_dim=16, n_encode_layers=1, critic_mode="stable_cost_v1")
    obs = _observation(5)
    cached = agent.backbone.encode(obs)
    cached[0].retain_grad()
    outputs = agent.get_critic_outputs_cached(obs, cached)
    loss = F.mse_loss(outputs["cost_normalized"], torch.ones(1, 2))
    loss = loss + F.binary_cross_entropy_with_logits(outputs["failure_logits"], torch.ones(1, 2))
    loss.backward()
    assert all(parameter.grad is None for parameter in agent.backbone.parameters())
    assert cached[0].grad is None
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in agent.critic.adapter.parameters())


def test_zero_initialized_actor_context_preserves_loaded_legacy_logits():
    torch.manual_seed(24)
    legacy = Agent(embedding_dim=16, n_encode_layers=1).eval()
    stable = Agent(embedding_dim=16, n_encode_layers=1, critic_mode="stable_cost_v1").eval()
    actor_state = {key: value for key, value in legacy.state_dict().items() if key.startswith("backbone.")}
    stable.load_state_dict(actor_state, strict=False)
    obs = _observation(7)
    with torch.no_grad():
        old_logits, _ = legacy.backbone(obs)
        new_logits, _ = stable.backbone(obs)
    torch.testing.assert_close(new_logits, old_logits, rtol=0, atol=0)
    legacy_wrapped = stateWrapper(obs, "cpu")
    assert set(legacy_wrapped.states) - {"observations"} <= MODEL_OBSERVATION_KEYS
    assert "customer_unserved" not in legacy_wrapped.states
    # The new residual context receives actor gradients despite initial zero output.
    logp = stable.get_action_and_value_cached(obs, action=torch.zeros(1, 2, dtype=torch.long))[1]
    (-logp.mean()).backward()
    assert stable.backbone.decoder.remaining_context[-1].weight.grad.abs().sum() > 0


def test_popart_preserves_raw_outputs_and_resets_only_head_optimizer_state():
    torch.manual_seed(11)
    trunk = torch.nn.Linear(3, 4).double()
    head = PopArtHead(4, beta=0.5, min_std=0.1).double()
    optimizer = torch.optim.AdamW(list(trunk.parameters()) + list(head.parameters()), lr=0.01)
    x = torch.randn(8, 3, dtype=torch.float64)
    loss = (head(trunk(x)) - 10).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    trunk_moments = copy.deepcopy(optimizer.state[trunk.weight])
    for targets in (torch.tensor([90., 120., 150.]), torch.tensor([700., 800., 900.])):
        old = head(trunk(x)).detach().clone()
        result = head.update_stats(targets, optimizer=optimizer)
        torch.testing.assert_close(head(trunk(x)), old, rtol=1e-10, atol=1e-10)
        assert result["optimizer_output_state"] == "reset"
        assert all(parameter not in optimizer.state for parameter in head.linear.parameters())
    torch.testing.assert_close(optimizer.state[trunk.weight]["exp_avg"], trunk_moments["exp_avg"])
    restored = PopArtHead(4, beta=0.5, min_std=0.1).double()
    restored.load_state_dict(head.state_dict())
    torch.testing.assert_close(restored(trunk(x)), head(trunk(x)))


def test_popart_weights_equalize_trajectories_and_ignore_masked_nan():
    head = PopArtHead(2, beta=0.1, min_std=0.1)
    # One trajectory has 3 states at 0; the other has 1 state at 10.
    targets = torch.tensor([0., 0., 0., 10., float("nan")])
    weights = torch.tensor([1 / 3, 1 / 3, 1 / 3, 1., float("nan")])
    active = torch.tensor([True, True, True, True, False])
    head.update_stats(targets, mask=active, weights=weights)
    assert head.mean.item() == pytest.approx(5.0)
    assert head.std.item() == pytest.approx(5.0)
    assert head.sample_count.item() == 4
    saved = copy.deepcopy(head.state_dict())
    with pytest.raises(ValueError, match="non-finite"):
        head.update_stats(torch.tensor([float("nan")]))
    for key, value in saved.items():
        torch.testing.assert_close(head.state_dict()[key], value)
