from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.critic_stability import (
    CriticGradientAccumulator,
    PPODiagnosticsAccumulator,
    should_capture_critic_gradients,
    value_loss_per_transition,
)
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.models.attention_model_wrapper import (
    Critic,
)
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.trainer import (
    build_critic_value_diagnostics,
    evaluate_policy_loss,
)


def test_smooth_l1_matches_huber_and_bounds_an_outlier_gradient() -> None:
    value = torch.tensor([0.0, 0.0], requires_grad=True)
    target = torch.tensor([0.25, 1_000.0])

    loss = value_loss_per_transition(
        value,
        target,
        loss_type="smooth_l1",
        beta=1.0,
        residual_scale=1.0,
    )
    expected = F.smooth_l1_loss(
        value,
        target,
        beta=1.0,
        reduction="none",
    )
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    assert loss.tolist() == pytest.approx([0.03125, 999.5])

    loss.sum().backward()
    assert value.grad is not None
    # The inlier remains quadratic while the extreme residual has unit slope.
    torch.testing.assert_close(
        value.grad,
        torch.tensor([-0.25, -1.0]),
        rtol=0,
        atol=0,
    )


def test_critic_gradient_scale_changes_only_its_backbone_input_gradient() -> None:
    torch.manual_seed(90210)
    critic = Critic(hidden_size=8)
    features = torch.randn(2, 3, 8)
    upstream = torch.randn(2, 3, 1)

    def run(scale: float):
        critic.zero_grad(set_to_none=True)
        current = features.detach().clone().requires_grad_(True)
        critic.backbone_grad_scale = scale
        output = critic((None, current))
        (output * upstream).sum().backward()
        assert current.grad is not None
        parameter_gradients = [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in critic.parameters()
        ]
        return output.detach().clone(), current.grad.detach().clone(), parameter_gradients

    default_output, default_input_grad, default_head_grads = run(1.0)
    scaled_output, scaled_input_grad, scaled_head_grads = run(0.1)

    # Gradient scaling must not change the value function or critic-head update.
    torch.testing.assert_close(scaled_output, default_output, rtol=0, atol=0)
    assert any(
        gradient is not None and torch.count_nonzero(gradient)
        for gradient in default_head_grads
    )
    for default, scaled in zip(default_head_grads, scaled_head_grads):
        if default is None:
            assert scaled is None
        else:
            assert scaled is not None
            torch.testing.assert_close(scaled, default, rtol=0, atol=0)
    torch.testing.assert_close(
        scaled_input_grad,
        default_input_grad * 0.1,
        rtol=1e-6,
        atol=1e-12,
    )


def test_ppo_diagnostics_exclude_nonfinite_padding_but_report_active_nonfinite() -> None:
    diagnostics = PPODiagnosticsAccumulator()
    diagnostics.update(
        log_ratio=torch.tensor([0.0, float("nan"), torch.log(torch.tensor(1.25)), float("inf")]),
        value_residual=torch.tensor([1.0, float("nan"), -2.0, float("inf")]),
        value_loss=torch.tensor([0.5, float("nan"), 1.5, float("inf")]),
        valid=torch.tensor([True, False, True, False]),
        clip_coef=0.2,
    )
    summary = diagnostics.summary()

    assert summary["valid_transition_evaluations"] == 2
    assert summary["ratio"]["count"] == 2
    assert summary["ratio"]["nonfinite_count"] == 0
    assert summary["ppo_clipped_transition_evaluations"] == 1
    assert summary["ppo_clip_fraction"] == pytest.approx(0.5)
    assert summary["raw_value_mse"] == pytest.approx(2.5)
    assert summary["configured_value_loss_mean"] == pytest.approx(1.0)

    diagnostics.update(
        log_ratio=torch.tensor([float("nan")]),
        value_residual=torch.tensor([float("inf")]),
        value_loss=torch.tensor([float("nan")]),
        valid=torch.tensor([True]),
        clip_coef=0.2,
    )
    summary = diagnostics.summary()
    assert summary["valid_transition_evaluations"] == 3
    assert summary["ratio"]["count"] == 3
    assert summary["ratio"]["nonfinite_count"] == 1
    assert summary["raw_squared_value_residual"]["nonfinite_count"] == 1
    assert summary["configured_value_loss"]["nonfinite_count"] == 1


def test_gradient_diagnostics_zero_disables_capture_and_positive_cadence_is_stable() -> None:
    # The default must be a true no-op for existing configs and non-standard
    # test agents that do not expose TERRAN's backbone/critic split.
    assert not should_capture_critic_gradients(
        epoch=7, start_epoch=7, every_epochs=0
    )
    assert not should_capture_critic_gradients(
        epoch=25, start_epoch=7, every_epochs=0
    )

    # A resumed session captures its first epoch, then fixed logical epochs.
    assert should_capture_critic_gradients(
        epoch=7, start_epoch=7, every_epochs=25
    )
    assert not should_capture_critic_gradients(
        epoch=8, start_epoch=7, every_epochs=25
    )
    assert should_capture_critic_gradients(
        epoch=25, start_epoch=7, every_epochs=25
    )


class _LossProbeBackbone(torch.nn.Module):
    device = torch.device("cpu")

    def encode(self, _observation):
        return None


class _LossProbeAgent(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _LossProbeBackbone()
        self.value = torch.nn.Parameter(torch.tensor(0.0))

    def get_action_and_value_cached(self, observation, action, state):
        del observation, state
        shape = tuple(action.shape)
        zeros = torch.zeros(shape)
        value = self.value.expand(*shape, 1)
        return action, zeros, zeros, value, None


def test_nonfinite_padded_target_cannot_leak_into_value_gradient() -> None:
    agent = _LossProbeAgent()
    batch = SimpleNamespace(
        observations=[{}],
        actions=torch.zeros((1, 1, 2), dtype=torch.long),
        old_logprobs=torch.zeros((1, 1, 2)),
        valid=torch.tensor([[[True, False]]]),
    )
    returns = torch.tensor([[[1.0, float("nan")]]])
    advantages = torch.zeros_like(returns)
    cfg = {
        "training": {
            "clip_coef": 0.2,
            "vf_coef": 0.1,
            "ent_coef": 0.0,
            "value_loss_type": "smooth_l1",
            "value_loss_beta": 1.0,
            "value_residual_scale": 1.0,
        }
    }

    total, _, value_loss, _ = evaluate_policy_loss(
        agent,
        batch,
        returns,
        advantages,
        cfg,
        "cpu",
    )
    assert value_loss.item() == pytest.approx(0.5)
    total.backward()
    assert agent.value.grad is not None
    assert torch.isfinite(agent.value.grad)
    assert agent.value.grad.item() == pytest.approx(-0.1)


def test_value_diagnostics_group_valid_transitions_and_ignore_padding() -> None:
    valid = torch.tensor(
        [
            [[True, True], [True, False]],
            [[True, False], [True, False]],
        ]
    )
    values = torch.tensor(
        [
            [[1.0, 2.0], [3.0, float("nan")]],
            [[4.0, float("nan")], [5.0, float("nan")]],
        ]
    )
    returns = torch.tensor(
        [
            [[2.0, 4.0], [6.0, float("nan")]],
            [[8.0, float("nan")], [10.0, float("nan")]],
        ]
    )
    batch = SimpleNamespace(
        valid=valid,
        values=values,
        rollout_budget_exhausted=torch.tensor(
            [[False, True], [False, False]]
        ),
        final_infos=[
            {"success": [True, False]},
            {"success": [False, False]},
        ],
    )
    critic_config = {
        "value_loss_type": "smooth_l1",
        "value_loss_beta": 1.0,
        "value_residual_scale": 1.0,
        "critic_backbone_grad_scale": 0.1,
        "critic_gradient_diagnostics_every_epochs": 25,
    }

    result = build_critic_value_diagnostics(
        [(batch, returns, returns - values)], critic_config
    )
    populations = result["populations"]
    expected = {
        "overall": (3, 5, 6.0, 3.0, 3.0, 11.0, 2.5),
        "success": (1, 2, 5.0, 2.5, 2.5, 8.5, 2.0),
        "rollout_budget_exhausted": (1, 1, 4.0, 2.0, 2.0, 4.0, 1.5),
        "other_terminal": (1, 2, 8.0, 4.0, 4.0, 17.0, 3.5),
    }
    for name, (
        trajectories,
        transitions,
        target_mean,
        prediction_mean,
        residual_mean,
        raw_mse,
        configured_loss,
    ) in expected.items():
        row = populations[name]
        assert row["trajectory_count"] == trajectories
        assert row["valid_transition_count"] == transitions
        assert row["return_target"]["mean"] == pytest.approx(target_mean)
        assert row["rollout_value_prediction"]["mean"] == pytest.approx(
            prediction_mean
        )
        assert row["value_residual_g_minus_v"]["mean"] == pytest.approx(
            residual_mean
        )
        assert row["raw_value_mse"] == pytest.approx(raw_mse)
        assert row["configured_value_loss"] == pytest.approx(configured_loss)
        for field in (
            "return_target",
            "rollout_value_prediction",
            "value_residual_g_minus_v",
            "raw_squared_value_residual",
        ):
            assert row[field]["count"] == transitions
            assert row[field]["nonfinite_count"] == 0


class _TinyActorCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2, bias=False)
        self.critic = torch.nn.Linear(2, 1, bias=False)


def _diagnostic_objectives(
    agent: _TinyActorCritic,
    inputs: torch.Tensor,
    policy_weights: torch.Tensor,
    value_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = agent.backbone(inputs)
    policy = (features * policy_weights).sum(dim=-1).mean()
    value = agent.critic(features).squeeze(-1)
    weighted_value = 0.1 * F.smooth_l1_loss(
        value,
        value_targets,
        beta=1.0,
        reduction="mean",
    )
    return policy, weighted_value


def test_chunked_critic_gradient_diagnostics_equal_one_full_objective() -> None:
    torch.manual_seed(1701)
    inputs = torch.randn(5, 2)
    policy_weights = torch.randn(5, 2)
    targets = torch.randn(5)

    full_agent = _TinyActorCritic()
    chunked_agent = _TinyActorCritic()
    chunked_agent.load_state_dict(full_agent.state_dict())

    full = CriticGradientAccumulator(full_agent)
    full_policy, full_value = _diagnostic_objectives(
        full_agent, inputs, policy_weights, targets
    )
    full.accumulate(
        policy_objective=full_policy,
        weighted_value_objective=full_value,
        transition_count=5,
    )

    chunked = CriticGradientAccumulator(chunked_agent)
    for start, end in ((0, 2), (2, 5)):
        fraction = (end - start) / 5.0
        policy, value = _diagnostic_objectives(
            chunked_agent,
            inputs[start:end],
            policy_weights[start:end],
            targets[start:end],
        )
        # Training weights each chunk mean by its share of valid transitions.
        chunked.accumulate(
            policy_objective=policy * fraction,
            weighted_value_objective=value * fraction,
            transition_count=end - start,
        )

    full_summary = full.summary()
    chunked_summary = chunked.summary()
    assert chunked_summary["valid_transition_count"] == 5
    assert chunked_summary["step_chunk_count"] == 2
    for field in (
        "policy_shared_grad_norm",
        "weighted_value_shared_grad_norm",
        "weighted_value_critic_head_grad_norm",
        "policy_value_shared_grad_dot",
        "policy_value_shared_grad_cosine",
        "combined_policy_value_shared_grad_norm",
    ):
        assert chunked_summary[field] == pytest.approx(
            full_summary[field], rel=1e-6, abs=1e-12
        )
    assert all(parameter.grad is None for parameter in chunked_agent.parameters())
