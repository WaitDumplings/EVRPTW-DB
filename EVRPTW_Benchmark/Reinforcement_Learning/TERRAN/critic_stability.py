"""Small, training-safe helpers for TERRAN critic stability diagnostics.

The helpers in this module deliberately do not change reward, return, or value
units.  They only make the value-loss contract explicit and collect detached
statistics from tensors that the PPO update already computes.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


VALUE_LOSS_TYPES = frozenset({"mse", "smooth_l1"})


def resolve_critic_stability_config(training: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and resolve the critic controls that affect learned parameters."""

    loss_type = str(training.get("value_loss_type", "mse")).strip().lower()
    if loss_type not in VALUE_LOSS_TYPES:
        raise ValueError(
            "training.value_loss_type must be one of "
            f"{sorted(VALUE_LOSS_TYPES)}"
        )
    beta = float(training.get("value_loss_beta", 1.0))
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("training.value_loss_beta must be finite and positive")
    residual_scale = float(training.get("value_residual_scale", 1.0))
    if not math.isfinite(residual_scale) or residual_scale <= 0.0:
        raise ValueError(
            "training.value_residual_scale must be finite and positive"
        )
    backbone_grad_scale = float(
        training.get("critic_backbone_grad_scale", 1.0)
    )
    if not math.isfinite(backbone_grad_scale) or backbone_grad_scale < 0.0:
        raise ValueError(
            "training.critic_backbone_grad_scale must be finite and non-negative"
        )
    diagnostics_every = int(
        training.get("critic_gradient_diagnostics_every_epochs", 0) or 0
    )
    if diagnostics_every < 0:
        raise ValueError(
            "training.critic_gradient_diagnostics_every_epochs must be non-negative"
        )
    return {
        "value_loss_type": loss_type,
        "value_loss_beta": beta,
        "value_residual_scale": residual_scale,
        "critic_backbone_grad_scale": backbone_grad_scale,
        "critic_gradient_diagnostics_every_epochs": diagnostics_every,
    }


def value_loss_per_transition(
    value: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str,
    beta: float,
    residual_scale: float,
) -> torch.Tensor:
    """Return an unreduced value loss while preserving V/G/A output units.

    ``residual_scale`` scales only the loss residual.  The model prediction,
    return target, and advantage remain in normalized-objective reward units.
    """

    residual = (value - target) / float(residual_scale)
    if loss_type == "mse":
        return residual.square()
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(
            residual,
            torch.zeros_like(residual),
            beta=float(beta),
            reduction="none",
        )
    raise ValueError(f"unsupported value loss type: {loss_type!r}")


class _StreamingMoments:
    """Exact finite moments for detached tensors without retaining samples."""

    def __init__(self) -> None:
        self.count = 0
        self.finite_count = 0
        self.nonfinite_count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, values: torch.Tensor) -> None:
        data = values.detach().reshape(-1)
        self.count += int(data.numel())
        if not data.numel():
            return
        finite = data[torch.isfinite(data)].double()
        finite_count = int(finite.numel())
        self.finite_count += finite_count
        self.nonfinite_count += int(data.numel()) - finite_count
        if not finite_count:
            return
        self.total += float(finite.sum().cpu())
        self.total_square += float(finite.square().sum().cpu())
        self.minimum = min(self.minimum, float(finite.min().cpu()))
        self.maximum = max(self.maximum, float(finite.max().cpu()))

    def merge(
        self,
        *,
        count: int,
        finite_count: int,
        total: float,
        total_square: float,
        minimum: float,
        maximum: float,
    ) -> None:
        """Merge already-reduced detached statistics into this accumulator."""

        self.count += int(count)
        self.finite_count += int(finite_count)
        self.nonfinite_count += int(count) - int(finite_count)
        if not finite_count:
            return
        self.total += float(total)
        self.total_square += float(total_square)
        self.minimum = min(self.minimum, float(minimum))
        self.maximum = max(self.maximum, float(maximum))

    def summary(self) -> dict[str, Any]:
        if not self.finite_count:
            return {
                "count": self.count,
                "finite_count": 0,
                "nonfinite_count": self.nonfinite_count,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        mean = self.total / self.finite_count
        variance = max(self.total_square / self.finite_count - mean * mean, 0.0)
        return {
            "count": self.count,
            "finite_count": self.finite_count,
            "nonfinite_count": self.nonfinite_count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


class PPODiagnosticsAccumulator:
    """Detached PPO ratio/KL and value-error diagnostics over valid transitions."""

    def __init__(self) -> None:
        self.ratio = _StreamingMoments()
        self.log_ratio = _StreamingMoments()
        self.approx_kl = _StreamingMoments()
        self.raw_squared_value_residual = _StreamingMoments()
        self.configured_value_loss = _StreamingMoments()
        self.clipped_count = 0
        self.valid_count = 0
        self.clip_coefficients: set[float] = set()

    @torch.no_grad()
    def update(
        self,
        *,
        log_ratio: torch.Tensor,
        value_residual: torch.Tensor,
        value_loss: torch.Tensor,
        valid: torch.Tensor,
        clip_coef: float,
    ) -> None:
        active = valid.bool()
        active_log_ratio = log_ratio[active].detach()
        active_ratio = torch.exp(active_log_ratio)
        active_residual = value_residual[active].detach()
        active_value_loss = value_loss[active].detach()
        approx_kl = (active_ratio - 1.0) - active_log_ratio
        count = int(active_log_ratio.numel())
        self.valid_count += count
        self.clip_coefficients.add(float(clip_coef))
        if not count:
            return

        # This collector is called once per PPO step chunk.  Reduce every
        # metric on device and transfer one small packed tensor so diagnostics
        # add only one GPU-to-CPU synchronization per chunk.
        metric_values = (
            (self.ratio, active_ratio),
            (self.log_ratio, active_log_ratio),
            (self.approx_kl, approx_kl),
            (self.raw_squared_value_residual, active_residual.square()),
            (self.configured_value_loss, active_value_loss),
        )
        packed_rows = []
        for _, values in metric_values:
            values = values.reshape(-1).double()
            finite = torch.isfinite(values)
            zero = torch.zeros((), dtype=values.dtype, device=values.device)
            cleaned = torch.where(finite, values, zero)
            packed_rows.append(
                torch.stack(
                    (
                        finite.sum(dtype=values.dtype),
                        cleaned.sum(),
                        cleaned.square().sum(),
                        torch.where(finite, values, math.inf).min(),
                        torch.where(finite, values, -math.inf).max(),
                    )
                )
            )
        clipped = ((active_ratio - 1.0).abs() > float(clip_coef)).sum(
            dtype=torch.float64
        )
        packed = torch.cat((torch.stack(packed_rows).reshape(-1), clipped[None]))
        reduced = packed.cpu().tolist()
        for index, (moments, _) in enumerate(metric_values):
            finite_count, total, total_square, minimum, maximum = reduced[
                index * 5 : (index + 1) * 5
            ]
            moments.merge(
                count=count,
                finite_count=int(finite_count),
                total=total,
                total_square=total_square,
                minimum=minimum,
                maximum=maximum,
            )
        self.clipped_count += int(reduced[-1])

    def summary(self) -> dict[str, Any]:
        clip_coef = (
            next(iter(self.clip_coefficients))
            if len(self.clip_coefficients) == 1
            else None
        )
        approx_kl = self.approx_kl.summary()
        raw_mse = self.raw_squared_value_residual.summary()
        configured = self.configured_value_loss.summary()
        return {
            "population": "all valid transitions evaluated across PPO updates",
            "valid_transition_evaluations": self.valid_count,
            "ratio": self.ratio.summary(),
            "log_ratio": self.log_ratio.summary(),
            "approx_kl": approx_kl,
            "approx_kl_mean": approx_kl["mean"],
            "ppo_clip_coef": clip_coef,
            "ppo_clip_fraction": (
                self.clipped_count / self.valid_count if self.valid_count else None
            ),
            "ppo_clipped_transition_evaluations": self.clipped_count,
            "raw_value_mse": raw_mse["mean"],
            "raw_squared_value_residual": raw_mse,
            "configured_value_loss_mean": configured["mean"],
            "configured_value_loss": configured,
        }


def should_capture_critic_gradients(
    *, epoch: int, start_epoch: int, every_epochs: int,
) -> bool:
    """Capture on the first epoch of this session and fixed logical epochs."""

    interval = int(every_epochs)
    return interval > 0 and (
        int(epoch) == int(start_epoch) or int(epoch) % interval == 0
    )


def _parameter_tuple(module: torch.nn.Module) -> tuple[torch.nn.Parameter, ...]:
    return tuple(parameter for parameter in module.parameters() if parameter.requires_grad)


class CriticGradientAccumulator:
    """Accumulate exact component gradients for one normal optimizer update.

    ``torch.autograd.grad`` returns gradients without modifying ``parameter.grad``.
    Detached CPU float64 accumulators keep the occasional diagnostic from adding
    persistent GPU memory.  Chunk objectives must already carry the exact same
    valid-transition weights used by the normal backward pass.
    """

    def __init__(self, agent: torch.nn.Module) -> None:
        self.shared_parameters = _parameter_tuple(agent.backbone)
        self.critic_parameters = _parameter_tuple(agent.critic)
        self.policy_shared: list[torch.Tensor | None] = [
            None for _ in self.shared_parameters
        ]
        self.value_shared: list[torch.Tensor | None] = [
            None for _ in self.shared_parameters
        ]
        self.value_critic: list[torch.Tensor | None] = [
            None for _ in self.critic_parameters
        ]
        self.transition_count = 0
        self.chunk_count = 0

    @staticmethod
    def _add(
        destination: list[torch.Tensor | None],
        gradients: Sequence[torch.Tensor | None],
    ) -> None:
        for index, gradient in enumerate(gradients):
            if gradient is None:
                continue
            detached = gradient.detach().to(device="cpu", dtype=torch.float64)
            if destination[index] is None:
                destination[index] = detached.clone()
            else:
                destination[index].add_(detached)

    def accumulate(
        self,
        *,
        policy_objective: torch.Tensor,
        weighted_value_objective: torch.Tensor,
        transition_count: int,
    ) -> None:
        policy_gradients = torch.autograd.grad(
            policy_objective,
            self.shared_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        value_gradients = torch.autograd.grad(
            weighted_value_objective,
            self.shared_parameters + self.critic_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        split = len(self.shared_parameters)
        self._add(self.policy_shared, policy_gradients)
        self._add(self.value_shared, value_gradients[:split])
        self._add(self.value_critic, value_gradients[split:])
        self.transition_count += int(transition_count)
        self.chunk_count += 1

    @staticmethod
    def _squared_norm(parts: Sequence[torch.Tensor | None]) -> float:
        return sum(
            float(part.square().sum()) for part in parts if part is not None
        )

    @staticmethod
    def _dot(
        left: Sequence[torch.Tensor | None],
        right: Sequence[torch.Tensor | None],
    ) -> float:
        return sum(
            float(a.mul(b).sum())
            for a, b in zip(left, right)
            if a is not None and b is not None
        )

    def summary(self) -> dict[str, Any]:
        policy_sq = self._squared_norm(self.policy_shared)
        value_shared_sq = self._squared_norm(self.value_shared)
        value_critic_sq = self._squared_norm(self.value_critic)
        policy_norm = math.sqrt(max(policy_sq, 0.0))
        value_shared_norm = math.sqrt(max(value_shared_sq, 0.0))
        dot = self._dot(self.policy_shared, self.value_shared)
        cosine = (
            dot / (policy_norm * value_shared_norm)
            if policy_norm > 0.0 and value_shared_norm > 0.0
            else None
        )
        combined = [
            (a if b is None else b)
            if a is None
            else (a if b is None else a + b)
            for a, b in zip(self.policy_shared, self.value_shared)
        ]
        return {
            "population": "first optimizer update of one frozen rollout",
            "valid_transition_count": self.transition_count,
            "step_chunk_count": self.chunk_count,
            "policy_shared_grad_norm": policy_norm,
            "weighted_value_shared_grad_norm": value_shared_norm,
            "weighted_value_critic_head_grad_norm": math.sqrt(
                max(value_critic_sq, 0.0)
            ),
            "policy_value_shared_grad_dot": dot,
            "policy_value_shared_grad_cosine": cosine,
            "combined_policy_value_shared_grad_norm": math.sqrt(
                max(self._squared_norm(combined), 0.0)
            ),
            "policy_component_includes_entropy": False,
            "normal_parameter_grad_buffers_modified": False,
            "accumulation": "exact sum of valid-transition-weighted chunk gradients",
        }


__all__ = [
    "CriticGradientAccumulator",
    "PPODiagnosticsAccumulator",
    "resolve_critic_stability_config",
    "should_capture_critic_gradients",
    "value_loss_per_transition",
]
