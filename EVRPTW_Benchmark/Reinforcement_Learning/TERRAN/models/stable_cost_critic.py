"""Variable-size task context and an isolated cost critic with shared PopArt.

The customer pooling follows CaliRoute's dynamic-context idea, but uses the
actual unserved set instead of the currently feasible action set. All scales
share parameters and one set of target statistics; no customer-count table is
used. Cost predictions remain in the original cost units.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


TASK_SCALAR_DIM = 11


def remaining_task_summary(node_embeddings: torch.Tensor, state):
    """Return [B, trajectories, D] unserved mean and fixed-width task scalars.

    Cached embeddings are in INTERNAL [depot, stations, customers] order.
    ``customer_unserved`` contains customers only, in their original order.
    Matrix multiplication avoids a [B, trajectories, customers, D] expansion.
    """
    n_cus = int(state.states["cus_loc"].size(1))
    n_rs = int(state.states["rs_loc"].size(1))
    required = (
        "customer_unserved", "dispatch_paid", "remaining_step_budget",
        "episode_step_budget",
    )
    missing = [key for key in required if key not in state.states]
    if missing:
        raise ValueError(f"stable_cost_v1 observation missing fields: {missing}")
    unserved = state.states["customer_unserved"].bool()
    if unserved.ndim != 3 or unserved.shape[-1] != n_cus:
        raise ValueError("customer_unserved must have shape [batch, trajectories, customers]")
    customers = node_embeddings[:, 1 + n_rs : 1 + n_rs + n_cus]
    if customers.shape[:2] != (unserved.shape[0], n_cus):
        raise ValueError("cached customer embeddings do not match customer_unserved")
    included = unserved.to(dtype=node_embeddings.dtype)
    count = included.sum(-1)
    pooled = torch.bmm(included, customers) / count.clamp_min(1.0).unsqueeze(-1)
    # The original external demand is already expressed in vehicle capacities.
    demand = state.states["demand"][:, 1 : 1 + n_cus].to(node_embeddings.dtype)
    if demand.ndim == 3:
        demand = demand.squeeze(-1)
    remaining_demand = torch.bmm(included, demand.unsqueeze(-1)).squeeze(-1)
    total_demand = demand.sum(-1, keepdim=True).clamp_min(1e-8)

    def scalar(value, name):
        value = value.to(dtype=node_embeddings.dtype, device=node_embeddings.device)
        if value.shape != count.shape:
            raise ValueError(f"{name} must have shape {tuple(count.shape)}, got {tuple(value.shape)}")
        return value

    budget = scalar(state.states["remaining_step_budget"], "remaining_step_budget").clamp_min(0)
    episode_budget = scalar(state.states["episode_step_budget"], "episode_step_budget").clamp_min(1)
    features = torch.stack(
        (
            torch.log1p(count),
            count / max(n_cus, 1),
            torch.log1p(remaining_demand.clamp_min(0)),
            remaining_demand / total_demand,
            scalar(state.used_capacity, "current_load"),
            scalar(state.used_battery, "current_battery"),
            scalar(state.current_time, "current_time"),
            scalar(state.states["dispatch_paid"], "dispatch_paid"),
            budget / episode_budget,
            torch.log1p(budget),
            (state.get_current_node() == 0).to(node_embeddings.dtype),
        ),
        dim=-1,
    )
    return pooled, features


class PopArtHead(nn.Module):
    """One affine value normalization shared by all problem sizes.

    Statistics are running target moments, not gradient-trained parameters.
    Updating them rescales the final layer to preserve raw predictions. Adam
    state for that layer is reset after a coordinate change; trunk optimizer
    state is retained. This explicit policy avoids reusing moments expressed in
    the old output coordinates without claiming optimizer-trajectory invariance.
    """

    def __init__(self, input_dim: int, *, beta: float = 0.01, min_std: float = 1.0):
        super().__init__()
        if not math.isfinite(beta) or not 0.0 < beta <= 1.0:
            raise ValueError("PopArt beta must be finite and in (0, 1]")
        if not math.isfinite(min_std) or min_std <= 0:
            raise ValueError("PopArt min_std must be finite and positive")
        self.beta = float(beta)
        self.min_std = float(min_std)
        self.linear = nn.Linear(input_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.register_buffer("mean", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("second_moment", torch.tensor(min_std ** 2, dtype=torch.float64))
        self.register_buffer("std", torch.tensor(min_std, dtype=torch.float64))
        self.register_buffer("sample_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("update_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_weight", torch.tensor(0.0, dtype=torch.float64))

    def normalize(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets - self.mean.to(targets)) / self.std.to(targets)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std.to(values) + self.mean.to(values)

    def normalized(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.denormalize(self.normalized(features))

    @torch.no_grad()
    def update_stats(
        self,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        weights: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Update once from a logical batch, including concatenated CPU targets.

        The caller freezes these statistics during all PPO reuse of that batch.
        Non-finite padding is allowed only outside ``mask``; active non-finite
        targets fail before any state or parameters are changed.
        """
        data = targets.detach()
        if weights is not None and weights.shape != data.shape:
            raise ValueError("PopArt target and weight shapes must match")
        weight = torch.ones_like(data, dtype=torch.float64) if weights is None else weights.detach().to(device=data.device, dtype=torch.float64)
        if mask is not None:
            if mask.shape != data.shape:
                raise ValueError("PopArt target and valid mask shapes must match")
            active = mask.to(device=data.device, dtype=torch.bool)
            data = data[active]
            weight = weight[active]
        data = data.reshape(-1).to(dtype=torch.float64)
        weight = weight.reshape(-1)
        count = int(data.numel())
        if not count:
            return {"updated": False, "sample_count": int(self.sample_count.item())}
        if not bool(torch.isfinite(data).all().item()):
            raise ValueError("PopArt received non-finite active targets")
        if not bool((torch.isfinite(weight) & (weight >= 0)).all().item()):
            raise ValueError("PopArt weights must be finite and non-negative")
        mass = weight.sum()
        if not bool(torch.isfinite(mass).item()):
            raise ValueError("PopArt total weight is non-finite")
        if not bool((mass > 0).item()):
            return {"updated": False, "sample_count": int(self.sample_count.item())}
        batch_mean = (weight * data).sum().div(mass).to(self.mean)
        batch_second = (weight * data.square()).sum().div(mass).to(self.second_moment)
        if not bool(torch.isfinite(batch_second).item()):
            raise ValueError("PopArt target second moment is non-finite")
        rate = 1.0 if int(self.update_count.item()) == 0 else self.beta
        new_mean = torch.lerp(self.mean, batch_mean, rate)
        new_second = torch.lerp(self.second_moment, batch_second, rate)
        new_std = (new_second - new_mean.square()).clamp_min(self.min_std ** 2).sqrt()
        ratio = self.std / new_std
        shift = (self.mean - new_mean) / new_std
        self.linear.weight.mul_(ratio.to(self.linear.weight))
        self.linear.bias.mul_(ratio.to(self.linear.bias)).add_(shift.to(self.linear.bias))
        self.mean.copy_(new_mean)
        self.second_moment.copy_(new_second)
        self.std.copy_(new_std)
        self.sample_count.add_(count)
        self.update_count.add_(1)
        self.total_weight.add_(mass.to(self.total_weight))
        if optimizer is not None:
            for parameter in self.linear.parameters():
                optimizer.state.pop(parameter, None)
        return {
            "updated": True,
            "mean": float(self.mean.item()),
            "std": float(self.std.item()),
            "sample_count": int(self.sample_count.item()),
            "update_count": int(self.update_count.item()),
            "batch_weight": float(mass.item()),
            "optimizer_output_state": "reset" if optimizer is not None else "not_provided",
        }


class StableCostCritic(nn.Module):
    """Independent adapter over detached encoder features and true task state."""

    def __init__(self, hidden_size: int, *, popart_beta: float = 0.01, popart_min_std: float = 1.0):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(3 * hidden_size + TASK_SCALAR_DIM, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        for layer in self.adapter:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(layer.bias)
        self.popart = PopArtHead(hidden_size, beta=popart_beta, min_std=popart_min_std)
        self.failure_head = nn.Linear(hidden_size, 1)
        nn.init.orthogonal_(self.failure_head.weight, gain=0.01)
        nn.init.zeros_(self.failure_head.bias)

    def forward(self, node_embeddings: torch.Tensor, state) -> dict[str, torch.Tensor]:
        encoded = node_embeddings.detach()
        pooled, scalars = remaining_task_summary(encoded, state)
        current_index = state.get_current_node()
        current = encoded.gather(1, current_index.unsqueeze(-1).expand(-1, -1, encoded.size(-1)))
        depot = encoded[:, :1].expand(-1, current_index.size(1), -1)
        features = self.adapter(torch.cat((pooled, current, depot, scalars.detach()), dim=-1))
        normalized = self.popart.normalized(features).squeeze(-1)
        return {
            "cost_value": self.popart.denormalize(normalized),
            "cost_normalized": normalized,
            "failure_logits": self.failure_head(features).squeeze(-1),
        }
