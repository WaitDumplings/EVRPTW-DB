from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(np.asarray(value), device=device)


class SequenceInstanceNorm(nn.Module):
    """InstanceNorm over the terminal dimension, matching upstream RRNCO."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(value.transpose(1, 2)).transpose(1, 2)


class ContextualGate(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * embedding_dim, 2 * embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * embedding_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, coordinate: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        weight = self.network(torch.cat((coordinate, relation), dim=-1))
        return weight * coordinate + (1.0 - weight) * relation


class DirectedDistanceExpert(nn.Module):
    """Separate sampled outgoing/incoming road-distance summaries (RRNCO ANE)."""

    def __init__(self, embedding_dim: int, sample_size: int = 25) -> None:
        super().__init__()
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        self.sample_size = int(sample_size)
        self.row = nn.Linear(self.sample_size, embedding_dim)
        self.col = nn.Linear(self.sample_size, embedding_dim)

    def _indices(self, distance: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = distance.shape
        adjusted = distance.clone()
        diagonal = torch.arange(nodes, device=distance.device)
        adjusted[:, diagonal, diagonal] = torch.finfo(distance.dtype).max
        probabilities = adjusted.clamp_min(1e-6).reciprocal()
        probabilities[:, diagonal, diagonal] = 0.0
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sampled = torch.multinomial(
            probabilities.reshape(batch * nodes, nodes),
            self.sample_size,
            replacement=self.sample_size > max(nodes - 1, 1),
        )
        return sampled.reshape(batch, nodes, self.sample_size)

    def forward(self, distance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self._indices(distance)
        outgoing = distance.gather(2, indices).sort(dim=-1).values
        incoming = distance.transpose(1, 2).gather(2, indices).sort(dim=-1).values
        return self.row(outgoing), self.col(incoming)


class RRNCOInitialEmbedding(nn.Module):
    """Adaptive node embedding with EVRPTW attributes and directed road relations."""

    def __init__(self, embedding_dim: int, sample_size: int) -> None:
        super().__init__()
        self.coordinate_depot = nn.Linear(2, embedding_dim)
        self.coordinate_terminal = nn.Linear(3, embedding_dim)
        self.distance_expert = DirectedDistanceExpert(embedding_dim, sample_size)
        self.row_gate = ContextualGate(embedding_dim)
        self.col_gate = ContextualGate(embedding_dim)
        self.attributes = nn.Linear(10, embedding_dim)
        self.row_combine = nn.Linear(2 * embedding_dim, embedding_dim)
        self.col_combine = nn.Linear(2 * embedding_dim, embedding_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        coordinates: torch.Tensor,
        distance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        depot = coordinates[:, :1]
        terminals = coordinates[:, 1:]
        angle = torch.atan2(
            terminals[..., 1] - depot[..., 1],
            terminals[..., 0] - depot[..., 0],
        )[..., None]
        coordinate = torch.cat(
            (
                self.coordinate_depot(depot),
                self.coordinate_terminal(torch.cat((terminals, angle), -1)),
            ),
            dim=1,
        )
        outgoing, incoming = self.distance_expert(distance)
        attributes = self.attributes(node_features)
        row = self.row_combine(torch.cat((self.row_gate(coordinate, outgoing), attributes), -1))
        col = self.col_combine(torch.cat((self.col_gate(coordinate, incoming), attributes), -1))
        return row, col


class RelationBiasFusion(nn.Module):
    """RRNCO neural adaptive bias extended from D/T/angle to D/T/E/angle."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        def expert() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(1, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )

        self.distance = expert()
        self.duration = expert()
        self.energy = expert()
        self.angle = expert()
        self.gate = nn.Sequential(
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, 4),
        )
        self.temperature = nn.Parameter(torch.tensor(5.0))
        self.output = nn.Linear(embedding_dim, 1)

    def forward(
        self,
        coordinates: torch.Tensor,
        distance: torch.Tensor,
        duration: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        delta = coordinates[:, :, None, :] - coordinates[:, None, :, :]
        angle = torch.atan2(delta[..., 1], delta[..., 0]) / math.pi
        channels = (
            self.distance(distance[..., None]),
            self.duration(duration[..., None]),
            self.energy(energy[..., None]),
            self.angle(angle[..., None]),
        )
        gate = torch.softmax(
            self.gate(torch.cat(channels, dim=-1)) / self.temperature.exp(), dim=-1
        )
        fused = sum(gate[..., index, None] * value for index, value in enumerate(channels))
        return self.output(fused).squeeze(-1)


class AFTFull(nn.Module):
    """Attention-free token mixing used by the released RRNCO implementation."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.project = nn.Linear(embedding_dim, embedding_dim)

    def forward(
        self, row: torch.Tensor, col: torch.Tensor, adaptive_bias: torch.Tensor
    ) -> torch.Tensor:
        query = torch.sigmoid(self.query(row))
        key = torch.softmax(self.key(col), dim=1)
        value = self.value(col)
        # Preserve the public implementation's softmax-then-exp semantics.
        relation = torch.exp(torch.softmax(adaptive_bias, dim=-1))
        exp_key = torch.exp(key)
        numerator = torch.bmm(relation, exp_key * value)
        denominator = torch.bmm(relation, exp_key).clamp_min(1e-12)
        return self.project(query * (numerator / denominator))


class RRNCOBlock(nn.Module):
    def __init__(self, embedding_dim: int, feedforward_hidden: int) -> None:
        super().__init__()
        self.row_norm = SequenceInstanceNorm(embedding_dim)
        self.col_norm = SequenceInstanceNorm(embedding_dim)
        self.mix_norm = SequenceInstanceNorm(embedding_dim)
        self.feed_forward_norm = SequenceInstanceNorm(embedding_dim)
        self.bias = RelationBiasFusion(embedding_dim)
        self.aft = AFTFull(embedding_dim)
        self.combine = nn.Linear(embedding_dim, embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_hidden),
            nn.ReLU(),
            nn.Linear(feedforward_hidden, embedding_dim),
        )

    def forward(
        self,
        row: torch.Tensor,
        col: torch.Tensor,
        distance: torch.Tensor,
        duration: torch.Tensor,
        energy: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        normalized_row = self.row_norm(row)
        normalized_col = self.col_norm(col)
        bias = self.bias(coordinates, distance, duration, energy)
        residual = self.mix_norm(row + self.combine(
            self.aft(normalized_row, normalized_col, bias)
        ))
        return self.feed_forward_norm(residual + self.feed_forward(residual))


class RRNCOLayer(nn.Module):
    def __init__(self, embedding_dim: int, feedforward_hidden: int) -> None:
        super().__init__()
        self.row = RRNCOBlock(embedding_dim, feedforward_hidden)
        self.col = RRNCOBlock(embedding_dim, feedforward_hidden)

    def forward(
        self,
        row: torch.Tensor,
        col: torch.Tensor,
        distance: torch.Tensor,
        duration: torch.Tensor,
        energy: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_row = self.row(row, col, distance, duration, energy, coordinates)
        next_col = self.col(
            col,
            row,
            distance.transpose(1, 2),
            duration.transpose(1, 2),
            energy.transpose(1, 2),
            coordinates,
        )
        return next_row, next_col


@dataclass(frozen=True)
class RRNCOFixedContext:
    row_embeddings: torch.Tensor
    col_embeddings: torch.Tensor
    distance: torch.Tensor
    duration: torch.Tensor
    energy: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_value: torch.Tensor
    logit_key: torch.Tensor


class RRNCOEVPolicy(nn.Module):
    """RRNCO road-relation encoder with the canonical EVRPTW-DB decoder state."""

    node_feature_dim = 10

    def __init__(
        self,
        embedding_dim: int = 128,
        n_encode_layers: int = 6,
        n_heads: int = 8,
        feedforward_hidden: int = 512,
        distance_sample_size: int = 25,
        tanh_clipping: float = 10.0,
    ) -> None:
        super().__init__()
        if embedding_dim % n_heads:
            raise ValueError("embedding_dim must be divisible by n_heads")
        if n_encode_layers <= 0:
            raise ValueError("n_encode_layers must be positive")
        self.embedding_dim = int(embedding_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.embedding_dim // self.n_heads
        self.tanh_clipping = float(tanh_clipping)
        self.initial = RRNCOInitialEmbedding(embedding_dim, distance_sample_size)
        self.encoder = nn.ModuleList(
            RRNCOLayer(embedding_dim, feedforward_hidden)
            for _ in range(int(n_encode_layers))
        )
        self.step_context = nn.Linear(embedding_dim + 3, embedding_dim, bias=False)
        self.project_nodes = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.pointer_output = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.pointer_feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )
        self.distance_weight = nn.Parameter(torch.tensor(1.0))
        self.duration_weight = nn.Parameter(torch.tensor(1.0))
        self.energy_weight = nn.Parameter(torch.tensor(1.0))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def initial_state(self, batch_size: int, n_traj: int) -> None:
        del batch_size, n_traj
        return None

    def encode(
        self,
        observation: dict[str, Any],
        distance: Any,
        travel_time: Any,
        energy: Any,
    ) -> RRNCOFixedContext:
        coordinates, node_features = self._node_features(observation)
        distance_tensor = _tensor(distance, self.device).float()
        duration_tensor = _tensor(travel_time, self.device).float()
        energy_tensor = _tensor(energy, self.device).float()
        if not (
            distance_tensor.ndim == 3
            and distance_tensor.shape == duration_tensor.shape == energy_tensor.shape
            and distance_tensor.shape[1] == distance_tensor.shape[2] == coordinates.shape[1]
        ):
            raise ValueError("D/T/E matrices must be batched squares matching terminals")
        if not all(
            torch.isfinite(value).all()
            for value in (distance_tensor, duration_tensor, energy_tensor)
        ):
            raise ValueError("RRNCO-EV requires finite D/T/E matrices")
        row, col = self.initial(node_features, coordinates, distance_tensor)
        for layer in self.encoder:
            row, col = layer(
                row, col, distance_tensor, duration_tensor, energy_tensor, coordinates
            )
        glimpse_key, glimpse_value, logit_key = self.project_nodes(col).chunk(3, dim=-1)
        return RRNCOFixedContext(
            row_embeddings=row,
            col_embeddings=col,
            distance=distance_tensor,
            duration=duration_tensor,
            energy=energy_tensor,
            glimpse_key=glimpse_key,
            glimpse_value=glimpse_value,
            logit_key=logit_key,
        )

    def logits(
        self,
        observation: dict[str, Any],
        fixed: RRNCOFixedContext,
        state: None = None,
    ) -> tuple[torch.Tensor, None]:
        del state
        last = _tensor(observation["last_node_idx"], self.device).long()
        action_mask = _tensor(observation["action_mask"], self.device).bool()
        current_load = _tensor(observation["current_load"], self.device).float()
        current_battery = _tensor(observation["current_battery"], self.device).float()
        current_time = _tensor(observation["current_time"], self.device).float()
        if last.ndim == 1:
            last = last[:, None]
            action_mask = action_mask[:, None, :]
            current_load = current_load[:, None]
            current_battery = current_battery[:, None]
            current_time = current_time[:, None]
        batch, trajectories = last.shape
        if fixed.row_embeddings.shape[0] != batch:
            raise ValueError("observation batch does not match encoded graph batch")
        expanded = fixed.row_embeddings[:, None].expand(-1, trajectories, -1, -1)
        index = last[..., None, None].expand(-1, -1, 1, self.embedding_dim)
        previous = torch.gather(expanded, 2, index).squeeze(2)
        query = self.step_context(
            torch.cat(
                (
                    previous,
                    current_load[..., None],
                    current_battery[..., None],
                    current_time[..., None],
                ),
                dim=-1,
            )
        )
        infeasible = ~action_mask
        glimpse = self._pointer_glimpse(query, fixed, infeasible)
        compatibility = torch.matmul(glimpse, fixed.logit_key.transpose(-2, -1))
        compatibility = compatibility / math.sqrt(self.embedding_dim)
        base_logits = self.tanh_clipping * torch.tanh(compatibility)
        batch_index = torch.arange(batch, device=self.device)[:, None].expand_as(last)
        edge_bias = (
            F.softplus(self.distance_weight) * fixed.distance[batch_index, last]
            + F.softplus(self.duration_weight) * fixed.duration[batch_index, last]
            + F.softplus(self.energy_weight) * fixed.energy[batch_index, last]
        )
        return (base_logits - edge_bias).masked_fill(infeasible, -torch.inf), None

    def _pointer_glimpse(
        self,
        query: torch.Tensor,
        fixed: RRNCOFixedContext,
        infeasible: torch.Tensor,
    ) -> torch.Tensor:
        """RRNCO pointer attention using all cached K/V/logit projections."""

        batch, trajectories, _ = query.shape
        nodes = fixed.glimpse_key.shape[1]
        q = query.reshape(batch, trajectories, self.n_heads, self.head_dim).transpose(1, 2)
        k = fixed.glimpse_key.reshape(
            batch, nodes, self.n_heads, self.head_dim
        ).transpose(1, 2)
        v = fixed.glimpse_value.reshape(
            batch, nodes, self.n_heads, self.head_dim
        ).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        expanded_mask = infeasible[:, None, :, :]
        scores = scores.masked_fill(expanded_mask, -torch.inf)
        weights = torch.softmax(scores, dim=-1).masked_fill(expanded_mask, 0.0)
        heads = torch.matmul(weights, v).transpose(1, 2).reshape(
            batch, trajectories, self.embedding_dim
        )
        glimpse = query + self.pointer_output(heads)
        return glimpse + self.pointer_feed_forward(glimpse)

    def _node_features(
        self, observation: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        depot = _tensor(observation["depot_loc"], self.device).float()
        customers = _tensor(observation["cus_loc"], self.device).float()
        stations = _tensor(observation["rs_loc"], self.device).float()
        demand = _tensor(observation["demand"], self.device).float()
        time_window = _tensor(observation["time_window"], self.device).float()
        service = _tensor(observation["service_time"], self.device).float()
        charge = _tensor(observation["charging_time_ratio"], self.device).float()
        if depot.ndim == 2:
            depot = depot[:, None]
        coordinates = torch.cat((depot, customers, stations), dim=1)
        batch, nodes, _ = coordinates.shape
        if demand.shape != (batch, nodes):
            raise ValueError("node attributes do not match terminal count")
        node_type = torch.zeros(batch, nodes, 3, device=self.device)
        node_type[:, 0, 0] = 1.0
        node_type[:, 1 : 1 + customers.shape[1], 1] = 1.0
        node_type[:, 1 + customers.shape[1] :, 2] = 1.0
        features = torch.cat(
            (
                coordinates,
                demand[..., None],
                time_window,
                service[..., None],
                charge[..., None],
                node_type,
            ),
            dim=-1,
        )
        return coordinates, features


__all__ = ["RRNCOEVPolicy", "RRNCOFixedContext"]
