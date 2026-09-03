from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


def _tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(np.asarray(value), device=device)


class EdgeMultiHeadAttention(nn.Module):
    """Attention where each query-candidate pair has its own edge features."""

    def __init__(
        self,
        query_dim: int,
        pair_dim: int,
        embedding_dim: int,
        n_heads: int,
    ) -> None:
        super().__init__()
        if embedding_dim % n_heads:
            raise ValueError("embedding_dim must be divisible by n_heads")
        self.embedding_dim = int(embedding_dim)
        self.n_heads = int(n_heads)
        self.head_dim = embedding_dim // n_heads
        self.query = nn.Linear(query_dim, embedding_dim, bias=False)
        self.key = nn.Linear(pair_dim, embedding_dim, bias=False)
        self.value = nn.Linear(pair_dim, embedding_dim, bias=False)
        self.output = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        pair: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, query_count, candidate_count, _ = pair.shape
        q = self.query(query).view(
            batch_size,
            query_count,
            self.n_heads,
            self.head_dim,
        )
        k = self.key(pair).view(
            batch_size,
            query_count,
            candidate_count,
            self.n_heads,
            self.head_dim,
        )
        v = self.value(pair).view_as(k)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 1, 2, 4)
        scores = torch.einsum("bhqd,bhqkd->bhqk", q, k) / math.sqrt(
            self.head_dim
        )
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, :, :], -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        values = torch.einsum("bhqk,bhqkd->bhqd", weights, v)
        values = values.permute(0, 2, 1, 3).reshape(
            batch_size,
            query_count,
            self.embedding_dim,
        )
        return self.output(values)


class EdgeAwareGATLayer(nn.Module):
    """Paper-aligned simultaneous node/edge update (Chen et al., Eqs. 4-5)."""

    def __init__(self, embedding_dim: int, n_heads: int) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.node_attention = EdgeMultiHeadAttention(
            query_dim=embedding_dim,
            pair_dim=2 * embedding_dim,
            embedding_dim=embedding_dim,
            n_heads=n_heads,
        )
        self.node_aggregate = nn.Linear(embedding_dim, embedding_dim)
        self.node_aggregate_norm = nn.BatchNorm1d(embedding_dim)
        self.node_combine = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.node_combine_norm = nn.BatchNorm1d(embedding_dim)

        self.edge_self = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.edge_source = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.edge_target = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.edge_aggregate = nn.Linear(embedding_dim, embedding_dim)
        self.edge_aggregate_norm = nn.BatchNorm1d(embedding_dim)
        self.edge_combine = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.edge_combine_norm = nn.BatchNorm1d(embedding_dim)

    @staticmethod
    def _norm(module: nn.BatchNorm1d, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape
        return module(value.reshape(-1, shape[-1])).reshape(shape)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_nodes = nodes.size(1)
        target_nodes = nodes[:, None, :, :].expand(-1, num_nodes, -1, -1)
        node_pairs = torch.cat((target_nodes, edges), dim=-1)
        node_message = self.node_attention(nodes, node_pairs)
        node_message = torch.relu(
            self._norm(
                self.node_aggregate_norm,
                self.node_aggregate(node_message),
            )
        )

        source = nodes[:, :, None, :]
        target = nodes[:, None, :, :]
        edge_message = (
            self.edge_self(edges)
            + self.edge_source(source)
            + self.edge_target(target)
        )
        edge_message = torch.relu(
            self._norm(
                self.edge_aggregate_norm,
                self.edge_aggregate(edge_message),
            )
        )
        next_nodes = torch.relu(
            nodes
            + self._norm(
                self.node_combine_norm,
                self.node_combine(node_message),
            )
        )
        next_edges = torch.relu(
            edges
            + self._norm(
                self.edge_combine_norm,
                self.edge_combine(edge_message),
            )
        )
        return next_nodes, next_edges


@dataclass(frozen=True)
class DRLTSFixedContext:
    node_embeddings: torch.Tensor
    edge_embeddings: torch.Tensor


@dataclass(frozen=True)
class DRLTSRecurrentState:
    hidden: torch.Tensor


class DRLTSPolicy(nn.Module):
    """Paper-guided DRL-TS policy with documented EVRPTW-DB features."""

    node_feature_dim = 6
    edge_feature_dim = 4

    def __init__(
        self,
        embedding_dim: int = 128,
        n_encode_layers: int = 2,
        n_heads: int = 8,
        nearest_neighbors: int = 10,
        tanh_clipping: float = 10.0,
    ) -> None:
        super().__init__()
        if embedding_dim % n_heads:
            raise ValueError("embedding_dim must be divisible by n_heads")
        if n_encode_layers <= 0:
            raise ValueError("n_encode_layers must be positive")
        self.embedding_dim = int(embedding_dim)
        self.nearest_neighbors = int(nearest_neighbors)
        self.tanh_clipping = float(tanh_clipping)
        self.node_projection = nn.Linear(self.node_feature_dim, embedding_dim)
        self.edge_projection = nn.Linear(self.edge_feature_dim, embedding_dim)
        self.encoder = nn.ModuleList(
            EdgeAwareGATLayer(embedding_dim, n_heads)
            for _ in range(int(n_encode_layers))
        )
        self.context_projection = nn.Linear(3, embedding_dim)
        self.decoder_gru = nn.GRUCell(embedding_dim, embedding_dim)
        self.glimpse = EdgeMultiHeadAttention(
            query_dim=2 * embedding_dim,
            pair_dim=2 * embedding_dim,
            embedding_dim=embedding_dim,
            n_heads=n_heads,
        )
        self.compatibility_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.compatibility_key = nn.Linear(
            2 * embedding_dim,
            embedding_dim,
            bias=False,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def initial_state(self, batch_size: int, n_traj: int) -> DRLTSRecurrentState:
        return DRLTSRecurrentState(
            hidden=torch.zeros(
                int(batch_size),
                int(n_traj),
                self.embedding_dim,
                device=self.device,
            )
        )

    def encode(
        self,
        observation: dict[str, Any],
        distance: Any,
        travel_time: Any,
        energy: Any,
    ) -> DRLTSFixedContext:
        node_features = self._node_features(observation)
        edge_features = self._edge_features(distance, travel_time, energy)
        nodes = self.node_projection(node_features)
        edges = self.edge_projection(edge_features)
        for layer in self.encoder:
            nodes, edges = layer(nodes, edges)
        return DRLTSFixedContext(nodes, edges)

    def logits(
        self,
        observation: dict[str, Any],
        fixed: DRLTSFixedContext,
        state: DRLTSRecurrentState,
    ) -> tuple[torch.Tensor, DRLTSRecurrentState]:
        last = _tensor(observation["last_node_idx"], self.device).long()
        action_mask = _tensor(observation["action_mask"], self.device).bool()
        if last.ndim == 1:
            last = last[:, None]
            action_mask = action_mask[:, None, :]
        batch_size, n_traj = last.shape
        num_nodes = fixed.node_embeddings.size(1)
        nodes = fixed.node_embeddings[:, None, :, :].expand(
            -1,
            n_traj,
            -1,
            -1,
        )
        node_index = last[:, :, None, None].expand(
            -1,
            -1,
            1,
            self.embedding_dim,
        )
        previous = torch.gather(nodes, 2, node_index).squeeze(2)
        hidden = self.decoder_gru(
            previous.reshape(-1, self.embedding_dim),
            state.hidden.reshape(-1, self.embedding_dim),
        ).reshape(batch_size, n_traj, self.embedding_dim)
        next_state = DRLTSRecurrentState(hidden=hidden)

        remaining_capacity = 1.0 - _tensor(
            observation["current_load"],
            self.device,
        ).float()
        context = torch.stack(
            (
                _tensor(observation["current_time"], self.device).float(),
                remaining_capacity,
                _tensor(observation["remaining_battery"], self.device).float(),
            ),
            dim=-1,
        )
        context = self.context_projection(context)

        edge_index = last[:, :, None, None].expand(
            -1,
            -1,
            1,
            num_nodes,
        )
        expanded_edges = fixed.edge_embeddings[:, None, :, :, :].expand(
            -1,
            n_traj,
            -1,
            -1,
            -1,
        )
        edge_from_previous = torch.gather(
            expanded_edges,
            2,
            edge_index[..., None].expand(-1, -1, -1, -1, self.embedding_dim),
        ).squeeze(2)
        pairs = torch.cat((nodes, edge_from_previous), dim=-1)
        glimpse = self.glimpse(
            torch.cat((hidden, context), dim=-1),
            pairs,
            mask=action_mask,
        )
        query = self.compatibility_query(glimpse)
        key = self.compatibility_key(pairs)
        compatibility = torch.einsum("btd,btnd->btn", query, key) / math.sqrt(
            self.embedding_dim
        )
        logits = self.tanh_clipping * torch.tanh(compatibility)
        return logits.masked_fill(~action_mask, -torch.inf), next_state

    def _node_features(self, observation: dict[str, Any]) -> torch.Tensor:
        demand = _tensor(observation["demand"], self.device).float()
        time_window = _tensor(observation["time_window"], self.device).float()
        service = _tensor(observation["service_time"], self.device).float()
        charging_power = _tensor(observation["charging_power"], self.device).float()
        batch_size, num_nodes = demand.shape
        node_type = torch.ones(
            batch_size,
            num_nodes,
            device=self.device,
            dtype=demand.dtype,
        )
        customer_count = int(
            _tensor(observation["cus_loc"], self.device).shape[1]
        )
        node_type[:, 0] = 0.0
        node_type[:, 1 + customer_count :] = -1.0
        return torch.cat(
            (
                demand[..., None],
                time_window,
                service[..., None],
                charging_power[..., None],
                node_type[..., None],
            ),
            dim=-1,
        )

    def _edge_features(
        self,
        distance: Any,
        travel_time: Any,
        energy: Any,
    ) -> torch.Tensor:
        distance_tensor = _tensor(distance, self.device).float()
        time_tensor = _tensor(travel_time, self.device).float()
        energy_tensor = _tensor(energy, self.device).float()
        if (
            distance_tensor.ndim != 3
            or distance_tensor.shape[-1] != distance_tensor.shape[-2]
        ):
            raise ValueError("edge matrices must have shape (batch, nodes, nodes)")
        batch_size, num_nodes, _ = distance_tensor.shape
        adjacency = torch.zeros_like(distance_tensor)
        diagonal = torch.eye(num_nodes, device=self.device, dtype=torch.bool)[None]
        adjacency = adjacency.masked_fill(diagonal, -1.0)
        if num_nodes > 1 and self.nearest_neighbors > 0:
            count = min(self.nearest_neighbors, num_nodes - 1)
            ranked = distance_tensor.masked_fill(diagonal, torch.inf)
            nearest = torch.topk(ranked, count, dim=-1, largest=False).indices
            adjacency.scatter_(-1, nearest, 1.0)
            adjacency = adjacency.masked_fill(diagonal, -1.0)
        return torch.stack(
            (distance_tensor, time_tensor, energy_tensor, adjacency),
            dim=-1,
        ).reshape(batch_size, num_nodes, num_nodes, self.edge_feature_dim)


__all__ = ["DRLTSFixedContext", "DRLTSPolicy", "DRLTSRecurrentState"]
