from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from ..DRL_TS.model import EdgeMultiHeadAttention


def _tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(np.asarray(value), device=device)


class DirectedGraphAttentionLayer(nn.Module):
    """Multi-head attention restricted to the paper's time-window graph."""

    def __init__(self, embedding_dim: int, n_heads: int) -> None:
        super().__init__()
        if embedding_dim % n_heads:
            raise ValueError("embedding_dim must be divisible by n_heads")
        self.embedding_dim = int(embedding_dim)
        self.n_heads = int(n_heads)
        self.head_dim = embedding_dim // n_heads
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.output = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = nodes.shape
        q = self.query(nodes).view(
            batch_size, num_nodes, self.n_heads, self.head_dim
        ).transpose(1, 2)
        k = self.key(nodes).view(
            batch_size, num_nodes, self.n_heads, self.head_dim
        ).transpose(1, 2)
        v = self.value(nodes).view(
            batch_size, num_nodes, self.n_heads, self.head_dim
        ).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~adjacency[:, None, :, :], -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        message = torch.matmul(weights, v).transpose(1, 2).reshape(
            batch_size, num_nodes, self.embedding_dim
        )
        nodes = self.norm1(nodes + self.output(message))
        return self.norm2(nodes + self.feed_forward(nodes))


class EdgeEnhancedEncoderLayer(nn.Module):
    """Directed node attention conditioned on travel-time and energy edges."""

    def __init__(self, embedding_dim: int, n_heads: int) -> None:
        super().__init__()
        self.attention = EdgeMultiHeadAttention(
            query_dim=embedding_dim,
            pair_dim=2 * embedding_dim,
            embedding_dim=embedding_dim,
            n_heads=n_heads,
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        num_nodes = nodes.size(1)
        target = nodes[:, None, :, :].expand(-1, num_nodes, -1, -1)
        pair = torch.cat((target, edges), dim=-1)
        nodes = self.norm1(nodes + self.attention(nodes, pair))
        return self.norm2(nodes + self.feed_forward(nodes))


@dataclass(frozen=True)
class EdgeDirectFixedContext:
    node_embeddings: torch.Tensor
    edge_embeddings: torch.Tensor
    graph_embedding: torch.Tensor


class EdgeDirectHomogeneousPolicy(nn.Module):
    """Edge-DIRECT-H: the paper architecture's homogeneous-fleet special case.

    EVRPTW-DB has an unlimited homogeneous fleet, so the heterogeneous vehicle
    decoder has one admissible class.  It remains an explicit learned module,
    but its categorical log-probability is exactly zero.  The node decoder and
    both directed edge encoders remain non-degenerate.
    """

    node_feature_dim = 10

    def __init__(
        self,
        embedding_dim: int = 128,
        n_encode_layers: int = 3,
        n_heads: int = 8,
        tanh_clipping: float = 10.0,
    ) -> None:
        super().__init__()
        if embedding_dim % n_heads:
            raise ValueError("embedding_dim must be divisible by n_heads")
        if n_encode_layers <= 0:
            raise ValueError("n_encode_layers must be positive")
        self.embedding_dim = int(embedding_dim)
        self.tanh_clipping = float(tanh_clipping)
        self.node_projection = nn.Linear(self.node_feature_dim, embedding_dim)
        self.time_window_encoder = nn.ModuleList(
            DirectedGraphAttentionLayer(embedding_dim, n_heads)
            for _ in range(int(n_encode_layers))
        )
        self.edge_projection = nn.Linear(2, embedding_dim)
        self.edge_node_projection = nn.Linear(2 * embedding_dim, embedding_dim)
        self.edge_encoder = nn.ModuleList(
            EdgeEnhancedEncoderLayer(embedding_dim, n_heads)
            for _ in range(int(n_encode_layers))
        )

        # A retained one-class vehicle decoder. Static unit capacity/battery and
        # canonical dynamic state produce a vehicle context for the node decoder.
        self.vehicle_static = nn.Linear(2, embedding_dim)
        self.vehicle_dynamic = nn.Linear(3, embedding_dim)
        self.vehicle_decoder = nn.Sequential(
            nn.Linear(3 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.node_query = nn.Linear(4 * embedding_dim, embedding_dim)
        self.node_glimpse = EdgeMultiHeadAttention(
            query_dim=embedding_dim,
            pair_dim=2 * embedding_dim,
            embedding_dim=embedding_dim,
            n_heads=n_heads,
        )
        self.logit_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.logit_key = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(
        self,
        observation: dict[str, Any],
        travel_time: Any,
        energy: Any,
        time_window_travel_time: Any | None = None,
    ) -> EdgeDirectFixedContext:
        node_features = self._node_features(observation)
        travel = _tensor(travel_time, self.device).float()
        energy_tensor = _tensor(energy, self.device).float()
        if travel.ndim != 3 or travel.shape != energy_tensor.shape:
            raise ValueError("travel-time and energy matrices must be batched squares")
        adjacency_travel = (
            travel
            if time_window_travel_time is None
            else _tensor(time_window_travel_time, self.device).float()
        )
        if adjacency_travel.shape != travel.shape:
            raise ValueError("time-window travel matrix has an invalid shape")
        adjacency = self.time_window_adjacency(observation, adjacency_travel)
        nodes = self.node_projection(node_features)
        for layer in self.time_window_encoder:
            nodes = layer(nodes, adjacency)

        edges = self.edge_projection(torch.stack((travel, energy_tensor), dim=-1))
        nodes = self.edge_node_projection(
            torch.cat((nodes, self.node_projection(node_features)), dim=-1)
        )
        for layer in self.edge_encoder:
            nodes = layer(nodes, edges)
        return EdgeDirectFixedContext(
            node_embeddings=nodes,
            edge_embeddings=edges,
            graph_embedding=nodes.mean(dim=1),
        )

    def logits(
        self,
        observation: dict[str, Any],
        fixed: EdgeDirectFixedContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        last = _tensor(observation["last_node_idx"], self.device).long()
        action_mask = _tensor(observation["action_mask"], self.device).bool()
        if last.ndim == 1:
            last = last[:, None]
            action_mask = action_mask[:, None, :]
        batch_size, n_traj = last.shape
        num_nodes = fixed.node_embeddings.size(1)
        nodes = fixed.node_embeddings[:, None, :, :].expand(-1, n_traj, -1, -1)
        node_index = last[:, :, None, None].expand(-1, -1, 1, self.embedding_dim)
        previous = torch.gather(nodes, 2, node_index).squeeze(2)

        remaining_capacity = 1.0 - _tensor(
            observation["current_load"], self.device
        ).float()
        remaining_battery = _tensor(
            observation["remaining_battery"], self.device
        ).float()
        current_time = _tensor(observation["current_time"], self.device).float()
        dynamic = torch.stack(
            (remaining_capacity, remaining_battery, current_time), dim=-1
        )
        static_vehicle = self.vehicle_static(
            torch.ones(batch_size, n_traj, 2, device=self.device)
        )
        graph = fixed.graph_embedding[:, None, :].expand(-1, n_traj, -1)
        vehicle = self.vehicle_decoder(
            torch.cat((static_vehicle, self.vehicle_dynamic(dynamic), graph), dim=-1)
        )
        # softmax over the single homogeneous vehicle is one; log p(vehicle)=0.
        vehicle_log_probability = torch.zeros(
            batch_size, n_traj, device=self.device, dtype=nodes.dtype
        )

        expanded_edges = fixed.edge_embeddings[:, None, :, :, :].expand(
            -1, n_traj, -1, -1, -1
        )
        edge_index = last[:, :, None, None, None].expand(
            -1, -1, 1, num_nodes, self.embedding_dim
        )
        outgoing = torch.gather(expanded_edges, 2, edge_index).squeeze(2)
        pair = torch.cat((nodes, outgoing), dim=-1)
        query = self.node_query(
            torch.cat(
                (graph, previous, vehicle, self.vehicle_dynamic(dynamic)), dim=-1
            )
        )
        glimpse = self.node_glimpse(query, pair, mask=action_mask)
        compatibility = torch.einsum(
            "btd,btnd->btn",
            self.logit_query(glimpse),
            self.logit_key(pair),
        ) / math.sqrt(self.embedding_dim)
        logits = self.tanh_clipping * torch.tanh(compatibility)
        return logits.masked_fill(~action_mask, -torch.inf), vehicle_log_probability

    def time_window_adjacency(
        self,
        observation: dict[str, Any],
        travel_time: torch.Tensor,
    ) -> torch.Tensor:
        tw = _tensor(observation["time_window"], self.device).float()
        earliest_i = tw[:, :, 0, None]
        latest_i = tw[:, :, 1, None]
        earliest_j = tw[:, None, :, 0]
        latest_j = tw[:, None, :, 1]
        lower = torch.maximum(earliest_i, earliest_j - travel_time)
        upper = torch.minimum(latest_i, latest_j - travel_time)
        adjacency = lower <= upper + 1e-9
        diagonal = torch.eye(
            travel_time.size(1), device=self.device, dtype=torch.bool
        )[None]
        return adjacency | diagonal

    def _node_features(self, observation: dict[str, Any]) -> torch.Tensor:
        coordinates = torch.cat(
            (
                _tensor(observation["depot_loc"], self.device).float(),
                _tensor(observation["cus_loc"], self.device).float(),
                _tensor(observation["rs_loc"], self.device).float(),
            ),
            dim=1,
        )
        demand = _tensor(observation["demand"], self.device).float()
        time_window = _tensor(observation["time_window"], self.device).float()
        service = _tensor(observation["service_time"], self.device).float()
        charging_power = _tensor(observation["charging_power"], self.device).float()
        batch_size, num_nodes = demand.shape
        customer_count = int(_tensor(observation["cus_loc"], self.device).shape[1])
        node_type = torch.zeros(
            batch_size, num_nodes, 3, device=self.device, dtype=demand.dtype
        )
        node_type[:, 0, 0] = 1.0
        node_type[:, 1 : 1 + customer_count, 1] = 1.0
        node_type[:, 1 + customer_count :, 2] = 1.0
        return torch.cat(
            (
                coordinates,
                demand[..., None],
                time_window,
                service[..., None],
                charging_power[..., None],
                node_type,
            ),
            dim=-1,
        )


__all__ = ["EdgeDirectFixedContext", "EdgeDirectHomogeneousPolicy"]
