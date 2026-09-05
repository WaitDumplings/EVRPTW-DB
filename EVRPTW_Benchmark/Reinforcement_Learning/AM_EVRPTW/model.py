from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .third_party.attention_learn_to_route.graph_encoder import (
    GraphAttentionEncoder,
    MultiHeadAttention,
)


@dataclass(frozen=True)
class AMFixedContext:
    """Encoder outputs and decoder projections cached for one rollout."""

    node_embeddings: torch.Tensor
    graph_context: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_value: torch.Tensor
    logit_key: torch.Tensor
    attention_key: torch.Tensor | None = None
    attention_value: torch.Tensor | None = None


def _tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(np.asarray(value), device=device)


class AMEVRPTWPolicy(nn.Module):
    """Kool et al. Attention Model with an EVRP-TW state adapter.

    The attention encoder, cached attention decoder, tanh clipping, and rollout
    training interface follow the ICLR 2019 AM design.  The input embeddings and
    decoder context are extended to expose the constraints of the shared
    EVRPTW-B environment.  Feasibility remains external to the network and is
    enforced by the canonical action mask.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        n_encode_layers: int = 3,
        n_heads: int = 8,
        tanh_clipping: float = 10.0,
        normalization: str = "batch",
    ) -> None:
        super().__init__()
        if embedding_dim != hidden_dim:
            raise ValueError("AM requires embedding_dim == hidden_dim")
        if embedding_dim % n_heads:
            raise ValueError("embedding_dim must be divisible by n_heads")
        self.embedding_dim = int(embedding_dim)
        self.n_heads = int(n_heads)
        self.tanh_clipping = float(tanh_clipping)

        # The upstream CVRP model uses a dedicated depot projection.  Customer
        # and station inputs share a projection and are distinguished by a
        # one-hot node type.  Node feature order is documented in ADAPTATION.md.
        self.init_embed_depot = nn.Linear(2, embedding_dim)
        self.init_embed_node = nn.Linear(10, embedding_dim)
        self.encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=n_encode_layers,
            normalization=normalization,
        )

        self.project_node_embeddings = nn.Linear(
            embedding_dim, 3 * embedding_dim, bias=False
        )
        self.project_fixed_context = nn.Linear(
            embedding_dim, embedding_dim, bias=False
        )
        self.project_step_context = nn.Linear(
            embedding_dim + 3, embedding_dim, bias=False
        )
        self.glimpse = MultiHeadAttention(
            n_heads=n_heads,
            input_dim=embedding_dim,
            embed_dim=embedding_dim,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(
        self, observation: dict[str, Any], *, cache_decoder: bool = True
    ) -> AMFixedContext:
        node_embeddings = self._initial_embeddings(observation)
        encoded, graph_embedding = self.encoder(node_embeddings)
        glimpse_key, glimpse_value, logit_key = self.project_node_embeddings(
            encoded
        ).chunk(3, dim=-1)
        attention_key, attention_value = (
            self._project_glimpse_memory(encoded)
            if cache_decoder
            else (None, None)
        )
        return AMFixedContext(
            node_embeddings=encoded,
            graph_context=self.project_fixed_context(graph_embedding)[:, None, :],
            glimpse_key=glimpse_key,
            glimpse_value=glimpse_value,
            logit_key=logit_key,
            attention_key=attention_key,
            attention_value=attention_value,
        )

    def _project_glimpse_memory(
        self, nodes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cache the actual glimpse weights, retaining the rollout autograd graph.

        The legacy ``glimpse_key/value`` fields use different parameters and
        cannot substitute for MultiHeadAttention's W_key and W_val. No cache is
        stored on the module, so it cannot outlive a rollout/optimizer update.
        """
        batch_size, graph_size, input_dim = nodes.shape
        flat = nodes.contiguous().view(-1, input_dim)
        shape = (self.glimpse.n_heads, batch_size, graph_size, -1)
        return (
            torch.matmul(flat, self.glimpse.W_key).view(shape),
            torch.matmul(flat, self.glimpse.W_val).view(shape),
        )

    def _cached_glimpse(
        self, query: torch.Tensor, fixed: AMFixedContext, mask: torch.Tensor
    ) -> torch.Tensor:
        """The upstream attention operations with only fixed K/V reused."""
        batch_size, n_query, input_dim = query.shape
        q = torch.matmul(
            query.contiguous().view(-1, input_dim), self.glimpse.W_query
        ).view(self.glimpse.n_heads, batch_size, n_query, -1)
        compatibility = self.glimpse.norm_factor * torch.matmul(
            q, fixed.attention_key.transpose(2, 3)
        )
        expanded_mask = mask.view(1, batch_size, n_query, -1).expand_as(
            compatibility
        )
        compatibility = compatibility.masked_fill(expanded_mask, -torch.inf)
        attention = torch.softmax(compatibility, dim=-1)
        # Match the upstream all-masked-row handling exactly.
        attention = attention.masked_fill(expanded_mask, 0.0)
        heads = torch.matmul(attention, fixed.attention_value)
        return torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(
                -1, self.glimpse.n_heads * self.glimpse.val_dim
            ),
            self.glimpse.W_out.view(-1, self.glimpse.embed_dim),
        ).view(batch_size, n_query, self.glimpse.embed_dim)

    def logits(
        self,
        observation: dict[str, Any],
        fixed: AMFixedContext,
    ) -> torch.Tensor:
        last = _tensor(observation["last_node_idx"], self.device).long()
        current_load = _tensor(observation["current_load"], self.device).float()
        current_battery = _tensor(
            observation["current_battery"], self.device
        ).float()
        current_time = _tensor(observation["current_time"], self.device).float()
        action_mask = _tensor(observation["action_mask"], self.device).bool()

        if last.ndim == 1:
            last = last[:, None]
            current_load = current_load[:, None]
            current_battery = current_battery[:, None]
            current_time = current_time[:, None]
            action_mask = action_mask[:, None, :]

        batch_size, n_traj = last.shape
        if fixed.node_embeddings.size(0) != batch_size:
            raise ValueError("observation batch does not match encoded graph batch")
        gather_index = last[..., None].expand(-1, -1, self.embedding_dim)
        expanded_nodes = fixed.node_embeddings[:, None, :, :].expand(
            -1, n_traj, -1, -1
        )
        previous = torch.gather(
            expanded_nodes,
            2,
            gather_index[:, :, None, :],
        ).squeeze(2)
        step_context = torch.cat(
            (
                previous,
                current_load[..., None],
                current_battery[..., None],
                current_time[..., None],
            ),
            dim=-1,
        )
        query = fixed.graph_context + self.project_step_context(step_context)
        infeasible = ~action_mask
        if fixed.attention_key is not None and fixed.attention_value is not None:
            glimpse = self._cached_glimpse(query, fixed, infeasible)
        else:
            glimpse = self.glimpse(
                query, fixed.node_embeddings, mask=infeasible
            )
        compatibility = torch.matmul(
            glimpse,
            fixed.logit_key.transpose(-2, -1),
        ) / math.sqrt(self.embedding_dim)
        logits = self.tanh_clipping * torch.tanh(compatibility)
        return logits.masked_fill(infeasible, -torch.inf)

    def _initial_embeddings(self, observation: dict[str, Any]) -> torch.Tensor:
        device = self.device
        depot = _tensor(observation["depot_loc"], device).float()
        customers = _tensor(observation["cus_loc"], device).float()
        stations = _tensor(observation["rs_loc"], device).float()
        demand = _tensor(observation["demand"], device).float()
        time_window = _tensor(observation["time_window"], device).float()
        service = _tensor(observation["service_time"], device).float()
        charging_time_ratio = _tensor(observation["charging_time_ratio"], device).float()

        if depot.ndim == 2:
            depot = depot[:, None, :]
        coordinates = torch.cat((depot, customers, stations), dim=1)
        batch_size, num_nodes, _ = coordinates.shape
        num_customers = customers.size(1)
        num_stations = stations.size(1)
        if demand.shape != (batch_size, num_nodes):
            raise ValueError("demand tensor does not match terminal count")

        node_type = torch.zeros(
            batch_size, num_nodes, 3, device=device, dtype=coordinates.dtype
        )
        node_type[:, 0, 0] = 1.0
        node_type[:, 1 : 1 + num_customers, 1] = 1.0
        if num_stations:
            node_type[:, 1 + num_customers :, 2] = 1.0

        features = torch.cat(
            (
                coordinates,
                demand[..., None],
                time_window,
                service[..., None],
                charging_time_ratio[..., None],
                node_type,
            ),
            dim=-1,
        )
        depot_embedding = self.init_embed_depot(depot)
        non_depot_embedding = self.init_embed_node(features[:, 1:, :])
        return torch.cat((depot_embedding, non_depot_embedding), dim=1)
