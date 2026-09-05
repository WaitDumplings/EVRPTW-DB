from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


def _tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(np.asarray(value), device=device)


@dataclass(frozen=True)
class RecurrentState:
    hidden: torch.Tensor
    cell: torch.Tensor


@dataclass(frozen=True)
class EVRPTWRLFixedContext:
    """Static inputs and parameter-dependent edge messages for one rollout."""

    static_features: torch.Tensor
    node_type: torch.Tensor
    edge_message: torch.Tensor


class EVRPTWRLPolicy(nn.Module):
    """Structure2Vec, context attention, and LSTM decoder from Lin et al.

    The equations follow Sections IV--V of the IEEE T-ITS paper.  The shared
    EVRPTW environment supplies the canonical dynamic state and action mask.
    See ``ADAPTATION.md`` for every benchmark-specific feature extension.
    """

    node_feature_dim = 10

    def __init__(
        self,
        embedding_dim: int = 128,
        structure2vec_rounds: int = 3,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if structure2vec_rounds <= 0:
            raise ValueError("structure2vec_rounds must be positive")
        self.embedding_dim = int(embedding_dim)
        self.structure2vec_rounds = int(structure2vec_rounds)

        # Equation (6): theta_1 through theta_5.
        self.local_projection = nn.Linear(self.node_feature_dim, embedding_dim)
        self.global_projection = nn.Linear(3, embedding_dim)
        self.neighbor_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.edge_direction = nn.Parameter(torch.empty(embedding_dim))
        self.edge_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Equations (7)--(11) and the recurrent decoder.
        self.decoder = nn.LSTMCell(embedding_dim, embedding_dim)
        self.context_projection = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)
        self.context_score = nn.Linear(embedding_dim, 1, bias=False)
        self.choice_projection = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)
        self.choice_score = nn.Linear(embedding_dim, 1, bias=False)
        self.reset_parameters()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def reset_parameters(self) -> None:
        for name, parameter in self.named_parameters():
            if "weight" in name and parameter.ndim >= 2:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
        nn.init.uniform_(
            self.edge_direction,
            -1.0 / np.sqrt(self.embedding_dim),
            1.0 / np.sqrt(self.embedding_dim),
        )

    def initial_state(self, batch_size: int, n_traj: int) -> RecurrentState:
        shape = (int(batch_size), int(n_traj), self.embedding_dim)
        return RecurrentState(
            hidden=torch.zeros(shape, device=self.device),
            cell=torch.zeros(shape, device=self.device),
        )

    def logits(
        self,
        observation: dict[str, Any],
        normalized_travel_time: Any,
        state: RecurrentState,
        *,
        fixed: EVRPTWRLFixedContext | None = None,
    ) -> tuple[torch.Tensor, RecurrentState]:
        local = self._local_features(observation, fixed=fixed)
        batch_size, n_traj, num_nodes, _ = local.shape
        global_state = self._global_features(observation)
        if global_state.shape[:2] != (batch_size, n_traj):
            raise ValueError("global and local trajectory dimensions do not match")

        local_hat = self.local_projection(local)
        global_hat = self.global_projection(global_state)[:, :, None, :]
        edge_message = (
            fixed.edge_message
            if fixed is not None
            else self._edge_message(normalized_travel_time, batch_size, num_nodes)
        )

        embedding = local_hat
        for _ in range(self.structure2vec_rounds):
            neighbor_sum = embedding.sum(dim=2, keepdim=True) - embedding
            embedding = torch.relu(
                local_hat
                + global_hat
                + self.neighbor_projection(neighbor_sum)
                + edge_message
            )

        last = _tensor(observation["last_node_idx"], self.device).long()
        gather = last[:, :, None, None].expand(-1, -1, 1, self.embedding_dim)
        decoder_input = torch.gather(local_hat, 2, gather).squeeze(2)
        flat_input = decoder_input.reshape(-1, self.embedding_dim)
        flat_hidden = state.hidden.reshape(-1, self.embedding_dim)
        flat_cell = state.cell.reshape(-1, self.embedding_dim)
        next_hidden, next_cell = self.decoder(flat_input, (flat_hidden, flat_cell))
        next_state = RecurrentState(
            hidden=next_hidden.reshape(batch_size, n_traj, self.embedding_dim),
            cell=next_cell.reshape(batch_size, n_traj, self.embedding_dim),
        )

        action_mask = _tensor(observation["action_mask"], self.device).bool()
        hidden_nodes = next_state.hidden[:, :, None, :].expand(-1, -1, num_nodes, -1)
        context_compatibility = self.context_score(
            torch.tanh(self.context_projection(torch.cat((embedding, hidden_nodes), dim=-1)))
        ).squeeze(-1)
        context_compatibility = context_compatibility.masked_fill(~action_mask, -torch.inf)
        context_weights = torch.softmax(context_compatibility, dim=-1)
        context = torch.sum(context_weights[..., None] * embedding, dim=2)

        context_nodes = context[:, :, None, :].expand(-1, -1, num_nodes, -1)
        logits = self.choice_score(
            torch.tanh(self.choice_projection(torch.cat((embedding, context_nodes), dim=-1)))
        ).squeeze(-1)
        return logits.masked_fill(~action_mask, -torch.inf), next_state

    def encode_static(
        self, observation: dict[str, Any], normalized_travel_time: Any
    ) -> EVRPTWRLFixedContext:
        """Prepare immutable data once; do not detach parameter-dependent values.

        Demand/global features, Structure2Vec and the recurrent state remain
        dynamic. The context is caller-owned and must not cross optimizer steps.
        """
        static, node_type = self._static_local_features(observation)
        batch_size, _, num_nodes, _ = static.shape
        return EVRPTWRLFixedContext(
            static_features=static,
            node_type=node_type,
            edge_message=self._edge_message(
                normalized_travel_time, batch_size, num_nodes
            ),
        )

    def _edge_message(
        self, normalized_travel_time: Any, batch_size: int, num_nodes: int
    ) -> torch.Tensor:
        edge_time = _tensor(normalized_travel_time, self.device).float()
        if edge_time.shape != (batch_size, num_nodes, num_nodes):
            raise ValueError("normalized travel-time matrix has an invalid shape")
        edge_row_sum = edge_time.sum(dim=-1)[:, None, :, None]
        edge_message = torch.relu(edge_row_sum * self.edge_direction)
        return self.edge_projection(edge_message)

    def _static_local_features(
        self, observation: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        customers = _tensor(observation["cus_loc"], self.device).float()
        coordinates = torch.cat(
            (
                _tensor(observation["depot_loc"], self.device).float(),
                customers,
                _tensor(observation["rs_loc"], self.device).float(),
            ),
            dim=1,
        )
        time_window = _tensor(observation["time_window"], self.device).float()
        service = _tensor(observation["service_time"], self.device).float()
        charging_time_ratio = _tensor(observation["charging_time_ratio"], self.device).float()
        batch_size, num_nodes, _ = coordinates.shape
        static = torch.cat(
            (
                coordinates,
                time_window,
                service[..., None],
                charging_time_ratio[..., None],
            ),
            dim=-1,
        )
        static = static[:, None, :, :]
        node_type = torch.zeros(
            batch_size,
            1,
            num_nodes,
            3,
            device=self.device,
            dtype=static.dtype,
        )
        customer_count = int(customers.shape[1])
        node_type[:, :, 0, 0] = 1.0
        node_type[:, :, 1 : 1 + customer_count, 1] = 1.0
        node_type[:, :, 1 + customer_count :, 2] = 1.0
        return static, node_type

    def _local_features(
        self,
        observation: dict[str, Any],
        *,
        fixed: EVRPTWRLFixedContext | None = None,
    ) -> torch.Tensor:
        remaining_demand = _tensor(observation["remaining_demand"], self.device).float()
        batch_size, n_traj, num_nodes = remaining_demand.shape
        static, node_type = (
            (fixed.static_features, fixed.node_type)
            if fixed is not None
            else self._static_local_features(observation)
        )
        if static.shape[:3] != (batch_size, 1, num_nodes):
            raise ValueError("static and dynamic local feature dimensions do not match")
        return torch.cat(
            (
                static.expand(-1, n_traj, -1, -1),
                remaining_demand[..., None],
                node_type.expand(-1, n_traj, -1, -1),
            ),
            dim=-1,
        )

    def _global_features(self, observation: dict[str, Any]) -> torch.Tensor:
        current_time = _tensor(observation["current_time"], self.device).float()
        remaining_battery = _tensor(
            observation["remaining_battery"], self.device
        ).float()
        remaining_vehicle = _tensor(
            observation["remaining_vehicle_ratio"], self.device
        ).float()
        return torch.stack((current_time, remaining_battery, remaining_vehicle), dim=-1)
