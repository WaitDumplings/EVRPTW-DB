from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..AM_EVRPTW.rollout import stack_observations
from .model import DRLTSPolicy


@dataclass
class DRLTSRollout:
    training_cost: torch.Tensor
    objective_distance_km: torch.Tensor
    log_likelihood: torch.Tensor
    feasible: torch.Tensor
    served_customers: torch.Tensor
    capacity_violation: torch.Tensor
    time_violation: torch.Tensor
    energy_violation: torch.Tensor
    infos: list[dict[str, Any]]
    runtime_s: float
    environment_transitions: int


def normalized_edge_matrices(
    envs: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distance_rows = []
    time_rows = []
    energy_rows = []
    for env in envs:
        unwrapped = env.unwrapped
        distance = np.asarray(unwrapped.distance_km, dtype=np.float32)
        travel_time = np.asarray(unwrapped.travel_time_s, dtype=np.float32)
        energy = np.asarray(unwrapped.energy_kwh, dtype=np.float32)
        distance_rows.append(distance / max(float(np.max(distance)), 1e-12))
        time_rows.append(travel_time / max(float(np.max(travel_time)), 1e-12))
        energy_rows.append(energy / max(float(np.max(energy)), 1e-12))
    return (
        np.stack(distance_rows, axis=0),
        np.stack(time_rows, axis=0),
        np.stack(energy_rows, axis=0),
    )


def rollout(
    policy: DRLTSPolicy,
    envs: Sequence[Any],
    *,
    decode_type: str,
    max_steps: int,
    seed: int,
    soft_constraints: bool,
    capacity_penalty: float = 1.0,
    time_penalty: float = 1.0,
    energy_penalty: float = 1.0,
    incomplete_penalty: float = 100.0,
) -> DRLTSRollout:
    if decode_type not in {"sampling", "greedy"}:
        raise ValueError("decode_type must be sampling or greedy")
    observations: list[dict[str, np.ndarray]] = []
    infos: list[dict[str, Any]] = []
    for index, env in enumerate(envs):
        observation, info = env.reset(seed=int(seed) + index)
        observations.append(observation)
        infos.append(info)
    batch = stack_observations(observations)
    distance, travel_time, energy = normalized_edge_matrices(envs)
    fixed = policy.encode(batch, distance, travel_time, energy)
    batch_size = len(envs)
    n_traj = int(envs[0].unwrapped.n_traj)
    state = policy.initial_state(batch_size, n_traj)
    done = np.zeros((batch_size, n_traj), dtype=bool)
    environment_transitions = 0
    log_likelihood = torch.zeros(batch_size, n_traj, device=policy.device)
    started = time.perf_counter()

    for _ in range(int(max_steps)):
        batch = stack_observations(observations)
        logits, state = policy.logits(batch, fixed, state)
        distribution = torch.distributions.Categorical(logits=logits)
        actions = (
            torch.argmax(logits, dim=-1)
            if decode_type == "greedy"
            else distribution.sample()
        )
        environment_transitions += int(np.count_nonzero(~done))
        active = torch.as_tensor(~done, device=policy.device)
        log_likelihood = log_likelihood + distribution.log_prob(actions) * active
        action_array = actions.detach().cpu().numpy().astype(np.int64)

        next_observations: list[dict[str, np.ndarray]] = []
        next_infos: list[dict[str, Any]] = []
        for env_index, (env, action) in enumerate(zip(envs, action_array)):
            observation, _, terminated, truncated, info = env.step(action)
            done[env_index] |= np.asarray(terminated) | np.asarray(truncated)
            next_observations.append(observation)
            next_infos.append(info)
        observations = next_observations
        infos = next_infos
        if done.all():
            break

    objective = np.stack(
        [np.asarray(info["objective_distance_km"], dtype=np.float64) for info in infos]
    )
    served = np.stack(
        [np.asarray(info["served_customers"], dtype=np.int64) for info in infos]
    )
    completed = np.stack(
        [np.asarray(info["success"], dtype=bool) for info in infos]
    )
    zeros = np.zeros_like(objective)
    if soft_constraints:
        capacity_violation = np.stack(
            [
                np.asarray(
                    info["capacity_violation_normalized"],
                    dtype=np.float64,
                )
                for info in infos
            ]
        )
        time_violation = np.stack(
            [
                np.asarray(info["time_violation_normalized"], dtype=np.float64)
                for info in infos
            ]
        )
        energy_violation = np.stack(
            [
                np.asarray(info["energy_violation_normalized"], dtype=np.float64)
                for info in infos
            ]
        )
    else:
        capacity_violation = zeros.copy()
        time_violation = zeros.copy()
        energy_violation = zeros.copy()
    no_violation = (
        (capacity_violation <= 1e-9)
        & (time_violation <= 1e-9)
        & (energy_violation <= 1e-9)
    )
    feasible = completed & no_violation
    customer_count = np.asarray(
        [env.unwrapped.num_customers for env in envs],
        dtype=np.float64,
    )[:, None]
    distance_scale = np.asarray(
        [env.unwrapped.reward_distance_scale_km for env in envs],
        dtype=np.float64,
    )[:, None]
    incomplete_fraction = 1.0 - served / np.maximum(customer_count, 1.0)
    training_cost = (
        objective / np.maximum(distance_scale, 1e-12)
        + float(capacity_penalty) * capacity_violation
        + float(time_penalty) * time_violation
        + float(energy_penalty) * energy_violation
        + (~completed)
        * (float(incomplete_penalty) * (1.0 + incomplete_fraction))
    )
    return DRLTSRollout(
        training_cost=torch.as_tensor(training_cost, device=policy.device).float(),
        objective_distance_km=torch.as_tensor(objective, device=policy.device).float(),
        log_likelihood=log_likelihood,
        feasible=torch.as_tensor(feasible, device=policy.device),
        served_customers=torch.as_tensor(served, device=policy.device),
        capacity_violation=torch.as_tensor(
            capacity_violation,
            device=policy.device,
        ).float(),
        time_violation=torch.as_tensor(time_violation, device=policy.device).float(),
        energy_violation=torch.as_tensor(
            energy_violation,
            device=policy.device,
        ).float(),
        infos=infos,
        runtime_s=float(time.perf_counter() - started),
        environment_transitions=environment_transitions,
    )


__all__ = ["DRLTSRollout", "normalized_edge_matrices", "rollout"]
