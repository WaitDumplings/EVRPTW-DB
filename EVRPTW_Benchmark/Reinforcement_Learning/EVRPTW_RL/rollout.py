from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..AM_EVRPTW.rollout import stack_observations
from .model import EVRPTWRLPolicy


@dataclass
class EVRPTWRLRollout:
    training_cost: torch.Tensor
    objective_distance_km: torch.Tensor
    log_likelihood: torch.Tensor
    feasible: torch.Tensor
    served_customers: torch.Tensor
    station_visits: torch.Tensor
    infos: list[dict[str, Any]]
    runtime_s: float
    environment_transitions: int
    trajectory_steps: torch.Tensor
    rollout_budget_exhausted: torch.Tensor


def _normalized_travel_time(envs: Sequence[Any]) -> np.ndarray:
    rows = []
    for env in envs:
        unwrapped = env.unwrapped
        matrix = np.asarray(unwrapped.travel_time_s, dtype=np.float32)
        rows.append(matrix / max(float(unwrapped.horizon_s), 1e-12))
    return np.stack(rows, axis=0)


def rollout(
    policy: EVRPTWRLPolicy,
    envs: Sequence[Any],
    *,
    decode_type: str,
    max_steps: int,
    seed: int,
    station_visit_penalty: float = 0.3,
    incomplete_penalty: float = 100.0,
) -> EVRPTWRLRollout:
    if decode_type not in {"sampling", "greedy"}:
        raise ValueError("decode_type must be sampling or greedy")
    observations: list[dict[str, np.ndarray]] = []
    infos: list[dict[str, Any]] = []
    for index, env in enumerate(envs):
        observation, info = env.reset(seed=int(seed) + index)
        observations.append(observation)
        infos.append(info)
    batch = stack_observations(observations)
    batch_size = len(envs)
    n_traj = int(envs[0].unwrapped.n_traj)
    state = policy.initial_state(batch_size, n_traj)
    travel_time = _normalized_travel_time(envs)
    done = np.zeros((batch_size, n_traj), dtype=bool)
    environment_transitions = 0
    trajectory_steps = np.zeros_like(done, dtype=np.int64)
    station_visits = np.zeros((batch_size, n_traj), dtype=np.int64)
    log_likelihood = torch.zeros(batch_size, n_traj, device=policy.device)
    start = time.perf_counter()

    for _ in range(int(max_steps)):
        batch = stack_observations(observations)
        logits, state = policy.logits(batch, travel_time, state)
        distribution = torch.distributions.Categorical(logits=logits)
        actions = (
            torch.argmax(logits, dim=-1)
            if decode_type == "greedy"
            else distribution.sample()
        )
        environment_transitions += int(np.count_nonzero(~done))
        trajectory_steps += (~done).astype(np.int64)
        active = torch.as_tensor(~done, device=policy.device)
        log_likelihood = log_likelihood + distribution.log_prob(actions) * active
        action_array = actions.detach().cpu().numpy().astype(np.int64)

        next_observations: list[dict[str, np.ndarray]] = []
        next_infos: list[dict[str, Any]] = []
        for env_index, (env, action) in enumerate(zip(envs, action_array)):
            station_start = int(env.unwrapped.station_start)
            station_visits[env_index] += (
                (~done[env_index]) & (action >= station_start)
            ).astype(np.int64)
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
    feasible = np.stack([np.asarray(info["success"], dtype=bool) for info in infos])
    customer_count = np.asarray(
        [env.unwrapped.num_customers for env in envs], dtype=np.float64
    )[:, None]
    distance_scale = np.asarray(
        [env.unwrapped.reward_distance_scale_km for env in envs], dtype=np.float64
    )[:, None]
    incomplete_fraction = 1.0 - served / np.maximum(customer_count, 1.0)
    training_cost = (
        objective / np.maximum(distance_scale, 1e-12)
        + float(station_visit_penalty) * station_visits
        + (~feasible)
        * (float(incomplete_penalty) * (1.0 + incomplete_fraction))
    )
    return EVRPTWRLRollout(
        training_cost=torch.as_tensor(training_cost, device=policy.device).float(),
        objective_distance_km=torch.as_tensor(objective, device=policy.device).float(),
        log_likelihood=log_likelihood,
        feasible=torch.as_tensor(feasible, device=policy.device),
        served_customers=torch.as_tensor(served, device=policy.device),
        station_visits=torch.as_tensor(station_visits, device=policy.device),
        infos=infos,
        runtime_s=float(time.perf_counter() - start),
        environment_transitions=environment_transitions,
        trajectory_steps=torch.as_tensor(
            trajectory_steps, device=policy.device
        ),
        rollout_budget_exhausted=torch.as_tensor(
            ~done, device=policy.device
        ),
    )
