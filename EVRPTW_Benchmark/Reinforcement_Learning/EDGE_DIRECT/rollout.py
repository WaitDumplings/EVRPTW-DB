from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..AM_EVRPTW.rollout import stack_observations
from ..DRL_TS.rollout import normalized_edge_matrices
from .model import EdgeDirectHomogeneousPolicy


@dataclass
class EdgeDirectRollout:
    cost_km: torch.Tensor
    log_likelihood: torch.Tensor
    feasible: torch.Tensor
    served_customers: torch.Tensor
    infos: list[dict[str, Any]]
    runtime_s: float


def rollout(
    policy: EdgeDirectHomogeneousPolicy,
    envs: Sequence[Any],
    *,
    decode_type: str,
    max_steps: int,
    seed: int,
    incomplete_penalty_km: float,
) -> EdgeDirectRollout:
    if decode_type not in {"sampling", "greedy"}:
        raise ValueError("decode_type must be sampling or greedy")
    observations: list[dict[str, np.ndarray]] = []
    infos: list[dict[str, Any]] = []
    for index, env in enumerate(envs):
        observation, info = env.reset(seed=int(seed) + index)
        observations.append(observation)
        infos.append(info)
    batch = stack_observations(observations)
    _, travel_time, energy = normalized_edge_matrices(envs)
    time_window_travel_time = np.stack(
        [
            np.asarray(env.unwrapped.travel_time_s, dtype=np.float32)
            / max(float(env.unwrapped.horizon_s), 1e-12)
            for env in envs
        ],
        axis=0,
    )
    fixed = policy.encode(
        batch,
        travel_time,
        energy,
        time_window_travel_time=time_window_travel_time,
    )
    batch_size = len(envs)
    n_traj = int(envs[0].unwrapped.n_traj)
    done = np.zeros((batch_size, n_traj), dtype=bool)
    log_likelihood = torch.zeros(batch_size, n_traj, device=policy.device)
    started = time.perf_counter()

    for _ in range(int(max_steps)):
        batch = stack_observations(observations)
        logits, vehicle_log_probability = policy.logits(batch, fixed)
        distribution = torch.distributions.Categorical(logits=logits)
        actions = (
            torch.argmax(logits, dim=-1)
            if decode_type == "greedy"
            else distribution.sample()
        )
        active = torch.as_tensor(~done, device=policy.device)
        log_likelihood = log_likelihood + (
            distribution.log_prob(actions) + vehicle_log_probability
        ) * active
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
    feasible = np.stack(
        [np.asarray(info["success"], dtype=bool) for info in infos]
    )
    customer_count = np.asarray(
        [env.unwrapped.num_customers for env in envs], dtype=np.float64
    )[:, None]
    incomplete_fraction = 1.0 - served / np.maximum(customer_count, 1.0)
    training_cost = objective + (~feasible) * (
        float(incomplete_penalty_km) * (1.0 + incomplete_fraction)
    )
    return EdgeDirectRollout(
        cost_km=torch.as_tensor(training_cost, device=policy.device).float(),
        log_likelihood=log_likelihood,
        feasible=torch.as_tensor(feasible, device=policy.device),
        served_customers=torch.as_tensor(served, device=policy.device),
        infos=infos,
        runtime_s=float(time.perf_counter() - started),
    )


__all__ = ["EdgeDirectRollout", "rollout"]
