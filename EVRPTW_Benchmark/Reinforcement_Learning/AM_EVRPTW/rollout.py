from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .model import AMEVRPTWPolicy


def stack_observations(rows: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.stack([row[key] for row in rows], axis=0) for key in rows[0]}


@dataclass
class AMRollout:
    cost_km: torch.Tensor
    log_likelihood: torch.Tensor
    feasible: torch.Tensor
    served_customers: torch.Tensor
    infos: list[dict[str, Any]]
    runtime_s: float
    environment_transitions: int
    trajectory_steps: torch.Tensor
    rollout_budget_exhausted: torch.Tensor


def rollout(
    policy: AMEVRPTWPolicy,
    envs: Sequence[Any],
    *,
    decode_type: str,
    max_steps: int,
    seed: int,
    incomplete_penalty_km: float,
) -> AMRollout:
    if decode_type not in {"sampling", "greedy"}:
        raise ValueError("decode_type must be sampling or greedy")
    observations: list[dict[str, np.ndarray]] = []
    infos: list[dict[str, Any]] = []
    for index, env in enumerate(envs):
        obs, info = env.reset(seed=int(seed) + index)
        observations.append(obs)
        infos.append(info)
    batch = stack_observations(observations)
    fixed = policy.encode(batch)
    n_traj = int(envs[0].unwrapped.n_traj)
    done = np.zeros((len(envs), n_traj), dtype=bool)
    environment_transitions = 0
    trajectory_steps = np.zeros_like(done, dtype=np.int64)
    log_likelihood = torch.zeros(
        len(envs), n_traj, device=policy.device, dtype=torch.float32
    )
    start = time.perf_counter()

    for _ in range(int(max_steps)):
        batch = stack_observations(observations)
        logits = policy.logits(batch, fixed)
        distribution = torch.distributions.Categorical(logits=logits)
        if decode_type == "greedy":
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = distribution.sample()
        environment_transitions += int(np.count_nonzero(~done))
        trajectory_steps += (~done).astype(np.int64)
        active = torch.as_tensor(~done, device=policy.device)
        log_likelihood = log_likelihood + distribution.log_prob(actions) * active
        action_array = actions.detach().cpu().numpy().astype(np.int64)

        next_observations: list[dict[str, np.ndarray]] = []
        next_infos: list[dict[str, Any]] = []
        for env_index, (env, action) in enumerate(zip(envs, action_array)):
            obs, _, terminated, truncated, info = env.step(action)
            done[env_index] |= np.asarray(terminated) | np.asarray(truncated)
            next_observations.append(obs)
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
    return AMRollout(
        cost_km=torch.as_tensor(training_cost, device=policy.device).float(),
        log_likelihood=log_likelihood,
        feasible=torch.as_tensor(feasible, device=policy.device),
        served_customers=torch.as_tensor(served, device=policy.device),
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
