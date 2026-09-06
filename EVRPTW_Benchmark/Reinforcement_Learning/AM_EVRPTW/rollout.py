from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..common.objective import resolve_objective
from ..common.reward_contract import (
    RewardScaleContract,
    classify_rollout_failure_reasons,
)
from .model import AMEVRPTWPolicy


def stack_observations(rows: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.stack([row[key] for row in rows], axis=0) for key in rows[0]}


def rollout_objective_arrays(
    envs: Sequence[Any], infos: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep physical distance separate from the configured scalar objective."""
    values, scales, vehicles, unit_costs = [], [], [], []
    for env, info in zip(envs, infos):
        base = env.unwrapped
        config = getattr(base, "objective_config", None)
        config = resolve_objective(
            config.to_dict() if hasattr(config, "to_dict") else config
        )
        if config.is_cost and "vehicles_started" not in info:
            raise ValueError("cost rollout requires vehicles_started in environment info")
        if config.is_cost and not hasattr(base, "reward_objective_scale"):
            raise ValueError("cost rollout requires reward_objective_scale")
        started = np.asarray(
            info.get("vehicles_started", info.get("vehicle_count", 0)), dtype=np.float64
        )
        distance = np.asarray(info["objective_distance_km"], dtype=np.float64)
        values.append(np.asarray(config.value(distance, started), dtype=np.float64))
        vehicles.append(np.broadcast_to(started, distance.shape))
        scales.append(float(getattr(
            base, "reward_objective_scale",
            getattr(base, "reward_distance_scale_km", 1.0),
        )))
        unit_costs.append(float(config.distance_unit_cost))
    return (
        np.stack(values), np.asarray(scales)[:, None],
        np.stack(vehicles), np.asarray(unit_costs)[:, None],
    )


@dataclass
class AMRollout:
    # Historical distance-plus-km-penalty diagnostic; never contains currency.
    cost_km: torch.Tensor
    training_cost: torch.Tensor
    objective_value: torch.Tensor
    objective_distance_km: torch.Tensor
    vehicles_started: torch.Tensor
    log_likelihood: torch.Tensor
    feasible: torch.Tensor
    served_customers: torch.Tensor
    infos: list[dict[str, Any]]
    runtime_s: float
    environment_transitions: int
    trajectory_steps: torch.Tensor
    rollout_budget_exhausted: torch.Tensor
    # CPU-only diagnostics; never used to recompute the training objective.
    training_cost_components: dict[str, torch.Tensor] | None = None
    reward_objective_scale: torch.Tensor | None = None
    failure_reasons: np.ndarray | None = None


def rollout(
    policy: AMEVRPTWPolicy,
    envs: Sequence[Any],
    *,
    decode_type: str,
    max_steps: int,
    seed: int,
    incomplete_penalty_km: float,
    use_static_cache: bool = True,
    compute_log_likelihood: bool = True,
    reward_contract: RewardScaleContract | None = None,
) -> AMRollout:
    if decode_type not in {"sampling", "greedy"}:
        raise ValueError("decode_type must be sampling or greedy")
    if reward_contract is not None:
        reward_contract.validate_envs(envs)
    observations: list[dict[str, np.ndarray]] = []
    infos: list[dict[str, Any]] = []
    for index, env in enumerate(envs):
        obs, info = env.reset(seed=int(seed) + index)
        observations.append(obs)
        infos.append(info)
    # Report model setup plus decoding; instance/env reset remains excluded.
    start = time.perf_counter()
    batch = stack_observations(observations)
    fixed = policy.encode(batch, cache_decoder=use_static_cache)
    n_traj = int(envs[0].unwrapped.n_traj)
    done = np.zeros((len(envs), n_traj), dtype=bool)
    environment_transitions = 0
    trajectory_steps = np.zeros_like(done, dtype=np.int64)
    log_likelihood = torch.zeros(
        len(envs), n_traj, device=policy.device, dtype=torch.float32
    )
    for _ in range(int(max_steps)):
        batch = stack_observations(observations)
        logits = policy.logits(batch, fixed)
        distribution = (
            torch.distributions.Categorical(logits=logits)
            if decode_type == "sampling" or compute_log_likelihood
            else None
        )
        if decode_type == "greedy":
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = distribution.sample()
        environment_transitions += int(np.count_nonzero(~done))
        trajectory_steps += (~done).astype(np.int64)
        if compute_log_likelihood:
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
    incomplete_fraction = np.clip(
        1.0 - served / np.maximum(customer_count, 1.0), 0.0, 1.0
    )
    legacy_cost_km = objective + (~feasible) * (
        float(incomplete_penalty_km) * (1.0 + incomplete_fraction)
    )
    objective_value, objective_scale, vehicles_started, distance_unit_cost = (
        rollout_objective_arrays(envs, infos)
    )
    # The AM auxiliary parameter retains its historical km-equivalent meaning.
    # Convert it to the active objective unit, then normalize base and penalty
    # together. Distance mode remains the original km cost divided by its scale.
    base_components = {
        "base_objective": objective_value / np.maximum(objective_scale, 1e-12),
        "base_distance_term": objective * distance_unit_cost / np.maximum(objective_scale, 1e-12),
        "base_vehicle_term": (
            objective_value - objective * distance_unit_cost
        ) / np.maximum(objective_scale, 1e-12),
    }
    if reward_contract is None:
        terminal_penalty = (
            (~feasible) * float(incomplete_penalty_km)
            * (1.0 + incomplete_fraction) * distance_unit_cost
        ) / np.maximum(objective_scale, 1e-12)
        diagnostic_components = {
            **base_components,
            "incomplete_penalty": terminal_penalty,
        }
    else:
        failed = (~feasible).astype(np.float64)
        terminal_failure_base = failed * reward_contract.failure_base
        terminal_unserved = (
            failed * reward_contract.unserved_coefficient * incomplete_fraction
        )
        terminal_penalty = terminal_failure_base + terminal_unserved
        diagnostic_components = {
            **base_components,
            "terminal_failure_base": terminal_failure_base,
            "terminal_unserved": terminal_unserved,
            "terminal_task_total": terminal_penalty,
        }
    training_cost = base_components["base_objective"] + terminal_penalty
    failure_reasons = classify_rollout_failure_reasons(
        infos,
        done=done,
        success=feasible,
        served_customers=served,
        customer_count=customer_count,
    )
    return AMRollout(
        cost_km=torch.as_tensor(legacy_cost_km, device=policy.device).float(),
        training_cost=torch.as_tensor(training_cost, device=policy.device).float(),
        objective_value=torch.as_tensor(objective_value, device=policy.device).float(),
        objective_distance_km=torch.as_tensor(objective, device=policy.device).float(),
        vehicles_started=torch.as_tensor(vehicles_started, device=policy.device).float(),
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
        training_cost_components={
            name: torch.as_tensor(value) for name, value in diagnostic_components.items()
        },
        reward_objective_scale=torch.as_tensor(objective_scale),
        failure_reasons=failure_reasons,
    )
