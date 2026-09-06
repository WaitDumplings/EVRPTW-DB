from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..AM_EVRPTW.rollout import rollout_objective_arrays, stack_observations
from ..common.method_auxiliary import MethodAuxiliaryProfile
from ..common.reward_contract import (
    RewardScaleContract,
    classify_rollout_failure_reasons,
)
from .model import EVRPTWRLPolicy


@dataclass
class EVRPTWRLRollout:
    training_cost: torch.Tensor
    objective_value: torch.Tensor
    vehicles_started: torch.Tensor
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
    # CPU-only diagnostics; never used to recompute the training objective.
    training_cost_components: dict[str, torch.Tensor] | None = None
    reward_objective_scale: torch.Tensor | None = None
    failure_reasons: np.ndarray | None = None
    method_auxiliary_diagnostics: dict[str, torch.Tensor] | None = None


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
    use_static_cache: bool = True,
    compute_log_likelihood: bool = True,
    reward_contract: RewardScaleContract | None = None,
    method_auxiliary_profile: MethodAuxiliaryProfile | None = None,
) -> EVRPTWRLRollout:
    if decode_type not in {"sampling", "greedy"}:
        raise ValueError("decode_type must be sampling or greedy")
    if reward_contract is not None:
        reward_contract.validate_envs(envs)
        if method_auxiliary_profile is None:
            raise ValueError(
                "EVRPTW-RL reward-contract rollout requires its frozen method "
                "auxiliary profile"
            )
    if method_auxiliary_profile is not None:
        method_auxiliary_profile.require_method("evrptw_rl")
        expected_profile = {
            "profile_id": "evrptw_rl_legal_station_fraction_v1",
            "applicability": "formal_training_with_shared_reward_contract",
            "aggregation": "executed_legal_station_visit_count",
            "denominator": "num_customers",
            "step_clip": None,
            "component_clip": None,
            "weights": {"station_visit": 0.3},
        }
        actual_profile = {
            "profile_id": method_auxiliary_profile.profile_id,
            "applicability": method_auxiliary_profile.applicability,
            "aggregation": method_auxiliary_profile.aggregation,
            "denominator": method_auxiliary_profile.denominator,
            "step_clip": method_auxiliary_profile.step_clip,
            "component_clip": method_auxiliary_profile.component_clip,
            "weights": dict(method_auxiliary_profile.weights),
        }
        if actual_profile != expected_profile:
            raise ValueError(
                "unsupported EVRPTW-RL method auxiliary profile semantics"
            )
        if reward_contract is None:
            raise ValueError(
                "EVRPTW-RL formal method auxiliary profile requires a reward contract"
            )
        if float(station_visit_penalty) != float(
            method_auxiliary_profile.weights["station_visit"]
        ):
            raise ValueError(
                "EVRPTW-RL station auxiliary weight disagrees with its frozen profile"
            )
    observations: list[dict[str, np.ndarray]] = []
    infos: list[dict[str, Any]] = []
    for index, env in enumerate(envs):
        observation, info = env.reset(seed=int(seed) + index)
        observations.append(observation)
        infos.append(info)
    # Include static-cache construction so cached/uncached timings are comparable.
    # Instance/env reset remains outside the reported setup-plus-decode interval.
    start = time.perf_counter()
    batch = stack_observations(observations)
    batch_size = len(envs)
    n_traj = int(envs[0].unwrapped.n_traj)
    state = policy.initial_state(batch_size, n_traj)
    travel_time = _normalized_travel_time(envs)
    fixed = policy.encode_static(batch, travel_time) if use_static_cache else None
    done = np.zeros((batch_size, n_traj), dtype=bool)
    environment_transitions = 0
    trajectory_steps = np.zeros_like(done, dtype=np.int64)
    station_visits = np.zeros((batch_size, n_traj), dtype=np.int64)
    log_likelihood = torch.zeros(batch_size, n_traj, device=policy.device)
    for _ in range(int(max_steps)):
        batch = stack_observations(observations)
        logits, state = policy.logits(batch, travel_time, state, fixed=fixed)
        distribution = (
            torch.distributions.Categorical(logits=logits)
            if decode_type == "sampling" or compute_log_likelihood
            else None
        )
        actions = (
            torch.argmax(logits, dim=-1)
            if decode_type == "greedy"
            else distribution.sample()
        )
        environment_transitions += int(np.count_nonzero(~done))
        trajectory_steps += (~done).astype(np.int64)
        if compute_log_likelihood:
            active = torch.as_tensor(~done, device=policy.device)
            log_likelihood = log_likelihood + distribution.log_prob(actions) * active
        action_array = actions.detach().cpu().numpy().astype(np.int64)

        next_observations: list[dict[str, np.ndarray]] = []
        next_infos: list[dict[str, Any]] = []
        for env_index, (env, action) in enumerate(zip(envs, action_array)):
            station_start = int(env.unwrapped.station_start)
            num_nodes = int(env.unwrapped.num_nodes)
            active = ~done[env_index]
            in_range = (action >= station_start) & (action < num_nodes)
            legal = np.zeros_like(active)
            candidates = np.flatnonzero(active & in_range)
            if candidates.size:
                action_mask = np.asarray(
                    observations[env_index]["action_mask"], dtype=bool
                )
                legal[candidates] = action_mask[candidates, action[candidates]]
            station_visits[env_index] += legal.astype(np.int64)
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
    incomplete_fraction = np.clip(
        1.0 - served / np.maximum(customer_count, 1.0), 0.0, 1.0
    )
    objective_value, objective_scale, vehicles_started, distance_unit_cost = rollout_objective_arrays(envs, infos)
    base_objective = objective_value / np.maximum(objective_scale, 1e-12)
    if method_auxiliary_profile is None:
        station_penalty = float(station_visit_penalty) * station_visits
        station_component_name = "station_visit_penalty"
        normalized_station_visits = station_visits.astype(np.float64)
    else:
        if not np.all(customer_count == customer_count[0, 0]):
            raise ValueError(
                "EVRPTW-RL station auxiliary requires fixed num_customers"
            )
        normalized_station_visits = station_visits / np.maximum(
            customer_count, 1.0
        )
        station_penalty = float(station_visit_penalty) * normalized_station_visits
        station_component_name = "station_visit_auxiliary"
    if reward_contract is None:
        terminal_penalty = (~feasible) * (
            float(incomplete_penalty) * (1.0 + incomplete_fraction)
        )
        terminal_name = "incomplete_penalty"
    else:
        failed = (~feasible).astype(np.float64)
        terminal_failure_base = failed * reward_contract.failure_base
        terminal_unserved = (
            failed * reward_contract.unserved_coefficient * incomplete_fraction
        )
        terminal_penalty = terminal_failure_base + terminal_unserved
        terminal_name = None
    training_cost = base_objective + station_penalty + terminal_penalty
    diagnostic_components = {
        "base_objective": base_objective,
        "base_distance_term": objective * distance_unit_cost / np.maximum(objective_scale, 1e-12),
        "base_vehicle_term": (
            objective_value - objective * distance_unit_cost
        ) / np.maximum(objective_scale, 1e-12),
        station_component_name: station_penalty,
    }
    if terminal_name is not None:
        diagnostic_components[terminal_name] = terminal_penalty
    else:
        diagnostic_components.update(
            {
                "terminal_failure_base": terminal_failure_base,
                "terminal_unserved": terminal_unserved,
                "terminal_task_total": terminal_penalty,
            }
        )
    failure_reasons = classify_rollout_failure_reasons(
        infos,
        done=done,
        success=feasible,
        served_customers=served,
        customer_count=customer_count,
    )
    return EVRPTWRLRollout(
        training_cost=torch.as_tensor(training_cost, device=policy.device).float(),
        objective_value=torch.as_tensor(objective_value, device=policy.device).float(),
        vehicles_started=torch.as_tensor(vehicles_started, device=policy.device).float(),
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
        training_cost_components={
            name: torch.as_tensor(value) for name, value in diagnostic_components.items()
        },
        reward_objective_scale=torch.as_tensor(objective_scale),
        failure_reasons=failure_reasons,
        method_auxiliary_diagnostics=(
            None
            if method_auxiliary_profile is None
            else {
                "station_visits_raw": torch.as_tensor(station_visits),
                "station_visit_denominator": torch.as_tensor(customer_count),
                "station_visits_normalized": torch.as_tensor(
                    normalized_station_visits
                ),
            }
        ),
    )
