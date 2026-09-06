from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..AM_EVRPTW.rollout import rollout_objective_arrays, stack_observations
from ..common.reward_contract import (
    RewardScaleContract,
    classify_rollout_failure_reasons,
)
from .model import DRLTSPolicy
from .soft_env import (
    DEFAULT_SOFT_VIOLATION_COMPONENT_CLIP,
    DEFAULT_SOFT_VIOLATION_STEP_CLIP,
    SOFT_VIOLATION_CONTRACT_ID,
    SOFT_VIOLATION_DENOMINATOR,
)


@dataclass
class DRLTSRollout:
    training_cost: torch.Tensor
    objective_value: torch.Tensor
    vehicles_started: torch.Tensor
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
    trajectory_steps: torch.Tensor
    rollout_budget_exhausted: torch.Tensor
    # CPU-only diagnostics; never used to recompute the training objective.
    training_cost_components: dict[str, torch.Tensor] | None = None
    reward_objective_scale: torch.Tensor | None = None
    failure_reasons: np.ndarray | None = None
    # Raw sums/counts and pre-weight bounded components for scale audits.
    soft_violation_diagnostics: dict[str, torch.Tensor] | None = None


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
        distance_rows.append(
            distance / max(float(unwrapped.reward_distance_scale_km), 1e-12)
        )
        time_rows.append(travel_time / max(float(unwrapped.horizon_s), 1e-12))
        energy_rows.append(
            energy / max(float(unwrapped.battery_capacity_kwh), 1e-12)
        )
    return (
        np.stack(distance_rows, axis=0),
        np.stack(time_rows, axis=0),
        np.stack(energy_rows, axis=0),
    )


def bounded_soft_violation_component(
    clipped_transition_sum: np.ndarray,
    num_customers: np.ndarray,
    component_clip: float,
) -> np.ndarray:
    """Normalize an accumulated soft violation without a dilution shortcut.

    The denominator is the fixed instance customer count, never the number of
    actions taken. Thus adding a zero-violation depot/station transition cannot
    make an already incurred violation cheaper. The caller has already clipped
    each normalized transition excess; this final clip keeps each component
    bounded even when a route uses more than ``N`` travel transitions.
    """

    values = np.asarray(clipped_transition_sum, dtype=np.float64)
    denominator = np.asarray(num_customers, dtype=np.float64)
    clip = float(component_clip)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("clipped soft violation sums must be finite and nonnegative")
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise ValueError("soft violation normalization requires positive customer counts")
    if not np.isfinite(clip) or clip <= 0.0:
        raise ValueError("soft violation component clip must be finite and positive")
    return np.minimum(values / denominator, clip)


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
    compute_log_likelihood: bool = True,
    reward_contract: RewardScaleContract | None = None,
    soft_violation_contract_id: str = SOFT_VIOLATION_CONTRACT_ID,
    soft_violation_step_clip: float = DEFAULT_SOFT_VIOLATION_STEP_CLIP,
    soft_violation_component_clip: float = DEFAULT_SOFT_VIOLATION_COMPONENT_CLIP,
    soft_violation_denominator: str = SOFT_VIOLATION_DENOMINATOR,
) -> DRLTSRollout:
    if decode_type not in {"sampling", "greedy"}:
        raise ValueError("decode_type must be sampling or greedy")
    if reward_contract is not None:
        reward_contract.validate_envs(envs)
    if str(soft_violation_contract_id) != SOFT_VIOLATION_CONTRACT_ID:
        raise ValueError(
            "unsupported DRL-TS soft violation contract: "
            f"{soft_violation_contract_id!r}"
        )
    if str(soft_violation_denominator) != SOFT_VIOLATION_DENOMINATOR:
        raise ValueError("DRL-TS soft violation denominator must be num_customers")
    step_clip = float(soft_violation_step_clip)
    component_clip = float(soft_violation_component_clip)
    if not np.isfinite(step_clip) or step_clip <= 0.0:
        raise ValueError("soft violation step clip must be finite and positive")
    if not np.isfinite(component_clip) or component_clip <= 0.0:
        raise ValueError("soft violation component clip must be finite and positive")
    for name, value in (
        ("capacity_penalty", capacity_penalty),
        ("time_penalty", time_penalty),
        ("energy_penalty", energy_penalty),
    ):
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    observations: list[dict[str, np.ndarray]] = []
    infos: list[dict[str, Any]] = []
    for index, env in enumerate(envs):
        observation, info = env.reset(seed=int(seed) + index)
        observations.append(observation)
        infos.append(info)
    # Report edge normalization/encoding and decoding, excluding env reset.
    started = time.perf_counter()
    batch = stack_observations(observations)
    distance, travel_time, energy = normalized_edge_matrices(envs)
    fixed = policy.encode(batch, distance, travel_time, energy)
    batch_size = len(envs)
    n_traj = int(envs[0].unwrapped.n_traj)
    state = policy.initial_state(batch_size, n_traj)
    done = np.zeros((batch_size, n_traj), dtype=bool)
    environment_transitions = 0
    trajectory_steps = np.zeros_like(done, dtype=np.int64)
    log_likelihood = torch.zeros(batch_size, n_traj, device=policy.device)
    for _ in range(int(max_steps)):
        batch = stack_observations(observations)
        logits, state = policy.logits(batch, fixed, state)
        distribution = (
            torch.distributions.Categorical(logits=logits)
            if compute_log_likelihood or decode_type == "sampling"
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
    zero_counts = np.zeros_like(objective, dtype=np.int64)
    if soft_constraints:
        raw_capacity_violation = np.stack(
            [
                np.asarray(
                    info["capacity_violation_raw_sum"],
                    dtype=np.float64,
                )
                for info in infos
            ]
        )
        raw_time_violation = np.stack(
            [
                np.asarray(info["time_violation_raw_sum"], dtype=np.float64)
                for info in infos
            ]
        )
        raw_energy_violation = np.stack(
            [
                np.asarray(info["energy_violation_raw_sum"], dtype=np.float64)
                for info in infos
            ]
        )
        clipped_capacity_sum = np.stack(
            [
                np.asarray(info["capacity_violation_clipped_sum"], dtype=np.float64)
                for info in infos
            ]
        )
        clipped_time_sum = np.stack(
            [
                np.asarray(info["time_violation_clipped_sum"], dtype=np.float64)
                for info in infos
            ]
        )
        clipped_energy_sum = np.stack(
            [
                np.asarray(info["energy_violation_clipped_sum"], dtype=np.float64)
                for info in infos
            ]
        )
        capacity_applicable = np.stack(
            [
                np.asarray(
                    info["capacity_violation_applicable_transitions"],
                    dtype=np.int64,
                )
                for info in infos
            ]
        )
        time_applicable = np.stack(
            [
                np.asarray(
                    info["time_violation_applicable_transitions"], dtype=np.int64
                )
                for info in infos
            ]
        )
        energy_applicable = np.stack(
            [
                np.asarray(
                    info["energy_violation_applicable_transitions"], dtype=np.int64
                )
                for info in infos
            ]
        )
        for info in infos:
            if info.get("soft_violation_contract_id") != SOFT_VIOLATION_CONTRACT_ID:
                raise RuntimeError("soft environment contract id mismatch")
            if float(info.get("soft_violation_step_clip", np.nan)) != step_clip:
                raise RuntimeError("soft environment step clip mismatch")
            if info.get("soft_violation_denominator") != SOFT_VIOLATION_DENOMINATOR:
                raise RuntimeError("soft environment denominator mismatch")
        customer_denominator = np.asarray(
            [env.unwrapped.num_customers for env in envs], dtype=np.float64
        )[:, None]
        if np.any(customer_denominator <= 0.0):
            raise RuntimeError("soft violation normalization requires customers")
        # Fixed N (rather than observed transition count) prevents a policy from
        # diluting an old violation by inserting zero-violation depot/CS travel.
        capacity_violation = bounded_soft_violation_component(
            clipped_capacity_sum, customer_denominator, component_clip
        )
        time_violation = bounded_soft_violation_component(
            clipped_time_sum, customer_denominator, component_clip
        )
        energy_violation = bounded_soft_violation_component(
            clipped_energy_sum, customer_denominator, component_clip
        )
    else:
        raw_capacity_violation = zeros.copy()
        raw_time_violation = zeros.copy()
        raw_energy_violation = zeros.copy()
        clipped_capacity_sum = zeros.copy()
        clipped_time_sum = zeros.copy()
        clipped_energy_sum = zeros.copy()
        capacity_applicable = zero_counts.copy()
        time_applicable = zero_counts.copy()
        energy_applicable = zero_counts.copy()
        capacity_violation = zeros.copy()
        time_violation = zeros.copy()
        energy_violation = zeros.copy()
    no_violation = (
        (raw_capacity_violation <= 1e-9)
        & (raw_time_violation <= 1e-9)
        & (raw_energy_violation <= 1e-9)
    )
    feasible = completed & no_violation
    customer_count = np.asarray(
        [env.unwrapped.num_customers for env in envs],
        dtype=np.float64,
    )[:, None]
    incomplete_fraction = np.clip(
        1.0 - served / np.maximum(customer_count, 1.0), 0.0, 1.0
    )
    objective_value, objective_scale, vehicles_started, distance_unit_cost = rollout_objective_arrays(envs, infos)
    base_objective = objective_value / np.maximum(objective_scale, 1e-12)
    if reward_contract is None:
        terminal_penalty = (~completed) * (
            float(incomplete_penalty) * (1.0 + incomplete_fraction)
        )
        terminal_name = "incomplete_penalty"
    else:
        # Stage 1 deliberately allows constraint violations.  A completed
        # soft trajectory pays its measured violation terms but not the hard
        # failure floor; every incomplete soft/hard trajectory pays it once.
        failed = (~completed).astype(np.float64)
        terminal_failure_base = failed * reward_contract.failure_base
        terminal_unserved = (
            failed * reward_contract.unserved_coefficient * incomplete_fraction
        )
        terminal_penalty = terminal_failure_base + terminal_unserved
        terminal_name = None
    capacity_auxiliary = float(capacity_penalty) * capacity_violation
    time_auxiliary = float(time_penalty) * time_violation
    energy_auxiliary = float(energy_penalty) * energy_violation
    soft_auxiliary_total = capacity_auxiliary + time_auxiliary + energy_auxiliary
    training_cost = base_objective + soft_auxiliary_total + terminal_penalty
    diagnostic_components = {
        "base_objective": base_objective,
        "base_distance_term": objective * distance_unit_cost / np.maximum(objective_scale, 1e-12),
        "base_vehicle_term": (
            objective_value - objective * distance_unit_cost
        ) / np.maximum(objective_scale, 1e-12),
        "capacity_penalty": capacity_auxiliary,
        "time_penalty": time_auxiliary,
        "energy_penalty": energy_auxiliary,
        "soft_auxiliary_total": soft_auxiliary_total,
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
        success=completed,
        served_customers=served,
        customer_count=customer_count,
    )
    if soft_constraints:
        completed_with_violation = completed & (~feasible)
        failure_reasons[completed_with_violation] = "completed_with_soft_violation"
        for row, info in enumerate(infos):
            info["failure_reason"] = failure_reasons[row].copy()
    return DRLTSRollout(
        training_cost=torch.as_tensor(training_cost, device=policy.device).float(),
        objective_value=torch.as_tensor(objective_value, device=policy.device).float(),
        vehicles_started=torch.as_tensor(vehicles_started, device=policy.device).float(),
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
        soft_violation_diagnostics={
            "soft_capacity_raw_sum": torch.as_tensor(raw_capacity_violation),
            "soft_time_raw_sum": torch.as_tensor(raw_time_violation),
            "soft_energy_raw_sum": torch.as_tensor(raw_energy_violation),
            "soft_capacity_clipped_sum": torch.as_tensor(clipped_capacity_sum),
            "soft_time_clipped_sum": torch.as_tensor(clipped_time_sum),
            "soft_energy_clipped_sum": torch.as_tensor(clipped_energy_sum),
            "soft_capacity_applicable_transitions": torch.as_tensor(
                capacity_applicable
            ),
            "soft_time_applicable_transitions": torch.as_tensor(time_applicable),
            "soft_energy_applicable_transitions": torch.as_tensor(
                energy_applicable
            ),
            "soft_capacity_component_unweighted": torch.as_tensor(
                capacity_violation
            ),
            "soft_time_component_unweighted": torch.as_tensor(time_violation),
            "soft_energy_component_unweighted": torch.as_tensor(energy_violation),
        },
    )


__all__ = [
    "DRLTSRollout",
    "bounded_soft_violation_component",
    "normalized_edge_matrices",
    "rollout",
]
