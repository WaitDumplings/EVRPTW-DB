from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from .protocol_trainers import train_reinforce_data_passes
from .route_info import finalize_route_infos
from .stage2_data import make_envs
from .objective import objective_from_args
from .method_auxiliary import method_auxiliary_from_args
from .reward_contract import reward_contract_from_args
from .training_protocol import require_training_rollout_steps, require_validation_decoding


def _max_steps(envs: list[Any]) -> int:
    return max(env.unwrapped.max_steps for env in envs)


def _finalize_validation_result(envs: list[Any], result: Any) -> None:
    # Full-info route export used to run inside the timed rollout loop. Keep
    # its deferred terminal cost in the same reported inference-time boundary.
    started = time.perf_counter()
    result.infos = finalize_route_infos(envs, result.infos)
    result.runtime_s += time.perf_counter() - started


def _training_reward_scale(args: Any, pool: Any) -> float:
    mode = "single_customer_repair_median"
    scale = float(pool.reward_distance_scale_km(mode))
    args.reward_distance_scale_mode = f"dataset_{mode}"
    args.reward_distance_scale_km = scale
    args.reward_distance_scale_metadata = dict(pool.reward_scale_metadata)
    return scale


def run_am(args: Any, pool: Any, policy: Any, optimizer: Any) -> None:
    from ..AM_EVRPTW.rollout import rollout

    training_rollout_steps = require_training_rollout_steps(args)
    validation_decode_type, validation_candidates = require_validation_decoding(args)
    reward_distance_scale_km = _training_reward_scale(args, pool)
    objective_config = objective_from_args(args)
    reward_contract = reward_contract_from_args(
        args, objective=objective_config, scale=getattr(args, "scale", None)
    )

    def solve(
        active, instances, decode_type, seed, max_steps=None, candidate_count=None,
    ):
        n_traj = (
            int(candidate_count)
            if candidate_count is not None
            else (args.samples_per_instance if decode_type == "sampling" else 1)
        )
        envs = make_envs(
            instances,
            n_traj=n_traj,
            info_level="light",
            reward_distance_scale_km=reward_distance_scale_km,
            reward_objective_scale=(
                reward_contract.objective_scale if reward_contract else None
            ),
            invalid_action_penalty=0.0 if reward_contract else -10.0,
            objective_config=objective_config,
        )
        result = rollout(
            active,
            envs,
            decode_type=decode_type,
            max_steps=_max_steps(envs) if max_steps is None else int(max_steps),
            seed=seed,
            compute_log_likelihood=(
                candidate_count is None and decode_type == "sampling"
            ),
            incomplete_penalty_km=args.incomplete_penalty_km,
            reward_contract=reward_contract,
        )
        if candidate_count is not None:
            _finalize_validation_result(envs, result)
        return result

    def objective(result):
        values = np.stack(
            [np.asarray(info["objective_distance_km"]) for info in result.infos]
        )
        return torch.as_tensor(values, device=policy.device).float()

    train_reinforce_data_passes(
        method="AM-EVRPTW",
        args=args,
        pool=pool,
        policy=policy,
        optimizer=optimizer,
        make_actor=lambda instances, _soft, seed: solve(
            policy, instances, "sampling", seed, training_rollout_steps
        ),
        make_baseline=lambda active, instances, _soft, seed: solve(
            active, instances, "greedy", seed, training_rollout_steps
        ),
        training_cost=lambda result: result.training_cost,
        objective_distance=objective,
        feasible=lambda result: result.feasible,
        validation_solve=lambda active, instance, seed: solve(
            active,
            [instance],
            validation_decode_type,
            seed,
            candidate_count=validation_candidates,
        ).infos[0],
        legacy_batch_size=args.batch_size,
    )


def run_evrptw_rl(args: Any, pool: Any, policy: Any, optimizer: Any) -> None:
    from ..EVRPTW_RL.rollout import rollout

    training_rollout_steps = require_training_rollout_steps(args)
    validation_decode_type, validation_candidates = require_validation_decoding(args)
    reward_distance_scale_km = _training_reward_scale(args, pool)
    objective_config = objective_from_args(args)
    reward_contract = reward_contract_from_args(
        args, objective=objective_config, scale=getattr(args, "scale", None)
    )
    method_auxiliary_profile = method_auxiliary_from_args(
        args, expected_method="evrptw_rl"
    )
    if reward_contract is not None and method_auxiliary_profile is None:
        raise ValueError(
            "formal EVRPTW-RL reward-contract training requires "
            "--method-auxiliary-profile"
        )

    def solve(
        active, instances, decode_type, seed, max_steps=None, candidate_count=None,
    ):
        n_traj = (
            int(candidate_count)
            if candidate_count is not None
            else (args.samples_per_instance if decode_type == "sampling" else 1)
        )
        envs = make_envs(
            instances,
            n_traj=n_traj,
            info_level="light",
            reward_distance_scale_km=reward_distance_scale_km,
            reward_objective_scale=(
                reward_contract.objective_scale if reward_contract else None
            ),
            invalid_action_penalty=0.0 if reward_contract else -10.0,
            objective_config=objective_config,
        )
        result = rollout(
            active,
            envs,
            decode_type=decode_type,
            max_steps=_max_steps(envs) if max_steps is None else int(max_steps),
            seed=seed,
            compute_log_likelihood=(
                candidate_count is None and decode_type == "sampling"
            ),
            station_visit_penalty=args.station_visit_penalty,
            incomplete_penalty=args.incomplete_penalty,
            reward_contract=reward_contract,
            method_auxiliary_profile=method_auxiliary_profile,
        )
        if candidate_count is not None:
            _finalize_validation_result(envs, result)
        return result

    train_reinforce_data_passes(
        method="EVRPTW-RL",
        args=args,
        pool=pool,
        policy=policy,
        optimizer=optimizer,
        make_actor=lambda instances, _soft, seed: solve(
            policy, instances, "sampling", seed, training_rollout_steps
        ),
        make_baseline=lambda active, instances, _soft, seed: solve(
            active, instances, "greedy", seed, training_rollout_steps
        ),
        training_cost=lambda result: result.training_cost,
        objective_distance=lambda result: result.objective_distance_km,
        feasible=lambda result: result.feasible,
        validation_solve=lambda active, instance, seed: solve(
            active,
            [instance],
            validation_decode_type,
            seed,
            candidate_count=validation_candidates,
        ).infos[0],
        legacy_batch_size=args.batch_size,
    )


def run_drl_ts(args: Any, pool: Any, policy: Any, optimizer: Any) -> None:
    from ..DRL_TS.env import DRLTSHardConstraintEnv
    from ..DRL_TS.rollout import rollout
    from ..DRL_TS.soft_env import (
        DEFAULT_SOFT_VIOLATION_COMPONENT_CLIP,
        DEFAULT_SOFT_VIOLATION_STEP_CLIP,
        DRLTSSoftConstraintEnv,
        SOFT_VIOLATION_AGGREGATION,
        SOFT_VIOLATION_APPLICABILITY,
        SOFT_VIOLATION_CONTRACT_ID,
        SOFT_VIOLATION_DENOMINATOR,
    )

    training_rollout_steps = require_training_rollout_steps(args)
    validation_decode_type, validation_candidates = require_validation_decoding(args)
    reward_distance_scale_km = _training_reward_scale(args, pool)
    objective_config = objective_from_args(args)
    reward_contract = reward_contract_from_args(
        args, objective=objective_config, scale=getattr(args, "scale", None)
    )
    method_auxiliary_profile = method_auxiliary_from_args(
        args, expected_method="drl_ts"
    )
    if reward_contract is not None and method_auxiliary_profile is None:
        raise ValueError(
            "formal DRL-TS reward-contract training requires "
            "--method-auxiliary-profile"
        )
    if reward_contract is None and method_auxiliary_profile is not None:
        raise ValueError(
            "DRL-TS formal method auxiliary profile requires a reward contract"
        )
    if method_auxiliary_profile is not None:
        expected_profile = {
            "profile_id": SOFT_VIOLATION_CONTRACT_ID,
            "applicability": SOFT_VIOLATION_APPLICABILITY,
            "aggregation": SOFT_VIOLATION_AGGREGATION,
            "denominator": SOFT_VIOLATION_DENOMINATOR,
            "weights": {
                "capacity": 1.0,
                "energy": 1.0,
                "time_window": 1.0,
            },
        }
        actual_profile = {
            "profile_id": method_auxiliary_profile.profile_id,
            "applicability": method_auxiliary_profile.applicability,
            "aggregation": method_auxiliary_profile.aggregation,
            "denominator": method_auxiliary_profile.denominator,
            "weights": dict(method_auxiliary_profile.weights),
        }
        if (
            actual_profile != expected_profile
            or method_auxiliary_profile.step_clip is None
            or method_auxiliary_profile.component_clip is None
        ):
            raise ValueError(
                f"unsupported DRL-TS soft auxiliary profile: {actual_profile!r}"
            )
        args.soft_violation_contract_id = method_auxiliary_profile.profile_id
        args.soft_violation_step_clip = float(method_auxiliary_profile.step_clip)
        args.soft_violation_component_clip = float(
            method_auxiliary_profile.component_clip
        )
        args.soft_violation_denominator = method_auxiliary_profile.denominator
        args.capacity_penalty = float(method_auxiliary_profile.weights["capacity"])
        args.time_penalty = float(method_auxiliary_profile.weights["time_window"])
        args.energy_penalty = float(method_auxiliary_profile.weights["energy"])
    soft_violation_kwargs = {
        "soft_violation_contract_id": getattr(
            args, "soft_violation_contract_id", SOFT_VIOLATION_CONTRACT_ID
        ),
        "soft_violation_step_clip": float(
            getattr(
                args,
                "soft_violation_step_clip",
                DEFAULT_SOFT_VIOLATION_STEP_CLIP,
            )
        ),
        "soft_violation_component_clip": float(
            getattr(
                args,
                "soft_violation_component_clip",
                DEFAULT_SOFT_VIOLATION_COMPONENT_CLIP,
            )
        ),
        "soft_violation_denominator": getattr(
            args, "soft_violation_denominator", SOFT_VIOLATION_DENOMINATOR
        ),
    }
    for field, value in soft_violation_kwargs.items():
        setattr(args, field, value)

    def solve(
        active, instances, soft, decode_type, seed, max_steps=None,
        candidate_count=None,
    ):
        n_traj = (
            int(candidate_count)
            if candidate_count is not None
            else (args.samples_per_instance if decode_type == "sampling" else 1)
        )
        if soft:
            envs = [
                DRLTSSoftConstraintEnv(
                    instance=instance,
                    n_traj=n_traj,
                    reward_mode="distance",
                    charging_mode="station_power_full",
                    matrix_mode="canonical",
                    info_level="light",
                    reward_distance_scale_km=reward_distance_scale_km,
                    reward_objective_scale=(
                        reward_contract.objective_scale if reward_contract else None
                    ),
                    invalid_action_penalty=0.0 if reward_contract else -10.0,
                    objective_config=objective_config,
                    **{
                        key: value
                        for key, value in soft_violation_kwargs.items()
                        if key != "soft_violation_component_clip"
                    },
                )
                for instance in instances
            ]
        else:
            envs = [
                DRLTSHardConstraintEnv(
                    instance=instance,
                    n_traj=n_traj,
                    reward_mode="distance",
                    charging_mode="station_power_full",
                    matrix_mode="canonical",
                    info_level="light",
                    reward_distance_scale_km=reward_distance_scale_km,
                    reward_objective_scale=(
                        reward_contract.objective_scale if reward_contract else None
                    ),
                    invalid_action_penalty=0.0 if reward_contract else -10.0,
                    objective_config=objective_config,
                )
                for instance in instances
            ]
        result = rollout(
            active,
            envs,
            decode_type=decode_type,
            max_steps=_max_steps(envs) if max_steps is None else int(max_steps),
            seed=seed,
            compute_log_likelihood=(
                candidate_count is None and decode_type == "sampling"
            ),
            soft_constraints=soft,
            capacity_penalty=args.capacity_penalty,
            time_penalty=args.time_penalty,
            energy_penalty=args.energy_penalty,
            incomplete_penalty=args.incomplete_penalty,
            reward_contract=reward_contract,
            **soft_violation_kwargs,
        )
        if candidate_count is not None:
            _finalize_validation_result(envs, result)
        return result

    train_reinforce_data_passes(
        method="DRL-TS",
        args=args,
        pool=pool,
        policy=policy,
        optimizer=optimizer,
        make_actor=lambda instances, soft, seed: solve(
            policy, instances, soft, "sampling", seed, training_rollout_steps
        ),
        make_baseline=lambda active, instances, soft, seed: solve(
            active, instances, soft, "greedy", seed, training_rollout_steps
        ),
        training_cost=lambda result: result.training_cost,
        objective_distance=lambda result: result.objective_distance_km,
        feasible=lambda result: result.feasible,
        validation_solve=lambda active, instance, seed: solve(
            active,
            [instance],
            False,
            validation_decode_type,
            seed,
            candidate_count=validation_candidates,
        ).infos[0],
        legacy_batch_size=args.batch_size,
        soft_stage_fraction=args.soft_stage_fraction,
        soft_stage_end_epoch=getattr(args, "soft_stage_end_epoch", None),
    )


__all__ = ["run_am", "run_drl_ts", "run_evrptw_rl"]
