from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .protocol_trainers import train_reinforce_data_passes
from .stage2_data import make_envs
from .training_protocol import require_training_rollout_steps, require_validation_decoding


def _max_steps(envs: list[Any]) -> int:
    return max(env.unwrapped.max_steps for env in envs)


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
            info_level=(
                "full"
                if candidate_count is not None or decode_type == "greedy"
                else "light"
            ),
            reward_distance_scale_km=reward_distance_scale_km,
        )
        return rollout(
            active,
            envs,
            decode_type=decode_type,
            max_steps=_max_steps(envs) if max_steps is None else int(max_steps),
            seed=seed,
            incomplete_penalty_km=args.incomplete_penalty_km,
        )

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
        training_cost=lambda result: result.cost_km / reward_distance_scale_km,
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
            info_level=(
                "full"
                if candidate_count is not None or decode_type == "greedy"
                else "light"
            ),
            reward_distance_scale_km=reward_distance_scale_km,
        )
        return rollout(
            active,
            envs,
            decode_type=decode_type,
            max_steps=_max_steps(envs) if max_steps is None else int(max_steps),
            seed=seed,
            station_visit_penalty=args.station_visit_penalty,
            incomplete_penalty=args.incomplete_penalty,
        )

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
    from ..DRL_TS.soft_env import DRLTSSoftConstraintEnv

    training_rollout_steps = require_training_rollout_steps(args)
    validation_decode_type, validation_candidates = require_validation_decoding(args)
    reward_distance_scale_km = _training_reward_scale(args, pool)

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
                    info_level=(
                        "full"
                        if candidate_count is not None or decode_type == "greedy"
                        else "light"
                    ),
                    reward_distance_scale_km=reward_distance_scale_km,
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
                    info_level=(
                        "full"
                        if candidate_count is not None or decode_type == "greedy"
                        else "light"
                    ),
                    reward_distance_scale_km=reward_distance_scale_km,
                )
                for instance in instances
            ]
        return rollout(
            active,
            envs,
            decode_type=decode_type,
            max_steps=_max_steps(envs) if max_steps is None else int(max_steps),
            seed=seed,
            soft_constraints=soft,
            capacity_penalty=args.capacity_penalty,
            time_penalty=args.time_penalty,
            energy_penalty=args.energy_penalty,
            incomplete_penalty=args.incomplete_penalty,
        )

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
    )


__all__ = ["run_am", "run_drl_ts", "run_evrptw_rl"]
