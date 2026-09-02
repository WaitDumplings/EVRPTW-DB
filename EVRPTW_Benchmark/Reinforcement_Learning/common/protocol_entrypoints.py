from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .protocol_trainers import train_reinforce_data_passes
from .stage2_data import make_envs


def _max_steps(envs: list[Any]) -> int:
    return max(env.unwrapped.max_steps for env in envs)


def run_am(args: Any, pool: Any, policy: Any, optimizer: Any) -> None:
    from ..AM_EVRPTW.rollout import rollout

    def solve(active, instances, decode_type, seed):
        envs = make_envs(
            instances,
            n_traj=args.samples_per_instance if decode_type == "sampling" else 1,
            info_level="full" if decode_type == "greedy" else "light",
        )
        return rollout(
            active,
            envs,
            decode_type=decode_type,
            max_steps=_max_steps(envs),
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
            policy, instances, "sampling", seed
        ),
        make_baseline=lambda active, instances, _soft, seed: solve(
            active, instances, "greedy", seed
        ),
        training_cost=lambda result: result.cost_km,
        objective_distance=objective,
        feasible=lambda result: result.feasible,
        validation_solve=lambda active, instance, seed: solve(
            active, [instance], "greedy", seed
        ).infos[0],
        legacy_batch_size=args.batch_size,
    )


def run_evrptw_rl(args: Any, pool: Any, policy: Any, optimizer: Any) -> None:
    from ..EVRPTW_RL.rollout import rollout

    def solve(active, instances, decode_type, seed):
        envs = make_envs(
            instances,
            n_traj=args.samples_per_instance if decode_type == "sampling" else 1,
            info_level="full" if decode_type == "greedy" else "light",
        )
        return rollout(
            active,
            envs,
            decode_type=decode_type,
            max_steps=_max_steps(envs),
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
            policy, instances, "sampling", seed
        ),
        make_baseline=lambda active, instances, _soft, seed: solve(
            active, instances, "greedy", seed
        ),
        training_cost=lambda result: result.training_cost,
        objective_distance=lambda result: result.objective_distance_km,
        feasible=lambda result: result.feasible,
        validation_solve=lambda active, instance, seed: solve(
            active, [instance], "greedy", seed
        ).infos[0],
        legacy_batch_size=args.batch_size,
    )


def run_drl_ts(args: Any, pool: Any, policy: Any, optimizer: Any) -> None:
    from ..DRL_TS.rollout import rollout
    from ..DRL_TS.soft_env import DRLTSSoftConstraintEnv

    def solve(active, instances, soft, decode_type, seed):
        n_traj = args.samples_per_instance if decode_type == "sampling" else 1
        if soft:
            envs = [
                DRLTSSoftConstraintEnv(
                    instance=instance,
                    n_traj=n_traj,
                    reward_mode="distance",
                    charging_mode="station_power_full",
                    matrix_mode="canonical",
                    info_level="full" if decode_type == "greedy" else "light",
                )
                for instance in instances
            ]
        else:
            envs = make_envs(
                instances,
                n_traj=n_traj,
                info_level="full" if decode_type == "greedy" else "light",
            )
        return rollout(
            active,
            envs,
            decode_type=decode_type,
            max_steps=_max_steps(envs),
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
            policy, instances, soft, "sampling", seed
        ),
        make_baseline=lambda active, instances, soft, seed: solve(
            active, instances, soft, "greedy", seed
        ),
        training_cost=lambda result: result.training_cost,
        objective_distance=lambda result: result.objective_distance_km,
        feasible=lambda result: result.feasible,
        validation_solve=lambda active, instance, seed: solve(
            active, [instance], False, "greedy", seed
        ).infos[0],
        legacy_batch_size=args.batch_size,
        soft_stage_fraction=args.soft_stage_fraction,
    )


__all__ = ["run_am", "run_drl_ts", "run_evrptw_rl"]
