from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))
sys.path.insert(0, str(REPO_ROOT))

from evrptw_core.io import load_instance
from gymnasium import Wrapper

from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import (
    EVRPTWVectorEnv,
    EVRPTWVectorEnvFast,
)

from .pbrs import PotentialRewardConfig, PotentialRewardWrapper


class TERRANRolloutHorizonWrapper(Wrapper):
    """Turn TERRAN's trainer rollout budget into a Gymnasium truncation.

    The shared EVRPTW environment has its own safety horizon.  Formal TERRAN
    training may intentionally collect fewer transitions, so reaching that
    registered budget must be surfaced as a terminal transition.  The outer
    PBRS wrapper can then apply its remaining-customer failure penalty.
    """

    def __init__(self, env, max_rollout_steps: int) -> None:
        super().__init__(env)
        self.max_rollout_steps = int(max_rollout_steps)
        if self.max_rollout_steps <= 0:
            raise ValueError("max_rollout_steps must be positive")
        self._elapsed_steps = 0

    def reset(self, **kwargs: Any):
        self._elapsed_steps = 0
        obs, info = self.env.reset(**kwargs)
        out_info = dict(info)
        out_info["rollout_horizon_steps"] = self.max_rollout_steps
        return obs, out_info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        terminated_array = np.asarray(terminated, dtype=bool)
        truncated_array = np.asarray(truncated, dtype=bool)
        budget_exhausted = np.zeros_like(terminated_array, dtype=bool)
        if self._elapsed_steps >= self.max_rollout_steps:
            budget_exhausted = ~(terminated_array | truncated_array)
            truncated_array = truncated_array | budget_exhausted

        served = np.asarray(info["served_customers"], dtype=np.int32)
        num_customers = max(int(getattr(self.unwrapped, "num_customers", 0)), 1)
        remaining = np.maximum(num_customers - served, 0)
        out_info = dict(info)
        out_info["rollout_horizon_steps"] = self.max_rollout_steps
        out_info["rollout_budget_exhausted"] = budget_exhausted.copy()
        out_info["remaining_customers"] = remaining.astype(np.int32, copy=False)
        out_info["remaining_customer_fraction"] = (
            remaining.astype(np.float32) / float(num_customers)
        )
        return obs, reward, terminated_array, truncated_array, out_info


class OnlineInstanceResetWrapper(Wrapper):
    """Refresh the wrapped EVRPTW env with a new sampled instance at reset."""

    def __init__(self, env, instance_sampler: Callable[[], Any]):
        super().__init__(env)
        self.instance_sampler = instance_sampler
        self._bootstrap_pending = True

    def reset(self, **kwargs: Any):
        if self._bootstrap_pending:
            self._bootstrap_pending = False
            return self.env.reset(**kwargs)
        options = dict(kwargs.pop("options", {}) or {})
        options["instance"] = self.instance_sampler()
        return self.env.reset(options=options, **kwargs)


def make_terran_env(
    instance_path: str | Path | None = None,
    instance: Any | None = None,
    instance_sampler: Callable[[], Any] | None = None,
    n_traj: int = 50,
    reward_mode: str = "distance",
    pbrs_config: PotentialRewardConfig | None = None,
    rollout_horizon_steps: int | None = None,
    **env_kwargs: Any,
):
    """Create the shared EVRPTW env with optional online sampling and PBRS."""
    if instance is None and instance_sampler is None:
        if instance_path is None:
            raise ValueError("Provide instance, instance_path, or instance_sampler.")
        instance = load_instance(instance_path)
    if instance is None and instance_sampler is not None:
        instance = instance_sampler()

    use_fast_env = bool(env_kwargs.pop("use_fast_env", True))
    info_level = str(env_kwargs.pop("info_level", "full"))
    use_jit_mask = bool(env_kwargs.pop("use_jit_mask", True))
    if use_fast_env:
        env = EVRPTWVectorEnvFast(
            instance=instance,
            n_traj=n_traj,
            reward_mode=reward_mode,
            info_level=info_level,
            use_jit_mask=use_jit_mask,
            **env_kwargs,
        )
    else:
        env = EVRPTWVectorEnv(instance=instance, n_traj=n_traj, reward_mode=reward_mode, **env_kwargs)
    if instance_sampler is not None:
        env = OnlineInstanceResetWrapper(env, instance_sampler)
    if rollout_horizon_steps is not None:
        env = TERRANRolloutHorizonWrapper(env, rollout_horizon_steps)
    if pbrs_config is not None and (
        pbrs_config.use_customer_pbrs
        or pbrs_config.use_repair_distance_pbrs
        or pbrs_config.use_feasible_ratio_pbrs
        or pbrs_config.use_terminal_heuristic
    ):
        env = PotentialRewardWrapper(env, pbrs_config)
    return env


__all__ = [
    "OnlineInstanceResetWrapper",
    "TERRANRolloutHorizonWrapper",
    "make_terran_env",
]
