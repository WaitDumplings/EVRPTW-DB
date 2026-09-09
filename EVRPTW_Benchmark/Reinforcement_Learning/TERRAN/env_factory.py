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
from gymnasium import Wrapper, spaces

from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import (
    EVRPTWVectorEnv,
    EVRPTWVectorEnvFast,
)

from .pbrs import PotentialRewardConfig, PotentialRewardWrapper


class StableCostObservationWrapper(Wrapper):
    """Raw economic transitions and sufficient compact state for two critics.

    Service completion is a separate event, never an economic reward.  The
    visited ledger (rather than demand or feasibility) defines remaining work.
    """

    training_mode = "stable_cost_v1"

    def __init__(self, env, rollout_horizon_steps: int | None = None):
        super().__init__(env)
        if not self.unwrapped.objective_config.is_cost:
            raise ValueError("stable_cost_v1 requires an economic cost objective")
        self.unwrapped.training_mode = self.training_mode
        self.rollout_horizon_steps = rollout_horizon_steps
        self._elapsed_steps = 0
        self._last_objective = None
        self._finished = None
        self._last_result = None

    def _augment(self, obs, info, finished, newly_finished=None):
        base = self.unwrapped
        limit = int(base.max_steps)
        if self.rollout_horizon_steps is not None:
            limit = min(limit, int(self.rollout_horizon_steps))
        remaining = max(limit - self._elapsed_steps, 0)
        unserved = (~base.visited[:, 1 : 1 + base.num_customers]).copy()
        out_obs = dict(obs)
        out_obs.update(
            customer_unserved=unserved,
            dispatch_paid=(base.last != 0).copy(),
            remaining_step_budget=np.where(finished, 0, remaining).astype(np.float32),
            episode_step_budget=np.full(base.n_traj, limit, dtype=np.float32),
        )
        success = np.asarray(info["success"], dtype=bool)
        out_info = dict(info)
        out_info.update(
            training_mode=self.training_mode,
            reward_unit="USD",
            failure_terminal=(
                np.zeros(base.n_traj, dtype=bool)
                if newly_finished is None else newly_finished & ~success
            ),
            remaining_customers=unserved.sum(axis=-1).astype(np.int32),
            unserved_fraction=(unserved.sum(axis=-1) / max(base.num_customers, 1)).astype(np.float32),
        )
        return out_obs, out_info

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        base = self.unwrapped
        self._elapsed_steps = 0
        self._last_result = None
        self._last_objective = np.asarray(info["objective_value"], dtype=np.float64).copy()
        self._finished = np.zeros(base.n_traj, dtype=bool)
        # Online reset may change customer count, so spaces follow the instance.
        self.observation_space = spaces.Dict({
            **self.env.observation_space.spaces,
            "customer_unserved": spaces.MultiBinary([base.n_traj, base.num_customers]),
            "dispatch_paid": spaces.MultiBinary(base.n_traj),
            "remaining_step_budget": spaces.Box(0.0, np.inf, (base.n_traj,), dtype=np.float32),
            "episode_step_budget": spaces.Box(0.0, np.inf, (base.n_traj,), dtype=np.float32),
        })
        return self._augment(obs, info, base.terminated | base.truncated)

    def step(self, action):
        if self._last_objective is None:
            raise RuntimeError("StableCostObservationWrapper.step called before reset")
        if self._last_result is not None and self._finished.all():
            # The horizon wrapper truncates externally. Do not let subsequent
            # padded calls advance the underlying environment past that limit.
            obs, terminated, truncated, info = self._last_result
            info = {**info, "failure_terminal": np.zeros_like(self._finished)}
            return obs, np.zeros_like(self._finished, dtype=np.float32), terminated.copy(), truncated.copy(), info
        obs, _, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        objective = np.asarray(info["objective_value"], dtype=np.float64)
        reward = -(objective - self._last_objective).astype(np.float32)
        finished = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        newly_finished = finished & ~self._finished
        self._last_objective = objective.copy()
        self._finished = finished.copy()
        obs, info = self._augment(obs, info, finished, newly_finished)
        self._last_result = (obs, np.asarray(terminated).copy(), np.asarray(truncated).copy(), info)
        return obs, reward, terminated, truncated, info


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
        failure_reason = np.asarray(
            info.get(
                "failure_reason",
                np.full_like(served, "in_progress", dtype=object),
            ),
            dtype=object,
        ).copy()
        failure_reason[budget_exhausted & (remaining > 0)] = (
            "rollout_budget_exhausted"
        )
        failure_reason[budget_exhausted & (remaining == 0)] = (
            "rollout_budget_exhausted_not_returned"
        )
        out_info["rollout_horizon_steps"] = self.max_rollout_steps
        out_info["rollout_budget_exhausted"] = budget_exhausted.copy()
        out_info["failure_reason"] = failure_reason
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
    n_traj: int = 100,
    reward_mode: str = "distance",
    pbrs_config: PotentialRewardConfig | None = None,
    rollout_horizon_steps: int | None = None,
    training_mode: str = "legacy",
    **env_kwargs: Any,
):
    """Create the shared EVRPTW env with optional online sampling and PBRS."""
    if training_mode not in {"legacy", "stable_cost_v1"}:
        raise ValueError(f"Unknown TERRAN training_mode: {training_mode}")
    if training_mode == "stable_cost_v1":
        if pbrs_config is not None and any((
            pbrs_config.use_customer_pbrs, pbrs_config.use_repair_distance_pbrs,
            pbrs_config.use_feasible_ratio_pbrs, pbrs_config.use_terminal_heuristic,
            pbrs_config.use_terminal_task_penalty,
        )):
            raise ValueError("stable_cost_v1 does not permit PBRS or terminal reward penalties")
        reward_mode = "distance"  # Name is legacy; objective_config selects USD.
        env_kwargs.update(normalize_reward=False, invalid_action_penalty=0.0, success_bonus=0.0)
        env_kwargs.pop("reward_objective_scale", None)
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
    if training_mode == "stable_cost_v1":
        env = StableCostObservationWrapper(env, rollout_horizon_steps)
    if pbrs_config is not None and (
        pbrs_config.use_customer_pbrs
        or pbrs_config.use_repair_distance_pbrs
        or pbrs_config.use_feasible_ratio_pbrs
        or pbrs_config.use_terminal_heuristic
        or pbrs_config.use_terminal_task_penalty
    ):
        env = PotentialRewardWrapper(env, pbrs_config)
    return env


__all__ = [
    "OnlineInstanceResetWrapper",
    "TERRANRolloutHorizonWrapper",
    "StableCostObservationWrapper",
    "make_terran_env",
]
