from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from evrptw_core.schema import merge_route_sequences

from ..common.route_info import finalize_route_infos
from ..common.objective import resolve_objective
from ..common.reward_contract import classify_rollout_failure_reasons
from ..common.training_diagnostics import summarize_values
from .models.attention_model_wrapper import (
    DYNAMIC_OBSERVATION_KEYS,
    STATIC_OBSERVATION_KEYS,
)


STABLE_STATE_KEYS = (
    "customer_unserved", "dispatch_paid", "remaining_step_budget", "episode_step_budget",
)


def stack_observations(observations: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = observations[0].keys()
    return {key: np.stack([obs[key] for obs in observations], axis=0) for key in keys}


def stack_policy_observations(
    observations: Sequence[dict[str, np.ndarray]],
    static: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compact model-only snapshot; immutable instance arrays are shared in time.

    A rollout resets once and never replaces an instance between steps. Dynamic
    arrays are still copied on every call so later env mutations cannot alter
    the stored PPO transitions. PBRS and the env continue to see full observations.
    """
    if static is None:
        static = {
            key: np.stack([obs[key] for obs in observations], axis=0)
            for key in STATIC_OBSERVATION_KEYS if key in observations[0]
        }
    dynamic_keys = DYNAMIC_OBSERVATION_KEYS + (
        STABLE_STATE_KEYS if "customer_unserved" in observations[0] else ()
    )
    dynamic = {
        key: np.stack([obs[key] for obs in observations], axis=0)
        for key in dynamic_keys if key in observations[0]
    }
    return {**static, **dynamic}, static


def tensor_from_array(value: Any, device: str | torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(np.asarray(value), device=device)


def _sync_cuda(device: str | torch.device) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def sample_actions(
    agent,
    obs_batch: dict[str, np.ndarray],
    decode_mode: str,
    device: str | torch.device,
    cached_state=None,
):
    logits_tuple = (
        agent.backbone(obs_batch)
        if cached_state is None
        else agent.backbone.decode(obs_batch, cached_state)
    )
    logits = logits_tuple[0]
    dist = torch.distributions.Categorical(logits=logits)
    if decode_mode == "greedy":
        actions = torch.argmax(logits, dim=-1)
    elif decode_mode == "sample":
        actions = dist.sample()
    else:
        raise ValueError(f"Unknown decode_mode={decode_mode!r}")
    logprob = dist.log_prob(actions)
    entropy = dist.entropy()
    value = agent.critic((logits_tuple[0], logits_tuple[1])).squeeze(-1)
    return actions, logprob, entropy, value, logits


def sample_eval_actions(agent, obs_batch, decode_mode: str, cached_state=None):
    """Action-only inference, with optional encoder cache for eval-mode models.

    Keep this evaluation helper restricted to eval mode so future mode-dependent
    encoder layers cannot silently invalidate its cache.
    """
    if cached_state is not None:
        if agent.training:
            raise ValueError("TERRAN encoder caching requires agent.eval()")
        logits, _ = agent.backbone.decode(obs_batch, cached_state)
    else:
        logits, _ = agent.backbone(obs_batch)
    if decode_mode == "greedy":
        return torch.argmax(logits, dim=-1)
    if decode_mode == "sample":
        return torch.distributions.Categorical(logits=logits).sample()
    raise ValueError(f"Unknown decode_mode={decode_mode!r}")


@dataclass
class RolloutBatch:
    observations: list[dict[str, np.ndarray]]
    actions: torch.Tensor
    old_logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    valid: torch.Tensor
    entropies: torch.Tensor
    final_infos: list[dict[str, Any]]
    timings: dict[str, float]
    trajectory_steps: torch.Tensor
    rollout_budget_exhausted: torch.Tensor
    reward_diagnostics: dict[str, float]
    # stable_cost_v1 only. Time-major labels share rewards.shape; episode
    # outcomes have shape (num_envs, n_traj). Legacy constructors stay valid.
    old_cost_values: torch.Tensor | None = None
    old_failure_values: torch.Tensor | None = None
    old_failure_logits: torch.Tensor | None = None
    cost_returns: torch.Tensor | None = None
    failure_returns: torch.Tensor | None = None
    unserved_returns: torch.Tensor | None = None
    terminal_failure: torch.Tensor | None = None
    unserved_fraction: torch.Tensor | None = None


def summarize_rollout_outcomes(
    infos: Sequence[dict[str, Any]],
    rollout_budget_exhausted: np.ndarray | torch.Tensor | None = None,
) -> dict[str, Any]:
    """Count mutually exclusive trajectory outcomes for FFP monitoring.

    ``non_horizon_infeasible`` has one deliberately narrow definition:
    ``(~success) & (~rollout_budget_exhausted)``. In particular, a rollout
    that merely reaches the caller's step budget is not reported as an FFP
    dead end. The optional mask is authoritative; the persisted info flag and
    terminal-reason names are backward-compatible fallbacks for callers that
    do not retain the rollout tensor.
    """

    explicit = None
    if rollout_budget_exhausted is not None:
        if isinstance(rollout_budget_exhausted, torch.Tensor):
            explicit = rollout_budget_exhausted.detach().cpu().numpy()
        else:
            explicit = np.asarray(rollout_budget_exhausted)
        explicit = np.asarray(explicit, dtype=bool)
        if explicit.ndim == 1 and len(infos) == 1:
            explicit = explicit.reshape(1, -1)
        if explicit.ndim != 2 or explicit.shape[0] != len(infos):
            raise ValueError(
                "rollout_budget_exhausted must have shape (num_envs, n_traj)"
            )

    trajectory_count = 0
    success_count = 0
    budget_count = 0
    non_horizon_count = 0
    reason_counts: dict[str, int] = {}
    for row_index, info in enumerate(infos):
        success = np.asarray(info.get("success", []), dtype=bool).reshape(-1)
        count = int(success.size)
        if not count:
            continue
        reasons = np.asarray(
            info.get("failure_reason", np.full(count, "terminal_failure")),
            dtype=object,
        ).reshape(-1)
        if reasons.size != count:
            raise ValueError("failure_reason and success trajectory counts disagree")

        if explicit is not None:
            budget = np.asarray(explicit[row_index], dtype=bool).reshape(-1)
            if budget.size != count:
                raise ValueError(
                    "rollout_budget_exhausted and success trajectory counts disagree"
                )
        elif "rollout_budget_exhausted" in info:
            budget = np.asarray(
                info["rollout_budget_exhausted"], dtype=bool
            ).reshape(-1)
            if budget.size != count:
                raise ValueError(
                    "rollout_budget_exhausted and success trajectory counts disagree"
                )
        else:
            # Legacy final infos did not persist the boolean horizon mask.
            budget = np.asarray(
                [
                    str(value) == "environment_step_limit"
                    or str(value).startswith("rollout_budget_exhausted")
                    for value in reasons
                ],
                dtype=bool,
            )

        # A successfully completed trajectory is never a failed horizon case,
        # even if a malformed legacy info record happens to set both flags.
        budget_failure = (~success) & budget
        non_horizon = (~success) & (~budget)
        for value in reasons[non_horizon]:
            reason = str(value)
            if reason in {"", "None", "in_progress", "success"}:
                reason = "terminal_failure"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        trajectory_count += count
        success_count += int(success.sum())
        budget_count += int(budget_failure.sum())
        non_horizon_count += int(non_horizon.sum())

    denominator = max(trajectory_count, 1)
    if trajectory_count != success_count + budget_count + non_horizon_count:
        raise RuntimeError("trajectory outcome monitoring is not exhaustive")
    return {
        "trajectory_count": trajectory_count,
        "success_count": success_count,
        "success_rate": success_count / denominator,
        "rollout_budget_exhausted_count": budget_count,
        "rollout_budget_exhausted_rate": budget_count / denominator,
        "non_horizon_infeasible_count": non_horizon_count,
        "non_horizon_infeasible_rate": non_horizon_count / denominator,
        "non_horizon_infeasible_reason_counts": dict(sorted(reason_counts.items())),
    }


def _reported_rollout_budget_exhaustion(
    infos: Sequence[dict[str, Any]], n_traj: int
) -> np.ndarray:
    """Read every environment-side indication that a step budget was hit.

    The TERRAN horizon wrapper emits an explicit boolean array. Bare shared
    environments instead terminate with ``environment_step_limit``; treating
    that terminal reason as an FFP failure would make the diagnostic depend on
    whether the wrapper happened to be installed.
    """

    rows: list[np.ndarray] = []
    for info in infos:
        explicit = np.asarray(
            info.get(
                "rollout_budget_exhausted",
                np.zeros(n_traj, dtype=bool),
            ),
            dtype=bool,
        ).reshape(-1)
        if explicit.size != n_traj:
            raise ValueError(
                "rollout_budget_exhausted and environment trajectory counts disagree"
            )
        reasons = np.asarray(
            info.get("failure_reason", np.full(n_traj, "in_progress")),
            dtype=object,
        ).reshape(-1)
        if reasons.size != n_traj:
            raise ValueError(
                "failure_reason and environment trajectory counts disagree"
            )
        reason_horizon = np.asarray(
            [
                str(reason) == "environment_step_limit"
                or str(reason).startswith("rollout_budget_exhausted")
                for reason in reasons
            ],
            dtype=bool,
        )
        rows.append(explicit | reason_horizon)
    return np.stack(rows, axis=0)


def reset_envs(envs, seed: int | None = None):
    observations = []
    infos = []
    for idx, env in enumerate(envs):
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = int(seed) + idx
        obs, info = env.reset(**kwargs)
        observations.append(obs)
        infos.append(info)
    return observations, infos


def step_envs(envs, actions: np.ndarray):
    observations, rewards, dones, infos = [], [], [], []
    for env, action in zip(envs, actions):
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
        rewards.append(reward)
        dones.append(np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool))
        infos.append(info)
    return observations, np.asarray(rewards, dtype=np.float32), np.asarray(dones, dtype=bool), infos


@contextmanager
def _defer_route_info(envs, enabled: bool):
    """Defer fast-env route export without changing the caller's info setting."""
    changed = []
    try:
        if enabled:
            for env in envs:
                base = env.unwrapped
                if getattr(base, "info_level", None) == "full":
                    changed.append(base)
                    base.info_level = "light"
        yield
    finally:
        for base in changed:
            base.info_level = "full"


@contextmanager
def _scoped_eval_sampling_seed(
    seed: int | None,
    device: str | torch.device,
    *,
    enabled: bool,
):
    """Seed evaluation sampling without consuming the caller's training RNG."""

    if not enabled or seed is None:
        yield
        return
    target = torch.device(device)
    cuda_devices: list[int] = []
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
        cuda_devices = [
            torch.cuda.current_device() if target.index is None else int(target.index)
        ]
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.random.default_generator.manual_seed(int(seed))
        if cuda_devices:
            torch.cuda.default_generators[cuda_devices[0]].manual_seed(int(seed))
        yield


class BoundedBaseRewardStats:
    """CPU-only logging accumulator; never used to construct rewards or losses.

    Moments cover every finite active transition. A fixed integer hash of the
    global active-transition index chooses at most K values for approximate
    quantiles, independent of chunking and without consuming a random stream.
    """

    def __init__(self, max_quantile_samples: int = 8192) -> None:
        self.limit = int(max_quantile_samples)
        if self.limit < 1:
            raise ValueError("max_quantile_samples must be positive")
        self.count = self.finite_count = 0
        self.mean = self.m2 = 0.0
        self.minimum, self.maximum = np.inf, -np.inf
        self.samples = np.empty(0, dtype=np.float64)
        self.priorities = np.empty(0, dtype=np.uint64)
        self.threshold = np.iinfo(np.uint64).max

    def update(self, values: np.ndarray, mask: np.ndarray) -> None:
        active = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)].reshape(-1)
        start = self.count
        self.count += int(active.size)
        finite_mask = np.isfinite(active)
        finite = active[finite_mask]
        if not finite.size:
            return
        chunk_count = int(finite.size)
        chunk_mean = float(finite.mean())
        delta = chunk_mean - self.mean
        total_count = self.finite_count + chunk_count
        self.m2 += float(np.square(finite - chunk_mean).sum()) + delta * delta * self.finite_count * chunk_count / total_count
        self.mean += delta * chunk_count / total_count
        self.finite_count = total_count
        self.minimum = min(self.minimum, float(finite.min()))
        self.maximum = max(self.maximum, float(finite.max()))

        # SplitMix64 is a deterministic permutation, not a training RNG call.
        priority = np.arange(start, self.count, dtype=np.uint64)[finite_mask]
        priority = priority + np.uint64(0x9E3779B97F4A7C15)
        priority = (priority ^ (priority >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        priority = (priority ^ (priority >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        priority = priority ^ (priority >> np.uint64(31))
        if self.samples.size >= self.limit:
            keep = priority < self.threshold
            finite, priority = finite[keep], priority[keep]
        if not finite.size:
            return
        samples = np.concatenate((self.samples, finite))
        priorities = np.concatenate((self.priorities, priority))
        if samples.size > self.limit:
            keep = np.argpartition(priorities, self.limit - 1)[:self.limit]
            samples, priorities = samples[keep], priorities[keep]
        self.samples, self.priorities = samples, priorities
        self.threshold = self.priorities.max()

    def summary(self) -> dict[str, Any]:
        result = summarize_values(self.samples, max_quantile_samples=self.limit)
        result.update(
            count=self.count, finite_count=self.finite_count,
            nonfinite_count=self.count - self.finite_count,
            mean=self.mean if self.finite_count and np.isfinite(self.mean) else None,
            std=float(np.sqrt(max(self.m2 / self.finite_count, 0.0))) if self.finite_count and np.isfinite(self.m2) else None,
            min=self.minimum if self.finite_count else None,
            max=self.maximum if self.finite_count else None,
            quantile_method="linear_on_deterministic_splitmix64_bottom_k_finite_active_observations",
            quantiles_approximate=self.finite_count > self.limit,
        )
        return result


def collect_rollout(
    agent,
    envs,
    rollout_steps: int,
    decode_mode: str,
    device: str | torch.device,
    seed: int | None = None,
    profile_timing: bool = False,
    compact_observations: bool = True,
    cache_static_embeddings: bool = True,
    reward_discount_factor: float = 1.0,
    base_reward_stats: BoundedBaseRewardStats | None = None,
    storage_device: str | torch.device | None = None,
    collect_reward_diagnostics: bool = True,
) -> RolloutBatch:
    """Collect transitions; optional reward diagnostics never affect training.

    Disabling diagnostics skips the legacy per-component CPU aggregation and
    returns an empty ``reward_diagnostics`` mapping. An explicitly supplied
    ``base_reward_stats`` accumulator is still updated.
    """
    if int(rollout_steps) <= 0:
        raise ValueError("rollout_steps must be positive")
    stable_mode = getattr(agent, "critic_mode", "legacy") == "stable_cost_v1"
    stable_envs = [getattr(env.unwrapped, "training_mode", "legacy") == "stable_cost_v1" for env in envs]
    if any(stable_envs) != stable_mode or (stable_mode and not all(stable_envs)):
        raise ValueError("stable_cost_v1 actor and rollout environments must agree")
    if stable_mode and float(reward_discount_factor) != 1.0:
        raise ValueError("stable_cost_v1 requires gamma=1 for complete Monte Carlo returns")
    storage_device = device if storage_device is None else storage_device
    total_start = time.perf_counter()
    reset_start = time.perf_counter()
    observations, infos = reset_envs(envs, seed=seed)
    reset_time_s = time.perf_counter() - reset_start
    done = np.zeros((len(envs), envs[0].unwrapped.n_traj), dtype=bool)
    if stable_mode:
        # Reset can already reveal an impossible instance. Its depot sentinel
        # is padding, not a sampled active action; its failure still counts.
        done = np.stack([env.unwrapped.terminated | env.unwrapped.truncated for env in envs])
    obs_steps: list[dict[str, np.ndarray]] = []
    actions_steps = []
    logprob_steps = []
    reward_steps = []
    done_steps = []
    value_steps = []
    failure_value_steps = []
    failure_logit_steps = []
    valid_steps = []
    entropy_steps = []
    model_action_time_s = 0.0
    env_step_time_s = 0.0
    stack_obs_time_s = 0.0
    static_obs = None
    static_device = None
    cached_state = None
    reward_component_keys = (
        "base",
        "distance",
        "objective",
        "electricity_cost",
        "vehicle_cost",
        "base_non_objective",
        "base_non_distance",
        "pbrs_customer",
        "pbrs_repair_distance",
        "pbrs_feasible_ratio",
        "terminal_heuristic",
        "terminal_task_total",
        "terminal_success_bonus",
        "terminal_failure_base",
        "terminal_unserved",
        "shaped",
        "pbrs_total",
        "shaping_total",
    )
    reward_diagnostics = {}
    if collect_reward_diagnostics:
        reward_diagnostics = {
            "active_count": 0.0,
            "customer_action_count": 0.0,
            "noncustomer_action_count": 0.0,
        }
        for key in reward_component_keys:
            reward_diagnostics[f"{key}_sum"] = 0.0
            reward_diagnostics[f"{key}_discounted_sum"] = 0.0
            reward_diagnostics[f"{key}_abs_sum"] = 0.0
            reward_diagnostics[f"{key}_customer_action_sum"] = 0.0
            reward_diagnostics[f"{key}_noncustomer_action_sum"] = 0.0

    for step_index in range(int(rollout_steps)):
        valid = ~done
        if stable_mode:
            # The collector's limit may be stricter than the env safety limit.
            # Store that actual budget in the replay observation as well.
            observations = [
                {**obs,
                 "remaining_step_budget": np.minimum(obs["remaining_step_budget"], int(rollout_steps) - step_index).astype(np.float32),
                 "episode_step_budget": np.minimum(obs["episode_step_budget"], int(rollout_steps)).astype(np.float32)}
                for obs in observations
            ]
        stack_start = time.perf_counter()
        if compact_observations:
            obs_batch, static_obs = stack_policy_observations(observations, static_obs)
        else:
            obs_batch = stack_observations(observations)
        stack_obs_time_s += time.perf_counter() - stack_start
        if profile_timing:
            _sync_cuda(device)
        model_start = time.perf_counter()
        with torch.no_grad():
            if compact_observations:
                if static_device is None:
                    static_device = {key: tensor_from_array(value, device) for key, value in static_obs.items()}
                model_obs = {**obs_batch, **static_device}
            else:
                model_obs = obs_batch
            if cache_static_embeddings and cached_state is None:
                # Static instance features and model parameters do not change
                # during no-grad rollout collection. PPO recomputes the encoder
                # with gradients during its update.
                cached_state = agent.backbone.encode(model_obs)
            if stable_mode:
                actions, logprob, entropy, value, _, critic_outputs = agent.get_action_and_value_cached(
                    model_obs, state=cached_state, decode_mode=decode_mode,
                    return_critic_outputs=True,
                )
                value = critic_outputs["cost_value"]
                failure_logits = critic_outputs["failure_logits"]
                failure_value_steps.append(failure_logits.sigmoid().detach().to(storage_device))
                failure_logit_steps.append(failure_logits.detach().to(storage_device))
            else:
                actions, logprob, entropy, value, _ = sample_actions(
                    agent, model_obs, decode_mode=decode_mode, device=device,
                    cached_state=cached_state,
                )
        if profile_timing:
            _sync_cuda(device)
        model_action_time_s += time.perf_counter() - model_start
        action_np = actions.detach().cpu().numpy().astype(np.int64)
        env_start = time.perf_counter()
        previous_infos = infos
        next_observations, reward_np, step_done, infos = step_envs(envs, action_np)
        env_step_time_s += time.perf_counter() - env_start

        if collect_reward_diagnostics:
            # Aggregate reward components while the pre-step active mask is still
            # available.  Keeping only float64 sums adds negligible memory and makes
            # scale failures observable in every formal epoch.
            for env_index, (env, info) in enumerate(zip(envs, infos)):
                active = np.asarray(valid[env_index], dtype=bool)
                active_count = int(active.sum())
                if active_count == 0:
                    continue
                components = info.get("reward_components")
                if components is None:
                    base = np.asarray(reward_np[env_index], dtype=np.float64)
                    arrays = {
                        "base": base,
                        "distance": base,
                        "objective": base,
                        "electricity_cost": np.zeros_like(base),
                        "vehicle_cost": np.zeros_like(base),
                        "base_non_objective": np.zeros_like(base),
                        "base_non_distance": np.zeros_like(base),
                        "pbrs_customer": np.zeros_like(base),
                        "pbrs_repair_distance": np.zeros_like(base),
                        "pbrs_feasible_ratio": np.zeros_like(base),
                        "terminal_heuristic": np.zeros_like(base),
                        "terminal_task_total": np.zeros_like(base),
                        "terminal_success_bonus": np.zeros_like(base),
                        "terminal_failure_base": np.zeros_like(base),
                        "terminal_unserved": np.zeros_like(base),
                        "shaped": base,
                    }
                    previous = previous_infos[env_index]
                    if "objective_value" in info and "objective_value" in previous:
                        normalized = bool(getattr(env.unwrapped, "normalize_reward", False))
                        scale = float(env.unwrapped.reward_objective_scale) if normalized else 1.0
                        distance_scale = float(env.unwrapped.reward_distance_scale_km) if normalized else 1.0
                        arrays["objective"] = -(np.asarray(info["objective_value"]) - np.asarray(previous["objective_value"])) / scale
                        arrays["distance"] = -(np.asarray(info["objective_distance_km"]) - np.asarray(previous["objective_distance_km"])) / distance_scale
                        arrays["base_non_objective"] = base - arrays["objective"]
                        arrays["base_non_distance"] = arrays["base_non_objective"]
                        for key in ("electricity_cost", "vehicle_cost"):
                            if info.get(f"{key}_usd") is not None:
                                arrays[key] = -(np.asarray(info[f"{key}_usd"]) - np.asarray(previous[f"{key}_usd"])) / scale
                else:
                    arrays = {
                        key: np.asarray(components.get(key, np.zeros_like(reward_np[env_index])), dtype=np.float64)
                        for key in reward_component_keys
                        if key not in {"pbrs_total", "shaping_total"}
                    }
                arrays["pbrs_total"] = (
                    arrays["pbrs_customer"]
                    + arrays["pbrs_repair_distance"]
                    + arrays["pbrs_feasible_ratio"]
                )
                # ``terminal_task_total`` is part of the canonical task, not
                # auxiliary shaping.  Keep this long-standing diagnostic field but
                # narrow its semantics to PBRS plus the legacy heuristic only.
                arrays["shaping_total"] = (
                    arrays["pbrs_total"] + arrays["terminal_heuristic"]
                )
                if base_reward_stats is not None:
                    base_reward_stats.update(arrays["base"], active)
                num_customers = int(getattr(env.unwrapped, "num_customers", 0))
                customer_action = (
                    active
                    & (action_np[env_index] >= 1)
                    & (action_np[env_index] <= num_customers)
                )
                noncustomer_action = active & ~customer_action
                reward_diagnostics["active_count"] += active_count
                reward_diagnostics["customer_action_count"] += int(
                    customer_action.sum()
                )
                reward_diagnostics["noncustomer_action_count"] += int(
                    noncustomer_action.sum()
                )
                for key, array in arrays.items():
                    active_values = array[active]
                    reward_diagnostics[f"{key}_sum"] += float(active_values.sum())
                    reward_diagnostics[f"{key}_discounted_sum"] += float(
                        (float(reward_discount_factor) ** step_index)
                        * active_values.sum()
                    )
                    reward_diagnostics[f"{key}_abs_sum"] += float(
                        np.abs(active_values).sum()
                    )
                    reward_diagnostics[f"{key}_customer_action_sum"] += float(
                        array[customer_action].sum()
                    )
                    reward_diagnostics[f"{key}_noncustomer_action_sum"] += float(
                        array[noncustomer_action].sum()
                    )

        elif base_reward_stats is not None:
            # A caller may disable diagnostics while still using the separate
            # bounded base-reward accumulator. Preserve that contract cheaply.
            for env_index, info in enumerate(infos):
                active = valid[env_index]
                if not active.any():
                    continue
                components = info.get("reward_components")
                base = (reward_np[env_index] if components is None else
                        components.get("base", np.zeros_like(reward_np[env_index])))
                base_reward_stats.update(np.asarray(base, dtype=np.float64), active)

        obs_steps.append(obs_batch)
        actions_steps.append(actions.detach().to(storage_device))
        logprob_steps.append(logprob.detach().to(storage_device))
        entropy_steps.append(entropy.detach().to(storage_device))
        reward_steps.append(tensor_from_array(reward_np, storage_device).float())
        done_steps.append(tensor_from_array(step_done, storage_device).bool())
        value_steps.append(value.detach().to(storage_device))
        valid_steps.append(tensor_from_array(valid, storage_device).bool())

        observations = next_observations
        done = done | step_done
        if done.all():
            break

    total_time_s = time.perf_counter() - total_start
    explicit_budget_exhaustion = _reported_rollout_budget_exhaustion(
        infos, done.shape[1]
    )
    # Preserve the legacy diagnostic for callers that construct an environment
    # without the TERRAN horizon wrapper, while retaining the explicit flag for
    # trajectories that were truncated exactly at the registered budget.
    rollout_budget_exhausted = explicit_budget_exhaustion | (~done)
    success = np.stack(
        [np.asarray(info.get("success"), dtype=bool) for info in infos], axis=0
    )
    served_customers = np.stack(
        [
            np.asarray(info.get("served_customers"), dtype=np.int32)
            for info in infos
        ],
        axis=0,
    )
    customer_count = np.asarray(
        [[int(getattr(env.unwrapped, "num_customers", 0))] for env in envs],
        dtype=np.int32,
    )
    classify_rollout_failure_reasons(
        infos,
        done=done,
        success=success,
        served_customers=served_customers,
        customer_count=customer_count,
    )
    for row_index, info in enumerate(infos):
        # Persist the authoritative mask alongside the reason so downstream
        # logging does not have to infer horizon status from a string label.
        info["rollout_budget_exhausted"] = rollout_budget_exhausted[
            row_index
        ].copy()
        if stable_mode:
            info["failure_terminal"] = np.asarray(info.get("failure_terminal", np.zeros(done.shape[1], dtype=bool))) | ~done[row_index]
    rewards_tensor = torch.stack(reward_steps, dim=0)
    dones_tensor = torch.stack(done_steps, dim=0)
    valid_tensor = torch.stack(valid_steps, dim=0)
    values_tensor = torch.stack(value_steps, dim=0)
    stable_fields = {}
    if stable_mode:
        # An actual collection limit is a task failure in this contract. There
        # is no bootstrap into an uncollected continuation.
        dones_tensor[-1] |= tensor_from_array(~done, storage_device).bool()
        failures = tensor_from_array(~success, storage_device).bool()
        unserved = tensor_from_array(
            np.clip(1.0 - served_customers / np.maximum(customer_count, 1), 0.0, 1.0),
            storage_device,
        ).float()
        stable_fields = dict(
            old_cost_values=values_tensor,
            old_failure_values=torch.stack(failure_value_steps, dim=0),
            old_failure_logits=torch.stack(failure_logit_steps, dim=0),
            cost_returns=compute_returns(-rewards_tensor, dones_tensor, 1.0) * valid_tensor,
            failure_returns=failures.float().unsqueeze(0).expand_as(rewards_tensor) * valid_tensor,
            unserved_returns=unserved.unsqueeze(0).expand_as(rewards_tensor) * valid_tensor,
            terminal_failure=failures,
            unserved_fraction=unserved,
        )
    return RolloutBatch(
        observations=obs_steps,
        actions=torch.stack(actions_steps, dim=0),
        old_logprobs=torch.stack(logprob_steps, dim=0),
        rewards=rewards_tensor,
        dones=dones_tensor,
        values=values_tensor,
        valid=valid_tensor,
        entropies=torch.stack(entropy_steps, dim=0),
        final_infos=infos,
        trajectory_steps=valid_tensor.sum(dim=0),
        rollout_budget_exhausted=tensor_from_array(
            rollout_budget_exhausted, storage_device
        ).bool(),
        reward_diagnostics=reward_diagnostics,
        timings={
            "rollout_total_time_s": float(total_time_s),
            "rollout_reset_time_s": float(reset_time_s),
            "rollout_stack_obs_time_s": float(stack_obs_time_s),
            "rollout_model_action_time_s": float(model_action_time_s),
            "rollout_env_step_time_s": float(env_step_time_s),
            "rollout_interaction_time_s": float(model_action_time_s + env_step_time_s),
        },
        **stable_fields,
    )


def compute_returns(rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.size(0))):
        running = rewards[step] + float(gamma) * running * (~dones[step]).float()
        returns[step] = running
    return returns


def select_best_trajectory(info: dict[str, Any], include_routes: bool = True) -> dict[str, Any]:
    success = np.asarray(info["success"], dtype=bool)
    distance = np.asarray(info["objective_distance_km"], dtype=np.float64)
    objective = np.asarray(info.get("objective_value", distance), dtype=np.float64)
    served = np.asarray(info["served_customers"], dtype=np.int32)
    if np.any(success):
        candidates = np.where(success)[0]
        selected = int(candidates[np.argmin(objective[candidates])])
        feasible = True
    else:
        max_served = int(served.max()) if served.size else 0
        candidates = np.where(served == max_served)[0]
        selected = int(candidates[np.argmin(objective[candidates])]) if candidates.size else 0
        feasible = False
    row = {
        "selected_traj_idx": selected,
        "feasible": feasible,
        "objective_distance_km": float(distance[selected]),
        "objective_value": float(objective[selected]),
        "vehicle_count": int(np.asarray(info["vehicle_count"])[selected]),
        "served_customers": int(served[selected]),
    }
    for key in ("objective_cost_usd", "electricity_cost_usd", "vehicle_cost_usd", "vehicles_started"):
        values = info.get(key)
        row[key] = float(np.asarray(values)[selected]) if values is not None else None
    objective_config = resolve_objective(info.get("objective_config"))
    row["objective_mode"] = objective_config.mode
    row["objective_unit"] = objective_config.unit
    row["objective_profile_id"] = objective_config.profile_id
    if include_routes and "routes" in info:
        routes = info["routes"][selected]
        route_sequence = merge_route_sequences(routes)
        row["route_sequence_json"] = json.dumps(route_sequence)
        row["routes_json"] = json.dumps(routes)
    return row


def rollout_single_instance(
    agent,
    env,
    decode_mode: str,
    max_steps: int,
    device: str | torch.device,
    seed: int | None = None,
    include_routes: bool = True,
    cache_static_embeddings: bool = True,
    compact_observations: bool = True,
    final_routes_only: bool = True,
):
    row = rollout_eval_batch(
        agent, [env], decode_mode, max_steps, device, seed=seed,
        include_routes=include_routes,
        cache_static_embeddings=cache_static_embeddings,
        compact_observations=compact_observations,
        final_routes_only=final_routes_only,
    )[0]
    row.pop("batch_runtime_s", None)
    return row


def rollout_eval_batch(
    agent,
    envs,
    decode_mode: str,
    max_steps: int,
    device: str | torch.device,
    seed: int | None = None,
    include_routes: bool = False,
    return_final_info: bool = False,
    cache_static_embeddings: bool = True,
    compact_observations: bool = True,
    final_routes_only: bool = True,
):
    """Evaluate fixed instances, retaining opt-out switches for A/B validation.

    Encoder caching is enabled only if the caller already selected eval mode;
    this function never changes model mode.
    """
    if not envs:
        return []
    with _scoped_eval_sampling_seed(
        seed,
        device,
        enabled=decode_mode == "sample",
    ), _defer_route_info(envs, final_routes_only):
        observations, infos = reset_envs(envs, seed=seed)
        n_traj = int(envs[0].unwrapped.n_traj)
        done = np.zeros((len(envs), n_traj), dtype=bool)
        static_obs = None
        static_device = None
        cached_state = None
        start = time.perf_counter()
        for _ in range(int(max_steps)):
            if compact_observations:
                obs_batch, static_obs = stack_policy_observations(observations, static_obs)
            else:
                obs_batch = stack_observations(observations)
            with torch.no_grad():
                if compact_observations:
                    if static_device is None:
                        static_device = {key: tensor_from_array(value, device) for key, value in static_obs.items()}
                    model_obs = {**obs_batch, **static_device}
                else:
                    model_obs = obs_batch
                if cache_static_embeddings and not agent.training and cached_state is None:
                    cached_state = agent.backbone.encode(model_obs)
                actions = sample_eval_actions(agent, model_obs, decode_mode, cached_state)
            action_np = actions.detach().cpu().numpy().astype(np.int64)
            observations, _, step_done, infos = step_envs(envs, action_np)
            done = done | step_done
            if done.all():
                break
        explicit_budget_exhaustion = _reported_rollout_budget_exhaustion(
            infos, n_traj
        )
        # Evaluation environments intentionally have no training horizon
        # wrapper. Any trajectory still active when this loop exhausts its
        # registered max_steps is nevertheless a horizon case, not an FFP
        # failure.
        rollout_budget_exhausted = explicit_budget_exhaustion | (~done)
        success = np.stack(
            [np.asarray(info.get("success"), dtype=bool) for info in infos],
            axis=0,
        )
        served_customers = np.stack(
            [
                np.asarray(info.get("served_customers"), dtype=np.int32)
                for info in infos
            ],
            axis=0,
        )
        customer_count = np.asarray(
            [[int(getattr(env.unwrapped, "num_customers", 0))] for env in envs],
            dtype=np.int32,
        )
        classify_rollout_failure_reasons(
            infos,
            done=done,
            success=success,
            served_customers=served_customers,
            customer_count=customer_count,
        )
        for row_index, info in enumerate(infos):
            info["rollout_budget_exhausted"] = rollout_budget_exhausted[
                row_index
            ].copy()
        if final_routes_only and (include_routes or return_final_info):
            infos = finalize_route_infos(envs, infos)
        # Include both the initial encoder/cache preparation above and the final
        # route export in reported inference time, not just the decoder loop.
        elapsed = time.perf_counter() - start
    per_instance_runtime = float(elapsed) / max(len(envs), 1)
    rows: list[dict[str, Any]] = []
    for row_index, info in enumerate(infos):
        row = select_best_trajectory(info, include_routes=include_routes)
        outcome = summarize_rollout_outcomes(
            [info], rollout_budget_exhausted[row_index : row_index + 1]
        )
        row.update(
            candidate_trajectory_count=outcome["trajectory_count"],
            candidate_success_count=outcome["success_count"],
            candidate_success_rate=outcome["success_rate"],
            candidate_rollout_budget_exhausted_count=outcome[
                "rollout_budget_exhausted_count"
            ],
            candidate_rollout_budget_exhausted_rate=outcome[
                "rollout_budget_exhausted_rate"
            ],
            candidate_non_horizon_infeasible_count=outcome[
                "non_horizon_infeasible_count"
            ],
            candidate_non_horizon_infeasible_rate=outcome[
                "non_horizon_infeasible_rate"
            ],
            candidate_non_horizon_infeasible_reason_counts=json.dumps(
                outcome["non_horizon_infeasible_reason_counts"], sort_keys=True
            ),
            no_success_all_candidates_non_horizon_infeasible=bool(
                outcome["success_count"] == 0
                and outcome["non_horizon_infeasible_count"]
                == outcome["trajectory_count"]
            ),
        )
        row["runtime_s"] = per_instance_runtime
        row["batch_runtime_s"] = float(elapsed)
        if return_final_info:
            row["_final_info"] = info
        rows.append(row)
    return rows
