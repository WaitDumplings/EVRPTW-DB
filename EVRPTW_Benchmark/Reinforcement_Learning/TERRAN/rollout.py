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
from .models.attention_model_wrapper import (
    DYNAMIC_OBSERVATION_KEYS,
    STATIC_OBSERVATION_KEYS,
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
    dynamic = {
        key: np.stack([obs[key] for obs in observations], axis=0)
        for key in DYNAMIC_OBSERVATION_KEYS if key in observations[0]
    }
    return {**static, **dynamic}, static


def tensor_from_array(value: Any, device: str | torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(np.asarray(value), device=device)


def _sync_cuda(device: str | torch.device) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def sample_actions(agent, obs_batch: dict[str, np.ndarray], decode_mode: str, device: str | torch.device):
    logits_tuple = agent.backbone(obs_batch)
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

    Training dropout draws must remain step-wise. Do not reuse an encoded graph
    in training mode, even when gradients are disabled.
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


def collect_rollout(agent, envs, rollout_steps: int, decode_mode: str, device: str | torch.device, seed: int | None = None, profile_timing: bool = False, compact_observations: bool = True) -> RolloutBatch:
    total_start = time.perf_counter()
    reset_start = time.perf_counter()
    observations, infos = reset_envs(envs, seed=seed)
    reset_time_s = time.perf_counter() - reset_start
    done = np.zeros((len(envs), envs[0].unwrapped.n_traj), dtype=bool)
    obs_steps: list[dict[str, np.ndarray]] = []
    actions_steps = []
    logprob_steps = []
    reward_steps = []
    done_steps = []
    value_steps = []
    valid_steps = []
    entropy_steps = []
    model_action_time_s = 0.0
    env_step_time_s = 0.0
    stack_obs_time_s = 0.0
    static_obs = None
    static_device = None

    for _ in range(int(rollout_steps)):
        valid = ~done
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
            # Deliberately keep the full per-step encoder call: training-mode
            # encoder dropout and its RNG consumption are unchanged.
            actions, logprob, entropy, value, _ = sample_actions(agent, model_obs, decode_mode=decode_mode, device=device)
        if profile_timing:
            _sync_cuda(device)
        model_action_time_s += time.perf_counter() - model_start
        action_np = actions.detach().cpu().numpy().astype(np.int64)
        env_start = time.perf_counter()
        next_observations, reward_np, step_done, infos = step_envs(envs, action_np)
        env_step_time_s += time.perf_counter() - env_start

        obs_steps.append(obs_batch)
        actions_steps.append(actions.detach())
        logprob_steps.append(logprob.detach())
        entropy_steps.append(entropy.detach())
        reward_steps.append(tensor_from_array(reward_np, device).float())
        done_steps.append(tensor_from_array(step_done, device).bool())
        value_steps.append(value.detach())
        valid_steps.append(tensor_from_array(valid, device).bool())

        observations = next_observations
        done = done | step_done
        if done.all():
            break

    total_time_s = time.perf_counter() - total_start
    explicit_budget_exhaustion = np.stack(
        [
            np.asarray(
                info.get(
                    "rollout_budget_exhausted",
                    np.zeros(done.shape[1], dtype=bool),
                ),
                dtype=bool,
            )
            for info in infos
        ],
        axis=0,
    )
    # Preserve the legacy diagnostic for callers that construct an environment
    # without the TERRAN horizon wrapper, while retaining the explicit flag for
    # trajectories that were truncated exactly at the registered budget.
    rollout_budget_exhausted = explicit_budget_exhaustion | (~done)
    return RolloutBatch(
        observations=obs_steps,
        actions=torch.stack(actions_steps, dim=0),
        old_logprobs=torch.stack(logprob_steps, dim=0),
        rewards=torch.stack(reward_steps, dim=0),
        dones=torch.stack(done_steps, dim=0),
        values=torch.stack(value_steps, dim=0),
        valid=torch.stack(valid_steps, dim=0),
        entropies=torch.stack(entropy_steps, dim=0),
        final_infos=infos,
        trajectory_steps=torch.stack(valid_steps, dim=0).sum(dim=0),
        rollout_budget_exhausted=tensor_from_array(
            rollout_budget_exhausted, device
        ).bool(),
        timings={
            "rollout_total_time_s": float(total_time_s),
            "rollout_reset_time_s": float(reset_time_s),
            "rollout_stack_obs_time_s": float(stack_obs_time_s),
            "rollout_model_action_time_s": float(model_action_time_s),
            "rollout_env_step_time_s": float(env_step_time_s),
            "rollout_interaction_time_s": float(model_action_time_s + env_step_time_s),
        },
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
    objective = np.asarray(info["objective_distance_km"], dtype=np.float64)
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
        "objective_distance_km": float(objective[selected]),
        "vehicle_count": int(np.asarray(info["vehicle_count"])[selected]),
        "served_customers": int(served[selected]),
    }
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
    this function never changes model mode or training dropout semantics.
    """
    if not envs:
        return []
    with _defer_route_info(envs, final_routes_only):
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
        if final_routes_only and (include_routes or return_final_info):
            infos = finalize_route_infos(envs, infos)
        # Include both the initial encoder/cache preparation above and the final
        # route export in reported inference time, not just the decoder loop.
        elapsed = time.perf_counter() - start
    per_instance_runtime = float(elapsed) / max(len(envs), 1)
    rows: list[dict[str, Any]] = []
    for info in infos:
        row = select_best_trajectory(info, include_routes=include_routes)
        row["runtime_s"] = per_instance_runtime
        row["batch_runtime_s"] = float(elapsed)
        if return_final_info:
            row["_final_info"] = info
        rows.append(row)
    return rows
