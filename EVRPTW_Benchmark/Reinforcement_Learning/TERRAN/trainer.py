from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))
sys.path.insert(0, str(REPO_ROOT))

from evrptw_core.io import iter_instances

from .async_instances import AsyncInstancePool
from ..common import Stage2TaskPool
from ..common.data_pass import DataPassState
from ..common.evaluation import select_min_verified_distance
from ..common.training_protocol import append_jsonl, atomic_json
from .data_pool import FixedDatasetInstancePool, OnlineInstancePool, Stage2TERRANPool
from .env_factory import make_terran_env
from .models import Agent
from .models.attention_model_wrapper import (
    DYNAMIC_OBSERVATION_KEYS,
    STATIC_OBSERVATION_KEYS,
)
from .pbrs import PotentialRewardConfig
from .rollout import collect_rollout, compute_returns, rollout_eval_batch


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        local = Path(__file__).resolve().parent / "configs" / cfg_path
        cfg_path = local if local.exists() else cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    denom = torch.clamp(mask_f.sum(), min=1.0)
    return (value * mask_f).sum() / denom


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync_cuda(device: str | torch.device) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _resolve_repo_path(path: str | Path | None) -> Path | None:
    if path is None or str(path) == "":
        return None
    out = Path(path)
    return out if out.is_absolute() else REPO_ROOT / out


def _eval_instance_batches(
    eval_path: Path,
    num_customers: int,
    num_charging_stations: int,
    batch_size: int,
    limit: int | None = None,
    num_batches_limit: int | None = None,
):
    max_count = None if limit is None else int(limit)
    if num_batches_limit is not None:
        by_batches = max(1, int(batch_size)) * int(num_batches_limit)
        max_count = by_batches if max_count is None else min(max_count, by_batches)
    batch = []
    seen = 0
    for instance in iter_instances(eval_path, num_customers=num_customers, num_charging_stations=num_charging_stations):
        if max_count is not None and seen >= max_count:
            break
        batch.append(instance)
        seen += 1
        if len(batch) >= max(1, int(batch_size)):
            yield batch
            batch = []
    if batch:
        yield batch


def _pbrs_enabled(cfg: dict[str, Any]) -> bool:
    pbrs = cfg.get("pbrs", {}) or {}
    return bool(
        pbrs.get("use_customer_pbrs", False)
        or pbrs.get("use_repair_distance_pbrs", False)
        or pbrs.get("use_feasible_ratio_pbrs", False)
        or pbrs.get("use_terminal_heuristic", False)
    )


def pbrs_scale_for_epoch(cfg: dict[str, Any], epoch: int, total_epochs: int) -> float:
    if not _pbrs_enabled(cfg):
        return 0.0
    pbrs = cfg.get("pbrs", {}) or {}
    annealing = pbrs.get("annealing", {}) or {}
    if not bool(annealing.get("enabled", False)):
        return float(annealing.get("start_scale", 1.0))
    start_scale = float(annealing.get("start_scale", 1.0))
    end_scale = float(annealing.get("end_scale", 0.2))
    start_epoch = max(1, int(annealing.get("start_epoch", 1)))
    end_epoch = max(start_epoch, int(annealing.get("end_epoch", total_epochs)))
    schedule = str(annealing.get("schedule", "cosine")).lower()
    if epoch <= start_epoch:
        return max(start_scale, 0.0)
    if epoch >= end_epoch:
        return max(end_scale, 0.0)
    progress = (float(epoch) - float(start_epoch)) / max(float(end_epoch - start_epoch), 1.0)
    progress = min(max(progress, 0.0), 1.0)
    if schedule == "linear":
        weight = progress
    elif schedule == "exponential":
        if start_scale <= 0 or end_scale <= 0:
            weight = progress
            return max(start_scale + (end_scale - start_scale) * weight, 0.0)
        return max(start_scale * ((end_scale / start_scale) ** progress), 0.0)
    elif schedule == "constant":
        return max(start_scale, 0.0)
    else:
        weight = 0.5 - 0.5 * math.cos(math.pi * progress)
    return max(start_scale + (end_scale - start_scale) * weight, 0.0)


def set_pbrs_reward_scale(envs: Sequence[Any], scale: float) -> None:
    for env in envs:
        current = env
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            setter = getattr(current, "set_reward_scale", None)
            if callable(setter):
                setter(scale)
                break
            current = getattr(current, "env", None)


def build_pbrs_config(cfg: dict[str, Any]) -> PotentialRewardConfig | None:
    pbrs = cfg.get("pbrs", {})
    config = PotentialRewardConfig(
        use_customer_pbrs=bool(pbrs.get("use_customer_pbrs", False)),
        use_repair_distance_pbrs=bool(pbrs.get("use_repair_distance_pbrs", False)),
        use_feasible_ratio_pbrs=bool(pbrs.get("use_feasible_ratio_pbrs", False)),
        use_terminal_heuristic=bool(pbrs.get("use_terminal_heuristic", False)),
        customer_pbrs_mode=str(pbrs.get("customer_pbrs_mode", "progress")),
        gamma=float(cfg.get("training", {}).get("gamma", 0.99)),
        alpha=float(pbrs.get("alpha", 2.0)),
        beta=float(pbrs.get("beta", 0.5)),
        customer_pbrs_coef=float(pbrs.get("customer_pbrs_coef", 1.0)),
        customer_progress_budget=float(pbrs.get("customer_progress_budget", 0.5)),
        customer_progress_mix=float(pbrs.get("customer_progress_mix", 0.5)),
        repair_progress_coef=float(pbrs.get("repair_progress_coef", 0.5)),
        feasible_ratio_coef=float(pbrs.get("feasible_ratio_coef", 0.0)),
        pbrs_clip=pbrs.get("pbrs_clip", None),
        success_bonus=float(pbrs.get("success_bonus", 0.1)),
        failure_penalty=float(pbrs.get("failure_penalty", 0.5)),
    )
    if not (
        config.use_customer_pbrs
        or config.use_repair_distance_pbrs
        or config.use_feasible_ratio_pbrs
        or config.use_terminal_heuristic
    ):
        return None
    return config


def _configure_dataset_reward_scale(cfg: dict[str, Any], pool: Any) -> None:
    env_cfg = cfg.setdefault("env", {})
    mode = str(env_cfg.get("reward_distance_scale_mode", "single_customer_repair_median"))
    if not mode.startswith("dataset_"):
        return
    base_mode = mode[len("dataset_") :]
    scale_fn = getattr(pool, "reward_distance_scale_km", None)
    if not callable(scale_fn):
        raise ValueError(
            "reward_distance_scale_mode uses dataset_ prefix, but the training pool "
            "does not provide dataset-level reward scale statistics."
        )
    scale = float(scale_fn(base_mode))
    env_cfg["reward_distance_scale_mode"] = base_mode
    env_cfg["reward_distance_scale_km"] = scale
    cfg.setdefault("normalization", {})["reward_distance_scale_km"] = scale
    cfg["normalization"]["reward_distance_scale_mode"] = mode
    cfg["normalization"]["reward_distance_scale_base_mode"] = base_mode
    cfg["normalization"]["reward_distance_scale_source"] = getattr(pool, "region_pool_status", "dataset")


def make_envs(cfg: dict[str, Any], seed: int):
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    num_envs = int(train_cfg.get("num_envs_per_gpu", 128))
    stage2_dataset_path = data_cfg.get("stage2_dataset_path")
    train_dataset_path = (
        data_cfg.get("train_dataset_path")
        or data_cfg.get("instance_dataset_path")
        or data_cfg.get("fixed_train_path")
    )
    if stage2_dataset_path not in (None, ""):
        pool = Stage2TERRANPool(
            dataset_path=_resolve_repo_path(stage2_dataset_path),
            family_root=_resolve_repo_path(data_cfg.get("stage2_family_root")),
            scale=data_cfg.get("stage2_scale", data_cfg.get("num_customers")),
            split_ids=data_cfg.get("stage2_split_ids", "train"),
            track_ids=data_cfg.get("stage2_track_ids", "train"),
            city_slugs=data_cfg.get("stage2_city_slugs"),
            seed=seed,
            cache_size=int(data_cfg.get("stage2_cache_size", 4)),
            completed_data_passes=int(data_cfg.get("stage2_completed_data_passes", 0)),
            completed_samples=int(data_cfg.get("stage2_completed_samples", 0)),
            training_stream_path=_resolve_repo_path(
                data_cfg.get("stage2_training_stream_path")
            ),
            representation=str(data_cfg.get("stage2_training_representation", "G")),
            euclidean_manifest=_resolve_repo_path(
                data_cfg.get("stage2_euclidean_manifest")
            ),
        )
    elif train_dataset_path not in (None, ""):
        pool = FixedDatasetInstancePool(
            dataset_path=train_dataset_path,
            num_customers=int(data_cfg.get("num_customers", 15)),
            num_charging_stations=int(data_cfg.get("num_charging_stations", 3)),
            seed=seed,
            sample_mode=str(data_cfg.get("train_sample_mode", "shuffle_cycle")),
        )
    else:
        common_pool_kwargs = dict(
            config_path=data_cfg.get("generator_config", "configs/amazon_hierarchy.yaml"),
            num_regions=int(data_cfg.get("mother_board_pool_size", 32)),
            mother_num_customers=int(data_cfg.get("mother_num_customers", 5000)),
            mother_num_charging_stations=int(data_cfg.get("mother_num_charging_stations", 120)),
            num_customers=int(data_cfg.get("num_customers", 15)),
            num_charging_stations=int(data_cfg.get("num_charging_stations", 3)),
            region_reuse_limit=int(data_cfg.get("region_reuse_limit", 200)),
            seed=seed,
            max_attempts_per_instance=data_cfg.get("max_attempts_per_instance"),
            territory_pool_path=data_cfg.get("territory_pool_path"),
            region_pool_path=data_cfg.get("region_pool_path"),
            region_pool_shuffle=bool(data_cfg.get("territory_pool_shuffle", data_cfg.get("region_pool_shuffle", True))),
            region_pool_replacement_policy=str(data_cfg.get("region_pool_replacement_policy", "cycle")),
        )
        if bool(data_cfg.get("async_instance_prefetch", False)):
            workers = int(data_cfg.get("async_instance_workers", min(8, max(1, num_envs))))
            queue_batches = int(data_cfg.get("async_instance_queue_batches", 2))
            regions_per_worker = data_cfg.get("async_regions_per_worker", None)
            pool = AsyncInstancePool(
                **common_pool_kwargs,
                num_workers=workers,
                queue_size=max(workers * 2, num_envs * max(1, queue_batches)),
                regions_per_worker=None if regions_per_worker is None else int(regions_per_worker),
                multiprocessing_context=str(data_cfg.get("async_multiprocessing_context", "spawn")),
                get_timeout_s=float(data_cfg.get("async_get_timeout_s", 300.0)),
            )
            pool.start()
        else:
            pool = OnlineInstancePool(**common_pool_kwargs)
    _configure_dataset_reward_scale(cfg, pool)
    pbrs_config = build_pbrs_config(cfg)
    env_cfg = dict(cfg.get("env", {}) or {})
    # The collector's registered budget is part of the training environment's
    # terminal semantics.  Evaluation environments are built separately and do
    # not receive this wrapper.
    env_cfg["rollout_horizon_steps"] = int(train_cfg.get("rollout_steps", 0))
    if env_cfg["rollout_horizon_steps"] <= 0:
        raise ValueError("training.rollout_steps must be positive")
    if bool(env_cfg.get("use_fast_env", True)):
        env_cfg.setdefault("info_level", "light")
    envs = [
        make_terran_env(
            instance_sampler=pool.sample,
            n_traj=int(train_cfg.get("n_traj", 100)),
            pbrs_config=pbrs_config,
            **env_cfg,
        )
        for _ in range(num_envs)
    ]
    return envs, pool

def evaluate_fixed_dataset(
    agent: Agent,
    cfg: dict[str, Any],
    seed: int,
    epoch: int,
    device: str | torch.device,
) -> dict[str, Any]:
    eval_cfg = cfg.get("evaluation", {})
    candidate_seed = int(eval_cfg.get("eval_seed", seed + 910_000_000))
    data_cfg = cfg.get("data", {})
    num_customers = int(data_cfg.get("num_customers", 15))
    num_cs = int(data_cfg.get("num_charging_stations", 3))
    eval_path = _resolve_repo_path(eval_cfg.get("eval_path"))
    n_traj = int(eval_cfg.get("eval_n_traj", 100))
    decode_mode = str(eval_cfg.get("eval_decode_mode", "sample"))
    configured_max_steps = eval_cfg.get("eval_max_steps")
    limit = eval_cfg.get("eval_limit", None)
    batch_size = max(1, int(eval_cfg.get("eval_batch_size", 1)))
    num_batches_limit = eval_cfg.get("eval_num_batches", None)
    eval_save_routes = bool(eval_cfg.get("eval_save_routes", False))
    eval_info_level = str(eval_cfg.get("eval_info_level", "light"))
    require_verifier = bool(
        eval_cfg.get("eval_require_independent_verifier", False)
    )
    if eval_path is None or not eval_path.exists():
        return {
            "eval_num_instances": 0,
            "eval_complete_and_feasible": 0,
            "eval_n_traj": n_traj,
            "eval_batch_size": batch_size,
            "eval_num_batches": 0,
            "eval_decode_mode": decode_mode,
            "eval_info_level": eval_info_level,
            "eval_save_routes": eval_save_routes,
            "eval_feasible_rate": np.nan,
            "eval_avg_objective_distance_km": np.nan,
            "eval_avg_vehicle_count": np.nan,
            "eval_avg_runtime_s": np.nan,
            "eval_status": f"missing_eval_path:{eval_path}",
        }

    if eval_cfg.get("eval_scale"):
        pool = Stage2TaskPool(
            dataset_path=eval_path,
            family_root=_resolve_repo_path(eval_cfg.get("eval_family_root")),
            scale=str(eval_cfg["eval_scale"]),
            split_ids=str(eval_cfg.get("eval_split_ids", "val")),
            track_ids=str(eval_cfg.get("eval_track_ids", "validation")),
            seed=int(seed) + 900_000,
            representation=str(eval_cfg.get("eval_representation", "G")),
            euclidean_manifest=_resolve_repo_path(
                eval_cfg.get("eval_euclidean_manifest")
            ),
        )
        fixed_instances = list(pool.first(limit=limit))
        instance_batches = [
            fixed_instances[offset : offset + batch_size]
            for offset in range(0, len(fixed_instances), batch_size)
        ]
        if num_batches_limit is not None:
            instance_batches = instance_batches[: int(num_batches_limit)]
    else:
        instance_batches = _eval_instance_batches(
            eval_path,
            num_customers,
            num_cs,
            batch_size,
            limit,
            num_batches_limit,
        )

    was_training = agent.training
    agent.eval()
    rows: list[dict[str, Any]] = []
    num_batches = 0
    seen_before_batch = 0
    for instances in instance_batches:
        eval_env_cfg = dict(cfg.get("env", {}) or {})
        if bool(eval_env_cfg.get("use_fast_env", True)):
            eval_env_cfg["info_level"] = (
                "full"
                if require_verifier or eval_save_routes
                else eval_info_level
            )
        envs = [
            make_terran_env(instance=instance, n_traj=n_traj, **eval_env_cfg)
            for instance in instances
        ]
        max_steps = (
            max(env.unwrapped.max_steps for env in envs)
            if configured_max_steps is None
            else int(configured_max_steps)
        )
        batch_rows = rollout_eval_batch(
            agent,
            envs,
            decode_mode=decode_mode,
            max_steps=max_steps,
            device=device,
            seed=candidate_seed + seen_before_batch,
            include_routes=eval_save_routes,
            return_final_info=require_verifier,
        )
        for instance, row in zip(instances, batch_rows):
            row["instance_id"] = instance.instance_id
            if require_verifier:
                info = row.pop("_final_info")
                _, routes, verification = select_min_verified_distance(
                    instance, info
                )
                row["feasible"] = bool(verification["passed"])
                row["objective_distance_km"] = float(
                    verification["objective_distance_km"]
                )
                row["vehicle_count"] = len(routes)
                row["verifier_passed"] = bool(verification["passed"])
        rows.extend(batch_rows)
        num_batches += 1
        seen_before_batch += len(instances)
    if was_training:
        agent.train()
    if not rows:
        return {
            "eval_num_instances": 0,
            "eval_complete_and_feasible": 0,
            "eval_n_traj": n_traj,
            "eval_batch_size": batch_size,
            "eval_num_batches": 0,
            "eval_decode_mode": decode_mode,
            "eval_info_level": eval_info_level,
            "eval_save_routes": eval_save_routes,
            "eval_feasible_rate": np.nan,
            "eval_avg_objective_distance_km": np.nan,
            "eval_avg_vehicle_count": np.nan,
            "eval_avg_runtime_s": np.nan,
            "eval_status": f"no_instances:{eval_path}",
        }

    feasible_rows = [row for row in rows if row["feasible"]]
    return {
        "eval_num_instances": len(rows),
        "eval_complete_and_feasible": len(feasible_rows),
        "eval_n_traj": n_traj,
        "eval_batch_size": batch_size,
        "eval_num_batches": num_batches,
        "eval_decode_mode": decode_mode,
        "eval_info_level": eval_info_level,
        "eval_save_routes": eval_save_routes,
        "eval_independent_verifier": require_verifier,
        "eval_feasible_rate": len(feasible_rows) / len(rows),
        "eval_avg_objective_distance_km": (
            float(
                np.mean(
                    [row["objective_distance_km"] for row in feasible_rows]
                )
            )
            if feasible_rows
            else np.nan
        ),
        "eval_avg_vehicle_count": (
            float(np.mean([row["vehicle_count"] for row in feasible_rows]))
            if feasible_rows
            else np.nan
        ),
        "eval_avg_runtime_s": float(
            np.mean([row["runtime_s"] for row in rows])
        ),
        "eval_status": "ok",
    }


def summarize_train_infos(final_infos: list[dict[str, Any]]) -> dict[str, Any]:
    if not final_infos:
        return {
            "train_feasible_rate": np.nan,
            "train_avg_best_objective_distance_km": np.nan,
            "train_avg_vehicle_count": np.nan,
            "train_avg_served_customers": np.nan,
        }
    feasible_flags = []
    best_objectives = []
    vehicle_counts = []
    served_counts = []
    for info in final_infos:
        success = np.asarray(info.get("success", []), dtype=bool)
        objective = np.asarray(info.get("objective_distance_km", []), dtype=np.float64)
        vehicle = np.asarray(info.get("vehicle_count", []), dtype=np.float64)
        served = np.asarray(info.get("served_customers", []), dtype=np.float64)
        if objective.size == 0:
            continue
        feasible_flags.extend(success.tolist())
        served_counts.extend(served.tolist())
        if np.any(success):
            candidates = np.where(success)[0]
            selected = int(candidates[np.argmin(objective[candidates])])
            best_objectives.append(float(objective[selected]))
            vehicle_counts.append(float(vehicle[selected]) if vehicle.size else np.nan)
    return {
        "train_feasible_rate": float(np.mean(feasible_flags)) if feasible_flags else np.nan,
        "train_avg_best_objective_distance_km": float(np.mean(best_objectives)) if best_objectives else np.nan,
        "train_avg_vehicle_count": float(np.mean(vehicle_counts)) if vehicle_counts else np.nan,
        "train_avg_served_customers": float(np.mean(served_counts)) if served_counts else np.nan,
    }


def _format_float(value: Any, precision: int = 4) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_f):
        return "nan"
    return f"{value_f:.{precision}f}"


def _debug_log(debug_enabled: bool, debug_file, message: str) -> None:
    if not debug_enabled:
        return
    print(message, flush=True)
    if debug_file is not None:
        debug_file.write(message + "\n")
        debug_file.flush()


def _slice_obs_by_env(obs: dict[str, Any], env_indices: Sequence[int] | np.ndarray) -> dict[str, Any]:
    indices = np.asarray(env_indices, dtype=np.int64)
    max_index = int(indices.max()) if indices.size else -1
    out: dict[str, Any] = {}
    for key, value in obs.items():
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape[0] > max_index:
            out[key] = arr[indices]
        else:
            out[key] = value
    return out


def evaluate_policy_loss(
    agent,
    batch,
    returns,
    advantages,
    cfg,
    device,
    env_indices: Sequence[int] | np.ndarray | None = None,
    step_start: int = 0,
    step_end: int | None = None,
):
    del device
    clip_coef = float(cfg["training"].get("clip_coef", 0.2))
    vf_coef = float(cfg["training"].get("vf_coef", 0.5))
    ent_coef = float(cfg["training"].get("ent_coef", 0.01))
    if env_indices is None:
        env_indices = np.arange(batch.actions.size(1), dtype=np.int64)
    else:
        env_indices = np.asarray(env_indices, dtype=np.int64)

    if step_end is None:
        step_end = len(batch.observations)
    step_start = max(0, int(step_start))
    step_end = min(len(batch.observations), int(step_end))
    if step_start >= step_end:
        raise ValueError(f"empty PPO step range: [{step_start}, {step_end})")

    # Static node embeddings are identical across rollout steps. Encode once for
    # the selected env minibatch, then reuse cached K/V/logit projections while
    # each step supplies its own dynamic state. For large Cus1000-style graphs,
    # callers can invoke this function on time chunks to avoid retaining all
    # decoder graphs until a single backward pass.
    first_obs = batch.observations[0]
    static_mb = _slice_obs_by_env(
        {key: first_obs[key] for key in STATIC_OBSERVATION_KEYS if key in first_obs},
        env_indices,
    )
    # Reuse the immutable instance tensors within this exact PPO time chunk.
    # The encoder is still run once per chunk, with unchanged training dropout;
    # no encoded state is reused across chunks or optimizer updates.
    static_device = {
        key: torch.as_tensor(value, device=agent.backbone.device)
        for key, value in static_mb.items()
    }

    def model_observation(obs):
        dynamic_mb = _slice_obs_by_env(
            {key: obs[key] for key in DYNAMIC_OBSERVATION_KEYS if key in obs},
            env_indices,
        )
        return {**static_device, **dynamic_mb}

    cached_state = agent.backbone.encode(model_observation(first_obs))

    policy_losses = []
    value_losses = []
    entropy_losses = []
    for step in range(step_start, step_end):
        obs = batch.observations[step]
        obs_mb = model_observation(obs)
        actions = batch.actions[step, env_indices].long()
        old_logprob = batch.old_logprobs[step, env_indices]
        _, new_logprob, entropy, value, _ = agent.get_action_and_value_cached(
            obs_mb,
            action=actions,
            state=cached_state,
        )
        value = value.squeeze(-1)
        ratio = torch.exp(new_logprob - old_logprob)
        adv = advantages[step, env_indices]
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv
        valid = batch.valid[step, env_indices]
        policy_losses.append(-masked_mean(torch.minimum(unclipped, clipped), valid))
        value_losses.append(masked_mean(F.mse_loss(value, returns[step, env_indices], reduction="none"), valid))
        entropy_losses.append(masked_mean(entropy, valid))
    policy_loss = torch.stack(policy_losses).mean()
    value_loss = torch.stack(value_losses).mean()
    entropy_loss = torch.stack(entropy_losses).mean()
    total = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss
    return total, policy_loss.detach(), value_loss.detach(), entropy_loss.detach()


def save_checkpoint(path: Path, agent: Agent, optimizer: torch.optim.Optimizer, cfg: dict[str, Any], epoch: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "seed": int(seed),
            "config": cfg,
            "model_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def train_from_config(cfg: dict[str, Any], seed: int, device: str | None = None, overrides: dict[str, Any] | None = None) -> Path:
    cfg = deep_update(cfg, overrides or {})
    set_seed(seed)
    train_cfg = cfg["training"]
    training_started = time.perf_counter()
    eval_cfg = cfg.get("evaluation", {})
    model_cfg = cfg.get("model", {})
    run_name = str(cfg.get("run_name", "TERRAN"))
    num_customers = int(cfg["data"].get("num_customers", 15))
    num_cs = int(cfg["data"].get("num_charging_stations", 3))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = Agent(
        embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        tanh_clipping=float(model_cfg.get("tanh_clipping", 15.0)),
        n_encode_layers=int(model_cfg.get("n_encode_layers", 3)),
        device=device,
        use_graph_token=bool(model_cfg.get("use_graph_token", False)),
        use_dynamic_embedding=bool(model_cfg.get("use_dynamic_embedding", False)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        agent.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        eps=1e-5,
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    initial_env_start = time.perf_counter()
    envs, pool = make_envs(cfg, seed)
    initial_env_pool_time_s = time.perf_counter() - initial_env_start
    gamma = float(train_cfg.get("gamma", 0.99))
    epochs = int(train_cfg.get("epochs", 1000))
    rollout_steps = int(train_cfg.get("rollout_steps", 64))
    ppo_epochs = int(train_cfg.get("ppo_update_epochs", 4))
    num_minibatches = max(1, int(train_cfg.get("num_minibatches", 1)))
    gradient_accumulation_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    logical_microbatches_per_epoch = max(
        1, int(train_cfg.get("logical_microbatches_per_epoch", 1))
    )
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 50))
    eval_interval = int(eval_cfg.get("eval_interval", 0) or 0)
    validation_seed = int(eval_cfg.get("eval_seed", seed + 910_000_000))
    debug_enabled = bool(train_cfg.get("debug", False))
    debug_log_every = max(1, int(train_cfg.get("debug_log_every", 1)))
    profile_timing = bool(train_cfg.get("profile_timing", False))
    ppo_step_chunk_size = int(train_cfg.get("ppo_step_chunk_size", 0) or 0)
    protocol_cfg = cfg.get("protocol", {})
    registered_effective_batch = int(
        protocol_cfg.get(
            "logical_environments_per_epoch",
            len(envs) * logical_microbatches_per_epoch,
        )
        or len(envs) * logical_microbatches_per_epoch
    )
    if len(envs) * logical_microbatches_per_epoch != registered_effective_batch:
        raise ValueError(
            "TERRAN physical rollout count does not match the registered "
            "effective batch"
        )
    environment_transitions_total = int(
        protocol_cfg.get("environment_transitions", 0) or 0
    )
    optimizer_steps_total = int(protocol_cfg.get("optimizer_steps", 0) or 0)
    start_epoch = 1
    resume_checkpoint = protocol_cfg.get("resume_checkpoint")
    if resume_checkpoint:
        payload = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        agent.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload["epoch"]) + 1

    if cfg.get("output_dir"):
        out_root = Path(cfg["output_dir"])
        ckpt_dir = out_root / "checkpoints"
        log_dir = out_root / "logs"
    else:
        out_root = REPO_ROOT / "EVRPTW_Benchmark/Reinforcement_Learning/TERRAN"
        ckpt_dir = out_root / "checkpoints" / f"Cus_{num_customers}_CS_{num_cs}" / run_name / f"seed_{seed}"
        log_dir = out_root / "logs" / f"Cus_{num_customers}_CS_{num_cs}" / run_name / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_log.csv"
    eval_log_path = log_dir / "eval_log.csv"
    debug_log_path = log_dir / "debug_log.txt"
    validation_history_path = out_root / "validation_history.jsonl"
    validation_summary_path = out_root / "validation_summary.json"
    validation_summary_within_path = out_root / "validation_summary_within_5000.json"
    validation_summary_overall_path = out_root / "validation_summary_overall.json"
    best_checkpoint_path = out_root / "best.ckpt"
    best_within_minimum_path = out_root / "best_within_5000.ckpt"
    best_overall_path = out_root / "best_overall.ckpt"
    selected_checkpoint_path = out_root / "checkpoint_selected.pt"
    best_eval_key = (-math.inf, -math.inf)
    best_within_minimum_key = (-math.inf, -math.inf)
    previous_overall_path = (
        validation_summary_overall_path
        if validation_summary_overall_path.is_file()
        else validation_summary_path
    )
    if previous_overall_path.is_file():
        previous_validation = json.loads(
            previous_overall_path.read_text(encoding="utf-8")
        )
        previous_distance = previous_validation.get(
            "mean_verified_distance_km"
        )
        best_eval_key = (
            float(previous_validation["complete_and_feasible_rate"]),
            (
                -float(previous_distance)
                if previous_distance is not None
                else -math.inf
            ),
        )
    previous_within_path = (
        validation_summary_within_path
        if validation_summary_within_path.is_file()
        else validation_summary_path
    )
    if previous_within_path.is_file():
        previous_within = json.loads(previous_within_path.read_text(encoding="utf-8"))
        previous_within_distance = previous_within.get("mean_verified_distance_km")
        best_within_minimum_key = (
            float(previous_within["complete_and_feasible_rate"]),
            (
                -float(previous_within_distance)
                if previous_within_distance is not None
                else -math.inf
            ),
        )
    minimum_training_epochs = int(
        train_cfg.get("minimum_training_epochs", epochs) or epochs
    )
    scheduled_validation_epochs = {
        int(value) for value in train_cfg.get("validation_epochs", [])
    }
    early_stop_patience = int(
        train_cfg.get("early_stop_patience_validations", 0) or 0
    )
    if early_stop_patience < 0:
        raise ValueError("early_stop_patience_validations cannot be negative")
    early_stop_start_epoch = int(
        train_cfg.get("early_stop_start_epoch", 0) or 0
    )
    if early_stop_start_epoch < 0:
        raise ValueError("early_stop_start_epoch cannot be negative")
    if early_stop_start_epoch >= epochs:
        raise ValueError("early_stop_start_epoch must be smaller than epochs")
    completed_validation_checks = 0
    validation_checks_without_improvement = 0
    history_best_key = (-math.inf, -math.inf)
    if validation_history_path.is_file():
        for line in validation_history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            distance = row.get("mean_verified_distance_km")
            key = (
                float(row["complete_and_feasible_rate"]),
                -float(distance) if distance is not None else -math.inf,
            )
            completed_validation_checks += 1
            logical_epoch = int(row.get("logical_epoch", 0) or 0)
            if key > history_best_key:
                history_best_key = key
                validation_checks_without_improvement = 0
            elif logical_epoch > early_stop_start_epoch:
                validation_checks_without_improvement += 1
            else:
                validation_checks_without_improvement = 0
    early_stopped = False
    early_stop_epoch: int | None = None
    completed_epoch = start_epoch - 1
    exposure_checkpoints = tuple(int(value) for value in protocol_cfg.get("exposure_checkpoints", []))
    gpu_hour_checkpoints = tuple(float(value) for value in protocol_cfg.get("gpu_hour_checkpoints", []))
    saved_exposure = {
        value for value in exposure_checkpoints
        if (ckpt_dir / f"checkpoint_customer_exposure_{value}.pt").is_file()
    }
    saved_gpu_hours = {
        value for value in gpu_hour_checkpoints
        if (ckpt_dir / f"checkpoint_gpu_hours_{value:g}.pt").is_file()
    }

    train_fields = [
        "epoch",
        "reward_mean",
        "policy_loss",
        "value_loss",
        "entropy",
        "samples_seen",
        "environment_transitions",
        "environment_transitions_total",
        "optimizer_steps_total",
        "num_envs",
        "n_traj",
        "rollout_steps",
        "trajectory_count",
        "mean_trajectory_steps",
        "trajectory_steps_p50",
        "trajectory_steps_p90",
        "trajectory_steps_p99",
        "trajectory_steps_max",
        "rollout_budget_exhausted_count",
        "rollout_budget_exhausted_rate",
        "num_minibatches",
        "gradient_accumulation_steps",
        "logical_microbatches_per_epoch",
        "effective_instances_per_optimizer_step",
        "pbrs_scale",
        "initial_env_pool_time_s",
        "rollout_reset_time_s",
        "rollout_stack_obs_time_s",
        "rollout_model_action_time_s",
        "rollout_env_step_time_s",
        "rollout_interaction_time_s",
        "rollout_total_time_s",
        "ppo_update_time_s",
        "eval_wall_time_s",
        "epoch_wall_time_s",
        "train_feasible_rate",
        "train_avg_best_objective_distance_km",
        "train_avg_vehicle_count",
        "train_avg_served_customers",
        "eval_avg_objective_distance_km",
        "eval_avg_vehicle_count",
        "eval_feasible_rate",
        "eval_avg_runtime_s",
        "eval_num_instances",
        "eval_n_traj",
        "eval_batch_size",
        "eval_num_batches",
        "eval_decode_mode",
        "eval_info_level",
        "eval_save_routes",
        "eval_status",
    ]
    eval_fields = [
        "epoch",
        "eval_avg_objective_distance_km",
        "eval_avg_vehicle_count",
        "eval_feasible_rate",
        "eval_avg_runtime_s",
        "eval_num_instances",
        "eval_complete_and_feasible",
        "eval_independent_verifier",
        "eval_n_traj",
        "eval_batch_size",
        "eval_num_batches",
        "eval_decode_mode",
        "eval_info_level",
        "eval_save_routes",
        "eval_status",
    ]

    log_mode = "a" if start_epoch > 1 else "w"
    needs_header = log_mode == "w" or not log_path.exists()
    with log_path.open(log_mode, newline="", encoding="utf-8") as f, eval_log_path.open(log_mode, newline="", encoding="utf-8") as ef, debug_log_path.open(log_mode, encoding="utf-8") as df:
        writer = csv.DictWriter(f, fieldnames=train_fields)
        eval_writer = csv.DictWriter(ef, fieldnames=eval_fields)
        if needs_header:
            writer.writeheader()
            eval_writer.writeheader()
        _debug_log(
            debug_enabled,
            df,
            f"[Init] run={run_name} seed={seed} device={device} epochs={epochs} "
            f"n_traj={train_cfg.get('n_traj', 100)} rollout_steps={rollout_steps} "
            f"num_envs={train_cfg.get('num_envs_per_gpu', 128)} minibatches={num_minibatches} "
            f"accum_grad={gradient_accumulation_steps} "
            f"n_encode_layers={model_cfg.get('n_encode_layers', 3)} "
            f"initial_env_pool_time_s={initial_env_pool_time_s:.3f} "
            f"eval_interval={eval_interval} eval_n_traj={eval_cfg.get('eval_n_traj', 100)} "
            f"eval_batch_size={eval_cfg.get('eval_batch_size', 1)} "
            f"eval_info_level={eval_cfg.get('eval_info_level', 'light')} "
            f"pbrs_annealing={cfg.get('pbrs', {}).get('annealing', {})}",
        )
        for epoch in range(start_epoch, epochs + 1):
            completed_epoch = epoch
            epoch_seed = seed + epoch * 100_000
            torch.manual_seed(epoch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(epoch_seed)
            epoch_start = time.perf_counter()
            pbrs_scale = pbrs_scale_for_epoch(cfg, epoch, epochs)
            set_pbrs_reward_scale(envs, pbrs_scale)
            agent.train()
            rollout_records: list[tuple[Any, torch.Tensor, torch.Tensor]] = []
            rollout_timings: dict[str, float] = {}
            final_infos: list[dict[str, Any]] = []
            trajectory_parts: list[np.ndarray] = []
            reward_sum = 0.0
            reward_count = 0
            environment_transitions = 0
            rollout_budget_exhausted_count = 0
            for microbatch_index in range(logical_microbatches_per_epoch):
                batch = collect_rollout(
                    agent,
                    envs,
                    rollout_steps=rollout_steps,
                    decode_mode="sample",
                    device=device,
                    seed=seed + epoch * 100_000 + microbatch_index,
                    profile_timing=profile_timing,
                )
                returns = compute_returns(batch.rewards, batch.dones, gamma=gamma)
                advantages = returns - batch.values
                rollout_records.append((batch, returns, advantages))
                valid_count = int(batch.valid.sum().item())
                environment_transitions += valid_count
                reward_count += valid_count
                if valid_count:
                    reward_sum += float(batch.rewards[batch.valid].sum().detach().cpu())
                trajectory_parts.append(
                    batch.trajectory_steps.detach().cpu().numpy().reshape(-1)
                )
                final_infos.extend(batch.final_infos)
                rollout_budget_exhausted_count += int(
                    batch.rollout_budget_exhausted.sum().detach().cpu()
                )
                for key, value in batch.timings.items():
                    rollout_timings[key] = rollout_timings.get(key, 0.0) + float(value)

            environment_transitions_total += environment_transitions
            trajectory_steps = np.concatenate(trajectory_parts)
            trajectory_count = int(trajectory_steps.size)
            all_advantages = torch.cat(
                [advantages[batch.valid] for batch, _, advantages in rollout_records]
            )
            if all_advantages.numel() > 1:
                advantage_mean = all_advantages.mean()
                advantage_std = all_advantages.std(unbiased=False)
                rollout_records = [
                    (
                        batch,
                        returns,
                        (advantages - advantage_mean) / (advantage_std + 1e-8),
                    )
                    for batch, returns, advantages in rollout_records
                ]

            losses = []
            num_envs = int(rollout_records[0][0].actions.size(1))
            minibatches = min(num_minibatches, num_envs)
            effective_instances = num_envs * logical_microbatches_per_epoch
            if profile_timing:
                _sync_cuda(device)
            ppo_start = time.perf_counter()
            epoch_rng = np.random.default_rng(epoch_seed + 17)
            if logical_microbatches_per_epoch == 1:
                batch, returns, advantages = rollout_records[0]
                env_order = np.arange(num_envs, dtype=np.int64)
                total_steps = int(batch.actions.size(0))
                chunk_size = (
                    ppo_step_chunk_size
                    if ppo_step_chunk_size > 0
                    else total_steps
                )
                chunk_size = max(1, min(chunk_size, total_steps))
                for _ in range(ppo_epochs):
                    epoch_rng.shuffle(env_order)
                    split_indices = [
                        indices
                        for indices in np.array_split(env_order, minibatches)
                        if indices.size > 0
                    ]
                    for group_start in range(
                        0, len(split_indices), gradient_accumulation_steps
                    ):
                        accum_group = split_indices[
                            group_start : group_start + gradient_accumulation_steps
                        ]
                        if not accum_group:
                            continue
                        optimizer.zero_grad(set_to_none=True)
                        group_policy = 0.0
                        group_value = 0.0
                        group_entropy = 0.0
                        group_size = float(len(accum_group))
                        for env_indices in accum_group:
                            weighted_policy = 0.0
                            weighted_value = 0.0
                            weighted_entropy = 0.0
                            for step_start in range(0, total_steps, chunk_size):
                                step_end = min(
                                    step_start + chunk_size, total_steps
                                )
                                chunk_weight = float(
                                    step_end - step_start
                                ) / max(float(total_steps), 1.0)
                                loss, policy_loss, value_loss, entropy = (
                                    evaluate_policy_loss(
                                        agent,
                                        batch,
                                        returns,
                                        advantages.detach(),
                                        cfg,
                                        device,
                                        env_indices=env_indices,
                                        step_start=step_start,
                                        step_end=step_end,
                                    )
                                )
                                (loss * chunk_weight / group_size).backward()
                                weighted_policy += (
                                    policy_loss.item() * chunk_weight
                                )
                                weighted_value += value_loss.item() * chunk_weight
                                weighted_entropy += entropy.item() * chunk_weight
                            group_policy += weighted_policy / group_size
                            group_value += weighted_value / group_size
                            group_entropy += weighted_entropy / group_size
                        torch.nn.utils.clip_grad_norm_(
                            agent.parameters(),
                            float(train_cfg.get("max_grad_norm", 1.0)),
                        )
                        optimizer.step()
                        optimizer_steps_total += 1
                        losses.append(
                            (group_policy, group_value, group_entropy)
                        )
            else:
                # Multiple physical rollout buffers form one effective batch.
                # PPO gradients are weighted by their base-instance share and
                # accumulated before each optimizer step, keeping GPU residency
                # at the registered physical batch size.
                for _ in range(ppo_epochs):
                    optimizer.zero_grad(set_to_none=True)
                    group_policy = 0.0
                    group_value = 0.0
                    group_entropy = 0.0
                    for record_index, (batch, returns, advantages) in enumerate(
                        rollout_records
                    ):
                        record_envs = int(batch.actions.size(1))
                        env_order = np.arange(record_envs, dtype=np.int64)
                        epoch_rng.shuffle(env_order)
                        record_minibatches = min(num_minibatches, record_envs)
                        split_indices = [
                            indices
                            for indices in np.array_split(
                                env_order, record_minibatches
                            )
                            if indices.size > 0
                        ]
                        total_steps = int(batch.actions.size(0))
                        chunk_size = (
                            ppo_step_chunk_size
                            if ppo_step_chunk_size > 0
                            else total_steps
                        )
                        chunk_size = max(1, min(chunk_size, total_steps))
                        for env_indices in split_indices:
                            instance_weight = (
                                float(len(env_indices))
                                / float(effective_instances)
                            )
                            for step_start in range(
                                0, total_steps, chunk_size
                            ):
                                step_end = min(
                                    step_start + chunk_size, total_steps
                                )
                                chunk_weight = float(
                                    step_end - step_start
                                ) / max(float(total_steps), 1.0)
                                loss, policy_loss, value_loss, entropy = (
                                    evaluate_policy_loss(
                                        agent,
                                        batch,
                                        returns,
                                        advantages.detach(),
                                        cfg,
                                        device,
                                        env_indices=env_indices,
                                        step_start=step_start,
                                        step_end=step_end,
                                    )
                                )
                                weight = instance_weight * chunk_weight
                                (loss * weight).backward()
                                group_policy += policy_loss.item() * weight
                                group_value += value_loss.item() * weight
                                group_entropy += entropy.item() * weight
                    torch.nn.utils.clip_grad_norm_(
                        agent.parameters(),
                        float(train_cfg.get("max_grad_norm", 1.0)),
                    )
                    optimizer.step()
                    optimizer_steps_total += 1
                    losses.append((group_policy, group_value, group_entropy))
            if profile_timing:
                _sync_cuda(device)
            ppo_update_time_s = time.perf_counter() - ppo_start
            reward_mean = reward_sum / max(reward_count, 1)
            loss_arr = np.asarray(losses, dtype=float)
            train_summary = summarize_train_infos(final_infos)
            if epoch % debug_log_every == 0:
                _debug_log(
                    debug_enabled,
                    df,
                    "[Train] "
                    f"epoch={epoch}/{epochs} samples={pool.sample_count} "
                    f"reward={_format_float(reward_mean)} "
                    f"policy_loss={_format_float(loss_arr[:, 0].mean())} "
                    f"value_loss={_format_float(loss_arr[:, 1].mean())} "
                    f"entropy={_format_float(loss_arr[:, 2].mean())} "
                    f"train_fr={_format_float(train_summary['train_feasible_rate'])} "
                    f"train_obj={_format_float(train_summary['train_avg_best_objective_distance_km'])} "
                    f"train_veh={_format_float(train_summary['train_avg_vehicle_count'])} "
                    f"served={_format_float(train_summary['train_avg_served_customers'])} "
                    f"pbrs_scale={pbrs_scale:.4f} "
                    f"timing_reset={rollout_timings.get('rollout_reset_time_s', 0.0):.3f}s "
                    f"timing_model={rollout_timings.get('rollout_model_action_time_s', 0.0):.3f}s "
                    f"timing_env={rollout_timings.get('rollout_env_step_time_s', 0.0):.3f}s "
                    f"timing_ppo={ppo_update_time_s:.3f}s",
                )
            eval_row: dict[str, Any] = {}
            eval_wall_time_s = 0.0
            early_stop_due = False
            should_eval = (
                epoch in scheduled_validation_epochs
                if scheduled_validation_epochs
                else eval_interval > 0 and (epoch % eval_interval == 0 or epoch == epochs)
            )
            if should_eval:
                eval_start = time.perf_counter()
                eval_row = evaluate_fixed_dataset(
                    agent, cfg, seed=seed, epoch=epoch, device=device
                )
                eval_wall_time_s = time.perf_counter() - eval_start
                eval_writer.writerow({"epoch": epoch, **eval_row})
                ef.flush()
                if eval_row.get("eval_status") == "ok":
                    verified_distance = eval_row.get(
                        "eval_avg_objective_distance_km"
                    )
                    validation = {
                        "schema": "drl_validation_summary_v1",
                        "split": "validation",
                        "logical_epoch": epoch,
                        "validation_seed": validation_seed,
                        "instances": int(eval_row["eval_num_instances"]),
                        "complete_and_feasible": int(
                            eval_row["eval_complete_and_feasible"]
                        ),
                        "complete_and_feasible_rate": float(
                            eval_row["eval_feasible_rate"]
                        ),
                        "mean_verified_distance_km": (
                            float(verified_distance)
                            if verified_distance is not None
                            and np.isfinite(float(verified_distance))
                            else None
                        ),
                        "validation_wall_time_s": eval_wall_time_s,
                        "verifier_summary_passed": bool(
                            eval_row.get("eval_independent_verifier", False)
                            and int(eval_row["eval_complete_and_feasible"])
                            == int(eval_row["eval_num_instances"])
                        ),
                    }
                    selection_key = (
                        validation["complete_and_feasible_rate"],
                        (
                            -validation["mean_verified_distance_km"]
                            if validation["mean_verified_distance_km"]
                            is not None
                            else -math.inf
                        ),
                    )
                    is_best_overall = selection_key > best_eval_key
                    is_best_within_minimum = bool(
                        epoch <= minimum_training_epochs
                        and selection_key > best_within_minimum_key
                    )
                    if is_best_overall:
                        validation_checks_without_improvement = 0
                    elif epoch > early_stop_start_epoch:
                        validation_checks_without_improvement += 1
                    else:
                        validation_checks_without_improvement = 0
                    completed_validation_checks += 1
                    early_stop_eligible = epoch > early_stop_start_epoch
                    early_stop_due = bool(
                        early_stop_patience
                        and early_stop_eligible
                        and validation_checks_without_improvement
                        >= early_stop_patience
                    )
                    validation.update(
                        {
                            "checkpoint_selected": is_best_within_minimum,
                            "best_within_minimum_selected": is_best_within_minimum,
                            "best_overall_selected": is_best_overall,
                            "minimum_training_epochs": minimum_training_epochs,
                            "validation_checks_without_improvement": validation_checks_without_improvement,
                            "early_stop_start_epoch": early_stop_start_epoch,
                            "early_stop_eligible": early_stop_eligible,
                            "early_stop_due": early_stop_due,
                        }
                    )
                    append_jsonl(validation_history_path, validation)
                    if is_best_overall:
                        best_eval_key = selection_key
                        save_checkpoint(
                            best_overall_path, agent, optimizer, cfg, epoch, seed
                        )
                        atomic_json(validation_summary_overall_path, validation)
                    if is_best_within_minimum:
                        best_within_minimum_key = selection_key
                        save_checkpoint(
                            best_within_minimum_path, agent, optimizer, cfg, epoch, seed
                        )
                        save_checkpoint(
                            best_checkpoint_path, agent, optimizer, cfg, epoch, seed
                        )
                        save_checkpoint(
                            selected_checkpoint_path, agent, optimizer, cfg, epoch, seed
                        )
                        atomic_json(validation_summary_path, validation)
                        atomic_json(validation_summary_within_path, validation)
                _debug_log(
                    debug_enabled,
                    df,
                    "[Eval] "
                    f"epoch={epoch}/{epochs} n={eval_row.get('eval_num_instances')} "
                    f"n_traj={eval_row.get('eval_n_traj')} "
                    f"batch={eval_row.get('eval_batch_size')}x{eval_row.get('eval_num_batches')} "
                    f"mode={eval_row.get('eval_decode_mode')} "
                    f"info={eval_row.get('eval_info_level')} "
                    f"fr={_format_float(eval_row.get('eval_feasible_rate'))} "
                    f"obj={_format_float(eval_row.get('eval_avg_objective_distance_km'))} "
                    f"veh={_format_float(eval_row.get('eval_avg_vehicle_count'))} "
                    f"runtime={_format_float(eval_row.get('eval_avg_runtime_s'))} "
                    f"eval_wall={eval_wall_time_s:.3f}s "
                    f"status={eval_row.get('eval_status')}",
                )
            epoch_wall_time_s = time.perf_counter() - epoch_start
            writer.writerow(
                {
                    "epoch": epoch,
                    "reward_mean": reward_mean,
                    "policy_loss": float(loss_arr[:, 0].mean()),
                    "value_loss": float(loss_arr[:, 1].mean()),
                    "entropy": float(loss_arr[:, 2].mean()),
                    "samples_seen": pool.sample_count,
                    "environment_transitions": environment_transitions,
                    "environment_transitions_total": environment_transitions_total,
                    "optimizer_steps_total": optimizer_steps_total,
                    "num_envs": num_envs,
                    "n_traj": int(train_cfg.get("n_traj", 100)),
                    "rollout_steps": rollout_steps,
                    "num_minibatches": minibatches,
                    "trajectory_count": trajectory_count,
                    "mean_trajectory_steps": float(trajectory_steps.mean()),
                    "trajectory_steps_p50": float(np.quantile(trajectory_steps, 0.50)),
                    "trajectory_steps_p90": float(np.quantile(trajectory_steps, 0.90)),
                    "trajectory_steps_p99": float(np.quantile(trajectory_steps, 0.99)),
                    "trajectory_steps_max": int(trajectory_steps.max()),
                    "rollout_budget_exhausted_count": (
                        rollout_budget_exhausted_count
                    ),
                    "rollout_budget_exhausted_rate": (
                        rollout_budget_exhausted_count / trajectory_count
                    ),
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "logical_microbatches_per_epoch": logical_microbatches_per_epoch,
                    "effective_instances_per_optimizer_step": (
                        effective_instances
                        if logical_microbatches_per_epoch > 1
                        else int(np.ceil(num_envs / max(minibatches, 1)))
                        * gradient_accumulation_steps
                    ),
                    "pbrs_scale": pbrs_scale,
                    "initial_env_pool_time_s": initial_env_pool_time_s,
                    "rollout_reset_time_s": rollout_timings.get("rollout_reset_time_s", ""),
                    "rollout_stack_obs_time_s": rollout_timings.get("rollout_stack_obs_time_s", ""),
                    "rollout_model_action_time_s": rollout_timings.get("rollout_model_action_time_s", ""),
                    "rollout_env_step_time_s": rollout_timings.get("rollout_env_step_time_s", ""),
                    "rollout_interaction_time_s": rollout_timings.get("rollout_interaction_time_s", ""),
                    "rollout_total_time_s": rollout_timings.get("rollout_total_time_s", ""),
                    "ppo_update_time_s": ppo_update_time_s,
                    "eval_wall_time_s": eval_wall_time_s,
                    "epoch_wall_time_s": epoch_wall_time_s,
                    "train_feasible_rate": train_summary.get("train_feasible_rate", ""),
                    "train_avg_best_objective_distance_km": train_summary.get("train_avg_best_objective_distance_km", ""),
                    "train_avg_vehicle_count": train_summary.get("train_avg_vehicle_count", ""),
                    "train_avg_served_customers": train_summary.get("train_avg_served_customers", ""),
                    "eval_avg_objective_distance_km": eval_row.get("eval_avg_objective_distance_km", ""),
                    "eval_avg_vehicle_count": eval_row.get("eval_avg_vehicle_count", ""),
                    "eval_feasible_rate": eval_row.get("eval_feasible_rate", ""),
                    "eval_avg_runtime_s": eval_row.get("eval_avg_runtime_s", ""),
                    "eval_num_instances": eval_row.get("eval_num_instances", ""),
                    "eval_n_traj": eval_row.get("eval_n_traj", ""),
                    "eval_batch_size": eval_row.get("eval_batch_size", ""),
                    "eval_num_batches": eval_row.get("eval_num_batches", ""),
                    "eval_decode_mode": eval_row.get("eval_decode_mode", ""),
                    "eval_info_level": eval_row.get("eval_info_level", ""),
                    "eval_save_routes": eval_row.get("eval_save_routes", ""),
                    "eval_status": eval_row.get("eval_status", ""),
                }
            )
            f.flush()
            if should_eval or epoch % checkpoint_interval == 0 or epoch == epochs:
                save_checkpoint(ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt", agent, optimizer, cfg, epoch, seed)
            observed_exposure = int(pool.sample_count) * num_customers
            observed_gpu_hours = (time.perf_counter() - training_started) / 3600.0
            schedules = (
                ("customer_exposure", exposure_checkpoints, saved_exposure, observed_exposure),
                ("gpu_hours", gpu_hour_checkpoints, saved_gpu_hours, observed_gpu_hours),
            )
            for axis, thresholds, saved, observed in schedules:
                for requested in thresholds:
                    if requested in saved or observed < requested:
                        continue
                    suffix = str(requested) if axis == "customer_exposure" else f"{requested:g}"
                    snapshot = ckpt_dir / f"checkpoint_{axis}_{suffix}.pt"
                    save_checkpoint(snapshot, agent, optimizer, cfg, epoch, seed)
                    saved.add(requested)
                    with (log_dir / "checkpoint_events.jsonl").open("a", encoding="utf-8") as events:
                        events.write(
                            json.dumps(
                                {
                                    "schema": "drl_training_checkpoint_event_v1",
                                    "axis": axis,
                                    "requested": requested,
                                    "observed_customer_exposures": observed_exposure,
                                    "observed_gpu_hours": observed_gpu_hours,
                                    "epoch": epoch,
                                    "path": str(snapshot),
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
            protocol_epochs = int(protocol_cfg.get("epochs_per_pass", 0) or 0)
            if (
                protocol_cfg
                and protocol_epochs > 0
                and (should_eval or epoch % protocol_epochs == 0)
            ):
                latest = out_root / "checkpoint_latest.pt"
                save_checkpoint(
                    latest,
                    agent,
                    optimizer,
                    cfg,
                    epoch,
                    seed,
                )
                completed = epoch // protocol_epochs
                state = DataPassState(
                    protocol_id=str(protocol_cfg["protocol_id"]),
                    completed_data_passes=completed,
                    instances_seen=int(pool.sample_count),
                    customer_exposures=int(pool.sample_count) * num_customers,
                    optimizer_steps=optimizer_steps_total,
                    environment_transitions=environment_transitions_total,
                    last_checkpoint=str(latest),
                )
                state.atomic_write(out_root / "data_pass_state.json")
            if early_stop_due and epoch < epochs:
                early_stopped = True
                early_stop_epoch = epoch
                break
    atomic_json(
        out_root / "early_stop_state.json",
        {
            "schema": "drl_early_stop_state_v1",
            "requested_training_epochs": epochs,
            "completed_training_epochs": completed_epoch,
            "completed_validation_checkpoints": completed_validation_checks,
            "validation_checks_without_improvement": validation_checks_without_improvement,
            "early_stop_patience_validations": early_stop_patience,
            "early_stop_start_epoch": early_stop_start_epoch,
            "early_stopped": early_stopped,
            "early_stop_epoch": early_stop_epoch,
        },
    )
    save_checkpoint(
        ckpt_dir / "checkpoint_final.pt", agent, optimizer, cfg, completed_epoch, seed
    )
    close_pool = getattr(pool, "close", None)
    if callable(close_pool):
        close_pool(terminate=True)
    return ckpt_dir / "checkpoint_final.pt"
