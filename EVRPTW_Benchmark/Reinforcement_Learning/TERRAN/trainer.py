from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Mapping, Sequence
from types import SimpleNamespace
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))
sys.path.insert(0, str(REPO_ROOT))

from evrptw_core.io import iter_instances
from evrptw_core.schema import merge_route_sequences

from .async_instances import AsyncInstancePool
from ..common import Stage2TaskPool
from ..common.data_pass import DataPassState
from ..common.evaluation import select_min_verified_objective
from ..common.objective import objective_from_checkpoint, resolve_objective
from ..common.reward_contract import RewardContract, load_reward_contract
from ..common.training_diagnostics import summarize_values
from ..common.training_protocol import (
    append_jsonl,
    atomic_json,
    build_adamw_optimizer,
    resolved_training_signature_digest,
    resolved_training_signature_from_args,
    validation_key,
)
from ..common.training_stream import (
    STREAM_CONTRACT_SCHEMA,
    STREAM_INTEGRITY_MODE_RUNTIME_REVERIFIED,
    training_stream_contract_digest,
)
from .data_pool import FixedDatasetInstancePool, OnlineInstancePool, Stage2TERRANPool
from .env_factory import make_terran_env
from .models import Agent
from .models.attention_model_wrapper import (
    DYNAMIC_OBSERVATION_KEYS,
    STATIC_OBSERVATION_KEYS,
)
from .pbrs import (
    TERMINAL_TASK_REWARD_UNIT,
    PotentialRewardConfig,
)
from .rollout import (
    BoundedBaseRewardStats,
    collect_rollout,
    compute_returns,
    rollout_eval_batch,
    summarize_rollout_outcomes,
)

OBJECTIVE_EVAL_FIELDS = (
    "eval_objective_mode", "eval_objective_unit", "eval_avg_objective",
    "eval_avg_objective_cost_usd", "eval_avg_electricity_cost_usd", "eval_avg_vehicle_cost_usd",
)
OBJECTIVE_REWARD_COMPONENTS = ("objective", "electricity_cost", "vehicle_cost", "base_non_objective")
EVAL_OUTCOME_FIELDS = (
    "eval_candidate_trajectory_count",
    "eval_candidate_success_count",
    "eval_candidate_success_rate",
    "eval_candidate_rollout_budget_exhausted_count",
    "eval_candidate_rollout_budget_exhausted_rate",
    "eval_candidate_non_horizon_infeasible_count",
    "eval_candidate_non_horizon_infeasible_rate",
    "eval_candidate_non_horizon_infeasible_reason_counts",
    "eval_no_success_all_candidates_non_horizon_infeasible_instance_count",
    "eval_no_success_all_candidates_non_horizon_infeasible_instance_rate",
)


def _summarize_tensor_parts(parts, max_quantile_samples: int = 8192) -> dict[str, Any]:
    """Pool detached buffer statistics without concatenating physical rollouts.

    Full moments are merged by population size. For multiple physical buffers,
    quantiles use global evenly spaced finite-observation positions; at most K
    selected values (never complete reward/return buffers) cross to the CPU.
    """
    parts = list(parts)
    if not parts:
        return summarize_values(np.empty(0))
    summaries = [summarize_values(values, mask, max_quantile_samples) for values, mask in parts]
    if len(parts) == 1:
        return summaries[0]
    result = summarize_values(np.empty(0))
    result.update({key: sum(row[key] for row in summaries) for key in ("count", "finite_count", "nonfinite_count")})
    size = result["finite_count"]
    if not size:
        return result
    nonempty = [row for row in summaries if row["finite_count"]]
    if all(row["mean"] is not None and row["std"] is not None for row in nonempty):
        mean = sum(row["mean"] * row["finite_count"] / size for row in nonempty)
        variance = sum((row["std"] ** 2 + (row["mean"] - mean) ** 2) * row["finite_count"] / size for row in nonempty)
        result.update(mean=mean, std=math.sqrt(max(variance, 0.0)))
    result.update(min=min(row["min"] for row in nonempty), max=max(row["max"] for row in nonempty))
    positions = np.linspace(0, size - 1, min(size, max_quantile_samples)).round().astype(np.int64)
    sample_parts, offset = [], 0
    with torch.no_grad():
        for (values, mask), summary in zip(parts, summaries):
            local = positions[(positions >= offset) & (positions < offset + summary["finite_count"])] - offset
            offset += summary["finite_count"]
            if not local.size:
                continue
            data = values.detach().reshape(-1)
            flat_mask = mask.reshape(-1) if mask is not None else None
            finite_seen = 0
            for start in range(0, data.numel(), 262144):
                chunk = data[start:start + 262144]
                if flat_mask is not None:
                    chunk = chunk[flat_mask[start:start + 262144]]
                finite = chunk[torch.isfinite(chunk)]
                chosen = local[(local >= finite_seen) & (local < finite_seen + finite.numel())] - finite_seen
                finite_seen += finite.numel()
                if chosen.size:
                    indices = torch.as_tensor(chosen, device=finite.device)
                    sample_parts.append(finite[indices].double().cpu().numpy())
    sampled = summarize_values(np.concatenate(sample_parts), max_quantile_samples=max_quantile_samples)
    result.update({key: sampled[key] for key in ("p05", "p50", "p95", "quantile_sample_count")})
    return result


def _json_safe_diagnostics(value):
    if isinstance(value, dict):
        return {key: _json_safe_diagnostics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_diagnostics(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


@torch.no_grad()
def build_epoch_reward_diagnostics(
    *, cfg, epoch, session_id, start_epoch, resume_from, rollout_records,
    raw_advantages, base_reward_stats, reward_components, normalization_records,
    preclip_norms, optimizer_steps_total, pbrs_scale,
) -> dict[str, Any]:
    """Logging-only view of tensors already consumed by the completed PPO update."""
    distributions = {
        "base_reward_per_active_step": {"population": "masked_active_transitions", **base_reward_stats.summary()},
        "shaped_reward_per_active_step": {"population": "masked_active_transitions", **_summarize_tensor_parts((batch.rewards, batch.valid) for batch, _, _ in rollout_records)},
        "shaped_return_to_go": {"population": "masked_active_transition_return_to_go", **_summarize_tensor_parts((returns, batch.valid) for batch, returns, _ in rollout_records)},
        "advantage_before_normalization": {"population": "masked_active_transitions", **summarize_values(raw_advantages)},
        "advantage_after_normalization": {"population": "masked_active_transitions", **_summarize_tensor_parts((advantages, batch.valid) for batch, _, advantages in rollout_records)},
    }
    initial_returns, trajectory_totals = [], []
    for batch, returns, _ in rollout_records:
        active_trajectory, first = batch.valid.max(dim=0)
        initial_returns.append((returns.gather(0, first.unsqueeze(0)).squeeze(0), active_trajectory))
        dtype = torch.float32 if batch.rewards.device.type == "mps" else torch.float64
        total = torch.zeros_like(batch.rewards[0], dtype=dtype)
        step_chunk = max(1, 262144 // max(batch.rewards[0].numel(), 1))
        for start in range(0, batch.rewards.size(0), step_chunk):
            active_rewards = torch.where(batch.valid[start:start + step_chunk], batch.rewards[start:start + step_chunk], 0.0)
            total += active_rewards.sum(dim=0, dtype=dtype)
        trajectory_totals.append((total, active_trajectory))
    distributions["shaped_initial_return_per_trajectory"] = {"population": "one_initial_return_per_nonempty_trajectory", **_summarize_tensor_parts(initial_returns)}
    distributions["shaped_total_per_trajectory"] = {"population": "one_undiscounted_masked_reward_sum_per_nonempty_trajectory", **_summarize_tensor_parts(trajectory_totals)}
    active_count = int(raw_advantages.numel())
    trajectory_count = distributions["shaped_initial_return_per_trajectory"]["count"]
    normalized = [row["normalize_reward"] for row in normalization_records]
    scale_mode = cfg.get("normalization", {}).get("reward_distance_scale_mode", cfg.get("env", {}).get("reward_distance_scale_mode", "single_customer_repair_median"))
    normalization = {
        "objective": resolve_objective(cfg.get("objective")).to_dict(),
        "training_pool_metadata": dict(cfg.get("normalization", {})),
        "reward_distance_scale_mode": scale_mode,
        "reward_distance_scale_source": cfg.get("normalization", {}).get("reward_distance_scale_source", "explicit_env_scale" if cfg.get("env", {}).get("reward_distance_scale_km") is not None else "per_instance"),
        "normalize_reward": normalized[0] if normalized and len(set(normalized)) == 1 else "mixed_or_unavailable",
        "distance_scale_km": summarize_values([row["distance_scale_km"] for row in normalization_records]),
        "objective_scale": summarize_values([row["objective_scale"] for row in normalization_records]),
        "applied_reward_divisor": summarize_values([row["objective_scale"] if row["normalize_reward"] else 1.0 for row in normalization_records]),
        "scale_population": "one_scale_per_environment_reset_in_this_epoch",
        "gamma": training_gamma(cfg), "pbrs_annealing_scale": pbrs_scale,
        "advantage_normalization_applied": active_count > 1,
        "advantage_normalization_epsilon": 1e-8,
    }
    names = sorted(key.removesuffix("_sum") for key in reward_components if key.endswith("_sum") and not key.endswith(("_abs_sum", "_discounted_sum", "_customer_action_sum", "_noncustomer_action_sum")))
    components = {
        name: {
            "active_step_sum": reward_components[f"{name}_sum"],
            "active_step_mean": reward_components[f"{name}_sum"] / max(active_count, 1),
            "active_step_abs_mean": reward_components.get(f"{name}_abs_sum", 0.0) / max(active_count, 1),
            "mean_sum_per_trajectory": reward_components[f"{name}_sum"] / max(trajectory_count, 1),
            "mean_discounted_sum_per_trajectory": reward_components.get(f"{name}_discounted_sum") / max(trajectory_count, 1) if f"{name}_discounted_sum" in reward_components else None,
        } for name in names
    }
    norm_values = torch.stack(preclip_norms) if preclip_norms else torch.empty(0)
    max_grad_norm = float(cfg.get("training", {}).get("max_grad_norm", 1.0))
    clip_flags = norm_values > max_grad_norm
    clip_stats = summarize_values(clip_flags, torch.isfinite(norm_values))
    return _json_safe_diagnostics({
        "schema": "drl_reward_diagnostics_v1", "method": "TERRAN", "epoch": int(epoch), "logical_epoch": int(epoch),
        "session_id": session_id, "start_epoch": int(start_epoch), "resume_from": str(resume_from) if resume_from else None,
        "optimizer_steps_total": int(optimizer_steps_total), "normalization": normalization,
        "mask": {"source": "RolloutBatch.valid (active before action; terminal transition included, finished padding excluded)", "active_transition_count": active_count, "trajectory_count": trajectory_count},
        "distributions": distributions, "components": components,
        "component_relationships": {
            "objective": ["electricity_cost", "vehicle_cost"],
            "base": ["objective", "base_non_objective"],
            "pbrs_total": [
                "pbrs_customer",
                "pbrs_repair_distance",
                "pbrs_feasible_ratio",
            ],
            "terminal_task_total": [
                "terminal_success_bonus",
                "terminal_failure_base",
                "terminal_unserved",
            ],
            "shaping_total": [
                "pbrs_total",
                "terminal_heuristic",
            ],
            "shaped": [
                "base",
                "shaping_total",
                "terminal_task_total",
            ],
            "derived_totals_not_additional_components": [
                "pbrs_total",
                "terminal_task_total",
                "shaping_total",
                "shaped",
            ],
        },
        "component_units": "actual training reward components after configured environment normalization; distance is the separate historical km-scale diagnostic, shaping_total contains only auxiliary PBRS plus the disabled-by-contract legacy heuristic, and terminal_task_total is a separate non-annealed task outcome reward (success bonus plus signed failure terms); these are not raw USD costs",
        "gradients": {"preclip_global_norm": summarize_values(norm_values), "max_grad_norm": max_grad_norm, "clip_fraction": clip_stats["mean"], "clip_fraction_finite_optimizer_steps": clip_stats["count"], "clip_fraction_definition": "returned_preclip_norm > max_grad_norm on finite norms", "capture": "return value of the existing single clip_grad_norm_ call after all backward accumulation, before optimizer.step"},
    })


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


def training_gamma(cfg: dict[str, Any]) -> float:
    """Use the same finite-episode discount for returns and potential shaping."""
    gamma = float(cfg.get("training", {}).get("gamma", 1.0))
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("training.gamma must be finite and in [0, 1]")
    return gamma


def terminal_success_bonus(cfg: Mapping[str, Any]) -> float:
    """Resolve the canonical completion reward in normalized-cost units."""

    pbrs = cfg.get("pbrs", {}) or {}
    if not isinstance(pbrs, Mapping):
        raise ValueError("TERRAN pbrs configuration must be a mapping")
    value = float(pbrs.get("terminal_success_bonus", 0.0))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "TERRAN pbrs.terminal_success_bonus must be finite and non-negative"
        )
    return value


def _selected_reward_contract_terms(
    cfg: dict[str, Any], contract: RewardContract, *, source: str,
):
    """Resolve and audit the immutable per-scale terms stored in a config.

    A contract digest identifies the complete multi-scale JSON, not the scale
    used by one training run.  Checkpoints therefore have to preserve both the
    selected scale and every derived runtime value that affected rewards.
    """

    data = cfg.get("data", {})
    raw_scale = data.get("stage2_scale")
    if raw_scale in (None, ""):
        try:
            raw_scale = f"Cus{int(data['num_customers'])}"
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"TERRAN {source} reward contract is missing its selected scale"
            ) from error
    try:
        terms = contract.for_scale(raw_scale, resolve_objective(cfg.get("objective")))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"TERRAN {source} reward contract selection is invalid: {error}"
        ) from error

    completion_bonus = terminal_success_bonus(cfg)
    expected = {
        ("training", "reward_contract_id"): terms.contract_id,
        ("normalization", "reward_contract_id"): terms.contract_id,
        ("normalization", "reward_contract_sha256"): terms.digest,
        ("normalization", "reward_contract_scale"): terms.scale_label,
        ("normalization", "reward_objective_scale"): terms.objective_scale,
        ("normalization", "failure_base"): terms.failure_base,
        ("normalization", "unserved_coefficient"): terms.unserved_coefficient,
        ("normalization", "terran_terminal_success_bonus"): completion_bonus,
        ("normalization", "terran_terminal_success_bonus_unit"):
            TERMINAL_TASK_REWARD_UNIT,
        ("normalization", "terran_terminal_success_bonus_equivalent_usd"):
            completion_bonus * terms.objective_scale,
        ("env", "normalize_reward"): True,
        ("env", "reward_objective_scale"): terms.objective_scale,
        ("env", "invalid_action_penalty"): 0.0,
        ("env", "success_bonus"): 0.0,
        ("pbrs", "use_terminal_heuristic"): False,
        ("pbrs", "use_terminal_task_penalty"): True,
        ("pbrs", "success_bonus"): 0.0,
        ("pbrs", "terminal_success_bonus"): completion_bonus,
        ("pbrs", "failure_base"): terms.failure_base,
        ("pbrs", "unserved_coefficient"): terms.unserved_coefficient,
    }
    for path, expected_value in expected.items():
        section, field = path
        actual = cfg.get(section, {}).get(field)
        if actual != expected_value:
            dotted = ".".join(path)
            raise ValueError(
                f"TERRAN {source} derived reward contract field {dotted} "
                f"does not match the frozen {terms.scale_label} terms"
            )
    return terms


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _validated_training_stream_sha(
    cfg: Mapping[str, Any], *, source: str,
) -> str | None:
    protocol = cfg.get("protocol", {}) or {}
    if not isinstance(protocol, Mapping):
        raise ValueError(f"TERRAN {source} protocol provenance is invalid")
    snapshot = protocol.get("training_stream_contract_snapshot")
    digest = protocol.get("training_stream_contract_sha256")
    if snapshot is None and digest is None:
        return None
    if not isinstance(snapshot, Mapping) or not isinstance(digest, str):
        raise ValueError(
            f"TERRAN {source} training-stream contract is incomplete"
        )
    if snapshot.get("schema") != STREAM_CONTRACT_SCHEMA:
        raise ValueError(
            f"TERRAN {source} training-stream contract schema is invalid"
        )
    computed = training_stream_contract_digest(snapshot)
    if snapshot.get("sha256") != computed or digest != computed:
        raise ValueError(
            f"TERRAN {source} training-stream contract digest is inconsistent"
        )
    return digest


def resolved_terran_scientific_fields(
    cfg: Mapping[str, Any], *, seed: int,
) -> dict[str, Any]:
    """Resolve the exact TERRAN controls that may change learned parameters.

    This is the method-specific portion of the common resolved-training
    signature.  It intentionally excludes dynamic resume counters and output
    paths while freezing PPO batching, validation selection, and the formal
    protocol boundary with the same defaults/transforms used by the trainer.
    """

    training = cfg.get("training", {}) or {}
    evaluation = cfg.get("evaluation", {}) or {}
    protocol = cfg.get("protocol", {}) or {}
    if not all(isinstance(section, Mapping) for section in (training, evaluation, protocol)):
        raise ValueError("TERRAN scientific signature requires mapping config sections")

    epochs = int(training.get("epochs", 1000))
    num_envs = int(training.get("num_envs_per_gpu", 128))
    n_traj = int(training.get("n_traj", 100))
    rollout_steps = int(training.get("rollout_steps", 64))
    logical_microbatches = max(
        1, int(training.get("logical_microbatches_per_epoch", 1))
    )
    effective_batch = num_envs * logical_microbatches
    ppo_step_chunk = int(training.get("ppo_step_chunk_size", 0) or 0)
    if ppo_step_chunk <= 0:
        ppo_step_chunk = 0

    for field, expected in (
        ("physical_batch_size", num_envs),
        ("effective_batch_size", effective_batch),
        ("logical_environments_per_epoch", effective_batch),
        ("training_rollout_steps", rollout_steps),
    ):
        configured = protocol.get(field)
        if configured is not None and int(configured) != expected:
            raise ValueError(
                f"TERRAN protocol {field}={configured!r} disagrees with "
                f"resolved trainer value {expected!r}"
            )

    eval_seed = int(evaluation.get("eval_seed", int(seed) + 910_000_000))
    raw_decode = str(evaluation.get("eval_decode_mode", "sample")).lower()
    if raw_decode in {"sample", "sampling"}:
        eval_decode = "sample"
    elif raw_decode == "greedy":
        eval_decode = "greedy"
    else:
        raise ValueError(f"unsupported TERRAN evaluation decode mode: {raw_decode}")
    eval_n_traj = int(evaluation.get("eval_n_traj", 100))
    eval_interval = int(evaluation.get("eval_interval", 0) or 0)
    eval_limit = _optional_int(evaluation.get("eval_limit"))
    eval_max_steps = _optional_int(evaluation.get("eval_max_steps"))
    protocol_eval_max_steps = _optional_int(
        protocol.get("validation_rollout_steps")
    )
    if (
        protocol.get("protocol_id") == "drl_rq_protocol_frozen_v1"
        and protocol_eval_max_steps is None
    ):
        raise ValueError(
            "formal TERRAN protocol requires validation rollout steps"
        )
    if (
        protocol_eval_max_steps is not None
        and protocol_eval_max_steps != eval_max_steps
    ):
        raise ValueError(
            "TERRAN protocol validation rollout steps disagree with evaluation"
        )
    eval_num_batches = _optional_int(evaluation.get("eval_num_batches"))
    eval_batch_size = max(1, int(evaluation.get("eval_batch_size", 1)))

    scheduled_epochs = [
        int(value) for value in training.get("validation_epochs", [])
    ]
    protocol_schedule = protocol.get("scheduled_validation_epochs")
    if protocol_schedule is not None and [
        int(value) for value in protocol_schedule
    ] != scheduled_epochs:
        raise ValueError(
            "TERRAN protocol validation schedule disagrees with trainer schedule"
        )
    early_stop_patience = int(
        training.get("early_stop_patience_validations", 0) or 0
    )
    early_stop_start = int(training.get("early_stop_start_epoch", 0) or 0)
    for field, expected in (
        ("early_stop_patience_validations", early_stop_patience),
        ("early_stop_start_epoch", early_stop_start),
    ):
        configured = protocol.get(field)
        if configured is not None and int(configured) != expected:
            raise ValueError(
                f"TERRAN protocol {field} disagrees with trainer configuration"
            )

    protocol_seed = protocol.get("validation_seed")
    if protocol_seed is not None and int(protocol_seed) != eval_seed:
        raise ValueError("TERRAN protocol validation seed disagrees with evaluation")
    protocol_candidates = protocol.get("validation_candidates")
    if protocol_candidates is not None and int(protocol_candidates) != eval_n_traj:
        raise ValueError(
            "TERRAN protocol validation candidates disagree with evaluation"
        )
    protocol_decode = protocol.get("validation_decode_type")
    expected_protocol_decode = "sampling" if eval_decode == "sample" else "greedy"
    if protocol_decode is not None and str(protocol_decode) != expected_protocol_decode:
        raise ValueError(
            "TERRAN protocol validation decode type disagrees with evaluation"
        )

    minimum_epochs = int(training.get("minimum_training_epochs", epochs) or epochs)
    validation_every = _optional_int(protocol.get("validation_every_epochs"))
    post_minimum_every = _optional_int(
        training.get(
            "post_minimum_validation_every_epochs",
            protocol.get("post_minimum_validation_every_epochs"),
        )
    )
    validation_checkpoints = int(
        protocol.get("validation_checkpoints", len(scheduled_epochs) or 1)
    )
    stream_sha = _validated_training_stream_sha(
        cfg, source="scientific signature"
    )
    completion_bonus = terminal_success_bonus(cfg)
    protocol_completion_bonus = protocol.get("terran_terminal_success_bonus")
    if (
        protocol_completion_bonus is not None
        and float(protocol_completion_bonus) != completion_bonus
    ):
        raise ValueError(
            "TERRAN protocol terminal success bonus disagrees with task reward"
        )
    result = {
        "method": "TERRAN",
        "task_reward": {
            "terminal_success_bonus": completion_bonus,
            "unit": TERMINAL_TASK_REWARD_UNIT,
        },
        "training": {
            "epochs": epochs,
            "num_envs_per_gpu": num_envs,
            "n_traj": n_traj,
            "rollout_steps": rollout_steps,
            "logical_microbatches_per_epoch": logical_microbatches,
            "ppo_update_epochs": int(training.get("ppo_update_epochs", 4)),
            "num_minibatches": max(1, int(training.get("num_minibatches", 1))),
            "gradient_accumulation_steps": max(
                1, int(training.get("gradient_accumulation_steps", 1))
            ),
            "ppo_step_chunk_size": ppo_step_chunk,
            "clip_coef": float(training.get("clip_coef", 0.2)),
            "vf_coef": float(training.get("vf_coef", 0.5)),
            "ent_coef": float(training.get("ent_coef", 0.01)),
            "learning_rate": float(training.get("learning_rate", 1e-4)),
            "max_grad_norm": float(training.get("max_grad_norm", 1.0)),
            "gamma": training_gamma(dict(cfg)),
        },
        "evaluation": {
            "seed": eval_seed,
            "decode_mode": eval_decode,
            "n_traj": eval_n_traj,
            "limit": eval_limit,
            "max_steps": eval_max_steps,
            "batch_size": eval_batch_size,
            "num_batches": eval_num_batches,
            "interval": eval_interval,
        },
        "protocol": {
            "protocol_id": str(protocol.get("protocol_id", "")),
            "physical_batch_size": num_envs,
            "effective_batch_size": effective_batch,
            "training_rollout_steps": rollout_steps,
            "validation_rollout_steps": eval_max_steps,
            "training_stream_contract_sha256": stream_sha,
            "minimum_training_epochs": minimum_epochs,
            "validation_every_epochs": validation_every,
            "post_minimum_validation_every_epochs": post_minimum_every,
            "scheduled_validation_epochs": scheduled_epochs,
            "validation_checkpoints": validation_checkpoints,
            "early_stop_patience_validations": early_stop_patience,
            "early_stop_start_epoch": early_stop_start,
        },
    }
    warm_start = protocol.get("warm_start")
    has_warm_start = bool(protocol.get("warm_start_checkpoint")) or warm_start is not None
    if has_warm_start:
        mode = warm_start_epoch_mode(cfg)
        result["protocol"]["warm_start_epoch_mode"] = mode
    if warm_start is not None:
        if not isinstance(warm_start, Mapping):
            raise ValueError("TERRAN warm-start provenance must be a mapping")
        result["protocol"]["warm_start"] = json.loads(
            json.dumps(dict(warm_start), sort_keys=True, allow_nan=False)
        )
        result["protocol"]["planned_training_epochs"] = int(
            protocol.get(
                "planned_training_epochs",
                (
                    epochs - int(warm_start.get("source_epoch", 0))
                    if mode == "continue_global"
                    else epochs
                ),
            )
        )
    return result


def _resolved_terran_training_signature(
    cfg: Mapping[str, Any], *, seed: int,
) -> dict[str, Any]:
    method_fields = resolved_terran_scientific_fields(cfg, seed=seed)
    data = cfg.get("data", {}) or {}
    training = method_fields["training"]
    evaluation = method_fields["evaluation"]
    protocol = method_fields["protocol"]
    raw_scale = data.get("stage2_scale")
    if raw_scale in (None, ""):
        raw_scale = f"Cus{int(data.get('num_customers', 15))}"
    stream_path = (cfg.get("protocol", {}) or {}).get("training_stream_path")
    if stream_path is None:
        stream_path = data.get("stage2_training_stream_path")
    effective = int(protocol["effective_batch_size"])
    planned_training_epochs = int(
        (cfg.get("protocol", {}) or {}).get(
            "planned_training_epochs", training["epochs"]
        )
    )
    exposure_budget = (
        planned_training_epochs
        * effective
        * int(str(raw_scale).removeprefix("Cus"))
        if stream_path is not None
        else None
    )
    eval_decode_type = (
        "sampling" if evaluation["decode_mode"] == "sample" else "greedy"
    )
    proxy = SimpleNamespace(
        protocol_id=protocol["protocol_id"],
        seed=int(seed),
        stage2_scale=str(raw_scale),
        training_representation=str(
            data.get("stage2_training_representation", "G")
        ),
        training_epochs=int(training["epochs"]),
        minimum_training_epochs=protocol["minimum_training_epochs"],
        training_rollout_steps=int(training["rollout_steps"]),
        validation_rollout_steps=evaluation["max_steps"],
        physical_batch_size=int(protocol["physical_batch_size"]),
        effective_batch_size=effective,
        n_traj=int(training["n_traj"]),
        customer_exposure_budget=exposure_budget,
        training_stream_contract_sha256=protocol[
            "training_stream_contract_sha256"
        ],
        validation_limit=evaluation["limit"],
        validation_decode_type=eval_decode_type,
        validation_candidates=int(evaluation["n_traj"]),
        validation_seed=int(evaluation["seed"]),
        validation_every_epochs=protocol["validation_every_epochs"],
        post_minimum_validation_every_epochs=protocol[
            "post_minimum_validation_every_epochs"
        ],
        validation_checkpoints=int(protocol["validation_checkpoints"]),
        early_stop_patience_validations=protocol[
            "early_stop_patience_validations"
        ],
        early_stop_start_epoch=protocol["early_stop_start_epoch"],
        final_validation_limit=(cfg.get("protocol", {}) or {}).get(
            "final_validation_limit"
        ),
        soft_stage_end_epoch=None,
        optimizer=(cfg.get("training", {}) or {}).get("optimizer"),
        weight_decay=(cfg.get("training", {}) or {}).get("weight_decay"),
        reward_contract_sha256=(cfg.get("normalization", {}) or {}).get(
            "reward_contract_sha256"
        ),
        method_auxiliary_sha256=None,
        training_stream_path=stream_path,
        validation_dataset_path=(cfg.get("evaluation", {}) or {}).get(
            "eval_path"
        ),
        euclidean_manifest=(cfg.get("evaluation", {}) or {}).get(
            "eval_euclidean_manifest",
            data.get("stage2_euclidean_manifest"),
        ),
        resolved_training_method_fields=method_fields,
    )
    return resolved_training_signature_from_args(proxy)


def _freeze_resolved_terran_training_signature(
    cfg: dict[str, Any], *, seed: int,
) -> dict[str, Any]:
    signature = _resolved_terran_training_signature(cfg, seed=seed)
    protocol = cfg.setdefault("protocol", {})
    existing = protocol.get("resolved_training_signature")
    if existing is not None and existing != signature:
        raise ValueError(
            "TERRAN current resolved training signature is inconsistent"
        )
    protocol["resolved_training_method_fields"] = signature["method_specific"]
    protocol["resolved_training_signature"] = signature
    protocol["resolved_training_signature_sha256"] = signature["sha256"]
    return signature


_LEGACY_CONTINUE_GLOBAL_WARM_START_KEYS = frozenset(
    {
        "schema",
        "source_checkpoint_path",
        "source_checkpoint_sha256",
        "source_epoch",
        "source_seed",
        "model_state_dict_loaded",
        "optimizer_state_dict_loaded",
        "optimizer_reset",
        "optimizer_name",
    }
)
_CONTINUE_GLOBAL_WARM_START_COMPATIBILITY_KEYS = frozenset(
    {
        "epoch_mode",
        "method",
        "checkpoint",
        "epoch_reset",
        "data_stream_cursor_reset",
        "validation_state_reset",
        "early_stop_state_reset",
        "source_baseline_evaluated",
    }
)


def _legacy_continue_global_warm_start_provenance(
    cfg: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Recognize the exact pre-mode TERRAN warm-start provenance shape.

    The first A6000 continuation checkpoints used the current schema name but
    predated ``epoch_mode``.  That schema had only one meaning: continue the
    source checkpoint's global epoch.  Keep this detector deliberately narrow
    so no other signature drift is hidden during resume validation.
    """

    protocol = cfg.get("protocol", {}) or {}
    provenance = protocol.get("warm_start") if isinstance(protocol, Mapping) else None
    if not isinstance(provenance, Mapping):
        return None
    if set(provenance) != _LEGACY_CONTINUE_GLOBAL_WARM_START_KEYS:
        return None
    try:
        source_epoch = int(provenance["source_epoch"])
        source_sha = str(provenance["source_checkpoint_sha256"])
        source_path = str(provenance["source_checkpoint_path"])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        provenance.get("schema") != "terran_weights_only_warm_start_v1"
        or source_epoch < 0
        or not source_path
        or len(source_sha) != 64
        or provenance.get("model_state_dict_loaded") is not True
        or provenance.get("optimizer_state_dict_loaded") is not False
        or provenance.get("optimizer_reset") is not True
        or str(provenance.get("optimizer_name", "")).lower() != "adamw"
    ):
        return None
    return dict(provenance)


def _project_signature_to_legacy_continue_global(
    signature: Mapping[str, Any],
    *,
    legacy_provenance: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Remove only fields added when epoch-mode provenance was introduced."""

    projected = deepcopy(dict(signature))
    method = projected.get("method_specific")
    method_protocol = method.get("protocol") if isinstance(method, Mapping) else None
    signature_provenance = (
        method_protocol.get("warm_start")
        if isinstance(method_protocol, Mapping)
        else None
    )
    if not isinstance(method_protocol, dict) or not isinstance(
        signature_provenance, Mapping
    ):
        return None
    if any(
        signature_provenance.get(key) != legacy_provenance.get(key)
        for key in _LEGACY_CONTINUE_GLOBAL_WARM_START_KEYS
    ):
        return None
    extra_keys = set(signature_provenance).difference(
        _LEGACY_CONTINUE_GLOBAL_WARM_START_KEYS
    )
    if not extra_keys.issubset(
        _CONTINUE_GLOBAL_WARM_START_COMPATIBILITY_KEYS
    ):
        return None
    expected_extras = {
        "epoch_mode": "continue_global",
        "method": "TERRAN",
        "checkpoint": legacy_provenance["source_checkpoint_path"],
        "epoch_reset": False,
        "data_stream_cursor_reset": True,
        "validation_state_reset": True,
        "early_stop_state_reset": True,
        "source_baseline_evaluated": True,
    }
    if any(
        signature_provenance.get(key) != expected_extras[key]
        for key in extra_keys
    ):
        return None
    declared_mode = method_protocol.get("warm_start_epoch_mode")
    if declared_mode not in (None, "continue_global"):
        return None
    method_protocol.pop("warm_start_epoch_mode", None)
    method_protocol["warm_start"] = {
        key: signature_provenance[key]
        for key in _LEGACY_CONTINUE_GLOBAL_WARM_START_KEYS
    }
    projected["sha256"] = resolved_training_signature_digest(projected)
    return projected


def _validate_resume_training_signature(
    cfg: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    current_seed: int | None,
) -> None:
    saved_cfg = payload.get("config", {}) or {}
    current_protocol = cfg.get("protocol", {}) or {}
    saved_protocol = saved_cfg.get("protocol", {}) or {}
    current = current_protocol.get("resolved_training_signature")
    saved = saved_protocol.get("resolved_training_signature")
    if current is None and saved is None:
        return
    if not isinstance(current, Mapping) or not isinstance(saved, Mapping):
        raise ValueError(
            "TERRAN resume resolved training signature is missing; start a fresh run"
        )
    resolved_current_seed = (
        int(current_seed)
        if current_seed is not None
        else int(current.get("seed"))
    )
    saved_seed = payload.get("seed", saved.get("seed"))
    if saved_seed is None:
        raise ValueError(
            "TERRAN resume checkpoint is missing its training seed"
        )
    expected_current = _resolved_terran_training_signature(
        cfg, seed=resolved_current_seed
    )
    expected_saved = _resolved_terran_training_signature(
        saved_cfg, seed=int(saved_seed)
    )
    legacy_warm_start = _legacy_continue_global_warm_start_provenance(saved_cfg)
    legacy_expected_saved = (
        _project_signature_to_legacy_continue_global(
            expected_saved,
            legacy_provenance=legacy_warm_start,
        )
        if legacy_warm_start is not None
        else None
    )
    if (
        dict(current) != expected_current
        or current.get("sha256")
        != resolved_training_signature_digest(dict(current))
        or current_protocol.get("resolved_training_method_fields")
        != current.get("method_specific")
    ):
        raise ValueError(
            "TERRAN current resolved training signature is inconsistent"
        )
    saved_matches_expected = (
        dict(saved) == expected_saved
        or dict(saved) == legacy_expected_saved
    )
    if (
        not saved_matches_expected
        or saved.get("sha256") != resolved_training_signature_digest(dict(saved))
        or saved_protocol.get("resolved_training_method_fields")
        != saved.get("method_specific")
    ):
        raise ValueError(
            "TERRAN resume checkpoint resolved training signature is inconsistent"
        )
    comparison_current = dict(current)
    if legacy_warm_start is not None:
        projected_current = _project_signature_to_legacy_continue_global(
            current,
            legacy_provenance=legacy_warm_start,
        )
        if projected_current is not None:
            comparison_current = projected_current
    if (
        comparison_current != dict(saved)
        and not _allows_legacy_training_prefix_extension(
            saved=dict(saved), current=comparison_current
        )
    ):
        raise ValueError(
            "TERRAN resume resolved training signature mismatch; start a fresh run"
        )


def _allows_legacy_training_prefix_extension(
    *, saved: dict[str, Any], current: dict[str, Any],
) -> bool:
    """Keep only the historical non-formal epoch-prefix resume behavior.

    The frozen RQ protocol is immutable.  A legacy direct caller may increase
    only its terminal epoch when early stopping is disabled; fields derived
    directly from that epoch are normalized solely for this comparison.
    """

    if (
        saved.get("protocol_id") == "drl_rq_protocol_frozen_v1"
        or current.get("protocol_id") == "drl_rq_protocol_frozen_v1"
        or saved.get("protocol_id") != current.get("protocol_id")
    ):
        return False
    try:
        saved_epoch = int(saved["training_epochs"])
        current_epoch = int(current["training_epochs"])
    except (KeyError, TypeError, ValueError):
        return False
    if current_epoch < saved_epoch:
        return False
    saved_method = saved.get("method_specific")
    current_method = current.get("method_specific")
    if not isinstance(saved_method, dict) or not isinstance(current_method, dict):
        return False
    saved_protocol = saved_method.get("protocol")
    current_protocol = current_method.get("protocol")
    saved_training = saved_method.get("training")
    current_training = current_method.get("training")
    if not all(
        isinstance(value, dict)
        for value in (
            saved_protocol,
            current_protocol,
            saved_training,
            current_training,
        )
    ):
        return False
    if (
        int(saved_protocol.get("early_stop_patience_validations") or 0) != 0
        or int(current_protocol.get("early_stop_patience_validations") or 0) != 0
        or saved.get("minimum_training_epochs") != saved_epoch
        or current.get("minimum_training_epochs") != current_epoch
        or saved_protocol.get("minimum_training_epochs") != saved_epoch
        or current_protocol.get("minimum_training_epochs") != current_epoch
        or saved_training.get("epochs") != saved_epoch
        or current_training.get("epochs") != current_epoch
    ):
        return False

    saved_normalized = deepcopy(saved)
    current_normalized = deepcopy(current)
    for signature in (saved_normalized, current_normalized):
        signature.pop("sha256", None)
        signature["training_epochs"] = saved_epoch
        signature["minimum_training_epochs"] = saved_epoch
        method = signature["method_specific"]
        method["training"]["epochs"] = saved_epoch
        method["protocol"]["minimum_training_epochs"] = saved_epoch
    return saved_normalized == current_normalized


def _validate_task_reward_compatibility(
    cfg: dict[str, Any],
    payload: Mapping[str, Any],
    *,
    operation: str,
) -> None:
    """Validate the semantics that make pretrained policy weights meaningful."""

    saved_cfg = payload.get("config", {})
    if not isinstance(saved_cfg, Mapping):
        raise ValueError(f"TERRAN {operation} checkpoint is missing its config")
    saved_training = saved_cfg.get("training", {})
    if not isinstance(saved_training, Mapping):
        raise ValueError(
            f"TERRAN {operation} checkpoint has invalid training configuration"
        )
    if "gamma" not in saved_training:
        raise ValueError(
            f"TERRAN {operation} checkpoint is missing training.gamma; "
            "start a fresh run"
        )
    if training_gamma(dict(saved_cfg)) != training_gamma(cfg):
        raise ValueError(
            f"TERRAN {operation} gamma mismatch; start a fresh run in a new "
            "output directory"
        )
    current_contract = cfg.get("training", {}).get("reward_contract_id")
    if saved_training.get("reward_contract_id") != current_contract:
        raise ValueError(
            f"TERRAN {operation} reward contract mismatch; start a fresh run"
        )
    current_snapshot = cfg.get("reward_contract")
    saved_snapshot = saved_cfg.get("reward_contract")
    if (current_snapshot is None) != (saved_snapshot is None):
        raise ValueError(
            f"TERRAN {operation} reward contract snapshot mismatch; "
            "start a fresh run"
        )
    if current_snapshot is not None:
        try:
            current_frozen = RewardContract.from_payload(current_snapshot)
            saved_frozen = RewardContract.from_payload(saved_snapshot)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"TERRAN {operation} reward contract snapshot is invalid; "
                "start a fresh run"
            ) from error
        if current_frozen.digest != saved_frozen.digest:
            raise ValueError(
                f"TERRAN {operation} reward contract digest mismatch; "
                "start a fresh run"
            )
        current_terms = _selected_reward_contract_terms(
            cfg, current_frozen, source="current configuration",
        )
        saved_terms = _selected_reward_contract_terms(
            dict(saved_cfg), saved_frozen, source=f"{operation} checkpoint",
        )
        if current_terms.scale_label != saved_terms.scale_label:
            raise ValueError(
                f"TERRAN {operation} reward contract scale mismatch; "
                "start a fresh run"
            )
        if current_terms.to_dict() != saved_terms.to_dict():
            raise ValueError(
                f"TERRAN {operation} derived reward contract mismatch; "
                "start a fresh run"
            )
    current_pbrs = cfg.get("pbrs_reward_semantics")
    saved_pbrs = saved_cfg.get("pbrs_reward_semantics")
    if (current_pbrs is None) != (saved_pbrs is None):
        raise ValueError(
            f"TERRAN {operation} PBRS shaping snapshot mismatch; start a fresh run"
        )
    if current_pbrs is not None:
        current_resolved = _resolved_pbrs_reward_semantics(cfg)
        saved_resolved = _resolved_pbrs_reward_semantics(dict(saved_cfg))
        if current_pbrs != current_resolved or saved_pbrs != saved_resolved:
            raise ValueError(
                f"TERRAN {operation} PBRS shaping snapshot is inconsistent "
                "with its config"
            )
        if current_pbrs != saved_pbrs:
            raise ValueError(
                f"TERRAN {operation} PBRS shaping semantics mismatch; "
                "start a fresh run"
            )
    current_objective = resolve_objective(cfg.get("objective"))
    try:
        objective_from_checkpoint(dict(payload), override=current_objective)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"TERRAN {operation} objective configuration mismatch; "
            f"start a fresh run: {error}"
        ) from error


def validate_resume_reward_contract(cfg: dict[str, Any], payload: dict[str, Any]) -> None:
    """Do not silently continue a discounted/older reward run as a new protocol."""
    _validate_task_reward_compatibility(cfg, payload, operation="resume")
    saved_training = payload.get("config", {}).get("training", {})
    current_stream_sha = _validated_training_stream_sha(
        cfg, source="current configuration"
    )
    saved_stream_sha = _validated_training_stream_sha(
        payload.get("config", {}), source="resume checkpoint"
    )
    if current_stream_sha != saved_stream_sha:
        raise ValueError(
            "TERRAN resume training-stream contract mismatch; start a fresh run"
        )
    _validate_resume_training_signature(
        cfg,
        payload,
        current_seed=None,
    )
    current_optimizer = cfg.get("training", {}).get("optimizer")
    current_weight_decay = cfg.get("training", {}).get("weight_decay")
    if current_optimizer is not None or current_weight_decay is not None:
        if current_optimizer is None or current_weight_decay is None:
            raise ValueError("TERRAN current optimizer contract is incomplete")
        if str(saved_training.get("optimizer", "")).lower() != str(
            current_optimizer
        ).lower():
            raise ValueError("TERRAN resume optimizer mismatch; start a fresh run")
        try:
            saved_weight_decay = float(saved_training["weight_decay"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "TERRAN resume checkpoint is missing optimizer weight decay; "
                "start a fresh run"
            ) from error
        if saved_weight_decay != float(current_weight_decay):
            raise ValueError(
                "TERRAN resume optimizer weight decay mismatch; start a fresh run"
            )
        param_groups = payload.get("optimizer_state_dict", {}).get(
            "param_groups", []
        )
        if not param_groups or any(
            float(group.get("weight_decay", float("nan")))
            != float(current_weight_decay)
            for group in param_groups
        ):
            raise ValueError(
                "TERRAN resume optimizer state weight decay mismatch; "
                "start a fresh run"
            )


def _warm_start_model_signature(cfg: Mapping[str, Any]) -> dict[str, Any]:
    model = cfg.get("model", {}) or {}
    if not isinstance(model, Mapping):
        raise ValueError("TERRAN warm-start model configuration must be a mapping")
    return {
        "embedding_dim": int(model.get("embedding_dim", 256)),
        "tanh_clipping": float(model.get("tanh_clipping", 15.0)),
        "n_encode_layers": int(model.get("n_encode_layers", 3)),
        "use_graph_token": bool(model.get("use_graph_token", False)),
        "use_dynamic_embedding": bool(model.get("use_dynamic_embedding", False)),
    }


def _warm_start_scale_signature(cfg: Mapping[str, Any]) -> dict[str, Any]:
    data = cfg.get("data", {}) or {}
    if not isinstance(data, Mapping):
        raise ValueError("TERRAN warm-start data configuration must be a mapping")
    raw_scale = data.get("stage2_scale")
    num_customers = int(
        data.get(
            "num_customers",
            str(raw_scale).removeprefix("Cus") if raw_scale not in (None, "") else 15,
        )
    )
    scale = str(raw_scale) if raw_scale not in (None, "") else f"Cus{num_customers}"
    return {
        "scale": scale,
        "num_customers": num_customers,
        "num_charging_stations": int(data.get("num_charging_stations", 3)),
        "representation": str(data.get("stage2_training_representation", "G")),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_warm_start_checkpoint(
    cfg: dict[str, Any],
    payload: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
) -> int:
    """Validate a weights-only initialization and return its global epoch."""

    protocol = cfg.get("protocol", {}) or {}
    if warm_start_epoch_mode(cfg) != "continue_global":
        raise ValueError(
            "global-epoch warm-start validation requires continue_global mode"
        )
    provenance = protocol.get("warm_start") if isinstance(protocol, Mapping) else None
    if not isinstance(provenance, Mapping):
        raise ValueError("TERRAN warm start requires frozen protocol provenance")
    if provenance.get("schema") != "terran_weights_only_warm_start_v1":
        raise ValueError("TERRAN warm-start provenance schema is invalid")
    source = Path(checkpoint_path).expanduser().resolve(strict=True)
    if str(source) != provenance.get("source_checkpoint_path"):
        raise ValueError("TERRAN warm-start source path disagrees with provenance")
    actual_digest = _sha256_file(source)
    if actual_digest != provenance.get("source_checkpoint_sha256"):
        raise ValueError("TERRAN warm-start checkpoint SHA256 mismatch")
    try:
        source_epoch = int(payload["epoch"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("TERRAN warm-start checkpoint has no valid epoch") from error
    if source_epoch < 0 or source_epoch != int(provenance.get("source_epoch", -1)):
        raise ValueError("TERRAN warm-start source epoch disagrees with provenance")
    source_seed = payload.get("seed")
    expected_source_seed = int(source_seed) if source_seed is not None else None
    if provenance.get("source_seed") != expected_source_seed:
        raise ValueError("TERRAN warm-start source seed disagrees with provenance")
    if int(cfg.get("training", {}).get("epochs", 0)) <= source_epoch:
        raise ValueError(
            "TERRAN warm-start target epoch must be greater than source epoch"
        )
    expected_planned_epochs = int(cfg["training"]["epochs"]) - source_epoch
    if int(protocol.get("planned_training_epochs", -1)) != expected_planned_epochs:
        raise ValueError("TERRAN warm-start planned epoch budget is inconsistent")
    if int(protocol.get("warm_start_epoch_offset", source_epoch)) != source_epoch:
        raise ValueError("TERRAN warm-start global epoch offset is inconsistent")
    if (
        provenance.get("model_state_dict_loaded") is not True
        or provenance.get("optimizer_state_dict_loaded") is not False
        or provenance.get("optimizer_reset") is not True
        or str(provenance.get("optimizer_name", "")).lower() != "adamw"
    ):
        raise ValueError("TERRAN warm-start optimizer-reset provenance is invalid")
    if not isinstance(payload.get("model_state_dict"), Mapping):
        raise ValueError("TERRAN warm-start checkpoint is missing model_state_dict")

    _validate_task_reward_compatibility(cfg, payload, operation="warm-start")
    saved_cfg = payload.get("config", {})
    if _warm_start_model_signature(cfg) != _warm_start_model_signature(saved_cfg):
        raise ValueError("TERRAN warm-start model configuration mismatch")
    if _warm_start_scale_signature(cfg) != _warm_start_scale_signature(saved_cfg):
        raise ValueError("TERRAN warm-start scale configuration mismatch")
    return source_epoch


def warm_start_epoch_mode(cfg: Mapping[str, Any]) -> str:
    """Return the explicit weights-only warm-start epoch policy.

    Upstream warm starts restart the logical training schedule at epoch one.
    The calibrated Cus1000 continuation profile instead keeps the source
    checkpoint's global epoch so PBRS, validation, and the fixed budget remain
    on the original schedule.
    """

    protocol = cfg.get("protocol", {}) or {}
    if not isinstance(protocol, Mapping):
        raise ValueError("TERRAN protocol configuration must be a mapping")
    provenance = protocol.get("warm_start")
    nested_mode = (
        provenance.get("epoch_mode")
        if isinstance(provenance, Mapping)
        else None
    )
    configured_mode = protocol.get("warm_start_epoch_mode")
    mode = str(
        configured_mode
        if configured_mode is not None
        else nested_mode
        if nested_mode is not None
        # The original 8584788 provenance schema predated the mode field and
        # unambiguously represented global-epoch continuation.
        else "continue_global"
        if isinstance(provenance, Mapping)
        else "reset"
    ).strip().lower()
    if mode not in {"reset", "continue_global"}:
        raise ValueError(
            "TERRAN warm_start_epoch_mode must be 'reset' or "
            "'continue_global'"
        )
    if nested_mode is not None and str(nested_mode) != mode:
        raise ValueError(
            "TERRAN warm-start epoch mode disagrees with its provenance"
        )
    if protocol.get("warm_start_epoch_offset") is not None:
        source_epoch = (
            int(provenance.get("source_epoch", 0))
            if isinstance(provenance, Mapping)
            else 0
        )
        expected_offset = source_epoch if mode == "continue_global" else 0
        if int(protocol["warm_start_epoch_offset"]) != expected_offset:
            raise ValueError(
                "TERRAN warm-start epoch offset disagrees with its mode"
            )
    return mode


def validate_warm_start_contract(
    cfg: dict[str, Any],
    payload: dict[str, Any],
    *,
    current_seed: int,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate scientific compatibility without inheriting training state."""

    saved_cfg = payload.get("config")
    if not isinstance(saved_cfg, dict):
        raise ValueError("TERRAN warm-start checkpoint is missing its frozen config")
    _validate_task_reward_compatibility(cfg, payload, operation="warm-start")
    if _warm_start_model_signature(saved_cfg) != _warm_start_model_signature(cfg):
        raise ValueError("TERRAN warm-start model architecture mismatch")
    if _warm_start_scale_signature(saved_cfg) != _warm_start_scale_signature(cfg):
        raise ValueError("TERRAN warm-start scale configuration mismatch")
    saved_protocol = saved_cfg.get("protocol", {})
    saved_signature = saved_protocol.get("resolved_training_signature", {})
    current_protocol = cfg.get("protocol", {})
    current_signature = current_protocol.get("resolved_training_signature", {})
    for field in ("scale", "training_representation"):
        if saved_signature.get(field) != current_signature.get(field):
            raise ValueError(f"TERRAN warm-start {field} mismatch")
    if int(payload.get("seed", -1)) != int(current_seed):
        raise ValueError("TERRAN warm-start seed mismatch")
    if not isinstance(payload.get("model_state_dict"), Mapping):
        raise ValueError("TERRAN warm-start checkpoint is missing model weights")
    try:
        source_epoch = int(payload["epoch"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("TERRAN warm-start checkpoint has no valid epoch") from error
    if source_epoch < 0:
        raise ValueError("TERRAN warm-start checkpoint epoch cannot be negative")
    source = Path(
        checkpoint_path
        if checkpoint_path is not None
        else current_protocol["warm_start_checkpoint"]
    ).expanduser().resolve(strict=True)
    declared = current_protocol.get("warm_start")
    if declared is not None:
        if not isinstance(declared, Mapping):
            raise ValueError("TERRAN warm-start provenance must be a mapping")
        if str(declared.get("source_checkpoint_path", "")) != str(source):
            raise ValueError("TERRAN warm-start source path disagrees with provenance")
        if int(declared.get("source_epoch", -1)) != source_epoch:
            raise ValueError("TERRAN warm-start source epoch disagrees with provenance")
        if declared.get("source_seed") != int(payload["seed"]):
            raise ValueError("TERRAN warm-start source seed disagrees with provenance")
        declared_digest = str(declared.get("source_checkpoint_sha256", ""))
        if declared_digest != _sha256_file(source):
            raise ValueError("TERRAN warm-start checkpoint SHA256 mismatch")
    mode = warm_start_epoch_mode(cfg)
    return {
        "checkpoint": str(source),
        "method": "TERRAN",
        "source_epoch": source_epoch,
        "source_seed": int(payload["seed"]),
        "source_checkpoint_path": str(source),
        "source_checkpoint_sha256": _sha256_file(source),
        "model_state_dict_loaded": True,
        "optimizer_state_dict_loaded": False,
        "optimizer_name": "adamw",
        "optimizer_reset": True,
        "warm_start_epoch_mode": mode,
        "epoch_reset": mode == "reset",
        "validation_state_reset": True,
        "early_stop_state_reset": True,
    }


def validate_fresh_training_output(cfg: dict[str, Any]) -> None:
    """A fresh launch may have launcher logs, but must not inherit training history."""
    if not cfg.get("output_dir"):
        return
    output = Path(cfg["output_dir"])
    evidence = [
        output / name
        for name in (
            "checkpoint_latest.pt", "checkpoint_selected.pt", "data_pass_state.json",
            "best.ckpt", "best_overall.ckpt", "best_within_5000.ckpt",
            "training_result.json", "validation_history.jsonl", "validation_summary.json",
            "validation_summary_overall.json", "validation_summary_within_5000.json",
            "warm_start_initial.ckpt",
            "logs/train_log.csv", "logs/eval_log.csv",
        )
        if (output / name).exists()
    ]
    checkpoint_dir = output / "checkpoints"
    if checkpoint_dir.is_dir():
        evidence.extend(checkpoint_dir.iterdir())
    if evidence:
        raise FileExistsError(
            "TERRAN fresh training output already contains training evidence: "
            f"{evidence[0]}. Use same-contract --resume or a new output directory."
        )


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_bool = mask.bool()
    denom = torch.clamp(mask_bool.sum(), min=1).to(dtype=value.dtype)
    return torch.where(mask_bool, value, torch.zeros_like(value)).sum() / denom


def _valid_transition_count(
    batch,
    env_indices: Sequence[int] | np.ndarray,
    step_start: int = 0,
    step_end: int | None = None,
) -> int:
    """Count active transitions in one PPO env/time slice."""
    if step_end is None:
        step_end = int(batch.valid.size(0))
    return int(
        batch.valid[int(step_start) : int(step_end), env_indices].sum().item()
    )


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
        use_terminal_task_penalty=bool(
            pbrs.get("use_terminal_task_penalty", False)
        ),
        customer_pbrs_mode=str(pbrs.get("customer_pbrs_mode", "progress")),
        gamma=training_gamma(cfg),
        alpha=float(pbrs.get("alpha", 2.0)),
        beta=float(pbrs.get("beta", 0.5)),
        customer_pbrs_coef=float(pbrs.get("customer_pbrs_coef", 1.0)),
        customer_progress_budget=float(pbrs.get("customer_progress_budget", 0.5)),
        customer_progress_mix=float(pbrs.get("customer_progress_mix", 0.5)),
        repair_progress_coef=float(pbrs.get("repair_progress_coef", 0.5)),
        feasible_ratio_coef=float(pbrs.get("feasible_ratio_coef", 0.0)),
        pbrs_clip=pbrs.get("pbrs_clip", None),
        terminal_success_bonus=terminal_success_bonus(cfg),
        success_bonus=float(pbrs.get("success_bonus", 0.1)),
        failure_penalty=float(pbrs.get("failure_penalty", 0.5)),
        failure_base=float(pbrs.get("failure_base", 0.0)),
        unserved_coefficient=float(pbrs.get("unserved_coefficient", 1.0)),
    )
    if not (
        config.use_customer_pbrs
        or config.use_repair_distance_pbrs
        or config.use_feasible_ratio_pbrs
        or config.use_terminal_heuristic
        or config.use_terminal_task_penalty
    ):
        return None
    return config


def _resolved_pbrs_reward_semantics(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return every reward-affecting PBRS value with defaults resolved.

    The shared reward contract intentionally does not hash method-specific
    shaping.  TERRAN nevertheless has to reject a resume when its potential,
    coefficients, or annealing schedule changed.  In particular, materialize
    the default annealing end epoch now so extending ``training.epochs`` cannot
    silently change a checkpoint's shaping schedule.
    """

    pbrs = cfg.get("pbrs", {}) or {}
    training = cfg.get("training", {}) or {}
    total_epochs = int(training.get("epochs", 1000))
    if total_epochs <= 0:
        raise ValueError("training.epochs must be positive before freezing PBRS")
    raw_annealing = pbrs.get("annealing", {}) or {}
    start_epoch = max(1, int(raw_annealing.get("start_epoch", 1)))
    end_epoch = max(
        start_epoch,
        int(raw_annealing.get("end_epoch", total_epochs)),
    )
    schedule = str(raw_annealing.get("schedule", "cosine")).lower()
    if schedule not in {"cosine", "linear", "exponential", "constant"}:
        raise ValueError(f"unsupported PBRS annealing schedule: {schedule!r}")
    annealing = {
        "enabled": bool(raw_annealing.get("enabled", False)),
        "start_scale": float(raw_annealing.get("start_scale", 1.0)),
        "end_scale": float(raw_annealing.get("end_scale", 0.2)),
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "schedule": schedule,
    }
    if not all(
        math.isfinite(float(annealing[name]))
        and float(annealing[name]) >= 0.0
        for name in ("start_scale", "end_scale")
    ):
        raise ValueError("PBRS annealing scales must be finite and non-negative")
    config = build_pbrs_config(cfg)
    return {
        "schema": "terran_pbrs_reward_semantics_v2",
        "wrapper_enabled": config is not None,
        "potential_reward_config": asdict(config) if config is not None else None,
        "annealing": dict(annealing),
    }


def _freeze_pbrs_reward_semantics(cfg: dict[str, Any]) -> dict[str, Any]:
    """Materialize resolved PBRS semantics into the live config/checkpoint."""

    snapshot = _resolved_pbrs_reward_semantics(cfg)
    cfg.setdefault("training", {})["epochs"] = int(
        cfg.get("training", {}).get("epochs", 1000)
    )
    cfg.setdefault("pbrs", {})["annealing"] = dict(snapshot["annealing"])
    cfg["pbrs_reward_semantics"] = snapshot
    return snapshot


def _configure_reward_contract(cfg: dict[str, Any]) -> None:
    """Resolve and freeze the common task scale before any env is created."""

    source = cfg.get("reward_contract")
    if source in (None, ""):
        return
    if isinstance(source, (str, Path)):
        contract = load_reward_contract(source)
        cfg["reward_contract_source_path"] = str(contract.source_path)
    elif isinstance(source, dict):
        contract = RewardContract.from_payload(source)
    else:
        raise TypeError("reward_contract must be a JSON path or frozen mapping")
    objective = resolve_objective(cfg.get("objective"))
    data = cfg.get("data", {})
    raw_scale = data.get("stage2_scale")
    if raw_scale in (None, ""):
        raw_scale = f"Cus{int(data.get('num_customers', 0))}"
    terms = contract.for_scale(raw_scale, objective)
    completion_bonus = terminal_success_bonus(cfg)
    cfg["reward_contract"] = contract.to_dict()
    cfg.setdefault("training", {})["reward_contract_id"] = terms.contract_id
    cfg.setdefault("normalization", {}).update(
        {
            "reward_contract_id": terms.contract_id,
            "reward_contract_sha256": terms.digest,
            "reward_contract_scale": terms.scale_label,
            "reward_objective_scale": terms.objective_scale,
            "failure_base": terms.failure_base,
            "unserved_coefficient": terms.unserved_coefficient,
            "terran_terminal_success_bonus": completion_bonus,
            "terran_terminal_success_bonus_unit": TERMINAL_TASK_REWARD_UNIT,
            "terran_terminal_success_bonus_equivalent_usd": (
                completion_bonus * terms.objective_scale
            ),
            "reward_objective_scale_source": "frozen_training_reference_contract",
        }
    )
    cfg.setdefault("env", {}).update(
        {
            "normalize_reward": True,
            "reward_objective_scale": terms.objective_scale,
            "invalid_action_penalty": 0.0,
            "success_bonus": 0.0,
        }
    )
    cfg.setdefault("pbrs", {}).update(
        {
            "use_terminal_heuristic": False,
            "use_terminal_task_penalty": True,
            "success_bonus": 0.0,
            "terminal_success_bonus": completion_bonus,
            "failure_base": terms.failure_base,
            "unserved_coefficient": terms.unserved_coefficient,
        }
    )
    _freeze_pbrs_reward_semantics(cfg)


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
    _configure_reward_contract(cfg)
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
            training_stream_contract_sha256=(
                (cfg.get("protocol", {}) or {}).get(
                    "training_stream_contract_sha256"
                )
            ),
            training_stream_contract_snapshot=(
                (cfg.get("protocol", {}) or {}).get(
                    "training_stream_contract_snapshot"
                )
            ),
            stream_integrity_mode=str(
                (cfg.get("protocol", {}) or {}).get(
                    "stream_integrity_mode",
                    STREAM_INTEGRITY_MODE_RUNTIME_REVERIFIED,
                )
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
    env_cfg["objective_config"] = resolve_objective(cfg.get("objective"))
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


def _decode_outcome_reason_counts(
    value: Any, *, expected_count: int, context: str
) -> dict[str, int]:
    try:
        decoded = json.loads(value) if isinstance(value, str) else dict(value)
        normalized = {
            str(reason): int(count) for reason, count in decoded.items()
        }
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {context} reason-count mapping") from exc
    if any(count < 0 for count in normalized.values()):
        raise ValueError(f"negative count in {context} reason-count mapping")
    if sum(normalized.values()) != int(expected_count):
        raise RuntimeError(
            f"{context} reason counts do not sum to non-horizon infeasible count"
        )
    return normalized


def summarize_eval_candidate_outcomes(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-instance candidate diagnostics produced by eval rollout."""

    candidate_count = 0
    success_count = 0
    budget_count = 0
    non_horizon_count = 0
    all_non_horizon_instances = 0
    reason_counts: Counter[str] = Counter()
    monitored_instances = 0
    for row in rows:
        if "candidate_trajectory_count" not in row:
            # Older/custom rollout adapters remain valid; their result simply
            # has no candidate-level FFP diagnostic.
            continue
        monitored_instances += 1
        candidate_count += int(row["candidate_trajectory_count"])
        success_count += int(row.get("candidate_success_count", 0))
        budget_count += int(row.get("candidate_rollout_budget_exhausted_count", 0))
        row_non_horizon_count = int(
            row.get("candidate_non_horizon_infeasible_count", 0)
        )
        non_horizon_count += row_non_horizon_count
        all_non_horizon_instances += int(
            bool(row.get("no_success_all_candidates_non_horizon_infeasible", False))
        )
        encoded_reasons = row.get(
            "candidate_non_horizon_infeasible_reason_counts", "{}"
        )
        reason_counts.update(
            _decode_outcome_reason_counts(
                encoded_reasons,
                expected_count=row_non_horizon_count,
                context="per-instance candidate",
            )
        )

    if candidate_count and candidate_count != (
        success_count + budget_count + non_horizon_count
    ):
        raise RuntimeError("evaluation candidate outcome monitoring is not exhaustive")
    return {
        "eval_candidate_trajectory_count": candidate_count,
        "eval_candidate_success_count": success_count,
        "eval_candidate_success_rate": (
            success_count / candidate_count if candidate_count else None
        ),
        "eval_candidate_rollout_budget_exhausted_count": budget_count,
        "eval_candidate_rollout_budget_exhausted_rate": (
            budget_count / candidate_count if candidate_count else None
        ),
        "eval_candidate_non_horizon_infeasible_count": non_horizon_count,
        "eval_candidate_non_horizon_infeasible_rate": (
            non_horizon_count / candidate_count if candidate_count else None
        ),
        "eval_candidate_non_horizon_infeasible_reason_counts": json.dumps(
            dict(sorted(reason_counts.items())), sort_keys=True
        ),
        "eval_no_success_all_candidates_non_horizon_infeasible_instance_count": (
            all_non_horizon_instances
        ),
        "eval_no_success_all_candidates_non_horizon_infeasible_instance_rate": (
            all_non_horizon_instances / monitored_instances
            if monitored_instances
            else None
        ),
    }


def evaluate_fixed_dataset(
    agent: Agent,
    cfg: dict[str, Any],
    seed: int,
    epoch: int,
    device: str | torch.device,
) -> dict[str, Any]:
    eval_cfg = cfg.get("evaluation", {})
    objective_config = resolve_objective(cfg.get("objective"))
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
    ) or objective_config.is_cost
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
        eval_env_cfg["objective_config"] = objective_config
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
                selected, routes, verification = select_min_verified_objective(
                    instance, info, objective_config
                )
                row["selected_traj_idx"] = selected
                row["feasible"] = bool(verification["passed"])
                row["objective_distance_km"] = float(
                    verification["objective_distance_km"]
                )
                row["vehicle_count"] = len(routes)
                row["verifier_passed"] = bool(verification["passed"])
                if eval_save_routes:
                    row["routes_json"] = json.dumps(routes)
                    row["route_sequence_json"] = json.dumps(merge_route_sequences(routes))
                for key in ("objective_value", "objective_cost_usd", "electricity_cost_usd", "vehicle_cost_usd", "vehicles_started", "objective_mode", "objective_unit"):
                    row[key] = verification.get(key)
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
    outcome_summary = summarize_eval_candidate_outcomes(rows)
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
        "eval_objective_mode": objective_config.mode,
        "eval_objective_unit": objective_config.unit,
        "eval_avg_objective": float(np.mean([row.get("objective_value", row["objective_distance_km"]) for row in feasible_rows])) if feasible_rows else np.nan,
        **{
            f"eval_avg_{name}": float(np.mean([row[name] for row in feasible_rows]))
            if feasible_rows and objective_config.is_cost else None
            for name in ("objective_cost_usd", "electricity_cost_usd", "vehicle_cost_usd")
        },
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
        **outcome_summary,
        "eval_status": "ok",
    }


def validation_summary_from_eval_row(
    eval_row: Mapping[str, Any],
    *,
    objective_config: Any,
    logical_epoch: int,
    validation_seed: int,
    validation_wall_time_s: float,
) -> dict[str, Any] | None:
    """Convert one fixed-dataset evaluation into selection evidence."""

    if eval_row.get("eval_status") != "ok":
        return None
    verified_distance = eval_row.get("eval_avg_objective_distance_km")
    validation: dict[str, Any] = {
        "schema": "drl_validation_summary_v1",
        "split": "validation",
        "logical_epoch": int(logical_epoch),
        "validation_seed": int(validation_seed),
        "instances": int(eval_row["eval_num_instances"]),
        "complete_and_feasible": int(eval_row["eval_complete_and_feasible"]),
        "complete_and_feasible_rate": float(eval_row["eval_feasible_rate"]),
        "mean_verified_distance_km": (
            float(verified_distance)
            if verified_distance is not None
            and np.isfinite(float(verified_distance))
            else None
        ),
        "validation_wall_time_s": float(validation_wall_time_s),
        "verifier_summary_passed": bool(
            eval_row.get("eval_independent_verifier", False)
            and int(eval_row["eval_complete_and_feasible"])
            == int(eval_row["eval_num_instances"])
        ),
    }
    candidate_count = eval_row.get("eval_candidate_trajectory_count")
    if candidate_count is not None:
        candidate_non_horizon_count = int(
            eval_row.get("eval_candidate_non_horizon_infeasible_count", 0)
        )
        decoded_reasons = _decode_outcome_reason_counts(
            eval_row.get(
                "eval_candidate_non_horizon_infeasible_reason_counts", "{}"
            ),
            expected_count=candidate_non_horizon_count,
            context="validation candidate aggregate",
        )
        validation.update(
            {
                "candidate_trajectory_count": int(candidate_count),
                "candidate_success_count": int(
                    eval_row.get("eval_candidate_success_count", 0)
                ),
                "candidate_success_rate": eval_row.get(
                    "eval_candidate_success_rate"
                ),
                "candidate_rollout_budget_exhausted_count": int(
                    eval_row.get(
                        "eval_candidate_rollout_budget_exhausted_count", 0
                    )
                ),
                "candidate_rollout_budget_exhausted_rate": eval_row.get(
                    "eval_candidate_rollout_budget_exhausted_rate"
                ),
                "candidate_non_horizon_infeasible_count": (
                    candidate_non_horizon_count
                ),
                "candidate_non_horizon_infeasible_rate": eval_row.get(
                    "eval_candidate_non_horizon_infeasible_rate"
                ),
                "candidate_non_horizon_infeasible_reason_counts": decoded_reasons,
                "no_success_all_candidates_non_horizon_infeasible_instance_count": int(
                    eval_row.get(
                        "eval_no_success_all_candidates_non_horizon_infeasible_instance_count",
                        0,
                    )
                ),
                "no_success_all_candidates_non_horizon_infeasible_instance_rate": eval_row.get(
                    "eval_no_success_all_candidates_non_horizon_infeasible_instance_rate"
                ),
            }
        )
    verified_objective = eval_row.get(
        "eval_avg_objective", verified_distance
    )
    validation.update(
        objective_mode=objective_config.mode,
        objective_unit=objective_config.unit,
        objective_config=objective_config.to_dict(),
        mean_verified_objective=(
            float(verified_objective)
            if verified_objective is not None
            and np.isfinite(float(verified_objective))
            else None
        ),
        mean_verified_vehicle_count=eval_row.get("eval_avg_vehicle_count"),
        mean_verified_objective_cost_usd=eval_row.get(
            "eval_avg_objective_cost_usd"
        ),
        mean_verified_cost_usd=eval_row.get("eval_avg_objective_cost_usd"),
        mean_verified_electricity_cost_usd=eval_row.get(
            "eval_avg_electricity_cost_usd"
        ),
        mean_verified_vehicle_cost_usd=eval_row.get(
            "eval_avg_vehicle_cost_usd"
        ),
    )
    return validation


def summarize_train_infos(final_infos: list[dict[str, Any]]) -> dict[str, Any]:
    if not final_infos:
        return {
            "train_feasible_rate": np.nan,
            "train_avg_best_objective_distance_km": np.nan,
            "train_avg_best_objective": np.nan,
            "train_avg_vehicle_count": np.nan,
            "train_avg_served_customers": np.nan,
            "terminal_outcome_reason_counts": {},
        }
    feasible_flags = []
    best_objectives = []
    best_distances = []
    vehicle_counts = []
    served_counts = []
    reason_counts: Counter[str] = Counter()
    for info in final_infos:
        success = np.asarray(info.get("success", []), dtype=bool)
        distance = np.asarray(info.get("objective_distance_km", []), dtype=np.float64)
        objective = np.asarray(info.get("objective_value", distance), dtype=np.float64)
        vehicle = np.asarray(info.get("vehicle_count", []), dtype=np.float64)
        served = np.asarray(info.get("served_customers", []), dtype=np.float64)
        reasons = np.asarray(info.get("failure_reason", []), dtype=object)
        if objective.size == 0:
            continue
        feasible_flags.extend(success.tolist())
        served_counts.extend(served.tolist())
        reason_counts.update(str(value) for value in reasons.reshape(-1))
        if np.any(success):
            candidates = np.where(success)[0]
            selected = int(candidates[np.argmin(objective[candidates])])
            best_objectives.append(float(objective[selected]))
            best_distances.append(float(distance[selected]))
            vehicle_counts.append(float(vehicle[selected]) if vehicle.size else np.nan)
    return {
        "train_feasible_rate": float(np.mean(feasible_flags)) if feasible_flags else np.nan,
        "train_avg_best_objective_distance_km": float(np.mean(best_distances)) if best_distances else np.nan,
        "train_avg_best_objective": float(np.mean(best_objectives)) if best_objectives else np.nan,
        "train_avg_vehicle_count": float(np.mean(vehicle_counts)) if vehicle_counts else np.nan,
        "train_avg_served_customers": float(np.mean(served_counts)) if served_counts else np.nan,
        "terminal_outcome_reason_counts": dict(sorted(reason_counts.items())),
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
    # No encoded state is reused across chunks or optimizer updates because the
    # encoder graph is consumed by backward and parameters may then change.
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

    policy_terms = []
    value_terms = []
    entropy_terms = []
    valid_masks = []
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
        policy_terms.append(-torch.minimum(unclipped, clipped))
        value_terms.append(
            F.mse_loss(
                value,
                returns[step, env_indices],
                reduction="none",
            )
        )
        entropy_terms.append(entropy)
        valid_masks.append(valid)
    active = torch.stack(valid_masks)
    policy_loss = masked_mean(torch.stack(policy_terms), active)
    value_loss = masked_mean(torch.stack(value_terms), active)
    entropy_loss = masked_mean(torch.stack(entropy_terms), active)
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


def apply_training_initialization(
    agent: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    resume_payload: Mapping[str, Any] | None = None,
    warm_start_payload: Mapping[str, Any] | None = None,
    warm_start_mode: str = "continue_global",
) -> int:
    """Load a continuation checkpoint or weights-only initialization."""

    if resume_payload is not None and warm_start_payload is not None:
        raise ValueError("resume and weights-only warm start are mutually exclusive")
    if resume_payload is not None:
        agent.load_state_dict(resume_payload["model_state_dict"])
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        return int(resume_payload["epoch"]) + 1
    if warm_start_payload is not None:
        agent.load_state_dict(warm_start_payload["model_state_dict"], strict=True)
        if optimizer.state:
            raise RuntimeError("TERRAN warm-start AdamW optimizer was not reset")
        if warm_start_mode == "reset":
            return 1
        if warm_start_mode == "continue_global":
            return int(warm_start_payload["epoch"]) + 1
        raise ValueError(
            "TERRAN warm_start_mode must be 'reset' or 'continue_global'"
        )
    return 1


def train_from_config(cfg: dict[str, Any], seed: int, device: str | None = None, overrides: dict[str, Any] | None = None) -> Path:
    cfg = deep_update(cfg, overrides or {})
    set_seed(seed)
    train_cfg = cfg["training"]
    optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
    if optimizer_name != "adamw":
        raise ValueError(
            f"unsupported TERRAN optimizer: {optimizer_name}; expected adamw"
        )
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    if not np.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("TERRAN weight_decay must be finite and non-negative")
    train_cfg["optimizer"] = optimizer_name
    train_cfg["weight_decay"] = weight_decay
    gamma = training_gamma(cfg)
    objective_config = resolve_objective(cfg.get("objective"))
    cfg["objective"] = objective_config.to_dict()
    _configure_reward_contract(cfg)
    protocol_id = str(cfg.get("protocol", {}).get("protocol_id", "")).strip()
    if objective_config.is_cost and protocol_id and cfg.get("reward_contract") is None:
        raise ValueError(
            "formal TERRAN cost training requires a frozen reward contract"
        )
    # Contract configuration freezes this after injecting terminal terms. Legacy
    # runs still need an explicit snapshot so any new checkpoint is resume-safe.
    _freeze_pbrs_reward_semantics(cfg)
    if objective_config.is_cost:
        if gamma != 1.0:
            raise ValueError("TERRAN cost objective requires undiscounted training.gamma=1")
        if cfg.get("reward_contract") is None:
            train_cfg["reward_contract_id"] = "terran_undiscounted_energy_vehicle_pbrs_v1"
    elif train_cfg.get("reward_contract_id") == "terran_undiscounted_energy_vehicle_pbrs_v1":
        train_cfg["reward_contract_id"] = "terran_undiscounted_distance_pbrs_v1"
    # Persist the resolved value even for callers using the default. Resume
    # must never guess which discount produced an existing checkpoint.
    train_cfg["gamma"] = gamma
    _freeze_resolved_terran_training_signature(cfg, seed=seed)
    protocol_cfg = cfg.get("protocol", {})
    resume_checkpoint = protocol_cfg.get("resume_checkpoint")
    warm_start_checkpoint = protocol_cfg.get("warm_start_checkpoint")
    if resume_checkpoint and warm_start_checkpoint:
        raise ValueError(
            "TERRAN resume and weights-only warm start are mutually exclusive"
        )
    inherited_warm_start = protocol_cfg.get("warm_start")
    warm_start_mode = (
        warm_start_epoch_mode(cfg)
        if warm_start_checkpoint or inherited_warm_start is not None
        else "reset"
    )
    warm_start_continues_global_epoch = (
        warm_start_checkpoint is not None
        and warm_start_mode == "continue_global"
    )
    warm_start_accounting_source_epoch = (
        int(inherited_warm_start.get("source_epoch", 0))
        if warm_start_mode == "continue_global"
        and isinstance(inherited_warm_start, Mapping)
        else 0
    )
    resume_payload = None
    warm_start_payload = None
    warm_start_provenance: dict[str, Any] | None = None
    warm_start_source_epoch: int | None = None
    if resume_checkpoint:
        resume_payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        validate_resume_reward_contract(cfg, resume_payload)
    elif warm_start_checkpoint:
        warm_start_payload = torch.load(
            warm_start_checkpoint, map_location="cpu", weights_only=False
        )
        warm_start_provenance = validate_warm_start_contract(
            cfg,
            warm_start_payload,
            current_seed=seed,
            checkpoint_path=warm_start_checkpoint,
        )
        warm_start_source_epoch = int(warm_start_provenance["source_epoch"])
        warm_start_accounting_source_epoch = (
            warm_start_source_epoch
            if warm_start_continues_global_epoch
            else 0
        )
        if warm_start_continues_global_epoch:
            strict_source_epoch = validate_warm_start_checkpoint(
                cfg,
                warm_start_payload,
                checkpoint_path=warm_start_checkpoint,
            )
            if strict_source_epoch != warm_start_source_epoch:
                raise RuntimeError(
                    "TERRAN warm-start validators disagree on source epoch"
                )
        validate_fresh_training_output(cfg)
    else:
        validate_fresh_training_output(cfg)
    training_started = time.perf_counter()
    eval_cfg = cfg.get("evaluation", {})
    if warm_start_continues_global_epoch and not eval_cfg.get("eval_path"):
        raise ValueError(
            "TERRAN continue_global warm start requires fixed validation data "
            "so the initialization incumbent can be selected"
        )
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
    optimizer = build_adamw_optimizer(
        agent.parameters(),
        learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
        eps=1e-5,
        weight_decay=weight_decay,
    )
    initial_env_start = time.perf_counter()
    envs, pool = make_envs(cfg, seed)
    initial_env_pool_time_s = time.perf_counter() - initial_env_start
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
    cache_rollout_encoder = bool(train_cfg.get("cache_rollout_encoder", True))
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
    start_epoch = apply_training_initialization(
        agent,
        optimizer,
        resume_payload=resume_payload,
        warm_start_payload=warm_start_payload,
        warm_start_mode=warm_start_mode,
    )
    if resume_payload is not None:
        del resume_payload
    if warm_start_payload is not None:
        assert warm_start_source_epoch is not None
        expected_start_epoch = (
            warm_start_source_epoch + 1
            if warm_start_continues_global_epoch
            else 1
        )
        if start_epoch != expected_start_epoch:
            raise RuntimeError("TERRAN warm-start epoch initialization failed")
        del warm_start_payload

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
    reward_diagnostics_path = (out_root if cfg.get("output_dir") else log_dir) / "reward_diagnostics.jsonl"
    diagnostic_session_id = f"{os.getpid()}-{time.time_ns()}"
    validation_history_path = out_root / "validation_history.jsonl"
    validation_summary_path = out_root / "validation_summary.json"
    validation_summary_within_path = out_root / "validation_summary_within_5000.json"
    validation_summary_overall_path = out_root / "validation_summary_overall.json"
    best_checkpoint_path = out_root / "best.ckpt"
    best_within_minimum_path = out_root / "best_within_5000.ckpt"
    best_overall_path = out_root / "best_overall.ckpt"
    selected_checkpoint_path = out_root / "checkpoint_selected.pt"
    warm_start_initial_checkpoint_path = out_root / "warm_start_initial.ckpt"
    minimum_training_epochs = int(
        train_cfg.get("minimum_training_epochs", epochs) or epochs
    )
    if warm_start_continues_global_epoch:
        assert warm_start_source_epoch is not None
        save_checkpoint(
            warm_start_initial_checkpoint_path,
            agent,
            optimizer,
            cfg,
            warm_start_source_epoch,
            seed,
        )
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
        best_eval_key = validation_key(previous_validation)
    previous_within_path = (
        validation_summary_within_path
        if validation_summary_within_path.is_file()
        else validation_summary_path
    )
    if previous_within_path.is_file():
        previous_within = json.loads(previous_within_path.read_text(encoding="utf-8"))
        previous_within_epoch = int(
            previous_within.get("logical_epoch", minimum_training_epochs) or 0
        )
        if previous_within_epoch <= minimum_training_epochs:
            best_within_minimum_key = validation_key(previous_within)
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
            key = validation_key(row)
            is_warm_start_baseline = bool(
                row.get("warm_start_initialization", False)
            )
            if not is_warm_start_baseline:
                completed_validation_checks += 1
            logical_epoch = int(row.get("logical_epoch", 0) or 0)
            if key > history_best_key:
                history_best_key = key
                validation_checks_without_improvement = 0
            elif (
                not is_warm_start_baseline
                and logical_epoch > early_stop_start_epoch
            ):
                validation_checks_without_improvement += 1
            elif not is_warm_start_baseline:
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
        "reward_base_mean",
        "reward_distance_mean",
        "reward_base_non_distance_mean",
        "reward_pbrs_customer_mean",
        "reward_pbrs_repair_distance_mean",
        "reward_terminal_heuristic_mean",
        "reward_terminal_task_total_mean",
        "reward_terminal_success_bonus_mean",
        "reward_terminal_failure_base_mean",
        "reward_terminal_unserved_mean",
        "reward_pbrs_total_mean",
        "reward_shaping_total_mean",
        "reward_base_abs_mean",
        "reward_distance_abs_mean",
        "reward_pbrs_total_abs_mean",
        "reward_pbrs_to_base_abs_ratio",
        "reward_pbrs_to_distance_abs_ratio",
        "reward_base_per_trajectory",
        "reward_distance_per_trajectory",
        "reward_pbrs_total_per_trajectory",
        "reward_terminal_heuristic_per_trajectory",
        "reward_terminal_task_total_per_trajectory",
        "reward_terminal_success_bonus_per_trajectory",
        "reward_discounted_distance_per_trajectory",
        "reward_discounted_pbrs_total_per_trajectory",
        "reward_discounted_terminal_heuristic_per_trajectory",
        "reward_discounted_terminal_task_total_per_trajectory",
        "reward_discounted_terminal_success_bonus_per_trajectory",
        "customer_action_reward_base_mean",
        "customer_action_reward_pbrs_total_mean",
        "noncustomer_action_reward_pbrs_total_mean",
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
        "terminal_outcome_reason_counts",
        "successful_trajectory_count",
        "successful_trajectory_rate",
        "mean_trajectory_steps",
        "trajectory_steps_p50",
        "trajectory_steps_p90",
        "trajectory_steps_p99",
        "trajectory_steps_max",
        "rollout_budget_exhausted_count",
        "rollout_budget_exhausted_rate",
        "non_horizon_infeasible_count",
        "non_horizon_infeasible_rate",
        "non_horizon_infeasible_reason_counts",
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
    train_fields.extend(EVAL_OUTCOME_FIELDS)
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
    eval_fields.extend(EVAL_OUTCOME_FIELDS)
    train_fields.extend(["train_avg_best_objective", "objective_mode", "objective_unit", "reward_objective_scale"])
    train_fields.extend(OBJECTIVE_EVAL_FIELDS)
    eval_fields.extend(OBJECTIVE_EVAL_FIELDS)
    train_fields.extend(
        f"reward_{component}_{suffix}"
        for component in OBJECTIVE_REWARD_COMPONENTS
        for suffix in ("mean", "per_trajectory", "discounted_per_trajectory")
    )

    log_mode = "a" if resume_checkpoint else "w"
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
        if warm_start_continues_global_epoch:
            assert warm_start_source_epoch is not None
            baseline_start = time.perf_counter()
            baseline_eval_row = evaluate_fixed_dataset(
                agent,
                cfg,
                seed=seed,
                epoch=warm_start_source_epoch,
                device=device,
            )
            baseline_wall_time_s = time.perf_counter() - baseline_start
            eval_writer.writerow(
                {"epoch": warm_start_source_epoch, **baseline_eval_row}
            )
            ef.flush()
            baseline_validation = validation_summary_from_eval_row(
                baseline_eval_row,
                objective_config=objective_config,
                logical_epoch=warm_start_source_epoch,
                validation_seed=validation_seed,
                validation_wall_time_s=baseline_wall_time_s,
            )
            if baseline_validation is None:
                raise RuntimeError(
                    "TERRAN warm-start initialization validation did not complete"
                )
            baseline_key = validation_key(baseline_validation)
            baseline_validation.update(
                {
                    "warm_start_initialization": True,
                    "warm_start_source_checkpoint": str(
                        warm_start_checkpoint
                    ),
                    "checkpoint_selected": True,
                    "best_within_minimum_selected": True,
                    "best_overall_selected": True,
                    "minimum_training_epochs": minimum_training_epochs,
                    "validation_checks_without_improvement": 0,
                    "early_stop_start_epoch": early_stop_start_epoch,
                    "early_stop_eligible": False,
                    "early_stop_due": False,
                }
            )
            append_jsonl(validation_history_path, baseline_validation)
            best_eval_key = baseline_key
            best_within_minimum_key = baseline_key
            history_best_key = baseline_key
            shutil.copy2(warm_start_initial_checkpoint_path, best_overall_path)
            shutil.copy2(
                warm_start_initial_checkpoint_path, best_within_minimum_path
            )
            shutil.copy2(warm_start_initial_checkpoint_path, best_checkpoint_path)
            shutil.copy2(
                warm_start_initial_checkpoint_path, selected_checkpoint_path
            )
            atomic_json(validation_summary_overall_path, baseline_validation)
            atomic_json(validation_summary_within_path, baseline_validation)
            atomic_json(validation_summary_path, baseline_validation)
            _debug_log(
                debug_enabled,
                df,
                "[WarmStartEval] "
                f"epoch={warm_start_source_epoch}/{epochs} "
                f"fr={_format_float(baseline_eval_row.get('eval_feasible_rate'))} "
                f"obj={_format_float(baseline_eval_row.get('eval_avg_objective', baseline_eval_row.get('eval_avg_objective_distance_km')))}"
                f"{objective_config.unit} "
                f"veh={_format_float(baseline_eval_row.get('eval_avg_vehicle_count'))} "
                f"eval_wall={baseline_wall_time_s:.3f}s",
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
            rollout_budget_parts: list[np.ndarray] = []
            reward_sum = 0.0
            reward_count = 0
            reward_diagnostics: dict[str, float] = {}
            base_reward_stats = BoundedBaseRewardStats()
            normalization_records = []
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
                    cache_static_embeddings=cache_rollout_encoder,
                    reward_discount_factor=gamma,
                    base_reward_stats=base_reward_stats,
                )
                for env in envs:
                    base_env = getattr(env, "unwrapped", env)
                    normalization_records.append({
                        "normalize_reward": bool(getattr(base_env, "normalize_reward", False)),
                        "distance_scale_km": float(getattr(base_env, "reward_distance_scale_km", 1.0)),
                        "objective_scale": float(getattr(base_env, "reward_objective_scale", 1.0)),
                    })
                returns = compute_returns(batch.rewards, batch.dones, gamma=gamma)
                advantages = returns - batch.values
                rollout_records.append((batch, returns, advantages))
                valid_count = int(batch.valid.sum().item())
                environment_transitions += valid_count
                reward_count += valid_count
                if valid_count:
                    reward_sum += float(batch.rewards[batch.valid].sum().detach().cpu())
                batch_reward_diagnostics = getattr(
                    batch, "reward_diagnostics", None
                )
                if batch_reward_diagnostics is None:
                    fallback_reward_sum = float(
                        batch.rewards[batch.valid].sum().detach().cpu()
                    )
                    fallback_reward_abs_sum = float(
                        batch.rewards[batch.valid].abs().sum().detach().cpu()
                    )
                    batch_reward_diagnostics = {
                        "active_count": float(valid_count),
                        "base_sum": fallback_reward_sum,
                        "base_abs_sum": fallback_reward_abs_sum,
                        "distance_sum": fallback_reward_sum,
                        "distance_abs_sum": fallback_reward_abs_sum,
                        "shaped_sum": fallback_reward_sum,
                        "shaped_abs_sum": fallback_reward_abs_sum,
                    }
                for key, value in batch_reward_diagnostics.items():
                    reward_diagnostics[key] = (
                        reward_diagnostics.get(key, 0.0) + float(value)
                    )
                trajectory_parts.append(
                    batch.trajectory_steps.detach().cpu().numpy().reshape(-1)
                )
                final_infos.extend(batch.final_infos)
                rollout_budget_parts.append(
                    batch.rollout_budget_exhausted.detach().cpu().numpy()
                )
                rollout_budget_exhausted_count += int(
                    batch.rollout_budget_exhausted.sum().detach().cpu()
                )
                for key, value in batch.timings.items():
                    rollout_timings[key] = rollout_timings.get(key, 0.0) + float(value)

            environment_transitions_total += environment_transitions
            trajectory_steps = np.concatenate(trajectory_parts)
            trajectory_count = int(trajectory_steps.size)
            outcome_summary = summarize_rollout_outcomes(
                final_infos, np.concatenate(rollout_budget_parts, axis=0)
            )
            if outcome_summary["trajectory_count"] != trajectory_count:
                raise RuntimeError(
                    "trajectory outcome monitoring count disagrees with rollout buffers"
                )
            if (
                outcome_summary["rollout_budget_exhausted_count"]
                != rollout_budget_exhausted_count
            ):
                raise RuntimeError(
                    "trajectory horizon monitoring count disagrees with rollout buffers"
                )
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
            preclip_norms = []
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
                        group_transition_count = sum(
                            _valid_transition_count(
                                batch,
                                env_indices,
                                step_end=total_steps,
                            )
                            for env_indices in accum_group
                        )
                        if group_transition_count <= 0:
                            raise RuntimeError(
                                "PPO optimizer group contains no valid transitions"
                            )
                        for env_indices in accum_group:
                            weighted_policy = 0.0
                            weighted_value = 0.0
                            weighted_entropy = 0.0
                            for step_start in range(0, total_steps, chunk_size):
                                step_end = min(
                                    step_start + chunk_size, total_steps
                                )
                                chunk_transition_count = _valid_transition_count(
                                    batch,
                                    env_indices,
                                    step_start,
                                    step_end,
                                )
                                if chunk_transition_count <= 0:
                                    continue
                                chunk_weight = (
                                    float(chunk_transition_count)
                                    / float(group_transition_count)
                                )
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
                                (loss * chunk_weight).backward()
                                weighted_policy += (
                                    policy_loss.item() * chunk_weight
                                )
                                weighted_value += value_loss.item() * chunk_weight
                                weighted_entropy += entropy.item() * chunk_weight
                            group_policy += weighted_policy
                            group_value += weighted_value
                            group_entropy += weighted_entropy
                        preclip_norm = torch.nn.utils.clip_grad_norm_(
                            agent.parameters(),
                            float(train_cfg.get("max_grad_norm", 1.0)),
                        )
                        preclip_norms.append(preclip_norm.detach())
                        optimizer.step()
                        optimizer_steps_total += 1
                        losses.append(
                            (group_policy, group_value, group_entropy)
                        )
            else:
                # Multiple physical rollout buffers form one effective batch.
                # PPO gradients are weighted by their active-transition share and
                # accumulated before each optimizer step, keeping GPU residency
                # at the registered physical batch size.
                effective_transition_count = sum(
                    int(batch.valid.sum().item())
                    for batch, _, _ in rollout_records
                )
                if effective_transition_count <= 0:
                    raise RuntimeError(
                        "PPO effective batch contains no valid transitions"
                    )
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
                            for step_start in range(
                                0, total_steps, chunk_size
                            ):
                                step_end = min(
                                    step_start + chunk_size, total_steps
                                )
                                chunk_transition_count = _valid_transition_count(
                                    batch,
                                    env_indices,
                                    step_start,
                                    step_end,
                                )
                                if chunk_transition_count <= 0:
                                    continue
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
                                weight = (
                                    float(chunk_transition_count)
                                    / float(effective_transition_count)
                                )
                                (loss * weight).backward()
                                group_policy += policy_loss.item() * weight
                                group_value += value_loss.item() * weight
                                group_entropy += entropy.item() * weight
                    preclip_norm = torch.nn.utils.clip_grad_norm_(
                        agent.parameters(),
                        float(train_cfg.get("max_grad_norm", 1.0)),
                    )
                    preclip_norms.append(preclip_norm.detach())
                    optimizer.step()
                    optimizer_steps_total += 1
                    losses.append((group_policy, group_value, group_entropy))
            if profile_timing:
                _sync_cuda(device)
            ppo_update_time_s = time.perf_counter() - ppo_start
            reward_mean = reward_sum / max(reward_count, 1)
            component_count = max(reward_diagnostics.get("active_count", 0.0), 1.0)
            customer_action_count = max(
                reward_diagnostics.get("customer_action_count", 0.0), 1.0
            )
            noncustomer_action_count = max(
                reward_diagnostics.get("noncustomer_action_count", 0.0), 1.0
            )
            reward_base_mean = (
                reward_diagnostics.get("base_sum", 0.0) / component_count
            )
            reward_distance_mean = (
                reward_diagnostics.get("distance_sum", 0.0) / component_count
            )
            reward_base_non_distance_mean = (
                reward_diagnostics.get("base_non_distance_sum", 0.0)
                / component_count
            )
            reward_pbrs_customer_mean = (
                reward_diagnostics.get("pbrs_customer_sum", 0.0)
                / component_count
            )
            reward_pbrs_repair_distance_mean = (
                reward_diagnostics.get("pbrs_repair_distance_sum", 0.0)
                / component_count
            )
            reward_terminal_heuristic_mean = (
                reward_diagnostics.get("terminal_heuristic_sum", 0.0)
                / component_count
            )
            reward_terminal_task_total_mean = (
                reward_diagnostics.get("terminal_task_total_sum", 0.0)
                / component_count
            )
            reward_terminal_success_bonus_mean = (
                reward_diagnostics.get("terminal_success_bonus_sum", 0.0)
                / component_count
            )
            reward_terminal_failure_base_mean = (
                reward_diagnostics.get("terminal_failure_base_sum", 0.0)
                / component_count
            )
            reward_terminal_unserved_mean = (
                reward_diagnostics.get("terminal_unserved_sum", 0.0)
                / component_count
            )
            reward_pbrs_total_mean = (
                reward_diagnostics.get("pbrs_total_sum", 0.0)
                / component_count
            )
            reward_shaping_total_mean = (
                reward_diagnostics.get("shaping_total_sum", 0.0)
                / component_count
            )
            reward_base_abs_mean = (
                reward_diagnostics.get("base_abs_sum", 0.0) / component_count
            )
            reward_distance_abs_mean = (
                reward_diagnostics.get("distance_abs_sum", 0.0)
                / component_count
            )
            reward_pbrs_total_abs_mean = (
                reward_diagnostics.get("pbrs_total_abs_sum", 0.0)
                / component_count
            )
            reward_pbrs_to_base_abs_ratio = (
                reward_pbrs_total_abs_mean / max(reward_base_abs_mean, 1e-12)
            )
            reward_pbrs_to_distance_abs_ratio = (
                reward_pbrs_total_abs_mean
                / max(reward_distance_abs_mean, 1e-12)
            )
            loss_arr = np.asarray(losses, dtype=float)
            train_summary = summarize_train_infos(final_infos)
            if epoch % debug_log_every == 0:
                _debug_log(
                    debug_enabled,
                    df,
                    "[Train] "
                    f"epoch={epoch}/{epochs} samples={pool.sample_count} "
                    f"reward={_format_float(reward_mean)} "
                    f"distance={_format_float(reward_distance_mean, 6)} "
                    f"base_other={_format_float(reward_base_non_distance_mean, 6)} "
                    f"pbrs={_format_float(reward_pbrs_total_mean, 6)} "
                    f"pbrs_abs/distance_abs={_format_float(reward_pbrs_to_distance_abs_ratio, 3)} "
                    f"policy_loss={_format_float(loss_arr[:, 0].mean())} "
                    f"value_loss={_format_float(loss_arr[:, 1].mean())} "
                    f"entropy={_format_float(loss_arr[:, 2].mean())} "
                    f"train_fr={_format_float(train_summary['train_feasible_rate'])} "
                    f"train_obj={_format_float(train_summary['train_avg_best_objective'])}{objective_config.unit} "
                    f"train_veh={_format_float(train_summary['train_avg_vehicle_count'])} "
                    f"served={_format_float(train_summary['train_avg_served_customers'])} "
                    f"non_horizon_infeasible="
                    f"{outcome_summary['non_horizon_infeasible_count']}/"
                    f"{outcome_summary['trajectory_count']} "
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
                validation = validation_summary_from_eval_row(
                    eval_row,
                    objective_config=objective_config,
                    logical_epoch=epoch,
                    validation_seed=validation_seed,
                    validation_wall_time_s=eval_wall_time_s,
                )
                if validation is not None:
                    selection_key = validation_key(validation)
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
                            # Formal evaluation follows the best validation
                            # checkpoint across the full run, including the
                            # optional post-minimum tail.
                            "checkpoint_selected": is_best_overall,
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
                        shutil.copy2(best_overall_path, best_checkpoint_path)
                        shutil.copy2(best_overall_path, selected_checkpoint_path)
                        atomic_json(validation_summary_overall_path, validation)
                        atomic_json(validation_summary_path, validation)
                    if is_best_within_minimum:
                        best_within_minimum_key = selection_key
                        save_checkpoint(
                            best_within_minimum_path, agent, optimizer, cfg, epoch, seed
                        )
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
                    f"obj={_format_float(eval_row.get('eval_avg_objective', eval_row.get('eval_avg_objective_distance_km')))}{objective_config.unit} "
                    f"veh={_format_float(eval_row.get('eval_avg_vehicle_count'))} "
                    f"candidate_non_horizon_infeasible="
                    f"{eval_row.get('eval_candidate_non_horizon_infeasible_count', 'n/a')}/"
                    f"{eval_row.get('eval_candidate_trajectory_count', 'n/a')} "
                    f"runtime={_format_float(eval_row.get('eval_avg_runtime_s'))} "
                    f"eval_wall={eval_wall_time_s:.3f}s "
                    f"status={eval_row.get('eval_status')}",
                )
            epoch_wall_time_s = time.perf_counter() - epoch_start
            writer.writerow(
                {
                    "epoch": epoch,
                    "reward_mean": reward_mean,
                    "reward_base_mean": reward_base_mean,
                    "reward_distance_mean": reward_distance_mean,
                    "reward_base_non_distance_mean": reward_base_non_distance_mean,
                    **{
                        f"reward_{component}_{suffix}": reward_diagnostics.get(f"{component}_{stat}", 0.0) / denominator
                        for component in OBJECTIVE_REWARD_COMPONENTS
                        for suffix, stat, denominator in (
                            ("mean", "sum", component_count),
                            ("per_trajectory", "sum", max(trajectory_count, 1)),
                            ("discounted_per_trajectory", "discounted_sum", max(trajectory_count, 1)),
                        )
                    },
                    "objective_mode": objective_config.mode,
                    "objective_unit": objective_config.unit,
                    "reward_objective_scale": float(getattr(getattr(envs[0], "unwrapped", envs[0]), "reward_objective_scale", 1.0)),
                    "train_avg_best_objective": train_summary.get("train_avg_best_objective"),
                    "terminal_outcome_reason_counts": json.dumps(
                        train_summary.get("terminal_outcome_reason_counts", {}),
                        sort_keys=True,
                    ),
                    **{field: eval_row.get(field) for field in OBJECTIVE_EVAL_FIELDS},
                    "reward_pbrs_customer_mean": reward_pbrs_customer_mean,
                    "reward_pbrs_repair_distance_mean": reward_pbrs_repair_distance_mean,
                    "reward_terminal_heuristic_mean": reward_terminal_heuristic_mean,
                    "reward_terminal_task_total_mean": reward_terminal_task_total_mean,
                    "reward_terminal_success_bonus_mean": (
                        reward_terminal_success_bonus_mean
                    ),
                    "reward_terminal_failure_base_mean": reward_terminal_failure_base_mean,
                    "reward_terminal_unserved_mean": reward_terminal_unserved_mean,
                    "reward_pbrs_total_mean": reward_pbrs_total_mean,
                    "reward_shaping_total_mean": reward_shaping_total_mean,
                    "reward_base_abs_mean": reward_base_abs_mean,
                    "reward_distance_abs_mean": reward_distance_abs_mean,
                    "reward_pbrs_total_abs_mean": reward_pbrs_total_abs_mean,
                    "reward_pbrs_to_base_abs_ratio": reward_pbrs_to_base_abs_ratio,
                    "reward_pbrs_to_distance_abs_ratio": reward_pbrs_to_distance_abs_ratio,
                    "reward_base_per_trajectory": (
                        reward_diagnostics.get("base_sum", 0.0)
                        / max(trajectory_count, 1)
                    ),
                    "reward_distance_per_trajectory": (
                        reward_diagnostics.get("distance_sum", 0.0)
                        / max(trajectory_count, 1)
                    ),
                    "reward_pbrs_total_per_trajectory": (
                        reward_diagnostics.get("pbrs_total_sum", 0.0)
                        / max(trajectory_count, 1)
                    ),
                    "reward_terminal_heuristic_per_trajectory": (
                        reward_diagnostics.get("terminal_heuristic_sum", 0.0)
                        / max(trajectory_count, 1)
                    ),
                    "reward_terminal_task_total_per_trajectory": (
                        reward_diagnostics.get("terminal_task_total_sum", 0.0)
                        / max(trajectory_count, 1)
                    ),
                    "reward_terminal_success_bonus_per_trajectory": (
                        reward_diagnostics.get("terminal_success_bonus_sum", 0.0)
                        / max(trajectory_count, 1)
                    ),
                    "reward_discounted_distance_per_trajectory": (
                        reward_diagnostics.get("distance_discounted_sum", 0.0)
                        / max(trajectory_count, 1)
                    ),
                    "reward_discounted_pbrs_total_per_trajectory": (
                        reward_diagnostics.get("pbrs_total_discounted_sum", 0.0)
                        / max(trajectory_count, 1)
                    ),
                    "reward_discounted_terminal_heuristic_per_trajectory": (
                        reward_diagnostics.get(
                            "terminal_heuristic_discounted_sum", 0.0
                        )
                        / max(trajectory_count, 1)
                    ),
                    "reward_discounted_terminal_task_total_per_trajectory": (
                        reward_diagnostics.get(
                            "terminal_task_total_discounted_sum", 0.0
                        )
                        / max(trajectory_count, 1)
                    ),
                    "reward_discounted_terminal_success_bonus_per_trajectory": (
                        reward_diagnostics.get(
                            "terminal_success_bonus_discounted_sum", 0.0
                        )
                        / max(trajectory_count, 1)
                    ),
                    "customer_action_reward_base_mean": (
                        reward_diagnostics.get(
                            "base_customer_action_sum", 0.0
                        )
                        / customer_action_count
                    ),
                    "customer_action_reward_pbrs_total_mean": (
                        reward_diagnostics.get(
                            "pbrs_total_customer_action_sum", 0.0
                        )
                        / customer_action_count
                    ),
                    "noncustomer_action_reward_pbrs_total_mean": (
                        reward_diagnostics.get(
                            "pbrs_total_noncustomer_action_sum", 0.0
                        )
                        / noncustomer_action_count
                    ),
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
                    "successful_trajectory_count": outcome_summary[
                        "success_count"
                    ],
                    "successful_trajectory_rate": outcome_summary["success_rate"],
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
                    "non_horizon_infeasible_count": outcome_summary[
                        "non_horizon_infeasible_count"
                    ],
                    "non_horizon_infeasible_rate": outcome_summary[
                        "non_horizon_infeasible_rate"
                    ],
                    "non_horizon_infeasible_reason_counts": json.dumps(
                        outcome_summary["non_horizon_infeasible_reason_counts"],
                        sort_keys=True,
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
                    **{
                        field: eval_row.get(field, "")
                        for field in EVAL_OUTCOME_FIELDS
                    },
                }
            )
            f.flush()
            diagnostics_start = time.perf_counter()
            diagnostic_record = build_epoch_reward_diagnostics(
                cfg=cfg, epoch=epoch, session_id=diagnostic_session_id,
                start_epoch=start_epoch, resume_from=resume_checkpoint,
                rollout_records=rollout_records, raw_advantages=all_advantages,
                base_reward_stats=base_reward_stats, reward_components=reward_diagnostics,
                normalization_records=normalization_records, preclip_norms=preclip_norms,
                optimizer_steps_total=optimizer_steps_total, pbrs_scale=pbrs_scale,
            )
            diagnostic_record["warm_start"] = protocol_cfg.get("warm_start")
            diagnostic_record["diagnostics_compute_wall_time_s"] = time.perf_counter() - diagnostics_start
            append_jsonl(reward_diagnostics_path, diagnostic_record)
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
                source_epoch = warm_start_accounting_source_epoch
                local_completed_epochs = epoch - source_epoch
                completed = local_completed_epochs // protocol_epochs
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
