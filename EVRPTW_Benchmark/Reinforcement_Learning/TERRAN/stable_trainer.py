"""Cost-preserving PPO for TERRAN, with independent value optimization.

The legacy benchmark trainer remains the default. This opt-in engine uses the
same constructive policy, data adapter and independent solution verifier.
Rollouts are frozen on CPU; only one physical environment/time chunk occupies
the accelerator. Economic targets are always positive USD return-to-go.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ..common.objective import resolve_objective
from ..common.training_protocol import append_jsonl, atomic_json
from .models import Agent
from .models.attention_model_wrapper import STATIC_OBSERVATION_KEYS, STABLE_DYNAMIC_OBSERVATION_KEYS
from .rollout import collect_rollout

SCHEMA = "terran_stable_cost_v1"
BATCH_RESUME_FIELDS = ("num_envs_per_gpu", "effective_batch_size",
                       "logical_microbatches_per_epoch", "ppo_step_chunk_size")


@dataclass
class StableState:
    epoch: int = 0
    phase: str = "feasibility"
    lambda_usd: float = 0.0
    actor_scale: float = 1.0
    scale_phase: str = ""
    success_streak: int = 0
    sample_count: int = 0
    transitions: int = 0
    actor_steps: int = 0
    best_feasible_rate: float = -1.0
    best_cost: float = math.inf


def resolve_stable_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(config)
    training = cfg.setdefault("training", {})
    if training.get("algorithm") != "stable_cost_v1":
        raise ValueError("stable trainer requires training.algorithm=stable_cost_v1")
    if float(training.get("gamma", 1.0)) != 1.0:
        raise ValueError("stable_cost_v1 requires gamma=1")
    if cfg.get("reward_contract"):
        raise ValueError("stable_cost_v1 cannot reuse a legacy reference reward contract")
    objective = resolve_objective(cfg.get("objective"))
    if not objective.is_cost:
        raise ValueError("stable_cost_v1 requires the energy_vehicle_cost objective")
    cfg["objective"] = objective.to_dict()
    model = cfg.setdefault("model", {})
    if model.get("critic_mode", "stable_cost_v1") != "stable_cost_v1":
        raise ValueError("stable training and model critic_mode disagree")
    model["critic_mode"] = "stable_cost_v1"
    for name, value in cfg.get("pbrs", {}).items():
        if name.startswith("use_") and value:
            raise ValueError(f"stable_cost_v1 requires pbrs.{name}=false")
    cfg["pbrs"] = {"use_customer_pbrs": False, "use_repair_distance_pbrs": False,
                   "use_feasible_ratio_pbrs": False, "use_terminal_heuristic": False,
                   "use_terminal_task_penalty": False, "terminal_success_bonus": 0.0}
    cfg.setdefault("env", {}).update(training_mode="stable_cost_v1", normalize_reward=False,
                                      success_bonus=0.0, invalid_action_penalty=0.0,
                                      reward_distance_scale_mode="single_customer_repair_median")
    cfg["env"].pop("reward_objective_scale", None)
    cfg["normalization"] = {"reward_unit": "USD", "reward_divisor": 1.0,
                              "critic": "shared_popart", "schema": SCHEMA}
    defaults = dict(popart_beta=0.01, popart_min_std=1.0, failure_target=0.01,
                    dual_lr=0.05, dual_initial=1.0, dual_max=1000.0,
                    warmup_success_rate=0.8, warmup_min_epochs=3, warmup_max_epochs=200,
                    actor_scale_floor=1.0, target_kl=0.02, actor_kl_backtracks=3, critic_grad_norm=1.0,
                    log_interval=1)
    stable = {**defaults, **cfg.get("stable_cost", {})}
    for name, value in stable.items():
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid stable_cost.{name}: {value!r}")
    for name in ("failure_target", "warmup_success_rate"):
        if not 0 <= stable[name] <= 1:
            raise ValueError(f"stable_cost.{name} must lie in [0,1]")
    for name in ("popart_min_std", "actor_scale_floor", "critic_grad_norm", "target_kl"):
        if stable[name] <= 0:
            raise ValueError(f"stable_cost.{name} must be positive")
    if not 0 < stable["popart_beta"] <= 1 or stable["dual_initial"] > stable["dual_max"]:
        raise ValueError("invalid PopArt rate or dual bounds")
    if stable["warmup_min_epochs"] < 1 or stable["warmup_max_epochs"] < stable["warmup_min_epochs"]:
        raise ValueError("invalid feasibility curriculum duration")
    cfg["stable_cost"] = stable
    cfg.setdefault("data", {})["stage2_record_sample_ids"] = True
    for name in ("epochs", "num_envs_per_gpu", "n_traj", "rollout_steps",
                 "logical_microbatches_per_epoch", "ppo_step_chunk_size", "ppo_update_epochs"):
        value = int(training.get(name, 1))
        if value < 1:
            raise ValueError(f"training.{name} must be positive")
        training[name] = value
    if training["n_traj"] < 2:
        raise ValueError("feasibility LOO baseline requires n_traj>=2")
    effective = training["num_envs_per_gpu"] * training["logical_microbatches_per_epoch"]
    if int(training.get("effective_batch_size", effective)) != effective:
        raise ValueError("effective_batch_size disagrees with physical batch and microbatch count")
    training["effective_batch_size"] = effective
    if int(training.get("num_minibatches", 1)) != 1 or int(training.get("gradient_accumulation_steps", 1)) != 1:
        raise ValueError("stable_cost_v1 uses logical_microbatches_per_epoch for one logical optimizer batch")
    for name, default in (("learning_rate", 1e-4), ("critic_learning_rate", 1e-4),
                          ("max_grad_norm", 1.0), ("clip_coef", 0.2)):
        value = float(training.get(name, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{name} must be finite and positive")
        training[name] = value
    for name, default in (("ent_coef", 0.01), ("weight_decay", 0.01)):
        value = float(training.get(name, default))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"training.{name} must be finite and non-negative")
        training[name] = value
    training["gamma"] = 1.0
    training["reward_contract_id"] = SCHEMA
    if not cfg.get("output_dir"):
        raise ValueError("stable training requires a dedicated output_dir")
    if training.get("resume_checkpoint") and training.get("actor_warm_start"):
        raise ValueError("resume and actor warm start are mutually exclusive")
    if not isinstance(training.get("allow_batch_resize_resume", False), bool):
        raise ValueError("training.allow_batch_resize_resume must be boolean")
    if training.get("allow_batch_resize_resume") and not training.get("resume_checkpoint"):
        raise ValueError("allow_batch_resize_resume requires a resume checkpoint")
    return cfg


def config_signature(cfg: dict[str, Any]) -> str:
    """Exclude locations/schedules that do not alter the next training update."""
    payload = {k: deepcopy(cfg.get(k, {})) for k in ("objective", "model", "training", "stable_cost", "env", "data")}
    for key in ("resume_checkpoint", "actor_warm_start", "checkpoint_interval", "debug",
                "allow_batch_resize_resume"):
        payload["training"].pop(key, None)
    payload["data"].pop("stage2_completed_samples", None)
    payload["data"].pop("stage2_completed_data_passes", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def validate_resume_checkpoint(payload: dict, cfg: dict, *, seed: int,
                               source: str | Path) -> dict[str, Any]:
    """Validate stored provenance before considering the explicit batch exception.

    Hash the checkpoint's saved configuration directly: re-resolving it first
    could hide a stale or altered configuration. Data sampler cursors are
    intentionally excluded by config_signature and restored from StableState.
    """
    original = payload.get("config")
    original_signature = payload.get("training_signature")
    if (payload.get("schema") != SCHEMA or not isinstance(original, dict)
            or config_signature(original) != original_signature):
        raise ValueError("stable checkpoint original configuration signature is invalid")
    if payload.get("seed") != seed:
        raise ValueError("stable checkpoint seed mismatch")
    signature = config_signature(cfg)
    changed = {key: {"old": original["training"].get(key), "new": cfg["training"].get(key)}
               for key in BATCH_RESUME_FIELDS
               if original["training"].get(key) != cfg["training"].get(key)}
    allowed = bool(cfg["training"].get("allow_batch_resize_resume", False))
    if original_signature != signature:
        if not allowed:
            raise ValueError("stable checkpoint/config training signature mismatch; "
                             "batch changes require explicit allow_batch_resize_resume")
        comparable_old, comparable_new = deepcopy(original), deepcopy(cfg)
        for comparable in (comparable_old, comparable_new):
            for key in BATCH_RESUME_FIELDS:
                comparable["training"].pop(key, None)
        if config_signature(comparable_old) != config_signature(comparable_new):
            raise ValueError("batch resize resume permits only changes to "
                             + ", ".join(BATCH_RESUME_FIELDS))
    return {"source_checkpoint": str(Path(source).resolve()),
            "source_epoch": payload["stable_state"]["epoch"],
            "source_sample_count": payload["stable_state"]["sample_count"],
            "source_training_signature": original_signature,
            "training_signature": signature, "allow_batch_resize_resume": allowed,
            "changed_batch_fields": changed,
            "old_batch_geometry": {key: original["training"].get(key) for key in BATCH_RESUME_FIELDS},
            "new_batch_geometry": {key: cfg["training"].get(key) for key in BATCH_RESUME_FIELDS},
            "optimizer_reset": False}


def load_actor_weights(agent: Agent, checkpoint: str | Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = payload["model_state_dict"]
    target = agent.state_dict()
    copied, skipped = {}, []
    for key, tensor in source.items():
        if key.startswith("critic."):
            skipped.append(key)
            continue
        if key not in target or target[key].shape != tensor.shape:
            raise ValueError(f"incompatible actor checkpoint tensor: {key}")
        copied[key] = tensor
    if not copied:
        raise ValueError("checkpoint contains no compatible TERRAN actor weights")
    missing = [key for key in target if not key.startswith("critic.") and key not in copied]
    if any("remaining" not in key and "stable" not in key for key in missing):
        raise ValueError(f"actor warm start missing unexpected parameters: {missing}")
    agent.load_state_dict(copied, strict=False)
    return {"path": str(Path(checkpoint).resolve()), "source_epoch": payload.get("epoch"),
            "copied_actor_tensors": len(copied), "new_actor_tensors": missing,
            "discarded_critic_tensors": len(skipped), "optimizer_reset": True}


def trajectory_state_weights(valid: torch.Tensor) -> torch.Tensor:
    lengths = valid.sum(dim=0).clamp_min(1)
    return valid.float() / lengths.unsqueeze(0)


def make_advantages(batch, state: StableState) -> torch.Tensor:
    if state.phase == "feasibility":
        score = batch.terminal_failure.float() + batch.unserved_fraction
        baseline = (score.sum(dim=-1, keepdim=True) - score) / (score.shape[-1] - 1)
        advantage = -(score - baseline).unsqueeze(0).expand_as(batch.cost_returns)
    else:
        advantage = -(batch.cost_returns - batch.old_cost_values) - state.lambda_usd * (
            batch.failure_returns - batch.old_failure_values)
    return torch.where(batch.valid, advantage, 0.0)


def update_dual(state: StableState, failure_rate: float, cfg: dict, cost_std: float) -> None:
    if state.phase != "cost":
        return
    settings = cfg["stable_cost"]
    # Units remain USD even as the preconditioner changes. No per-N table.
    vehicle_unit = max(resolve_objective(cfg["objective"]).vehicle_unit_cost, 1.0)
    step_scale = max(vehicle_unit, float(cost_std))
    state.lambda_usd = float(np.clip(
        state.lambda_usd + settings["dual_lr"] * step_scale * (failure_rate - settings["failure_target"]),
        0.0, settings["dual_max"] * vehicle_unit))


def _atomic_checkpoint(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def _model_obs(observation: dict, device, static: dict | None = None) -> dict:
    if static is None:
        static = {key: torch.as_tensor(observation[key], device=device)
                  for key in STATIC_OBSERVATION_KEYS if key in observation}
    return {**static, **{key: torch.as_tensor(observation[key], device=device)
                         for key in STABLE_DYNAMIC_OBSERVATION_KEYS if key in observation}}


def loss_chunk(agent, batch, advantages, cfg, start: int, end: int, *,
               trajectory_denominator: int, time_unit: int):
    """A chunk contributes to one logical update; no optimizer steps inside.

    Policy sums over actual actions, with one fixed budget-based numerical
    factor shared by all trajectories. Critic averages within each trajectory.
    Neither policy advantage nor return is divided by its trajectory's length.
    """
    device = next(agent.parameters()).device
    static = {key: torch.as_tensor(batch.observations[0][key], device=device)
              for key in STATIC_OBSERVATION_KEYS if key in batch.observations[0]}
    cache = agent.backbone.encode(_model_obs(batch.observations[0], device, static))
    weights = trajectory_state_weights(batch.valid)
    policy_parts, critic_parts, entropy_parts = [], [], []
    diagnostics = dict(kl_sum=0.0, clip_sum=0.0, count=0, cost_sq_sum=0.0,
                       failure_brier_sum=0.0)
    for step in range(start, end):
        valid = batch.valid[step].to(device)
        if not valid.any():
            continue
        obs = _model_obs(batch.observations[step], device, static)
        actions = batch.actions[step].to(device)
        result = agent.get_action_and_value_cached(obs, action=actions, state=cache,
                                                    return_critic_outputs=True)
        _, logprob, entropy, _, _, values = result
        new_logprob = logprob[valid]
        old_logprob = batch.old_logprobs[step].to(device)[valid]
        adv = advantages[step].to(device)[valid]
        logratio = new_logprob - old_logprob
        ratio = logratio.exp()
        clip = cfg["training"]["clip_coef"]
        policy_parts.append(-torch.minimum(ratio * adv, ratio.clamp(1-clip, 1+clip) * adv).sum())
        entropy_parts.append(entropy[valid].sum())
        target = batch.cost_returns[step].to(device)[valid]
        cost_prediction = values["cost_value"][valid]
        # Both tensors are in USD; normalizing only the error is equivalent to
        # comparing normalized targets and the output-preserving PopArt head.
        std = agent.critic.popart.std.to(cost_prediction)
        value_error = (cost_prediction - target) / std
        failure = batch.failure_returns[step].to(device)[valid]
        failure_logit = values["failure_logits"][valid]
        loss_values = value_error.square() + F.binary_cross_entropy_with_logits(
            failure_logit, failure, reduction="none")
        weight = weights[step].to(device)[valid]
        critic_parts.append((loss_values * weight).sum() / trajectory_denominator)
        with torch.no_grad():
            diagnostics["kl_sum"] += float(((ratio - 1) - logratio).sum())
            diagnostics["clip_sum"] += float(((ratio - 1).abs() > clip).sum())
            diagnostics["count"] += int(valid.sum())
            diagnostics["cost_sq_sum"] += float((cost_prediction - target).square().sum())
            diagnostics["failure_brier_sum"] += float((failure_logit.sigmoid() - failure).square().sum())
    if not policy_parts:
        return None
    policy_loss = torch.stack(policy_parts).sum() / (trajectory_denominator * time_unit)
    entropy = torch.stack(entropy_parts).sum() / (trajectory_denominator * time_unit)
    critic_loss = torch.stack(critic_parts).sum()
    total = policy_loss - cfg["training"]["ent_coef"] * entropy + critic_loss
    return total, policy_loss.detach(), critic_loss.detach(), entropy.detach(), diagnostics


@torch.no_grad()
def rollout_policy_kl(agent, records) -> float:
    """Empirical KL on every stored active action, with one cache per instance."""
    device = next(agent.parameters()).device
    total, count = 0.0, 0
    for batch in records:
        static = {key: torch.as_tensor(batch.observations[0][key], device=device)
                  for key in STATIC_OBSERVATION_KEYS if key in batch.observations[0]}
        cache = agent.backbone.encode(_model_obs(batch.observations[0], device, static))
        for step, observation in enumerate(batch.observations):
            valid = batch.valid[step].to(device)
            if not valid.any():
                continue
            logits, _ = agent.backbone.decode(_model_obs(observation, device, static), cache)
            logprob = torch.distributions.Categorical(logits=logits).log_prob(batch.actions[step].to(device))
            diff = logprob[valid] - batch.old_logprobs[step].to(device)[valid]
            total += float((diff.exp() - 1 - diff).sum())
            count += int(valid.sum())
    return total / max(count, 1)


def checked_actor_step(agent, optimizer, records, cfg) -> tuple[bool, float, int]:
    """Backtrack an Adam proposal without losing its pre-update moment state.

    This is an empirical stored-sample guard, not a formal trust-region proof.
    Rejected proposals never change the critic or the actual rollout buffers.
    """
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    before = [p.detach().clone() for p in parameters]
    optimizer_before = deepcopy(optimizer.state_dict())
    learning_rates = [group["lr"] for group in optimizer.param_groups]
    attempts = int(cfg["stable_cost"]["actor_kl_backtracks"]) + 1
    for attempt in range(attempts):
        if attempt:
            with torch.no_grad():
                for parameter, original in zip(parameters, before):
                    parameter.copy_(original)
            optimizer.load_state_dict(deepcopy(optimizer_before))
        for group, rate in zip(optimizer.param_groups, learning_rates):
            group["lr"] = rate * (0.5 ** attempt)
        optimizer.step()
        kl = rollout_policy_kl(agent, records)
        if math.isfinite(kl) and kl <= cfg["stable_cost"]["target_kl"]:
            return True, kl, attempt
    with torch.no_grad():
        for parameter, original in zip(parameters, before):
            parameter.copy_(original)
    optimizer.load_state_dict(optimizer_before)
    for group, rate in zip(optimizer.param_groups, learning_rates):
        group["lr"] = rate * (0.5 ** attempts)
    return False, rollout_policy_kl(agent, records), attempts


def optimize_rollouts(agent, actor_optimizer, critic_optimizer, records, cfg, state):
    settings, training = cfg["stable_cost"], cfg["training"]
    raw_advantages = [make_advantages(batch, state) for batch in records]
    count = sum(int(batch.valid.sum()) for batch in records)
    if count == 0:
        raise RuntimeError("logical rollout contains no active actions; check initial instance feasibility")
    rms = math.sqrt(sum(float(adv[batch.valid].double().square().sum())
                        for batch, adv in zip(records, raw_advantages)) / max(count, 1))
    floor = 1e-3 if state.phase == "feasibility" else settings["actor_scale_floor"]
    if state.scale_phase != state.phase:
        state.actor_scale = max(floor, rms)
        state.scale_phase = state.phase
    used_scale = max(floor, state.actor_scale)
    advantages = [adv / used_scale for adv in raw_advantages]
    targets = torch.cat([batch.cost_returns.flatten() for batch in records])
    weights = torch.cat([trajectory_state_weights(batch.valid).flatten() for batch in records])
    agent.critic.popart.update_stats(targets, mask=weights > 0, weights=weights,
                                     optimizer=critic_optimizer)
    del targets, weights, raw_advantages
    trajectory_count = sum(int(batch.valid.any(dim=0).sum()) for batch in records)
    actor_parameters = [p for name, p in agent.named_parameters() if not name.startswith("critic.")]
    critic_parameters = list(agent.critic.parameters())
    metrics = dict(actor_loss=0.0, critic_loss=0.0, entropy=0.0, actor_grad_norm=0.0,
                   critic_grad_norm=0.0, approx_kl=0.0, clip_fraction=0.0,
                   cost_rmse_usd=0.0, failure_brier=0.0, ppo_updates=0, actor_backtracks=0,
                   advantage_rms_raw=rms, actor_scale_used=used_scale,
                   zero_advantage_fraction=sum(int((adv[batch.valid] == 0).sum())
                       for batch, adv in zip(records, advantages)) / max(count, 1))
    for update in range(training["ppo_update_epochs"]):
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        diag = dict(kl_sum=0.0, clip_sum=0.0, count=0, cost_sq_sum=0.0, failure_brier_sum=0.0)
        update_losses = [0.0, 0.0, 0.0]
        for batch, advantage in zip(records, advantages):
            steps = batch.actions.shape[0]
            for start in range(0, steps, training["ppo_step_chunk_size"]):
                result = loss_chunk(agent, batch, advantage, cfg, start,
                                    min(steps, start + training["ppo_step_chunk_size"]),
                                    trajectory_denominator=trajectory_count,
                                    time_unit=training["rollout_steps"])
                if result is None:
                    continue
                loss, policy, critic, entropy, chunk_diag = result
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite TERRAN stable PPO loss")
                loss.backward()
                for i, value in enumerate((policy, critic, entropy)):
                    update_losses[i] += float(value)
                for key, value in chunk_diag.items():
                    diag[key] += value
                del loss, result
        kl = diag["kl_sum"] / max(diag["count"], 1)
        # Refuse further actor reuse once the stored-policy KL is already high.
        # The first update is evaluated at the rollout policy (KL ~= 0).
        actor_norm = torch.nn.utils.clip_grad_norm_(actor_parameters, training["max_grad_norm"],
                                                    error_if_nonfinite=True)
        critic_norm = torch.nn.utils.clip_grad_norm_(critic_parameters, settings["critic_grad_norm"],
                                                     error_if_nonfinite=True)
        accepted, post_kl, backtracks = checked_actor_step(agent, actor_optimizer, records, cfg)
        metrics["actor_backtracks"] += backtracks
        if accepted:
            state.actor_steps += 1
            metrics["ppo_updates"] += 1
        critic_optimizer.step()
        metrics.update(actor_loss=update_losses[0], critic_loss=update_losses[1], entropy=update_losses[2],
                       actor_grad_norm=float(actor_norm), critic_grad_norm=float(critic_norm),
                       approx_kl=post_kl, pre_update_kl=kl,
                       actor_learning_rate=actor_optimizer.param_groups[0]["lr"],
                       clip_fraction=diag["clip_sum"] / max(diag["count"], 1),
                       cost_rmse_usd=math.sqrt(diag["cost_sq_sum"] / max(diag["count"], 1)),
                       failure_brier=diag["failure_brier_sum"] / max(diag["count"], 1))
        if not accepted:
            break
    # Statistics affect the next logical batch, never this batch's PPO reuse.
    state.actor_scale = max(floor, 0.9 * state.actor_scale + 0.1 * rms)
    return metrics


def train_stable_cost(config: dict[str, Any], seed: int, device=None) -> Path:
    from .trainer import make_envs, evaluate_fixed_dataset, set_seed, validation_summary_from_eval_row
    cfg = resolve_stable_config(config)
    training, settings = cfg["training"], cfg["stable_cost"]
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "checkpoint_latest.pt"
    resume_path = training.get("resume_checkpoint")
    if not resume_path and any((output / name).exists() for name in
                              ("checkpoint_latest.pt", "metrics.jsonl", "training_contract.json")):
        raise FileExistsError(f"existing run requires explicit resume: {output}")
    source_index = cfg.get("data", {}).get("stage2_dataset_path")
    if source_index:
        actual = hashlib.sha256(Path(source_index).read_bytes()).hexdigest()
        expected = cfg["data"].get("training_index_sha256")
        if expected and actual != expected:
            raise ValueError("training index changed since the configuration was prepared")
        cfg["data"]["training_index_sha256"] = actual
    signature = config_signature(cfg)
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = cfg["model"]
    agent = Agent(embedding_dim=int(model.get("embedding_dim", 256)),
                  n_encode_layers=int(model.get("n_encode_layers", 3)),
                  tanh_clipping=float(model.get("tanh_clipping", 15.0)), device=device,
                  critic_mode="stable_cost_v1", popart_beta=settings["popart_beta"],
                  popart_min_std=settings["popart_min_std"]).to(device)
    actor_parameters = [p for name, p in agent.named_parameters() if not name.startswith("critic.")]
    actor_optimizer = torch.optim.AdamW(actor_parameters, lr=training["learning_rate"], eps=1e-5,
                                         weight_decay=training["weight_decay"])
    critic_optimizer = torch.optim.AdamW(agent.critic.parameters(), lr=training["critic_learning_rate"],
                                          eps=1e-5, weight_decay=training["weight_decay"])
    vehicle_unit = max(resolve_objective(cfg["objective"]).vehicle_unit_cost, 1.0)
    state = StableState(lambda_usd=settings["dual_initial"] * vehicle_unit)
    initialization = None
    resume = None
    if resume_path:
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        resume = validate_resume_checkpoint(payload, cfg, seed=seed, source=resume_path)
        agent.load_state_dict(payload["model_state_dict"])
        actor_optimizer.load_state_dict(payload["optimizer_state_dict"])
        critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
        state = StableState(**payload["stable_state"])
        initialization = payload.get("initialization")
        cfg["data"]["stage2_completed_samples"] = state.sample_count
        del payload
    elif training.get("actor_warm_start"):
        initialization = load_actor_weights(agent, training["actor_warm_start"])
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    atomic_json(output / "training_contract.json", {"schema": SCHEMA, "training_signature": signature,
        "seed": seed, "objective": cfg["objective"], "reward_unit": "USD", "gamma": 1.0,
        "initialization": initialization, "resume": resume,
        "sample_mode": "seeded_shuffle_cycle_without_replacement",
        "effective_instances": training["num_envs_per_gpu"] * training["logical_microbatches_per_epoch"],
        "policy_time_unit": training["rollout_steps"], "critic_weighting": "trajectory_mean",
        "popart_optimizer_state": "reset_output_layer_moments_on_stats_update"})
    envs, pool = make_envs(cfg, seed)
    started = time.perf_counter()
    progress_path = output / "progress.json"
    metrics_path = output / "metrics.jsonl"
    atomic_json(progress_path, {"status": "running", "epoch": state.epoch, "pid": os.getpid()})

    def save():
        _atomic_checkpoint(latest, {"schema": SCHEMA, "training_signature": signature,
            "epoch": state.epoch, "seed": seed, "config": cfg,
            "model_state_dict": agent.state_dict(), "optimizer_state_dict": actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": critic_optimizer.state_dict(), "stable_state": asdict(state),
            "initialization": initialization, "resume": resume})

    try:
        for epoch in range(state.epoch + 1, training["epochs"] + 1):
            epoch_start = time.perf_counter()
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            agent.train()
            records = []
            set_seed(seed + epoch * 100_000)
            collect_start = time.perf_counter()
            for microbatch in range(training["logical_microbatches_per_epoch"]):
                batch = collect_rollout(agent, envs, rollout_steps=training["rollout_steps"],
                    decode_mode="sample", device=device, seed=seed + epoch * 100_000 + microbatch,
                    cache_static_embeddings=True, reward_discount_factor=1.0, storage_device="cpu",
                    collect_reward_diagnostics=False)
                if batch.cost_returns is None or batch.old_failure_values is None:
                    raise RuntimeError("stable rollout fields missing")
                records.append(batch)
                atomic_json(progress_path, {"status": "running", "pid": os.getpid(), "epoch": epoch,
                    "stage": "collect", "phase": state.phase, "microbatch": microbatch + 1,
                    "microbatches": training["logical_microbatches_per_epoch"],
                    "epoch_wall_s": time.perf_counter() - epoch_start})
                if epoch == 1 and (microbatch == 0 or (microbatch + 1) % 8 == 0):
                    print(f"[StableTERRAN] collecting epoch={epoch} microbatch={microbatch+1}/"
                          f"{training['logical_microbatches_per_epoch']}", flush=True)
            collect_wall_s = time.perf_counter() - collect_start
            failure_rate = float(torch.cat([batch.terminal_failure.flatten() for batch in records]).float().mean())
            unserved = float(torch.cat([batch.unserved_fraction.flatten() for batch in records]).mean())
            phase_used, lambda_used = state.phase, state.lambda_usd
            atomic_json(progress_path, {"status": "running", "pid": os.getpid(), "epoch": epoch,
                                        "stage": "update", "phase": state.phase})
            update_start = time.perf_counter()
            metrics = optimize_rollouts(agent, actor_optimizer, critic_optimizer, records, cfg, state)
            update_wall_s = time.perf_counter() - update_start
            state.epoch = epoch
            state.sample_count = int(pool.sample_count)
            if hasattr(pool, "drain_sampled_view_ids"):
                sampled_ids = pool.drain_sampled_view_ids()
                append_jsonl(output / "sampled_view_ids.jsonl", {"epoch": epoch,
                    "start_cursor": state.sample_count - len(sampled_ids),
                    "end_cursor": state.sample_count, "view_ids": sampled_ids})
            valid_transitions = sum(int(batch.valid.sum()) for batch in records)
            state.transitions += valid_transitions
            popart_std = float(agent.critic.popart.std)
            update_dual(state, failure_rate, cfg, popart_std)
            if state.phase == "feasibility":
                state.success_streak = state.success_streak + 1 if 1-failure_rate >= settings["warmup_success_rate"] else 0
                if state.success_streak >= settings["warmup_min_epochs"]:
                    state.phase = "cost"
            cost_means = [float(batch.cost_returns[0].mean()) for batch in records]
            effective_instances = sum(batch.actions.shape[1] for batch in records)
            epoch_wall_s = time.perf_counter() - epoch_start
            rollout_timings: dict[str, float] = {}
            for record in records:
                for key, value in record.timings.items():
                    rollout_timings[key] = rollout_timings.get(key, 0.0) + float(value)
            row = {"schema": SCHEMA, "epoch": epoch, "phase": phase_used, "next_phase": state.phase,
                   "failure_rate": failure_rate, "success_rate": 1-failure_rate, "unserved_fraction": unserved,
                   "cost_accrued_mean_usd": float(np.mean(cost_means)), "lambda_usd": lambda_used,
                   "lambda_next_usd": state.lambda_usd, "popart_std_usd": popart_std,
                   "effective_instances": effective_instances,
                   "n_traj": training["n_traj"], "samples_seen": state.sample_count,
                   "transitions": state.transitions, "epoch_wall_s": epoch_wall_s,
                   "collect_wall_s": collect_wall_s, "update_wall_s": update_wall_s,
                   "instances_per_s": effective_instances / max(epoch_wall_s, 1e-9),
                   "valid_transitions_per_s": valid_transitions / max(epoch_wall_s, 1e-9),
                   "wall_s": time.perf_counter()-started, **rollout_timings, **metrics}
            if str(device).startswith("cuda"):
                row.update(cuda_peak_allocated_mb=torch.cuda.max_memory_allocated(device) / 2**20,
                           cuda_peak_reserved_mb=torch.cuda.max_memory_reserved(device) / 2**20)
            append_jsonl(metrics_path, row)
            atomic_json(progress_path, {"status": "running", "pid": os.getpid(), **row})
            print(f"[StableTERRAN] epoch={epoch} phase={phase_used} success={1-failure_rate:.4f} "
                  f"cost={row['cost_accrued_mean_usd']:.2f}USD critic={metrics['critic_loss']:.4f} "
                  f"KL={metrics['approx_kl']:.5f} wall={row['epoch_wall_s']:.1f}s", flush=True)
            del records, batch
            eval_cfg = cfg.get("evaluation", {})
            interval = int(eval_cfg.get("eval_interval", 0))
            if interval and epoch % interval == 0:
                eval_start = time.perf_counter()
                eval_row = evaluate_fixed_dataset(agent, cfg, seed, epoch, device)
                validation = validation_summary_from_eval_row(eval_row, objective_config=resolve_objective(cfg["objective"]),
                    logical_epoch=epoch, validation_seed=int(eval_cfg.get("eval_seed", seed+910_000_000)),
                    validation_wall_time_s=time.perf_counter()-eval_start)
                if validation is None:
                    raise RuntimeError(f"configured stable validation did not run: {eval_row}")
                append_jsonl(output / "validation_history.jsonl", validation)
                atomic_json(output / "validation_summary.json", validation)
                rate = float(validation["complete_and_feasible_rate"])
                cost = validation.get("mean_verified_objective_cost_usd")
                value = float(cost) if cost is not None else math.inf
                if (rate, -value) > (state.best_feasible_rate, -state.best_cost):
                    state.best_feasible_rate, state.best_cost = rate, value
                    save()
                    shutil.copy2(latest, output / "best.ckpt")
            if epoch == 1 or epoch % int(training.get("checkpoint_interval", 25)) == 0 or epoch == training["epochs"]:
                save()
            if state.phase == "feasibility" and epoch >= settings["warmup_max_epochs"]:
                save()
                raise RuntimeError("feasibility curriculum did not meet its success threshold; cost optimization was not enabled")
        save()
        atomic_json(progress_path, {"status": "complete", "pid": os.getpid(), "epoch": state.epoch})
    except BaseException as error:
        # Preserve the last committed checkpoint; an interrupted gradient update
        # is deliberately never advertised as a resumable epoch.
        atomic_json(progress_path, {"status": "failed", "pid": os.getpid(), "epoch": state.epoch,
                                    "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        for env in envs:
            env.close()
        pool.close()
    return latest
