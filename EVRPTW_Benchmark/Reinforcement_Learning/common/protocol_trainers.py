from __future__ import annotations

import json
import math
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.stats import ttest_rel

from .training_protocol import (
    append_jsonl,
    atomic_json,
    grouped_batches,
    load_state,
    make_validation_pool,
    require_registered_batches,
    validation_key,
    verified_validation,
)


def _customer_count(scale: str) -> int:
    value = str(scale).lower().removeprefix("cus")
    if not value.isdigit() or int(value) <= 0:
        raise ValueError(f"invalid scale: {scale}")
    return int(value)


def _peak_gpu_bytes(device: str) -> int:
    return int(torch.cuda.max_memory_allocated(device)) if str(device).startswith("cuda") else 0


def _save_checkpoint(
    path: Path,
    *,
    method: str,
    data_pass: int,
    policy: torch.nn.Module,
    baseline: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: Any,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "method": method,
        "data_pass": int(data_pass),
        "model": policy.state_dict(),
        "baseline": baseline.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "protocol_id": args.protocol_id,
    }
    payload.update(extra or {})
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    *,
    policy: torch.nn.Module,
    baseline: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    protocol_id: str,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=policy.device, weights_only=False)
    if payload.get("protocol_id") != protocol_id:
        raise ValueError("checkpoint protocol does not match requested protocol")
    policy.load_state_dict(payload["model"])
    baseline.load_state_dict(payload["baseline"])
    optimizer.load_state_dict(payload["optimizer"])
    return payload


def train_reinforce_data_passes(
    *,
    method: str,
    args: Any,
    pool: Any,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    make_actor: Callable[[list[Any], bool, int], Any],
    make_baseline: Callable[[torch.nn.Module, list[Any], bool, int], Any],
    training_cost: Callable[[Any], torch.Tensor],
    objective_distance: Callable[[Any], torch.Tensor],
    feasible: Callable[[Any], torch.Tensor],
    validation_solve: Callable[[torch.nn.Module, Any, int], dict[str, Any]],
    legacy_batch_size: int,
    soft_stage_fraction: float = 0.0,
) -> None:
    """Run the frozen corpus-pass protocol for the three REINFORCE baselines.

    State is committed only after a complete pass and its checkpoint.  An
    interrupted pass is intentionally replayed from its deterministic shuffle
    on resume, avoiding ambiguous mid-batch optimizer state.
    """

    if args.data_passes is None or args.data_passes <= 0:
        raise ValueError("protocol mode requires a positive --data-passes")
    if args.max_batches_per_pass is not None and not args.pilot_mode:
        raise ValueError("--max-batches-per-pass is allowed only with --pilot-mode")
    physical, effective = require_registered_batches(args, legacy_batch_size)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = load_state(output, args.protocol_id, args.resume)
    checkpoint = output / "checkpoint_latest.pt"
    selected_checkpoint = output / "checkpoint_selected.pt"
    baseline = deepcopy(policy).eval()
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)
    resume_extra: dict[str, Any] = {}
    if args.resume:
        if not checkpoint.exists():
            raise FileNotFoundError(f"resume checkpoint is missing: {checkpoint}")
        resume_extra = _load_checkpoint(
            checkpoint,
            policy=policy,
            baseline=baseline,
            optimizer=optimizer,
            protocol_id=args.protocol_id,
        )
        if int(resume_extra.get("data_pass", -1)) != state.completed_data_passes:
            raise ValueError("checkpoint and data-pass state disagree")
    elif checkpoint.exists():
        raise FileExistsError(f"existing checkpoint requires --resume: {checkpoint}")

    validation_pool = make_validation_pool(args, scale=args.scale, seed=args.seed)
    validation_instances = (
        list(validation_pool.first(limit=args.validation_limit))
        if validation_pool is not None
        else []
    )
    best_key = tuple(resume_extra.get("best_validation_key", [-math.inf, -math.inf]))
    baseline_probe_size = max(
        0, int(getattr(args, "baseline_eval_size", 64))
    )
    baseline_probe_instances = list(
        pool.first(limit=min(baseline_probe_size, len(pool)))
    )
    ema_cost = resume_extra.get("ema_cost")
    optimizer_steps = int(state.optimizer_steps)
    environment_transitions_total = int(state.environment_transitions)
    run_started = time.perf_counter()
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)

    first_pass = state.completed_data_passes + 1
    for data_pass in range(first_pass, int(args.data_passes) + 1):
        pass_started = time.perf_counter()
        soft = bool(soft_stage_fraction and data_pass <= int(args.data_passes * soft_stage_fraction))
        sums = {"loss": 0.0, "cost": 0.0, "distance": 0.0, "feasible": 0.0}
        instances_seen = 0
        transition_count = 0
        complete_pass = args.max_batches_per_pass is None
        batches = pool.data_pass_batches(data_pass, physical)
        for group_index, batch_group in enumerate(
            grouped_batches(
                batches,
                effective_batch_size=effective,
                max_batches=args.max_batches_per_pass,
            )
        ):
            group_size = sum(len(batch) for batch in batch_group)
            optimizer.zero_grad(set_to_none=True)
            for sub_index, instances in enumerate(batch_group):
                rollout_seed = int(args.seed) + data_pass * 10_000_000 + group_index * 1000 + sub_index
                torch.manual_seed(rollout_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(rollout_seed)
                actor = make_actor(instances, soft, rollout_seed)
                actor_cost = training_cost(actor)
                if method == "EVRPTW-RL" and optimizer_steps < int(args.ema_warmup_steps):
                    observed = float(actor_cost.mean().detach().cpu())
                    ema_cost = observed if ema_cost is None else args.ema_decay * ema_cost + (1.0 - args.ema_decay) * observed
                    baseline_cost = torch.full_like(actor_cost, float(ema_cost))
                else:
                    with torch.no_grad():
                        baseline_result = make_baseline(baseline, instances, soft, rollout_seed)
                    baseline_cost = training_cost(baseline_result)
                advantage = (actor_cost - baseline_cost).detach()
                loss = (advantage * actor.log_likelihood).mean()
                (loss * (len(instances) / max(group_size, 1))).backward()
                count = len(instances)
                instances_seen += count
                sums["loss"] += float(loss.detach().cpu()) * count
                sums["cost"] += float(actor_cost.mean().detach().cpu()) * count
                sums["distance"] += float(objective_distance(actor).mean().detach().cpu()) * count
                sums["feasible"] += float(feasible(actor).float().mean().detach().cpu()) * count
                transition_count += int(actor.environment_transitions)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer_steps += 1
            if method == "EVRPTW-RL" and optimizer_steps == int(args.ema_warmup_steps):
                baseline.load_state_dict(policy.state_dict())

        if instances_seen == 0:
            raise RuntimeError("data pass yielded no training instances")
        if complete_pass and instances_seen != len(pool):
            raise RuntimeError(f"incomplete data pass: {instances_seen} != {len(pool)}")

        # Keep each method's rollout-baseline update, but never use a test set.
        baseline_updated = False
        paired_t_pvalue: float | None = None
        baseline_probe = baseline_probe_instances
        if baseline_probe:
            actor_costs = []
            baseline_costs = []
            for index, instance in enumerate(baseline_probe):
                seed = int(args.seed) + 700_000 + index
                with torch.no_grad():
                    actor_result = make_baseline(policy, [instance], False, seed)
                    baseline_result = make_baseline(baseline, [instance], False, seed)
                actor_costs.append(float(training_cost(actor_result).mean().cpu()))
                baseline_costs.append(float(training_cost(baseline_result).mean().cpu()))
            test = ttest_rel(actor_costs, baseline_costs, alternative="less")
            paired_t_pvalue = float(test.pvalue)
            baseline_updated = bool(
                np.mean(actor_costs) < np.mean(baseline_costs)
                and np.isfinite(test.pvalue)
                and paired_t_pvalue < float(args.baseline_alpha)
            )
            if baseline_updated:
                baseline.load_state_dict(policy.state_dict())

        validation: dict[str, Any] | None = None
        should_validate = bool(
            validation_instances
            and (
                data_pass % int(args.validation_every_passes) == 0
                or data_pass == int(args.data_passes)
            )
        )
        if should_validate:
            policy.eval()
            validation = verified_validation(
                validation_instances,
                lambda instance, seed: validation_solve(policy, instance, seed),
                seed=int(args.seed) + data_pass * 100_000,
            )
            validation.update({"data_pass": data_pass, "split": "validation"})
            append_jsonl(output / "validation_history.jsonl", validation)

        is_best = bool(
            validation is not None and validation_key(validation) > tuple(best_key)
        )
        if is_best:
            best_key = validation_key(validation)
        extra = {
            "best_validation_key": list(best_key),
            "ema_cost": ema_cost,
            "pilot_partial_pass": not complete_pass,
        }
        _save_checkpoint(
            checkpoint,
            method=method,
            data_pass=data_pass if complete_pass else state.completed_data_passes,
            policy=policy,
            baseline=baseline,
            optimizer=optimizer,
            args=args,
            extra=extra,
        )
        if is_best:
            shutil.copy2(checkpoint, selected_checkpoint)
            atomic_json(output / "validation_summary.json", validation)
        if not selected_checkpoint.exists() and args.pilot_mode:
            shutil.copy2(checkpoint, selected_checkpoint)

        row = {
            "schema": "drl_data_pass_history_v1",
            "method": method,
            "protocol_id": args.protocol_id,
            "data_pass": data_pass,
            "pass_complete": complete_pass,
            "training_stage": "soft" if soft else "hard",
            "instances_seen": instances_seen,
            "customer_exposures": instances_seen * _customer_count(args.scale),
            "optimizer_steps_total": optimizer_steps,
            "environment_transitions": transition_count,
            "environment_transitions_total": environment_transitions_total + transition_count,
            "physical_batch_size": physical,
            "effective_batch_size": effective,
            "mean_loss": sums["loss"] / instances_seen,
            "mean_training_cost": sums["cost"] / instances_seen,
            "mean_objective_distance_km": sums["distance"] / instances_seen,
            "mean_environment_feasible_rate": sums["feasible"] / instances_seen,
            "paired_t_pvalue": paired_t_pvalue,
            "baseline_updated": baseline_updated,
            "pass_wall_time_s": time.perf_counter() - pass_started,
            "run_wall_time_s": time.perf_counter() - run_started,
            "peak_gpu_memory_bytes": _peak_gpu_bytes(args.device),
        }
        append_jsonl(output / "train_history.jsonl", row)
        print(json.dumps(row, sort_keys=True), flush=True)
        environment_transitions_total += transition_count
        if complete_pass:
            state.completed_data_passes = data_pass
            state.instances_seen += instances_seen
            state.customer_exposures += instances_seen * _customer_count(args.scale)
            state.optimizer_steps = optimizer_steps
            state.environment_transitions = environment_transitions_total
            state.last_checkpoint = str(checkpoint)
            state.atomic_write(output / "data_pass_state.json")
        else:
            break

    terminal = {
        "schema": "drl_training_result_v1",
        "status": "pilot_partial" if args.max_batches_per_pass is not None else "passed",
        "method": method,
        "protocol_id": args.protocol_id,
        "requested_data_passes": int(args.data_passes),
        "completed_data_passes": int(state.completed_data_passes),
        "instances_seen": int(
            state.instances_seen
            if args.max_batches_per_pass is None
            else state.instances_seen + instances_seen
        ),
        "customer_exposures": int(
            state.customer_exposures
            if args.max_batches_per_pass is None
            else state.customer_exposures
            + instances_seen * _customer_count(args.scale)
        ),
        "optimizer_steps": int(optimizer_steps),
        "environment_transitions": int(environment_transitions_total),
        "selected_checkpoint": str(selected_checkpoint if selected_checkpoint.exists() else checkpoint),
        "peak_gpu_memory_bytes": _peak_gpu_bytes(args.device),
        "wall_time_s": time.perf_counter() - run_started,
    }
    atomic_json(output / "training_result.json", terminal)


__all__ = ["train_reinforce_data_passes"]
