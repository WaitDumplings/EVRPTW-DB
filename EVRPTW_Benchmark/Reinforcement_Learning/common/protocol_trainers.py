from __future__ import annotations

import json
import math
import shutil
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.stats import ttest_rel

from .method_auxiliary import (
    assert_checkpoint_method_auxiliary,
    method_auxiliary_from_args,
)
from .objective import objective_from_args, objective_from_checkpoint
from .reward_contract import (
    assert_checkpoint_reward_contract,
    reward_contract_from_args,
)
from .training_diagnostics import summarize_values
from .training_stream import (
    assert_checkpoint_training_stream_contract,
    training_stream_contract_from_args,
)
from .training_protocol import (
    assert_checkpoint_training_signature,
    append_jsonl,
    atomic_json,
    freeze_resolved_training_signature,
    grouped_batches,
    load_state,
    make_validation_pool,
    parse_float_checkpoints,
    parse_int_checkpoints,
    require_registered_batches,
    require_validation_decoding,
    validation_epochs,
    validation_key,
    verified_validation,
)


def prepare_training_objective(args: Any):
    """Freeze objective values and prevent a new cost run inheriting old output."""
    config = objective_from_args(args)
    args.objective = config.to_dict()
    active_reward_contract = reward_contract_from_args(
        args, objective=config, scale=getattr(args, "scale", None)
    )
    training_stream_contract_from_args(
        args,
        required=(getattr(args, "protocol_id", None) == "drl_rq_protocol_frozen_v1"),
    )
    if (
        getattr(args, "validation_seed", None) is None
        and getattr(args, "seed", None) is not None
    ):
        args.validation_seed = int(args.seed) + 910_000_000
    freeze_resolved_training_signature(args)
    resume = bool(getattr(args, "resume", False))
    warm_start = getattr(args, "warm_start_checkpoint", None)
    if resume and warm_start is not None:
        raise ValueError("--resume and --warm-start-checkpoint are mutually exclusive")
    formal = (
        getattr(args, "data_passes", None) is not None
        or getattr(args, "training_epochs", None) is not None
    )
    if formal and config.is_cost and active_reward_contract is None:
        raise ValueError(
            "formal cost training requires a frozen reward contract"
        )
    if resume and not formal:
        raise ValueError("standalone training does not implement --resume; use a formal training protocol")
    if warm_start is not None and not formal:
        raise ValueError(
            "standalone training does not implement warm-start; use a formal training protocol"
        )
    if warm_start is not None and not Path(warm_start).is_file():
        raise FileNotFoundError(f"warm-start checkpoint is missing: {warm_start}")
    output = Path(args.output_dir)
    if resume and (output / "checkpoint_latest.pt").is_file():
        payload = torch.load(output / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
        objective_from_checkpoint(payload, override=config)
        assert_checkpoint_reward_contract(payload, args)
        assert_checkpoint_training_stream_contract(payload, args)
        assert_checkpoint_training_signature(payload, args)
        expected_auxiliary_method = getattr(args, "method_auxiliary_method", None)
        if expected_auxiliary_method is not None:
            assert_checkpoint_method_auxiliary(
                payload, args, expected_method=expected_auxiliary_method
            )
    if config.is_cost and not resume:
        evidence = list(output.glob("checkpoint*.pt")) + list(output.glob("best*.ckpt"))
        evidence.extend(
            output / name for name in (
                "data_pass_state.json", "train_history.jsonl", "logical_epoch_history.jsonl",
                "validation_history.jsonl", "validation_summary.json", "training_result.json",
            ) if (output / name).exists()
        )
        if (output / "checkpoints").is_dir():
            evidence.extend((output / "checkpoints").iterdir())
        if evidence:
            raise FileExistsError("fresh cost training requires a new output directory without training history")
    return config


def _collect_diagnostic_values(
    destination: dict[str, list[np.ndarray]], name: str, values: Any,
) -> None:
    """Detach one microbatch; storage lives for one logical update only."""
    if values is None:
        return
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    destination.setdefault(name, []).append(
        np.asarray(values, dtype=np.float64).reshape(-1).copy()
    )


def _summarize_diagnostic_groups(
    groups: dict[str, list[np.ndarray]],
) -> dict[str, dict[str, Any]]:
    return {
        name: summarize_values(np.concatenate(parts))
        for name, parts in groups.items() if parts
    }


def _collect_reinforce_diagnostics(
    *, actor: Any, actor_cost: torch.Tensor, baseline_cost: torch.Tensor,
    advantage: torch.Tensor, raw_distance: torch.Tensor,
    active_objective: torch.Tensor, started_vehicles: torch.Tensor,
    objective_config: Any, distributions: dict[str, list[np.ndarray]],
    components: dict[str, list[np.ndarray]], scales: dict[str, list[np.ndarray]],
) -> None:
    for name, values in (
        ("actor_training_cost", actor_cost),
        ("baseline_training_cost", torch.broadcast_to(baseline_cost, actor_cost.shape)),
        ("pre_loss_advantage", advantage),
        ("raw_objective_value", active_objective),
        ("raw_distance_km", raw_distance),
        ("vehicles_started", started_vehicles),
    ):
        _collect_diagnostic_values(distributions, name, values)
    if objective_config.is_cost:
        # Detached CPU accounting, not inputs to the policy loss.
        distributions.setdefault("raw_electricity_cost_usd", []).append(
            distributions["raw_distance_km"][-1] * objective_config.distance_unit_cost
        )
        distributions.setdefault("raw_vehicle_cost_usd", []).append(
            distributions["vehicles_started"][-1] * objective_config.vehicle_unit_cost
        )
    for name, values in (getattr(actor, "training_cost_components", None) or {}).items():
        _collect_diagnostic_values(components, name, values)
    for name, values in (getattr(actor, "soft_violation_diagnostics", None) or {}).items():
        _collect_diagnostic_values(distributions, name, values)
    for name, values in (
        getattr(actor, "method_auxiliary_diagnostics", None) or {}
    ).items():
        _collect_diagnostic_values(distributions, name, values)
    _collect_diagnostic_values(scales, "objective_scale", getattr(actor, "reward_objective_scale", None))


def _append_reinforce_diagnostics(
    *, output: Path, method: str, args: Any, objective_config: Any,
    session: dict[str, Any], data_pass: int, logical_epoch: int,
    optimizer_steps: int, soft: bool, baseline_kind: str,
    distributions: dict[str, list[np.ndarray]],
    components: dict[str, list[np.ndarray]], scales: dict[str, list[np.ndarray]],
    pre_clip_norm: Any,
) -> None:
    """Positive trajectory-cost diagnostics, never inputs to learning."""
    diagnostics_started = time.perf_counter()
    gradient = summarize_values(pre_clip_norm)
    threshold = float(args.max_grad_norm)
    pre_clip_value = gradient["mean"]
    component_summaries = _summarize_diagnostic_groups(components)
    row = {
        "schema": "drl_reward_diagnostics_v1",
        "method": method,
        "protocol_id": args.protocol_id,
        **session,
        "data_pass": int(data_pass),
        "logical_epoch": int(logical_epoch),
        "optimizer_steps_total": int(optimizer_steps),
        "training_stage": "soft" if soft else "hard",
        "sample_unit": "candidate_trajectory_including_failed_or_truncated",
        "cost_sign_convention": "positive_cost_lower_is_better_not_step_reward",
        "advantage_definition": "actor_training_cost_minus_actual_baseline_cost_before_loss",
        "baseline_kind": baseline_kind,
        "objective_config": objective_config.to_dict(),
        "objective_unit": objective_config.unit,
        "normalization": {
            "training_cost_unit": "dimensionless",
            "base_objective_normalized": True,
            "advantage_standardized": False,
            "reward_objective_scale": _summarize_diagnostic_groups(scales).get("objective_scale", summarize_values([])),
            "objective_scale_sample_unit": "base_instance_not_trajectory",
            "objective_scale_unit": objective_config.unit,
            "reward_distance_scale_km": getattr(args, "reward_distance_scale_km", None),
            "reward_distance_scale_mode": getattr(args, "reward_distance_scale_mode", None),
            "reward_distance_scale_metadata": getattr(args, "reward_distance_scale_metadata", None),
            "reward_contract_id": getattr(args, "reward_contract_id", None),
            "reward_contract_sha256": getattr(args, "reward_contract_sha256", None),
            "failure_base": getattr(args, "reward_failure_base", None),
            "unserved_coefficient": getattr(
                args, "reward_unserved_coefficient", None
            ),
            "method_auxiliary_profile_id": getattr(
                args, "method_auxiliary_profile_id", None
            ),
            "method_auxiliary_sha256": getattr(
                args, "method_auxiliary_sha256", None
            ),
            "method_auxiliary_applicability": getattr(
                args, "method_auxiliary_applicability", None
            ),
            "method_auxiliary_aggregation": getattr(
                args, "method_auxiliary_aggregation", None
            ),
            "method_auxiliary_denominator": getattr(
                args, "method_auxiliary_denominator", None
            ),
            "method_auxiliary_step_clip": getattr(
                args, "method_auxiliary_step_clip", None
            ),
            "method_auxiliary_component_clip": getattr(
                args, "method_auxiliary_component_clip", None
            ),
            "method_auxiliary_weights": getattr(
                args, "method_auxiliary_weights", None
            ),
            "method_auxiliary_formula": (
                "weight*executed_legal_station_visits/num_customers"
                if getattr(args, "method_auxiliary_method", None) == "evrptw_rl"
                else None
            ),
            "soft_violation_contract_id": getattr(
                args, "soft_violation_contract_id", None
            ),
            "soft_violation_step_clip": getattr(
                args, "soft_violation_step_clip", None
            ),
            "soft_violation_component_clip": getattr(
                args, "soft_violation_component_clip", None
            ),
            "soft_violation_denominator": getattr(
                args, "soft_violation_denominator", None
            ),
            "soft_violation_formula": (
                "weight*min(component_clip,"
                "sum_valid_transitions(min(normalized_excess,step_clip))"
                "/num_customers)"
                if getattr(args, "soft_violation_contract_id", None)
                else None
            ),
        },
        "distributions": _summarize_diagnostic_groups(distributions),
        "components": component_summaries,
        "component_unit": "dimensionless_positive_trajectory_cost",
        "component_relationships": {
            "base_objective": ["base_distance_term", "base_vehicle_term"],
            "training_cost": [
                name for name in component_summaries
                if name not in {
                    "base_distance_term",
                    "base_vehicle_term",
                    "terminal_task_total",
                    "terminal_failure_penalty",
                    "soft_auxiliary_total",
                }
            ],
            "terminal_task_total": [
                "terminal_failure_base", "terminal_unserved"
            ],
            "soft_auxiliary_total": [
                name for name in (
                    "capacity_penalty", "time_penalty", "energy_penalty"
                ) if name in component_summaries
            ],
            "station_visit_auxiliary": [
                "station_visits_raw", "station_visit_denominator",
                "station_visits_normalized",
            ] if "station_visit_auxiliary" in component_summaries else [],
            "base_distance_term_meaning": "electricity_cost_normalized" if objective_config.is_cost else "distance_normalized",
            "base_vehicle_term_meaning": "vehicle_cost_normalized" if objective_config.is_cost else "zero_no_vehicle_term",
        },
        "gradients": {
            "pre_clip_norm": gradient,
            "clip_max_norm": threshold,
            "clipping_fraction": (
                float(pre_clip_value > threshold) if pre_clip_value is not None else None
            ),
            "clipping_definition": "finite_pre_clip_norm_exceeds_clip_max_norm",
            "optimizer_updates": 1,
            "scope": "after_all_weighted_microbatch_backward_before_existing_clip",
        },
    }
    row["diagnostics_compute_wall_time_s"] = time.perf_counter() - diagnostics_started
    row["diagnostics_compute_wall_time_scope"] = "summary_and_row_creation_excluding_collection_and_file_write"
    append_jsonl(output / "reward_diagnostics.jsonl", row)


def same_instance_leave_one_out(actor_cost: torch.Tensor) -> torch.Tensor:
    """Cost baseline from other independent trajectories of the same instance."""
    if actor_cost.ndim != 2 or actor_cost.shape[1] < 2:
        raise ValueError(
            "leave_one_out requires a [batch, trajectories] cost tensor with at least 2 trajectories"
        )
    detached = actor_cost.detach()
    return (detached.sum(dim=1, keepdim=True) - detached) / (detached.shape[1] - 1)


def paper_ema_baseline_due(method: str, optimizer_steps: int, args: Any) -> bool:
    """Return whether the method uses EMA warmup.

    AM follows its publication; RRNCO-EV deliberately matches the AM schedule
    for the controlled architecture comparison.
    """

    step = int(optimizer_steps)
    if step < 0:
        raise ValueError("optimizer_steps cannot be negative")
    if method in {"AM-EVRPTW", "RRNCO-EV"}:
        warmup_steps = int(args.steps_per_epoch) * int(args.baseline_warmup_epochs)
        return step < warmup_steps
    if method == "EVRPTW-RL":
        return step < int(args.ema_warmup_steps)
    return False


def paper_baseline_eval_due(method: str, optimizer_steps: int, args: Any) -> bool:
    """Return whether the publication's rollout-baseline comparison is due.

    training_epochs in the benchmark protocol counts optimizer updates, not
    the much larger paper epochs. Consequently the schedule is expressed in
    optimizer steps: AM uses its published 2,500 batches per epoch, while
    EVRPTW-RL uses its published post-warmup 100-step interval. RRNCO-EV
    deliberately matches AM for this controlled comparison. DRL-TS is not
    assigned a schedule here because the full manuscript/source is unavailable.
    """

    step = int(optimizer_steps)
    if step <= 0:
        return False
    if method in {"AM-EVRPTW", "RRNCO-EV"}:
        interval = int(args.steps_per_epoch)
        return interval > 0 and step % interval == 0
    if method == "EVRPTW-RL":
        warmup = int(args.ema_warmup_steps)
        interval = int(args.baseline_eval_interval)
        return step > warmup and interval > 0 and step % interval == 0
    return False


def _customer_count(scale: str) -> int:
    value = str(scale).lower().removeprefix("cus")
    if not value.isdigit() or int(value) <= 0:
        raise ValueError(f"invalid scale: {scale}")
    return int(value)


def _resolve_soft_stage_contract(
    *,
    method: str,
    fixed_epochs: int | None,
    total_passes: int,
    soft_stage_fraction: float,
    soft_stage_end_epoch: int | None,
) -> dict[str, Any] | None:
    """Freeze the exact DRL-TS soft-to-hard reward-stage boundary.

    An absolute epoch boundary takes precedence over the legacy fraction.  In
    that mode the fraction is deliberately recorded as inactive, so changing
    an ignored CLI default cannot invalidate a checkpoint.  When the fraction
    determines the boundary, both the exact fraction and its resolved integer
    boundary are frozen.
    """

    if method != "DRL-TS":
        return None
    fraction = float(soft_stage_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("soft-stage fraction must be finite and in [0, 1]")
    if soft_stage_end_epoch is not None:
        if fixed_epochs is None:
            raise ValueError("an absolute soft-stage boundary requires --training-epochs")
        end_epoch = int(soft_stage_end_epoch)
        if not 0 <= end_epoch <= fixed_epochs:
            raise ValueError("soft-stage end epoch must be in [0, training epochs]")
        return {
            "schema": "drl_ts_soft_stage_contract_v1",
            "mode": "absolute_epoch",
            "soft_stage_end_epoch": end_epoch,
            "soft_stage_fraction": None,
            "resolved_soft_stage_end_epoch": end_epoch,
            "resolved_soft_stage_end_data_pass": None,
        }
    if fixed_epochs is not None:
        resolved_end = int(fixed_epochs * fraction)
        return {
            "schema": "drl_ts_soft_stage_contract_v1",
            "mode": "fraction_of_training_epochs",
            "soft_stage_end_epoch": None,
            "soft_stage_fraction": fraction,
            "resolved_soft_stage_end_epoch": resolved_end,
            "resolved_soft_stage_end_data_pass": None,
        }
    resolved_pass = int(total_passes * fraction)
    return {
        "schema": "drl_ts_soft_stage_contract_v1",
        "mode": "fraction_of_data_passes",
        "soft_stage_end_epoch": None,
        "soft_stage_fraction": fraction,
        "resolved_soft_stage_end_epoch": None,
        "resolved_soft_stage_end_data_pass": resolved_pass,
    }


def _assert_checkpoint_soft_stage_contract(payload: dict[str, Any], args: Any) -> None:
    expected = getattr(args, "soft_stage_contract_snapshot", None)
    checkpoint = payload.get("soft_stage_contract")
    checkpoint_args = payload.get("args", {})
    embedded = (
        checkpoint_args.get("soft_stage_contract_snapshot")
        if isinstance(checkpoint_args, dict)
        else None
    )
    if expected is None:
        if checkpoint is not None or embedded is not None:
            raise ValueError(
                "checkpoint soft-stage contract mismatch; start a fresh run"
            )
        return
    if checkpoint != expected or embedded != expected:
        raise ValueError(
            "checkpoint soft-stage contract mismatch; start a fresh run"
        )


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
    reward_contract_from_args(
        args,
        objective=objective_from_args(args),
        scale=getattr(args, "scale", None),
    )
    freeze_resolved_training_signature(args)
    expected_auxiliary_method = getattr(args, "method_auxiliary_method", None)
    if expected_auxiliary_method is not None:
        method_auxiliary_from_args(
            args, expected_method=expected_auxiliary_method
        )
    payload = {
        "method": method,
        "data_pass": int(data_pass),
        "model": policy.state_dict(),
        "baseline": baseline.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "protocol_id": args.protocol_id,
        "objective_config": objective_from_args(args).to_dict(),
        "reward_contract": getattr(args, "reward_contract_snapshot", None),
        "method_auxiliary_profile": getattr(
            args, "method_auxiliary_snapshot", None
        ),
        "training_stream_contract": getattr(
            args, "training_stream_contract_snapshot", None
        ),
        "resolved_training_signature": getattr(
            args, "resolved_training_signature", None
        ),
        "soft_stage_contract": getattr(
            args, "soft_stage_contract_snapshot", None
        ),
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
    objective_config: Any = None,
    optimizer_name: str | None = None,
    optimizer_weight_decay: float | None = None,
    reward_contract_args: Any = None,
) -> dict[str, Any]:
    policy_device = getattr(policy, "device", None)
    if policy_device is None:
        policy_device = next(policy.parameters()).device
    payload = torch.load(path, map_location=policy_device, weights_only=False)
    if payload.get("protocol_id") != protocol_id:
        raise ValueError("checkpoint protocol does not match requested protocol")
    objective_from_checkpoint(payload, override=objective_config)
    if reward_contract_args is not None:
        assert_checkpoint_reward_contract(payload, reward_contract_args)
        assert_checkpoint_training_stream_contract(payload, reward_contract_args)
        _assert_checkpoint_soft_stage_contract(payload, reward_contract_args)
        assert_checkpoint_training_signature(payload, reward_contract_args)
        expected_auxiliary_method = getattr(
            reward_contract_args, "method_auxiliary_method", None
        )
        checkpoint_auxiliary = payload.get("method_auxiliary_profile")
        if expected_auxiliary_method is not None:
            assert_checkpoint_method_auxiliary(
                payload,
                reward_contract_args,
                expected_method=expected_auxiliary_method,
            )
        elif checkpoint_auxiliary is not None:
            raise ValueError(
                "checkpoint has a method auxiliary profile but the run does not"
            )
    if optimizer_name is not None or optimizer_weight_decay is not None:
        if optimizer_name is None or optimizer_weight_decay is None:
            raise ValueError("incomplete requested optimizer contract")
        requested_name = str(optimizer_name).lower()
        requested_weight_decay = float(optimizer_weight_decay)
        saved_args = payload.get("args", {})
        if str(saved_args.get("optimizer", "")).lower() != requested_name:
            raise ValueError(
                "checkpoint optimizer mismatch; start a fresh run"
            )
        try:
            saved_weight_decay = float(saved_args["weight_decay"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "checkpoint is missing optimizer weight decay; start a fresh run"
            ) from error
        if saved_weight_decay != requested_weight_decay:
            raise ValueError(
                "checkpoint optimizer weight decay mismatch; start a fresh run"
            )
        param_groups = payload.get("optimizer", {}).get("param_groups", [])
        if not param_groups or any(
            float(group.get("weight_decay", float("nan")))
            != requested_weight_decay
            for group in param_groups
        ):
            raise ValueError(
                "checkpoint optimizer state weight decay mismatch; start a fresh run"
            )
        if requested_name == "adamw" and not isinstance(
            optimizer, torch.optim.AdamW
        ):
            raise ValueError("requested AdamW contract requires an AdamW optimizer")
    policy.load_state_dict(payload["model"])
    baseline.load_state_dict(payload["baseline"])
    optimizer.load_state_dict(payload["optimizer"])
    return payload


def _load_warm_start_checkpoint(
    path: Path,
    *,
    method: str,
    policy: torch.nn.Module,
    baseline: torch.nn.Module,
    objective_config: Any,
    contract_args: Any,
) -> dict[str, Any]:
    """Load compatible model weights while resetting all training state."""

    policy_device = getattr(policy, "device", None)
    if policy_device is None:
        policy_device = next(policy.parameters()).device
    payload = torch.load(path, map_location=policy_device, weights_only=False)
    if payload.get("method") != method:
        raise ValueError(
            f"warm-start method mismatch: {payload.get('method')!r} != {method!r}"
        )
    objective_from_checkpoint(payload, override=objective_config)
    assert_checkpoint_reward_contract(payload, contract_args)
    _assert_checkpoint_soft_stage_contract(payload, contract_args)
    expected_auxiliary_method = getattr(
        contract_args, "method_auxiliary_method", None
    )
    if expected_auxiliary_method is not None:
        assert_checkpoint_method_auxiliary(
            payload,
            contract_args,
            expected_method=expected_auxiliary_method,
        )
    saved_args = payload.get("args", {}) or {}
    if not isinstance(saved_args, dict):
        saved_args = vars(saved_args)
    for field in ("scale", "training_representation", "seed"):
        requested = getattr(contract_args, field, None)
        saved = saved_args.get(field)
        if requested is not None and saved is not None and str(saved) != str(requested):
            raise ValueError(
                f"warm-start {field} mismatch: {saved!r} != {requested!r}"
            )
    state_dict = payload.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError("warm-start checkpoint is missing model weights")
    policy.load_state_dict(state_dict)
    baseline.load_state_dict(policy.state_dict())
    return {
        "checkpoint": str(path.resolve()),
        "method": payload.get("method"),
        "source_logical_epoch": int(payload.get("logical_epoch", 0) or 0),
        "source_data_pass": int(payload.get("data_pass", 0) or 0),
        "optimizer_reset": True,
        "epoch_reset": True,
        "baseline_history_reset": True,
        "validation_state_reset": True,
        "early_stop_state_reset": True,
    }


def _save_registered_snapshots(
    *,
    output: Path,
    method: str,
    args: Any,
    data_pass: int,
    policy: torch.nn.Module,
    baseline: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    observed_exposure: int,
    observed_gpu_hours: float,
    exposure_checkpoints: tuple[int, ...],
    gpu_hour_checkpoints: tuple[float, ...],
    saved_exposure: set[int],
    saved_gpu_hours: set[float],
) -> None:
    schedules = (
        ("customer_exposure", exposure_checkpoints, saved_exposure, observed_exposure),
        ("gpu_hours", gpu_hour_checkpoints, saved_gpu_hours, observed_gpu_hours),
    )
    for axis, thresholds, saved, observed in schedules:
        for requested in thresholds:
            if requested in saved or observed < requested:
                continue
            suffix = str(requested) if axis == "customer_exposure" else f"{requested:g}"
            snapshot = output / f"checkpoint_{axis}_{suffix}.pt"
            _save_checkpoint(
                snapshot,
                method=method,
                data_pass=data_pass,
                policy=policy,
                baseline=baseline,
                optimizer=optimizer,
                args=args,
                extra={
                    "checkpoint_axis": axis,
                    "requested_checkpoint": requested,
                    "observed_customer_exposures": int(observed_exposure),
                    "observed_gpu_hours": float(observed_gpu_hours),
                },
            )
            saved.add(requested)
            append_jsonl(
                output / "checkpoint_events.jsonl",
                {
                    "schema": "drl_training_checkpoint_event_v1",
                    "axis": axis,
                    "requested": requested,
                    "observed_customer_exposures": int(observed_exposure),
                    "observed_gpu_hours": float(observed_gpu_hours),
                    "path": str(snapshot),
                },
            )


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
    soft_stage_end_epoch: int | None = None,
) -> None:
    """Run a fixed-batch protocol for the three REINFORCE baselines.

    Formal fixed-epoch jobs take a deterministic prefix of a seeded shuffle and
    never need to traverse the full training index. The legacy complete-pass
    mode remains available for old explicit CLI invocations.
    """
    reinforce_baseline = getattr(args, "reinforce_baseline", "paper")
    if reinforce_baseline not in {"paper", "leave_one_out"}:
        raise ValueError(f"unsupported REINFORCE baseline: {reinforce_baseline}")
    use_leave_one_out = reinforce_baseline == "leave_one_out"
    if use_leave_one_out and method != "RRNCO-EV":
        raise ValueError("leave_one_out is an explicit RRNCO-EV experiment only")
    if use_leave_one_out and int(getattr(args, "samples_per_instance", 2)) < 2:
        raise ValueError("leave_one_out requires at least 2 trajectories per instance")
    validation_decode_type, validation_candidates = require_validation_decoding(args)
    objective_config = prepare_training_objective(args)
    validation_seed = int(
        getattr(args, "validation_seed", None)
        if getattr(args, "validation_seed", None) is not None
        else int(args.seed) + 910_000_000
    )

    fixed_epochs = getattr(args, "training_epochs", None)
    if fixed_epochs is not None:
        fixed_epochs = int(fixed_epochs)
        if fixed_epochs <= 0:
            raise ValueError("--training-epochs must be positive")
        if args.data_passes is not None:
            raise ValueError("choose --training-epochs or --data-passes, not both")
        total_passes = 1
    else:
        if args.data_passes is None or args.data_passes <= 0:
            raise ValueError("protocol mode requires --training-epochs or --data-passes")
        total_passes = int(args.data_passes)
    soft_stage_fraction = float(soft_stage_fraction)
    soft_stage_contract = _resolve_soft_stage_contract(
        method=method,
        fixed_epochs=fixed_epochs,
        total_passes=total_passes,
        soft_stage_fraction=soft_stage_fraction,
        soft_stage_end_epoch=soft_stage_end_epoch,
    )
    args.soft_stage_contract_snapshot = soft_stage_contract
    if soft_stage_contract is not None:
        soft_stage_end_epoch = soft_stage_contract["soft_stage_end_epoch"]
    if args.max_batches_per_pass is not None and not args.pilot_mode:
        raise ValueError("--max-batches-per-pass is allowed only with --pilot-mode")
    if fixed_epochs is not None and args.max_batches_per_pass is not None:
        raise ValueError("--training-epochs cannot be combined with --max-batches-per-pass")
    validation_every_epochs = int(
        getattr(args, "validation_every_epochs", None)
        or fixed_epochs
        or 1
    )
    minimum_training_epochs = int(
        getattr(args, "minimum_training_epochs", None)
        or fixed_epochs
        or 1
    )
    post_minimum_validation_every_epochs = int(
        getattr(args, "post_minimum_validation_every_epochs", None)
        or validation_every_epochs
    )
    scheduled_validation_epochs: tuple[int, ...] = ()
    if fixed_epochs is not None:
        scheduled_validation_epochs = validation_epochs(
            fixed_epochs,
            initial_interval=validation_every_epochs,
            minimum_epochs=minimum_training_epochs,
            post_minimum_interval=post_minimum_validation_every_epochs,
        )
        if int(args.validation_checkpoints) != len(scheduled_validation_epochs):
            raise ValueError(
                "fixed-epoch validation checkpoint count does not match "
                "the configured two-phase validation schedule"
            )
    early_stop_patience = int(
        getattr(args, "early_stop_patience_validations", 0) or 0
    )
    if early_stop_patience < 0:
        raise ValueError("--early-stop-patience-validations cannot be negative")
    if early_stop_patience and fixed_epochs is None:
        raise ValueError("early stopping is supported only with --training-epochs")
    early_stop_start_epoch = int(
        getattr(args, "early_stop_start_epoch", 0) or 0
    )
    if early_stop_start_epoch < 0:
        raise ValueError("--early-stop-start-epoch cannot be negative")
    if early_stop_start_epoch and fixed_epochs is None:
        raise ValueError("delayed early stopping requires --training-epochs")
    if fixed_epochs is not None and early_stop_start_epoch >= fixed_epochs:
        raise ValueError(
            "--early-stop-start-epoch must be smaller than --training-epochs"
        )
    if early_stop_patience and early_stop_start_epoch < minimum_training_epochs:
        raise ValueError(
            "early stopping cannot start before --minimum-training-epochs"
        )
    physical, effective = require_registered_batches(args, legacy_batch_size)
    stream_path = getattr(args, "training_stream_path", None)
    expected_fixed_instances = fixed_epochs * effective if fixed_epochs is not None else None
    if (
        fixed_epochs is not None
        and stream_path is None
        and expected_fixed_instances > len(pool)
    ):
        raise ValueError("fixed training budget exceeds the no-replacement training pool")
    if stream_path is not None and fixed_epochs is None:
        raise ValueError("an explicit training stream requires --training-epochs")
    if stream_path is not None:
        customer_budget = getattr(args, "customer_exposure_budget", None)
        expected_budget = int(expected_fixed_instances) * _customer_count(args.scale)
        if customer_budget is None or int(customer_budget) != expected_budget:
            raise ValueError("explicit training stream requires an exact customer-exposure budget")
    if effective % physical and stream_path is None:
        raise ValueError(
            "a remainder physical batch requires an explicit training stream"
        )
    microbatches_per_epoch = math.ceil(effective / physical)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = load_state(output, args.protocol_id, args.resume)
    checkpoint = output / "checkpoint_latest.pt"
    selected_checkpoint = output / "checkpoint_selected.pt"
    best_checkpoint = output / "best.ckpt"
    best_within_minimum_checkpoint = output / "best_within_5000.ckpt"
    best_overall_checkpoint = output / "best_overall.ckpt"
    validation_summary = output / "validation_summary.json"
    validation_summary_within_minimum = (
        output / "validation_summary_within_5000.json"
    )
    validation_summary_overall = output / "validation_summary_overall.json"
    baseline = deepcopy(policy).eval()
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)
    warm_start_provenance: dict[str, Any] | None = None
    warm_start_checkpoint = getattr(args, "warm_start_checkpoint", None)
    if warm_start_checkpoint is not None:
        warm_start_provenance = _load_warm_start_checkpoint(
            Path(warm_start_checkpoint),
            method=method,
            policy=policy,
            baseline=baseline,
            objective_config=objective_config,
            contract_args=args,
        )
        args.warm_start_provenance = warm_start_provenance
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
            objective_config=objective_config,
            optimizer_name=getattr(args, "optimizer", None),
            optimizer_weight_decay=getattr(args, "weight_decay", None),
            reward_contract_args=args,
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
    completed_logical_epochs = (
        int(resume_extra.get("logical_epoch", 0))
        if fixed_epochs is not None
        else 0
    )
    if fixed_epochs is not None and not 0 <= completed_logical_epochs <= fixed_epochs:
        raise ValueError("resume checkpoint has an invalid logical epoch")
    if fixed_epochs is not None and completed_logical_epochs and stream_path is None:
        raise ValueError("fixed-epoch resume requires an explicit training stream")
    best_key = tuple(resume_extra.get("best_validation_key", [-math.inf, -math.inf]))
    best_within_minimum_key = tuple(
        resume_extra.get("best_within_minimum_key", [-math.inf, -math.inf])
    )
    validation_checks_without_improvement = int(
        resume_extra.get("validation_checks_without_improvement", 0)
    )
    completed_validation_checks = int(
        resume_extra.get("completed_validation_checks", 0)
    )
    early_stopped = False
    early_stop_epoch: int | None = None
    terminal_logical_epoch = completed_logical_epochs
    baseline_probe_size = (
        0 if use_leave_one_out else max(0, int(getattr(args, "baseline_eval_size", 64)))
    )
    baseline_probe_instances = list(
        pool.first(limit=min(baseline_probe_size, len(pool)))
    )
    baseline_eval_count = int(resume_extra.get("baseline_eval_count", 0))
    baseline_update_count = int(resume_extra.get("baseline_update_count", 0))
    ema_cost = resume_extra.get("ema_cost")
    optimizer_steps = int(state.optimizer_steps)
    starting_optimizer_steps = optimizer_steps
    environment_transitions_total = int(state.environment_transitions)
    starting_state_instances = int(state.instances_seen)
    starting_state_exposures = int(state.customer_exposures)
    diagnostic_session = {
        "session_id": str(time.time_ns()),
        "resume_requested": bool(args.resume),
        "resume_checkpoint": str(checkpoint) if args.resume else None,
        "warm_start_requested": warm_start_checkpoint is not None,
        "warm_start_provenance": warm_start_provenance,
        "session_start_optimizer_steps": starting_optimizer_steps,
        "session_start_logical_epoch": completed_logical_epochs,
        "session_start_completed_data_passes": int(state.completed_data_passes),
    }
    run_started = time.perf_counter()
    exposure_checkpoints = parse_int_checkpoints(getattr(args, "exposure_checkpoints", ""))
    gpu_hour_checkpoints = parse_float_checkpoints(getattr(args, "gpu_hour_checkpoints", ""))
    saved_exposure = {
        value for value in exposure_checkpoints
        if (output / f"checkpoint_customer_exposure_{value}.pt").is_file()
    }
    saved_gpu_hours = {
        value for value in gpu_hour_checkpoints
        if (output / f"checkpoint_gpu_hours_{value:g}.pt").is_file()
    }
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)

    # A fixed-update run is one logical pass even while an interrupted or
    # explicitly extended budget has completed_data_passes=1 from its previous
    # terminal checkpoint. Resume from logical_epoch, not from the pass flag.
    first_pass = (
        1 if fixed_epochs is not None else state.completed_data_passes + 1
    )
    for data_pass in range(first_pass, total_passes + 1):
        pass_started = time.perf_counter()
        soft = bool(soft_stage_fraction and data_pass <= int(total_passes * soft_stage_fraction))
        training_stage = "soft" if soft else "hard"
        if fixed_epochs is not None and (soft_stage_fraction or soft_stage_end_epoch is not None):
            training_stage = "mixed"
        sums = {key: 0.0 for key in ("loss", "cost", "distance", "objective", "vehicles_started", "feasible")}
        instances_seen = 0
        transition_count = 0
        trajectory_steps: list[int] = []
        rollout_budget_exhausted_count = 0
        failure_reason_counts: Counter[str] = Counter()
        complete_pass = fixed_epochs is not None or args.max_batches_per_pass is None
        remaining_fixed_epochs = (
            fixed_epochs - completed_logical_epochs
            if fixed_epochs is not None
            else None
        )
        max_batches = (
            None
            if fixed_epochs is not None and stream_path is not None
            else (
                remaining_fixed_epochs * microbatches_per_epoch
                if fixed_epochs is not None
                else args.max_batches_per_pass
            )
        )
        batches = (
            pool.stream_batches(
                stream_path,
                physical,
                start=completed_logical_epochs * effective,
                stop=expected_fixed_instances,
                logical_batch_size=effective,
                **(
                    {"training_stream_contract_sha256": str(
                        args.training_stream_contract_sha256
                    )}
                    if getattr(args, "training_stream_contract_sha256", None)
                    else {}
                ),
            )
            if stream_path is not None
            else pool.data_pass_batches(data_pass, physical)
        )
        for group_index, batch_group in enumerate(
            grouped_batches(
                batches,
                effective_batch_size=effective,
                max_batches=max_batches,
            )
        ):
            logical_epoch_started = time.perf_counter()
            logical_epoch = completed_logical_epochs + group_index + 1
            terminal_logical_epoch = logical_epoch
            group_soft = soft
            if fixed_epochs is not None and soft_stage_end_epoch is not None:
                group_soft = logical_epoch <= soft_stage_end_epoch
            elif fixed_epochs is not None and soft_stage_fraction:
                group_soft = logical_epoch <= int(
                    fixed_epochs * soft_stage_fraction
                )
            group_size = sum(len(batch) for batch in batch_group)
            group_sums = {key: 0.0 for key in sums}
            group_transitions = 0
            group_trajectory_steps: list[int] = []
            group_exhausted = 0
            group_failure_reason_counts: Counter[str] = Counter()
            diagnostic_distributions: dict[str, list[np.ndarray]] = {}
            diagnostic_components: dict[str, list[np.ndarray]] = {}
            diagnostic_scales: dict[str, list[np.ndarray]] = {}
            optimizer.zero_grad(set_to_none=True)
            for sub_index, instances in enumerate(batch_group):
                rollout_seed = int(args.seed) + data_pass * 10_000_000 + logical_epoch * 1000 + sub_index
                torch.manual_seed(rollout_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(rollout_seed)
                actor = make_actor(instances, group_soft, rollout_seed)
                actor_cost = training_cost(actor)
                if use_leave_one_out:
                    baseline_cost = same_instance_leave_one_out(actor_cost)
                    if actor.log_likelihood.shape != actor_cost.shape:
                        raise ValueError("leave_one_out cost and log_likelihood shapes must match")
                    baseline_kind = "same_instance_leave_one_out"
                else:
                    use_ema = paper_ema_baseline_due(method, optimizer_steps, args)
                    if use_ema:
                        observed = float(actor_cost.mean().detach().cpu())
                        ema_cost = (
                            observed
                            if ema_cost is None
                            else args.ema_decay * ema_cost
                            + (1.0 - args.ema_decay) * observed
                        )
                        baseline_cost = torch.full_like(actor_cost, float(ema_cost))
                    else:
                        with torch.no_grad():
                            baseline_result = make_baseline(baseline, instances, group_soft, rollout_seed)
                        baseline_cost = training_cost(baseline_result)
                    baseline_kind = "paper_ema" if use_ema else "greedy_rollout"
                advantage = (actor_cost - baseline_cost).detach()
                loss = (advantage * actor.log_likelihood).mean()
                (loss * (len(instances) / max(group_size, 1))).backward()
                count = len(instances)
                instances_seen += count
                raw_distance = objective_distance(actor)
                active_objective = getattr(actor, "objective_value", None)
                if active_objective is None:
                    if objective_config.is_cost:
                        raise ValueError("cost training requires a named objective_value")
                    active_objective = raw_distance
                started_vehicles = getattr(actor, "vehicles_started", torch.zeros_like(raw_distance))
                _collect_reinforce_diagnostics(
                    actor=actor, actor_cost=actor_cost, baseline_cost=baseline_cost,
                    advantage=advantage, raw_distance=raw_distance,
                    active_objective=active_objective, started_vehicles=started_vehicles,
                    objective_config=objective_config, distributions=diagnostic_distributions,
                    components=diagnostic_components, scales=diagnostic_scales,
                )
                metrics = {
                    "loss": float(loss.detach().cpu()) * count,
                    "cost": float(actor_cost.mean().detach().cpu()) * count,
                    "distance": float(raw_distance.mean().detach().cpu()) * count,
                    "objective": float(active_objective.mean().detach().cpu()) * count,
                    "vehicles_started": float(started_vehicles.mean().detach().cpu()) * count,
                    "feasible": float(feasible(actor).float().mean().detach().cpu()) * count,
                }
                for key, value in metrics.items():
                    sums[key] += value
                    group_sums[key] += value
                actor_transitions = int(actor.environment_transitions)
                actor_steps = actor.trajectory_steps.detach().cpu().reshape(-1).tolist()
                actor_exhausted = int(actor.rollout_budget_exhausted.sum().detach().cpu())
                transition_count += actor_transitions
                group_transitions += actor_transitions
                trajectory_steps.extend(actor_steps)
                group_trajectory_steps.extend(actor_steps)
                rollout_budget_exhausted_count += actor_exhausted
                group_exhausted += actor_exhausted
                actor_failure_reasons = getattr(actor, "failure_reasons", None)
                if actor_failure_reasons is not None:
                    observed_reasons = Counter(
                        str(value)
                        for value in np.asarray(actor_failure_reasons, dtype=object).reshape(-1)
                    )
                    failure_reason_counts.update(observed_reasons)
                    group_failure_reason_counts.update(observed_reasons)
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer_steps += 1
            _append_reinforce_diagnostics(
                output=output, method=method, args=args, objective_config=objective_config,
                session=diagnostic_session, data_pass=data_pass, logical_epoch=logical_epoch,
                optimizer_steps=optimizer_steps, soft=group_soft,
                baseline_kind=baseline_kind,
                distributions=diagnostic_distributions, components=diagnostic_components,
                scales=diagnostic_scales, pre_clip_norm=pre_clip_norm,
            )
            if method == "EVRPTW-RL" and optimizer_steps == int(args.ema_warmup_steps):
                baseline.load_state_dict(policy.state_dict())
            baseline_updated = False
            paired_t_pvalue: float | None = None
            if baseline_probe_instances and paper_baseline_eval_due(
                method, optimizer_steps, args
            ):
                policy.eval()
                actor_costs = []
                baseline_costs = []
                for probe_index, instance in enumerate(baseline_probe_instances):
                    probe_seed = (
                        int(args.seed)
                        + 700_000
                        + optimizer_steps * 10_000
                        + probe_index
                    )
                    with torch.no_grad():
                        actor_result = make_baseline(
                            policy, [instance], False, probe_seed
                        )
                        baseline_result = make_baseline(
                            baseline, [instance], False, probe_seed
                        )
                    actor_costs.append(
                        float(training_cost(actor_result).mean().cpu())
                    )
                    baseline_costs.append(
                        float(training_cost(baseline_result).mean().cpu())
                    )
                test = ttest_rel(actor_costs, baseline_costs, alternative="less")
                paired_t_pvalue = float(test.pvalue)
                baseline_updated = bool(
                    np.mean(actor_costs) < np.mean(baseline_costs)
                    and np.isfinite(test.pvalue)
                    and paired_t_pvalue < float(args.baseline_alpha)
                )
                baseline_eval_count += 1
                if baseline_updated:
                    baseline.load_state_dict(policy.state_dict())
                    baseline_update_count += 1
                    baseline_probe_instances = list(
                        pool.sample(min(baseline_probe_size, len(pool)))
                    )
                append_jsonl(
                    output / "baseline_history.jsonl",
                    {
                        "schema": "drl_rollout_baseline_event_v1",
                        "method": method,
                        "optimizer_step": optimizer_steps,
                        "probe_instances": len(actor_costs),
                        "paired_t_pvalue": paired_t_pvalue,
                        "baseline_updated": baseline_updated,
                        "schedule_source": "publication",
                    },
                )
                policy.train()
            _save_registered_snapshots(
                output=output,
                method=method,
                args=args,
                data_pass=state.completed_data_passes,
                policy=policy,
                baseline=baseline,
                optimizer=optimizer,
                observed_exposure=(
                    starting_state_instances + instances_seen
                ) * _customer_count(args.scale),
                observed_gpu_hours=(time.perf_counter() - run_started) / 3600.0,
                exposure_checkpoints=exposure_checkpoints,
                gpu_hour_checkpoints=gpu_hour_checkpoints,
                saved_exposure=saved_exposure,
                saved_gpu_hours=saved_gpu_hours,
            )
            if fixed_epochs is not None:
                epoch_steps = np.asarray(group_trajectory_steps, dtype=np.int64)
                append_jsonl(
                    output / "logical_epoch_history.jsonl",
                    {
                        "schema": "drl_logical_epoch_history_v1",
                        "method": method,
                        "protocol_id": args.protocol_id,
                        "logical_epoch": logical_epoch,
                        "training_stage": "soft" if group_soft else "hard",
                        "instances_seen": group_size,
                        "customer_exposures": group_size * _customer_count(args.scale),
                        "physical_microbatches": len(batch_group),
                        "physical_batch_size": physical,
                        "effective_batch_size": effective,
                        "optimizer_steps_total": optimizer_steps,
                        "environment_transitions": group_transitions,
                        "mean_loss": group_sums["loss"] / group_size,
                        "mean_training_cost": group_sums["cost"] / group_size,
                        "mean_objective_distance_km": group_sums["distance"] / group_size,
                        "objective_mode": objective_config.mode,
                        "objective_unit": objective_config.unit,
                        "mean_objective_value": group_sums["objective"] / group_size,
                        "mean_objective_cost_usd": group_sums["objective"] / group_size if objective_config.is_cost else None,
                        "mean_electricity_cost_usd": group_sums["distance"] / group_size * objective_config.distance_unit_cost if objective_config.is_cost else None,
                        "mean_vehicle_cost_usd": group_sums["vehicles_started"] / group_size * objective_config.vehicle_unit_cost if objective_config.is_cost else None,
                        "mean_vehicles_started": group_sums["vehicles_started"] / group_size,
                        "mean_environment_feasible_rate": group_sums["feasible"] / group_size,
                        "mean_trajectory_steps": float(epoch_steps.mean()),
                        "rollout_budget_exhausted_rate": group_exhausted / max(epoch_steps.size, 1),
                        "terminal_outcome_reason_counts": dict(
                            sorted(group_failure_reason_counts.items())
                        ),
                        "baseline_eval_due": paired_t_pvalue is not None,
                        "paired_t_pvalue": paired_t_pvalue,
                        "baseline_updated": baseline_updated,
                        "epoch_wall_time_s": time.perf_counter() - logical_epoch_started,
                    },
                )
                validation_due = bool(
                    validation_instances
                    and logical_epoch in scheduled_validation_epochs
                )
                if validation_due:
                    policy.eval()
                    validation_started = time.perf_counter()
                    validation = verified_validation(
                        validation_instances,
                        lambda instance, seed: validation_solve(
                            policy, instance, seed
                        ),
                        seed=validation_seed,
                        objective_config=objective_config,
                    )
                    validation.update(
                        {
                            "validation_wall_time_s": time.perf_counter() - validation_started,
                            "logical_epoch": logical_epoch,
                            "validation_seed": validation_seed,
                            "split": "validation",
                            "decode_type": validation_decode_type,
                            "candidate_count": validation_candidates,
                        }
                    )
                    current_key = validation_key(validation)
                    is_best_overall = current_key > tuple(best_key)
                    is_best_within_minimum = bool(
                        logical_epoch <= minimum_training_epochs
                        and current_key > tuple(best_within_minimum_key)
                    )
                    if is_best_overall:
                        best_key = current_key
                        validation_checks_without_improvement = 0
                    elif logical_epoch > early_stop_start_epoch:
                        validation_checks_without_improvement += 1
                    else:
                        # Pre-gate validation still selects best.ckpt, but it must
                        # not consume patience intended for the mature phase.
                        validation_checks_without_improvement = 0
                    if is_best_within_minimum:
                        best_within_minimum_key = current_key
                    completed_validation_checks += 1
                    early_stop_eligible = logical_epoch > early_stop_start_epoch
                    early_stop_due = bool(
                        early_stop_patience
                        and early_stop_eligible
                        and validation_checks_without_improvement
                        >= early_stop_patience
                    )
                    validation.update(
                        {
                            # The formal alias follows the best checkpoint over
                            # the complete run, including the optional
                            # post-minimum early-stopping tail.  The fixed-budget
                            # selection remains available under its explicit
                            # best_within_5000 name.
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
                    append_jsonl(output / "validation_history.jsonl", validation)
                    extra = {
                        "logical_epoch": logical_epoch,
                        "best_validation_key": list(best_key),
                        "best_within_minimum_key": list(best_within_minimum_key),
                        "validation_checks_without_improvement": validation_checks_without_improvement,
                        "completed_validation_checks": completed_validation_checks,
                        "ema_cost": ema_cost,
                        "baseline_eval_count": baseline_eval_count,
                        "baseline_update_count": baseline_update_count,
                        "pilot_partial_pass": logical_epoch < fixed_epochs,
                    }
                    epoch_checkpoint = (
                        output / f"checkpoint_epoch_{logical_epoch:04d}.pt"
                    )
                    checkpoint_completed_passes = (
                        data_pass if logical_epoch == fixed_epochs else 0
                    )
                    _save_checkpoint(
                        epoch_checkpoint,
                        method=method,
                        data_pass=checkpoint_completed_passes,
                        policy=policy,
                        baseline=baseline,
                        optimizer=optimizer,
                        args=args,
                        extra=extra,
                    )
                    shutil.copy2(epoch_checkpoint, checkpoint)
                    if is_best_overall:
                        shutil.copy2(epoch_checkpoint, best_overall_checkpoint)
                        shutil.copy2(epoch_checkpoint, selected_checkpoint)
                        shutil.copy2(epoch_checkpoint, best_checkpoint)
                        atomic_json(validation_summary_overall, validation)
                        atomic_json(validation_summary, validation)
                    if is_best_within_minimum:
                        shutil.copy2(epoch_checkpoint, best_within_minimum_checkpoint)
                        atomic_json(validation_summary_within_minimum, validation)
                    state.completed_data_passes = checkpoint_completed_passes
                    state.instances_seen = starting_state_instances + instances_seen
                    state.customer_exposures = (
                        starting_state_exposures
                        + instances_seen * _customer_count(args.scale)
                    )
                    state.optimizer_steps = optimizer_steps
                    state.environment_transitions = (
                        environment_transitions_total + transition_count
                    )
                    state.last_checkpoint = str(checkpoint)
                    state.atomic_write(output / "data_pass_state.json")
                    policy.train()
                    if early_stop_due and logical_epoch < fixed_epochs:
                        early_stopped = True
                        early_stop_epoch = logical_epoch
                        complete_pass = False
                        break

        if instances_seen == 0:
            raise RuntimeError("data pass yielded no training instances")
        if fixed_epochs is not None:
            if (
                optimizer_steps - starting_optimizer_steps
                != terminal_logical_epoch - completed_logical_epochs
            ):
                raise RuntimeError(
                    "logical epoch count does not match optimizer update count"
                )
            expected_instances = (terminal_logical_epoch - completed_logical_epochs) * effective
            if instances_seen != expected_instances:
                raise RuntimeError(
                    f"incomplete fixed-epoch budget: {instances_seen} != {expected_instances}"
                )
        elif complete_pass and instances_seen != len(pool):
            raise RuntimeError(f"incomplete data pass: {instances_seen} != {len(pool)}")

        # Legacy complete-data-pass mode has no publication-equivalent update
        # unit, so retain its historical pass-end comparison. Fixed-update
        # protocol runs use the paper schedules inside the optimizer loop.
        baseline_updated = False
        paired_t_pvalue: float | None = None
        baseline_probe = baseline_probe_instances
        if fixed_epochs is None and baseline_probe:
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
                baseline_update_count += 1
                baseline_probe_instances = list(
                    pool.sample(min(baseline_probe_size, len(pool)))
                )
            baseline_eval_count += 1

        validation: dict[str, Any] | None = None
        should_validate = bool(
            fixed_epochs is None
            and validation_instances
            and (
                data_pass % int(args.validation_every_passes) == 0
                or data_pass == total_passes
            )
        )
        if should_validate:
            policy.eval()
            validation_started = time.perf_counter()
            validation = verified_validation(
                validation_instances,
                lambda instance, seed: validation_solve(policy, instance, seed),
                seed=validation_seed,
                objective_config=objective_config,
            )
            validation.update(
                {
                    "validation_wall_time_s": time.perf_counter() - validation_started,
                    "data_pass": data_pass,
                    "validation_seed": validation_seed,
                    "split": "validation",
                    "decode_type": validation_decode_type,
                    "candidate_count": validation_candidates,
                }
            )
            append_jsonl(output / "validation_history.jsonl", validation)

        is_best = bool(
            validation is not None and validation_key(validation) > tuple(best_key)
        )
        if is_best:
            best_key = validation_key(validation)
        extra = {
            "best_validation_key": list(best_key),
            "best_within_minimum_key": list(best_within_minimum_key),
            "ema_cost": ema_cost,
            "baseline_eval_count": baseline_eval_count,
            "baseline_update_count": baseline_update_count,
            "pilot_partial_pass": not complete_pass,
            "logical_epoch": (
                terminal_logical_epoch if fixed_epochs is not None else None
            ),
            "validation_checks_without_improvement": (
                validation_checks_without_improvement
            ),
            "completed_validation_checks": completed_validation_checks,
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
            shutil.copy2(checkpoint, best_checkpoint)
            atomic_json(output / "validation_summary.json", validation)
        if not selected_checkpoint.exists() and args.pilot_mode:
            shutil.copy2(checkpoint, selected_checkpoint)

        observed_steps = np.asarray(trajectory_steps, dtype=np.int64)
        trajectory_count = int(observed_steps.size)
        row = {
            "schema": "drl_data_pass_history_v1",
            "method": method,
            "protocol_id": args.protocol_id,
            "data_pass": data_pass,
            "pass_complete": complete_pass,
            "budget_mode": (
                "fixed_customer_exposure" if stream_path is not None else
                ("fixed_logical_epochs" if fixed_epochs is not None else "complete_data_passes")
            ),
            "training_epochs": fixed_epochs,
            "logical_environments_per_epoch": effective if fixed_epochs is not None else None,
            "training_stage": training_stage,
            "instances_seen": instances_seen,
            "customer_exposures": instances_seen * _customer_count(args.scale),
            "optimizer_steps_total": optimizer_steps,
            "environment_transitions": transition_count,
            "environment_transitions_total": environment_transitions_total + transition_count,
            "physical_batch_size": physical,
            "training_rollout_steps": int(args.training_rollout_steps),
            "soft_stage_end_epoch": soft_stage_end_epoch,
            "soft_stage_fraction": (
                soft_stage_contract["soft_stage_fraction"]
                if soft_stage_contract is not None else None
            ),
            "soft_stage_contract_snapshot": soft_stage_contract,
            "trajectory_count": trajectory_count,
            "mean_trajectory_steps": float(observed_steps.mean()),
            "trajectory_steps_p50": float(np.quantile(observed_steps, 0.50)),
            "trajectory_steps_p90": float(np.quantile(observed_steps, 0.90)),
            "trajectory_steps_p99": float(np.quantile(observed_steps, 0.99)),
            "trajectory_steps_max": int(observed_steps.max()),
            "rollout_budget_exhausted_count": rollout_budget_exhausted_count,
            "rollout_budget_exhausted_rate": (
                rollout_budget_exhausted_count / trajectory_count
            ),
            "terminal_outcome_reason_counts": dict(
                sorted(failure_reason_counts.items())
            ),
            "effective_batch_size": effective,
            "mean_loss": sums["loss"] / instances_seen,
            "mean_training_cost": sums["cost"] / instances_seen,
            "mean_objective_distance_km": sums["distance"] / instances_seen,
            "objective_mode": objective_config.mode,
            "objective_unit": objective_config.unit,
            "mean_objective_value": sums["objective"] / instances_seen,
            "mean_objective_cost_usd": sums["objective"] / instances_seen if objective_config.is_cost else None,
            "mean_electricity_cost_usd": sums["distance"] / instances_seen * objective_config.distance_unit_cost if objective_config.is_cost else None,
            "mean_vehicle_cost_usd": sums["vehicles_started"] / instances_seen * objective_config.vehicle_unit_cost if objective_config.is_cost else None,
            "mean_vehicles_started": sums["vehicles_started"] / instances_seen,
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
            state.instances_seen = starting_state_instances + instances_seen
            state.customer_exposures = (
                starting_state_exposures
                + instances_seen * _customer_count(args.scale)
            )
            state.optimizer_steps = optimizer_steps
            state.environment_transitions = environment_transitions_total
            state.last_checkpoint = str(checkpoint)
            state.atomic_write(output / "data_pass_state.json")
        else:
            break

    # Re-publish the formal aliases from their canonical overall artifacts at
    # the terminal boundary.  Besides making the contract explicit, this also
    # repairs aliases created by an older pre-tail-selection implementation
    # when such a run is resumed with the current code.
    if fixed_epochs is not None and best_overall_checkpoint.is_file():
        if not validation_summary_overall.is_file():
            raise RuntimeError(
                "best_overall.ckpt exists without validation_summary_overall.json"
            )
        shutil.copy2(best_overall_checkpoint, selected_checkpoint)
        shutil.copy2(best_overall_checkpoint, best_checkpoint)
        atomic_json(
            validation_summary,
            json.loads(validation_summary_overall.read_text(encoding="utf-8")),
        )

    final_validation_limit = int(
        getattr(args, "final_validation_limit", 0) or 0
    )
    final_validation_path = output / "validation_final_audit.json"
    if fixed_epochs is not None and final_validation_limit > 0:
        if validation_pool is None or not best_checkpoint.is_file():
            raise RuntimeError(
                "final validation audit requires a validation pool and best.ckpt"
            )
        _load_checkpoint(
            best_checkpoint,
            policy=policy,
            baseline=baseline,
            optimizer=optimizer,
            protocol_id=args.protocol_id,
            objective_config=objective_config,
            optimizer_name=getattr(args, "optimizer", None),
            optimizer_weight_decay=getattr(args, "weight_decay", None),
            reward_contract_args=args,
        )
        policy.eval()
        final_validation = verified_validation(
            validation_pool.first(limit=final_validation_limit),
            lambda instance, seed: validation_solve(policy, instance, seed),
            seed=int(args.seed) + 999_000_000,
            objective_config=objective_config,
        )
        if int(final_validation["instances"]) != final_validation_limit:
            raise RuntimeError(
                "final validation audit did not consume the registered view count: "
                f"{final_validation['instances']} != {final_validation_limit}"
            )
        selected_summary = json.loads(
            validation_summary.read_text(encoding="utf-8")
        )
        final_validation.update(
            {
                "schema": "drl_final_validation_audit_v1",
                "split": "validation",
                "selection_checkpoint": str(best_checkpoint),
                "selection_logical_epoch": selected_summary.get("logical_epoch"),
                "selection_changed": False,
                "decode_type": validation_decode_type,
                "candidate_count": validation_candidates,
            }
        )
        atomic_json(final_validation_path, final_validation)

    terminal = {
        "schema": "drl_training_result_v1",
        "status": "pilot_partial" if args.pilot_mode else ("early_stopped" if early_stopped else "passed"),
        "method": method,
        "protocol_id": args.protocol_id,
        "objective_config": objective_config.to_dict(),
        "objective_mode": objective_config.mode,
        "objective_unit": objective_config.unit,
        "reward_diagnostics": str(output / "reward_diagnostics.jsonl"),
        "reward_diagnostics_schema": "drl_reward_diagnostics_v1",
        "reward_distance_scale_km": getattr(args, "reward_distance_scale_km", None),
        "reward_distance_scale_mode": getattr(args, "reward_distance_scale_mode", None),
        "reward_distance_scale_metadata": getattr(args, "reward_distance_scale_metadata", None),
        "reward_contract_id": getattr(args, "reward_contract_id", None),
        "reward_contract_sha256": getattr(args, "reward_contract_sha256", None),
        "reward_contract_scale": getattr(args, "reward_contract_scale", None),
        "reward_contract_snapshot": getattr(args, "reward_contract_snapshot", None),
        "reward_objective_scale": getattr(args, "reward_objective_scale", None),
        "reward_failure_base": getattr(args, "reward_failure_base", None),
        "reward_unserved_coefficient": getattr(
            args, "reward_unserved_coefficient", None
        ),
        "method_auxiliary_profile_id": getattr(
            args, "method_auxiliary_profile_id", None
        ),
        "method_auxiliary_sha256": getattr(
            args, "method_auxiliary_sha256", None
        ),
        "method_auxiliary_snapshot": getattr(
            args, "method_auxiliary_snapshot", None
        ),
        "method_auxiliary_method": getattr(
            args, "method_auxiliary_method", None
        ),
        "method_auxiliary_applicability": getattr(
            args, "method_auxiliary_applicability", None
        ),
        "method_auxiliary_aggregation": getattr(
            args, "method_auxiliary_aggregation", None
        ),
        "method_auxiliary_denominator": getattr(
            args, "method_auxiliary_denominator", None
        ),
        "method_auxiliary_step_clip": getattr(
            args, "method_auxiliary_step_clip", None
        ),
        "method_auxiliary_component_clip": getattr(
            args, "method_auxiliary_component_clip", None
        ),
        "method_auxiliary_weights": getattr(
            args, "method_auxiliary_weights", None
        ),
        "soft_violation_contract_id": getattr(
            args, "soft_violation_contract_id", None
        ),
        "soft_violation_step_clip": getattr(
            args, "soft_violation_step_clip", None
        ),
        "soft_violation_component_clip": getattr(
            args, "soft_violation_component_clip", None
        ),
        "soft_violation_denominator": getattr(
            args, "soft_violation_denominator", None
        ),
        "budget_mode": (
            "fixed_customer_exposure" if stream_path is not None else
            ("fixed_logical_epochs" if fixed_epochs is not None else "complete_data_passes")
        ),
        "requested_training_epochs": fixed_epochs,
        "completed_training_epochs": terminal_logical_epoch if fixed_epochs is not None else None,
        "early_stopped": early_stopped,
        "early_stop_epoch": early_stop_epoch,
        "logical_environments_per_epoch": effective if fixed_epochs is not None else None,
        "requested_data_passes": int(args.data_passes) if args.data_passes is not None else None,
        "completed_data_passes": int(state.completed_data_passes),
        "training_rollout_steps": int(args.training_rollout_steps),
        "soft_stage_end_epoch": soft_stage_end_epoch,
        "soft_stage_fraction": (
            soft_stage_contract["soft_stage_fraction"]
            if soft_stage_contract is not None else None
        ),
        "soft_stage_contract_snapshot": soft_stage_contract,
        "instances_seen": int(state.instances_seen),
        "customer_exposures": int(state.customer_exposures),
        "training_stream_path": str(stream_path) if stream_path is not None else None,
        "training_stream_contract_sha256": getattr(
            args, "training_stream_contract_sha256", None
        ),
        "training_stream_contract_snapshot": getattr(
            args, "training_stream_contract_snapshot", None
        ),
        "stream_integrity_mode": getattr(args, "stream_integrity_mode", None),
        "resolved_training_signature_sha256": getattr(
            args, "resolved_training_signature_sha256", None
        ),
        "resolved_training_signature": getattr(
            args, "resolved_training_signature", None
        ),
        "warm_start_requested": warm_start_checkpoint is not None,
        "warm_start_provenance": warm_start_provenance,
        "optimizer_steps": int(optimizer_steps),
        "baseline_eval_count": baseline_eval_count,
        "baseline_update_count": baseline_update_count,
        "reinforce_baseline": reinforce_baseline,
        "environment_transitions": int(environment_transitions_total),
        "saved_exposure_checkpoints": sorted(saved_exposure),
        "saved_gpu_hour_checkpoints": sorted(saved_gpu_hours),
        "selected_checkpoint": str(selected_checkpoint if selected_checkpoint.exists() else checkpoint),
        "best_checkpoint": str(best_checkpoint if best_checkpoint.exists() else selected_checkpoint),
        "best_within_5000_checkpoint": str(best_within_minimum_checkpoint),
        "best_overall_checkpoint": str(best_overall_checkpoint),
        "minimum_training_epochs": minimum_training_epochs if fixed_epochs is not None else None,
        "validation_every_epochs": (
            validation_every_epochs if fixed_epochs is not None else None
        ),
        "post_minimum_validation_every_epochs": (
            post_minimum_validation_every_epochs if fixed_epochs is not None else None
        ),
        "scheduled_validation_epochs": list(scheduled_validation_epochs),
        "validation_checkpoints": int(args.validation_checkpoints),
        "completed_validation_checkpoints": completed_validation_checks,
        "early_stop_patience_validations": early_stop_patience,
        "early_stop_start_epoch": early_stop_start_epoch,
        "final_validation_limit": final_validation_limit,
        "validation_decode_type": validation_decode_type,
        "validation_candidates": validation_candidates,
        "validation_seed": validation_seed,
        "final_validation_audit": (
            str(final_validation_path) if final_validation_limit > 0 else None
        ),
        "peak_gpu_memory_bytes": _peak_gpu_bytes(args.device),
        "wall_time_s": time.perf_counter() - run_started,
    }
    atomic_json(output / "training_result.json", terminal)


__all__ = [
    "paper_baseline_eval_due",
    "paper_ema_baseline_due",
    "train_reinforce_data_passes",
]
