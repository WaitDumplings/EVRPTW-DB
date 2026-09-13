"""Opt-in synchronous REINFORCE; the existing single-GPU loop is unchanged.

Supported adapters are AM, RRNCO-EV and the two-stage DRL-TS. Every rank owns
a full replica, consumes a disjoint global-stream shard and sums normalized
gradients. Validation is sharded by contiguous instance positions, preserving
the original per-instance seeds and independent solution verification.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import tempfile
import time
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from scipy.stats import ttest_rel

from .data_pass import DataPassState
from .distributed import DistributedContext
from .protocol_trainers import (
    _append_reinforce_diagnostics, _collect_reinforce_diagnostics,
    _customer_count, _load_checkpoint, _load_warm_start_checkpoint,
    _save_checkpoint, _resolve_soft_stage_contract, paper_baseline_eval_due,
    paper_ema_baseline_due, same_instance_leave_one_out, prepare_training_objective,
)
from .training_protocol import (
    append_jsonl, atomic_json, assert_checkpoint_training_signature, load_state,
    make_validation_pool, require_registered_batches,
    require_validation_decoding, require_validation_rollout_steps,
    validation_epochs, validation_key, verified_validation,
)
from .training_stream import read_stream_view_ids


def configure_distributed_contract(args: Any, context: DistributedContext, *,
                                   method: str = "AM-EVRPTW") -> dict[str, Any]:
    physical, effective = require_registered_batches(args, int(args.batch_size))
    if effective % (physical * context.world_size):
        raise ValueError("effective batch must be divisible by physical batch times world size")
    contract = {
        "schema": "drl_synchronous_reinforce_v1",
        "world_size": context.world_size,
        "physical_batch_size_per_rank": physical,
        "effective_batch_size_global": effective,
        "microbatches_per_rank": effective // (physical * context.world_size),
        "gradient_reduction": "sum_with_global_environment_denominator",
        "batch_norm": "local_forward_rank_zero_buffers_after_each_update",
        "ema": "global_trajectory_mean_per_synchronous_microbatch",
        "validation": "contiguous_instance_shards_preserving_global_seeds",
        "resume_topology": "fixed_world_size_and_batch_contract",
    }
    if method in {"RRNCO-EV", "DRL-TS"}:
        baseline = getattr(args, "reinforce_baseline", "paper")
        contract["method"] = method
        contract["reinforce_baseline"] = baseline
        if baseline == "leave_one_out":
            contract["ema"] = "disabled_same_instance_leave_one_out"
            contract["baseline_scope"] = "other_trajectories_of_the_same_instance_on_the_same_rank"
        elif method == "DRL-TS":
            contract["ema"] = "disabled_native_greedy_rollout"
            contract["baseline_scope"] = "matching_soft_or_hard_stage_with_native_update_interval"
    args.distributed_training = True
    args.distributed_contract = contract
    # Keep the already-deployed AM signature byte-for-byte compatible. New
    # adapters retain architecture/graph/auxiliary metadata resolved by train.py.
    if method == "AM-EVRPTW":
        args.resolved_training_method_fields = {
            name: getattr(args, name, None) for name in (
                "learning_rate", "max_grad_norm", "embedding_dim", "n_encode_layers",
                "n_heads", "tanh_clipping", "steps_per_epoch", "baseline_warmup_epochs",
                "baseline_eval_size", "baseline_alpha", "ema_decay",
                "incomplete_penalty_km",
            )
        }
    elif method in {"RRNCO-EV", "DRL-TS"}:
        fields = deepcopy(getattr(args, "resolved_training_method_fields", None) or {})
        fields.update({name: getattr(args, name) for name in (
            "learning_rate", "max_grad_norm", "embedding_dim", "n_encode_layers",
            "n_heads", "tanh_clipping", "baseline_eval_size", "baseline_alpha",
            "incomplete_penalty",
        ) if hasattr(args, name)})
        fields["method"] = method
        if method == "RRNCO-EV":
            fields.update({name: getattr(args, name) for name in (
                "graph_mode", "aft_mode", "distance_sampling", "relation_chunk_size",
                "checkpoint_bias", "relation_temperature", "feedforward_hidden",
                "distance_sample_size", "reinforce_baseline", "steps_per_epoch",
                "baseline_warmup_epochs", "ema_decay",
            ) if hasattr(args, name)})
        else:
            fields.update({name: getattr(args, name) for name in (
                "nearest_neighbors", "activation_checkpoint_stride", "batches_per_epoch",
                "capacity_penalty", "time_penalty", "energy_penalty",
                "soft_violation_contract_id", "soft_violation_step_clip",
                "soft_violation_component_clip", "soft_violation_denominator",
            ) if hasattr(args, name)})
            if getattr(args, "soft_stage_contract_snapshot", None) is not None:
                fields["soft_stage_contract"] = deepcopy(args.soft_stage_contract_snapshot)
        args.resolved_training_method_fields = fields
    else:
        raise ValueError(f"unsupported distributed adapter: {method}")
    return contract


def shard_stream_epoch(view_ids: Sequence[str], *, logical_epoch: int,
                       physical_batch_size: int, effective_batch_size: int,
                       rank: int, world_size: int) -> list[list[str]]:
    """Partition one global batch without loading other ranks' instances."""
    physical, effective = int(physical_batch_size), int(effective_batch_size)
    if world_size < 1 or not 0 <= rank < world_size or physical < 1:
        raise ValueError("invalid rank/world size/physical batch")
    if effective <= 0 or effective % (physical * world_size):
        raise ValueError("effective batch must be divisible by physical batch times world size")
    begin = (int(logical_epoch) - 1) * effective
    if begin < 0 or begin + effective > len(view_ids):
        raise ValueError("logical epoch is outside the registered stream")
    return [list(view_ids[offset + rank * physical:offset + (rank + 1) * physical])
            for offset in range(begin, begin + effective, physical * world_size)]


def merge_validation_summaries(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Weight cost means by successful instances, never by shard size."""
    if not summaries:
        raise ValueError("no validation shards")
    result = dict(summaries[0])
    for summary in summaries[1:]:
        for key in ("schema", "objective_mode", "objective_config", "objective_unit"):
            if summary.get(key) != result.get(key):
                raise ValueError(f"validation shards disagree on {key}")
    instances = sum(int(part["instances"]) for part in summaries)
    passed = sum(int(part["complete_and_feasible"]) for part in summaries)
    result.update(instances=instances, complete_and_feasible=passed,
                  complete_and_feasible_rate=passed / max(instances, 1),
                  verifier_summary_passed=instances > 0 and passed == instances,
                  rows=[row for part in summaries for row in part.get("rows", [])])
    mean_keys = {key for part in summaries for key in part if key.startswith("mean_verified_")}
    for key in mean_keys:
        populated = [part for part in summaries if int(part["complete_and_feasible"]) > 0]
        values = [part.get(key) for part in populated]
        if not passed or all(value is None for value in values):
            result[key] = None
        elif any(value is None for value in values):
            raise ValueError(f"incomplete successful-shard statistic: {key}")
        else:
            result[key] = sum(float(part[key]) * int(part["complete_and_feasible"])
                              for part in populated) / passed
    return result


def capture_rank_rng(pool: Any, device: str) -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state(device).cpu()
                       if str(device).startswith("cuda") else None),
        "pool": deepcopy(pool.rng.bit_generator.state) if hasattr(pool, "rng") else None,
    }


def restore_rank_rng(state: dict[str, Any], pool: Any, device: str) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda") is not None and str(device).startswith("cuda"):
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device)
    if state.get("pool") is not None:
        pool.rng.bit_generator.state = state["pool"]


def atomic_copy(source: Path, destination: Path) -> None:
    """Publish a checkpoint alias only after its replacement has been fully copied."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".copy.tmp", dir=destination.parent)
    os.close(descriptor)
    try:
        shutil.copy2(source, temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def rollback_uncommitted_logs(output: Path, completed_epoch: int) -> None:
    """Back up and discard observations beyond the last durable optimizer state."""
    backup = output / f"resume_uncommitted_history_{time.time_ns()}"
    for name in ("logical_epoch_history.jsonl", "reward_diagnostics.jsonl",
                 "sampled_view_ids.jsonl", "baseline_history.jsonl", "validation_history.jsonl"):
        path = output / name
        if not path.is_file():
            continue
        original = path.read_text()
        lines = original.splitlines()
        retained = []
        changed = False
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise ValueError(f"corrupt committed history: {path}")
                changed = True
                continue
            epoch = row.get("logical_epoch", row.get("optimizer_step"))
            if epoch is None:
                raise ValueError(f"history row has no epoch: {path}")
            if int(epoch) <= completed_epoch:
                retained.append(line)
            else:
                changed = True
        if changed:
            backup.mkdir(exist_ok=True)
            shutil.copy2(path, backup / name)
            temporary = path.with_suffix(path.suffix + ".resume.tmp")
            temporary.write_text("".join(line + "\n" for line in retained))
            temporary.replace(path)


def _merge_diagnostic_parts(parts):
    merged = [{}, {}, {}]
    for part in parts:
        for target, group in zip(merged, part):
            for key, arrays in group.items():
                target.setdefault(key, []).extend(arrays)
    return merged


def train_distributed_reinforce_data_passes(
    *, method: str, args: Any, pool: Any, policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    make_actor: Callable[[list[Any], bool, int], Any],
    make_baseline: Callable[[torch.nn.Module, list[Any], bool, int], Any],
    training_cost: Callable[[Any], torch.Tensor],
    objective_distance: Callable[[Any], torch.Tensor],
    feasible: Callable[[Any], torch.Tensor],
    validation_solve: Callable[[torch.nn.Module, Any, int], dict[str, Any]],
    legacy_batch_size: int, soft_stage_fraction: float = 0.0,
    soft_stage_end_epoch: int | None = None,
) -> None:
    context = DistributedContext.current()
    with context.local_phase("protocol configuration"):
        if method not in {"AM-EVRPTW", "RRNCO-EV", "DRL-TS"}:
            raise ValueError(f"unsupported distributed adapter: {method}")
        if method != "DRL-TS" and (soft_stage_fraction or soft_stage_end_epoch is not None):
            raise ValueError("soft constraints are supported only by DRL-TS")
        reinforce_baseline = getattr(args, "reinforce_baseline", "paper")
        use_leave_one_out = reinforce_baseline == "leave_one_out"
        if reinforce_baseline not in {"paper", "leave_one_out"}:
            raise ValueError(f"unsupported REINFORCE baseline: {reinforce_baseline}")
        if use_leave_one_out and method != "RRNCO-EV":
            raise ValueError("distributed leave_one_out is supported only for RRNCO-EV")
        if use_leave_one_out and int(args.samples_per_instance) < 2:
            raise ValueError("leave_one_out requires at least two trajectories per instance")
        if getattr(args, "data_passes", None) is not None or not getattr(args, "training_epochs", None):
            raise ValueError("distributed training requires fixed --training-epochs")
        if getattr(args, "max_batches_per_pass", None) is not None:
            raise ValueError("distributed training does not use max-batches-per-pass")
        if not getattr(args, "training_stream_path", None):
            raise ValueError("distributed training requires an explicit training stream")
        if getattr(args, "exposure_checkpoints", "") or getattr(args, "gpu_hour_checkpoints", ""):
            raise ValueError("use epoch validation checkpoints for distributed training")
        soft_stage_contract = _resolve_soft_stage_contract(
            method=method, fixed_epochs=int(args.training_epochs), total_passes=1,
            soft_stage_fraction=soft_stage_fraction, soft_stage_end_epoch=soft_stage_end_epoch)
        # AM historically did not attach this optional field during setup.
        if method != "AM-EVRPTW":
            args.soft_stage_contract_snapshot = soft_stage_contract
        contract = configure_distributed_contract(args, context, method=method)
        validation_decode, validation_candidates = require_validation_decoding(args)
        require_validation_rollout_steps(args)
        objective = prepare_training_objective(args)
        physical, effective = require_registered_batches(args, legacy_batch_size)
        epochs = int(args.training_epochs)
        if epochs < 1:
            raise ValueError("training epochs must be positive")
        minimum = int(getattr(args, "minimum_training_epochs", None) or epochs)
        interval = int(getattr(args, "validation_every_epochs", None) or epochs)
        post_interval = int(getattr(args, "post_minimum_validation_every_epochs", None) or interval)
        scheduled = validation_epochs(epochs, initial_interval=interval,
                                      minimum_epochs=minimum, post_minimum_interval=post_interval)
        if int(args.validation_checkpoints) != len(scheduled):
            raise ValueError("validation checkpoint count differs from the configured epoch schedule")
        patience = int(getattr(args, "early_stop_patience_validations", 0) or 0)
        early_start = int(getattr(args, "early_stop_start_epoch", 0) or 0)
        if patience < 0 or early_start < 0 or (patience and early_start < minimum):
            raise ValueError("invalid early-stop patience/start; stopping cannot precede minimum epochs")
        if early_start >= epochs:
            raise ValueError("early-stop start must precede maximum training epochs")
        customers = _customer_count(args.scale)
        if int(getattr(args, "customer_exposure_budget", 0) or 0) != epochs * effective * customers:
            raise ValueError("customer-exposure budget must equal epochs * global effective batch * customers")
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        state = load_state(output, args.protocol_id, bool(args.resume))
        checkpoint = output / "checkpoint_latest.pt"
        baseline = deepcopy(policy).eval()
        for parameter in baseline.parameters():
            parameter.requires_grad_(False)
        payload = {}
        warm_start = None
        if args.resume:
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            payload = _load_checkpoint(checkpoint, policy=policy, baseline=baseline,
                optimizer=optimizer, protocol_id=args.protocol_id, objective_config=objective,
                optimizer_name=getattr(args, "optimizer", None),
                optimizer_weight_decay=getattr(args, "weight_decay", None), reward_contract_args=args)
            if payload.get("distributed_contract") != contract:
                raise ValueError("distributed checkpoint topology/batch contract differs")
        elif checkpoint.exists():
            raise FileExistsError(f"existing checkpoint requires --resume: {checkpoint}")
        elif getattr(args, "warm_start_checkpoint", None) is not None:
            warm_start = _load_warm_start_checkpoint(Path(args.warm_start_checkpoint),
                method=method, policy=policy, baseline=baseline,
                objective_config=objective, contract_args=args)
        completed = int(payload.get("logical_epoch", 0))
        cursor = completed * effective
        if not 0 <= completed <= epochs:
            raise ValueError("checkpoint logical epoch is outside the requested budget")
        if args.resume:
            if int(payload.get("stream_cursor", -1)) != cursor:
                raise ValueError("checkpoint epoch and global stream cursor disagree")
            embedded_state = payload.get("data_pass_state")
            if not isinstance(embedded_state, dict):
                raise ValueError("distributed checkpoint has no authoritative data-pass state")
            restored_state = DataPassState(**embedded_state)
            if (restored_state.protocol_id != args.protocol_id or restored_state.optimizer_steps != completed
                    or restored_state.instances_seen != cursor
                    or restored_state.customer_exposures != cursor * customers):
                raise ValueError("embedded checkpoint state disagrees with the global stream cursor")
            if state.optimizer_steps > completed or state.instances_seen > cursor:
                raise ValueError("data-pass state is ahead of the last durable checkpoint")
            # checkpoint_latest.pt is atomically published BEFORE its JSON sidecar.
            # A crash in that window can leave a lagging sidecar, which is repairable.
            state = restored_state
        view_ids = read_stream_view_ids(args.training_stream_path, stop=epochs * effective)
        task_map = pool._task_by_view_id
        missing = set(view_ids).difference(task_map)
        if missing:
            raise ValueError(f"training stream contains IDs outside the filtered pool: {sorted(missing)[:3]}")
        validation_pool = make_validation_pool(args, scale=args.scale, seed=args.seed)
        if validation_pool is not None and hasattr(args, "instance_cache_size"):
            validation_pool.cache_size = int(args.instance_cache_size)
        validation_tasks = ([] if validation_pool is None else
                            list(validation_pool.tasks[:int(args.validation_limit)]))
        if validation_pool is not None and len(validation_tasks) != int(args.validation_limit):
            raise ValueError("validation pool is smaller than the requested fixed cohort")
        probe_size = (0 if use_leave_one_out else
                      min(max(0, int(getattr(args, "baseline_eval_size", 64))), len(pool)))
        probe_ids = list(payload.get("baseline_probe_view_ids",
                                    [task.view_id for task in pool.tasks[:probe_size]]))
        if len(probe_ids) != probe_size or any(view_id not in task_map for view_id in probe_ids):
            raise ValueError("checkpoint baseline probe does not match the training pool")
        if args.resume:
            rng_states = payload.get("rank_rng_states", [])
            if len(rng_states) != context.world_size:
                raise ValueError("checkpoint is missing per-rank RNG states")
            restore_rank_rng(rng_states[context.rank], pool, args.device)
    if args.resume:
        def repair_resume_artifacts():
            rollback_uncommitted_logs(output, completed)
            state.atomic_write(output / "data_pass_state.json")
            for summary_name, checkpoint_names, json_names in (
                ("best_validation_summary", ("best.ckpt", "best_overall.ckpt", "checkpoint_selected.pt"),
                 ("validation_summary.json", "validation_summary_overall.json")),
                ("best_within_minimum_summary", ("best_within_5000.ckpt",),
                 ("validation_summary_within_5000.json",)),
            ):
                summary = payload.get(summary_name)
                if summary is None:
                    continue
                selected_epoch = int(summary["logical_epoch"])
                source = checkpoint if selected_epoch == completed else output / f"checkpoint_epoch_{selected_epoch:04d}.pt"
                if not source.is_file():
                    raise FileNotFoundError(f"cannot repair selected checkpoint alias: {source}")
                selected_payload = torch.load(source, map_location="cpu", weights_only=False)
                assert_checkpoint_training_signature(selected_payload, args)
                if int(selected_payload.get("logical_epoch", -1)) != selected_epoch:
                    raise ValueError("selected checkpoint epoch disagrees with its validation summary")
                for name in checkpoint_names:
                    atomic_copy(source, output / name)
                for name in json_names:
                    atomic_json(output / name, summary)
        context.main_call(repair_resume_artifacts)
    context.broadcast_model(policy)
    context.broadcast_model(baseline)
    identity = {"rank": context.rank, "pid": os.getpid(), "device": args.device}
    if hasattr(args, "instance_cache_size"):
        identity["host_instance_cache_size"] = int(args.instance_cache_size)
    initial_identities = context.gather_objects(identity)
    best_key = tuple(payload.get("best_validation_key", [-math.inf, -math.inf]))
    best_minimum = tuple(payload.get("best_within_minimum_key", [-math.inf, -math.inf]))
    best_summary = payload.get("best_validation_summary")
    best_minimum_summary = payload.get("best_within_minimum_summary")
    ema_cost = payload.get("ema_cost")
    checks_without_improvement = int(payload.get("validation_checks_without_improvement", 0))
    validation_checks = int(payload.get("completed_validation_checks", 0))
    baseline_evals = int(payload.get("baseline_eval_count", 0))
    baseline_updates = int(payload.get("baseline_update_count", 0))
    environment_transitions = int(state.environment_transitions)
    early_stopped = bool(payload.get("early_stopped", False))
    terminal_epoch = completed
    started = time.perf_counter()
    previous_wall = float(payload.get("total_wall_time_s", 0.0))
    session = {"session_id": str(time.time_ns()), "resume_requested": bool(args.resume),
               "session_start_optimizer_steps": completed,
               "distributed_world_size": context.world_size,
               "diagnostic_scope": "all_rank_candidate_trajectories"}
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    context.main_call(lambda: atomic_json(output / "distributed_workers.json", {
        "contract": contract, "workers": initial_identities,
        "resume": bool(args.resume), "stream_cursor": cursor}))

    def validate(tasks, seed):
        context.broadcast_buffers(policy)
        policy.eval()
        began = time.perf_counter()
        begin = len(tasks) * context.rank // context.world_size
        end = len(tasks) * (context.rank + 1) // context.world_size
        with context.local_phase("validation shard"):
            local = verified_validation(
                (validation_pool.instance(task) for task in tasks[begin:end]),
                lambda instance, item_seed: validation_solve(policy, instance, item_seed),
                seed=int(seed) + begin, objective_config=objective,
                cuda_rng_devices=([torch.device(args.device).index or 0]
                                  if str(args.device).startswith("cuda") else []))
        parts = context.gather_objects(local)
        combined = context.main_call(lambda: merge_validation_summaries(parts))
        combined["validation_wall_time_s"] = context.max_values([time.perf_counter() - began])[0]
        combined["validation_shards"] = context.world_size
        combined["validation_seed"] = int(seed)
        policy.train()
        return combined

    def save(epoch, *, epoch_artifact=False, selected=False, selected_minimum=False, validation=None):
        rank_rng = context.gather_objects(capture_rank_rng(pool, args.device))
        elapsed = context.max_values([time.perf_counter() - started])[0] + previous_wall
        state.optimizer_steps = epoch
        state.instances_seen = epoch * effective
        state.customer_exposures = state.instances_seen * customers
        state.environment_transitions = environment_transitions
        state.completed_data_passes = int(epoch == epochs)
        state.last_checkpoint = str(checkpoint)
        extra = {"logical_epoch": epoch, "stream_cursor": epoch * effective,
                 "distributed_contract": contract, "rank_rng_states": rank_rng,
                 "baseline_probe_view_ids": probe_ids, "ema_cost": ema_cost,
                 "data_pass_state": asdict(state),
                 "best_validation_summary": best_summary,
                 "best_within_minimum_summary": best_minimum_summary,
                 "best_validation_key": list(best_key), "best_within_minimum_key": list(best_minimum),
                 "validation_checks_without_improvement": checks_without_improvement,
                 "completed_validation_checks": validation_checks,
                 "baseline_eval_count": baseline_evals, "baseline_update_count": baseline_updates,
                 "early_stopped": early_stopped, "pilot_partial_pass": epoch < epochs,
                 "total_wall_time_s": elapsed, "total_gpu_hours": elapsed * context.world_size / 3600,
                 "warm_start_provenance": warm_start}
        def write():
            path = output / f"checkpoint_epoch_{epoch:04d}.pt" if epoch_artifact else checkpoint
            _save_checkpoint(path, method=method, data_pass=state.completed_data_passes,
                             policy=policy, baseline=baseline, optimizer=optimizer, args=args, extra=extra)
            if path != checkpoint:
                atomic_copy(path, checkpoint)
            if selected:
                for name in ("best.ckpt", "best_overall.ckpt", "checkpoint_selected.pt"):
                    atomic_copy(checkpoint, output / name)
                atomic_json(output / "validation_summary.json", validation)
                atomic_json(output / "validation_summary_overall.json", validation)
            if selected_minimum:
                atomic_copy(checkpoint, output / "best_within_5000.ckpt")
                atomic_json(output / "validation_summary_within_5000.json", validation)
            state.atomic_write(output / "data_pass_state.json")
        context.main_call(write)

    # A terminal early-stop checkpoint resumes as terminal, without consuming new data.
    epoch_range = () if early_stopped else range(completed + 1, epochs + 1)
    for epoch in epoch_range:
        epoch_started = time.perf_counter()
        soft = bool(soft_stage_contract is not None and
                    epoch <= soft_stage_contract["resolved_soft_stage_end_epoch"])
        policy.train()
        local_sums = {key: 0.0 for key in ("loss", "cost", "distance", "objective", "vehicles", "feasible")}
        local_steps = []
        local_transitions = 0
        local_exhausted = 0
        reasons = Counter()
        distributions, components, scales = {}, {}, {}
        local_ids = []
        optimizer.zero_grad(set_to_none=True)
        shards = shard_stream_epoch(view_ids, logical_epoch=epoch, physical_batch_size=physical,
                                    effective_batch_size=effective, rank=context.rank, world_size=context.world_size)
        context.main_call(lambda: atomic_json(output / "progress.json", {
            "status": "running", "logical_epoch": epoch, "completed_logical_epoch": epoch - 1,
            "phase": "training", "world_size": context.world_size, "workers": initial_identities}))
        for micro_index, ids in enumerate(shards):
            seed = int(args.seed) + 10_000_000 + epoch * 100_000 + micro_index * context.world_size + context.rank
            with context.local_phase("actor rollout"):
                instances = [pool.instance(task_map[view_id]) for view_id in ids]
                local_ids.extend(ids)
                torch.manual_seed(seed)
                actor = make_actor(instances, soft, seed)
                actor_cost = training_cost(actor)
                if not torch.isfinite(actor_cost).all():
                    raise FloatingPointError("non-finite actor training cost")
            use_ema = not use_leave_one_out and paper_ema_baseline_due(method, epoch - 1, args)
            if use_ema:
                cost_sum, cost_count = context.sum_values([float(actor_cost.detach().double().sum().cpu()), actor_cost.numel()])
                observed = cost_sum / cost_count
                ema_cost = observed if ema_cost is None else float(args.ema_decay) * ema_cost + (1 - float(args.ema_decay)) * observed
                baseline_cost = torch.full_like(actor_cost, float(ema_cost))
            with context.local_phase("baseline and backward"):
                if use_leave_one_out:
                    baseline_cost = same_instance_leave_one_out(actor_cost)
                    if actor.log_likelihood.shape != actor_cost.shape:
                        raise ValueError("leave_one_out cost and log_likelihood shapes must match")
                elif not use_ema:
                    with torch.no_grad():
                        reference = make_baseline(baseline, instances, soft, seed)
                        baseline_cost = training_cost(reference)
                    del reference
                advantage = (actor_cost - baseline_cost).detach()
                loss = (advantage * actor.log_likelihood).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite REINFORCE loss")
                (loss * (len(instances) / effective)).backward()
                distance = objective_distance(actor)
                value = getattr(actor, "objective_value", None)
                if value is None:
                    if objective.is_cost:
                        raise ValueError("cost training requires named objective_value")
                    value = distance
                vehicles = getattr(actor, "vehicles_started", torch.zeros_like(distance))
                _collect_reinforce_diagnostics(actor=actor, actor_cost=actor_cost,
                    baseline_cost=baseline_cost, advantage=advantage, raw_distance=distance,
                    active_objective=value, started_vehicles=vehicles, objective_config=objective,
                    distributions=distributions, components=components, scales=scales)
                values = (loss.detach(), actor_cost, distance, value, vehicles, feasible(actor).float())
                for key, tensor in zip(local_sums, values):
                    local_sums[key] += float(tensor.detach().mean().cpu()) * len(instances)
                local_transitions += int(actor.environment_transitions)
                local_steps.extend(actor.trajectory_steps.detach().cpu().reshape(-1).tolist())
                local_exhausted += int(actor.rollout_budget_exhausted.sum().detach().cpu())
                if getattr(actor, "failure_reasons", None) is not None:
                    reasons.update(str(reason) for reason in np.asarray(actor.failure_reasons).reshape(-1))
                del actor, actor_cost, loss, advantage, baseline_cost, instances
        context.sum_gradients(policy.parameters())
        with context.local_phase("optimizer update"):
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            if not torch.isfinite(pre_clip_norm):
                raise FloatingPointError("non-finite synchronized gradient norm")
            optimizer.step()
        context.broadcast_buffers(policy)
        terminal_epoch = epoch
        totals = context.sum_values([*local_sums.values(), local_transitions, local_exhausted])
        sums = dict(zip(local_sums, totals[:len(local_sums)]))
        transitions, exhausted = map(int, totals[-2:])
        environment_transitions += transitions
        grouped_steps = context.gather_objects(local_steps)
        steps = np.asarray([step for part in grouped_steps for step in part], dtype=np.int64)
        global_reasons = Counter()
        for counts in context.gather_objects(dict(reasons)):
            global_reasons.update(counts)
        diagnostics = _merge_diagnostic_parts(context.gather_objects((distributions, components, scales)))
        baseline_updated, pvalue = False, None
        if probe_ids and paper_baseline_eval_due(method, epoch, args):
            policy.eval()
            def compare_baseline():
                actor_costs, reference_costs = [], []
                for probe_index, view_id in enumerate(probe_ids):
                    instance = pool.instance(task_map[view_id])
                    probe_seed = int(args.seed) + 700_000 + epoch * 10_000 + probe_index
                    with torch.no_grad():
                        current = make_baseline(policy, [instance], soft, probe_seed)
                        reference = make_baseline(baseline, [instance], soft, probe_seed)
                    actor_costs.append(float(training_cost(current).mean().cpu()))
                    reference_costs.append(float(training_cost(reference).mean().cpu()))
                test = ttest_rel(actor_costs, reference_costs, alternative="less")
                p = float(test.pvalue)
                updated = bool(np.mean(actor_costs) < np.mean(reference_costs)
                               and np.isfinite(p) and p < float(args.baseline_alpha))
                replacement_ids = probe_ids
                if updated:
                    indices = pool.rng.integers(0, len(pool.tasks), size=probe_size)
                    replacement_ids = [pool.tasks[int(index)].view_id for index in indices]
                return updated, p, replacement_ids
            baseline_updated, pvalue, probe_ids = context.main_call(compare_baseline)
            baseline_evals += 1
            if baseline_updated:
                baseline.load_state_dict(policy.state_dict())
                baseline_updates += 1
            context.main_call(lambda: append_jsonl(output / "baseline_history.jsonl", {
                "schema": "drl_rollout_baseline_event_v1", "method": method,
                "optimizer_step": epoch, "probe_instances": probe_size,
                "paired_t_pvalue": pvalue, "baseline_updated": baseline_updated,
                "schedule_source": "native_adapter" if method == "DRL-TS" else "publication",
                "probe_training_stage": "soft" if soft else "hard"}))
            policy.train()
        epoch_wall = context.max_values([time.perf_counter() - epoch_started])[0]
        peak = (torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith("cuda") else 0)
        peaks = context.gather_objects(int(peak))
        row = {
            "schema": "drl_logical_epoch_history_v1", "method": method,
            "protocol_id": args.protocol_id, "logical_epoch": epoch,
            "training_stage": "soft" if soft else "hard",
            "instances_seen": effective, "customer_exposures": effective * customers,
            "physical_microbatches": len(shards) * context.world_size,
            "physical_batch_size": physical, "effective_batch_size": effective,
            "distributed_world_size": context.world_size, "physical_batch_scope": "per_rank",
            "optimizer_steps_total": epoch, "environment_transitions": transitions,
            "mean_loss": sums["loss"] / effective, "mean_training_cost": sums["cost"] / effective,
            "mean_objective_distance_km": sums["distance"] / effective,
            "objective_mode": objective.mode, "objective_unit": objective.unit,
            "mean_objective_value": sums["objective"] / effective,
            "mean_objective_cost_usd": sums["objective"] / effective if objective.is_cost else None,
            "mean_electricity_cost_usd": sums["distance"] / effective * objective.distance_unit_cost if objective.is_cost else None,
            "mean_vehicle_cost_usd": sums["vehicles"] / effective * objective.vehicle_unit_cost if objective.is_cost else None,
            "mean_vehicles_started": sums["vehicles"] / effective,
            "mean_environment_feasible_rate": sums["feasible"] / effective,
            "mean_trajectory_steps": float(steps.mean()),
            "rollout_budget_exhausted_rate": exhausted / max(len(steps), 1),
            "terminal_outcome_reason_counts": dict(sorted(global_reasons.items())),
            "baseline_kind": ("same_instance_leave_one_out" if use_leave_one_out else
                              "paper_ema" if use_ema else "greedy_rollout"), "ema_cost": ema_cost,
            "baseline_eval_due": pvalue is not None, "paired_t_pvalue": pvalue,
            "baseline_updated": baseline_updated, "epoch_wall_time_s": epoch_wall,
            "peak_gpu_allocated_bytes_per_rank": peaks,
            "global_stream_cursor": epoch * effective,
        }
        def write_training():
            _append_reinforce_diagnostics(output=output, method=method, args=args,
                objective_config=objective, session=session, data_pass=1, logical_epoch=epoch,
                optimizer_steps=epoch, soft=soft, baseline_kind=row["baseline_kind"],
                distributions=diagnostics[0], components=diagnostics[1], scales=diagnostics[2],
                pre_clip_norm=pre_clip_norm)
            append_jsonl(output / "logical_epoch_history.jsonl", row)
            print(json.dumps(row, sort_keys=True), flush=True)
        context.main_call(write_training)
        sampled = context.gather_objects(local_ids)
        ordered_ids = [view_id for start in range(0, len(sampled[0]), physical)
                       for part in sampled for view_id in part[start:start + physical]]
        context.main_call(lambda: append_jsonl(output / "sampled_view_ids.jsonl", {
            "logical_epoch": epoch, "start_cursor": (epoch - 1) * effective,
            "end_cursor": epoch * effective, "view_ids": ordered_ids}))
        if validation_tasks and epoch in scheduled:
            context.main_call(lambda: atomic_json(output / "progress.json", {
                "status": "running", "logical_epoch": epoch, "phase": "validation",
                "world_size": context.world_size, "workers": initial_identities}))
            validation = validate(validation_tasks, args.validation_seed)
            current_key = validation_key(validation)
            selected = current_key > best_key
            selected_minimum = epoch <= minimum and current_key > best_minimum
            if selected:
                best_key, checks_without_improvement = current_key, 0
            elif epoch > early_start:
                checks_without_improvement += 1
            else:
                checks_without_improvement = 0
            if selected_minimum:
                best_minimum = current_key
            validation_checks += 1
            early_stopped = bool(patience and epoch > early_start and checks_without_improvement >= patience)
            validation.update(logical_epoch=epoch, split="validation", decode_type=validation_decode,
                candidate_count=validation_candidates, checkpoint_selected=selected,
                best_overall_selected=selected, best_within_minimum_selected=selected_minimum,
                minimum_training_epochs=minimum, validation_checks_without_improvement=checks_without_improvement,
                early_stop_start_epoch=early_start, early_stop_eligible=epoch > early_start,
                early_stop_due=early_stopped)
            if selected:
                best_summary = deepcopy(validation)
            if selected_minimum:
                best_minimum_summary = deepcopy(validation)
            context.main_call(lambda: append_jsonl(output / "validation_history.jsonl", validation))
            save(epoch, epoch_artifact=True, selected=selected,
                 selected_minimum=selected_minimum, validation=validation)
            if early_stopped:
                break
    save(terminal_epoch)
    final_limit = int(getattr(args, "final_validation_limit", 0) or 0)
    if final_limit:
        with context.local_phase("final audit initialization"):
            if validation_pool is None or len(validation_pool.tasks) < final_limit:
                raise ValueError("final validation audit requires enough validation instances")
            _load_checkpoint(output / "best.ckpt", policy=policy, baseline=baseline, optimizer=optimizer,
                protocol_id=args.protocol_id, objective_config=objective,
                optimizer_name=getattr(args, "optimizer", None),
                optimizer_weight_decay=getattr(args, "weight_decay", None), reward_contract_args=args)
        audit = validate(list(validation_pool.tasks[:final_limit]), int(args.seed) + 999_000_000)
        audit.update(schema="drl_final_validation_audit_v1", split="validation", selection_changed=False,
                     selection_checkpoint=str(output / "best.ckpt"), decode_type=validation_decode,
                     candidate_count=validation_candidates)
        context.main_call(lambda: atomic_json(output / "validation_final_audit.json", audit))
    elapsed = context.max_values([time.perf_counter() - started])[0] + previous_wall
    result = {
        "schema": "drl_training_result_v1", "status": "pilot_partial" if args.pilot_mode else ("early_stopped" if early_stopped else "passed"),
        "method": method, "protocol_id": args.protocol_id,
        "objective_config": objective.to_dict(), "objective_mode": objective.mode,
        "objective_unit": objective.unit, "distributed_contract": contract,
        "requested_training_epochs": epochs, "completed_training_epochs": terminal_epoch,
        "early_stopped": early_stopped, "early_stop_epoch": terminal_epoch if early_stopped else None,
        "optimizer_steps": terminal_epoch, "instances_seen": terminal_epoch * effective,
        "customer_exposures": terminal_epoch * effective * customers,
        "environment_transitions": environment_transitions,
        "logical_environments_per_epoch": effective, "training_rollout_steps": args.training_rollout_steps,
        "physical_batch_size": physical, "effective_batch_size": effective,
        "baseline_eval_count": baseline_evals, "baseline_update_count": baseline_updates,
        "validation_checkpoints": args.validation_checkpoints, "completed_validation_checkpoints": validation_checks,
        "minimum_training_epochs": minimum, "scheduled_validation_epochs": list(scheduled),
        "early_stop_patience_validations": patience, "early_stop_start_epoch": early_start,
        "validation_decode_type": validation_decode, "validation_candidates": validation_candidates,
        "selected_checkpoint": str(output / "checkpoint_selected.pt"), "best_checkpoint": str(output / "best.ckpt"),
        "best_overall_checkpoint": str(output / "best_overall.ckpt"),
        "best_within_5000_checkpoint": str(output / "best_within_5000.ckpt"),
        "total_wall_time_s": elapsed, "total_gpu_hours": elapsed * context.world_size / 3600,
        "resolved_training_signature": getattr(args, "resolved_training_signature", None),
        "training_stream_contract_snapshot": getattr(args, "training_stream_contract_snapshot", None),
        "reward_contract_snapshot": getattr(args, "reward_contract_snapshot", None),
        "warm_start_provenance": warm_start,
    }
    if method != "AM-EVRPTW":
        result.update(reinforce_baseline=reinforce_baseline,
                      soft_stage_contract_snapshot=soft_stage_contract,
                      method_auxiliary_snapshot=getattr(args, "method_auxiliary_snapshot", None),
                      host_instance_cache_size=getattr(args, "instance_cache_size", None))
    context.main_call(lambda: atomic_json(output / "training_result.json", result))
    context.main_call(lambda: atomic_json(output / "progress.json", {
        "status": "completed", "logical_epoch": terminal_epoch,
        "early_stopped": early_stopped, "world_size": context.world_size, "workers": initial_identities}))
