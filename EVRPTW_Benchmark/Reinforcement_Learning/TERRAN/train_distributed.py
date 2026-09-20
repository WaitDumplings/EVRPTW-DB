"""Synchronous legacy-PPO TERRAN; launch with torchrun, one process per GPU.

This entry deliberately keeps the legacy actor/critic and PPO/PBRS loss. It
shares the single-GPU command line and supports reset warm starts for fresh
fixed-epoch Stage-2 runs.
No collective occurs inside a variable-length rollout or backward time chunk.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import fcntl
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from ..common.data_pass import DataPassState
from ..common.distributed import DistributedContext, close_distributed, initialize_distributed
from ..common.distributed_entrypoints import parse_distributed_args
from ..common.training_protocol import append_jsonl, atomic_json, build_adamw_optimizer, validation_key
from . import trainer as legacy
from .critic_stability import PPODiagnosticsAccumulator, resolve_critic_stability_config
from .models import Agent
from .protocol import finalize_protocol
from .rollout import collect_rollout, compute_returns, summarize_rollout_outcomes
from .train import parse_args, prepare_training


def configure_topology(cfg, context):
    """Freeze physical/global batching without changing the legacy architecture."""
    training, protocol = cfg["training"], cfg.get("protocol", {})
    if training.get("algorithm") == "stable_cost_v1":
        raise ValueError("this entry is legacy PPO, not stable_cost_v1")
    if protocol.get("resume_checkpoint"):
        raise ValueError("distributed legacy TERRAN resume is unsupported")
    if protocol.get("warm_start_checkpoint") and legacy.warm_start_epoch_mode(cfg) != "reset":
        raise ValueError("distributed legacy TERRAN warm start requires reset epoch mode")
    if not cfg["data"].get("stage2_dataset_path"):
        raise ValueError("distributed TERRAN requires a frozen Stage-2 pool")
    physical = int(training["num_envs_per_gpu"])
    effective = int(protocol.get("effective_batch_size") or physical * context.world_size)
    if physical < 1 or effective % (physical * context.world_size):
        raise ValueError("effective batch must be divisible by per-GPU physical batch times world size")
    if int(training.get("early_stop_patience_validations", 0)):
        raise ValueError("distributed legacy TERRAN requires fixed epochs with early stopping disabled")
    training["distributed_world_size"] = context.world_size
    training["logical_microbatches_per_epoch"] = effective // (physical * context.world_size)
    contract = {
        "schema": "terran_synchronous_legacy_ppo_v1",
        "world_size": context.world_size,
        "physical_batch_size_per_rank": physical,
        "effective_batch_size_global": effective,
        "microbatches_per_rank": training["logical_microbatches_per_epoch"],
        "gradient_reduction": "sum_with_global_valid_transition_denominator",
        "advantage_normalization": "global_valid_transition_population_mean_and_std",
        "batch_norm": "local_forward_rank_zero_buffers_after_each_update",
        "validation": "rank_zero_full_frozen_cohort_broadcast_result",
        "sampler": "contiguous_physical_shards_of_shared_seeded_global_stream",
        "resume": "unsupported_fail_closed",
        "warm_start": deepcopy(protocol.get("warm_start")),
    }
    cfg["distributed_contract"] = contract
    return contract


def normalize_global_advantages(records, context):
    """Population moments weight transitions, including unequal decode lengths."""
    values = torch.cat([advantage[batch.valid].double() for batch, _, advantage in records])
    count, total = context.sum_values([values.numel(), float(values.sum())])
    if not count:
        raise RuntimeError("PPO global batch contains no valid transitions")
    mean = total / count
    variance_sum = context.sum_values([float((values - mean).square().sum())])[0]
    std = math.sqrt(max(variance_sum / count, 0.0))
    if count > 1:
        records = [(batch, returns, (advantage - mean) / (std + 1e-8))
                   for batch, returns, advantage in records]
    return records, {"count": int(count), "mean": mean, "std": std}


def synchronous_ppo_update(agent, optimizer, records, cfg, device, context, *, epoch_seed):
    """Use the same minibatch/accumulation semantics as the legacy trainer.

    With one physical rollout, matching local minibatch groups form a global
    optimizer group. With logical rollout accumulation, all physical buffers
    form one update per PPO pass, exactly as in the single-device trainer.
    """
    training = cfg["training"]
    rng = np.random.default_rng(epoch_seed + 17 + context.rank * 1_000_003)
    diagnostics = PPODiagnosticsAccumulator()
    losses, norms = [], []
    for _ in range(int(training.get("ppo_update_epochs", 4))):
        groups = []
        for batch, returns, advantages in records:
            env_order = np.arange(int(batch.actions.size(1)), dtype=np.int64)
            rng.shuffle(env_order)
            splits = np.array_split(env_order, min(int(training.get("num_minibatches", 1)), len(env_order)))
            chunks = [(batch, returns, advantages, indices) for indices in splits if indices.size]
            if len(records) > 1:
                if not groups:
                    groups.append([])
                groups[0].extend(chunks)
            else:
                accumulation = max(1, int(training.get("gradient_accumulation_steps", 1)))
                groups.extend(chunks[i:i + accumulation] for i in range(0, len(chunks), accumulation))
        for group in groups:
            local_count = sum(legacy._valid_transition_count(batch, indices)
                              for batch, _, _, indices in group)
            denominator = context.sum_values([local_count])[0]
            if denominator <= 0:
                raise RuntimeError("PPO global optimizer group has no valid transitions")
            optimizer.zero_grad(set_to_none=True)
            local_losses = np.zeros(3, dtype=np.float64)
            with context.local_phase("legacy PPO backward"):
                for batch, returns, advantages, indices in group:
                    steps = int(batch.actions.size(0))
                    chunk_size = int(training.get("ppo_step_chunk_size", 0)) or steps
                    for start in range(0, steps, chunk_size):
                        end = min(start + chunk_size, steps)
                        count = legacy._valid_transition_count(batch, indices, start, end)
                        if not count:
                            continue
                        loss, policy, value, entropy = legacy.evaluate_policy_loss(
                            agent, batch, returns, advantages.detach(), cfg, device,
                            env_indices=indices, step_start=start, step_end=end,
                            ppo_diagnostics=diagnostics)
                        weight = count / denominator
                        (loss * weight).backward()
                        local_losses += np.asarray([policy.item(), value.item(), entropy.item()]) * weight
            context.sum_gradients(agent.parameters())
            norm = legacy._clip_grad_norm_finite(agent.parameters(), float(training.get("max_grad_norm", 1.0)))
            optimizer.step()
            context.broadcast_buffers(agent)
            losses.append(context.sum_values(local_losses))
            norms.append(float(norm))
    return {"losses": losses, "gradient_norms": norms, "optimizer_steps": len(losses),
            "rank_ppo_diagnostics": context.gather_objects(diagnostics.summary())}


def _checkpoint(path, agent, optimizer, cfg, epoch, seed):
    temporary = path.with_name(path.name + ".tmp")
    legacy.save_checkpoint(temporary, agent, optimizer, cfg, epoch, seed)
    temporary.replace(path)


def initialize_warm_start(agent, optimizer, cfg, *, seed):
    """Validate and load each rank before the initial model broadcast."""
    protocol = cfg.get("protocol", {})
    checkpoint = protocol.get("warm_start_checkpoint")
    if checkpoint is None:
        return 1
    if legacy.warm_start_epoch_mode(cfg) != "reset":
        raise ValueError("distributed legacy TERRAN warm start requires reset epoch mode")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    legacy.validate_warm_start_contract(cfg, payload, current_seed=seed, checkpoint_path=checkpoint)
    provenance = protocol.get("warm_start") or {}
    return legacy.apply_training_initialization(
        agent, optimizer, warm_start_payload=payload, warm_start_mode="reset",
        warm_start_objective_transition=bool(provenance.get("objective_transition", False)),
        warm_start_scale_transition=bool(provenance.get("scale_transition", False)),
    )


def train_distributed(cfg, *, seed, device, context):
    cfg = deepcopy(cfg)
    contract = configure_topology(cfg, context)
    training, protocol = cfg["training"], cfg.get("protocol", {})
    if str(training.get("optimizer", "adamw")).lower() != "adamw":
        raise ValueError("TERRAN optimizer must be AdamW")
    objective = legacy.resolve_objective(cfg.get("objective"))
    cfg["objective"] = objective.to_dict()
    legacy._configure_reward_contract(cfg)
    legacy._freeze_pbrs_reward_semantics(cfg)
    gamma = legacy.training_gamma(cfg)
    training["gamma"] = gamma
    if objective.is_cost and gamma != 1:
        raise ValueError("cost PPO requires undiscounted gamma=1")
    if objective.is_cost and protocol.get("protocol_id") and cfg.get("reward_contract") is None:
        raise ValueError("formal cost PPO requires a frozen reward contract")
    critic_config = resolve_critic_stability_config(training)
    training.update(critic_config)
    legacy._freeze_resolved_terran_training_signature(cfg, seed=seed)
    context.main_call(lambda: legacy.validate_fresh_training_output(cfg))
    legacy.set_seed(seed)
    with context.local_phase("model and pool initialization"):
        model_cfg = cfg.get("model", {})
        agent = Agent(
            embedding_dim=int(model_cfg.get("embedding_dim", 256)),
            tanh_clipping=float(model_cfg.get("tanh_clipping", 15)),
            n_encode_layers=int(model_cfg.get("n_encode_layers", 3)), device=device,
            use_graph_token=bool(model_cfg.get("use_graph_token", False)),
            use_dynamic_embedding=bool(model_cfg.get("use_dynamic_embedding", False))).to(device)
        agent.critic.backbone_grad_scale = critic_config["critic_backbone_grad_scale"]
        optimizer = build_adamw_optimizer(agent.parameters(), learning_rate=float(training.get("learning_rate", 1e-4)),
                                          eps=1e-5, weight_decay=float(training.get("weight_decay", .01)))
        if initialize_warm_start(agent, optimizer, cfg, seed=seed) != 1:
            raise RuntimeError("distributed TERRAN warm start did not reset to epoch one")
        local_cfg = deepcopy(cfg)
        local_cfg["data"]["distributed_rank"] = context.rank
        envs, pool = legacy.make_envs(local_cfg, seed)
    context.broadcast_model(agent)
    output = Path(cfg["output_dir"])
    def init_output():
        (output / "logs").mkdir(parents=True, exist_ok=True)
        (output / "checkpoints").mkdir(exist_ok=True)
        atomic_json(output / "distributed_contract.json", contract)
        atomic_json(output / "resolved_config.json", cfg)
    context.main_call(init_output)
    epochs = int(training["epochs"])
    effective = contract["effective_batch_size_global"]
    microbatches = contract["microbatches_per_rank"]
    minimum = int(training.get("minimum_training_epochs") or epochs)
    scheduled = set(training.get("validation_epochs", []))
    interval = int(cfg.get("evaluation", {}).get("eval_interval", 0))
    transitions_total = optimizer_steps = validation_checks = 0
    best_key = within_key = (-math.inf, -math.inf)
    csv_fields = ("epoch", "samples_seen", "environment_transitions_total", "optimizer_steps_total", "epoch_wall_time_s",
                  "train_feasible_rate", "train_avg_best_objective", "policy_loss", "value_loss", "entropy", "world_size")
    try:
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            epoch_seed = seed + epoch * 100_000
            legacy.set_seed(epoch_seed + context.rank * 1_000_003)
            scale = legacy.pbrs_scale_for_epoch(cfg, epoch, epochs)
            legacy.set_pbrs_reward_scale(envs, scale)
            agent.train()
            records, infos, horizons = [], [], []
            with context.local_phase("legacy PPO rollout"):
                for microbatch in range(microbatches):
                    batch = collect_rollout(agent, envs, rollout_steps=int(training["rollout_steps"]),
                                            decode_mode="sample", device=device,
                                            seed=epoch_seed + context.rank * 1_000_003 + microbatch,
                                            cache_static_embeddings=bool(training.get("cache_rollout_encoder", True)),
                                            reward_discount_factor=gamma)
                    returns = compute_returns(batch.rewards, batch.dones, gamma=gamma)
                    records.append((batch, returns, returns - batch.values))
                    infos.extend(batch.final_infos)
                    horizons.append(batch.rollout_budget_exhausted.detach().cpu().numpy())
            records, moments = normalize_global_advantages(records, context)
            transitions_total += moments["count"]
            update = synchronous_ppo_update(agent, optimizer, records, cfg, device, context, epoch_seed=epoch_seed)
            optimizer_steps += update["optimizer_steps"]
            parts = context.gather_objects({"infos": infos, "horizons": np.concatenate(horizons),
                                            "sampled_view_ids": pool.drain_sampled_view_ids()})
            samples = int(context.sum_values([pool.sample_count])[0])
            if samples != epoch * effective:
                raise RuntimeError(f"TERRAN global exposure mismatch: {samples} != {epoch * effective}")
            should_eval = epoch in scheduled if scheduled else bool(interval and (epoch % interval == 0 or epoch == epochs))
            def report():
                nonlocal best_key, within_key, validation_checks
                all_infos = [info for part in parts for info in part["infos"]]
                summary = legacy.summarize_train_infos(all_infos)
                outcomes = summarize_rollout_outcomes(all_infos, np.concatenate([part["horizons"] for part in parts]))
                loss = np.asarray(update["losses"]).mean(axis=0)
                validation = None
                if should_eval:
                    validation_checks += 1
                    eval_start = time.perf_counter()
                    row = legacy.evaluate_fixed_dataset(agent, cfg, seed=seed, epoch=epoch, device=device)
                    validation = legacy.validation_summary_from_eval_row(
                        row, objective_config=objective, logical_epoch=epoch,
                        validation_seed=int(cfg["evaluation"].get("eval_seed", seed + 910_000_000)),
                        validation_wall_time_s=time.perf_counter() - eval_start)
                    if validation is None:
                        raise RuntimeError("distributed TERRAN required validation did not complete")
                    key = validation_key(validation)
                    best, within = key > best_key, epoch <= minimum and key > within_key
                    validation.update(checkpoint_selected=best, best_overall_selected=best,
                                      best_within_minimum_selected=within, minimum_training_epochs=minimum,
                                      distributed_world_size=context.world_size)
                    append_jsonl(output / "validation_history.jsonl", validation)
                    if best:
                        best_key = key
                        for name in ("best_overall.ckpt", "best.ckpt", "checkpoint_selected.pt"):
                            _checkpoint(output / name, agent, optimizer, cfg, epoch, seed)
                        for name in ("validation_summary.json", "validation_summary_overall.json"):
                            atomic_json(output / name, validation)
                    if within:
                        within_key = key
                        _checkpoint(output / "best_within_5000.ckpt", agent, optimizer, cfg, epoch, seed)
                        atomic_json(output / "validation_summary_within_5000.json", validation)
                    print(f"[Eval] epoch={epoch}/{epochs} cost={validation.get('mean_verified_cost_usd')} "
                          f"feasible={validation['complete_and_feasible']}/{validation['instances']}", flush=True)
                wall = time.perf_counter() - started
                record = {"epoch": epoch, "samples_seen": samples, "environment_transitions_total": transitions_total,
                          "optimizer_steps_total": optimizer_steps, "epoch_wall_time_s": wall,
                          "train_feasible_rate": summary["train_feasible_rate"],
                          "train_avg_best_objective": summary["train_avg_best_objective"],
                          "policy_loss": float(loss[0]), "value_loss": float(loss[1]), "entropy": float(loss[2]),
                          "world_size": context.world_size}
                log = output / "logs" / "train_log.csv"
                with log.open("a", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=csv_fields)
                    if epoch == 1:
                        writer.writeheader()
                    writer.writerow(record)
                append_jsonl(output / "logical_epoch_history.jsonl", legacy._json_safe_diagnostics({
                    **record, "logical_epoch": epoch, "mean_environment_feasible_rate": summary["train_feasible_rate"],
                    "trajectory_success_rate": outcomes["success_rate"], "outcomes": outcomes,
                    "global_advantage_moments": moments, "gradient_norms": update["gradient_norms"],
                    "rank_ppo_diagnostics": update["rank_ppo_diagnostics"], "pbrs_scale": scale}))
                if any(part["sampled_view_ids"] for part in parts):
                    append_jsonl(output / "sampled_view_ids.jsonl", {"logical_epoch": epoch,
                                 "rank_shards": [part["sampled_view_ids"] for part in parts]})
                if should_eval or epoch == epochs or epoch % int(training.get("checkpoint_interval", 50)) == 0:
                    _checkpoint(output / "checkpoints" / f"checkpoint_epoch_{epoch:04d}.pt", agent, optimizer, cfg, epoch, seed)
                    _checkpoint(output / "checkpoint_latest.pt", agent, optimizer, cfg, epoch, seed)
                    DataPassState(protocol_id=str(protocol.get("protocol_id", "")),
                                  completed_data_passes=int(epoch == epochs), instances_seen=samples,
                                  customer_exposures=samples * int(cfg["data"]["num_customers"]),
                                  optimizer_steps=optimizer_steps, environment_transitions=transitions_total,
                                  last_checkpoint=str(output / "checkpoint_latest.pt")).atomic_write(output / "data_pass_state.json")
                print(f"[Train] epoch={epoch}/{epochs} samples={samples} cost={summary['train_avg_best_objective']:.4f} "
                      f"feasibility={summary['train_feasible_rate']:.4f} wall={wall:.2f}s ranks={context.world_size}", flush=True)
            context.main_call(report)
        final = output / "checkpoints" / "checkpoint_final.pt"
        def finish():
            _checkpoint(final, agent, optimizer, cfg, epochs, seed)
            atomic_json(output / "early_stop_state.json", {"requested_training_epochs": epochs,
                        "completed_training_epochs": epochs, "completed_validation_checkpoints": validation_checks,
                        "early_stopped": False, "early_stop_epoch": None})
        context.main_call(finish)
        return final
    finally:
        pool.close(terminate=True)


def main():
    args = parse_distributed_args(parse_args)
    if args.resume:
        raise ValueError("distributed legacy TERRAN resume is unsupported")
    if args.warm_start_checkpoint is not None and args.warm_start_epoch_mode != "reset":
        raise ValueError("distributed legacy TERRAN warm start requires reset epoch mode")
    if not args.training_epochs or args.data_passes is not None:
        raise ValueError("distributed TERRAN requires --training-epochs")
    if args.output_dir is None:
        raise ValueError("--output-dir is required")
    context, args.device = initialize_distributed(device=args.device or "cuda", backend=args.distributed_backend,
        timeout_seconds=args.distributed_timeout_seconds, expected_world_size=args.expected_world_size)
    lock = None
    try:
        def claim():
            nonlocal lock
            args.output_dir.mkdir(parents=True, exist_ok=True)
            lock = (args.output_dir / ".distributed.lock").open("a+")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        context.main_call(claim)
        cfg, overrides, meta = context.main_call(lambda: prepare_training(args))
        cfg = legacy.deep_update(cfg, overrides)
        cfg["data"]["stage2_cache_size"] = args.instance_cache_size
        checkpoint = train_distributed(cfg, seed=args.seed, device=args.device, context=context)
        memory_by_rank = context.gather_objects(
            int(torch.cuda.max_memory_allocated(args.device))
            if str(args.device).startswith("cuda") else 0)
        def finalize():
            finalize_protocol(args, checkpoint, meta)
            result_path = args.output_dir / "training_result.json"
            result = json.loads(result_path.read_text())
            result.update(distributed_contract=json.loads(
                (args.output_dir / "distributed_contract.json").read_text()),
                peak_gpu_memory_bytes_by_rank=memory_by_rank,
                peak_gpu_memory_bytes=max(memory_by_rank),
                allocated_gpu_hours=(result["wall_time_s"] * context.world_size / 3600
                                     if str(args.device).startswith("cuda") else 0))
            atomic_json(result_path, result)
        context.main_call(finalize)
        if context.is_main:
            print(f"Saved synchronized final checkpoint: {checkpoint}", flush=True)
    finally:
        if lock is not None:
            lock.close()
        close_distributed()


if __name__ == "__main__":
    main()
