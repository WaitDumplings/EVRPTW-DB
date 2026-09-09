"""Compare critic losses on one checkpoint and one frozen training rollout.

This diagnostic never takes an optimizer step or saves new model weights.
The two passes share rewards, values, actions, advantages and model parameters.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..common.stage2_data import Stage2TaskPool
from ..common.training_stream import read_stream_view_ids
from .critic_stability import CriticGradientAccumulator, PPODiagnosticsAccumulator
from .env_factory import make_terran_env
from .models import Agent
from .rollout import collect_rollout, compute_returns
from .trainer import build_pbrs_config, evaluate_policy_loss, pbrs_scale_for_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rollout-epoch", type=int, default=782)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-traj", type=int, default=50)
    parser.add_argument("--step-chunk-size", type=int, default=624)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.rollout_epoch, args.batch_size, args.n_traj, args.step_chunk_size) < 1:
        parser.error("epoch, batch size, trajectory count and chunk size must be positive")
    if args.output.exists():
        parser.error("diagnostic output already exists")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(payload["config"])
    data, training, model = cfg["data"], cfg["training"], cfg["model"]
    seed = int(payload["seed"])
    rollout_seed = seed + args.rollout_epoch * 100_000
    torch.manual_seed(rollout_seed)
    pool = Stage2TaskPool(
        dataset_path=data["stage2_dataset_path"], family_root=data["stage2_family_root"],
        scale=data["stage2_scale"], split_ids="train", track_ids="train", seed=seed,
        cache_size=args.batch_size, representation=data.get("stage2_training_representation", "G"),
        euclidean_manifest=data.get("stage2_euclidean_manifest"),
    )
    stream = read_stream_view_ids(data["stage2_training_stream_path"])
    # Locate the original training epoch's instances without changing the stream.
    offset = (args.rollout_epoch - 1) * int(training["num_envs_per_gpu"])
    ids = stream[offset:offset + args.batch_size]
    if len(ids) != args.batch_size:
        raise ValueError("requested rollout epoch lies outside the source training stream")
    tasks = {task.view_id: task for task in pool.tasks}
    env_kwargs = dict(cfg["env"])
    env_kwargs.update(objective_config=cfg["objective"], info_level="light")
    horizon = int(training["rollout_steps"])
    envs = [make_terran_env(instance=pool.instance(tasks[view_id]), n_traj=args.n_traj,
                            pbrs_config=build_pbrs_config(cfg), rollout_horizon_steps=horizon,
                            **env_kwargs) for view_id in ids]
    scale = pbrs_scale_for_epoch(cfg, args.rollout_epoch, int(training["epochs"]))
    for env in envs:
        env.set_reward_scale(scale)
    agent = Agent(**{key: model[key] for key in (
        "embedding_dim", "tanh_clipping", "n_encode_layers", "use_graph_token", "use_dynamic_embedding"
    ) if key in model}, device=args.device).to(args.device)
    agent.load_state_dict(payload["model_state_dict"], strict=True)
    agent.train()
    started = time.perf_counter()
    batch = collect_rollout(agent, envs, horizon, "sample", args.device, seed=rollout_seed)
    returns = compute_returns(batch.rewards, batch.dones, float(training["gamma"]))
    advantages = returns - batch.values
    active_adv = advantages[batch.valid]
    advantages = (advantages - active_adv.mean()) / (active_adv.std(unbiased=False) + 1e-8)
    indices = np.arange(args.batch_size)
    count = int(batch.valid.sum().item())
    report = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": int(payload["epoch"]),
              "rollout_epoch": args.rollout_epoch, "training_view_ids": ids,
              "batch_size": args.batch_size, "n_traj": args.n_traj, "horizon": horizon,
              "valid_transitions": count, "rollout_seconds": time.perf_counter() - started,
              "success_count": int(sum(np.asarray(info["success"]).sum() for info in batch.final_infos)),
              "horizon_count": int(batch.rollout_budget_exhausted.sum().item()),
              "optimizer_steps": 0, "variants": {}}
    variants = {
        "previous": {"value_loss_type": "mse", "value_loss_beta": 1.0,
                     "value_residual_scale": 1.0, "vf_coef": 0.5, "critic_backbone_grad_scale": 1.0},
        "stable_v1": {"value_loss_type": "smooth_l1", "value_loss_beta": 1.0,
                      "value_residual_scale": 1.0, "vf_coef": 0.1, "critic_backbone_grad_scale": 0.1},
    }
    for name, controls in variants.items():
        variant = copy.deepcopy(cfg)
        variant["training"].update(controls)
        agent.critic.backbone_grad_scale = controls["critic_backbone_grad_scale"]
        agent.zero_grad(set_to_none=True)
        gradients, ppo = CriticGradientAccumulator(agent), PPODiagnosticsAccumulator()
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for start in range(0, len(batch.observations), args.step_chunk_size):
            end = min(start + args.step_chunk_size, len(batch.observations))
            chunk_count = int(batch.valid[start:end].sum().item())
            if not chunk_count:
                continue
            fraction = chunk_count / count
            result = evaluate_policy_loss(agent, batch, returns, advantages, variant, args.device,
                                          env_indices=indices, step_start=start, step_end=end,
                                          ppo_diagnostics=ppo, return_loss_components=True)
            components = result[4]
            gradients.accumulate(policy_objective=components["policy_loss"] * fraction,
                                 weighted_value_objective=components["weighted_value_loss"] * fraction,
                                 transition_count=chunk_count)
            (result[0] * fraction).backward()
            del result, components
        norm = torch.sqrt(sum(parameter.grad.detach().double().square().sum()
                              for parameter in agent.parameters() if parameter.grad is not None))
        report["variants"][name] = {"controls": controls, "gradients": gradients.summary(),
                                    "ppo": ppo.summary(), "preclip_global_norm": float(norm),
                                    "seconds": time.perf_counter() - started,
                                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
                                    if args.device.startswith("cuda") else None}
        print(json.dumps({"variant": name, **report["variants"][name]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
