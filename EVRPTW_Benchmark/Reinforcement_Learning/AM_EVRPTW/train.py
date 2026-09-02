from __future__ import annotations

import argparse
import json
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_rel

from .data import Stage2TaskPool, make_envs
from .model import AMEVRPTWPolicy
from .rollout import rollout


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _max_steps(envs) -> int:
    return max(env.unwrapped.max_steps for env in envs)


def _greedy_costs(policy, instances, args) -> np.ndarray:
    envs = make_envs(instances, n_traj=1, info_level="light")
    with torch.no_grad():
        result = rollout(
            policy,
            envs,
            decode_type="greedy",
            max_steps=_max_steps(envs),
            seed=args.seed + 10_000,
            incomplete_penalty_km=args.incomplete_penalty_km,
        )
    return result.cost_km[:, 0].detach().cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AM-EVRPTW baseline.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--family-root", type=Path)
    parser.add_argument("--scale", default="Cus100")
    parser.add_argument("--split-ids", default="train")
    parser.add_argument("--track-ids")
    parser.add_argument("--city-slugs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples-per-instance", type=int, default=1)
    parser.add_argument("--baseline-eval-size", type=int, default=64)
    parser.add_argument("--baseline-alpha", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--n-encode-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--tanh-clipping", type=float, default=10.0)
    parser.add_argument("--incomplete-penalty-km", type=float, default=10000.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool = Stage2TaskPool(
        dataset_path=args.dataset_path,
        family_root=args.family_root,
        scale=args.scale,
        split_ids=args.split_ids,
        track_ids=args.track_ids,
        city_slugs=args.city_slugs,
        seed=args.seed,
    )
    baseline_instances = list(pool.first(limit=args.baseline_eval_size))
    policy = AMEVRPTWPolicy(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.embedding_dim,
        n_encode_layers=args.n_encode_layers,
        n_heads=args.n_heads,
        tanh_clipping=args.tanh_clipping,
    ).to(args.device)
    baseline = deepcopy(policy).eval()
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    history_path = args.output_dir / "train_history.jsonl"

    for epoch in range(args.epochs):
        policy.train()
        epoch_costs: list[float] = []
        epoch_feasible: list[float] = []
        start = time.perf_counter()
        for step in range(args.steps_per_epoch):
            instances = pool.sample(args.batch_size)
            actor_envs = make_envs(
                instances,
                n_traj=args.samples_per_instance,
                info_level="light",
            )
            actor = rollout(
                policy,
                actor_envs,
                decode_type="sampling",
                max_steps=_max_steps(actor_envs),
                seed=args.seed + epoch * args.steps_per_epoch + step,
                incomplete_penalty_km=args.incomplete_penalty_km,
            )
            baseline_envs = make_envs(
                instances,
                n_traj=args.samples_per_instance,
                info_level="light",
            )
            with torch.no_grad():
                baseline_result = rollout(
                    baseline,
                    baseline_envs,
                    decode_type="greedy",
                    max_steps=_max_steps(baseline_envs),
                    seed=args.seed + epoch * args.steps_per_epoch + step,
                    incomplete_penalty_km=args.incomplete_penalty_km,
                )
            advantage = (actor.cost_km - baseline_result.cost_km).detach()
            loss = (advantage * actor.log_likelihood).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            epoch_costs.append(float(actor.cost_km.mean().detach().cpu()))
            epoch_feasible.append(float(actor.feasible.float().mean().detach().cpu()))

        policy.eval()
        actor_costs = _greedy_costs(policy, baseline_instances, args)
        baseline_costs = _greedy_costs(baseline, baseline_instances, args)
        test = ttest_rel(actor_costs, baseline_costs, alternative="less")
        improved = bool(
            np.mean(actor_costs) < np.mean(baseline_costs)
            and np.isfinite(test.pvalue)
            and float(test.pvalue) < args.baseline_alpha
        )
        if improved:
            baseline.load_state_dict(policy.state_dict())
        row = {
            "epoch": epoch,
            "train_cost_km": float(np.mean(epoch_costs)),
            "train_feasible_rate": float(np.mean(epoch_feasible)),
            "greedy_actor_cost_km": float(np.mean(actor_costs)),
            "greedy_baseline_cost_km": float(np.mean(baseline_costs)),
            "paired_t_pvalue": float(test.pvalue),
            "baseline_updated": improved,
            "epoch_runtime_s": float(time.perf_counter() - start),
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        torch.save(
            {
                "method": "AM-EVRPTW",
                "epoch": epoch,
                "model": policy.state_dict(),
                "baseline": baseline.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            },
            args.output_dir / "checkpoint_latest.pt",
        )
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
