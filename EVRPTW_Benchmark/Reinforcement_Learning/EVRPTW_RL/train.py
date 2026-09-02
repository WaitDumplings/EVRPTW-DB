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

from ..common import Stage2TaskPool, make_envs
from ..common.protocol_entrypoints import run_evrptw_rl
from ..common.training_protocol import add_data_pass_arguments
from .model import EVRPTWRLPolicy
from .rollout import rollout


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the EVRPTW-RL baseline.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--family-root", type=Path)
    parser.add_argument("--scale", default="Cus100")
    parser.add_argument("--split-ids", default="train")
    parser.add_argument("--track-ids", default="train")
    parser.add_argument("--city-slugs")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--samples-per-instance", type=int, default=1)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--structure2vec-rounds", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--ema-warmup-steps", type=int, default=1_000)
    parser.add_argument("--ema-decay", type=float, default=0.9)
    parser.add_argument("--baseline-eval-interval", type=int, default=100)
    parser.add_argument("--baseline-eval-size", type=int, default=64)
    parser.add_argument("--baseline-alpha", type=float, default=0.05)
    parser.add_argument("--station-visit-penalty", type=float, default=0.3)
    parser.add_argument("--incomplete-penalty", type=float, default=100.0)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    add_data_pass_arguments(parser)
    return parser.parse_args()


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
            seed=args.seed + 20_000,
            station_visit_penalty=args.station_visit_penalty,
            incomplete_penalty=args.incomplete_penalty,
        )
    return result.training_cost[:, 0].detach().cpu().numpy()


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
    policy = EVRPTWRLPolicy(
        embedding_dim=args.embedding_dim,
        structure2vec_rounds=args.structure2vec_rounds,
    ).to(args.device)
    baseline = deepcopy(policy).eval()
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    if args.data_passes is not None:
        run_evrptw_rl(args, pool, policy, optimizer)
        return
    ema_cost: float | None = None
    baseline_instances = pool.sample(args.baseline_eval_size)
    history_path = args.output_dir / "train_history.jsonl"

    for iteration in range(1, args.iterations + 1):
        started = time.perf_counter()
        instances = pool.sample(args.batch_size)
        actor_envs = make_envs(
            instances,
            n_traj=args.samples_per_instance,
            info_level="light",
        )
        policy.train()
        actor = rollout(
            policy,
            actor_envs,
            decode_type="sampling",
            max_steps=_max_steps(actor_envs),
            seed=args.seed + iteration,
            station_visit_penalty=args.station_visit_penalty,
            incomplete_penalty=args.incomplete_penalty,
        )

        if iteration <= args.ema_warmup_steps:
            observed = float(actor.training_cost.mean().detach().cpu())
            ema_cost = (
                observed
                if ema_cost is None
                else args.ema_decay * ema_cost + (1.0 - args.ema_decay) * observed
            )
            baseline_cost = torch.full_like(actor.training_cost, float(ema_cost))
        else:
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
                    seed=args.seed + iteration,
                    station_visit_penalty=args.station_visit_penalty,
                    incomplete_penalty=args.incomplete_penalty,
                )
            baseline_cost = baseline_result.training_cost

        advantage = (actor.training_cost - baseline_cost).detach()
        loss = (advantage * actor.log_likelihood).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        optimizer.step()

        baseline_updated = False
        paired_t_pvalue: float | None = None
        if iteration == args.ema_warmup_steps:
            baseline.load_state_dict(policy.state_dict())
        elif (
            iteration > args.ema_warmup_steps
            and iteration % args.baseline_eval_interval == 0
        ):
            policy.eval()
            actor_costs = _greedy_costs(policy, baseline_instances, args)
            baseline_costs = _greedy_costs(baseline, baseline_instances, args)
            test = ttest_rel(actor_costs, baseline_costs, alternative="less")
            paired_t_pvalue = float(test.pvalue)
            baseline_updated = bool(
                np.mean(actor_costs) < np.mean(baseline_costs)
                and np.isfinite(test.pvalue)
                and paired_t_pvalue < args.baseline_alpha
            )
            if baseline_updated:
                baseline.load_state_dict(policy.state_dict())
                baseline_instances = pool.sample(args.baseline_eval_size)

        row = {
            "iteration": iteration,
            "loss": float(loss.detach().cpu()),
            "training_cost": float(actor.training_cost.mean().detach().cpu()),
            "objective_distance_km": float(
                actor.objective_distance_km.mean().detach().cpu()
            ),
            "feasible_rate": float(actor.feasible.float().mean().detach().cpu()),
            "mean_station_visits": float(actor.station_visits.float().mean().cpu()),
            "baseline_kind": (
                "ema" if iteration <= args.ema_warmup_steps else "greedy_rollout"
            ),
            "paired_t_pvalue": paired_t_pvalue,
            "baseline_updated": baseline_updated,
            "runtime_s": float(time.perf_counter() - started),
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if iteration % args.checkpoint_interval == 0 or iteration == args.iterations:
            torch.save(
                {
                    "method": "EVRPTW-RL",
                    "iteration": iteration,
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
