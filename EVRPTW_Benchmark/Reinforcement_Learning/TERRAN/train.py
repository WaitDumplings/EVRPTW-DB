from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .trainer import load_config, train_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TERRAN on online EVRPTW Cus15 instances.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-envs-per-gpu", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--n-traj", type=int, default=None)
    parser.add_argument("--num-minibatches", type=int, default=None)
    parser.add_argument("--mother-board-pool-size", type=int, default=None)
    parser.add_argument("--mother-num-customers", type=int, default=None)
    parser.add_argument("--mother-num-charging-stations", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-path", type=str, default=None)
    parser.add_argument("--eval-n-traj", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--debug", action="store_true", help="Print train/eval diagnostics and mirror them to debug_log.txt.")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug diagnostic logging.")
    parser.add_argument("--debug-log-every", type=int, default=None)
    parser.add_argument("--profile-timing", action="store_true", help="Synchronize CUDA around timed sections for accurate profiling.")
    parser.add_argument("--no-profile-timing", action="store_true", help="Disable CUDA synchronization used only for profiling.")
    parser.add_argument("--async-instance-prefetch", action="store_true")
    parser.add_argument("--no-async-instance-prefetch", action="store_true")
    parser.add_argument("--async-instance-workers", type=int, default=None)
    parser.add_argument("--async-instance-queue-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    overrides: dict[str, Any] = {"data": {}, "training": {}, "evaluation": {}}
    if args.mother_board_pool_size is not None:
        overrides["data"]["mother_board_pool_size"] = args.mother_board_pool_size
    if args.mother_num_customers is not None:
        overrides["data"]["mother_num_customers"] = args.mother_num_customers
    if args.mother_num_charging_stations is not None:
        overrides["data"]["mother_num_charging_stations"] = args.mother_num_charging_stations
    if args.async_instance_prefetch:
        overrides["data"]["async_instance_prefetch"] = True
    if args.no_async_instance_prefetch:
        overrides["data"]["async_instance_prefetch"] = False
    if args.async_instance_workers is not None:
        overrides["data"]["async_instance_workers"] = args.async_instance_workers
    if args.async_instance_queue_batches is not None:
        overrides["data"]["async_instance_queue_batches"] = args.async_instance_queue_batches
    if args.epochs is not None:
        overrides["training"]["epochs"] = args.epochs
    if args.num_envs_per_gpu is not None:
        overrides["training"]["num_envs_per_gpu"] = args.num_envs_per_gpu
    if args.rollout_steps is not None:
        overrides["training"]["rollout_steps"] = args.rollout_steps
    if args.n_traj is not None:
        overrides["training"]["n_traj"] = args.n_traj
    if args.num_minibatches is not None:
        overrides["training"]["num_minibatches"] = args.num_minibatches
    if args.debug:
        overrides["training"]["debug"] = True
    if args.no_debug:
        overrides["training"]["debug"] = False
    if args.debug_log_every is not None:
        overrides["training"]["debug_log_every"] = args.debug_log_every
    if args.profile_timing:
        overrides["training"]["profile_timing"] = True
    if args.no_profile_timing:
        overrides["training"]["profile_timing"] = False
    if args.eval_interval is not None:
        overrides["evaluation"]["eval_interval"] = args.eval_interval
    if args.eval_path is not None:
        overrides["evaluation"]["eval_path"] = args.eval_path
    if args.eval_n_traj is not None:
        overrides["evaluation"]["eval_n_traj"] = args.eval_n_traj
    if args.eval_limit is not None:
        overrides["evaluation"]["eval_limit"] = args.eval_limit
    if not overrides["data"] and not overrides["training"] and not overrides["evaluation"]:
        overrides = {}
    else:
        overrides = {key: value for key, value in overrides.items() if value}
    ckpt = train_from_config(cfg, seed=args.seed, device=args.device, overrides=overrides)
    print(f"Saved final checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
