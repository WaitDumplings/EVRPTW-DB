from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..common.training_protocol import add_data_pass_arguments
from .protocol import configure_protocol, finalize_protocol

from .trainer import load_config, train_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TERRAN on canonical Stage-2 or explicit legacy data."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-envs-per-gpu", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--ppo-step-chunk-size", type=int, default=None)
    parser.add_argument("--n-traj", type=int, default=None)
    parser.add_argument("--num-minibatches", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--service-territory-pool-size", "--mother-board-pool-size", dest="mother_board_pool_size", type=int, default=None)
    parser.add_argument("--train-dataset-path", type=str, default=None, help="Fixed train split bundle or directory, e.g. EVRPTW_Dataset/dataset_v1/dataset/train/Cus15")
    parser.add_argument("--stage2-dataset-path", type=str, default=None)
    parser.add_argument("--stage2-family-root", type=str, default=None)
    parser.add_argument("--stage2-scale", type=str, default=None)
    parser.add_argument("--stage2-split-ids", type=str, default=None)
    parser.add_argument("--stage2-track-ids", type=str, default=None)
    parser.add_argument("--stage2-city-slugs", type=str, default=None)
    parser.add_argument("--num-customers", type=int, default=None)
    parser.add_argument("--num-charging-stations", type=int, default=None)
    parser.add_argument("--train-sample-mode", type=str, default=None, choices=["shuffle_cycle", "cycle", "random"])
    parser.add_argument("--territory-pool-path", "--region-pool-path", dest="territory_pool_path", type=str, default=None)
    parser.add_argument("--territory-pool-shuffle", "--region-pool-shuffle", dest="territory_pool_shuffle", action="store_true")
    parser.add_argument("--no-territory-pool-shuffle", "--no-region-pool-shuffle", dest="no_territory_pool_shuffle", action="store_true")
    parser.add_argument("--territory-pool-replacement-policy", "--region-pool-replacement-policy", dest="territory_pool_replacement_policy", type=str, default=None, choices=["cycle", "generate"])
    parser.add_argument("--mother-num-customers", type=int, default=None)
    parser.add_argument("--mother-num-charging-stations", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-path", type=str, default=None)
    parser.add_argument("--eval-n-traj", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-num-batches", type=int, default=None)
    parser.add_argument("--eval-info-level", type=str, choices=["light", "full"], default=None)
    parser.add_argument("--eval-save-routes", action="store_true")
    parser.add_argument("--no-eval-save-routes", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Print train/eval diagnostics and mirror them to debug_log.txt.")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug diagnostic logging.")
    parser.add_argument("--debug-log-every", type=int, default=None)
    parser.add_argument("--profile-timing", action="store_true", help="Synchronize CUDA around timed sections for accurate profiling.")
    parser.add_argument("--no-profile-timing", action="store_true", help="Disable CUDA synchronization used only for profiling.")
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--cache-rollout-encoder",
        dest="cache_rollout_encoder",
        action="store_true",
        help="Encode immutable instance features once per training rollout.",
    )
    cache_group.add_argument(
        "--no-cache-rollout-encoder",
        dest="cache_rollout_encoder",
        action="store_false",
        help="Recompute the encoder at every rollout step for A/B diagnostics.",
    )
    parser.set_defaults(cache_rollout_encoder=None)
    parser.add_argument("--async-instance-prefetch", action="store_true")
    parser.add_argument("--no-async-instance-prefetch", action="store_true")
    parser.add_argument("--async-instance-workers", type=int, default=None)
    parser.add_argument("--async-instance-queue-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=Path)
    add_data_pass_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    overrides: dict[str, Any] = {"data": {}, "training": {}, "evaluation": {}}
    if args.mother_board_pool_size is not None:
        overrides["data"]["mother_board_pool_size"] = args.mother_board_pool_size
    if args.train_dataset_path is not None:
        overrides["data"]["train_dataset_path"] = args.train_dataset_path
    if args.stage2_dataset_path is not None:
        overrides["data"]["stage2_dataset_path"] = args.stage2_dataset_path
    if args.stage2_family_root is not None:
        overrides["data"]["stage2_family_root"] = args.stage2_family_root
    if args.stage2_scale is not None:
        overrides["data"]["stage2_scale"] = args.stage2_scale
    if args.stage2_split_ids is not None:
        overrides["data"]["stage2_split_ids"] = args.stage2_split_ids
    if args.stage2_track_ids is not None:
        overrides["data"]["stage2_track_ids"] = args.stage2_track_ids
    if args.stage2_city_slugs is not None:
        overrides["data"]["stage2_city_slugs"] = args.stage2_city_slugs
    if args.training_stream_path is not None:
        overrides["data"]["stage2_training_stream_path"] = str(
            args.training_stream_path
        )
    overrides["data"]["stage2_training_representation"] = args.training_representation
    if args.euclidean_manifest is not None:
        overrides["data"]["stage2_euclidean_manifest"] = str(
            args.euclidean_manifest
        )
    if args.num_customers is not None:
        overrides["data"]["num_customers"] = args.num_customers
    if args.num_charging_stations is not None:
        overrides["data"]["num_charging_stations"] = args.num_charging_stations
    if args.train_sample_mode is not None:
        overrides["data"]["train_sample_mode"] = args.train_sample_mode
    if args.territory_pool_path is not None:
        overrides["data"]["territory_pool_path"] = args.territory_pool_path
    if args.territory_pool_shuffle:
        overrides["data"]["territory_pool_shuffle"] = True
    if args.no_territory_pool_shuffle:
        overrides["data"]["territory_pool_shuffle"] = False
    if args.territory_pool_replacement_policy is not None:
        overrides["data"]["region_pool_replacement_policy"] = args.territory_pool_replacement_policy
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
    if args.ppo_step_chunk_size is not None:
        overrides["training"]["ppo_step_chunk_size"] = args.ppo_step_chunk_size
    if args.n_traj is not None:
        overrides["training"]["n_traj"] = args.n_traj
    if args.num_minibatches is not None:
        overrides["training"]["num_minibatches"] = args.num_minibatches
    if args.gradient_accumulation_steps is not None:
        overrides["training"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
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
    if args.cache_rollout_encoder is not None:
        overrides["training"]["cache_rollout_encoder"] = args.cache_rollout_encoder
    if args.eval_interval is not None:
        overrides["evaluation"]["eval_interval"] = args.eval_interval
    if args.eval_path is not None:
        overrides["evaluation"]["eval_path"] = args.eval_path
    if args.eval_n_traj is not None:
        overrides["evaluation"]["eval_n_traj"] = args.eval_n_traj
    if args.eval_limit is not None:
        overrides["evaluation"]["eval_limit"] = args.eval_limit
    if args.eval_batch_size is not None:
        overrides["evaluation"]["eval_batch_size"] = args.eval_batch_size
    if args.eval_num_batches is not None:
        overrides["evaluation"]["eval_num_batches"] = args.eval_num_batches
    if args.eval_info_level is not None:
        overrides["evaluation"]["eval_info_level"] = args.eval_info_level
    if args.eval_save_routes:
        overrides["evaluation"]["eval_save_routes"] = True
    if args.no_eval_save_routes:
        overrides["evaluation"]["eval_save_routes"] = False
    if args.output_dir is not None:
        overrides["output_dir"] = str(args.output_dir.resolve())
    overrides = {key: value for key, value in overrides.items() if value}
    overrides, protocol_meta = configure_protocol(args, overrides)
    ckpt = train_from_config(cfg, seed=args.seed, device=args.device, overrides=overrides)
    finalize_protocol(args, ckpt, protocol_meta)
    print(f"Saved final checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
