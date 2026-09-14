"""Synchronous EVRPTW-RL with its native EMA-to-greedy baseline transition."""
from __future__ import annotations

import math

from . import train as single_gpu
from .model import EVRPTWRLPolicy
from ..common.distributed_entrypoints import parse_distributed_args, run_distributed_adapter
from ..common.protocol_entrypoints import run_evrptw_rl


def parse_args(argv=None):
    return parse_distributed_args(single_gpu.parse_args, argv)


def prepare_method(args):
    single_gpu._configure_station_auxiliary(args)
    single_gpu.configure_method_fields(args)
    if args.ema_warmup_steps < 0:
        raise ValueError("--ema-warmup-steps must be nonnegative")
    if args.baseline_eval_interval <= 0:
        raise ValueError("--baseline-eval-interval must be positive")
    if not math.isfinite(args.ema_decay) or not 0.0 <= args.ema_decay <= 1.0:
        raise ValueError("--ema-decay must be finite and in [0, 1]")
    fields = dict(getattr(args, "resolved_training_method_fields", None) or {})
    fields.update({
        "architecture": "evrptw_rl_structure2vec_v1",
        "structure2vec_rounds": int(args.structure2vec_rounds),
        "rollout_baseline_schedule_source": "existing_evrptw_rl_adapter",
        "rollout_baseline_interval_optimizer_updates": int(args.baseline_eval_interval),
        "rollout_baseline_probe_source": "training_pool_only",
        "rollout_baseline_warmup_optimizer_updates": int(args.ema_warmup_steps),
        "rollout_baseline_warmup_transition": "copy_actor_after_last_ema_optimizer_update",
        "rollout_baseline_first_probe_rule": "step_greater_than_warmup_and_divisible_by_interval",
    })
    args.resolved_training_method_fields = fields


def build_policy(args):
    policy = EVRPTWRLPolicy(embedding_dim=args.embedding_dim,
                           structure2vec_rounds=args.structure2vec_rounds)
    policy.activation_checkpoint_stride = int(args.activation_checkpoint_stride)
    return policy


def main() -> None:
    run_distributed_adapter(args=parse_args(), method="EVRPTW-RL", single_seed=single_gpu.set_seed,
                            prepare_method=prepare_method, build_policy=build_policy, run=run_evrptw_rl)


if __name__ == "__main__":
    main()
