"""Shared opt-in torchrun setup for new REINFORCE adapters.

The deployed AM entry remains unchanged. Instance caches hold immutable CPU
instances independently per rank; sizing them changes I/O, not sampling or loss.
"""
from __future__ import annotations

import argparse
import fcntl
import sys

from .distributed import close_distributed, initialize_distributed
from .distributed_protocol import configure_distributed_contract
from .protocol_entrypoints import _training_reward_scale
from .protocol_trainers import prepare_training_objective
from .stage2_data import Stage2TaskPool
from .training_protocol import (
    build_adamw_optimizer, require_adamw, require_validation_rollout_steps,
)


def parse_distributed_args(single_parse, argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--distributed-backend", choices=("nccl", "gloo"))
    parser.add_argument("--distributed-timeout-seconds", type=int, default=7200)
    parser.add_argument("--expected-world-size", type=int)
    parser.add_argument("--instance-cache-size", type=int, default=256,
                        help="Maximum cached CPU instances per pool and worker; 0 disables caching.")
    options, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if options.instance_cache_size < 0:
        parser.error("--instance-cache-size must be nonnegative")
    original = sys.argv
    try:
        sys.argv = [original[0], *remaining]
        args = single_parse()
    finally:
        sys.argv = original
    for key, value in vars(options).items():
        setattr(args, key, value)
    return args


def run_distributed_adapter(*, args, method, single_seed, prepare_method, build_policy, run):
    context, args.device = initialize_distributed(
        device=args.device, backend=args.distributed_backend,
        timeout_seconds=args.distributed_timeout_seconds,
        expected_world_size=args.expected_world_size)
    lock = None
    try:
        def claim_output():
            nonlocal lock
            args.output_dir.mkdir(parents=True, exist_ok=True)
            lock = (args.output_dir / ".distributed.lock").open("a+")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("another distributed run owns this output directory") from error
        context.main_call(claim_output)
        with context.local_phase(f"{method} initialization"):
            if args.data_passes is not None or not args.training_epochs:
                raise ValueError("distributed adapters require fixed --training-epochs")
            prepare_method(args)
            configure_distributed_contract(args, context, method=method)
            require_validation_rollout_steps(args)
            prepare_training_objective(args)
            single_seed(args.seed)
            pool = Stage2TaskPool(
                dataset_path=args.dataset_path, family_root=args.family_root,
                scale=args.scale, split_ids=args.split_ids, track_ids=args.track_ids,
                city_slugs=args.city_slugs, seed=args.seed,
                cache_size=args.instance_cache_size,
                representation=args.training_representation, euclidean_manifest=args.euclidean_manifest)
            policy = build_policy(args).to(args.device)
            optimizer = build_adamw_optimizer(policy.parameters(), learning_rate=args.learning_rate,
                                              weight_decay=require_adamw(args))
            # Resolve dataset calibration before workers enter training collectives.
            _training_reward_scale(args, pool)
        run(args, pool, policy, optimizer)
    finally:
        if lock is not None:
            lock.close()
        close_distributed()
