"""Train a shared AM policy using torchrun; physical batch is per GPU, effective is global."""
from __future__ import annotations

import argparse
import fcntl
import sys

from . import train as single_gpu
from .data import Stage2TaskPool
from .model import AMEVRPTWPolicy
from ..common.distributed import close_distributed, initialize_distributed
from ..common.distributed_protocol import configure_distributed_contract
from ..common.protocol_entrypoints import _training_reward_scale, run_am
from ..common.protocol_trainers import prepare_training_objective
from ..common.training_protocol import (
    build_adamw_optimizer, require_adamw, require_validation_rollout_steps,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--distributed-backend", choices=("nccl", "gloo"))
    parser.add_argument("--distributed-timeout-seconds", type=int, default=7200)
    parser.add_argument("--expected-world-size", type=int)
    options, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    original = sys.argv
    try:
        sys.argv = [original[0], *remaining]
        args = single_gpu.parse_args()
    finally:
        sys.argv = original
    for key, value in vars(options).items():
        setattr(args, key, value)
    return args


def main() -> None:
    args = parse_args()
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
        with context.local_phase("AM initialization"):
            configure_distributed_contract(args, context)
            require_validation_rollout_steps(args)
            objective_config = prepare_training_objective(args)
            single_gpu.set_seed(args.seed)
            pool = Stage2TaskPool(
                dataset_path=args.dataset_path, family_root=args.family_root,
                objective_config=objective_config,
                scale=args.scale, split_ids=args.split_ids, track_ids=args.track_ids,
                city_slugs=args.city_slugs, seed=args.seed,
                representation=args.training_representation, euclidean_manifest=args.euclidean_manifest)
            policy = AMEVRPTWPolicy(
                embedding_dim=args.embedding_dim, hidden_dim=args.embedding_dim,
                n_encode_layers=args.n_encode_layers, n_heads=args.n_heads,
                tanh_clipping=args.tanh_clipping).to(args.device)
            optimizer = build_adamw_optimizer(policy.parameters(), learning_rate=args.learning_rate,
                                              weight_decay=require_adamw(args))
            # run_am reuses this cache. Failures while reading calibration views
            # must reach every worker before any enters the training collectives.
            _training_reward_scale(args, pool)
        run_am(args, pool, policy, optimizer)
    finally:
        if lock is not None:
            lock.close()
        close_distributed()


if __name__ == "__main__":
    main()
