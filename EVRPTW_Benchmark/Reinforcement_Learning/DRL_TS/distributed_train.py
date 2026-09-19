"""Synchronous DRL-TS with native staged rewards and always-hard validation."""
from __future__ import annotations

from . import train as single_gpu
from .model import DRLTSPolicy
from ..common.distributed_entrypoints import parse_distributed_args, run_distributed_adapter
from ..common.protocol_entrypoints import run_drl_ts
from ..common.protocol_trainers import _resolve_soft_stage_contract


def parse_args(argv=None):
    return parse_distributed_args(single_gpu.parse_args, argv)


def prepare_method(args):
    single_gpu._configure_soft_auxiliary(args)
    if args.batches_per_epoch <= 0:
        raise ValueError("--batches-per-epoch must be positive")
    if args.activation_checkpoint_stride < 0:
        raise ValueError("--activation-checkpoint-stride must be nonnegative")
    single_gpu.configure_method_fields(args)
    args.resolved_training_method_fields.update({
        "architecture": "drl_ts_native_v1",
        "edge_row_gather": "direct_index_v1",
    })
    if args.activation_checkpoint_stride:
        args.resolved_training_method_fields.update({
            "activation_checkpoint_stride": int(args.activation_checkpoint_stride),
            "activation_checkpoint_semantics": "decoder_only_nonreentrant_full_recurrent_gradient_rng_preserved",
        })
    args.soft_stage_contract_snapshot = _resolve_soft_stage_contract(
        method="DRL-TS", fixed_epochs=int(args.training_epochs), total_passes=1,
        soft_stage_fraction=args.soft_stage_fraction,
        soft_stage_end_epoch=args.soft_stage_end_epoch)


def build_policy(args):
    policy = DRLTSPolicy(
        embedding_dim=args.embedding_dim, n_encode_layers=args.n_encode_layers,
        n_heads=args.n_heads, nearest_neighbors=args.nearest_neighbors,
        tanh_clipping=args.tanh_clipping)
    policy.activation_checkpoint_stride = int(args.activation_checkpoint_stride)
    return policy


def main() -> None:
    run_distributed_adapter(args=parse_args(), method="DRL-TS", single_seed=single_gpu.set_seed,
                            prepare_method=prepare_method, build_policy=build_policy, run=run_drl_ts)


if __name__ == "__main__":
    main()
