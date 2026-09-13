"""Synchronous RRNCO-EV: per-rank physical batches, globally summed gradients."""
from __future__ import annotations

from . import train as single_gpu
from .model import RRNCOEVPolicy
from ..common.distributed_entrypoints import parse_distributed_args, run_distributed_adapter
from ..common.protocol_entrypoints import run_rrnco_ev


def parse_args(argv=None):
    return parse_distributed_args(single_gpu.parse_args, argv)


def build_policy(args):
    policy = RRNCOEVPolicy(
        embedding_dim=args.embedding_dim, n_encode_layers=args.n_encode_layers,
        n_heads=args.n_heads, feedforward_hidden=args.feedforward_hidden,
        distance_sample_size=args.distance_sample_size, tanh_clipping=args.tanh_clipping,
        graph_mode=args.graph_mode, aft_mode=args.aft_mode,
        distance_sampling=args.distance_sampling, relation_chunk_size=args.relation_chunk_size,
        checkpoint_bias=args.checkpoint_bias, relation_temperature=args.relation_temperature)
    if args.activation_checkpoint_stride:
        policy.activation_checkpoint_stride = int(args.activation_checkpoint_stride)
    return policy


def main() -> None:
    run_distributed_adapter(args=parse_args(), method="RRNCO-EV", single_seed=single_gpu.set_seed,
                            prepare_method=single_gpu.configure_method_fields,
                            build_policy=build_policy, run=run_rrnco_ev)


if __name__ == "__main__":
    main()
