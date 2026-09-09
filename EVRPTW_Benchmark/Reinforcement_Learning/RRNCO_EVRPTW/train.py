from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from ..common.protocol_entrypoints import run_rrnco_ev
from ..common.stage2_data import Stage2TaskPool
from ..common.training_protocol import (
    add_data_pass_arguments,
    build_adamw_optimizer,
    require_adamw,
)
from .model import RRNCOEVPolicy


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the RRNCO road-relation encoder in canonical EVRPTW-DB."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--family-root", type=Path)
    parser.add_argument("--scale", default="Cus100")
    parser.add_argument("--split-ids", default="train")
    parser.add_argument("--track-ids")
    parser.add_argument("--city-slugs")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--samples-per-instance", type=int, default=1)
    parser.add_argument("--baseline-eval-size", type=int, default=64)
    parser.add_argument("--baseline-alpha", type=float, default=0.05)
    parser.add_argument("--baseline-warmup-epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=2500)
    parser.add_argument("--ema-decay", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--n-encode-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--feedforward-hidden", type=int, default=512)
    parser.add_argument("--distance-sample-size", type=int, default=25)
    parser.add_argument("--tanh-clipping", type=float, default=10.0)
    parser.add_argument("--incomplete-penalty", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    add_data_pass_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_passes is None and args.training_epochs is None:
        raise ValueError("RRNCO-EV requires --training-epochs or --data-passes")
    set_seed(args.seed)
    pool = Stage2TaskPool(
        dataset_path=args.dataset_path,
        family_root=args.family_root,
        scale=args.scale,
        split_ids=args.split_ids,
        track_ids=args.track_ids,
        city_slugs=args.city_slugs,
        seed=args.seed,
        representation=args.training_representation,
        euclidean_manifest=args.euclidean_manifest,
    )
    policy = RRNCOEVPolicy(
        embedding_dim=args.embedding_dim,
        n_encode_layers=args.n_encode_layers,
        n_heads=args.n_heads,
        feedforward_hidden=args.feedforward_hidden,
        distance_sample_size=args.distance_sample_size,
        tanh_clipping=args.tanh_clipping,
    ).to(args.device)
    args.resolved_training_method_fields = {
        "architecture": "rrnco_ev_v1",
        "upstream_architecture": "ai4co/real-routing-nco",
        "embedding_dim": args.embedding_dim,
        "encoder_layers": args.n_encode_layers,
        "heads": args.n_heads,
        "feedforward_hidden": args.feedforward_hidden,
        "distance_sample_size": args.distance_sample_size,
        "relation_channels": [
            "directed_distance",
            "directed_time",
            "directed_energy",
            "angle",
        ],
        "constraint_source": "canonical_shared_environment_action_mask",
    }
    optimizer = build_adamw_optimizer(
        policy.parameters(),
        learning_rate=args.learning_rate,
        weight_decay=require_adamw(args),
    )
    run_rrnco_ev(args, pool, policy, optimizer)


if __name__ == "__main__":
    main()
