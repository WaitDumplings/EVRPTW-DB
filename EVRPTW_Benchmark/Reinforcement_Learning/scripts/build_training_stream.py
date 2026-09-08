#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from EVRPTW_Benchmark.Reinforcement_Learning.common.training_stream import (
    atomic_write_stream,
    build_training_stream,
    file_sha256,
    normalize_scale,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a frozen shared DRL training ID stream.")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--customer-exposures", type=int, required=True)
    parser.add_argument("--allowed-family-ids", type=Path)
    parser.add_argument(
        "--reuse-source-index-sha256",
        help=(
            "Reuse this previously verified source-index SHA256 instead of "
            "reading the index a second time solely to recompute its file digest"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def declared_sha256(value: str, *, option: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{option} must be exactly 64 hexadecimal characters")
    return digest


def main() -> None:
    args = parse_args()
    scale = normalize_scale(args.scale)
    customers = int(scale.removeprefix("Cus"))
    if args.customer_exposures <= 0 or args.customer_exposures % customers:
        raise ValueError("customer exposures must be positive and divisible by the scale")
    allowed = None
    if args.allowed_family_ids is not None:
        allowed = [
            line.strip()
            for line in args.allowed_family_ids.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    frame = pd.read_parquet(args.index)
    stream, manifest = build_training_stream(
        frame,
        scale=scale,
        seed=args.seed,
        sample_count=args.customer_exposures // customers,
        allowed_family_ids=allowed,
    )
    manifest["source_index"] = str(args.index)
    manifest["source_index_sha256"] = (
        declared_sha256(
            args.reuse_source_index_sha256,
            option="--reuse-source-index-sha256",
        )
        if args.reuse_source_index_sha256 is not None
        else file_sha256(args.index)
    )
    manifest["allowed_family_ids_source"] = (
        str(args.allowed_family_ids) if args.allowed_family_ids else None
    )
    manifest["allowed_family_ids_sha256"] = (
        file_sha256(args.allowed_family_ids) if args.allowed_family_ids else None
    )
    atomic_write_stream(args.output, stream, manifest)
    print(json.dumps({"output": str(args.output), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
