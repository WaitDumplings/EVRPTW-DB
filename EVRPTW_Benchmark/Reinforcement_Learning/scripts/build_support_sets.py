#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from EVRPTW_Benchmark.Reinforcement_Learning.common.support_selection import (
    select_parent_family_supports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze parent-family support sets for RQ2.")
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--family-metrics", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=73129)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    selection = select_parent_family_supports(
        pd.read_parquet(args.train_index),
        pd.read_parquet(args.family_metrics),
        fraction=args.fraction,
        seed=args.seed,
    )
    outputs = {
        "Full-support": selection.full_family_ids,
        "Random-10%-support": selection.random_family_ids,
        "Coverage-10%-support": selection.coverage_family_ids,
    }
    for name, family_ids in outputs.items():
        _atomic_text(
            args.output_dir / f"{name}.txt",
            "".join(f"{value}\n" for value in family_ids),
        )
    manifest = dict(selection.manifest)
    manifest.update(
        {
            "train_index": str(args.train_index),
            "family_metrics": str(args.family_metrics),
            "outputs": {name: f"{name}.txt" for name in outputs},
        }
    )
    _atomic_text(
        args.output_dir / "support_selection_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
