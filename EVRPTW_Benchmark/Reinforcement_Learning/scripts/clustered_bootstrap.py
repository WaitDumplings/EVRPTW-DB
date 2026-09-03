#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from EVRPTW_Benchmark.Reinforcement_Learning.common.clustered_bootstrap import (
    paired_parent_family_bootstrap,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired parent-family clustered DRL bootstrap.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--method-a", required=True)
    parser.add_argument("--method-b", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=49081)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.results) if args.results.suffix == ".parquet" else pd.read_csv(args.results)
    report = paired_parent_family_bootstrap(
        frame,
        method_a=args.method_a,
        method_b=args.method_b,
        replicates=args.replicates,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
