#!/usr/bin/env python3
"""Rebuild the Phase-1 corpus report from completed Stage-2 families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evrptw_stage2.metrics import aggregate_phase1_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-root", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate_phase1_metrics(args.instance_root)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
