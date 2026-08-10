#!/usr/bin/env python3
"""Run read-only source and configuration checks for a CLE build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evrptw_cle.preflight import preflight_profile
from evrptw_cle.util import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", type=Path, default=Path("configs/us_top10_cle_v1.json")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    profile = args.profile if args.profile.is_absolute() else root / args.profile
    report = preflight_profile(profile)
    if args.output:
        output = args.output if args.output.is_absolute() else root / args.output
        write_json(output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
