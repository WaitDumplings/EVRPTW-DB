#!/usr/bin/env python3
"""Build the compact Amazon station-day artifacts consumed by Stage 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evrptw_stage2.amazon import build_amazon_stage2_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-build-inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.model_build_inputs
    result = build_amazon_stage2_artifacts(
        route_data_path=source / "route_data.json",
        travel_times_path=source / "travel_times.json",
        package_data_path=source / "package_data.json",
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
