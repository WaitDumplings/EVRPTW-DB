#!/usr/bin/env python3
"""Preflight or extract one registered city from a frozen state GeoJSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evrptw_cle.building_registry import (
    extract_registered_city,
    preflight_registered_city,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-slug", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/top10_building_extraction_v1.json"),
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/buildings"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if args.preflight_only:
        result = preflight_registered_city(
            config_path=args.config,
            city_slug=args.city_slug,
            source_root=args.source_root,
            output_root=args.output_root,
            verify_source_hash=True,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        raise SystemExit(0 if result["passed"] else 1)

    result = extract_registered_city(
        config_path=args.config,
        city_slug=args.city_slug,
        source_root=args.source_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
