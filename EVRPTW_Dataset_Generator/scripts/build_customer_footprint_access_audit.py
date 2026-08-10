#!/usr/bin/env python3
"""Build footprint-to-road access sensitivity for one Gate 2 pilot city."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evrptw_cle.customer_access import build_footprint_access_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-slug", required=True)
    parser.add_argument("--locations", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--area-crs", required=True)
    args = parser.parse_args()
    report = build_footprint_access_audit(
        city_slug=args.city_slug,
        location_path=args.locations,
        graph_path=args.graph,
        output_dir=args.output_dir,
        area_crs=args.area_crs,
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
