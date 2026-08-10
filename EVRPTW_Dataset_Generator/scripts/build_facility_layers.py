#!/usr/bin/env python3
"""Build road-anchored AFDC charger and OSM depot candidate layers for one city."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evrptw_cle.facilities import build_facility_layers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-slug", required=True)
    parser.add_argument("--afdc", type=Path, required=True)
    parser.add_argument("--depot-candidates", type=Path, required=True)
    parser.add_argument("--depot-summary", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--area-crs", required=True)
    parser.add_argument("--max-road-access-m", type=float, default=250.0)
    args = parser.parse_args()
    manifest = build_facility_layers(
        city_slug=args.city_slug,
        afdc_path=args.afdc,
        depot_candidates_path=args.depot_candidates,
        depot_summary_path=args.depot_summary,
        boundary_path=args.boundary,
        graph_path=args.graph,
        output_dir=args.output_dir,
        area_crs=args.area_crs,
        max_road_access_m=args.max_road_access_m,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
