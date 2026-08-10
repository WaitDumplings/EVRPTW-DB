#!/usr/bin/env python3
"""Build one city's OSM terminal-only access connector layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evrptw_cle.terminal_access import build_terminal_access_layer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-slug", required=True)
    parser.add_argument("--pbf", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--operational-graph", type=Path, required=True)
    parser.add_argument("--output-graph", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--area-crs", required=True)
    parser.add_argument("--connection-tolerance-m", type=float, default=2.0)
    args = parser.parse_args()
    report = build_terminal_access_layer(
        city_slug=args.city_slug,
        pbf_file=args.pbf,
        boundary_file=args.boundary,
        operational_graph_path=args.operational_graph,
        output_graph_path=args.output_graph,
        output_report_path=args.output_report,
        area_crs=args.area_crs,
        connection_tolerance_m=args.connection_tolerance_m,
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
