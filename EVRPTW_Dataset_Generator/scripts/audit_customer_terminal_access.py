#!/usr/bin/env python3
"""Audit customer access with OSM terminal-only ways and a private sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evrptw_cle.customer_access import build_terminal_scenario_access_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-slug", required=True)
    parser.add_argument("--locations", type=Path, required=True)
    parser.add_argument("--operational-graph", type=Path, required=True)
    parser.add_argument("--terminal-graph", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--area-crs", required=True)
    args = parser.parse_args()
    report = build_terminal_scenario_access_audit(
        city_slug=args.city_slug,
        location_path=args.locations,
        operational_graph_path=args.operational_graph,
        terminal_graph_path=args.terminal_graph,
        output_dir=args.output_dir,
        area_crs=args.area_crs,
    )
    compact = {
        name: {
            "p99_m": scenario["distance_quantiles_m"]["p99"],
            "unit_coverage_100m": scenario["threshold_sensitivity"]["100"][
                "covered_modeled_unit_share"
            ],
            "unit_coverage_200m": scenario["threshold_sensitivity"]["200"][
                "covered_modeled_unit_share"
            ],
            "best_access_layer_counts": scenario["best_access_layer_counts"],
        }
        for name, scenario in report["scenarios"].items()
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
