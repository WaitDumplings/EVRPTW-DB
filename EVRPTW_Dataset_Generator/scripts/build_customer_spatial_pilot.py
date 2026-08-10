#!/usr/bin/env python3
"""Build a Gate 2 Microsoft/NSI containment-match pilot for one city."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evrptw_cle.customer_spatial import build_microsoft_nsi_spatial_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-slug", required=True)
    parser.add_argument("--building-dir", type=Path, required=True)
    parser.add_argument("--nsi-records", type=Path, required=True)
    parser.add_argument("--nsi-locations", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-unit-weighted-match-share", type=float, default=0.95)
    args = parser.parse_args()
    manifest = build_microsoft_nsi_spatial_pilot(
        city_slug=args.city_slug,
        building_dir=args.building_dir,
        nsi_records_path=args.nsi_records,
        nsi_locations_path=args.nsi_locations,
        boundary_path=args.boundary,
        output_dir=args.output_dir,
        minimum_unit_weighted_match_share=args.minimum_unit_weighted_match_share,
    )
    print(json.dumps(manifest["record_summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
