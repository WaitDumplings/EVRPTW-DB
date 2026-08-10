#!/usr/bin/env python3
"""Audit nearby Microsoft polygons for unmatched NSI customer records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evrptw_cle.customer_spatial import audit_unmatched_nearest_footprints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--building-dir", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--nsi-locations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--area-crs", required=True)
    args = parser.parse_args()
    report = audit_unmatched_nearest_footprints(
        building_dir=args.building_dir,
        crosswalk_path=args.crosswalk,
        nsi_locations_path=args.nsi_locations,
        output_dir=args.output_dir,
        area_crs=args.area_crs,
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
