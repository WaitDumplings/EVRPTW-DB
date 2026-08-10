#!/usr/bin/env python3
"""Resolve AFDC coordinates from reviewable AFDC/Census/OSM/manual evidence."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from evrptw_cle.charger_resolution import resolve_afdc_coordinates
from evrptw_cle.util import sha256_file, write_json


def _read_optional(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--afdc", type=Path, required=True)
    parser.add_argument("--census-results", type=Path)
    parser.add_argument("--osm-pois", type=Path)
    parser.add_argument("--manual-overrides", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite resolved AFDC table: {args.output}")
    result = resolve_afdc_coordinates(
        pd.read_csv(args.afdc, low_memory=False),
        census_results=_read_optional(args.census_results),
        osm_pois=_read_optional(args.osm_pois),
        manual_overrides=_read_optional(args.manual_overrides),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    manifest = {
        "schema": "evrptw_afdc_coordinate_resolution_v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "row_count": len(result),
        "resolution_status_counts": result["location_resolution_status"].value_counts().to_dict(),
        "inputs": {
            "afdc": {"path": str(args.afdc.resolve()), "sha256": sha256_file(args.afdc)},
            "census_results": str(args.census_results.resolve()) if args.census_results else None,
            "osm_pois": str(args.osm_pois.resolve()) if args.osm_pois else None,
            "manual_overrides": str(args.manual_overrides.resolve()) if args.manual_overrides else None,
        },
        "output": {"path": str(args.output.resolve()), "sha256": sha256_file(args.output)},
        "semantic_limit": (
            "Census points are address-access anchors, not exact charger coordinates; "
            "unresolved rows retain raw AFDC geometry."
        ),
    }
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
