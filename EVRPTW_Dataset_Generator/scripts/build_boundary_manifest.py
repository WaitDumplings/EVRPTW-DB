#!/usr/bin/env python3
"""Build a portable provenance manifest for the reviewed ten-city boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evrptw_cle.util import sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--boundary-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    preset = json.loads(args.preset.read_text(encoding="utf-8"))
    records = []
    for item in preset["cities"]:
        slug = item["slug"]
        source = json.loads(
            (args.metadata_root / slug / "metadata.json").read_text(encoding="utf-8")
        )
        admin = args.boundary_root / slug / "admin_boundary.geojson"
        land = args.boundary_root / slug / "land_boundary.geojson"
        water_sources = []
        for water in source["land_mask_qa"]["areawater_sources"]:
            water_sources.append(
                {
                    "county_geoid": water["county_geoid"],
                    "archive": Path(water["file"]).name,
                    "sha256": water["sha256"],
                }
            )
        records.append(
            {
                "slug": slug,
                "query": item["query"],
                "census_place_geoid": item["census_place_geoid"],
                "census_name": source["census_name"],
                "county_geoids": source["county_geoids"],
                "place_archive": Path(source["place_source_file"]).name,
                "place_archive_sha256": source["place_source_sha256"],
                "areawater_archives": water_sources,
                "census_land_area_km2": source["land_mask_qa"]["census_land_area_km2"],
                "derived_land_area_km2": source["land_mask_qa"]["derived_land_area_km2"],
                "land_area_relative_error_vs_census": source["land_mask_qa"][
                    "land_area_relative_error_vs_census"
                ],
                "admin_boundary_sha256": sha256_file(admin),
                "land_boundary_sha256": sha256_file(land),
            }
        )

    payload = {
        "schema": "evrptw_top10_boundary_manifest_v1",
        "boundary_source": "2025 U.S. Census TIGER/Line Places",
        "water_source": "2025 U.S. Census TIGER/Line Area Hydrography (AREAWATER)",
        "semantics": "admin boundary is city proper; land boundary is admin minus AREAWATER",
        "preset_id": preset["preset_id"],
        "cities": records,
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
