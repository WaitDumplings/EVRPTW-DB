#!/usr/bin/env python3
"""Build reproducible city-proper and land-only Census boundary layers.

The preset supplies Census Place GEOIDs.  For each city this script freezes the
official Place geometry as the administrative membership boundary and subtracts
intersecting Census AREAWATER polygons to build the land-only service mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union

TIGER_BASE = "https://www2.census.gov/geo/tiger/TIGER2025"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, retries: int = 4) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "evrptw-cle/0.2"}
            )
            with (
                urllib.request.urlopen(request, timeout=300) as response,
                partial.open("wb") as output,
            ):
                shutil.copyfileobj(response, output, length=1024 * 1024)
            partial.replace(destination)
            return destination
        except Exception:
            if partial.exists():
                partial.unlink()
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _polygonal_only(geometry: Any):
    geometry = make_valid(geometry)
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if isinstance(geometry, GeometryCollection):
        polygons = [
            part
            for part in geometry.geoms
            if isinstance(part, (Polygon, MultiPolygon))
        ]
        return unary_union(polygons) if polygons else Polygon()
    return Polygon()


def _select_place(place_archive: Path, geoid: str) -> gpd.GeoDataFrame:
    places = gpd.read_file(f"zip://{place_archive}")
    selected = places.loc[places["GEOID"].astype(str) == geoid].copy()
    if len(selected) != 1:
        raise RuntimeError(f"Census Place GEOID {geoid} matched {len(selected)} rows")
    return selected.to_crs("EPSG:4326")


def _county_geoids(
    city: gpd.GeoDataFrame, counties: gpd.GeoDataFrame, state_fips: str
) -> list[str]:
    local_crs = city.estimate_utm_crs()
    city_geometry = city.to_crs(local_crs).geometry.iloc[0]
    candidates = counties.loc[
        counties["STATEFP"].astype(str).str.zfill(2) == state_fips
    ].to_crs(local_crs)
    overlaps = candidates.geometry.map(lambda geometry: geometry.intersection(city_geometry).area)
    selected = candidates.loc[overlaps > 1.0]
    if selected.empty:
        raise RuntimeError("No positive-area county overlap for Census Place")
    return sorted(selected["GEOID"].astype(str).str.zfill(5).tolist())


def _build_land_mask(
    city: gpd.GeoDataFrame, county_geoids: list[str], source_root: Path
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    local_crs = city.estimate_utm_crs()
    city_local = city.to_crs(local_crs)
    admin = _polygonal_only(city_local.geometry.iloc[0])
    water_parts = []
    sources = []
    for county_geoid in county_geoids:
        archive = source_root / "areawater" / f"tl_2025_{county_geoid}_areawater.zip"
        _download(f"{TIGER_BASE}/AREAWATER/{archive.name}", archive)
        water = gpd.read_file(f"zip://{archive}").to_crs(local_crs)
        for geometry in water.loc[water.geometry.intersects(admin), "geometry"]:
            clipped = geometry.intersection(admin)
            if not clipped.is_empty:
                water_parts.append(clipped)
        sources.append(
            {
                "county_geoid": county_geoid,
                "file": f"areawater/{archive.name}",
                "sha256": _sha256(archive),
            }
        )
    water_union = unary_union(water_parts) if water_parts else Polygon()
    land_geometry = _polygonal_only(admin.difference(water_union))
    land = gpd.GeoDataFrame(
        [{"mask_semantics": "2025 Census Place minus 2025 Census AREAWATER"}],
        geometry=[land_geometry],
        crs=local_crs,
    ).to_crs("EPSG:4326")
    census_land_m2 = float(city.iloc[0]["ALAND"])
    census_water_m2 = float(city.iloc[0]["AWATER"])
    removed_water_m2 = admin.intersection(water_union).area
    return land, {
        "census_land_area_km2": census_land_m2 / 1_000_000,
        "census_water_area_km2": census_water_m2 / 1_000_000,
        "derived_admin_area_km2": admin.area / 1_000_000,
        "derived_land_area_km2": land_geometry.area / 1_000_000,
        "derived_removed_water_area_km2": removed_water_m2 / 1_000_000,
        "land_area_relative_error_vs_census": abs(land_geometry.area - census_land_m2)
        / max(census_land_m2, 1.0),
        "water_area_relative_error_vs_census": abs(removed_water_m2 - census_water_m2)
        / max(census_water_m2, 1.0),
        "land_mask_valid": bool(land_geometry.is_valid),
        "land_mask_empty": bool(land_geometry.is_empty),
        "areawater_sources": sources,
    }


def _write_city(
    item: dict[str, Any], counties: gpd.GeoDataFrame, source_root: Path, output_root: Path
) -> dict[str, Any]:
    geoid = str(item["census_place_geoid"])
    state_fips = geoid[:2]
    place_archive = source_root / "place" / f"tl_2025_{state_fips}_place.zip"
    _download(f"{TIGER_BASE}/PLACE/{place_archive.name}", place_archive)
    city = _select_place(place_archive, geoid)
    county_geoids = _county_geoids(city, counties, state_fips)
    land, qa = _build_land_mask(city, county_geoids, source_root)
    city_dir = output_root / item["slug"]
    city_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "STATEFP",
        "PLACEFP",
        "GEOID",
        "NAME",
        "NAMELSAD",
        "ALAND",
        "AWATER",
        "geometry",
    ]
    city[fields].to_file(city_dir / "admin_boundary.geojson", driver="GeoJSON")
    land.to_file(city_dir / "land_boundary.geojson", driver="GeoJSON")
    metadata = {
        "city_slug": item["slug"],
        "display_name": item.get("display_name", item["query"].split(",")[0]),
        "query": item["query"],
        "census_name": str(city.iloc[0]["NAMELSAD"]),
        "census_place_geoid": geoid,
        "county_geoids": county_geoids,
        "boundary_source": "2025 U.S. Census TIGER/Line Places",
        "water_source": "2025 U.S. Census TIGER/Line Area Hydrography (AREAWATER)",
        "admin_boundary_semantics": "city proper only",
        "land_boundary_semantics": "city proper minus official AREAWATER polygons",
        "place_source_file": f"place/{place_archive.name}",
        "place_source_sha256": _sha256(place_archive),
        "county_source_file": "county/tl_2025_us_county.zip",
        "county_source_sha256": _sha256(source_root / "county/tl_2025_us_county.zip"),
        "land_mask_qa": qa,
    }
    (city_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cities", nargs="*")
    args = parser.parse_args()
    preset = json.loads(args.preset.read_text(encoding="utf-8"))
    selected = set(args.cities or [item["slug"] for item in preset["cities"]])
    unknown = selected - {item["slug"] for item in preset["cities"]}
    if unknown:
        raise ValueError(f"Unknown city slugs: {sorted(unknown)}")
    county_archive = args.source_root / "county/tl_2025_us_county.zip"
    _download(f"{TIGER_BASE}/COUNTY/{county_archive.name}", county_archive)
    counties = gpd.read_file(f"zip://{county_archive}").to_crs("EPSG:4326")
    records = []
    for item in preset["cities"]:
        if item["slug"] not in selected:
            continue
        print(f"BOUNDARY {item['slug']}", flush=True)
        records.append(_write_city(item, counties, args.source_root, args.output_root))
    (args.output_root / "build_summary.json").write_text(
        json.dumps(
            {"preset_id": preset["preset_id"], "cities": records},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
