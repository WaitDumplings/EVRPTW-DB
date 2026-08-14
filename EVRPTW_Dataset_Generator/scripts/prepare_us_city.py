#!/usr/bin/env python3
"""Resolve one U.S. Census Place and materialize a single-city CLE build contract."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import build_census_city_boundaries as census_boundaries
import geopandas as gpd

from evrptw_cle.util import sha256_file, write_json


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    if not slug:
        raise ValueError(f"Cannot derive a city slug from {value!r}")
    return slug


def _normalized_place_name(value: str) -> str:
    value = re.sub(r"\b(city|town|village|borough|municipality|cdp)\b", "", value.casefold())
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _load_state(registry_path: Path, value: str) -> dict[str, str]:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    normalized = value.strip().casefold()
    matches = [
        item
        for item in payload["states"]
        if normalized in {item["name"].casefold(), item["abbr"].casefold(), item["fips"]}
    ]
    if len(matches) != 1:
        supported = ", ".join(item["abbr"] for item in payload["states"])
        raise ValueError(f"Unknown U.S. state {value!r}. Supported abbreviations: {supported}")
    return {str(key): str(item) for key, item in matches[0].items()}


def _resolve_place(
    places: gpd.GeoDataFrame,
    *,
    city: str,
    state_fips: str,
    census_place_geoid: str | None,
) -> gpd.GeoDataFrame:
    state_places = places.loc[
        places["STATEFP"].astype(str).str.zfill(2).eq(state_fips)
    ].copy()
    if census_place_geoid:
        selected = state_places.loc[
            state_places["GEOID"].astype(str).str.zfill(7).eq(
                str(census_place_geoid).zfill(7)
            )
        ].copy()
    else:
        target = _normalized_place_name(city)
        normalized_names = state_places["NAME"].astype(str).map(_normalized_place_name)
        selected = state_places.loc[normalized_names.eq(target)].copy()
    if len(selected) != 1:
        target = _normalized_place_name(city)
        candidates = state_places.loc[
            state_places["NAME"]
            .astype(str)
            .map(_normalized_place_name)
            .map(lambda value: target in value or value in target),
            ["NAME", "NAMELSAD", "GEOID"],
        ].head(20)
        raise ValueError(
            f"{city!r} resolved to {len(selected)} Census Places in state {state_fips}. "
            "Pass --census-place-geoid to disambiguate. Nearby candidates: "
            f"{candidates.to_dict(orient='records')}"
        )
    return selected.to_crs("EPSG:4326")


def _pbf_url_and_name(
    *,
    state: dict[str, str],
    geofabrik_region: str | None,
    pbf_url: str | None,
) -> tuple[str, str]:
    if pbf_url:
        filename = Path(urlparse(pbf_url).path).name
        if not filename.endswith(".osm.pbf"):
            raise ValueError("--pbf-url must end in .osm.pbf")
        return pbf_url, filename
    region = (geofabrik_region or state["geofabrik_slug"]).strip("/")
    filename = f"{region.rsplit('/', 1)[-1]}-latest.osm.pbf"
    return f"https://download.geofabrik.de/north-america/us/{region}-latest.osm.pbf", filename


def prepare_us_city(
    *,
    city: str,
    state_value: str,
    generator_root: Path,
    city_slug: str | None = None,
    census_place_geoid: str | None = None,
    geofabrik_region: str | None = None,
    pbf_url: str | None = None,
    microsoft_url: str | None = None,
    hpms_service_url: str | None = None,
    custom_root: Path | None = None,
) -> dict[str, Any]:
    """Create boundaries and path-stable configs for one city; sources may be absent."""

    generator_root = generator_root.resolve()
    repository_root = generator_root.parent
    state_registry = generator_root / "configs/us_states_v1.json"
    state = _load_state(state_registry, state_value)
    slug = city_slug or _slugify(city)
    custom_root = (
        custom_root.resolve()
        if custom_root is not None
        else generator_root / "work/us-city-adapter" / slug
    )
    config_root = custom_root / "configs"
    boundary_root = custom_root / "boundaries"
    census_source_root = generator_root / "data/sources/census-tiger-2025"

    place_archive = (
        census_source_root / "place" / f"tl_2025_{state['fips']}_place.zip"
    )
    census_boundaries._download(
        f"{census_boundaries.TIGER_BASE}/PLACE/{place_archive.name}", place_archive
    )
    places = gpd.read_file(f"zip://{place_archive}")
    selected = _resolve_place(
        places,
        city=city,
        state_fips=state["fips"],
        census_place_geoid=census_place_geoid,
    )
    geoid = str(selected.iloc[0]["GEOID"]).zfill(7)
    census_name = str(selected.iloc[0]["NAME"])
    display_name = census_name
    query = f"{display_name}, {state['name']}, USA"

    pbf_source_url, pbf_filename = _pbf_url_and_name(
        state=state,
        geofabrik_region=geofabrik_region,
        pbf_url=pbf_url,
    )
    pbf_path = generator_root / "data/sources/geofabrik" / pbf_filename
    city_boundary_root = boundary_root / slug
    city_preset_path = config_root / "city_preset.json"
    city_item = {
        "slug": slug,
        "display_name": display_name,
        "query": query,
        "census_place_geoid": geoid,
        "boundary_file": str(city_boundary_root / "admin_boundary.geojson"),
        "query_mask_file": str(city_boundary_root / "land_boundary.geojson"),
        "pbf_file": str(pbf_path),
        "pbf_source_url": pbf_source_url,
    }
    preset = {
        "preset_id": f"us_city_adapter_{slug}_2025",
        "selection_semantics": (
            "User-selected U.S. Census Place; city proper is the service-boundary unit."
        ),
        "boundary_vintage": "2025 U.S. Census TIGER/Line Places and AREAWATER",
        "cities": [city_item],
    }
    config_root.mkdir(parents=True, exist_ok=True)
    write_json(city_preset_path, preset)

    county_archive = census_source_root / "county/tl_2025_us_county.zip"
    census_boundaries._download(
        f"{census_boundaries.TIGER_BASE}/COUNTY/{county_archive.name}", county_archive
    )
    counties = gpd.read_file(f"zip://{county_archive}").to_crs("EPSG:4326")
    metadata = census_boundaries._write_city(
        city_item, counties, census_source_root, boundary_root
    )
    land_boundary = city_boundary_root / "land_boundary.geojson"
    area_crs = str(gpd.read_file(land_boundary).estimate_utm_crs())

    building_source_root = (
        generator_root / "data/sources/microsoft-us-building-footprints"
    )
    building_registry_path = config_root / "building_registry.json"
    building_registry = {
        "schema": "evrptw_city_building_registry_v1",
        "registry_id": f"us_city_adapter_{slug}_buildings_v1",
        "official_city_preset": str(city_preset_path),
        "source_dataset": "Microsoft USBuildingFootprints",
        "source_integrity_mode": "record_on_extraction",
        "source_acquisition_status": (
            "Public state GeoJSON; bytes, SHA-256, and feature count are recorded during "
            "the single extraction scan instead of pre-hashed in research mode."
        ),
        "expected_crs": "EPSG:4326",
        "expected_geometry_type": "Polygon",
        "expected_property_keys": ["capture_dates_range", "release"],
        "membership_rule": (
            "building representative point covered by the Census city land boundary"
        ),
        "polygon_policy": "retain the complete source polygon; never clip at the boundary",
        "building_id_rule": "msft_usbf_{city_slug}_{one_based_source_feature_index:09d}",
        "batch_size": 50000,
        "density_grid_m": 500.0,
        "cities": {
            slug: {
                "label": display_name,
                "state": state["name"],
                "state_fips": state["fips"],
                "source_file": state["microsoft_file"],
                "source_sha256": None,
                "source_bytes": None,
                "source_feature_count": None,
                "boundary_file": str(land_boundary),
                "boundary_sha256": sha256_file(land_boundary),
                "area_crs": area_crs,
            }
        },
    }
    write_json(building_registry_path, building_registry)

    hpms_source_root = generator_root / "data/sources/hpms"
    hpms_registry_path = config_root / "hpms_registry.json"
    write_json(
        hpms_registry_path,
        {
            "schema": "evrptw_hpms_source_registry_v1",
            "description": "Single-city official FHWA 2018 HPMS public extract.",
            "supported_extensions": [".geojson"],
            "cities": {slug: {"source_stem": slug}},
        },
    )

    base_profile = json.loads(
        (generator_root / "configs/us_11city_cle_v1.json").read_text(encoding="utf-8")
    )
    work_root = custom_root / "build"
    release_root = repository_root / "EVRPTW_Dataset/CLE_v1/us_custom" / slug
    raw_afdc = generator_root / "data/sources/afdc/afdc_us_public_available_electric.csv"
    resolved_afdc = (
        generator_root
        / f"data/sources/afdc/afdc_us_public_available_electric_resolved_{slug}.csv"
    )
    base_profile.update(
        {
            "profile_id": f"us_city_adapter_{slug}_v1",
            "city_preset": str(city_preset_path),
            "building_registry": str(building_registry_path),
            "boundary_root": str(boundary_root),
            "work_root": str(work_root),
            "release_root": str(release_root),
        }
    )
    base_profile["source_paths"] = {
        "microsoft_building_root": str(building_source_root),
        "afdc_raw_csv": str(raw_afdc),
        "afdc_resolved_csv": str(resolved_afdc),
        "hpms_raw_root": str(hpms_source_root),
        "hpms_source_registry": str(hpms_registry_path),
    }
    base_profile["speed_profile"]["moves_source"]["profile"] = str(
        generator_root / "configs/us_moves5_speed_profile_v1.json"
    )
    profile_path = config_root / "cle_profile.json"
    write_json(profile_path, base_profile)

    hpms_url = hpms_service_url or (
        "https://geo.dot.gov/server/rest/services/Hosted/"
        f"{state['hpms_service_token']}_2018_PR/FeatureServer"
    )
    ms_url = microsoft_url or (
        "https://minedbuildings.z5.web.core.windows.net/legacy/usbuildings-v2/"
        f"{state['microsoft_file']}.zip"
    )
    contract = {
        "schema": "evrptw_us_city_cle_contract_v1",
        "city": {
            "requested_name": city,
            "display_name": display_name,
            "slug": slug,
            "state": state,
            "census_place_geoid": geoid,
            "census_name": metadata["census_name"],
        },
        "configs": {
            "profile": str(profile_path),
            "preset": str(city_preset_path),
            "building_registry": str(building_registry_path),
            "hpms_registry": str(hpms_registry_path),
        },
        "boundaries": {
            "root": str(boundary_root),
            "admin": str(city_boundary_root / "admin_boundary.geojson"),
            "land": str(land_boundary),
        },
        "sources": {
            "pbf": {"url": pbf_source_url, "path": str(pbf_path)},
            "microsoft_buildings": {
                "url": ms_url,
                "path": str(building_source_root / state["microsoft_file"]),
                "state_file": state["microsoft_file"],
            },
            "hpms": {
                "service_url": hpms_url,
                "path": str(hpms_source_root / f"{slug}.geojson"),
            },
            "afdc": {
                "path": str(raw_afdc),
                "resolved_path": str(resolved_afdc),
                "census_address_anchors_path": str(
                    generator_root
                    / "data/sources/afdc"
                    / f"afdc_census_address_anchors_{state['abbr'].lower()}.csv"
                ),
                "osm_charging_pois_path": str(
                    generator_root
                    / "data/sources/osm"
                    / f"osm_charging_pois_{slug}.csv"
                ),
            },
        },
        "outputs": {"work_root": str(work_root), "release_root": str(release_root)},
        "custom_root": str(custom_root),
    }
    contract_path = custom_root / "city_contract.json"
    write_json(contract_path, contract)
    contract["contract_path"] = str(contract_path)
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--city-slug")
    parser.add_argument("--census-place-geoid")
    parser.add_argument("--geofabrik-region")
    parser.add_argument("--pbf-url")
    parser.add_argument("--microsoft-url")
    parser.add_argument("--hpms-service-url")
    parser.add_argument("--custom-root", type=Path)
    args = parser.parse_args()
    generator_root = Path(__file__).resolve().parents[1]
    contract = prepare_us_city(
        city=args.city,
        state_value=args.state,
        generator_root=generator_root,
        city_slug=args.city_slug,
        census_place_geoid=args.census_place_geoid,
        geofabrik_region=args.geofabrik_region,
        pbf_url=args.pbf_url,
        microsoft_url=args.microsoft_url,
        hpms_service_url=args.hpms_service_url,
        custom_root=args.custom_root,
    )
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
