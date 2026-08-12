from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from evrptw_cle.boundary_gate import verify_boundary_gate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _build_fixture(tmp_path: Path, *, land_outside: bool = False) -> tuple[Path, Path, Path]:
    config_dir = tmp_path / "configs"
    boundary_root = tmp_path / "boundaries"
    city_root = tmp_path / "cities"
    slug = "sample-city"
    geoid = "0100001"
    city_dir = boundary_root / slug
    city_dir.mkdir(parents=True)

    admin = gpd.GeoDataFrame(
        [{"GEOID": geoid}], geometry=[box(-1.0, -1.0, 1.0, 1.0)], crs="EPSG:4326"
    )
    land_geometry = box(-0.9, -0.9, 1.1 if land_outside else 0.9, 0.9)
    land = gpd.GeoDataFrame(
        [{"mask_semantics": "2025 Census Place minus 2025 Census AREAWATER"}],
        geometry=[land_geometry],
        crs="EPSG:4326",
    )
    admin_path = city_dir / "admin_boundary.geojson"
    land_path = city_dir / "land_boundary.geojson"
    admin.to_file(admin_path, driver="GeoJSON")
    land.to_file(land_path, driver="GeoJSON")

    archive_sha = "a" * 64
    metadata = {
        "city_slug": slug,
        "query": "Sample City, USA",
        "census_place_geoid": geoid,
        "boundary_source": "2025 U.S. Census TIGER/Line Places",
        "water_source": "2025 U.S. Census TIGER/Line Area Hydrography (AREAWATER)",
        "admin_boundary_semantics": "city proper only",
        "land_boundary_semantics": "city proper minus official AREAWATER polygons",
        "place_source_sha256": archive_sha,
        "county_source_sha256": archive_sha,
        "county_geoids": ["01001"],
        "land_mask_qa": {
            "land_area_relative_error_vs_census": 0.001,
            "land_mask_valid": True,
            "land_mask_empty": False,
            "derived_land_area_km2": 1.0,
            "areawater_sources": [
                {"county_geoid": "01001", "sha256": archive_sha}
            ],
        },
    }
    _write_json(city_dir / "metadata.json", metadata)

    manifest_city = {
        "slug": slug,
        "query": "Sample City, USA",
        "census_place_geoid": geoid,
        "admin_boundary_sha256": _sha256(admin_path),
        "land_boundary_sha256": _sha256(land_path),
        "place_archive_sha256": archive_sha,
        "county_geoids": ["01001"],
        "areawater_archives": [
            {"county_geoid": "01001", "sha256": archive_sha}
        ],
    }
    _write_json(
        boundary_root / "manifest.json",
        {
            "preset_id": "us_11city_population_v1",
            "boundary_source": "2025 U.S. Census TIGER/Line Places",
            "water_source": "2025 U.S. Census TIGER/Line Area Hydrography (AREAWATER)",
            "semantics": "admin boundary is city proper; land boundary is admin minus AREAWATER",
            "cities": [manifest_city],
        },
    )
    preset_path = config_dir / "preset.json"
    _write_json(
        preset_path,
        {
            "preset_id": "us_11city_population_v1",
            "selection_semantics": "Frozen city proper, not metro area.",
            "boundary_vintage": "2025 U.S. Census TIGER/Line Places and Area Hydrography",
            "cities": [
                {
                    "slug": slug,
                    "display_name": "Sample City",
                    "query": "Sample City, USA",
                    "census_place_geoid": geoid,
                    "boundary_file": "../boundaries/sample-city/admin_boundary.geojson",
                    "query_mask_file": "../boundaries/sample-city/land_boundary.geojson",
                }
            ],
        },
    )
    _write_json(
        city_root / slug / "manifest.json",
        {
            "selected_graph_role": "operational_routing",
            "graph_semantics": {
                "raw": "directed OSM MultiDiGraph clipped to the exact city boundary",
                "operational": "one weakly connected actual-OSM routing graph; outside-city roads are transit-only; no synthetic connector edges",
            },
            "operational_connectivity": {
                "passed": True,
                "connector_semantics": "actual OSM drive edges only; outside-city nodes are transit-only; no synthetic connector edges",
            },
            "provenance": {
                "boundary_source": {"source_sha256": _sha256(admin_path)},
                "query_mask_source": {"source_sha256": _sha256(land_path)},
            },
        },
    )
    return preset_path, boundary_root, city_root


def test_boundary_gate_passes_aligned_city_proper_contract(tmp_path: Path) -> None:
    preset, boundary_root, city_root = _build_fixture(tmp_path)
    report = verify_boundary_gate(preset, boundary_root, city_root, expected_city_count=1)
    assert report["passed"] is True
    assert report["summary"]["passed_city_count"] == 1


def test_boundary_gate_rejects_land_mask_outside_city_proper(tmp_path: Path) -> None:
    preset, boundary_root, city_root = _build_fixture(tmp_path, land_outside=True)
    report = verify_boundary_gate(preset, boundary_root, city_root, expected_city_count=1)
    assert report["passed"] is False
    assert any("outside city proper" in error for error in report["cities"][0]["errors"])


def test_boundary_gate_rejects_road_graph_boundary_hash_mismatch(tmp_path: Path) -> None:
    preset, boundary_root, city_root = _build_fixture(tmp_path)
    road_manifest_path = city_root / "sample-city" / "manifest.json"
    road_manifest = json.loads(road_manifest_path.read_text(encoding="utf-8"))
    road_manifest["provenance"]["boundary_source"]["source_sha256"] = "f" * 64
    _write_json(road_manifest_path, road_manifest)
    report = verify_boundary_gate(preset, boundary_root, city_root, expected_city_count=1)
    assert report["passed"] is False
    assert any("frozen admin boundary" in error for error in report["cities"][0]["errors"])
