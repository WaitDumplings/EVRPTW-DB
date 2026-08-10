"""Release Gate 1: reproducible city-boundary contract verification."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd

EXPECTED_PRESET_ID = "top10_us_cities_population_v1"
EXPECTED_BOUNDARY_SOURCE = "2025 U.S. Census TIGER/Line Places"
EXPECTED_WATER_SOURCE = "2025 U.S. Census TIGER/Line Area Hydrography (AREAWATER)"
EXPECTED_ADMIN_SEMANTICS = "city proper only"
EXPECTED_LAND_SEMANTICS = "city proper minus official AREAWATER polygons"
DEFAULT_MAX_LAND_AREA_RELATIVE_ERROR = 0.005


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _resolve_from_config(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _load_geometry(path: Path) -> tuple[gpd.GeoDataFrame, Any]:
    frame = gpd.read_file(path)
    if frame.empty or frame.geometry.is_empty.all():
        raise ValueError("geometry layer is empty")
    if frame.crs is None:
        raise ValueError("geometry layer has no CRS")
    return frame, frame.geometry.union_all()


def _append_if(condition: bool, errors: list[str], message: str) -> None:
    if condition:
        errors.append(message)


def verify_boundary_gate(
    preset_path: Path,
    boundary_root: Path,
    city_root: Path | None,
    *,
    expected_city_count: int = 10,
    max_land_area_relative_error: float = DEFAULT_MAX_LAND_AREA_RELATIVE_ERROR,
) -> dict[str, Any]:
    """Verify the official city-proper/service-mask and road-graph contract.

    ``city_root`` is required for the official hard gate. Passing ``None`` is useful
    only for boundary-production diagnostics and is recorded as such in the report.
    """

    preset_path = preset_path.resolve()
    boundary_root = boundary_root.resolve()
    city_root = city_root.resolve() if city_root is not None else None
    global_errors: list[str] = []

    preset = _read_json(preset_path)
    boundary_manifest_path = boundary_root / "manifest.json"
    if not boundary_manifest_path.exists():
        raise FileNotFoundError(f"missing boundary manifest: {boundary_manifest_path}")
    boundary_manifest = _read_json(boundary_manifest_path)

    _append_if(
        preset.get("preset_id") != EXPECTED_PRESET_ID,
        global_errors,
        f"preset_id must be {EXPECTED_PRESET_ID!r}",
    )
    selection_semantics = str(preset.get("selection_semantics", "")).lower()
    _append_if(
        "city proper" not in selection_semantics or "not metro" not in selection_semantics,
        global_errors,
        "selection_semantics must explicitly say city proper, not metro area",
    )
    _append_if(
        preset.get("boundary_vintage")
        != "2025 U.S. Census TIGER/Line Places and Area Hydrography",
        global_errors,
        "boundary_vintage is not the frozen 2025 Census Place/AREAWATER vintage",
    )
    _append_if(
        boundary_manifest.get("preset_id") != preset.get("preset_id"),
        global_errors,
        "boundary manifest preset_id does not match the preset",
    )
    _append_if(
        boundary_manifest.get("boundary_source") != EXPECTED_BOUNDARY_SOURCE,
        global_errors,
        "boundary manifest has the wrong boundary source",
    )
    _append_if(
        boundary_manifest.get("water_source") != EXPECTED_WATER_SOURCE,
        global_errors,
        "boundary manifest has the wrong water source",
    )
    _append_if(
        boundary_manifest.get("semantics")
        != "admin boundary is city proper; land boundary is admin minus AREAWATER",
        global_errors,
        "boundary manifest has the wrong boundary semantics",
    )

    preset_cities = preset.get("cities", [])
    manifest_cities = boundary_manifest.get("cities", [])
    _append_if(
        len(preset_cities) != expected_city_count,
        global_errors,
        f"preset contains {len(preset_cities)} cities; expected {expected_city_count}",
    )
    preset_slugs = [str(item.get("slug", "")) for item in preset_cities]
    preset_geoids = [str(item.get("census_place_geoid", "")) for item in preset_cities]
    _append_if(
        len(set(preset_slugs)) != len(preset_slugs), global_errors, "preset city slugs are not unique"
    )
    _append_if(
        len(set(preset_geoids)) != len(preset_geoids),
        global_errors,
        "preset Census Place GEOIDs are not unique",
    )
    invalid_geoids = [value for value in preset_geoids if re.fullmatch(r"\d{7}", value) is None]
    _append_if(
        bool(invalid_geoids),
        global_errors,
        f"invalid seven-digit Census Place GEOIDs: {invalid_geoids}",
    )

    manifest_by_slug = {str(item.get("slug", "")): item for item in manifest_cities}
    _append_if(
        set(manifest_by_slug) != set(preset_slugs),
        global_errors,
        "boundary manifest city set differs from the preset city set",
    )

    city_reports: list[dict[str, Any]] = []
    observed_area_errors: list[float] = []
    for item in preset_cities:
        slug = str(item.get("slug", ""))
        geoid = str(item.get("census_place_geoid", ""))
        errors: list[str] = []
        city_dir = boundary_root / slug
        admin_path = city_dir / "admin_boundary.geojson"
        land_path = city_dir / "land_boundary.geojson"
        metadata_path = city_dir / "metadata.json"

        configured_admin = _resolve_from_config(preset_path, str(item.get("boundary_file", "")))
        configured_land = _resolve_from_config(preset_path, str(item.get("query_mask_file", "")))
        _append_if(
            configured_admin != admin_path.resolve(),
            errors,
            "preset boundary_file does not resolve to the frozen admin boundary",
        )
        _append_if(
            configured_land != land_path.resolve(),
            errors,
            "preset query_mask_file does not resolve to the frozen land boundary",
        )

        missing = [
            path.name
            for path in (admin_path, land_path, metadata_path)
            if not path.exists()
        ]
        if missing:
            errors.append(f"missing boundary artifacts: {missing}")
            city_reports.append({"slug": slug, "geoid": geoid, "passed": False, "errors": errors})
            continue

        metadata = _read_json(metadata_path)
        manifest_city = manifest_by_slug.get(slug, {})
        expected_pairs = (
            (metadata.get("city_slug"), slug, "metadata city_slug"),
            (metadata.get("query"), item.get("query"), "metadata query"),
            (metadata.get("census_place_geoid"), geoid, "metadata GEOID"),
            (metadata.get("boundary_source"), EXPECTED_BOUNDARY_SOURCE, "metadata boundary source"),
            (metadata.get("water_source"), EXPECTED_WATER_SOURCE, "metadata water source"),
            (metadata.get("admin_boundary_semantics"), EXPECTED_ADMIN_SEMANTICS, "admin semantics"),
            (metadata.get("land_boundary_semantics"), EXPECTED_LAND_SEMANTICS, "land semantics"),
            (manifest_city.get("census_place_geoid"), geoid, "manifest GEOID"),
            (manifest_city.get("query"), item.get("query"), "manifest query"),
        )
        for actual, expected, label in expected_pairs:
            _append_if(actual != expected, errors, f"{label} mismatch: {actual!r} != {expected!r}")

        admin_sha = _sha256(admin_path)
        land_sha = _sha256(land_path)
        _append_if(
            manifest_city.get("admin_boundary_sha256") != admin_sha,
            errors,
            "admin boundary checksum differs from boundary manifest",
        )
        _append_if(
            manifest_city.get("land_boundary_sha256") != land_sha,
            errors,
            "land boundary checksum differs from boundary manifest",
        )
        for label, value in (
            ("metadata place archive", metadata.get("place_source_sha256")),
            ("metadata county archive", metadata.get("county_source_sha256")),
            ("manifest place archive", manifest_city.get("place_archive_sha256")),
        ):
            _append_if(not _is_sha256(value), errors, f"{label} has no valid SHA-256")
        _append_if(
            metadata.get("place_source_sha256") != manifest_city.get("place_archive_sha256"),
            errors,
            "metadata and manifest identify different Census Place archives",
        )
        _append_if(
            metadata.get("county_geoids") != manifest_city.get("county_geoids"),
            errors,
            "metadata and manifest county GEOID lists differ",
        )
        metadata_water_sources = {
            str(source.get("county_geoid")): source.get("sha256")
            for source in metadata.get("land_mask_qa", {}).get("areawater_sources", [])
        }
        manifest_water_sources = {
            str(source.get("county_geoid")): source.get("sha256")
            for source in manifest_city.get("areawater_archives", [])
        }
        _append_if(
            metadata_water_sources != manifest_water_sources,
            errors,
            "metadata and manifest identify different AREAWATER archives",
        )
        for county_geoid, archive_sha in metadata_water_sources.items():
            _append_if(
                re.fullmatch(r"\d{5}", county_geoid) is None or not _is_sha256(archive_sha),
                errors,
                f"invalid AREAWATER provenance for county {county_geoid!r}",
            )

        qa = metadata.get("land_mask_qa", {})
        area_error = qa.get("land_area_relative_error_vs_census")
        if isinstance(area_error, (int, float)):
            observed_area_errors.append(float(area_error))
            _append_if(
                area_error > max_land_area_relative_error,
                errors,
                "land area relative error exceeds "
                f"{max_land_area_relative_error:.4%}: {area_error:.4%}",
            )
        else:
            errors.append("land area relative error is missing or non-numeric")
        _append_if(qa.get("land_mask_valid") is not True, errors, "land mask QA is not valid")
        _append_if(qa.get("land_mask_empty") is not False, errors, "land mask QA says empty")
        _append_if(
            not isinstance(qa.get("derived_land_area_km2"), (int, float))
            or qa.get("derived_land_area_km2", 0) <= 0,
            errors,
            "derived land area is not positive",
        )

        try:
            admin_frame, admin_geometry = _load_geometry(admin_path)
            land_frame, land_geometry = _load_geometry(land_path)
            _append_if(not admin_geometry.is_valid, errors, "admin geometry is invalid")
            _append_if(not land_geometry.is_valid, errors, "land geometry is invalid")
            if "GEOID" not in admin_frame.columns:
                errors.append("admin GeoJSON has no GEOID field")
            else:
                admin_geoids = set(admin_frame["GEOID"].astype(str))
                _append_if(
                    admin_geoids != {geoid}, errors, f"admin GeoJSON GEOID is {admin_geoids}"
                )
            land_local = land_frame.to_crs(admin_frame.estimate_utm_crs())
            admin_local = admin_frame.to_crs(admin_frame.estimate_utm_crs())
            outside_area_m2 = float(
                land_local.geometry.union_all().difference(admin_local.geometry.union_all()).area
            )
            _append_if(
                outside_area_m2 > 1.0,
                errors,
                f"land service boundary extends {outside_area_m2:.3f} m2 outside city proper",
            )
        except (OSError, TypeError, ValueError) as error:
            errors.append(f"geometry validation failed: {error}")

        road_graph_aligned: bool | None = None
        if city_root is not None:
            road_manifest_path = city_root / slug / "manifest.json"
            if not road_manifest_path.exists():
                errors.append("road-graph manifest is missing")
                road_graph_aligned = False
            else:
                road_manifest = _read_json(road_manifest_path)
                provenance = road_manifest.get("provenance", {})
                graph_semantics = road_manifest.get("graph_semantics", {})
                operational = road_manifest.get("operational_connectivity", {})
                road_errors_before = len(errors)
                _append_if(
                    provenance.get("boundary_source", {}).get("source_sha256") != admin_sha,
                    errors,
                    "road graph does not use the frozen admin boundary",
                )
                _append_if(
                    provenance.get("query_mask_source", {}).get("source_sha256") != land_sha,
                    errors,
                    "road graph does not use the frozen land service mask",
                )
                raw_semantics = str(graph_semantics.get("raw", ""))
                operational_semantics = str(graph_semantics.get("operational", ""))
                _append_if(
                    "exact city boundary" not in raw_semantics,
                    errors,
                    "raw road graph is not declared as an exact city-boundary clip",
                )
                _append_if(
                    "outside-city roads are transit-only" not in operational_semantics
                    or "no synthetic connector edges" not in operational_semantics,
                    errors,
                    "operational road semantics do not enforce transit-only real connectors",
                )
                _append_if(
                    road_manifest.get("selected_graph_role") != "operational_routing",
                    errors,
                    "selected road graph is not operational_routing",
                )
                _append_if(
                    operational.get("passed") is not True,
                    errors,
                    "road-graph operational connectivity gate did not pass",
                )
                _append_if(
                    operational.get("connector_semantics")
                    != "actual OSM drive edges only; outside-city nodes are transit-only; "
                    "no synthetic connector edges",
                    errors,
                    "road-graph connector semantics are not frozen",
                )
                road_graph_aligned = len(errors) == road_errors_before

        city_reports.append(
            {
                "slug": slug,
                "display_name": item.get("display_name"),
                "census_place_geoid": geoid,
                "passed": not errors,
                "errors": errors,
                "admin_boundary_sha256": admin_sha,
                "land_boundary_sha256": land_sha,
                "land_area_relative_error_vs_census": area_error,
                "road_graph_aligned": road_graph_aligned,
            }
        )

    if city_root is None:
        global_errors.append(
            "city_root was not supplied; official road-graph alignment was not checked"
        )
    failed_cities = [item["slug"] for item in city_reports if not item["passed"]]
    passed = not global_errors and not failed_cities
    return {
        "schema": "evrptw_release_gate_v1",
        "gate_id": "GATE_01_BOUNDARY_CONTRACT",
        "gate_name": "City-proper service boundary and real-road routing envelope",
        "generated_utc": datetime.now(UTC).isoformat(),
        "passed": passed,
        "contract": {
            "official_city_unit": "2025 U.S. Census TIGER/Line Place (city proper)",
            "admin_boundary_role": "raw road-graph membership and bridge/tunnel retention",
            "service_boundary_role": "admin boundary minus Census AREAWATER; sole eligibility mask for customers, depots, and chargers",
            "routing_envelope_role": "real OSM roads outside city may connect city components but are transit_only",
            "synthetic_connector_edges_allowed": False,
            "max_land_area_relative_error_vs_census": max_land_area_relative_error,
        },
        "inputs": {
            "preset": str(preset_path),
            "boundary_root": str(boundary_root),
            "city_root": str(city_root) if city_root is not None else None,
        },
        "summary": {
            "expected_city_count": expected_city_count,
            "checked_city_count": len(city_reports),
            "passed_city_count": len(city_reports) - len(failed_cities),
            "failed_city_count": len(failed_cities),
            "failed_cities": failed_cities,
            "max_observed_land_area_relative_error_vs_census": (
                max(observed_area_errors) if observed_area_errors else None
            ),
        },
        "global_errors": global_errors,
        "cities": city_reports,
    }
