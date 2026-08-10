"""Preflight checks for a reproducible City Logistics Environment build."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .util import sha256_file

AFDC_REQUIRED_COLUMNS = {
    "ID",
    "Fuel Type Code",
    "Status Code",
    "Access Code",
    "Country",
    "Latitude",
    "Longitude",
    "EV Level2 EVSE Num",
    "EV DC Fast Count",
    "EV Connector Types",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_from(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _record_file(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
    }
    if path.is_file():
        record["bytes"] = path.stat().st_size
        observed_hash = sha256_file(path)
        record["sha256"] = observed_hash
        if expected_sha256:
            record["expected_sha256"] = expected_sha256
            record["hash_matches"] = observed_hash == expected_sha256
    return record


def preflight_profile(
    profile_path: Path,
    *,
    selected_slugs: set[str] | None = None,
) -> dict[str, Any]:
    """Validate source paths and cross-config contracts without changing data."""

    profile_path = profile_path.resolve()
    generator_root = profile_path.parents[1]
    profile = read_json(profile_path)
    errors: list[str] = []
    warnings: list[str] = []
    if profile.get("schema") != "evrptw_cle_build_profile_v1":
        errors.append("unsupported or missing CLE profile schema")

    preset_path = resolve_from(generator_root, profile["city_preset"])
    building_registry_path = resolve_from(generator_root, profile["building_registry"])
    for label, path in (
        ("city preset", preset_path),
        ("building registry", building_registry_path),
    ):
        if not path.is_file():
            errors.append(f"missing {label}: {path}")
    if errors:
        return {
            "schema": "evrptw_cle_preflight_v1",
            "passed": False,
            "errors": errors,
            "warnings": warnings,
        }

    preset = read_json(preset_path)
    building_registry = read_json(building_registry_path)
    all_cities = preset.get("cities", [])
    preset_slugs = {str(item["slug"]) for item in all_cities}
    building_slugs = set(building_registry.get("cities", {}))
    if preset_slugs != building_slugs:
        errors.append(
            "city preset and building registry differ: "
            f"preset_only={sorted(preset_slugs - building_slugs)}, "
            f"building_only={sorted(building_slugs - preset_slugs)}"
        )
    if selected_slugs is not None:
        unknown = selected_slugs - preset_slugs
        if unknown:
            errors.append(f"unknown selected city slugs: {sorted(unknown)}")
        cities = [item for item in all_cities if str(item["slug"]) in selected_slugs]
    else:
        cities = all_cities

    city_records: list[dict[str, Any]] = []
    pbf_records: dict[str, dict[str, Any]] = {}
    for item in cities:
        slug = str(item["slug"])
        boundary = resolve_from(preset_path.parent, item["boundary_file"])
        service_boundary = resolve_from(preset_path.parent, item["query_mask_file"])
        pbf = resolve_from(preset_path.parent, item["pbf_file"])
        pbf_records.setdefault(
            str(pbf),
            {
                **_record_file(pbf),
                "source_url": item.get("pbf_source_url"),
                "cities": [],
            },
        )["cities"].append(slug)
        building_entry = building_registry.get("cities", {}).get(slug, {})
        building_root = resolve_from(
            generator_root, profile["source_paths"]["microsoft_building_root"]
        )
        building_path = building_root / str(building_entry.get("source_file", ""))
        city_record = {
            "city_slug": slug,
            "census_place_geoid": item.get("census_place_geoid"),
            "admin_boundary": _record_file(boundary),
            "service_boundary": _record_file(service_boundary),
            "building_source": _record_file(
                building_path,
                expected_sha256=building_entry.get("source_sha256"),
            ),
            "pbf_path": str(pbf),
        }
        for field in ("admin_boundary", "service_boundary", "building_source"):
            if not city_record[field]["exists"]:
                errors.append(f"{slug}: missing {field}: {city_record[field]['path']}")
            if city_record[field].get("hash_matches") is False:
                errors.append(f"{slug}: SHA-256 mismatch for {field}")
        city_records.append(city_record)

    for record in pbf_records.values():
        if not record["exists"]:
            errors.append(
                f"missing OSM PBF for {','.join(record['cities'])}: {record['path']}"
            )

    source_paths = profile["source_paths"]
    raw_afdc = resolve_from(generator_root, source_paths["afdc_raw_csv"])
    resolved_afdc = resolve_from(generator_root, source_paths["afdc_resolved_csv"])
    selected_afdc = resolved_afdc if resolved_afdc.is_file() else raw_afdc
    afdc_record = _record_file(selected_afdc)
    afdc_record["selected_variant"] = "resolved" if selected_afdc == resolved_afdc else "raw"
    if not selected_afdc.is_file():
        errors.append(f"missing AFDC input: expected {resolved_afdc} or {raw_afdc}")
    else:
        header = pd.read_csv(selected_afdc, nrows=0)
        missing = AFDC_REQUIRED_COLUMNS - set(header.columns)
        if missing:
            errors.append(f"AFDC input is missing columns: {sorted(missing)}")
        if selected_afdc == raw_afdc:
            warnings.append(
                "resolved AFDC file is absent; raw coordinates will be retained with "
                "explicit unresolved/raw-source labels"
            )

    hpms_root = resolve_from(generator_root, source_paths["hpms_edge_match_root"])
    if not hpms_root.exists():
        warnings.append(
            "optional normalized HPMS-to-OSM edge matches are absent; functional class "
            "will use OSM fallback and legal speed will not receive HPMS fills"
        )
    if shutil.which("osmium") is None:
        errors.append("osmium executable is not available; install the conda environment")

    return {
        "schema": "evrptw_cle_preflight_v1",
        "profile_id": profile.get("profile_id"),
        "selected_city_slugs": [str(item["slug"]) for item in cities],
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "inputs": {
            "profile": _record_file(profile_path),
            "city_preset": _record_file(preset_path),
            "building_registry": _record_file(building_registry_path),
            "afdc": afdc_record,
            "hpms_edge_match_root": str(hpms_root),
            "pbf_sources": list(pbf_records.values()),
            "cities": city_records,
        },
    }
