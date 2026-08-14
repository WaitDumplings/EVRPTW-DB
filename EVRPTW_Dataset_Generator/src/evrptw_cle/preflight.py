"""Preflight checks for a reproducible City Logistics Environment build."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .hpms_match import discover_hpms_source
from .moves_speed import load_moves_speed_profile
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


def _record_large_source(path: Path) -> dict[str, Any]:
    """Record lightweight research preflight metadata without hashing a large file."""

    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        record["bytes"] = path.stat().st_size
    return record


def preflight_profile(
    profile_path: Path,
    *,
    selected_slugs: set[str] | None = None,
    hpms_source_root_override: Path | None = None,
    hpms_source_registry_override: Path | None = None,
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
        expected_building_sha = building_entry.get("source_sha256")
        building_source_record = (
            _record_file(building_path, expected_sha256=expected_building_sha)
            if expected_building_sha
            else _record_large_source(building_path)
        )
        city_record = {
            "city_slug": slug,
            "census_place_geoid": item.get("census_place_geoid"),
            "admin_boundary": _record_file(boundary),
            "service_boundary": _record_file(service_boundary),
            "building_source": building_source_record,
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
        elif profile.get("charging_policy", {}).get(
            "coordinate_evidence_required", False
        ):
            resolution_manifest_path = selected_afdc.with_suffix(".manifest.json")
            afdc_record["resolution_manifest"] = _record_file(
                resolution_manifest_path
            )
            if not resolution_manifest_path.is_file():
                errors.append(
                    f"resolved AFDC input lacks its resolution manifest: {resolution_manifest_path}"
                )
            else:
                resolution_manifest = read_json(resolution_manifest_path)
                resolution_inputs = resolution_manifest.get("inputs", {})
                required_inputs = profile.get("charging_policy", {}).get(
                    "required_resolution_inputs", []
                )
                missing_evidence = []
                for name in required_inputs:
                    evidence = resolution_inputs.get(name)
                    if not isinstance(evidence, dict) or not evidence.get(
                        "sha256"
                    ):
                        missing_evidence.append(name)
                        continue
                    evidence_path = Path(str(evidence.get("path", "")))
                    if not evidence_path.is_file() or sha256_file(
                        evidence_path
                    ) != evidence["sha256"]:
                        errors.append(
                            f"resolved AFDC coordinate evidence is unavailable or stale: {name}"
                        )
                if missing_evidence:
                    errors.append(
                        "resolved AFDC input lacks required coordinate evidence: "
                        f"{sorted(missing_evidence)}"
                    )
                resolved_hash = resolution_manifest.get("output", {}).get("sha256")
                if resolved_hash != afdc_record.get("sha256"):
                    errors.append(
                        "resolved AFDC table hash differs from its resolution manifest"
                    )
                status_counts = resolution_manifest.get(
                    "coordinate_validation_status_counts", {}
                )
                afdc_record["coordinate_validation_status_counts"] = status_counts
                if not status_counts:
                    errors.append(
                        "resolved AFDC manifest predates coordinate-validation tiers"
                    )

    hpms_records: list[dict[str, Any]] = []
    moves_profile_value = (
        profile.get("speed_profile", {}).get("moves_source", {}).get("profile")
    )
    moves_profile_path = (
        resolve_from(generator_root, moves_profile_value)
        if moves_profile_value
        else None
    )
    moves_profile_record = (
        _record_file(moves_profile_path) if moves_profile_path is not None else None
    )
    if moves_profile_path is None:
        errors.append("speed_profile.moves_source.profile is missing")
    elif not moves_profile_path.is_file():
        errors.append(f"missing compact MOVES5 speed profile: {moves_profile_path}")
    else:
        try:
            loaded_moves_profile = load_moves_speed_profile(moves_profile_path)
            moves_profile_record["profile_id"] = loaded_moves_profile["profile_id"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"invalid compact MOVES5 speed profile: {error}")
    hpms_raw_root_value = source_paths.get("hpms_raw_root")
    hpms_registry_value = source_paths.get("hpms_source_registry")
    hpms_required = bool(
        profile.get("speed_profile", {})
        .get("hpms_matching", {})
        .get("required", False)
    )
    hpms_raw_root = hpms_source_root_override or (
        resolve_from(generator_root, hpms_raw_root_value)
        if hpms_raw_root_value
        else None
    )
    hpms_registry_path = hpms_source_registry_override or (
        resolve_from(generator_root, hpms_registry_value)
        if hpms_registry_value
        else None
    )
    hpms_registry: dict[str, Any] = {}
    if hpms_registry_path is not None and hpms_registry_path.is_file():
        hpms_registry = read_json(hpms_registry_path)
    elif hpms_required:
        errors.append(f"missing HPMS source registry: {hpms_registry_path}")
    elif hpms_registry_path is not None:
        warnings.append(f"optional HPMS source registry is absent: {hpms_registry_path}")

    hpms_city_entries = hpms_registry.get("cities", {})
    for item in cities:
        slug = str(item["slug"])
        entry = hpms_city_entries.get(slug, {})
        source_stem = str(entry.get("source_stem", "")).strip()
        source_path = (
            discover_hpms_source(hpms_raw_root, source_stem)
            if hpms_raw_root is not None and source_stem
            else None
        )
        record = {
            "city_slug": slug,
            "source_stem": source_stem or None,
            "source": _record_large_source(source_path) if source_path else None,
        }
        hpms_records.append(record)
        if source_path is None:
            message = (
                f"{slug}: HPMS raw source is unavailable for source_stem="
                f"{source_stem or '<missing>'} under {hpms_raw_root}"
            )
            if hpms_required:
                errors.append(message)
            else:
                warnings.append(message)
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
            "moves5_speed_profile": moves_profile_record,
            "hpms_source_registry": (
                _record_file(hpms_registry_path) if hpms_registry_path else None
            ),
            "hpms_raw_root": str(hpms_raw_root) if hpms_raw_root else None,
            "hpms_sources": hpms_records,
            "pbf_sources": list(pbf_records.values()),
            "cities": city_records,
        },
    }
