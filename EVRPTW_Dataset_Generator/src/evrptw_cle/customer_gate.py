"""Release Gate 2: latent customer-location contract verification."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from .util import sha256_file

EXPECTED_CONTRACT_SCHEMA = "evrptw_customer_location_contract_v1"
EXPECTED_MANIFEST_SCHEMA = "evrptw_customer_cle_v2"
EXPECTED_RELEASE_STATUSES = {"gate02_candidate", "release_candidate", "release_eligible"}
ALLOWED_TYPES = {"house", "manufactured_home", "small_apt", "medium_apt", "large_apt"}
ALLOWED_BANDS = {"unit_1", "units_2_4", "units_5_19", "units_20_49", "units_50_plus"}
ALLOWED_CORE_TIERS = {"G1_U1", "G1_U2", "G2_U1", "G2_U2"}
ALLOWED_GEOMETRY_TIERS = {"G1_containment", "G2_near_area_consistent"}
ALLOWED_UNIT_TIERS = {"U1_positive_resunits", "U2_occtype_interval"}
ALLOWED_ANCHOR_METHODS = {
    "footprint_boundary_to_operational_edge_projection",
}
EXPECTED_GEOMETRY_SOURCE = "microsoft_usbuildingfootprints_polygon"
EXACT_MATCH_METHOD = "nsi_point_within_microsoft_footprint"
NEAR_MATCH_METHOD = "nearest_microsoft_footprint_area_consistent"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _unit_band(units: int) -> str:
    if units == 1:
        return "unit_1"
    if units <= 4:
        return "units_2_4"
    if units <= 19:
        return "units_5_19"
    if units <= 49:
        return "units_20_49"
    return "units_50_plus"


def _expected_type(units: int, raw_occtypes: Any) -> str:
    text = str(raw_occtypes).upper()
    if units == 1:
        return "manufactured_home" if "RES2" in text else "house"
    if units <= 4:
        return "small_apt"
    if units <= 19:
        return "medium_apt"
    return "large_apt"


def _required_manifest_checks(
    manifest: dict[str, Any],
    *,
    slug: str,
    boundary_sha: str,
    road_graph_sha: str,
    selected_threshold_m: float | None,
) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema") != EXPECTED_MANIFEST_SCHEMA:
        errors.append(
            f"manifest schema is {manifest.get('schema')!r}; expected {EXPECTED_MANIFEST_SCHEMA!r}"
        )
    if manifest.get("status") not in EXPECTED_RELEASE_STATUSES:
        errors.append(f"manifest status {manifest.get('status')!r} is pilot/not release-candidate")
    if manifest.get("city_slug") != slug:
        errors.append("manifest city_slug does not match the preset")
    if manifest.get("boundary", {}).get("sha256") != boundary_sha:
        errors.append("customer CLE layer does not use the frozen land service boundary")
    if manifest.get("road_graph", {}).get("sha256") != road_graph_sha:
        errors.append("customer CLE layer does not use the frozen operational road graph")
    road_access = manifest.get("road_access", {})
    source = manifest.get("source", {})
    if source.get("nsi", {}).get("dataset") != "USACE National Structure Inventory 2026 Base":
        errors.append("NSI 2026 source snapshot is not declared")
    microsoft = source.get("microsoft_footprints", {})
    if microsoft.get("dataset") != "Microsoft USBuildingFootprints":
        errors.append("Microsoft USBuildingFootprints source snapshot is not declared")
    if not microsoft.get("sha256"):
        errors.append("Microsoft footprint source SHA-256 is missing")
    expected_methods = [EXACT_MATCH_METHOD, NEAR_MATCH_METHOD]
    if manifest.get("physical_location", {}).get("core_match_methods") != expected_methods:
        errors.append("core physical-location match methods do not match the Gate 2 contract")
    if manifest.get("physical_location", {}).get("near_match_manual_audit_passed") is not True:
        errors.append("near-footprint geometry matches have not passed the manual audit")
    if manifest.get("classification", {}).get("default_house_count") != 0:
        errors.append("default_house_count must be exactly zero")
    if road_access.get("frozen_acceptance_threshold_m") != selected_threshold_m:
        errors.append("manifest road-access threshold differs from the frozen Gate 2 threshold")
    return errors


def _validate_location_table(
    location_path: Path,
    boundary_path: Path,
    required_columns: set[str],
    selected_threshold_m: float | None,
    minimum_match_share: float,
    minimum_type_match_share: float,
    minimum_access_share: float,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    frame = gpd.read_parquet(location_path)
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        return [f"latent table is missing required columns: {missing_columns}"], {
            "location_count": len(frame),
            "core_location_count": None,
        }
    if frame.empty:
        return ["latent location table is empty"], {"location_count": 0, "core_location_count": 0}
    if frame["latent_service_location_id"].duplicated().any():
        errors.append("latent_service_location_id is not unique")
    if frame["active_customer"].fillna(False).astype(bool).any():
        errors.append("Stage 1 table contains active customers")

    boundary = gpd.read_file(boundary_path).geometry.union_all()
    outside_count = int((~frame.geometry.map(boundary.covers)).sum())
    if outside_count:
        errors.append(f"{outside_count} locations fall outside the frozen service boundary")

    geometry_eligible = frame["geometry_core_eligible"].fillna(False).astype(bool)
    access_eligible = frame["road_access_default_eligible"].fillna(False).astype(bool)
    eligible = frame["customer_release_eligible"].fillna(False).astype(bool)
    if (eligible != (geometry_eligible & access_eligible)).any():
        errors.append(
            "customer_release_eligible is not geometry_core_eligible AND "
            "road_access_default_eligible"
        )
    core = frame.loc[eligible].copy()
    if core.empty:
        errors.append("no release-eligible core customer locations")
    if not set(core["service_location_type"].dropna()).issubset(ALLOWED_TYPES):
        errors.append("core table contains an unsupported service_location_type")
    if not set(core["residential_unit_band"].dropna()).issubset(ALLOWED_BANDS):
        errors.append("core table contains an unsupported residential_unit_band")
    if not set(core["source_confidence_tier"].dropna()).issubset(ALLOWED_CORE_TIERS):
        errors.append("core table contains a non-core source confidence tier")
    if not set(core["geometry_evidence_tier"].dropna()).issubset(ALLOWED_GEOMETRY_TIERS):
        errors.append("core table contains a non-core geometry evidence tier")
    if not set(core["unit_evidence_tier"].dropna()).issubset(ALLOWED_UNIT_TIERS):
        errors.append("core table contains a non-core unit evidence tier")
    if (core["geometry_source"] != EXPECTED_GEOMETRY_SOURCE).any():
        errors.append("core table contains a non-Microsoft physical geometry")
    if not set(core["geometry_match_method"].dropna()).issubset(
        {EXACT_MATCH_METHOD, NEAR_MATCH_METHOD}
    ):
        errors.append("core table contains an unsupported NSI/Microsoft match method")
    geometry_match_distance = pd.to_numeric(core["geometry_match_distance_m"], errors="coerce")
    area_factor = pd.to_numeric(core["footprint_area_factor"], errors="coerce")
    exact = core["geometry_evidence_tier"] == "G1_containment"
    near = core["geometry_evidence_tier"] == "G2_near_area_consistent"
    if geometry_match_distance.isna().any() or (geometry_match_distance < 0).any():
        errors.append("core table has invalid geometry_match_distance_m")
    if (core.loc[exact, "geometry_match_method"] != EXACT_MATCH_METHOD).any() or (
        geometry_match_distance.loc[exact] != 0
    ).any():
        errors.append("G1 containment rows do not have exact-containment geometry evidence")
    if (core.loc[near, "geometry_match_method"] != NEAR_MATCH_METHOD).any() or (
        geometry_match_distance.loc[near] > 10
    ).any() or (area_factor.loc[near] > 4).any():
        errors.append("G2 near-match rows exceed the frozen 10 m / area-factor-4 rule")
    expected_combined = (
        core["geometry_evidence_tier"].str.slice(0, 2)
        + "_"
        + core["unit_evidence_tier"].str.slice(0, 2)
    )
    if (expected_combined != core["source_confidence_tier"]).any():
        errors.append("source_confidence_tier is inconsistent with geometry/unit tiers")
    if core["microsoft_building_id"].isna().any():
        errors.append("core table has missing Microsoft building IDs")
    if core["quarantine_reason"].notna().any() and (
        core["quarantine_reason"].astype(str).str.strip() != ""
    ).any():
        errors.append("release-eligible rows carry a quarantine reason")

    units_numeric = pd.to_numeric(core["residential_units"], errors="coerce")
    invalid_units = units_numeric.isna() | (units_numeric < 1) | (units_numeric % 1 != 0)
    if invalid_units.any():
        errors.append("core residential_units must be positive integers")
    else:
        units = units_numeric.astype(int)
        lower = pd.to_numeric(core["residential_units_lower"], errors="coerce")
        upper = pd.to_numeric(core["residential_units_upper"], errors="coerce")
        if lower.isna().any() or (lower < 1).any() or (units < lower).any():
            errors.append("residential_units_lower does not bound the modeled point estimate")
        finite_upper = upper.notna()
        if (upper.loc[finite_upper] < lower.loc[finite_upper]).any() or (
            units.loc[finite_upper] > upper.loc[finite_upper]
        ).any():
            errors.append("residential_units_upper does not bound the modeled point estimate")
        expected_bands = units.map(_unit_band)
        if (expected_bands != core["residential_unit_band"].astype(str)).any():
            errors.append("residential_unit_band is inconsistent with residential_units")
        expected_types = [
            _expected_type(unit, raw)
            for unit, raw in zip(units, core["raw_nsi_occtypes"], strict=True)
        ]
        if (pd.Series(expected_types, index=core.index) != core["service_location_type"]).any():
            errors.append("service_location_type is inconsistent with units/RES2 evidence")

    if core["type_evidence"].astype(str).str.contains("default", case=False).any():
        errors.append("core table contains default type evidence")
    if core["units_evidence"].astype(str).str.contains("default", case=False).any():
        errors.append("core table contains default unit evidence")
    raw_occtypes = core["raw_nsi_occtypes"].astype(str).str.upper()
    if not raw_occtypes.str.contains(r"RES[123]", regex=True).all():
        errors.append("core table contains a row without ordinary-residential NSI evidence")
    if raw_occtypes.str.contains(r"RES[456]", regex=True).any():
        errors.append("core table contains institutional-residential NSI evidence")
    if not set(core["road_anchor_method"].dropna()).issubset(ALLOWED_ANCHOR_METHODS):
        errors.append("core table contains an unsupported road_anchor_method")
    if core["physical_edge_id"].isna().any():
        errors.append("core table has missing physical road anchors")
    if core[["edge_u", "edge_v", "edge_key"]].isna().any().any():
        errors.append("core table has incomplete directed edge identity")
    access_distance = pd.to_numeric(core["road_access_distance_m"], errors="coerce")
    if access_distance.isna().any() or (access_distance < 0).any():
        errors.append("core table has invalid road-access distance")
    if selected_threshold_m is not None and (access_distance > selected_threshold_m).any():
        errors.append("core table includes an over-threshold road-access connector")
    if selected_threshold_m is not None:
        row_threshold = pd.to_numeric(core["access_threshold_m"], errors="coerce")
        if row_threshold.isna().any() or not math.isclose(
            float(row_threshold.min()), float(selected_threshold_m)
        ) or not math.isclose(float(row_threshold.max()), float(selected_threshold_m)):
            errors.append("core row access_threshold_m is not the frozen global threshold")

    modeled_units = (
        pd.to_numeric(frame["residential_units"], errors="coerce").fillna(0).clip(lower=0)
    )
    total_unit_mass = float(modeled_units.sum())
    geometry_unit_mass = float(modeled_units.loc[geometry_eligible].sum())
    unit_weighted_match_share = (
        geometry_unit_mass / total_unit_mass if total_unit_mass > 0 else 0.0
    )
    if unit_weighted_match_share < minimum_match_share:
        errors.append(
            "unit-weighted Microsoft/NSI core match share "
            f"{unit_weighted_match_share:.4%} is below {minimum_match_share:.2%}"
        )

    type_unit_weighted_match_shares: dict[str, float] = {}
    for location_type, group in frame.groupby("service_location_type", observed=True):
        type_units = pd.to_numeric(group["residential_units"], errors="coerce").fillna(0)
        type_total = float(type_units.sum())
        type_geometry = float(
            type_units.loc[group["geometry_core_eligible"].fillna(False).astype(bool)].sum()
        )
        share = type_geometry / type_total if type_total > 0 else 0.0
        type_unit_weighted_match_shares[str(location_type)] = share
        if share < minimum_type_match_share:
            errors.append(
                f"{location_type} unit-weighted core match share {share:.4%} is below "
                f"{minimum_type_match_share:.2%}"
            )

    geometry_units = modeled_units.loc[geometry_eligible]
    access_unit_mass = float(modeled_units.loc[eligible].sum())
    access_denominator = float(geometry_units.sum())
    unit_weighted_access_share = (
        access_unit_mass / access_denominator if access_denominator > 0 else 0.0
    )
    if unit_weighted_access_share < minimum_access_share:
        errors.append(
            "unit-weighted default road-access coverage "
            f"{unit_weighted_access_share:.4%} is below {minimum_access_share:.2%}"
        )
    type_unit_weighted_access_shares: dict[str, float] = {}
    for location_type, group in frame.loc[geometry_eligible].groupby(
        "service_location_type", observed=True
    ):
        type_units = pd.to_numeric(group["residential_units"], errors="coerce").fillna(0)
        type_total = float(type_units.sum())
        type_access = float(
            type_units.loc[group["road_access_default_eligible"].fillna(False).astype(bool)].sum()
        )
        share = type_access / type_total if type_total > 0 else 0.0
        type_unit_weighted_access_shares[str(location_type)] = share

    summary = {
        "location_count": len(frame),
        "core_location_count": len(core),
        "quarantined_location_count": int((~eligible).sum()),
        "outside_service_boundary_count": outside_count,
        "unit_weighted_core_match_share": unit_weighted_match_share,
        "type_unit_weighted_core_match_shares": type_unit_weighted_match_shares,
        "unit_weighted_default_road_access_share": unit_weighted_access_share,
        "type_unit_weighted_default_road_access_shares": type_unit_weighted_access_shares,
        "service_location_type_counts": {
            str(key): int(value) for key, value in core["service_location_type"].value_counts().items()
        },
        "confidence_tier_counts": {
            str(key): int(value) for key, value in core["source_confidence_tier"].value_counts().items()
        },
    }
    return errors, summary


def verify_customer_gate(
    preset_path: Path,
    contract_path: Path,
    boundary_root: Path,
    city_root: Path,
    customer_root: Path,
    *,
    expected_city_count: int = 10,
) -> dict[str, Any]:
    """Verify the official ten-city latent-customer contract."""

    preset_path = preset_path.resolve()
    contract_path = contract_path.resolve()
    boundary_root = boundary_root.resolve()
    city_root = city_root.resolve()
    customer_root = customer_root.resolve()
    preset = _read_json(preset_path)
    contract = _read_json(contract_path)
    global_errors: list[str] = []

    if contract.get("schema") != EXPECTED_CONTRACT_SCHEMA:
        global_errors.append("customer contract schema is not supported")
    if contract.get("preset_id") != preset.get("preset_id"):
        global_errors.append("customer contract preset_id differs from the city preset")
    if contract.get("residential_scope", {}).get("default_house_allowed") is not False:
        global_errors.append("customer contract must explicitly forbid default_house")
    cities = preset.get("cities", [])
    if len(cities) != expected_city_count:
        global_errors.append(f"preset contains {len(cities)} cities; expected {expected_city_count}")

    access_contract = contract.get("road_access", {})
    selected_threshold = access_contract.get("selected_threshold_m")
    if access_contract.get("selection_status") != "frozen_from_ten_city_audit":
        global_errors.append("global road-access threshold is not frozen from a ten-city audit")
    if selected_threshold not in access_contract.get("candidate_thresholds_m", []):
        global_errors.append("selected road-access threshold is missing or not pre-registered")
        selected_threshold = None

    required_columns = set(contract.get("required_location_columns", []))
    minimum_match_share = float(
        contract.get("physical_location", {}).get(
            "minimum_city_unit_weighted_core_match_share", 1.0
        )
    )
    minimum_type_match_share = float(
        contract.get("physical_location", {}).get(
            "minimum_type_unit_weighted_core_match_share", 1.0
        )
    )
    minimum_access_share = float(
        access_contract.get("selection_rule", {}).get(
            "minimum_city_unit_weighted_coverage", 1.0
        )
    )
    city_reports: list[dict[str, Any]] = []
    for item in cities:
        slug = str(item["slug"])
        errors: list[str] = []
        customer_dir = customer_root / slug
        manifest_path = customer_dir / "customer_cle_manifest.json"
        boundary_path = boundary_root / slug / "land_boundary.geojson"
        road_manifest_path = city_root / slug / "manifest.json"
        if not manifest_path.exists():
            city_reports.append(
                {
                    "slug": slug,
                    "passed": False,
                    "artifact_status": "missing",
                    "errors": ["customer CLE-layer manifest is missing"],
                }
            )
            continue
        if not boundary_path.exists() or not road_manifest_path.exists():
            if not boundary_path.exists():
                errors.append("frozen land service boundary is missing")
            if not road_manifest_path.exists():
                errors.append("road-graph manifest is missing")
            city_reports.append(
                {
                    "slug": slug,
                    "passed": False,
                    "artifact_status": "present_but_unverifiable",
                    "errors": errors,
                }
            )
            continue

        manifest = _read_json(manifest_path)
        road_manifest = _read_json(road_manifest_path)
        operational_name = road_manifest.get("operational_graph", "graph_operational.graphml")
        operational_path = city_root / slug / operational_name
        boundary_sha = sha256_file(boundary_path)
        road_graph_sha = (
            sha256_file(operational_path)
            if operational_path.exists()
            else road_manifest.get("checksums", {}).get(operational_name, "")
        )
        errors.extend(
            _required_manifest_checks(
                manifest,
                slug=slug,
                boundary_sha=boundary_sha,
                road_graph_sha=road_graph_sha,
                selected_threshold_m=selected_threshold,
            )
        )

        location_path = customer_dir / "latent_service_locations.parquet"
        table_summary: dict[str, Any] = {}
        if not location_path.exists():
            errors.append("latent_service_locations.parquet is missing")
        else:
            table_errors, table_summary = _validate_location_table(
                location_path,
                boundary_path,
                required_columns,
                selected_threshold,
                minimum_match_share,
                minimum_type_match_share,
                minimum_access_share,
            )
            errors.extend(table_errors)
        expected_output_sha = manifest.get("output_sha256", {}).get("latent_service_locations")
        if location_path.exists() and expected_output_sha:
            if sha256_file(location_path) != expected_output_sha:
                errors.append("latent location table SHA-256 differs from manifest")
        elif location_path.exists():
            errors.append("latent location table SHA-256 is missing from manifest")

        city_reports.append(
            {
                "slug": slug,
                "passed": not errors,
                "artifact_status": manifest.get("status", "unknown"),
                "errors": errors,
                "table_summary": table_summary,
            }
        )

    failed_cities = [item["slug"] for item in city_reports if not item["passed"]]
    missing_cities = [
        item["slug"] for item in city_reports if item.get("artifact_status") == "missing"
    ]
    passed = not global_errors and not failed_cities
    return {
        "schema": "evrptw_release_gate_v1",
        "gate_id": "GATE_02_CUSTOMER_LOCATION_CONTRACT",
        "gate_name": "Physical residential service locations, modeled units, and road access",
        "generated_utc": datetime.now(UTC).isoformat(),
        "passed": passed,
        "contract_path": str(contract_path),
        "inputs": {
            "preset": str(preset_path),
            "boundary_root": str(boundary_root),
            "city_root": str(city_root),
            "customer_root": str(customer_root),
        },
        "summary": {
            "expected_city_count": expected_city_count,
            "checked_city_count": len(city_reports),
            "passed_city_count": len(city_reports) - len(failed_cities),
            "failed_city_count": len(failed_cities),
            "missing_customer_board_count": len(missing_cities),
            "missing_customer_boards": missing_cities,
            "selected_road_access_threshold_m": selected_threshold,
        },
        "global_errors": global_errors,
        "cities": city_reports,
    }
