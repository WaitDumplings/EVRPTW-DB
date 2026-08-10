"""Microsoft-footprint/NSI spatial matching for Gate 2 customer boards."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.strtree import STRtree

from .util import sha256_file, write_json

NSI_OCCUPANCY_INTERVALS: dict[str, tuple[int, int | None]] = {
    "RES3A": (2, 2),
    "RES3B": (3, 4),
    "RES3C": (5, 9),
    "RES3D": (10, 19),
    "RES3E": (20, 49),
    "RES3F": (50, None),
}


def nsi_unit_interval(occtype: str, resunits: Any) -> tuple[int, int, int | None, str]:
    """Return modeled point/lower/upper units without hiding NSI zero-value fallback."""

    try:
        units = int(resunits)
    except (TypeError, ValueError):
        units = 0
    if units > 0:
        return units, units, units, "nsi_resunits_positive"
    code = str(occtype or "").upper()
    if code.startswith("RES1") or code == "RES2":
        return 1, 1, 1, "nsi_occtype_lower_bound"
    for prefix, (lower, upper) in NSI_OCCUPANCY_INTERVALS.items():
        if code.startswith(prefix):
            return lower, lower, upper, "nsi_occtype_lower_bound"
    raise ValueError(f"No ordinary-residential unit rule for NSI occupancy {occtype!r}")


def service_location_type(units: int, raw_occtypes: str) -> str:
    if units == 1:
        return "manufactured_home" if "RES2" in raw_occtypes.upper() else "house"
    if units <= 4:
        return "small_apt"
    if units <= 19:
        return "medium_apt"
    return "large_apt"


def residential_unit_band(units: int) -> str:
    if units == 1:
        return "unit_1"
    if units <= 4:
        return "units_2_4"
    if units <= 19:
        return "units_5_19"
    if units <= 49:
        return "units_20_49"
    return "units_50_plus"


def resolve_containment_matches(
    nsi_points: gpd.GeoDataFrame, footprints: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Return one audit row per NSI point; ambiguous containment is never auto-resolved."""

    points = nsi_points.reset_index(drop=True)
    polygons = footprints[["building_id", "geometry"]].reset_index(drop=True)
    if points.crs != polygons.crs:
        polygons = polygons.to_crs(points.crs)
    joined = gpd.sjoin(
        points[["geometry"]],
        polygons,
        how="left",
        predicate="covered_by",
    )
    candidates = (
        joined.dropna(subset=["building_id"])
        .groupby(level=0, sort=False)["building_id"]
        .agg(list)
    )
    result = pd.DataFrame(index=points.index)
    result["footprint_match_count"] = candidates.map(len).reindex(points.index, fill_value=0)
    result["candidate_microsoft_building_ids"] = candidates.map(
        lambda values: json.dumps(sorted(str(value) for value in values))
    ).reindex(points.index, fill_value="[]")
    result["microsoft_building_id"] = candidates.map(
        lambda values: str(values[0]) if len(values) == 1 else None
    ).reindex(points.index)
    result["geometry_match_status"] = np.select(
        [result["footprint_match_count"] == 1, result["footprint_match_count"] > 1],
        ["matched_unique_containment", "quarantine_multiple_containment"],
        default="quarantine_no_containment",
    )
    result["geometry_match_method"] = np.where(
        result["footprint_match_count"] == 1,
        "nsi_point_within_microsoft_footprint",
        None,
    )
    result["geometry_match_distance_m"] = np.where(
        result["footprint_match_count"] == 1, 0.0, np.nan
    )
    return result.reset_index(drop=True)


def _read_footprint_parts(building_dir: Path) -> tuple[gpd.GeoDataFrame, list[dict[str, Any]]]:
    part_paths = sorted((building_dir / "footprints").glob("part-*.parquet"))
    if not part_paths:
        raise FileNotFoundError(f"No footprint parts under {building_dir / 'footprints'}")
    frames = [gpd.read_parquet(path) for path in part_paths]
    frame = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    if frame["building_id"].duplicated().any():
        raise ValueError("Microsoft building_id is not unique")
    provenance = [
        {"file": path.name, "sha256": sha256_file(path), "row_count": len(part)}
        for path, part in zip(part_paths, frames, strict=True)
    ]
    return frame, provenance


def _add_unit_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    intervals = [
        nsi_unit_interval(row.occtype, row.resunits) for row in frame.itertuples(index=False)
    ]
    result = frame.copy()
    result["normalized_units"] = [value[0] for value in intervals]
    result["unit_lower"] = [value[1] for value in intervals]
    result["unit_upper"] = [value[2] for value in intervals]
    result["unit_evidence"] = [value[3] for value in intervals]
    return result


def _aggregate_matched_locations(
    crosswalk: gpd.GeoDataFrame,
    footprints: gpd.GeoDataFrame,
    mixed_use_by_group: dict[str, bool],
) -> gpd.GeoDataFrame:
    matched = crosswalk.loc[
        crosswalk["geometry_evidence_tier"].isin(
            {"G1_containment", "G2_near_area_consistent"}
        )
    ].copy()
    matched["mixed_use_source"] = matched["group_key"].astype(str).map(mixed_use_by_group).fillna(
        False
    )
    grouped = matched.groupby("microsoft_building_id", sort=True, observed=True)
    aggregates = grouped.agg(
        residential_record_count=("fd_id", "size"),
        raw_nsi_resunits_sum=("resunits", "sum"),
        residential_units=("normalized_units", "sum"),
        residential_units_lower=("unit_lower", "sum"),
        fallback_unit_record_count=(
            "unit_evidence",
            lambda values: int((values == "nsi_occtype_lower_bound").sum()),
        ),
        mixed_use_flag=("mixed_use_source", "max"),
        geometry_match_distance_m=("geometry_match_distance_m", "max"),
        footprint_area_factor=("footprint_area_factor", "max"),
    )
    aggregates["geometry_evidence_tier"] = grouped["geometry_evidence_tier"].apply(
        lambda values: (
            "G2_near_area_consistent"
            if (values == "G2_near_area_consistent").any()
            else "G1_containment"
        )
    )
    upper_open = grouped["unit_upper"].apply(lambda values: bool(values.isna().any()))
    upper_sum = grouped["unit_upper"].sum(min_count=1)
    aggregates["residential_units_upper"] = upper_sum.mask(upper_open)
    aggregates["nsi_fd_ids"] = grouped["fd_id"].apply(
        lambda values: json.dumps(sorted(int(value) for value in values))
    )
    aggregates["raw_nsi_occtypes"] = grouped["occtype"].apply(
        lambda values: json.dumps(sorted({str(value) for value in values}))
    )
    aggregates = aggregates.reset_index()
    aggregates["residential_units"] = aggregates["residential_units"].astype(int)
    aggregates["residential_units_lower"] = aggregates["residential_units_lower"].astype(int)
    aggregates["units_evidence"] = np.where(
        aggregates["fallback_unit_record_count"] == 0,
        "nsi_resunits_positive",
        "nsi_occtype_lower_bound",
    )
    aggregates["unit_evidence_tier"] = np.where(
        aggregates["fallback_unit_record_count"] == 0,
        "U1_positive_resunits",
        "U2_occtype_interval",
    )
    aggregates["residential_unit_band"] = aggregates["residential_units"].map(
        residential_unit_band
    )
    aggregates["service_location_type"] = [
        service_location_type(units, raw)
        for units, raw in zip(
            aggregates["residential_units"], aggregates["raw_nsi_occtypes"], strict=True
        )
    ]
    aggregates["type_evidence"] = np.where(
        aggregates["fallback_unit_record_count"] == 0, "nsi_resunits", "nsi_occtype"
    )
    aggregates["source_confidence_tier"] = (
        aggregates["geometry_evidence_tier"].str.slice(0, 2)
        + "_"
        + aggregates["unit_evidence_tier"].str.slice(0, 2)
    )
    aggregates["geometry_source"] = "microsoft_usbuildingfootprints_polygon"
    aggregates["geometry_match_method"] = np.where(
        aggregates["geometry_evidence_tier"] == "G1_containment",
        "nsi_point_within_microsoft_footprint",
        "nearest_microsoft_footprint_area_consistent",
    )
    aggregates["geometry_manual_audit_status"] = np.where(
        aggregates["geometry_evidence_tier"] == "G1_containment",
        "not_required",
        "pending",
    )
    aggregates["active_customer"] = False
    aggregates["customer_release_eligible"] = False
    aggregates["quarantine_reason"] = "road_access_threshold_pending"

    footprint_fields = [
        "building_id",
        "source_feature_index",
        "source_release",
        "capture_dates_range",
        "footprint_area_m2",
        "location_lon",
        "location_lat",
        "geometry",
    ]
    locations = aggregates.merge(
        footprints[footprint_fields],
        left_on="microsoft_building_id",
        right_on="building_id",
        how="left",
        validate="one_to_one",
    ).drop(columns="building_id")
    locations["latent_service_location_id"] = "msft_nsi_" + locations[
        "microsoft_building_id"
    ].astype(str)
    return gpd.GeoDataFrame(locations, geometry="geometry", crs=footprints.crs)


def build_microsoft_nsi_spatial_pilot(
    *,
    city_slug: str,
    building_dir: Path,
    nsi_records_path: Path,
    nsi_locations_path: Path,
    boundary_path: Path,
    output_dir: Path,
    minimum_unit_weighted_match_share: float = 0.95,
) -> dict[str, Any]:
    """Build an audit-only containment crosswalk and matched physical-location table."""

    output_dir.mkdir(parents=True, exist_ok=True)
    footprints, footprint_parts = _read_footprint_parts(building_dir)
    nsi = gpd.read_parquet(nsi_records_path).reset_index(drop=True)
    if nsi["fd_id"].duplicated().any():
        raise ValueError("NSI fd_id is not unique")
    matches = resolve_containment_matches(nsi, footprints)
    crosswalk = gpd.GeoDataFrame(
        pd.concat([nsi.reset_index(drop=True), matches], axis=1),
        geometry="geometry",
        crs=nsi.crs,
    )
    crosswalk = _add_unit_evidence(crosswalk)
    footprint_area_by_id = footprints.set_index("building_id")["footprint_area_m2"]
    crosswalk["microsoft_footprint_area_m2"] = crosswalk["microsoft_building_id"].map(
        footprint_area_by_id
    )
    crosswalk["nsi_footprint_area_m2"] = (
        pd.to_numeric(crosswalk["ftprntsqft"], errors="coerce").fillna(0) * 0.09290304
    )
    smaller = np.minimum(
        crosswalk["microsoft_footprint_area_m2"], crosswalk["nsi_footprint_area_m2"]
    )
    larger = np.maximum(
        crosswalk["microsoft_footprint_area_m2"], crosswalk["nsi_footprint_area_m2"]
    )
    crosswalk["footprint_area_factor"] = np.where(smaller > 0, larger / smaller, np.nan)
    exact_match = crosswalk["geometry_match_status"] == "matched_unique_containment"
    crosswalk["geometry_evidence_tier"] = np.where(
        exact_match, "G1_containment", "G0_quarantine"
    )
    crosswalk["geometry_manual_audit_status"] = np.where(
        exact_match, "not_required", "not_applicable"
    )

    nsi_locations = gpd.read_parquet(nsi_locations_path)
    mixed_use_by_group = dict(
        zip(
            nsi_locations["group_key"].astype(str),
            nsi_locations["mixed_use_flag"].fillna(False).astype(bool),
            strict=True,
        )
    )
    locations = _aggregate_matched_locations(crosswalk, footprints, mixed_use_by_group)
    boundary = gpd.read_file(boundary_path).geometry.union_all()
    service_points = gpd.GeoSeries(
        gpd.points_from_xy(locations["location_lon"], locations["location_lat"]),
        crs="EPSG:4326",
    )
    outside_location_count = int((~service_points.map(boundary.covers)).sum())
    if outside_location_count:
        raise ValueError(
            f"{outside_location_count} matched footprint representative points fall outside service boundary"
        )
    boundary_straddling_footprint_count = int(
        (~locations.geometry.map(boundary.covers)).sum()
    )

    crosswalk_path = output_dir / "microsoft_nsi_record_crosswalk.parquet"
    location_path = output_dir / "matched_residential_footprints.parquet"
    crosswalk.to_parquet(crosswalk_path, index=False)
    locations.to_parquet(location_path, index=False)

    matched = crosswalk["geometry_match_status"] == "matched_unique_containment"
    total_unit_mass = float(crosswalk["normalized_units"].sum())
    matched_unit_mass = float(crosswalk.loc[matched, "normalized_units"].sum())
    unit_share = matched_unit_mass / total_unit_mass if total_unit_mass else 0.0
    by_occtype = {}
    for occtype, group in crosswalk.groupby("occtype", sort=True, observed=True):
        group_matched = group["geometry_match_status"] == "matched_unique_containment"
        mass = float(group["normalized_units"].sum())
        by_occtype[str(occtype)] = {
            "record_count": len(group),
            "matched_record_count": int(group_matched.sum()),
            "matched_record_share": float(group_matched.mean()),
            "modeled_unit_mass": mass,
            "matched_unit_mass": float(group.loc[group_matched, "normalized_units"].sum()),
            "matched_unit_share": (
                float(group.loc[group_matched, "normalized_units"].sum()) / mass if mass else 0.0
            ),
        }

    building_summary_path = building_dir / "building_summary.json"
    building_summary = json.loads(building_summary_path.read_text(encoding="utf-8"))
    manifest = {
        "schema": "evrptw_customer_spatial_match_pilot_v1",
        "status": "gate02_spatial_match_pilot_not_release_eligible",
        "city_slug": city_slug,
        "generated_utc": datetime.now(UTC).isoformat(),
        "sources": {
            "microsoft_footprints": {
                "dataset": "Microsoft USBuildingFootprints",
                "original_source_sha256": building_summary.get("source", {}).get("sha256"),
                "building_summary_sha256": sha256_file(building_summary_path),
                "parts": footprint_parts,
            },
            "nsi": {
                "dataset": "USACE National Structure Inventory 2026 Base",
                "ordinary_residential_records_sha256": sha256_file(nsi_records_path),
                "pilot_locations_sha256": sha256_file(nsi_locations_path),
            },
            "service_boundary_sha256": sha256_file(boundary_path),
        },
        "matching_contract": {
            "core_method": "nsi_point_within_microsoft_footprint",
            "ambiguous_policy": "quarantine; no smallest/nearest polygon tie-break",
            "unmatched_policy": "quarantine; no nearest-footprint promotion",
            "minimum_unit_weighted_match_share": minimum_unit_weighted_match_share,
        },
        "record_summary": {
            "microsoft_footprint_count": len(footprints),
            "ordinary_residential_nsi_record_count": len(crosswalk),
            "match_status_counts": {
                str(key): int(value)
                for key, value in crosswalk["geometry_match_status"].value_counts().items()
            },
            "matched_record_share": float(matched.mean()),
            "modeled_unit_mass": total_unit_mass,
            "matched_modeled_unit_mass": matched_unit_mass,
            "matched_modeled_unit_share": unit_share,
            "unit_weighted_match_gate_passed": unit_share >= minimum_unit_weighted_match_share,
            "by_nsi_occtype": by_occtype,
        },
        "location_summary": {
            "matched_physical_location_count": len(locations),
            "outside_service_boundary_representative_point_count": outside_location_count,
            "boundary_straddling_footprint_count": boundary_straddling_footprint_count,
            "boundary_straddling_policy": "retain complete footprint when its representative service point is inside; never clip the building polygon",
            "service_location_type_counts": {
                str(key): int(value)
                for key, value in locations["service_location_type"].value_counts().items()
            },
            "confidence_tier_counts": {
                str(key): int(value)
                for key, value in locations["source_confidence_tier"].value_counts().items()
            },
        },
        "outputs": {
            "record_crosswalk": str(crosswalk_path),
            "matched_residential_footprints": str(location_path),
        },
        "output_sha256": {
            "record_crosswalk": sha256_file(crosswalk_path),
            "matched_residential_footprints": sha256_file(location_path),
        },
        "known_limitations": [
            "Containment establishes a reproducible geometry crosswalk, not structure-level truth.",
            "Nearest-footprint fallbacks remain quarantined and are not analyzed as core locations.",
            "Road access has not yet been recomputed from footprint boundaries.",
            "All matched locations remain inactive and not release eligible.",
        ],
    }
    write_json(output_dir / "spatial_match_manifest.json", manifest)
    return manifest


def audit_unmatched_nearest_footprints(
    *,
    building_dir: Path,
    crosswalk_path: Path,
    nsi_locations_path: Path,
    output_dir: Path,
    area_crs: str,
) -> dict[str, Any]:
    """Measure whether quarantined NSI points have defensible nearby Microsoft polygons."""

    output_dir.mkdir(parents=True, exist_ok=True)
    footprints, _ = _read_footprint_parts(building_dir)
    crosswalk = gpd.read_parquet(crosswalk_path)
    unmatched = crosswalk.loc[
        crosswalk["geometry_match_status"] == "quarantine_no_containment"
    ].copy()
    footprint_local = footprints.to_crs(area_crs).reset_index(drop=True)
    unmatched_local = unmatched.to_crs(area_crs).reset_index(drop=True)
    polygon_geometries = np.asarray(footprint_local.geometry.to_numpy(), dtype=object)
    point_geometries = np.asarray(unmatched_local.geometry.to_numpy(), dtype=object)
    tree = STRtree(polygon_geometries)
    nearest_indices = np.asarray(tree.nearest(point_geometries), dtype=int)
    nearest_polygons = polygon_geometries[nearest_indices]
    distances = np.asarray(shapely.distance(point_geometries, nearest_polygons), dtype=float)
    nearest = footprint_local.iloc[nearest_indices].reset_index(drop=True)

    audit = unmatched.drop(columns="geometry").reset_index(drop=True).copy()
    audit["nearest_microsoft_building_id"] = nearest["building_id"].to_numpy()
    audit["nearest_distance_m"] = distances
    audit["microsoft_footprint_area_m2"] = nearest["footprint_area_m2"].to_numpy(dtype=float)
    audit["nsi_footprint_area_m2"] = (
        pd.to_numeric(audit["ftprntsqft"], errors="coerce").fillna(0).to_numpy(dtype=float)
        * 0.09290304
    )
    smaller = np.minimum(
        audit["microsoft_footprint_area_m2"].to_numpy(dtype=float),
        audit["nsi_footprint_area_m2"].to_numpy(dtype=float),
    )
    larger = np.maximum(
        audit["microsoft_footprint_area_m2"].to_numpy(dtype=float),
        audit["nsi_footprint_area_m2"].to_numpy(dtype=float),
    )
    audit["footprint_area_factor"] = np.where(smaller > 0, larger / smaller, np.inf)
    audit_path = output_dir / "unmatched_nearest_footprint_audit.parquet"
    gpd.GeoDataFrame(audit, geometry=unmatched.geometry.reset_index(drop=True), crs=unmatched.crs).to_parquet(
        audit_path, index=False
    )

    resolved = crosswalk.copy()
    if "geometry_evidence_tier" not in resolved:
        exact = resolved["geometry_match_status"] == "matched_unique_containment"
        resolved["geometry_evidence_tier"] = np.where(
            exact, "G1_containment", "G0_quarantine"
        )
        resolved["geometry_manual_audit_status"] = np.where(
            exact, "not_required", "not_applicable"
        )
    near_selected = (audit["nearest_distance_m"] <= 10) & (
        audit["footprint_area_factor"] <= 4
    )
    selected_near = audit.loc[
        near_selected,
        [
            "fd_id",
            "nearest_microsoft_building_id",
            "nearest_distance_m",
            "footprint_area_factor",
            "microsoft_footprint_area_m2",
        ],
    ].set_index("fd_id")
    resolved_fd = resolved["fd_id"].map(selected_near["nearest_microsoft_building_id"])
    promote = resolved_fd.notna() & (
        resolved["geometry_match_status"] == "quarantine_no_containment"
    )
    resolved.loc[promote, "microsoft_building_id"] = resolved_fd.loc[promote]
    resolved.loc[promote, "geometry_match_status"] = "candidate_near_area_consistent"
    resolved.loc[promote, "geometry_match_method"] = (
        "nearest_microsoft_footprint_area_consistent"
    )
    resolved.loc[promote, "geometry_match_distance_m"] = resolved.loc[promote, "fd_id"].map(
        selected_near["nearest_distance_m"]
    )
    resolved.loc[promote, "footprint_area_factor"] = resolved.loc[promote, "fd_id"].map(
        selected_near["footprint_area_factor"]
    )
    resolved.loc[promote, "microsoft_footprint_area_m2"] = resolved.loc[
        promote, "fd_id"
    ].map(selected_near["microsoft_footprint_area_m2"])
    resolved.loc[promote, "geometry_evidence_tier"] = "G2_near_area_consistent"
    resolved.loc[promote, "geometry_manual_audit_status"] = "pending"

    nsi_locations = gpd.read_parquet(nsi_locations_path)
    mixed_use_by_group = dict(
        zip(
            nsi_locations["group_key"].astype(str),
            nsi_locations["mixed_use_flag"].fillna(False).astype(bool),
            strict=True,
        )
    )
    resolved_locations = _aggregate_matched_locations(
        resolved, footprints, mixed_use_by_group
    )
    resolved_crosswalk_path = output_dir / "geometry_resolved_record_candidates.parquet"
    resolved_locations_path = output_dir / "geometry_resolved_footprint_candidates.parquet"
    resolved.to_parquet(resolved_crosswalk_path, index=False)
    resolved_locations.to_parquet(resolved_locations_path, index=False)

    thresholds = [1, 2, 5, 10, 20, 50, 100]
    area_factors: list[float | None] = [2.0, 4.0, 10.0, None]
    scenarios: dict[str, Any] = {}
    for threshold in thresholds:
        for factor in area_factors:
            selected = audit["nearest_distance_m"] <= threshold
            factor_label = "any" if factor is None else str(int(factor))
            if factor is not None:
                selected &= audit["footprint_area_factor"] <= factor
            key = f"distance_le_{threshold}m_area_factor_le_{factor_label}"
            unit_mass = float(audit.loc[selected, "normalized_units"].sum())
            scenarios[key] = {
                "additional_record_count": int(selected.sum()),
                "additional_record_share_of_unmatched": float(selected.mean()),
                "additional_modeled_unit_mass": unit_mass,
                "additional_unit_share_of_unmatched": (
                    unit_mass / float(audit["normalized_units"].sum())
                    if float(audit["normalized_units"].sum()) > 0
                    else 0.0
                ),
            }
    quantiles = np.quantile(distances, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1])
    by_occtype = {}
    for occtype, group in audit.groupby("occtype", sort=True, observed=True):
        by_occtype[str(occtype)] = {
            "record_count": len(group),
            "modeled_unit_mass": float(group["normalized_units"].sum()),
            "nearest_distance_quantiles_m": {
                key: float(value)
                for key, value in zip(
                    ("min", "p25", "p50", "p75", "p90", "p95", "p99", "max"),
                    np.quantile(
                        group["nearest_distance_m"],
                        [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1],
                    ),
                    strict=True,
                )
            },
        }
    core_geometry = resolved["geometry_evidence_tier"].isin(
        {"G1_containment", "G2_near_area_consistent"}
    )
    resolved_unit_mass = float(resolved.loc[core_geometry, "normalized_units"].sum())
    total_unit_mass = float(resolved["normalized_units"].sum())
    resolved_by_occtype = {}
    for occtype, group in resolved.groupby("occtype", sort=True, observed=True):
        group_core = group["geometry_evidence_tier"].isin(
            {"G1_containment", "G2_near_area_consistent"}
        )
        mass = float(group["normalized_units"].sum())
        core_mass = float(group.loc[group_core, "normalized_units"].sum())
        resolved_by_occtype[str(occtype)] = {
            "modeled_unit_mass": mass,
            "candidate_core_unit_mass": core_mass,
            "candidate_core_unit_share": core_mass / mass if mass else 0.0,
        }
    report = {
        "schema": "evrptw_customer_unmatched_nearest_audit_v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "semantics": "diagnostic only; no nearest-footprint row is promoted to core",
        "inputs": {
            "crosswalk_sha256": sha256_file(crosswalk_path),
            "building_dir": str(building_dir.resolve()),
            "area_crs": area_crs,
        },
        "summary": {
            "unmatched_record_count": len(audit),
            "unmatched_modeled_unit_mass": float(audit["normalized_units"].sum()),
            "nearest_distance_quantiles_m": {
                key: float(value)
                for key, value in zip(
                    ("min", "p25", "p50", "p75", "p90", "p95", "p99", "max"),
                    quantiles,
                    strict=True,
                )
            },
            "scenario_sensitivity": scenarios,
            "by_nsi_occtype": by_occtype,
            "frozen_development_rule_candidate": {
                "near_distance_max_m": 10,
                "footprint_area_factor_max": 4,
                "manual_audit_status": "pending",
                "candidate_core_record_count": int(core_geometry.sum()),
                "candidate_core_record_share": float(core_geometry.mean()),
                "candidate_core_modeled_unit_mass": resolved_unit_mass,
                "candidate_core_modeled_unit_share": (
                    resolved_unit_mass / total_unit_mass if total_unit_mass else 0.0
                ),
                "by_nsi_occtype": resolved_by_occtype,
            },
        },
        "outputs": {
            "unmatched_nearest_audit": str(audit_path),
            "geometry_resolved_record_candidates": str(resolved_crosswalk_path),
            "geometry_resolved_footprint_candidates": str(resolved_locations_path),
        },
        "output_sha256": {
            "unmatched_nearest_audit": sha256_file(audit_path),
            "geometry_resolved_record_candidates": sha256_file(resolved_crosswalk_path),
            "geometry_resolved_footprint_candidates": sha256_file(resolved_locations_path),
        },
    }
    write_json(output_dir / "unmatched_nearest_footprint_audit.json", report)
    return report
