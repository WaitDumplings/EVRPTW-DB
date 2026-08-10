"""Reviewable AFDC coordinate resolution without silent coordinate replacement."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return NON_ALNUM.sub(" ", str(value).strip().lower()).strip()


def address_key(
    street: Any,
    city: Any,
    state: Any,
    postal_code: Any,
) -> str:
    postal = normalize_text(postal_code).split(" ")[0]
    return "|".join(
        (normalize_text(street), normalize_text(city), normalize_text(state), postal[:5])
    )


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius_m = 6_371_008.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    value = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius_m * math.asin(math.sqrt(value))


def _osm_address_key(row: pd.Series) -> str:
    street = " ".join(
        part
        for part in (
            str(row.get("addr_housenumber") or "").strip(),
            str(row.get("addr_street") or "").strip(),
        )
        if part
    )
    return address_key(
        street,
        row.get("addr_city"),
        row.get("addr_state"),
        row.get("addr_postcode"),
    )


def resolve_afdc_coordinates(
    afdc: pd.DataFrame,
    *,
    census_results: pd.DataFrame | None = None,
    osm_pois: pd.DataFrame | None = None,
    manual_overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return AFDC rows with raw, address-anchor, and resolved geometry fields.

    Resolution precedence is a manually reviewed override, then an OSM charging
    POI with an exact normalized address, then the original AFDC coordinate.
    Census geocoding is retained as an address-access anchor and QA comparison;
    it is never mislabeled as the exact charger parking-space coordinate.
    """

    result = afdc.copy()
    result["afdc_id"] = pd.to_numeric(result["ID"], errors="raise").astype(int)
    result["raw_afdc_latitude"] = pd.to_numeric(result["Latitude"], errors="coerce")
    result["raw_afdc_longitude"] = pd.to_numeric(result["Longitude"], errors="coerce")
    result["normalized_address_key"] = result.apply(
        lambda row: address_key(
            row.get("Street Address"), row.get("City"), row.get("State"), row.get("ZIP")
        ),
        axis=1,
    )
    result["address_anchor_latitude"] = np.nan
    result["address_anchor_longitude"] = np.nan
    result["address_anchor_source"] = "not_available"
    result["census_match_status"] = "not_requested"

    if census_results is not None and not census_results.empty:
        census = census_results.copy()
        census["afdc_id"] = pd.to_numeric(census["afdc_id"], errors="raise").astype(int)
        census = census.drop_duplicates("afdc_id", keep="last").set_index("afdc_id")
        for index, row in result.iterrows():
            afdc_id = int(row["afdc_id"])
            if afdc_id not in census.index:
                continue
            match = census.loc[afdc_id]
            result.at[index, "census_match_status"] = str(
                match.get("census_match_status", "unknown")
            )
            try:
                lon = float(match["census_longitude"])
                lat = float(match["census_latitude"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(lon) and math.isfinite(lat):
                result.at[index, "address_anchor_longitude"] = lon
                result.at[index, "address_anchor_latitude"] = lat
                result.at[index, "address_anchor_source"] = "us_census_geocoder"

    osm_by_address: dict[str, list[pd.Series]] = {}
    if osm_pois is not None and not osm_pois.empty:
        osm = osm_pois.copy()
        osm["normalized_address_key"] = osm.apply(_osm_address_key, axis=1)
        for _, row in osm.loc[osm["normalized_address_key"].str.len() > 3].iterrows():
            osm_by_address.setdefault(str(row["normalized_address_key"]), []).append(row)

    manual_by_id: dict[int, pd.Series] = {}
    if manual_overrides is not None and not manual_overrides.empty:
        manual = manual_overrides.copy()
        manual["afdc_id"] = pd.to_numeric(manual["afdc_id"], errors="raise").astype(int)
        manual_by_id = {
            int(row["afdc_id"]): row
            for _, row in manual.drop_duplicates("afdc_id", keep="last").iterrows()
        }

    resolved_lon: list[float] = []
    resolved_lat: list[float] = []
    resolved_source: list[str] = []
    statuses: list[str] = []
    matched_osm_ids: list[str] = []
    raw_to_address: list[float | None] = []
    raw_to_osm: list[float | None] = []
    notes: list[str] = []
    for row in result.itertuples(index=False):
        raw_lon = float(row.raw_afdc_longitude)
        raw_lat = float(row.raw_afdc_latitude)
        afdc_id = int(row.afdc_id)
        selected_lon, selected_lat = raw_lon, raw_lat
        source = "afdc_raw"
        status = "raw_retained_no_corroborating_exact_geometry"
        osm_id = ""
        note = ""
        address_distance = None
        if math.isfinite(float(row.address_anchor_longitude)) and math.isfinite(
            float(row.address_anchor_latitude)
        ):
            address_distance = haversine_m(
                raw_lon,
                raw_lat,
                float(row.address_anchor_longitude),
                float(row.address_anchor_latitude),
            )

        osm_distance = None
        candidates = osm_by_address.get(str(row.normalized_address_key), [])
        if candidates:
            candidate = min(
                candidates,
                key=lambda item: haversine_m(
                    raw_lon,
                    raw_lat,
                    float(item["longitude"]),
                    float(item["latitude"]),
                ),
            )
            selected_lon = float(candidate["longitude"])
            selected_lat = float(candidate["latitude"])
            osm_distance = haversine_m(raw_lon, raw_lat, selected_lon, selected_lat)
            source = "osm_charging_poi_exact_address"
            status = "resolved_osm_exact_address"
            osm_id = str(candidate.get("source_osm_id", ""))

        if afdc_id in manual_by_id:
            override = manual_by_id[afdc_id]
            selected_lon = float(override["resolved_longitude"])
            selected_lat = float(override["resolved_latitude"])
            source = "manual_reviewed_override"
            status = "resolved_manual_review"
            note = str(override.get("review_note", ""))

        resolved_lon.append(selected_lon)
        resolved_lat.append(selected_lat)
        resolved_source.append(source)
        statuses.append(status)
        matched_osm_ids.append(osm_id)
        raw_to_address.append(address_distance)
        raw_to_osm.append(osm_distance)
        notes.append(note)

    result["resolved_longitude"] = resolved_lon
    result["resolved_latitude"] = resolved_lat
    result["resolved_geometry_source"] = resolved_source
    result["location_resolution_status"] = statuses
    result["matched_osm_charging_id"] = matched_osm_ids
    result["raw_to_address_anchor_m"] = raw_to_address
    result["raw_to_osm_poi_m"] = raw_to_osm
    result["coordinate_review_note"] = notes
    result["coordinate_validation_tier"] = np.select(
        [
            result["location_resolution_status"].eq("resolved_manual_review"),
            result["location_resolution_status"].eq("resolved_osm_exact_address"),
            result["address_anchor_source"].eq("us_census_geocoder"),
        ],
        [
            "V1_manual_reviewed_exact",
            "V2_osm_exact_address",
            "V3_census_address_corroborated",
        ],
        default="V0_uncorroborated_source_coordinate",
    )
    result["coordinate_validation_status"] = np.select(
        [
            result["coordinate_validation_tier"].isin(
                {"V1_manual_reviewed_exact", "V2_osm_exact_address"}
            ),
            result["coordinate_validation_tier"].eq(
                "V3_census_address_corroborated"
            ),
        ],
        [
            "exact_geometry_corroborated",
            "address_corroborated_exact_geometry_unverified",
        ],
        default="uncorroborated_source_coordinate",
    )
    result["coordinate_candidate_eligible"] = result[
        "coordinate_validation_status"
    ].ne("uncorroborated_source_coordinate")
    result["coordinate_release_eligible"] = result[
        "coordinate_validation_status"
    ].eq("exact_geometry_corroborated")
    return result
