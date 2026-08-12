#!/usr/bin/env python3
"""Build an audit-only OSM depot-candidate preview for one city.

The output is deliberately a candidate layer, not a claim that every mapped
warehouse is a parcel-delivery depot.  Explicitly named carrier/logistics
facilities are separated from generic warehouse proxies, and every retained
candidate is anchored to the directed operational road graph.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, MultiLineString, mapping

OSMIUM_FILTERS = [
    "nwr/building=warehouse",
    "nwr/building=industrial",
    "nwr/industrial=warehouse",
    "nwr/office=logistics",
    "nwr/amenity=post_depot",
    "nwr/landuse=industrial",
    "nwr/landuse=depot",
    "nwr/depot",
    "nwr/logistics",
]

# OSM keys such as ``logistics`` and ``depot`` are open-ended.  Treating every
# non-empty value as parcel/logistics evidence admitted unrelated features such
# as ``logistics=spaceflight`` and ``depot=taxi``.  Keep the accepted values
# explicit and auditable.
ALLOWED_LOGISTICS_VALUES = {
    "yes",
    "cargo",
    "courier",
    "delivery",
    "distribution",
    "freight",
    "fulfillment",
    "logistics",
    "mail",
    "parcel",
    "post",
    "warehouse",
}
ALLOWED_DEPOT_VALUES = {
    "yes",
    "cargo",
    "courier",
    "delivery",
    "distribution",
    "freight",
    "fulfillment",
    "logistics",
    "mail",
    "parcel",
    "post",
    "warehouse",
}

CARRIER_PATTERNS = [
    (re.compile(r"\bamazon\b", re.IGNORECASE), "Amazon"),
    (re.compile(r"\b(?:ups|united parcel service)\b", re.IGNORECASE), "UPS"),
    (re.compile(r"\b(?:fedex|federal express)\b", re.IGNORECASE), "FedEx"),
    (re.compile(r"\bdhl\b", re.IGNORECASE), "DHL"),
    (re.compile(r"\b(?:usps|united states postal service)\b", re.IGNORECASE), "USPS"),
    (re.compile(r"\bontrac\b", re.IGNORECASE), "OnTrac"),
    (re.compile(r"\blasership\b", re.IGNORECASE), "LaserShip"),
    (re.compile(r"\bgls\b", re.IGNORECASE), "GLS"),
]

FACILITY_PATTERN = re.compile(
    r"\b(?:delivery|distribution|fulfillment|sortation|hub|processing|annex|depot|warehouse|logistics|parcel|courier|postal)\b",
    re.IGNORECASE,
)

# Parcel-carrier retail counters are not vehicle-dispatch depots.  OSM can still
# tag these points as ``amenity=post_depot``, so that tag alone is insufficient
# for the strict pool.
RETAIL_COUNTER_PATTERN = re.compile(
    r"\b(?:the\s+ups\s+store|fedex\s+office|postalannex|mailboxes\s+etc|"
    r"print(?:ing)?\s*(?:and|&)\s*ship|retail\s+shipping\s+counter)\b",
    re.IGNORECASE,
)

ROAD_CLASSES = {"motorway", "trunk", "primary", "secondary", "tertiary"}
OUTPUT_COLUMNS = [
    "candidate_rank",
    "city_slug",
    "candidate_id",
    "source_osm_id",
    "source_osm_url",
    "facility_name",
    "carrier",
    "evidence_tier",
    "evidence_reason",
    "function_signal",
    "strict_candidate_class",
    "depot_evidence_class",
    "verification_status",
    "last_mile_function_verified",
    "benchmark_depot_class",
    "strict_candidate_eligible",
    "optional_candidate_eligible",
    "cle_candidate_eligible",
    "depot_release_eligible",
    "facility_geometry_type",
    "facility_area_m2",
    "facility_area_known",
    "longitude",
    "latitude",
    "address",
    "osm_tags_json",
    "road_anchor_node",
    "road_anchor_longitude",
    "road_anchor_latitude",
    "road_snap_distance_m",
    "road_snap_distance_qa_flag",
    "road_anchor_strategy",
    "anchor_inside_city",
    "anchor_transit_only",
    "anchor_scc_id",
    "anchor_scc_node_share",
    "operational_eligible",
    "quarantine_reason",
    "manual_verification_required",
]


def _text(value: Any) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return ""
    return str(value).strip()


def _bool(value: Any) -> bool:
    return _text(value).lower() in {"true", "1", "yes"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(arguments: list[str]) -> None:
    completed = subprocess.run(arguments, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(arguments)}\n"
            f"{completed.stderr.strip()}"
        )


def _source_osm_identity(export_id: str) -> tuple[str, str]:
    """Translate osmium export ids, including synthetic area ids, to OSM ids."""
    if not export_id:
        return "", ""
    prefix = export_id[0]
    numeric = int(export_id[1:])
    if prefix == "n":
        object_type, object_id = "node", numeric
    elif prefix == "w":
        object_type, object_id = "way", numeric
    elif prefix == "r":
        object_type, object_id = "relation", numeric
    elif prefix == "a" and numeric % 2 == 0:
        object_type, object_id = "way", numeric // 2
    elif prefix == "a":
        object_type, object_id = "relation", (numeric - 1) // 2
    else:
        return export_id, ""
    source_id = f"{object_type[0]}{object_id}"
    return source_id, f"https://www.openstreetmap.org/{object_type}/{object_id}"


def _carrier(row: pd.Series) -> str:
    haystack = " ".join(
        _text(row.get(column))
        for column in ("name", "official_name", "operator", "brand", "company", "owner")
    )
    for pattern, label in CARRIER_PATTERNS:
        if pattern.search(haystack):
            return label
    return ""


def _candidate_tags(row: pd.Series) -> dict[str, str]:
    columns = [
        "name",
        "official_name",
        "operator",
        "brand",
        "building",
        "building:use",
        "landuse",
        "industrial",
        "office",
        "shop",
        "amenity",
        "depot",
        "logistics",
        "access",
        "access:delivery",
        "addr:housenumber",
        "addr:street",
        "addr:city",
        "addr:postcode",
        "website",
        "source",
        "source:url",
    ]
    return {column: value for column in columns if (value := _text(row.get(column)))}


def _is_matched_object(row: pd.Series) -> bool:
    building = _text(row.get("building")).lower()
    landuse = _text(row.get("landuse")).lower()
    industrial = _text(row.get("industrial")).lower()
    office = _text(row.get("office")).lower()
    amenity = _text(row.get("amenity")).lower()
    depot_value = _text(row.get("depot")).lower()
    logistics_value = _text(row.get("logistics")).lower()
    return any(
        (
            building in {"warehouse", "industrial"},
            landuse in {"industrial", "depot", "logistics", "warehouse"},
            industrial in {"warehouse", "logistics", "distribution"},
            office == "logistics",
            amenity == "post_depot",
            depot_value in ALLOWED_DEPOT_VALUES,
            logistics_value in ALLOWED_LOGISTICS_VALUES,
        )
    )


def _evidence_tier(
    row: pd.Series,
    facility_area_m2: float,
    min_warehouse_area_m2: float,
) -> tuple[str, str, int]:
    carrier = _carrier(row)
    name = _text(row.get("name")) or _text(row.get("official_name"))
    combined = " ".join(
        _text(row.get(column)) for column in ("name", "official_name", "operator", "brand")
    )
    building = _text(row.get("building")).lower()
    landuse = _text(row.get("landuse")).lower()
    industrial = _text(row.get("industrial")).lower()
    office = _text(row.get("office")).lower()
    amenity = _text(row.get("amenity")).lower()
    depot_value = _text(row.get("depot")).lower()
    logistics_value = _text(row.get("logistics")).lower()
    explicit_depot = (
        amenity == "post_depot"
        or office == "logistics"
        or depot_value in ALLOWED_DEPOT_VALUES
        or logistics_value in ALLOWED_LOGISTICS_VALUES
    )
    warehouse = (
        building == "warehouse"
        or industrial == "warehouse"
        or landuse in {"warehouse", "logistics", "depot"}
    )
    industrial_site = building == "industrial" or landuse == "industrial"
    facility_hint = bool(FACILITY_PATTERN.search(combined))
    has_address = bool(_text(row.get("addr:street")))

    retail_haystack = " ".join(
        _text(row.get(column))
        for column in ("name", "official_name", "operator", "brand", "website")
    )
    if RETAIL_COUNTER_PATTERN.search(retail_haystack):
        return "C_industrial_proxy", "parcel-carrier retail counter", 0

    # A carrier name is not enough: it must be tied to a physical warehouse,
    # an explicit dispatch-function name, or an explicit depot/logistics tag.
    # Area is kept as a continuous attribute and sensitivity flag, not used as
    # ground-truth dispatch evidence.
    strong_dispatch_evidence = (
        warehouse or facility_hint or (explicit_depot and facility_area_m2 > 0)
    )
    carrier_facility = bool(carrier) and strong_dispatch_evidence
    if carrier == "USPS":
        usps_facility_hint = bool(
            re.search(r"\b(?:distribution|processing|annex|depot|warehouse)\b", name, re.IGNORECASE)
        )
        carrier_facility = (
            warehouse
            or usps_facility_hint
            or (amenity == "post_depot" and facility_area_m2 > 0)
        )

    if carrier_facility:
        reasons = []
        if carrier:
            reasons.append(f"named {carrier} facility")
        if explicit_depot:
            reasons.append("explicit depot/logistics tag")
        if warehouse:
            reasons.append("warehouse tag")
        if industrial_site and not warehouse:
            reasons.append("industrial-facility tag")
        if facility_hint:
            reasons.append("delivery/hub/logistics name")
        score = 6 + 3 * bool(explicit_depot) + 2 * bool(warehouse) + bool(has_address)
        return "A_osm_explicit", "; ".join(reasons), score

    if carrier and explicit_depot and not warehouse and not facility_hint:
        return (
            "C_industrial_proxy",
            "carrier-branded point lacks physical dispatch-facility evidence",
            0,
        )

    warehouse_proxy = warehouse or office == "logistics" or explicit_depot
    if warehouse_proxy:
        reasons = ["warehouse/logistics land-use evidence"]
        if facility_area_m2 >= min_warehouse_area_m2:
            reasons.append(f"area sensitivity flag >= {min_warehouse_area_m2:g} m2")
        elif facility_area_m2 > 0:
            reasons.append(f"area sensitivity flag < {min_warehouse_area_m2:g} m2")
        score = 2 + 2 * bool(explicit_depot) + bool(name) + bool(has_address)
        return "B_warehouse_proxy", "; ".join(reasons), score

    return "C_industrial_proxy", "generic industrial evidence only", 0


def _function_signal(row: pd.Series) -> str:
    amenity = _text(row.get("amenity")).lower()
    if amenity == "post_depot":
        return "osm_post_depot"
    combined = " ".join(
        _text(row.get(column))
        for column in ("name", "official_name", "operator", "brand", "depot", "logistics")
    )
    match = FACILITY_PATTERN.search(combined)
    return match.group(0).lower() if match else ""


def _address(row: pd.Series) -> str:
    parts = [
        " ".join(filter(None, [_text(row.get("addr:housenumber")), _text(row.get("addr:street"))])),
        _text(row.get("addr:city")),
        _text(row.get("addr:postcode")),
    ]
    return ", ".join(part for part in parts if part)


def _geometry_priority(geometry_type: str) -> int:
    return {"Polygon": 4, "MultiPolygon": 4, "Point": 3, "LineString": 2}.get(
        geometry_type, 1
    )


def _extract_osm_candidates(
    *, osmium: str, pbf_path: Path, boundary_path: Path, working_dir: Path
) -> gpd.GeoDataFrame:
    city_pbf = working_dir / "city.osm.pbf"
    candidates_pbf = working_dir / "depot-candidates.osm.pbf"
    candidates_geojson = working_dir / "depot-candidates.geojson"
    _run(
        [
            osmium,
            "extract",
            "--polygon",
            str(boundary_path),
            "--strategy",
            "complete_ways",
            "--overwrite",
            "--output",
            str(city_pbf),
            str(pbf_path),
        ]
    )
    _run(
        [
            osmium,
            "tags-filter",
            "--overwrite",
            "--output",
            str(candidates_pbf),
            str(city_pbf),
            *OSMIUM_FILTERS,
        ]
    )
    _run(
        [
            osmium,
            "export",
            "--overwrite",
            "--add-unique-id",
            "type_id",
            "--output-format",
            "geojson",
            "--output",
            str(candidates_geojson),
            str(candidates_pbf),
        ]
    )
    raw = gpd.read_file(candidates_geojson).to_crs(4326)
    raw = raw.loc[raw.apply(_is_matched_object, axis=1)].copy()
    identities = raw["id"].map(_source_osm_identity)
    raw["source_osm_id"] = identities.map(lambda pair: pair[0])
    raw["source_osm_url"] = identities.map(lambda pair: pair[1])
    raw["geometry_priority"] = raw.geometry.geom_type.map(_geometry_priority)
    raw = raw.sort_values("geometry_priority", ascending=False)
    raw = raw.drop_duplicates("source_osm_id", keep="first")
    return raw


def _classify_candidates(
    raw: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    min_warehouse_area_m2: float,
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    local_crs = boundary.estimate_utm_crs()
    projected = raw.to_crs(local_crs)
    raw["facility_area_m2"] = projected.geometry.area.round(1)
    tier_values = raw.apply(
        lambda row: _evidence_tier(
            row,
            float(row["facility_area_m2"]),
            min_warehouse_area_m2,
        ),
        axis=1,
    )
    raw["evidence_tier"] = tier_values.map(lambda value: value[0])
    raw["evidence_reason"] = tier_values.map(lambda value: value[1])
    raw["evidence_score"] = tier_values.map(lambda value: value[2])
    raw["carrier"] = raw.apply(_carrier, axis=1)
    raw["facility_name"] = raw.apply(
        lambda row: _text(row.get("name"))
        or _text(row.get("official_name"))
        or (_carrier(row) + " depot" if _carrier(row) else "Unnamed warehouse"),
        axis=1,
    )
    raw["address"] = raw.apply(_address, axis=1)
    raw["osm_tags_json"] = raw.apply(
        lambda row: json.dumps(_candidate_tags(row), ensure_ascii=False, sort_keys=True), axis=1
    )
    raw["facility_geometry_type"] = raw.geometry.geom_type
    raw["facility_area_known"] = raw["facility_geometry_type"].isin(
        {"Polygon", "MultiPolygon"}
    )
    raw["function_signal"] = raw.apply(_function_signal, axis=1)
    raw["strict_candidate_class"] = "warehouse_proxy"
    raw.loc[
        (raw["evidence_tier"] == "A_osm_explicit") & raw["carrier"].ne(""),
        "strict_candidate_class",
    ] = "carrier_facility_signal"
    raw.loc[
        (raw["evidence_tier"] == "A_osm_explicit") & raw["function_signal"].ne(""),
        "strict_candidate_class",
    ] = "dispatch_function_signal"
    raw["verification_status"] = "unverified_osm_candidate"
    raw["depot_release_eligible"] = False
    boundary_geometry = boundary.geometry.union_all()
    display_points = raw.geometry.representative_point()
    raw = raw.loc[display_points.covered_by(boundary_geometry)].copy()
    tier_counts = {
        tier: int(count) for tier, count in raw["evidence_tier"].value_counts().items()
    }
    retained = raw.loc[raw["evidence_tier"].isin({"A_osm_explicit", "B_warehouse_proxy"})]
    return gpd.GeoDataFrame(retained, geometry="geometry", crs=4326), tier_counts


def _anchor_candidates(
    candidates: gpd.GeoDataFrame,
    graph_path: Path,
    max_snap_distance_m: float,
    city_slug: str,
) -> tuple[gpd.GeoDataFrame, Any, dict[str, Any]]:
    graph = ox.load_graphml(graph_path)
    strong_components = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)
    node_to_scc: dict[Any, tuple[str, int]] = {}
    for index, component in enumerate(strong_components, start=1):
        scc_id = f"S{index:04d}"
        for node in component:
            node_to_scc[node] = (scc_id, len(component))

    nodes = ox.graph_to_gdfs(graph, edges=False).reset_index()
    node_id_column = "osmid" if "osmid" in nodes.columns else nodes.columns[0]
    nodes = nodes.rename(columns={node_id_column: "road_anchor_node"})
    nodes["anchor_inside_city"] = nodes["road_anchor_node"].map(
        lambda node: _bool(
            graph.nodes[node].get(
                "inside_city", graph.nodes[node].get("inside_service_boundary", True)
            )
        )
    )
    nodes["anchor_transit_only"] = nodes["road_anchor_node"].map(
        lambda node: _bool(graph.nodes[node].get("transit_only", False))
    )
    nodes["anchor_scc_id"] = nodes["road_anchor_node"].map(
        lambda node: node_to_scc[node][0]
    )
    eligible_nodes = nodes.loc[
        nodes["anchor_inside_city"]
        & ~nodes["anchor_transit_only"]
        & nodes["anchor_scc_id"].eq("S0001")
    ].copy()
    if eligible_nodes.empty:
        raise RuntimeError("Operational graph has no city-internal nodes in its largest SCC")
    local_crs = candidates.estimate_utm_crs()
    candidates_local = candidates.to_crs(local_crs)
    nodes_local = eligible_nodes.to_crs(local_crs)[["road_anchor_node", "geometry"]]
    joined = gpd.sjoin_nearest(
        candidates_local,
        nodes_local,
        how="left",
        distance_col="road_snap_distance_m",
    )
    joined["_candidate_index"] = joined.index
    joined = (
        joined.sort_values(["road_snap_distance_m", "road_anchor_node"])
        .drop_duplicates("_candidate_index", keep="first")
        .set_index("_candidate_index")
    )
    if not joined.index.is_unique or len(joined) != len(candidates):
        raise RuntimeError("Road anchoring did not preserve one row per facility candidate")
    anchored = joined.to_crs(4326)

    anchor_lons: list[float] = []
    anchor_lats: list[float] = []
    inside_values: list[bool] = []
    transit_values: list[bool] = []
    scc_ids: list[str] = []
    scc_shares: list[float] = []
    eligible_values: list[bool] = []
    quarantine_reasons: list[str] = []
    for _, row in anchored.iterrows():
        node = row["road_anchor_node"]
        attrs = graph.nodes[node]
        anchor_lons.append(round(float(attrs["x"]), 7))
        anchor_lats.append(round(float(attrs["y"]), 7))
        inside = _bool(attrs.get("inside_city", attrs.get("inside_service_boundary", True)))
        transit = _bool(attrs.get("transit_only", False))
        scc_id, scc_size = node_to_scc[node]
        in_largest_scc = scc_id == "S0001"
        distance_qa_flag = float(row["road_snap_distance_m"]) > max_snap_distance_m
        eligible = inside and not transit and in_largest_scc
        reasons = []
        if not inside or transit:
            reasons.append("anchor is outside the service-eligible city graph")
        if not in_largest_scc:
            reasons.append("anchor is outside the largest directed SCC")
        if distance_qa_flag:
            reasons.append(
                f"road snap exceeds {max_snap_distance_m:g} m QA reference; retained"
            )
        inside_values.append(inside)
        transit_values.append(transit)
        scc_ids.append(scc_id)
        scc_shares.append(round(scc_size / graph.number_of_nodes(), 6))
        eligible_values.append(eligible)
        quarantine_reasons.append("; ".join(reasons))

    anchored["road_anchor_longitude"] = anchor_lons
    anchored["road_anchor_latitude"] = anchor_lats
    anchored["anchor_inside_city"] = inside_values
    anchored["anchor_transit_only"] = transit_values
    anchored["anchor_scc_id"] = scc_ids
    anchored["anchor_scc_node_share"] = scc_shares
    anchored["operational_eligible"] = eligible_values
    anchored["quarantine_reason"] = quarantine_reasons
    anchored["manual_verification_required"] = True
    anchored["geometry"] = anchored.geometry.representative_point()
    anchored["longitude"] = anchored.geometry.x.round(7)
    anchored["latitude"] = anchored.geometry.y.round(7)
    anchored["road_snap_distance_m"] = anchored["road_snap_distance_m"].round(1)
    anchored["road_snap_distance_qa_flag"] = (
        anchored["road_snap_distance_m"] > max_snap_distance_m
    )
    anchored["road_anchor_strategy"] = (
        "nearest_city_internal_node_in_largest_directed_scc"
    )
    anchored["city_slug"] = city_slug
    tier_a = anchored["evidence_tier"].eq("A_osm_explicit")
    tier_b = anchored["evidence_tier"].eq("B_warehouse_proxy")
    anchored["benchmark_depot_class"] = np.where(
        tier_a,
        "strict_osm_candidate",
        "optional_warehouse_proxy",
    )
    anchored["strict_candidate_eligible"] = tier_a & anchored["operational_eligible"]
    anchored["optional_candidate_eligible"] = tier_b & anchored["operational_eligible"]
    anchored["cle_candidate_eligible"] = (
        anchored["strict_candidate_eligible"]
        | anchored["optional_candidate_eligible"]
    )
    anchored["last_mile_function_verified"] = False
    anchored["depot_evidence_class"] = "optional_warehouse_proxy_quarantined"
    anchored.loc[
        anchored["optional_candidate_eligible"], "depot_evidence_class"
    ] = "optional_warehouse_proxy"
    anchored.loc[tier_a, "depot_evidence_class"] = "strict_osm_candidate_quarantined"
    anchored.loc[
        anchored["strict_candidate_eligible"], "depot_evidence_class"
    ] = "strict_osm_candidate"

    anchored["tier_order"] = anchored["evidence_tier"].map(
        {"A_osm_explicit": 0, "B_warehouse_proxy": 1}
    )
    anchored = anchored.sort_values(
        [
            "tier_order",
            "operational_eligible",
            "evidence_score",
            "facility_area_m2",
            "road_snap_distance_m",
        ],
        ascending=[True, False, False, False, True],
    )
    anchored["candidate_rank"] = range(1, len(anchored) + 1)
    anchored["candidate_id"] = anchored.apply(
        lambda row: f"{city_slug}-depot-{int(row['candidate_rank']):04d}", axis=1
    )

    graph_summary = {
        "graph_path": str(graph_path),
        "node_count": graph.number_of_nodes(),
        "directed_edge_count": graph.number_of_edges(),
        "strong_component_count": len(strong_components),
        "largest_strong_component_nodes": len(strong_components[0]),
        "largest_strong_component_node_share": len(strong_components[0])
        / graph.number_of_nodes(),
        "service_eligible_anchor_node_count": len(eligible_nodes),
        "anchor_strategy": "nearest city-internal non-transit node in largest directed SCC",
        "max_road_snap_distance_m": max_snap_distance_m,
    }
    return gpd.GeoDataFrame(anchored, geometry="geometry", crs=4326), graph, graph_summary


def _normalize_highways(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {_text(item) for item in value}
    text = _text(value)
    for char in "[]'\"":
        text = text.replace(char, "")
    return {item.strip() for item in text.split(",") if item.strip()}


def _iter_lines(geometry: Any) -> Iterable[LineString]:
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms


def _road_context(graph: Any, land_geometry: Any, max_segments: int = 650) -> list[list[list[float]]]:
    edges = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True).reset_index()
    edges = edges.loc[edges["highway"].map(lambda value: bool(_normalize_highways(value) & ROAD_CLASSES))]
    edges = edges.sort_values("length", ascending=False)
    roads: list[list[list[float]]] = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    for geometry in edges.geometry:
        clipped = geometry.intersection(land_geometry)
        for line in _iter_lines(clipped):
            simplified = line.simplify(0.0007, preserve_topology=False)
            coords = [[round(x, 5), round(y, 5)] for x, y in simplified.coords]
            if len(coords) < 2:
                continue
            forward = tuple((point[0], point[1]) for point in coords)
            key = min(forward, tuple(reversed(forward)))
            if key in seen:
                continue
            seen.add(key)
            roads.append(coords)
            if len(roads) >= max_segments:
                return roads
    return roads


def _visual_payload(
    candidates: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    graph: Any,
    max_tier_b: int,
) -> dict[str, Any]:
    tier_a = candidates.loc[candidates["evidence_tier"] == "A_osm_explicit"]
    tier_b = candidates.loc[
        (candidates["evidence_tier"] == "B_warehouse_proxy")
        & candidates["operational_eligible"]
    ].head(max_tier_b)
    display = pd.concat([tier_a, tier_b]).sort_values("candidate_rank")
    points = []
    for _, row in display.iterrows():
        points.append(
            {
                "id": row["candidate_id"],
                "name": row["facility_name"],
                "carrier": row["carrier"],
                "tier": row["evidence_tier"],
                "eligible": bool(row["operational_eligible"]),
                "reason": row["evidence_reason"],
                "quarantine": row["quarantine_reason"],
                "lon": round(float(row["longitude"]), 6),
                "lat": round(float(row["latitude"]), 6),
                "anchor_lon": round(float(row["road_anchor_longitude"]), 6),
                "anchor_lat": round(float(row["road_anchor_latitude"]), 6),
                "snap_m": round(float(row["road_snap_distance_m"]), 1),
                "area_m2": round(float(row["facility_area_m2"]), 1),
                "address": row["address"],
                "url": row["source_osm_url"],
            }
        )
    land_geometry = boundary.geometry.union_all()
    return {
        "summary": {
            "tier_a": len(tier_a),
            "tier_a_eligible": int(tier_a["operational_eligible"].sum()),
            "tier_b_total": int(
                (candidates["evidence_tier"] == "B_warehouse_proxy").sum()
            ),
            "tier_b_shown": len(tier_b),
        },
        "boundary": mapping(land_geometry.simplify(0.0007, preserve_topology=True)),
        "roads": _road_context(graph, land_geometry),
        "candidates": points,
    }


def _html_fragment(
    payload: dict[str, Any], vendor_dir: Path, city_label: str
) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    d3_array = (vendor_dir / "d3-array-3.2.4.min.js").read_text(encoding="utf-8")
    d3_geo = (vendor_dir / "d3-geo-3.1.1.min.js").read_text(encoding="utf-8")
    template = '''<div id="depot-candidate-audit">
  <div class="viz-grid depot-stats" aria-label="Depot candidate summary">
    <div class="card viz-stat"><div class="text-small">Tier A explicit</div><div class="viz-stat-value" id="depot-a-count"></div></div>
    <div class="card viz-stat"><div class="text-small">Tier A road-eligible</div><div class="viz-stat-value" id="depot-a-eligible"></div></div>
    <div class="card viz-stat"><div class="text-small">Tier B warehouse proxies</div><div class="viz-stat-value" id="depot-b-count"></div></div>
  </div>
  <div class="viz-row text-small depot-legend" aria-label="Depot candidate legend">
    <span><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="4" fill="var(--viz-series-1, #0b6e99)"></circle></svg>Tier A: explicit carrier/logistics evidence</span>
    <span><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="4" fill="var(--viz-series-2, #e07a2d)"></circle></svg>Tier B: road-eligible optional pool shown (<span id="depot-b-shown"></span>)</span>
    <span><svg width="14" height="14" aria-hidden="true"><path d="M3 3L11 11M11 3L3 11" stroke="var(--destructive, #dc2626)" stroke-width="2"></path></svg>Quarantined</span>
  </div>
  <svg class="depot-map" viewBox="0 0 736 455" role="img" aria-label="__CITY_LABEL__ depot candidates and directed-road anchors"></svg>
  <div class="card depot-detail text-small" aria-live="polite"></div>
</div>
<style>
  #depot-candidate-audit { position: relative; width: 100%; color: var(--foreground, #1f2937); }
  #depot-candidate-audit .depot-stats { margin-bottom: 10px; }
  #depot-candidate-audit .depot-legend { justify-content: flex-start; gap: 14px; margin-bottom: 6px; }
  #depot-candidate-audit .depot-legend > span { display: inline-flex; align-items: center; gap: 4px; }
  #depot-candidate-audit .depot-map { display: block; width: 100%; height: auto; overflow: visible; }
  #depot-candidate-audit .land { fill: color-mix(in srgb, var(--muted, #d9e2e8) 42%, transparent); stroke: var(--border, #94a3b8); stroke-width: 1.1; }
  #depot-candidate-audit .road { fill: none; stroke: var(--muted-foreground, #64748b); stroke-opacity: .34; stroke-width: .65; vector-effect: non-scaling-stroke; }
  #depot-candidate-audit .snap { stroke: var(--muted-foreground, #64748b); stroke-opacity: .52; stroke-width: .7; vector-effect: non-scaling-stroke; }
  #depot-candidate-audit .candidate { cursor: pointer; stroke: var(--background, #ffffff); stroke-width: 1.1; vector-effect: non-scaling-stroke; }
  #depot-candidate-audit .quarantine { fill: none; stroke: var(--destructive, #dc2626); stroke-width: 2.1; vector-effect: non-scaling-stroke; pointer-events: none; }
  #depot-candidate-audit .depot-detail { margin-top: 8px; min-height: 40px; }
</style>
<script>__D3_ARRAY__
__D3_GEO__</script>
<script>
(() => {
  const root = document.getElementById("depot-candidate-audit");
  const payload = __DATA__;
  root.querySelector("#depot-a-count").textContent = payload.summary.tier_a;
  root.querySelector("#depot-a-eligible").textContent = payload.summary.tier_a_eligible;
  root.querySelector("#depot-b-count").textContent = payload.summary.tier_b_total;
  root.querySelector("#depot-b-shown").textContent = payload.summary.tier_b_shown;
  const svg = root.querySelector(".depot-map");
  const detail = root.querySelector(".depot-detail");
  const ns = "http://www.w3.org/2000/svg";
  const boundaryFeature = {type: "Feature", properties: {}, geometry: payload.boundary};
  const projection = d3.geoMercator().fitExtent([[10, 10], [726, 445]], boundaryFeature);
  const path = d3.geoPath(projection);
  function element(tag, className, parent = svg) {
    const node = document.createElementNS(ns, tag);
    if (className) node.setAttribute("class", className);
    parent.appendChild(node);
    return node;
  }
  const land = element("path", "land");
  land.setAttribute("d", path(boundaryFeature));
  const roads = element("g", "roads");
  payload.roads.forEach(coords => {
    const road = element("path", "road", roads);
    road.setAttribute("d", path({type: "LineString", coordinates: coords}));
  });
  const snaps = element("g", "snaps");
  payload.candidates.forEach(candidate => {
    const source = projection([candidate.lon, candidate.lat]);
    const anchor = projection([candidate.anchor_lon, candidate.anchor_lat]);
    const line = element("line", "snap", snaps);
    line.setAttribute("x1", source[0]);
    line.setAttribute("y1", source[1]);
    line.setAttribute("x2", anchor[0]);
    line.setAttribute("y2", anchor[1]);
  });
  const marks = element("g", "candidates");
  function show(candidate) {
    detail.replaceChildren();
    const strong = document.createElement("strong");
    strong.textContent = candidate.name;
    detail.appendChild(strong);
    detail.appendChild(document.createTextNode(
      ` · ${candidate.tier === "A_osm_explicit" ? "Tier A" : "Tier B"} · ${candidate.eligible ? "road-eligible" : "quarantined"} · snap ${candidate.snap_m.toFixed(1)} m`
    ));
    detail.appendChild(document.createElement("br"));
    detail.appendChild(document.createTextNode(candidate.reason));
    if (candidate.address) detail.appendChild(document.createTextNode(` · ${candidate.address}`));
    if (candidate.quarantine) detail.appendChild(document.createTextNode(` · ${candidate.quarantine}`));
    detail.appendChild(document.createTextNode(" · "));
    const link = document.createElement("a");
    link.href = candidate.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = "OSM evidence";
    detail.appendChild(link);
  }
  payload.candidates.forEach((candidate, index) => {
    const point = projection([candidate.lon, candidate.lat]);
    const mark = element("circle", "candidate", marks);
    mark.setAttribute("cx", point[0]);
    mark.setAttribute("cy", point[1]);
    mark.setAttribute("r", candidate.tier === "A_osm_explicit" ? 5.1 : 3.5);
    mark.setAttribute("fill", candidate.tier === "A_osm_explicit" ? "var(--viz-series-1, #0b6e99)" : "var(--viz-series-2, #e07a2d)");
    mark.setAttribute("aria-label", `${candidate.name}; ${candidate.tier}; ${candidate.eligible ? "road eligible" : "quarantined"}`);
    mark.addEventListener("click", () => show(candidate));
    if (!candidate.eligible) {
      const x = element("path", "quarantine", marks);
      x.setAttribute("d", `M${point[0]-5},${point[1]-5}L${point[0]+5},${point[1]+5}M${point[0]+5},${point[1]-5}L${point[0]-5},${point[1]+5}`);
    }
    if (index === 0) show(candidate);
  });
})();
</script>
'''
    return (
        template.replace("__D3_ARRAY__", d3_array)
        .replace("__D3_GEO__", d3_geo)
        .replace("__DATA__", encoded)
        .replace("__CITY_LABEL__", html.escape(city_label, quote=True))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--city-slug", default="los-angeles")
    parser.add_argument("--city-label")
    parser.add_argument(
        "--boundary-root", type=Path, default=Path("boundaries/us-11city-2025")
    )
    parser.add_argument("--city-root", type=Path, default=Path("data/cities"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit output directory; otherwise analysis/depot_preview/<snapshot-date>",
    )
    parser.add_argument(
        "--snapshot-date", default=datetime.now(UTC).date().isoformat()
    )
    parser.add_argument("--max-road-snap-m", type=float, default=250.0)
    parser.add_argument(
        "--min-warehouse-area-m2",
        type=float,
        default=1_000.0,
        help=(
            "Minimum OSM polygon area for an unnamed warehouse proxy. Explicit "
            "accepted depot/logistics tags are not area-gated."
        ),
    )
    parser.add_argument("--max-tier-b-on-map", type=int, default=5_000)
    parser.add_argument("--visualization-html", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    city_slug = args.city_slug
    city_label = args.city_label or city_slug.replace("-", " ").title()
    boundary_root = (
        args.boundary_root
        if args.boundary_root.is_absolute()
        else repo_root / args.boundary_root
    )
    boundary_path = boundary_root / city_slug / "land_boundary.geojson"
    city_root = args.city_root if args.city_root.is_absolute() else repo_root / args.city_root
    graph_path = city_root / city_slug / "graph_operational.graphml"
    city_manifest_path = city_root / city_slug / "manifest.json"
    city_manifest = json.loads(city_manifest_path.read_text(encoding="utf-8"))
    pbf_path = Path(city_manifest["provenance"]["osm"]["pbf_file"])
    if not pbf_path.is_absolute():
        pbf_path = repo_root / pbf_path
    output_dir = (
        args.output_dir
        if args.output_dir and args.output_dir.is_absolute()
        else repo_root / args.output_dir
        if args.output_dir
        else repo_root / "analysis/depot_preview" / args.snapshot_date
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    osmium = shutil.which("osmium")
    if osmium is None:
        raise RuntimeError("osmium-tool is required on PATH")
    boundary = gpd.read_file(boundary_path).to_crs(4326)
    with tempfile.TemporaryDirectory(prefix="evrptw-depot-") as temp_dir:
        raw = _extract_osm_candidates(
            osmium=osmium,
            pbf_path=pbf_path,
            boundary_path=boundary_path,
            working_dir=Path(temp_dir),
        )
    candidates, tier_counts = _classify_candidates(
        raw,
        boundary,
        args.min_warehouse_area_m2,
    )
    anchored, graph, graph_summary = _anchor_candidates(
        candidates, graph_path, args.max_road_snap_m, city_slug
    )

    # Recheck the final point representation after the projected road-anchoring
    # round trip.  For polygons touching a complex land boundary, recomputing a
    # representative point after CRS transforms can place it just outside even
    # though the pre-anchor representative point passed the boundary filter.
    boundary_geometry = boundary.geometry.union_all()
    anchored = anchored.loc[anchored.geometry.covered_by(boundary_geometry)].copy()

    point_output = anchored.copy()
    point_output[OUTPUT_COLUMNS].to_csv(
        output_dir / f"{city_slug}_depot_candidates.csv", index=False
    )
    point_output[OUTPUT_COLUMNS + ["geometry"]].to_file(
        output_dir / f"{city_slug}_depot_candidates.geojson", driver="GeoJSON"
    )
    summary = {
        "schema": "evrptw_depot_candidate_audit_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "city_slug": city_slug,
        "city_label": city_label,
        "source": {
            "dataset": "OpenStreetMap via frozen Geofabrik PBF",
            "pbf_path": str(pbf_path),
            "pbf_source_url": city_manifest["provenance"]["osm"]["pbf_source_url"],
            "pbf_replication_timestamp_utc": city_manifest["provenance"]["osm"][
                "pbf_replication_timestamp_utc"
            ],
            "pbf_sha256": city_manifest["provenance"]["osm"]["pbf_sha256"],
            "boundary_path": str(boundary_path),
            "boundary_sha256": _sha256(boundary_path),
            "osmium_filters": OSMIUM_FILTERS,
        },
        "tier_semantics": {
            "A_osm_explicit": "known parcel carrier plus physical warehouse, dispatch-function name, or explicit depot/logistics evidence; retail counters are excluded; still requires manual verification",
            "B_warehouse_proxy": "warehouse/logistics facility proxy without strict carrier-dispatch confirmation; area is retained as a continuous sensitivity attribute",
            "C_industrial_proxy": "generic industrial evidence, carrier retail counter, or carrier-branded point without physical dispatch-facility evidence; excluded from the official candidate layer",
        },
        "classification_policy": {
            "reference_warehouse_area_flag_m2": args.min_warehouse_area_m2,
            "warehouse_area_policy": "continuous attribute; no hard exclusion",
            "retail_counter_exclusion_pattern": RETAIL_COUNTER_PATTERN.pattern,
            "allowed_logistics_values": sorted(ALLOWED_LOGISTICS_VALUES),
            "allowed_depot_values": sorted(ALLOWED_DEPOT_VALUES),
            "cle_candidate_policy": (
                "road-eligible Tier A strict OSM candidates plus road-eligible "
                "Tier B optional warehouse proxies"
            ),
        },
        "raw_tier_counts": tier_counts,
        "retained_candidate_count": len(anchored),
        "tier_a_count": int((anchored["evidence_tier"] == "A_osm_explicit").sum()),
        "tier_b_count": int((anchored["evidence_tier"] == "B_warehouse_proxy").sum()),
        "operational_eligible_count": int(anchored["operational_eligible"].sum()),
        "tier_a_operational_eligible_count": int(
            (
                (anchored["evidence_tier"] == "A_osm_explicit")
                & anchored["operational_eligible"]
            ).sum()
        ),
        "strict_candidate_eligible_count": int(
            anchored["strict_candidate_eligible"].sum()
        ),
        "optional_candidate_eligible_count": int(
            anchored["optional_candidate_eligible"].sum()
        ),
        "dispatch_function_signal_count": int(
            (anchored["strict_candidate_class"] == "dispatch_function_signal").sum()
        ),
        "carrier_facility_signal_count": int(
            (anchored["strict_candidate_class"] == "carrier_facility_signal").sum()
        ),
        "cle_candidate_eligible_count": int(
            anchored["cle_candidate_eligible"].sum()
        ),
        "depot_release_eligible_count": int(anchored["depot_release_eligible"].sum()),
        "manual_verification_required_for_every_retained_candidate": True,
        "graph_gate": {
            **graph_summary,
            "graph_sha256": _sha256(graph_path),
        },
        "limitations": [
            "OSM coverage and facility tagging are incomplete and spatially heterogeneous.",
            "A named carrier warehouse is a depot candidate, not proof of present-day last-mile operations.",
            "Tier B candidates are land-use proxies and cannot be called parcel depots.",
            "Road anchoring tests directed graph access but not private driveway, gate, or truck-access rules.",
        ],
    }
    (output_dir / f"{city_slug}_depot_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    payload = _visual_payload(anchored, boundary, graph, args.max_tier_b_on_map)
    visual_path = args.visualization_html or (
        output_dir / f"{city_slug}_depot_candidate_audit.html"
    )
    visual_path.parent.mkdir(parents=True, exist_ok=True)
    visual_path.write_text(
        _html_fragment(
            payload, Path(__file__).resolve().parent / "vendor", city_label
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Visualization: {visual_path}")


if __name__ == "__main__":
    main()
