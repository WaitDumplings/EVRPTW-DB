#!/usr/bin/env python3
"""Build an audit-only AFDC charging-station preview for the top-ten cities.

This script intentionally stops before road anchoring or vehicle-connector
eligibility. It answers the narrower first question: where do currently
available, public AFDC electric stations fall inside each exact city land
boundary?
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import io
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.errors import GEOSException
from shapely.geometry import LineString, MultiLineString, mapping

API_ROOT = "https://developer.nlr.gov/api/alt-fuel-stations/v1"
ROAD_CLASSES = {"motorway", "trunk", "primary", "secondary"}
DISPLAY_NAMES = {
    "new-york": "New York City",
    "los-angeles": "Los Angeles",
    "chicago": "Chicago",
    "houston": "Houston",
    "dallas": "Dallas",
    "washington-dc": "Washington, DC",
    "boston": "Boston",
    "miami": "Miami",
    "san-francisco": "San Francisco",
    "seattle": "Seattle",
}


def _haversine_miles(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius_miles = 3958.7613
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius_miles * math.asin(math.sqrt(a))


def _query_geometry(boundary: gpd.GeoDataFrame) -> tuple[float, float, float]:
    local_crs = boundary.estimate_utm_crs()
    projected = boundary.to_crs(local_crs).geometry.union_all()
    center_projected = projected.centroid
    center = gpd.GeoSeries([center_projected], crs=local_crs).to_crs(4326).iloc[0]
    min_lon, min_lat, max_lon, max_lat = boundary.geometry.union_all().bounds
    radius = max(
        _haversine_miles(center.x, center.y, lon, lat)
        for lon in (min_lon, max_lon)
        for lat in (min_lat, max_lat)
    )
    return center.y, center.x, math.ceil(radius + 2.0)


def _download(url: str, *, attempts: int = 4) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "evrptw-cle/afdc-preview"},
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def _afdc_csv_url(api_key: str, latitude: float, longitude: float, radius: float) -> str:
    params = {
        "api_key": api_key,
        "fuel_type": "ELEC",
        "access": "public",
        "status": "E",
        "country": "US",
        "latitude": f"{latitude:.7f}",
        "longitude": f"{longitude:.7f}",
        "radius": f"{radius:.1f}",
        "limit": "all",
    }
    return f"{API_ROOT}/nearest.csv?{urllib.parse.urlencode(params)}"


def _redacted_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    safe_query = [(key, "REDACTED") if key == "api_key" else (key, value) for key, value in query]
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(safe_query), parsed.fragment)
    )


def _number(value: Any) -> int:
    try:
        if value is None or pd.isna(value) or str(value).strip() == "":
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _bool_or_none(value: Any) -> bool | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _connector_set(value: Any) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {token.strip() for token in str(value).replace(",", " ").split() if token.strip()}


def _station_category(row: pd.Series) -> str:
    if _number(row.get("EV DC Fast Count")) > 0:
        return "dc_fast"
    if _number(row.get("EV Level2 EVSE Num")) > 0:
        return "level2_only"
    return "other_or_unknown"


def _normalize_highways(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    text = str(value or "")
    for char in "[]'\"":
        text = text.replace(char, "")
    return {item.strip() for item in text.split(",") if item.strip()}


def _iter_lines(geometry: Any) -> Iterable[LineString]:
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms


def _road_context(graph_path: Path, land_geometry: Any, max_segments: int = 500) -> list[list[list[float]]]:
    """Stream the GraphML and retain only the longest display roads.

    Loading each 15--112 MB graph through OSMnx is unnecessary for this audit
    graphic. GraphML writes nodes before edges, so a streaming parser can keep
    node coordinates plus a small heap of major-road candidates.
    """
    key_names: dict[str, str] = {}
    node_xy: dict[str, tuple[float, float]] = {}
    longest: list[tuple[float, int, str, str, str | None]] = []
    sequence = 0
    candidate_limit = max_segments * 4
    for _, element in ET.iterparse(graph_path, events=("end",)):
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "key":
            key_names[element.attrib["id"]] = element.attrib.get("attr.name", "")
        elif tag == "node":
            values = {
                key_names.get(child.attrib.get("key", ""), ""): child.text
                for child in element
                if child.tag.rsplit("}", 1)[-1] == "data"
            }
            try:
                node_xy[element.attrib["id"]] = (float(values["x"]), float(values["y"]))
            except (KeyError, TypeError, ValueError):
                pass
        elif tag == "edge":
            values = {
                key_names.get(child.attrib.get("key", ""), ""): child.text
                for child in element
                if child.tag.rsplit("}", 1)[-1] == "data"
            }
            if _normalize_highways(values.get("highway")) & ROAD_CLASSES:
                try:
                    length = float(values.get("length") or 0.0)
                except (TypeError, ValueError):
                    length = 0.0
                item = (
                    length,
                    sequence,
                    element.attrib["source"],
                    element.attrib["target"],
                    values.get("geometry"),
                )
                sequence += 1
                if len(longest) < candidate_limit:
                    heapq.heappush(longest, item)
                elif item[0] > longest[0][0]:
                    heapq.heapreplace(longest, item)
        if tag in {"key", "node", "edge"}:
            element.clear()

    candidates: list[tuple[float, list[list[float]]]] = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    for length, _, u, v, geometry_text in sorted(longest, reverse=True):
        geometry = None
        if geometry_text:
            try:
                geometry = wkt.loads(geometry_text)
            except (GEOSException, ValueError, TypeError):
                geometry = None
        if geometry is None:
            if u not in node_xy or v not in node_xy:
                continue
            geometry = LineString([node_xy[u], node_xy[v]])
        clipped = geometry.intersection(land_geometry)
        for line in _iter_lines(clipped):
            if line.is_empty:
                continue
            simplified = line.simplify(0.0008, preserve_topology=False)
            coords = [[round(x, 5), round(y, 5)] for x, y in simplified.coords]
            if len(coords) < 2:
                continue
            forward = tuple((point[0], point[1]) for point in coords)
            reverse = tuple(reversed(forward))
            key = min(forward, reverse)
            if key in seen:
                continue
            seen.add(key)
            candidates.append((length or line.length, coords))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [coords for _, coords in candidates[:max_segments]]


def _safe_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _build_city_record(
    *,
    slug: str,
    boundary_path: Path,
    graph_path: Path,
    raw_csv_path: Path,
    api_key: str,
    refresh: bool,
    national_source: tuple[pd.DataFrame, Path, str] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    boundary = gpd.read_file(boundary_path).to_crs(4326)
    land_geometry = boundary.geometry.union_all()
    latitude, longitude, radius = _query_geometry(boundary)
    url = _afdc_csv_url(api_key, latitude, longitude, radius)

    if national_source is not None:
        national_frame, national_path, national_sha256 = national_source
        min_lon, min_lat, max_lon, max_lat = land_geometry.bounds
        raw = national_frame.loc[
            national_frame["Longitude"].between(min_lon, max_lon)
            & national_frame["Latitude"].between(min_lat, max_lat)
        ].copy()
        payload = None
    elif refresh or not raw_csv_path.exists():
        payload = _download(url)
        raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
        raw_csv_path.write_bytes(payload)
        raw = pd.read_csv(io.BytesIO(payload), low_memory=False)
    else:
        payload = raw_csv_path.read_bytes()
        raw = pd.read_csv(io.BytesIO(payload), low_memory=False)
    points = gpd.GeoSeries(
        gpd.points_from_xy(raw["Longitude"], raw["Latitude"]), crs=4326, index=raw.index
    )
    inside = points.covered_by(land_geometry)
    selected = raw.loc[inside].copy()
    selected["city_slug"] = slug
    selected["charging_category"] = selected.apply(_station_category, axis=1)
    selected["restricted_public"] = selected["Restricted Access"].map(_bool_or_none)
    selected["l1_ports"] = selected["EV Level1 EVSE Num"].map(_number)
    selected["l2_ports"] = selected["EV Level2 EVSE Num"].map(_number)
    selected["dc_fast_ports"] = selected["EV DC Fast Count"].map(_number)
    selected["total_reported_ports"] = (
        selected["l1_ports"] + selected["l2_ports"] + selected["dc_fast_ports"]
    )
    selected["has_ccs"] = selected["EV Connector Types"].map(
        lambda value: "J1772COMBO" in _connector_set(value)
    )
    selected["has_nacs"] = selected["EV Connector Types"].map(
        lambda value: "TESLA" in _connector_set(value)
    )
    selected["has_j1772"] = selected["EV Connector Types"].map(
        lambda value: "J1772" in _connector_set(value)
    )

    projected = boundary.to_crs(boundary.estimate_utm_crs()).geometry.union_all()
    land_area_km2 = projected.area / 1_000_000
    categories = Counter(selected["charging_category"])
    restricted = selected["restricted_public"] == True
    unknown_restriction = selected["restricted_public"].isna()
    summary = {
        "city_slug": slug,
        "city": DISPLAY_NAMES.get(slug, slug),
        "land_area_km2": round(land_area_km2, 2),
        "public_available_sites": len(selected),
        "sites_per_100_km2": round(len(selected) / land_area_km2 * 100, 2),
        "dc_fast_sites": int(categories["dc_fast"]),
        "level2_only_sites": int(categories["level2_only"]),
        "other_or_unknown_sites": int(categories["other_or_unknown"]),
        "restricted_public_sites": int(restricted.sum()),
        "restriction_unknown_sites": int(unknown_restriction.sum()),
        "ccs_sites": int(selected["has_ccs"].sum()),
        "nacs_sites": int(selected["has_nacs"].sum()),
        "j1772_sites": int(selected["has_j1772"].sum()),
        "reported_ports": int(selected["total_reported_ports"].sum()),
        "spatial_prefilter_candidate_sites": len(raw),
    }

    boundary_display = land_geometry.simplify(0.001, preserve_topology=True)
    stations = []
    for _, row in selected.iterrows():
        stations.append(
            {
                "id": int(row["ID"]),
                "name": _safe_text(row["Station Name"]),
                "lon": round(float(row["Longitude"]), 6),
                "lat": round(float(row["Latitude"]), 6),
                "category": row["charging_category"],
                "restricted": row["restricted_public"],
                "l2": int(row["l2_ports"]),
                "dc": int(row["dc_fast_ports"]),
                "connectors": _safe_text(row["EV Connector Types"]),
            }
        )
    visual_record = {
        "slug": slug,
        "name": DISPLAY_NAMES.get(slug, slug),
        "summary": summary,
        "boundary": mapping(boundary_display),
        "roads": _road_context(graph_path, land_geometry),
        "stations": stations,
    }
    if national_source is not None:
        source_record = {
            "city_slug": slug,
            "source_mode": "national_csv_snapshot",
            "raw_file": str(national_path),
            "raw_sha256": national_sha256,
            "bbox_candidate_sites": len(raw),
            "exact_city_land_sites": len(selected),
        }
    else:
        source_record = {
            "city_slug": slug,
            "source_mode": "nearest_api_query",
            "raw_file": str(raw_csv_path),
            "raw_sha256": hashlib.sha256(payload).hexdigest(),
            "query_url_redacted": _redacted_url(url),
            "query_center": {"latitude": latitude, "longitude": longitude},
            "query_radius_miles": radius,
            "query_candidate_sites": len(raw),
            "exact_city_land_sites": len(selected),
        }
    return visual_record, selected, source_record


def _html_fragment(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    vendor_dir = Path(__file__).resolve().parent / "vendor"
    d3_array = (vendor_dir / "d3-array-3.2.4.min.js").read_text(encoding="utf-8")
    d3_geo = (vendor_dir / "d3-geo-3.1.1.min.js").read_text(encoding="utf-8")
    template = '''<div id="afdc-top10-distribution">
  <div class="viz-row text-small" aria-label="Charging station legend">
    <span><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="4" fill="var(--viz-series-1)"></circle></svg> DC fast site</span>
    <span><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="3" fill="var(--viz-series-2)"></circle></svg> Level 2 only</span>
    <span><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="3" fill="var(--viz-series-3)"></circle></svg> Other / unknown</span>
    <span><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="4" fill="none" stroke="var(--foreground)" stroke-width="1.5"></circle></svg> Outer ring = restricted public access</span>
  </div>
  <div class="city-grid"></div>
  <div class="tooltip text-small" role="status" aria-live="polite"></div>
</div>
<style>
  #afdc-top10-distribution { position: relative; width: 100%; color: var(--foreground); }
  #afdc-top10-distribution .viz-row { justify-content: flex-start; gap: 14px; margin-bottom: 12px; }
  #afdc-top10-distribution .viz-row span { display: inline-flex; align-items: center; gap: 4px; }
  #afdc-top10-distribution .city-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px 16px; }
  #afdc-top10-distribution .city-panel { min-width: 0; }
  #afdc-top10-distribution .city-heading { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; margin-bottom: 4px; }
  #afdc-top10-distribution .city-heading h3 { margin: 0; font-weight: 500; }
  #afdc-top10-distribution .city-heading span { color: var(--muted-foreground); white-space: nowrap; }
  #afdc-top10-distribution .city-map { display: block; width: 100%; height: auto; overflow: visible; }
  #afdc-top10-distribution .land { fill: color-mix(in srgb, var(--muted) 42%, transparent); stroke: var(--border); stroke-width: 1.1; }
  #afdc-top10-distribution .road { fill: none; stroke: var(--muted-foreground); stroke-opacity: .30; stroke-width: .55; vector-effect: non-scaling-stroke; }
  #afdc-top10-distribution .station { stroke: var(--background); stroke-width: .45; vector-effect: non-scaling-stroke; }
  #afdc-top10-distribution .restricted-ring { fill: none; stroke: var(--foreground); stroke-width: .85; vector-effect: non-scaling-stroke; pointer-events: none; }
  #afdc-top10-distribution .tooltip { position: absolute; visibility: hidden; pointer-events: none; max-width: 250px; }
  @media (max-width: 580px) {
    #afdc-top10-distribution .city-grid { grid-template-columns: 1fr; }
  }
</style>
<script>__D3_ARRAY__
__D3_GEO__</script>
<script>
(() => {
  const root = document.getElementById("afdc-top10-distribution");
  const payload = __DATA__;
  const grid = root.querySelector(".city-grid");
  const tooltip = root.querySelector(".tooltip");
  const svgNamespace = "http://www.w3.org/2000/svg";
  const categoryColor = {
    dc_fast: "var(--viz-series-1)",
    level2_only: "var(--viz-series-2)",
    other_or_unknown: "var(--viz-series-3)"
  };

  function htmlElement(tag, className, parent) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    parent.appendChild(node);
    return node;
  }

  function svgElement(tag, className, parent) {
    const node = document.createElementNS(svgNamespace, tag);
    if (className) node.setAttribute("class", className);
    parent.appendChild(node);
    return node;
  }

  payload.cities.forEach(city => {
    const panel = htmlElement("section", "city-panel", grid);
    const heading = htmlElement("div", "city-heading", panel);
    htmlElement("h3", "", heading).textContent = city.name;
    htmlElement("span", "text-small", heading).textContent =
      `${city.summary.public_available_sites.toLocaleString()} sites · ${city.summary.dc_fast_sites.toLocaleString()} DC fast`;
    const svg = svgElement("svg", "city-map", panel);
    svg.setAttribute("viewBox", "0 0 360 235");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", `${city.name}: ${city.summary.public_available_sites} public available AFDC charging sites inside the city land boundary`);
    svgElement("title", "", svg).textContent = `${city.name} charging station distribution`;
    svgElement("desc", "", svg).textContent =
      `AFDC public available stations: ${city.summary.dc_fast_sites} DC fast, ${city.summary.level2_only_sites} Level 2 only, and ${city.summary.other_or_unknown_sites} other or unknown.`;

    const boundaryFeature = {type: "Feature", properties: {}, geometry: city.boundary};
    const projection = d3.geoMercator().fitExtent([[8, 8], [352, 227]], boundaryFeature);
    const path = d3.geoPath(projection);
    const landPath = svgElement("path", "land", svg);
    landPath.setAttribute("d", path(boundaryFeature));
    const roadsGroup = svgElement("g", "", svg);
    city.roads.forEach(coords => {
      const roadPath = svgElement("path", "road", roadsGroup);
      roadPath.setAttribute("d", path({type: "LineString", coordinates: coords}));
    });

    const stationsGroup = svgElement("g", "", svg);
    city.stations.forEach(station => {
      const projected = projection([station.lon, station.lat]);
      const point = svgElement("circle", "station", stationsGroup);
      point.setAttribute("cx", projected[0]);
      point.setAttribute("cy", projected[1]);
      point.setAttribute("r", station.category === "dc_fast" ? 2.15 : 1.55);
      point.setAttribute("fill", categoryColor[station.category]);
      point.setAttribute("aria-label", `${station.name}; ${station.category}; ${station.dc} DC fast ports; ${station.l2} Level 2 ports`);
      point.addEventListener("mouseenter", event => show(event, station));
      point.addEventListener("mouseleave", hide);
    });

    const ringsGroup = svgElement("g", "", svg);
    city.stations.filter(station => station.restricted === true).forEach(station => {
      const projected = projection([station.lon, station.lat]);
      const ring = svgElement("circle", "restricted-ring", ringsGroup);
      ring.setAttribute("cx", projected[0]);
      ring.setAttribute("cy", projected[1]);
      ring.setAttribute("r", station.category === "dc_fast" ? 3.25 : 2.55);
    });

    function show(event, station) {
      tooltip.replaceChildren();
      const strong = document.createElement("strong");
      strong.textContent = station.name || "Unnamed station";
      tooltip.appendChild(strong);
      tooltip.appendChild(document.createElement("br"));
      tooltip.appendChild(document.createTextNode(`${station.dc} DC fast · ${station.l2} Level 2`));
      tooltip.appendChild(document.createElement("br"));
      tooltip.appendChild(document.createTextNode(station.connectors || "Connector not reported"));
      if (station.restricted === true) {
        tooltip.appendChild(document.createElement("br"));
        tooltip.appendChild(document.createTextNode("Restricted public access"));
      }
      tooltip.style.visibility = "visible";
      const rootRect = root.getBoundingClientRect();
      const markRect = event.currentTarget.getBoundingClientRect();
      const left = Math.max(4, Math.min(markRect.left - rootRect.left + markRect.width / 2 + 8, rootRect.width - tooltip.offsetWidth - 4));
      const top = Math.max(4, markRect.top - rootRect.top - tooltip.offsetHeight - 6);
      tooltip.style.left = `${left}px`;
      tooltip.style.top = `${top}px`;
    }
    function hide() { tooltip.style.visibility = "hidden"; }
  });
})();
</script>
'''
    return (
        template.replace("__D3_ARRAY__", d3_array)
        .replace("__D3_GEO__", d3_geo)
        .replace("__DATA__", encoded)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--snapshot-date", default=datetime.now(UTC).date().isoformat())
    parser.add_argument("--api-key", default=os.environ.get("AFDC_API_KEY", "DEMO_KEY"))
    parser.add_argument(
        "--national-csv",
        type=Path,
        help="Use one AFDC national CSV snapshot instead of city-by-city API queries.",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--visualization-html", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    manifest = json.loads((repo_root / "boundaries/top10/manifest.json").read_text())
    slugs = [item["slug"] for item in manifest["cities"]]
    output_dir = repo_root / "analysis/charging_preview" / args.snapshot_date
    raw_dir = repo_root / "data/sources/afdc" / f"query-{args.snapshot_date}"
    output_dir.mkdir(parents=True, exist_ok=True)

    national_source: tuple[pd.DataFrame, Path, str] | None = None
    if args.national_csv is not None:
        national_path = args.national_csv.resolve()
        national_payload = national_path.read_bytes()
        national_frame = pd.read_csv(io.BytesIO(national_payload), low_memory=False)
        required_filters = {
            "Fuel Type Code": {"ELEC"},
            "Status Code": {"E"},
            "Access Code": {"public"},
            "Country": {"US"},
        }
        for column, allowed in required_filters.items():
            observed = set(national_frame[column].dropna().astype(str).unique())
            if observed != allowed:
                raise ValueError(
                    f"National CSV filter mismatch for {column}: expected {allowed}, observed {observed}"
                )
        national_source = (
            national_frame,
            national_path,
            hashlib.sha256(national_payload).hexdigest(),
        )

    cities: list[dict[str, Any]] = []
    station_frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for slug in slugs:
        print(f"Processing {slug}...", flush=True)
        city, stations, source = _build_city_record(
            slug=slug,
            boundary_path=repo_root / f"boundaries/top10/{slug}/land_boundary.geojson",
            graph_path=repo_root / f"data/cities/{slug}/graph_all.graphml",
            raw_csv_path=raw_dir / f"{slug}-public-available.csv",
            api_key=args.api_key,
            refresh=args.refresh,
            national_source=national_source,
        )
        cities.append(city)
        station_frames.append(stations)
        sources.append(source)

    summary = pd.DataFrame([city["summary"] for city in cities])
    summary.to_csv(output_dir / "top10_charging_summary.csv", index=False)
    selected_columns = [
        "city_slug",
        "ID",
        "Station Name",
        "Latitude",
        "Longitude",
        "Status Code",
        "Access Code",
        "Restricted Access",
        "charging_category",
        "EV Level1 EVSE Num",
        "EV Level2 EVSE Num",
        "EV DC Fast Count",
        "EV Connector Types",
        "EV Network",
        "Maximum Vehicle Class",
        "Facility Type",
        "Date Last Confirmed",
        "Updated At",
    ]
    all_stations = pd.concat(station_frames, ignore_index=True)
    all_stations[selected_columns].to_csv(output_dir / "top10_charging_sites.csv", index=False)

    if national_source is None:
        last_updated_url = f"{API_ROOT}/last-updated.json?{urllib.parse.urlencode({'api_key': args.api_key})}"
        try:
            last_updated = json.loads(_download(last_updated_url))
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            last_updated = {"error": str(exc)}
    else:
        last_updated = {
            "source": "downloaded national CSV filename and capture date",
            "snapshot_label": args.snapshot_date,
        }
    provenance = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "snapshot_label": args.snapshot_date,
        "source": (
            "DOE/NLR Alternative Fuels Data Center national CSV data download"
            if national_source
            else "DOE/NLR Alternative Fuels Data Center API"
        ),
        "source_mode": "national_csv_snapshot" if national_source else "nearest_api_queries",
        "filter": {"fuel_type": "ELEC", "access": "public", "status": "E", "country": "US"},
        "exact_spatial_filter": "2025 Census TIGER/Line Place land boundary",
        "afdc_last_updated_response": last_updated,
        "cities": sources,
        "limitations": [
            "Audit preview only; no road anchoring has been performed.",
            "No Rivian connector, power, hours, or vehicle-class eligibility filter has been applied.",
            "AFDC public stations with restricted_access=true remain visible and are counted separately.",
        ],
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    compact = {"source": provenance, "cities": cities}
    (output_dir / "compact_map_data.json").write_text(
        json.dumps(compact, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    if args.visualization_html:
        args.visualization_html.parent.mkdir(parents=True, exist_ok=True)
        args.visualization_html.write_text(_html_fragment(compact), encoding="utf-8")
        print(f"Visualization: {args.visualization_html}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
