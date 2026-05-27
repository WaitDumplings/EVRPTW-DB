from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_ROOT = REPO_ROOT / "EVRPTW_Dataset_Generator"
sys.path.insert(0, str(GENERATOR_ROOT))

from evrptw_hierarchy.configs.config import load_yaml
from evrptw_hierarchy.io.persistence import ensure_dir


REQUIRED_CSVS = {
    "road_nodes_csv": "normalized/road_nodes.csv",
    "road_edges_csv": "normalized/road_edges.csv",
    "customer_seed_csv": "normalized/customer_seed.csv",
    "charging_station_csv": "normalized/charging_station.csv",
    "depot_candidate_csv": "normalized/depot_candidate.csv",
}
PROJECTED_CRS = "EPSG:5070"
WGS84_CRS = "EPSG:4326"
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and normalize public geodata for Geo-AC-v1 service territories."
    )
    parser.add_argument("--city-config", type=Path, default=GENERATOR_ROOT / "configs/geo_ac_v1_us10.yaml")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "EVRPTW_Dataset" / "Geo_AC_v1" / "source_data")
    parser.add_argument("--config-out", type=Path, default=GENERATOR_ROOT / "configs/geo_ac_v1_us10.with_sources.yaml")
    parser.add_argument("--territory-id", action="append", default=None)
    parser.add_argument("--territory-limit", type=int, default=None)
    parser.add_argument("--road-source", choices=["osm", "overture", "tiger", "both"], default="both")
    parser.add_argument("--network-type", default="drive")
    parser.add_argument("--overpass-url", default="https://overpass-api.de/api")
    parser.add_argument("--tiger-year", type=int, default=2025)
    parser.add_argument("--acs-year", type=int, default=2024)
    parser.add_argument("--overture-release", default="latest")
    parser.add_argument("--depot-source", choices=["osm_or_fallback", "fallback"], default="osm_or_fallback")
    parser.add_argument("--nrel-api-key-env", default="NREL_API_KEY")
    parser.add_argument("--census-api-key-env", default="CENSUS_API_KEY")
    parser.add_argument("--max-depot-candidates", type=int, default=40)
    parser.add_argument("--force", action="store_true", help="Re-download cached public source files.")
    return parser.parse_args()


def _require_geospatial_deps() -> dict[str, Any]:
    missing = []
    modules: dict[str, Any] = {}
    for name in ["geopandas", "networkx", "osmnx", "pandas", "requests", "shapely", "yaml"]:
        try:
            modules[name] = __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        install = "conda run -n maojie python -m pip install geopandas shapely pyproj requests osmnx duckdb pyarrow"
        raise RuntimeError(f"Missing geospatial dependencies: {missing}. Install with: {install}")
    return modules


def _state_county(fips: str) -> tuple[str, str]:
    fips = str(fips).zfill(5)
    return fips[:2], fips[2:]


def _download(url: str, path: Path, requests_mod: Any, force: bool = False, params: dict[str, Any] | None = None) -> Path:
    ensure_dir(path.parent)
    if path.exists() and not force:
        return path
    response = requests_mod.get(url, params=params, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        out = float(value)
        if not math.isfinite(out):
            return float(default)
        return out
    except (TypeError, ValueError):
        return float(default)


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return str(path)


def _load_county_polygon(spec: dict[str, Any], raw_dir: Path, tiger_year: int, requests_mod: Any, gpd: Any, force: bool) -> Any:
    county_fips = str(spec["county_fips"]).zfill(5)
    url = f"https://www2.census.gov/geo/tiger/TIGER{tiger_year}/COUNTY/tl_{tiger_year}_us_county.zip"
    zip_path = _download(url, raw_dir / "tiger" / f"tl_{tiger_year}_us_county.zip", requests_mod, force=force)
    counties = gpd.read_file(zip_path).to_crs(WGS84_CRS)
    county = counties[counties["GEOID"].astype(str) == county_fips]
    if county.empty:
        raise ValueError(f"County FIPS {county_fips} not found in TIGER {tiger_year} county file.")
    return county.iloc[[0]].copy()


def _load_block_groups(
    spec: dict[str, Any],
    raw_dir: Path,
    tiger_year: int,
    acs_year: int,
    census_key: str,
    requests_mod: Any,
    gpd: Any,
    pd: Any,
    force: bool,
) -> Any:
    state_fips, county_fips = _state_county(str(spec["county_fips"]))
    url = f"https://www2.census.gov/geo/tiger/TIGER{tiger_year}/BG/tl_{tiger_year}_{state_fips}_bg.zip"
    zip_path = _download(url, raw_dir / "tiger" / f"tl_{tiger_year}_{state_fips}_bg.zip", requests_mod, force=force)
    bgs = gpd.read_file(zip_path).to_crs(WGS84_CRS)
    bgs = bgs[(bgs["STATEFP"].astype(str) == state_fips) & (bgs["COUNTYFP"].astype(str) == county_fips)].copy()
    if bgs.empty:
        raise ValueError(f"No TIGER block groups found for county FIPS {state_fips}{county_fips}.")

    cache_path = raw_dir / "acs" / f"acs{acs_year}_{state_fips}_{county_fips}_B25002_002E.json"
    ensure_dir(cache_path.parent)
    if cache_path.exists() and not force:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        params = {
            "get": "NAME,B25002_002E",
            "for": "block group:*",
            "in": f"state:{state_fips} county:{county_fips}",
            "key": census_key,
        }
        response = requests_mod.get(f"https://api.census.gov/data/{acs_year}/acs/acs5", params=params, timeout=120)
        response.raise_for_status()
        try:
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(
                f"Census ACS API did not return JSON for county FIPS {state_fips}{county_fips}. "
                f"Response starts with: {response.text[:200]!r}"
            ) from exc
        cache_path.write_text(json.dumps(payload), encoding="utf-8")

    if not payload or len(payload) < 2:
        raise ValueError(f"ACS response for county FIPS {state_fips}{county_fips} has no block-group rows.")
    header = payload[0]
    acs = pd.DataFrame(payload[1:], columns=header)
    acs["GEOID"] = acs["state"] + acs["county"] + acs["tract"] + acs["block group"]
    acs["occupancy"] = pd.to_numeric(acs["B25002_002E"], errors="coerce").fillna(0.0).clip(lower=0.0)
    joined = bgs.merge(acs[["GEOID", "occupancy"]], on="GEOID", how="left")
    joined["occupancy"] = joined["occupancy"].fillna(0.0)
    return joined


def _point_rows_from_geometries(gdf: Any, id_cols: list[str], value_cols: list[str]) -> list[dict[str, Any]]:
    projected = gdf.to_crs(PROJECTED_CRS).copy()
    projected["point_geom"] = projected.geometry.representative_point()
    points_projected = projected.set_geometry("point_geom")
    points_wgs84 = points_projected.to_crs(WGS84_CRS)
    rows = []
    for idx, row in points_projected.iterrows():
        point_proj = points_projected.geometry.loc[idx]
        point_wgs = points_wgs84.geometry.loc[idx]
        out = {
            "lon": float(point_wgs.x),
            "lat": float(point_wgs.y),
            "x_km": float(point_proj.x / 1000.0),
            "y_km": float(point_proj.y / 1000.0),
        }
        for col in id_cols + value_cols:
            out[col] = row.get(col, "")
        rows.append(out)
    return rows


def _export_customer_seed(block_groups: Any, out_path: Path) -> dict[str, Any]:
    block_groups = block_groups.copy()
    block_groups["community_id"] = block_groups["GEOID"].astype(str)
    block_groups["tract"] = block_groups["TRACTCE"].astype(str)
    block_groups["block_group"] = block_groups["BLKGRPCE"].astype(str)
    rows = _point_rows_from_geometries(block_groups, ["community_id", "tract", "block_group"], ["occupancy"])
    rows = sorted(rows, key=lambda row: str(row["community_id"]))
    _write_csv(
        out_path,
        rows,
        ["community_id", "tract", "block_group", "lon", "lat", "x_km", "y_km", "occupancy"],
    )
    occupancy = [_to_float(row["occupancy"]) for row in rows]
    return {
        "customer_seed_count": len(rows),
        "occupancy_total": float(sum(occupancy)),
        "positive_occupancy_seed_count": int(sum(value > 0 for value in occupancy)),
    }


def _largest_weak_component(graph: Any, nx: Any) -> tuple[Any, float]:
    total = max(int(graph.number_of_nodes()), 1)
    components = list(nx.weakly_connected_components(graph))
    if not components:
        return graph, 0.0
    largest = max(components, key=len)
    return graph.subgraph(largest).copy(), float(len(largest) / total)


def _export_osm_roads(
    county: Any,
    normalized_dir: Path,
    raw_dir: Path,
    network_type: str,
    overpass_url: str,
    ox: Any,
    nx: Any,
) -> dict[str, Any]:
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(ensure_dir(raw_dir / "osmnx_cache"))
    ox.settings.overpass_url = str(overpass_url).rstrip("/")
    polygon = county.geometry.iloc[0]
    graph = ox.graph_from_polygon(
        polygon,
        network_type=network_type,
        simplify=True,
        retain_all=True,
        truncate_by_edge=False,
    )
    graph, largest_share = _largest_weak_component(graph, nx)
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    nodes_gdf = nodes_gdf.reset_index().rename(columns={"osmid": "source_node_id"})
    nodes_gdf = nodes_gdf.set_geometry("geometry").set_crs(WGS84_CRS, allow_override=True)
    nodes_proj = nodes_gdf.to_crs(PROJECTED_CRS)

    node_map: dict[Any, str] = {}
    node_rows = []
    for idx, row in nodes_gdf.iterrows():
        node_id = f"n{idx:07d}"
        source_id = row["source_node_id"]
        node_map[source_id] = node_id
        proj = nodes_proj.iloc[idx].geometry
        node_rows.append({
            "node_id": node_id,
            "lon": float(row["x"]),
            "lat": float(row["y"]),
            "x_km": float(proj.x / 1000.0),
            "y_km": float(proj.y / 1000.0),
        })

    edge_best: dict[tuple[str, str], dict[str, Any]] = {}
    for (u_raw, v_raw, _key), row in edges_gdf.iterrows():
        if u_raw not in node_map or v_raw not in node_map:
            continue
        u = node_map[u_raw]
        v = node_map[v_raw]
        if u == v:
            continue
        length_km = _to_float(row.get("length"), default=0.0) / 1000.0
        if length_km <= 0:
            continue
        key = tuple(sorted((u, v)))
        prev = edge_best.get(key)
        if prev is None or length_km < float(prev["length_km"]):
            edge_best[key] = {"u": key[0], "v": key[1], "length_km": length_km, "source": "osm"}

    edge_rows = sorted(edge_best.values(), key=lambda row: (row["u"], row["v"]))
    _write_csv(normalized_dir / "road_nodes.csv", node_rows, ["node_id", "lon", "lat", "x_km", "y_km"])
    _write_csv(normalized_dir / "road_edges.csv", edge_rows, ["u", "v", "length_km", "source"])
    return {
        "osm_road_node_count": len(node_rows),
        "osm_road_edge_count": len(edge_rows),
        "osm_largest_component_node_share": largest_share,
        "osm_total_edge_length_km": float(sum(float(row["length_km"]) for row in edge_rows)),
        "osm_overpass_url": str(overpass_url).rstrip("/"),
    }


def _afdc_request(
    spec: dict[str, Any],
    raw_dir: Path,
    requests_mod: Any,
    api_key: str,
    force: bool,
    loosen_status: bool = False,
) -> dict[str, Any]:
    state = str(spec["state"]).upper()
    suffix = "all_status" if loosen_status else "status_E"
    cache_path = raw_dir / "afdc" / f"afdc_{state}_{suffix}.json"
    ensure_dir(cache_path.parent)
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    params = {
        "api_key": api_key,
        "fuel_type": "ELEC",
        "country": "US",
        "state": state,
        "access": "public",
        "limit": "all",
    }
    if not loosen_status:
        params["status"] = "E"
    response = requests_mod.get("https://developer.nrel.gov/api/alt-fuel-stations/v1.json", params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _export_chargers(
    spec: dict[str, Any],
    county: Any,
    normalized_dir: Path,
    raw_dir: Path,
    requests_mod: Any,
    gpd: Any,
    pd: Any,
    api_key: str,
    force: bool,
) -> dict[str, Any]:
    payload = _afdc_request(spec, raw_dir, requests_mod, api_key, force, loosen_status=False)
    rows = payload.get("fuel_stations", [])
    if not rows:
        payload = _afdc_request(spec, raw_dir, requests_mod, api_key, force, loosen_status=True)
        rows = payload.get("fuel_stations", [])
    if not rows:
        raise ValueError(f"AFDC returned no public ELEC stations for state {spec['state']}.")

    points = []
    for row in rows:
        lon = _to_float(row.get("longitude"), default=float("nan"))
        lat = _to_float(row.get("latitude"), default=float("nan"))
        if lon == lon and lat == lat:
            points.append({**row, "geometry": gpd.points_from_xy([lon], [lat])[0]})
    gdf = gpd.GeoDataFrame(points, geometry="geometry", crs=WGS84_CRS)
    if gdf.empty:
        raise ValueError(f"AFDC station rows for {spec['state']} have no usable coordinates.")
    gdf = gdf[gdf.geometry.within(county.geometry.iloc[0])].copy()
    if gdf.empty:
        raise ValueError(f"No AFDC public charging stations clipped inside {spec['territory_id']}.")

    gdf_proj = gdf.to_crs(PROJECTED_CRS)
    out_rows = []
    for idx, row in gdf.iterrows():
        proj = gdf_proj.loc[idx].geometry
        out_rows.append({
            "station_id": row.get("id", ""),
            "name": row.get("station_name", ""),
            "lon": float(row.geometry.x),
            "lat": float(row.geometry.y),
            "x_km": float(proj.x / 1000.0),
            "y_km": float(proj.y / 1000.0),
            "network": row.get("ev_network", ""),
            "level2_count": int(_to_float(row.get("ev_level2_evse_num"), 0.0)),
            "dc_fast_count": int(_to_float(row.get("ev_dc_fast_num"), 0.0)),
            "status": row.get("status_code", ""),
            "access": row.get("access_code", ""),
        })
    out_rows = sorted(
        out_rows,
        key=lambda row: (-int(row["dc_fast_count"]), -int(row["level2_count"]), str(row["station_id"])),
    )
    _write_csv(
        normalized_dir / "charging_station.csv",
        out_rows,
        ["station_id", "name", "lon", "lat", "x_km", "y_km", "network", "level2_count", "dc_fast_count", "status", "access"],
    )
    return {
        "charging_station_count": len(out_rows),
        "charging_station_dc_fast_total": int(sum(int(row["dc_fast_count"]) for row in out_rows)),
        "charging_station_level2_total": int(sum(int(row["level2_count"]) for row in out_rows)),
        "afdc_state_station_count_before_clip": len(rows),
    }


def _features_from_polygon(ox: Any, polygon: Any, tags: dict[str, Any]) -> Any:
    if hasattr(ox, "features_from_polygon"):
        return ox.features_from_polygon(polygon, tags)
    return ox.features.features_from_polygon(polygon, tags)


def _classify_depot(row: Any) -> tuple[int, str]:
    building = str(row.get("building", "")).lower()
    landuse = str(row.get("landuse", "")).lower()
    industrial = str(row.get("industrial", "")).lower()
    name = str(row.get("name", "")).lower()
    text = " ".join([building, landuse, industrial, name])
    if "warehouse" in text or "distribution" in text or "logistics" in text or "freight" in text:
        return 100, "warehouse_logistics"
    if building in {"industrial", "commercial"}:
        return 70, f"{building}_building"
    if landuse == "industrial":
        return 45, "industrial_landuse"
    return 20, "industrial_related"


def _select_spread(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return rows
    rows = sorted(rows, key=lambda row: (-float(row.get("_score", 0.0)), str(row.get("candidate_id", ""))))
    selected = [rows[0]]
    remaining = rows[1:]
    while remaining and len(selected) < count:
        best_idx = 0
        best_value = -1.0
        for idx, row in enumerate(remaining):
            dist = min(
                (float(row["x_km"]) - float(sel["x_km"])) ** 2 + (float(row["y_km"]) - float(sel["y_km"])) ** 2
                for sel in selected
            )
            value = dist + 0.001 * float(row.get("_score", 0.0))
            if value > best_value:
                best_idx = idx
                best_value = value
        selected.append(remaining.pop(best_idx))
    return selected


def _fallback_depots(county: Any, gpd: Any, count: int) -> list[dict[str, Any]]:
    projected = county.to_crs(PROJECTED_CRS)
    bounds = projected.total_bounds
    minx, miny, maxx, maxy = bounds
    points = []
    steps = max(4, int(count**0.5) + 3)
    for ix in range(steps):
        for iy in range(steps):
            x = minx + (ix + 0.5) / steps * (maxx - minx)
            y = miny + (iy + 0.5) / steps * (maxy - miny)
            points.append((x, y))
    point_gdf = gpd.GeoDataFrame(
        {"idx": list(range(len(points)))},
        geometry=gpd.points_from_xy([p[0] for p in points], [p[1] for p in points]),
        crs=PROJECTED_CRS,
    )
    point_gdf = point_gdf[point_gdf.geometry.within(projected.geometry.iloc[0])].copy()
    point_wgs = point_gdf.to_crs(WGS84_CRS)
    rows = []
    for pos, (idx, row) in enumerate(point_gdf.iterrows()):
        wgs = point_wgs.loc[idx].geometry
        rows.append({
            "candidate_id": f"fallback_depot_{pos:03d}",
            "lon": float(wgs.x),
            "lat": float(wgs.y),
            "x_km": float(row.geometry.x / 1000.0),
            "y_km": float(row.geometry.y / 1000.0),
            "source": "fallback_center_region",
            "source_id": "",
            "category": "fallback_center_region",
            "_score": 1.0,
        })
    return _select_spread(rows, count)


def _export_depots(
    spec: dict[str, Any],
    county: Any,
    normalized_dir: Path,
    raw_dir: Path,
    ox: Any,
    gpd: Any,
    max_candidates: int,
    depot_source: str,
) -> dict[str, Any]:
    min_candidates = int(spec.get("depot_candidate_count", 6))
    if depot_source == "fallback":
        rows = _fallback_depots(county, gpd, max(min_candidates, min(max_candidates, min_candidates + 4)))
        for idx, row in enumerate(rows):
            row["candidate_id"] = f"depot_{idx:03d}"
            row.pop("_score", None)
        _write_csv(
            normalized_dir / "depot_candidate.csv",
            rows,
            ["candidate_id", "lon", "lat", "x_km", "y_km", "source", "source_id", "category"],
        )
        return {
            "depot_candidate_count": len(rows),
            "depot_source_mode": "fallback_center_region",
            "fallback_depot_count": len(rows),
            "raw_osm_depot_feature_count": 0,
        }

    tags = {
        "building": ["warehouse", "industrial", "commercial"],
        "landuse": "industrial",
        "industrial": True,
        "freight": True,
        "warehouse": True,
    }
    try:
        features = _features_from_polygon(ox, county.geometry.iloc[0], tags)
    except Exception as exc:
        features = gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS)
        features.attrs["fetch_error"] = str(exc)
    if not features.empty:
        ensure_dir(raw_dir / "osm")
        features.to_file(raw_dir / "osm" / "depot_candidate_features.geojson", driver="GeoJSON")

    rows = []
    if not features.empty:
        features = features.to_crs(WGS84_CRS)
        projected = features.to_crs(PROJECTED_CRS)
        projected["point_geom"] = projected.geometry.representative_point()
        point_projected = projected.set_geometry("point_geom")
        point_wgs = point_projected.to_crs(WGS84_CRS)
        for pos, (idx, row) in enumerate(point_projected.iterrows()):
            point_projected_geom = point_projected.geometry.iloc[pos]
            point_wgs84 = point_wgs.geometry.iloc[pos]
            if not point_wgs84.within(county.geometry.iloc[0]):
                continue
            score, category = _classify_depot(row)
            source_id = "|".join(str(part) for part in idx) if isinstance(idx, tuple) else str(idx)
            rows.append({
                "candidate_id": f"osm_depot_{pos:05d}",
                "lon": float(point_wgs84.x),
                "lat": float(point_wgs84.y),
                "x_km": float(point_projected_geom.x / 1000.0),
                "y_km": float(point_projected_geom.y / 1000.0),
                "source": "osm_poi_or_building",
                "source_id": source_id,
                "category": category,
                "_score": float(score),
            })

    source_mode = "osm_poi_or_building"
    osm_candidate_count = len(rows)
    if len(rows) < min_candidates:
        needed = max(min_candidates, min(max_candidates, min_candidates + 4))
        rows.extend(_fallback_depots(county, gpd, needed - len(rows)))
        source_mode = "mixed_osm_and_fallback_center_region" if osm_candidate_count else "fallback_center_region"

    rows = _select_spread(rows, int(max_candidates))
    for idx, row in enumerate(rows):
        row["candidate_id"] = f"depot_{idx:03d}"
        row.pop("_score", None)
    _write_csv(
        normalized_dir / "depot_candidate.csv",
        rows,
        ["candidate_id", "lon", "lat", "x_km", "y_km", "source", "source_id", "category"],
    )
    return {
        "depot_candidate_count": len(rows),
        "depot_source_mode": source_mode,
        "fallback_depot_count": int(sum(str(row.get("source")) == "fallback_center_region" for row in rows)),
        "raw_osm_depot_feature_count": int(len(features)) if hasattr(features, "__len__") else 0,
    }


def _resolve_overture_release(release: str, requests_mod: Any) -> str:
    if release != "latest":
        return release
    response = requests_mod.get(
        "https://overturemaps-us-west-2.s3.amazonaws.com/",
        params={"list-type": "2", "prefix": "release/", "delimiter": "/"},
        timeout=60,
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    prefixes = []
    for elem in root.iter():
        if elem.tag.endswith("Prefix") and elem.text and elem.text.startswith("release/"):
            value = elem.text.split("/")[1]
            if value:
                prefixes.append(value)
    if not prefixes:
        raise ValueError("Could not resolve latest Overture release from public S3 listing.")
    return sorted(set(prefixes))[-1]


def _try_export_overture_roads(
    spec: dict[str, Any],
    county: Any,
    normalized_dir: Path,
    requests_mod: Any,
    release: str,
) -> dict[str, Any]:
    try:
        import duckdb
        from shapely import wkt
        import geopandas as gpd
    except Exception as exc:
        return {"overture_status": "missing_dependency", "overture_error": str(exc)}
    try:
        resolved = _resolve_overture_release(release, requests_mod)
        min_lon, min_lat, max_lon, max_lat = county.total_bounds
        path = f"s3://overturemaps-us-west-2/release/{resolved}/theme=transportation/type=segment/*"
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("INSTALL spatial; LOAD spatial;")
        con.execute("SET s3_region='us-west-2';")
        query = f"""
            SELECT id, ST_AsText(ST_GeomFromWKB(geometry)) AS wkt
            FROM read_parquet('{path}', hive_partitioning=1)
            WHERE bbox.xmin <= {max_lon}
              AND bbox.xmax >= {min_lon}
              AND bbox.ymin <= {max_lat}
              AND bbox.ymax >= {min_lat}
            LIMIT 500000
        """
        df = con.execute(query).fetchdf()
        con.close()
        if df.empty:
            return {"overture_status": "empty", "overture_release": resolved, "overture_segment_count": 0}
        source_ids = []
        geom = []
        for source_id, value in zip(df["id"].astype(str).tolist(), df["wkt"].tolist()):
            if value:
                source_ids.append(source_id)
                geom.append(wkt.loads(value))
        gdf = gpd.GeoDataFrame({"source_id": source_ids}, geometry=geom, crs=WGS84_CRS)
        gdf = gdf[gdf.intersects(county.geometry.iloc[0])].copy()
        if gdf.empty:
            return {"overture_status": "empty_after_clip", "overture_release": resolved, "overture_segment_count": 0}
        gdf.to_file(normalized_dir / "overture_transportation_segments.geojson", driver="GeoJSON")
        return {
            "overture_status": "ok",
            "overture_release": resolved,
            "overture_segment_count": int(len(gdf)),
        }
    except Exception as exc:
        return {"overture_status": "failed", "overture_error": str(exc)}


def _line_parts(geometry: Any) -> list[Any]:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type == "MultiLineString":
        return [part for part in geometry.geoms if not part.is_empty]
    if geometry.geom_type == "GeometryCollection":
        out = []
        for part in geometry.geoms:
            out.extend(_line_parts(part))
        return out
    return []


def _export_overture_roads(
    spec: dict[str, Any],
    county: Any,
    normalized_dir: Path,
    requests_mod: Any,
    release: str,
    gpd: Any,
) -> dict[str, Any]:
    try:
        import duckdb
        from pyproj import Transformer
        from shapely import wkb
    except Exception as exc:
        raise RuntimeError(f"Overture road export requires duckdb, pyproj, and shapely: {exc}") from exc

    resolved = _resolve_overture_release(release, requests_mod)
    min_lon, min_lat, max_lon, max_lat = county.total_bounds
    path = f"s3://overturemaps-us-west-2/release/{resolved}/theme=transportation/type=segment/*"
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    query = f"""
        SELECT id, class, subtype, connectors, geometry
        FROM read_parquet('{path}', hive_partitioning=1)
        WHERE bbox.xmin <= {max_lon}
          AND bbox.xmax >= {min_lon}
          AND bbox.ymin <= {max_lat}
          AND bbox.ymax >= {min_lat}
          AND subtype = 'road'
          AND class NOT IN ('footway', 'cycleway', 'path', 'steps', 'bridleway', 'pedestrian', 'track')
        LIMIT 1000000
    """
    df = con.execute(query).fetchdf()
    con.close()
    if df.empty:
        raise ValueError(f"Overture returned no road segments for {spec['territory_id']}.")

    county_geom = county.geometry.iloc[0]
    transformer = Transformer.from_crs(WGS84_CRS, PROJECTED_CRS, always_xy=True)
    node_lookup: dict[tuple[float, float], str] = {}
    node_rows: list[dict[str, Any]] = []
    edge_best: dict[tuple[str, str], dict[str, Any]] = {}
    raw_segments = int(len(df))
    clipped_parts = 0

    def node_for(lon: float, lat: float) -> str:
        key = (round(float(lon), 7), round(float(lat), 7))
        value = node_lookup.get(key)
        if value is not None:
            return value
        x, y = transformer.transform(float(lon), float(lat))
        value = f"n{len(node_rows):07d}"
        node_lookup[key] = value
        node_rows.append({
            "node_id": value,
            "lon": float(lon),
            "lat": float(lat),
            "x_km": float(x / 1000.0),
            "y_km": float(y / 1000.0),
        })
        return value

    for _, row in df.iterrows():
        geom_bytes = row.get("geometry")
        if geom_bytes is None:
            continue
        try:
            geom = wkb.loads(bytes(geom_bytes))
        except Exception:
            continue
        if not geom.intersects(county_geom):
            continue
        clipped = geom.intersection(county_geom)
        for part in _line_parts(clipped):
            coords = list(part.coords)
            if len(coords) < 2:
                continue
            start = coords[0]
            end = coords[-1]
            if start == end:
                continue
            u = node_for(float(start[0]), float(start[1]))
            v = node_for(float(end[0]), float(end[1]))
            if u == v:
                continue
            projected_coords = [transformer.transform(float(x), float(y)) for x, y in coords]
            length_m = 0.0
            for (x0, y0), (x1, y1) in zip(projected_coords[:-1], projected_coords[1:]):
                length_m += math.hypot(float(x1) - float(x0), float(y1) - float(y0))
            length_km = float(length_m / 1000.0)
            if length_km <= 0:
                continue
            key = tuple(sorted((u, v)))
            prev = edge_best.get(key)
            if prev is None or length_km < float(prev["length_km"]):
                edge_best[key] = {
                    "u": key[0],
                    "v": key[1],
                    "length_km": length_km,
                    "source": "overture",
                }
            clipped_parts += 1

    edge_rows = sorted(edge_best.values(), key=lambda row: (row["u"], row["v"]))
    if not node_rows or not edge_rows:
        raise ValueError(f"Overture road export produced an empty graph for {spec['territory_id']}.")
    _write_csv(normalized_dir / "road_nodes.csv", node_rows, ["node_id", "lon", "lat", "x_km", "y_km"])
    _write_csv(normalized_dir / "road_edges.csv", edge_rows, ["u", "v", "length_km", "source"])
    return {
        "overture_status": "used_for_road_graph",
        "overture_release": resolved,
        "overture_raw_road_segment_count": raw_segments,
        "overture_clipped_line_part_count": clipped_parts,
        "overture_road_node_count": len(node_rows),
        "overture_road_edge_count": len(edge_rows),
        "overture_total_edge_length_km": float(sum(float(row["length_km"]) for row in edge_rows)),
    }


def _export_tiger_roads(
    spec: dict[str, Any],
    county: Any,
    normalized_dir: Path,
    raw_dir: Path,
    tiger_year: int,
    requests_mod: Any,
    gpd: Any,
    force: bool,
) -> dict[str, Any]:
    from pyproj import Transformer

    county_fips = str(spec["county_fips"]).zfill(5)
    url = f"https://www2.census.gov/geo/tiger/TIGER{tiger_year}/ROADS/tl_{tiger_year}_{county_fips}_roads.zip"
    zip_path = _download(url, raw_dir / "tiger" / f"tl_{tiger_year}_{county_fips}_roads.zip", requests_mod, force=force)
    roads = gpd.read_file(zip_path).to_crs(WGS84_CRS)
    if roads.empty:
        raise ValueError(f"TIGER ROADS returned no rows for {spec['territory_id']}.")

    county_geom = county.geometry.iloc[0]
    transformer = Transformer.from_crs(WGS84_CRS, PROJECTED_CRS, always_xy=True)
    node_lookup: dict[tuple[float, float], str] = {}
    node_rows: list[dict[str, Any]] = []
    edge_best: dict[tuple[str, str], dict[str, Any]] = {}
    clipped_parts = 0

    def node_for(lon: float, lat: float) -> str:
        key = (round(float(lon), 7), round(float(lat), 7))
        value = node_lookup.get(key)
        if value is not None:
            return value
        x, y = transformer.transform(float(lon), float(lat))
        value = f"n{len(node_rows):07d}"
        node_lookup[key] = value
        node_rows.append({
            "node_id": value,
            "lon": float(lon),
            "lat": float(lat),
            "x_km": float(x / 1000.0),
            "y_km": float(y / 1000.0),
        })
        return value

    for _, row in roads.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        clipped = geom.intersection(county_geom) if not geom.within(county_geom) else geom
        for part in _line_parts(clipped):
            coords = list(part.coords)
            if len(coords) < 2:
                continue
            start = coords[0]
            end = coords[-1]
            if start == end:
                continue
            u = node_for(float(start[0]), float(start[1]))
            v = node_for(float(end[0]), float(end[1]))
            if u == v:
                continue
            projected_coords = [transformer.transform(float(x), float(y)) for x, y in coords]
            length_m = 0.0
            for (x0, y0), (x1, y1) in zip(projected_coords[:-1], projected_coords[1:]):
                length_m += math.hypot(float(x1) - float(x0), float(y1) - float(y0))
            length_km = float(length_m / 1000.0)
            if length_km <= 0:
                continue
            key = tuple(sorted((u, v)))
            prev = edge_best.get(key)
            if prev is None or length_km < float(prev["length_km"]):
                edge_best[key] = {
                    "u": key[0],
                    "v": key[1],
                    "length_km": length_km,
                    "source": "tiger_roads",
                }
            clipped_parts += 1

    edge_rows = sorted(edge_best.values(), key=lambda row: (row["u"], row["v"]))
    if not node_rows or not edge_rows:
        raise ValueError(f"TIGER road export produced an empty graph for {spec['territory_id']}.")
    _write_csv(normalized_dir / "road_nodes.csv", node_rows, ["node_id", "lon", "lat", "x_km", "y_km"])
    _write_csv(normalized_dir / "road_edges.csv", edge_rows, ["u", "v", "length_km", "source"])
    return {
        "tiger_road_source": f"TIGER{int(tiger_year)} ROADS",
        "tiger_raw_road_count": int(len(roads)),
        "tiger_clipped_line_part_count": int(clipped_parts),
        "tiger_road_node_count": len(node_rows),
        "tiger_road_edge_count": len(edge_rows),
        "tiger_total_edge_length_km": float(sum(float(row["length_km"]) for row in edge_rows)),
    }


def _write_preview(county: Any, normalized_dir: Path, qa_dir: Path, gpd: Any, pd: Any) -> None:
    frames = []
    county_preview = county[["GEOID", "NAME", "geometry"]].copy()
    county_preview["layer"] = "county"
    frames.append(county_preview[["layer", "GEOID", "NAME", "geometry"]])
    for filename, layer, id_col in [
        ("customer_seed.csv", "customer_seed", "community_id"),
        ("charging_station.csv", "charging_station", "station_id"),
        ("depot_candidate.csv", "depot_candidate", "candidate_id"),
    ]:
        path = normalized_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        gdf = gpd.GeoDataFrame(
            {
                "layer": layer,
                "GEOID": df.get(id_col, pd.Series([""] * len(df))).astype(str),
                "NAME": df.get("name", pd.Series([""] * len(df))).astype(str),
            },
            geometry=gpd.points_from_xy(df["lon"], df["lat"]),
            crs=WGS84_CRS,
        )
        frames.append(gdf[["layer", "GEOID", "NAME", "geometry"]])
    if frames:
        preview = pd.concat(frames, ignore_index=True)
        gpd.GeoDataFrame(preview, geometry="geometry", crs=WGS84_CRS).to_file(qa_dir / "preview_layers.geojson", driver="GeoJSON")


def _write_qa_report(path: Path, spec: dict[str, Any], summary: dict[str, Any]) -> None:
    lines = [
        f"# Geo-AC-v1 Public Geodata QA: {spec['territory_id']}",
        "",
        f"County: {spec.get('county_name', '')}, {spec.get('state', '')} ({spec.get('county_fips', '')})",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in sorted(summary):
        value = summary[key]
        if isinstance(value, float):
            value = f"{value:.6g}"
        lines.append(f"| `{key}` | {value} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _log_stage(territory_id: str, stage: str) -> None:
    print(json.dumps({"territory": territory_id, "stage": stage}, sort_keys=True), flush=True)


def _nonempty(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f) > 1


def _validate_outputs(normalized_dir: Path) -> dict[str, Any]:
    out = {}
    for filename in REQUIRED_CSVS.values():
        path = normalized_dir / Path(filename).name
        out[f"{path.stem}_nonempty"] = _nonempty(path)
    return out


def _qa_summary_for(data_root: Path) -> dict[str, Any]:
    path = data_root / "qa" / "qa_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _road_source_from_summary(summary: dict[str, Any], args: argparse.Namespace) -> str:
    if summary.get("osm_road_node_count"):
        return "osm_osmnx"
    if summary.get("overture_road_node_count"):
        return "overture_transportation"
    if summary.get("tiger_road_node_count"):
        return "tiger_line_roads"
    return {
        "osm": "osm_osmnx",
        "both": "osm_osmnx",
        "overture": "overture_transportation",
        "tiger": "tiger_line_roads",
    }.get(str(args.road_source), str(args.road_source))


def _update_config_with_sources(
    city_cfg: dict[str, Any],
    selected_ids: set[str],
    output_root: Path,
    config_out: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config_base = config_out.resolve().parent
    updated = dict(city_cfg)
    territories = []
    for raw in city_cfg.get("territories", []):
        spec = dict(raw)
        territory_id = str(spec.get("territory_id"))
        data_root = output_root.resolve() / territory_id
        existing_outputs = all(_nonempty(data_root / value) for value in REQUIRED_CSVS.values())
        if territory_id in selected_ids or existing_outputs:
            qa_summary = _qa_summary_for(data_root)
            spec["data_root"] = _relative_or_absolute(data_root, config_base)
            spec["source_files"] = dict(REQUIRED_CSVS)
            spec["data_source_versions"] = {
                "tiger_line": f"TIGER{int(qa_summary.get('tiger_year', args.tiger_year))}",
                "acs": f"ACS{int(qa_summary.get('acs_year', args.acs_year))} 5-year B25002_002E",
                "roads": _road_source_from_summary(qa_summary, args),
                "chargers": "afdc_nrel",
                "depots": str(qa_summary.get("depot_source_mode", "osm_poi_building_or_fallback_marked")),
                "overture_release": str(qa_summary.get("overture_release", args.overture_release)),
            }
            spec["data_filters"] = {
                "road_source": str(qa_summary.get("road_source", args.road_source)),
                "road_network_type": str(args.network_type),
                "overpass_url": str(args.overpass_url).rstrip("/"),
                "depot_source": str(qa_summary.get("depot_source_mode", args.depot_source)),
                "chargers": "fuel_type=ELEC, access=public, clipped_to_county",
                "customer_seed": "TIGER block-group representative points weighted by ACS B25002_002E",
            }
        territories.append(spec)
    updated["territories"] = territories
    return updated


def _selected_territories(city_cfg: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    territories = list(city_cfg.get("territories", []))
    if args.territory_id:
        keep = set(args.territory_id)
        territories = [spec for spec in territories if str(spec.get("territory_id")) in keep]
    if args.territory_limit is not None:
        territories = territories[: int(args.territory_limit)]
    if not territories:
        raise ValueError("No territories selected.")
    return [dict(spec) for spec in territories]


def main() -> None:
    args = parse_args()
    mods = _require_geospatial_deps()
    gpd = mods["geopandas"]
    nx = mods["networkx"]
    ox = mods["osmnx"]
    pd = mods["pandas"]
    requests_mod = mods["requests"]
    yaml_mod = mods["yaml"]

    nrel_api_key = os.environ.get(args.nrel_api_key_env)
    if not nrel_api_key:
        raise RuntimeError(f"{args.nrel_api_key_env} is required to fetch AFDC/NREL public charging stations.")
    census_key = os.environ.get(args.census_api_key_env)
    if not census_key:
        raise RuntimeError(f"{args.census_api_key_env} is required to fetch ACS block-group occupancy data.")

    city_cfg = load_yaml(args.city_config)
    territories = _selected_territories(city_cfg, args)
    timing_rows = []
    selected_ids: set[str] = set()
    output_root = Path(args.output_root)

    for spec in territories:
        selected_ids.add(str(spec["territory_id"]))
        territory_id = str(spec["territory_id"])
        root = ensure_dir(output_root / territory_id)
        raw_dir = ensure_dir(root / "raw")
        normalized_dir = ensure_dir(root / "normalized")
        qa_dir = ensure_dir(root / "qa")
        start = time.perf_counter()

        _log_stage(territory_id, "load_county_polygon")
        county = _load_county_polygon(spec, raw_dir, int(args.tiger_year), requests_mod, gpd, bool(args.force))
        _log_stage(territory_id, "load_block_groups_and_acs")
        block_groups = _load_block_groups(
            spec,
            raw_dir,
            int(args.tiger_year),
            int(args.acs_year),
            census_key,
            requests_mod,
            gpd,
            pd,
            bool(args.force),
        )
        summary: dict[str, Any] = {
            "territory_id": territory_id,
            "county_fips": str(spec["county_fips"]).zfill(5),
            "tiger_year": int(args.tiger_year),
            "acs_year": int(args.acs_year),
            "road_source": args.road_source,
            "county_area_km2": float(county.to_crs(PROJECTED_CRS).area.iloc[0] / 1_000_000.0),
            "block_group_count": int(len(block_groups)),
        }

        _log_stage(territory_id, "export_customer_seed")
        summary.update(_export_customer_seed(block_groups, normalized_dir / "customer_seed.csv"))
        if args.road_source in {"osm", "both"}:
            _log_stage(territory_id, "export_osm_roads")
            summary.update(_export_osm_roads(county, normalized_dir, raw_dir, args.network_type, args.overpass_url, ox, nx))
        elif args.road_source == "overture":
            _log_stage(territory_id, "export_overture_roads")
            summary.update(_export_overture_roads(spec, county, normalized_dir, requests_mod, args.overture_release, gpd))
        elif args.road_source == "tiger":
            _log_stage(territory_id, "export_tiger_roads")
            summary.update(_export_tiger_roads(spec, county, normalized_dir, raw_dir, int(args.tiger_year), requests_mod, gpd, bool(args.force)))
        else:
            raise ValueError(f"Unsupported road source: {args.road_source}")
        _log_stage(territory_id, "export_chargers")
        summary.update(_export_chargers(spec, county, normalized_dir, raw_dir, requests_mod, gpd, pd, nrel_api_key, bool(args.force)))
        _log_stage(territory_id, "export_depots")
        summary.update(_export_depots(spec, county, normalized_dir, raw_dir, ox, gpd, int(args.max_depot_candidates), args.depot_source))
        if args.road_source == "both":
            _log_stage(territory_id, "export_overture_qa")
            summary.update(_try_export_overture_roads(spec, county, normalized_dir, requests_mod, args.overture_release))
        _log_stage(territory_id, "write_qa")
        summary.update(_validate_outputs(normalized_dir))
        summary["wall_time_s"] = round(time.perf_counter() - start, 6)

        (qa_dir / "qa_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        _write_qa_report(qa_dir / "qa_report.md", spec, summary)
        _write_preview(county, normalized_dir, qa_dir, gpd, pd)
        timing_rows.append({
            "territory_id": territory_id,
            "wall_time_s": summary["wall_time_s"],
            "normalized_dir": str(normalized_dir),
        })
        print(json.dumps({"territory": territory_id, "normalized_dir": str(normalized_dir), "wall_time_s": summary["wall_time_s"]}, indent=2))

    updated_cfg = _update_config_with_sources(city_cfg, selected_ids, output_root, Path(args.config_out), args)
    ensure_dir(Path(args.config_out).parent)
    with Path(args.config_out).open("w", encoding="utf-8") as f:
        yaml_mod.safe_dump(updated_cfg, f, sort_keys=False)

    _write_csv(
        output_root / "etl_timing.csv",
        timing_rows,
        ["territory_id", "wall_time_s", "normalized_dir"],
    )
    print(json.dumps({"config_out": str(args.config_out), "territory_count": len(selected_ids)}, indent=2))


if __name__ == "__main__":
    main()
