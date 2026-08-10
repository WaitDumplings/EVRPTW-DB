from __future__ import annotations

import gzip
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import shapely
from shapely.geometry import Point, box, mapping
from shapely.prepared import prep
from shapely.strtree import STRtree

from .util import sha256_file, write_json

DEFAULT_NSI_API_URL = "https://nsi.sec.usace.army.mil/nsiapi/structures"
ORDINARY_RESIDENTIAL_PREFIXES = ("RES1", "RES2", "RES3")
INSTITUTIONAL_RESIDENTIAL_CODES = {"RES4", "RES5", "RES6"}
DELIVERY_ROAD_CLASSES = {
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "road",
}
EXCLUDED_ANCHOR_ROAD_CLASSES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "construction",
    "proposed",
    "raceway",
}
SELECTED_NSI_FIELDS = (
    "fd_id",
    "bid",
    "occtype",
    "st_damcat",
    "resunits",
    "sqft",
    "num_story",
    "ftprntid",
    "ftprntsrc",
    "ftprntsqft",
    "bldheight",
    "usastrucid",
    "cbfips",
    "med_yr_blt",
    "x",
    "y",
)


@dataclass(frozen=True)
class NSICustomerOptions:
    city_slug: str
    city_label: str
    boundary_file: Path
    graph_file: Path
    output_dir: Path
    area_crs: str
    api_url: str = DEFAULT_NSI_API_URL
    tile_size_m: float = 5_000.0
    density_grid_m: float = 500.0
    workers: int = 4
    timeout_s: int = 300
    retries: int = 4

    def validate(self) -> None:
        if self.tile_size_m <= 0:
            raise ValueError("tile_size_m must be positive")
        if self.density_grid_m <= 0:
            raise ValueError("density_grid_m must be positive")
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.retries <= 0:
            raise ValueError("retries must be positive")


def iter_rfc7464_features(chunks: Iterable[bytes]) -> Iterator[dict[str, Any]]:
    """Parse an RFC 7464 GeoJSON feature stream without loading it into memory."""
    buffer = b""
    saw_record_separator = False
    for chunk in chunks:
        if not chunk:
            continue
        buffer += chunk
        if b"\x1e" not in buffer:
            continue
        saw_record_separator = True
        parts = buffer.split(b"\x1e")
        buffer = parts.pop()
        for part in parts:
            stripped = part.strip()
            if stripped:
                yield json.loads(stripped)
    stripped = buffer.strip()
    if stripped:
        if saw_record_separator:
            yield json.loads(stripped)
        else:
            for line in stripped.splitlines():
                if line.strip():
                    yield json.loads(line)


def _file_chunks(handle, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    while True:
        chunk = handle.read(chunk_size)
        if not chunk:
            return
        yield chunk


def _load_boundary(path: Path) -> gpd.GeoDataFrame:
    frame = gpd.read_file(path).to_crs("EPSG:4326")
    geometry = frame.geometry.union_all()
    return gpd.GeoDataFrame({"geometry": [geometry]}, crs="EPSG:4326")


def build_query_tiles(
    boundary: gpd.GeoDataFrame,
    area_crs: str,
    tile_size_m: float,
) -> gpd.GeoDataFrame:
    projected = boundary.to_crs(area_crs)
    city = projected.geometry.iloc[0]
    min_x, min_y, max_x, max_y = city.bounds
    start_x = math.floor(min_x / tile_size_m) * tile_size_m
    start_y = math.floor(min_y / tile_size_m) * tile_size_m
    records: list[dict[str, Any]] = []
    geometries = []
    row = 0
    y = start_y
    while y < max_y:
        column = 0
        x = start_x
        while x < max_x:
            intersection = city.intersection(box(x, y, x + tile_size_m, y + tile_size_m))
            if not intersection.is_empty and intersection.area > 1.0:
                records.append(
                    {
                        "tile_id": f"r{row:03d}-c{column:03d}",
                        "tile_row": row,
                        "tile_column": column,
                        "query_area_m2": float(intersection.area),
                    }
                )
                geometries.append(intersection)
            x += tile_size_m
            column += 1
        y += tile_size_m
        row += 1
    tiles = gpd.GeoDataFrame(records, geometry=geometries, crs=area_crs)
    return tiles.to_crs("EPSG:4326")


def _request_body(geometry: Any) -> bytes:
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": mapping(geometry),
            }
        ],
    }
    return json.dumps(feature_collection, separators=(",", ":")).encode("utf-8")


def _download_tile(
    tile_id: str,
    geometry: Any,
    raw_dir: Path,
    api_url: str,
    timeout_s: int,
    retries: int,
) -> dict[str, Any]:
    destination = raw_dir / f"{tile_id}.geojsonseq.gz"
    request_sha256 = hashlib.sha256(_request_body(geometry)).hexdigest()
    if destination.exists() and destination.stat().st_size > 0:
        return {
            "tile_id": tile_id,
            "path": str(destination),
            "cached": True,
            "request_sha256": request_sha256,
            "response_sha256": sha256_file(destination),
            "compressed_bytes": destination.stat().st_size,
        }

    raw_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        f"{api_url}?fmt=fs",
        data=_request_body(geometry),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json-seq, application/geo+json, application/json",
            "User-Agent": "EVRPTW-DB-NSI-pilot/0.1 (research dataset construction)",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            digest = hashlib.sha256()
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    raise RuntimeError(f"NSI returned HTTP {status} for {tile_id}")
                with gzip.open(temporary, "wb", compresslevel=6) as compressed:
                    for chunk in _file_chunks(response):
                        digest.update(chunk)
                        compressed.write(chunk)
            temporary.replace(destination)
            return {
                "tile_id": tile_id,
                "path": str(destination),
                "cached": False,
                "request_sha256": request_sha256,
                "response_payload_sha256": digest.hexdigest(),
                "response_sha256": sha256_file(destination),
                "compressed_bytes": destination.stat().st_size,
            }
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as error:
            last_error = error
            if temporary.exists():
                temporary.unlink()
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"Failed NSI tile {tile_id} after {retries} attempts") from last_error


def _download_tiles(
    tiles: gpd.GeoDataFrame,
    raw_dir: Path,
    options: NSICustomerOptions,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=options.workers) as executor:
        futures = {
            executor.submit(
                _download_tile,
                row.tile_id,
                row.geometry,
                raw_dir,
                options.api_url,
                options.timeout_s,
                options.retries,
            ): row.tile_id
            for row in tiles.itertuples()
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(f"NSI tile {completed}/{len(futures)}: {result['tile_id']}", flush=True)
    return sorted(results, key=lambda item: item["tile_id"])


def _safe_number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _ordinary_residential(occtype: Any) -> bool:
    code = str(occtype or "").upper()
    return code.startswith(ORDINARY_RESIDENTIAL_PREFIXES)


def classify_service_location(resunits: float, canonical_occtype: str) -> tuple[str, str]:
    """Map evidence to one delivery-location class while retaining the original NSI code."""
    if resunits >= 20:
        return "large_apt", "summed_nsi_resunits"
    if resunits >= 5:
        return "medium_apt", "summed_nsi_resunits"
    if resunits >= 2:
        return "small_apt", "summed_nsi_resunits"
    code = canonical_occtype.upper()
    if code == "RES2":
        return "manufactured_home", "nsi_occtype_fallback"
    if code.startswith(("RES3F", "RES3E")):
        return "large_apt", "nsi_occtype_fallback"
    if code.startswith(("RES3D", "RES3C")):
        return "medium_apt", "nsi_occtype_fallback"
    if code.startswith("RES3"):
        return "small_apt", "nsi_occtype_fallback"
    return "house", "nsi_occtype_fallback"


def structure_group_key(properties: dict[str, Any]) -> tuple[str, str]:
    for field in ("ftprntid", "usastrucid", "bid"):
        value = properties.get(field)
        if value is not None and str(value).strip() not in {"", "0", "nan", "None"}:
            return f"{field}:{value}", field
    return f"fd_id:{properties.get('fd_id')}", "fd_id"


def _record_identity(properties: dict[str, Any]) -> str:
    fd_id = properties.get("fd_id")
    if fd_id is not None:
        return f"fd:{fd_id}"
    return (
        "fallback:"
        + hashlib.sha1(
            json.dumps(
                {
                    "bid": properties.get("bid"),
                    "x": properties.get("x"),
                    "y": properties.get("y"),
                    "occtype": properties.get("occtype"),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )


def _parse_nsi_tiles(
    raw_paths: list[Path],
    boundary: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    boundary_prepared = prep(boundary)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    occupancy_counts: Counter[str] = Counter()
    damage_category_counts: Counter[str] = Counter()
    institutional_counts: Counter[str] = Counter()
    outside_boundary_count = 0
    duplicate_record_count = 0
    parsed_record_count = 0

    for raw_path in raw_paths:
        with gzip.open(raw_path, "rb") as handle:
            for feature in iter_rfc7464_features(_file_chunks(handle)):
                parsed_record_count += 1
                properties = feature.get("properties") or {}
                identity = _record_identity(properties)
                if identity in seen:
                    duplicate_record_count += 1
                    continue
                seen.add(identity)
                coordinates = (feature.get("geometry") or {}).get("coordinates") or []
                if len(coordinates) < 2:
                    continue
                lon, lat = float(coordinates[0]), float(coordinates[1])
                if not boundary_prepared.covers(Point(lon, lat)):
                    outside_boundary_count += 1
                    continue
                occtype = str(properties.get("occtype") or "UNKNOWN").upper()
                st_damcat = str(properties.get("st_damcat") or "UNKNOWN").upper()
                occupancy_counts[occtype] += 1
                damage_category_counts[st_damcat] += 1
                if occtype in INSTITUTIONAL_RESIDENTIAL_CODES:
                    institutional_counts[occtype] += 1
                if not _ordinary_residential(occtype):
                    continue
                group_key, group_method = structure_group_key(properties)
                record = {field: properties.get(field) for field in SELECTED_NSI_FIELDS}
                record.update(
                    {
                        "lon": lon,
                        "lat": lat,
                        "group_key": group_key,
                        "group_method": group_method,
                        "source_record_identity": identity,
                    }
                )
                records.append(record)

    frame = pd.DataFrame.from_records(records)
    audit = {
        "api_feature_count_before_deduplication": parsed_record_count,
        "duplicate_api_record_count": duplicate_record_count,
        "unique_structure_record_count": len(seen),
        "outside_exact_boundary_count": outside_boundary_count,
        "ordinary_residential_record_count": len(frame),
        "institutional_residential_excluded_counts": dict(sorted(institutional_counts.items())),
        "occupancy_counts": dict(sorted(occupancy_counts.items())),
        "damage_category_counts": dict(sorted(damage_category_counts.items())),
    }
    return frame, audit


def _profile_residential_groups(
    raw_paths: list[Path],
    residential_group_keys: set[str],
    boundary: Any,
) -> dict[str, dict[str, Any]]:
    boundary_prepared = prep(boundary)
    seen: set[str] = set()
    total_counts: defaultdict[str, int] = defaultdict(int)
    nonres_counts: defaultdict[str, int] = defaultdict(int)
    categories: defaultdict[str, set[str]] = defaultdict(set)
    for raw_path in raw_paths:
        with gzip.open(raw_path, "rb") as handle:
            for feature in iter_rfc7464_features(_file_chunks(handle)):
                properties = feature.get("properties") or {}
                identity = _record_identity(properties)
                if identity in seen:
                    continue
                seen.add(identity)
                coordinates = (feature.get("geometry") or {}).get("coordinates") or []
                if len(coordinates) < 2 or not boundary_prepared.covers(
                    Point(float(coordinates[0]), float(coordinates[1]))
                ):
                    continue
                group_key, _ = structure_group_key(properties)
                if group_key not in residential_group_keys:
                    continue
                category = str(properties.get("st_damcat") or "UNKNOWN").upper()
                total_counts[group_key] += 1
                categories[group_key].add(category)
                if category != "RES":
                    nonres_counts[group_key] += 1
    return {
        key: {
            "all_nsi_record_count_at_group": total_counts[key],
            "nonresidential_record_count_at_group": nonres_counts[key],
            "damage_categories_at_group": "|".join(sorted(categories[key])),
        }
        for key in residential_group_keys
    }


def _group_residential_records(
    records: pd.DataFrame,
    profiles: dict[str, dict[str, Any]],
    area_crs: str,
) -> gpd.GeoDataFrame:
    if records.empty:
        raise ValueError("NSI query returned no ordinary residential records")
    numeric_fields = ("resunits", "sqft", "num_story", "ftprntsqft", "bldheight")
    for field in numeric_fields:
        records[field] = pd.to_numeric(records[field], errors="coerce")
    records["resunits_clean"] = records["resunits"].fillna(0).clip(lower=0)
    points = gpd.GeoDataFrame(
        records,
        geometry=gpd.points_from_xy(records["lon"], records["lat"]),
        crs="EPSG:4326",
    )
    projected = points.to_crs(area_crs)
    records = records.copy()
    records["projected_x"] = projected.geometry.x.to_numpy()
    records["projected_y"] = projected.geometry.y.to_numpy()

    grouped = records.groupby("group_key", sort=False, observed=True)
    aggregates = grouped.agg(
        group_method=("group_method", "first"),
        residential_record_count=("source_record_identity", "size"),
        residential_occtype_count=("occtype", "nunique"),
        resunits_total=("resunits_clean", "sum"),
        resunits_max=("resunits_clean", "max"),
        sqft_total=("sqft", "sum"),
        sqft_max=("sqft", "max"),
        footprint_sqft_max=("ftprntsqft", "max"),
        stories_max=("num_story", "max"),
        building_height_m_max=("bldheight", "max"),
        projected_x_min=("projected_x", "min"),
        projected_x_max=("projected_x", "max"),
        projected_y_min=("projected_y", "min"),
        projected_y_max=("projected_y", "max"),
    ).reset_index()

    canonical = (
        records.sort_values(
            ["group_key", "resunits_clean", "fd_id"],
            ascending=[True, False, True],
            na_position="last",
            kind="mergesort",
        )
        .drop_duplicates("group_key", keep="first")[
            ["group_key", "fd_id", "bid", "ftprntid", "usastrucid", "occtype", "lon", "lat"]
        ]
        .rename(
            columns={
                "fd_id": "canonical_fd_id",
                "occtype": "canonical_occtype",
            }
        )
    )
    locations = aggregates.merge(canonical, on="group_key", how="left", validate="one_to_one")
    locations["coordinate_spread_m"] = np.hypot(
        locations["projected_x_max"] - locations["projected_x_min"],
        locations["projected_y_max"] - locations["projected_y_min"],
    )
    profile_frame = pd.DataFrame.from_dict(profiles, orient="index")
    profile_frame.index.name = "group_key"
    locations = locations.merge(
        profile_frame.reset_index(), on="group_key", how="left", validate="one_to_one"
    )
    locations["all_nsi_record_count_at_group"] = locations["all_nsi_record_count_at_group"].fillna(
        locations["residential_record_count"]
    )
    locations["nonresidential_record_count_at_group"] = locations[
        "nonresidential_record_count_at_group"
    ].fillna(0)
    locations["mixed_use_flag"] = locations["nonresidential_record_count_at_group"] > 0
    classes = [
        classify_service_location(row.resunits_total, str(row.canonical_occtype or ""))
        for row in locations.itertuples()
    ]
    locations["service_location_type"] = [item[0] for item in classes]
    locations["type_evidence"] = [item[1] for item in classes]
    locations["latent_service_location_id"] = locations["group_key"].map(
        lambda value: "nsi_la_" + hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:16]
    )
    locations["source_semantics"] = "NSI_2026_modeled_structure_inventory"
    locations["active_customer"] = False
    locations = locations.drop(
        columns=[
            "projected_x_min",
            "projected_x_max",
            "projected_y_min",
            "projected_y_max",
        ]
    )
    return gpd.GeoDataFrame(
        locations,
        geometry=gpd.points_from_xy(locations["lon"], locations["lat"]),
        crs="EPSG:4326",
    )


def _parse_multivalue(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value)
    text = str(value or "")
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, list):
                return tuple(str(item) for item in parsed)
        except json.JSONDecodeError:
            pass
    return (text,) if text else ()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _eligible_physical_edges(graph_file: Path, area_crs: str) -> gpd.GeoDataFrame:
    graph = ox.load_graphml(graph_file)
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True, fill_edge_geometry=True)
    edges = edges.reset_index()
    records: list[dict[str, Any]] = []
    geometries = []
    seen_physical: set[str] = set()
    for row in edges.itertuples(index=False):
        attributes = row._asdict()
        if _as_bool(attributes.get("transit_only", False)):
            continue
        classes = _parse_multivalue(attributes.get("highway"))
        if not classes or not any(value in DELIVERY_ROAD_CLASSES for value in classes):
            continue
        if all(value in EXCLUDED_ANCHOR_ROAD_CLASSES for value in classes):
            continue
        geometry = attributes.get("geometry")
        if geometry is None or geometry.is_empty:
            continue
        canonical = shapely.normalize(geometry)
        physical_hash = hashlib.sha1(canonical.wkb).hexdigest()[:20]
        if physical_hash in seen_physical:
            continue
        seen_physical.add(physical_hash)
        records.append(
            {
                "physical_edge_id": physical_hash,
                "edge_u": str(attributes.get("u")),
                "edge_v": str(attributes.get("v")),
                "edge_key": str(attributes.get("key")),
                "highway": "|".join(classes),
                "road_name": "|".join(_parse_multivalue(attributes.get("name"))),
                "oneway": _as_bool(attributes.get("oneway", False)),
            }
        )
        geometries.append(geometry)
    if not records:
        raise ValueError("Operational graph contains no eligible non-transit delivery edges")
    return gpd.GeoDataFrame(records, geometry=geometries, crs=edges.crs).to_crs(area_crs)


def _attach_road_access(
    locations: gpd.GeoDataFrame,
    graph_file: Path,
    area_crs: str,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    projected_locations = locations.to_crs(area_crs)
    edges = (
        _eligible_physical_edges(graph_file, area_crs)
        .sort_values("physical_edge_id", kind="mergesort")
        .reset_index(drop=True)
    )
    edge_geometries = np.asarray(edges.geometry.to_numpy(), dtype=object)
    point_geometries = np.asarray(projected_locations.geometry.to_numpy(), dtype=object)
    tree = STRtree(edge_geometries)
    nearest_indices = np.asarray(tree.nearest(point_geometries), dtype=int)
    matched_geometries = edge_geometries[nearest_indices]
    distances = np.asarray(shapely.distance(point_geometries, matched_geometries), dtype=float)
    offsets = shapely.line_locate_point(matched_geometries, point_geometries)
    anchor_geometries = shapely.line_interpolate_point(matched_geometries, offsets)
    anchor_wgs84 = gpd.GeoSeries(anchor_geometries, crs=area_crs).to_crs("EPSG:4326")

    matched_edges = edges.iloc[nearest_indices].reset_index(drop=True)
    result = locations.reset_index(drop=True).copy()
    for field in (
        "physical_edge_id",
        "edge_u",
        "edge_v",
        "edge_key",
        "highway",
        "road_name",
        "oneway",
    ):
        result[field] = matched_edges[field].to_numpy()
    result["road_access_distance_m"] = distances
    result["road_anchor_lon"] = anchor_wgs84.x.to_numpy()
    result["road_anchor_lat"] = anchor_wgs84.y.to_numpy()
    result["road_access_band"] = pd.cut(
        distances,
        bins=[-np.inf, 50.0, 100.0, 200.0, np.inf],
        labels=["le_50m", "50_100m", "100_200m", "gt_200m"],
    ).astype(str)
    result["provisional_access_eligible_200m"] = distances <= 200.0
    result = gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:4326")
    thresholds = {
        str(int(threshold)): {
            "count": int((distances <= threshold).sum()),
            "share": float((distances <= threshold).mean()),
        }
        for threshold in (25.0, 50.0, 100.0, 200.0, 500.0)
    }
    quantiles = np.quantile(distances, [0.0, 0.5, 0.9, 0.95, 0.99, 1.0])
    audit = {
        "eligible_physical_edge_count": len(edges),
        "edge_filter": {
            "included_highway_classes": sorted(DELIVERY_ROAD_CLASSES),
            "excluded_highway_classes": sorted(EXCLUDED_ANCHOR_ROAD_CLASSES),
            "transit_only_edges_excluded": True,
        },
        "distance_quantiles_m": dict(
            zip(("min", "p50", "p90", "p95", "p99", "max"), map(float, quantiles), strict=True)
        ),
        "threshold_sensitivity": thresholds,
        "frozen_acceptance_threshold": None,
        "pilot_review_threshold_m": 200,
    }
    return result, audit


def _write_density_grid(
    locations: gpd.GeoDataFrame,
    output_path: Path,
    area_crs: str,
    grid_m: float,
) -> None:
    projected = locations.to_crs(area_crs)
    frame = locations.drop(columns="geometry").copy()
    frame["grid_x"] = np.floor(projected.geometry.x.to_numpy() / grid_m).astype(int)
    frame["grid_y"] = np.floor(projected.geometry.y.to_numpy() / grid_m).astype(int)
    frame["access_le_50m"] = frame["road_access_distance_m"] <= 50
    frame["access_le_100m"] = frame["road_access_distance_m"] <= 100
    frame["access_le_200m"] = frame["road_access_distance_m"] <= 200
    type_counts = pd.crosstab([frame["grid_x"], frame["grid_y"]], frame["service_location_type"])
    sums = frame.groupby(["grid_x", "grid_y"], observed=True).agg(
        service_location_count=("latent_service_location_id", "size"),
        housing_unit_estimate=("resunits_total", "sum"),
        access_le_50m=("access_le_50m", "sum"),
        access_le_100m=("access_le_100m", "sum"),
        access_le_200m=("access_le_200m", "sum"),
        mixed_use_count=("mixed_use_flag", "sum"),
    )
    grid = sums.join(type_counts, how="left").reset_index()
    for category in ("house", "manufactured_home", "small_apt", "medium_apt", "large_apt"):
        if category not in grid:
            grid[category] = 0
    geometries = [
        box(x * grid_m, y * grid_m, (x + 1) * grid_m, (y + 1) * grid_m)
        for x, y in zip(grid["grid_x"], grid["grid_y"], strict=True)
    ]
    gpd.GeoDataFrame(grid, geometry=geometries, crs=area_crs).to_crs("EPSG:4326").to_file(
        output_path, driver="GeoJSON"
    )


def _count_values(frame: pd.DataFrame, field: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame[field].value_counts(dropna=False).sort_index().items()
    }


def build_nsi_customer_cle(options: NSICustomerOptions) -> dict[str, Any]:
    """Build one city's NSI latent service pool and map it to its operational graph."""
    options.validate()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = options.output_dir / "raw_tiles"
    boundary = _load_boundary(options.boundary_file)
    city_geometry = boundary.geometry.iloc[0]
    frozen_boundary_path = options.output_dir / "service_boundary.geojson"
    boundary.to_file(frozen_boundary_path, driver="GeoJSON")
    tiles = build_query_tiles(boundary, options.area_crs, options.tile_size_m)
    query_tiles_path = options.output_dir / "nsi_query_tiles.geojson"
    tiles.to_file(query_tiles_path, driver="GeoJSON")

    started_at = datetime.now(UTC).isoformat()
    tile_downloads = _download_tiles(tiles, raw_dir, options)
    raw_paths = [Path(item["path"]) for item in tile_downloads]
    records, parse_audit = _parse_nsi_tiles(raw_paths, city_geometry)
    if records["fd_id"].duplicated().any():
        raise AssertionError("Residential fd_id values are not unique after API deduplication")
    records_path = options.output_dir / "nsi_ordinary_residential_records.parquet"
    gpd.GeoDataFrame(
        records,
        geometry=gpd.points_from_xy(records["lon"], records["lat"]),
        crs="EPSG:4326",
    ).to_parquet(records_path, index=False)

    profiles = _profile_residential_groups(
        raw_paths, set(records["group_key"].astype(str)), city_geometry
    )
    locations = _group_residential_records(records, profiles, options.area_crs)
    locations, road_audit = _attach_road_access(locations, options.graph_file, options.area_crs)
    if locations["latent_service_location_id"].duplicated().any():
        raise AssertionError("Latent service location IDs are not unique")
    if not bool(locations.geometry.map(city_geometry.covers).all()):
        raise AssertionError(
            "A derived latent service location falls outside the exact land boundary"
        )

    locations_path = options.output_dir / "latent_service_locations.parquet"
    locations.to_parquet(locations_path, index=False)
    density_path = options.output_dir / "latent_service_density_500m.geojson"
    _write_density_grid(locations, density_path, options.area_crs, options.density_grid_m)

    completed_at = datetime.now(UTC).isoformat()
    manifest = {
        "schema": "evrptw_nsi_customer_cle_v1",
        "status": "pilot_for_policy_review_not_frozen_release",
        "city_slug": options.city_slug,
        "city_label": options.city_label,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "source": {
            "dataset": "USACE National Structure Inventory 2026 Base",
            "api_url": options.api_url,
            "query_format": "POST exact tile-city intersection polygons; fmt=fs RFC 7464",
            "source_semantics": "nationally consistent modeled inventory, not verified customers",
            "tile_count": len(tiles),
            "tile_size_m": options.tile_size_m,
            "tile_downloads": tile_downloads,
        },
        "boundary": {
            "path": str(options.boundary_file.resolve()),
            "sha256": sha256_file(options.boundary_file),
            "frozen_output": str(frozen_boundary_path),
            "membership_rule": (
                f"NSI point covered by reviewed {options.city_label} land boundary"
            ),
            "outside_exact_boundary_count": parse_audit["outside_exact_boundary_count"],
        },
        "road_graph": {
            "path": str(options.graph_file.resolve()),
            "sha256": sha256_file(options.graph_file),
            "role": "existing connected operational OSM graph; no synthetic connector",
        },
        "record_audit": parse_audit,
        "grouping": {
            "priority": ["ftprntid", "usastrucid", "bid", "fd_id"],
            "purpose": "one physical service location per shared structure/footprint group",
            "stacked_residential_records": "distinct fd_id records are retained and resunits are summed",
            "latent_service_location_count": len(locations),
            "group_method_counts": _count_values(locations, "group_method"),
            "multi_residential_record_location_count": int(
                (locations["residential_record_count"] > 1).sum()
            ),
            "mixed_use_location_count": int(locations["mixed_use_flag"].sum()),
        },
        "classification": {
            "ordinary_residential_included": ["RES1*", "RES2", "RES3A-F"],
            "institutional_residential_excluded_from_v1": ["RES4", "RES5", "RES6"],
            "unit_rules": {
                "house": "0-1 estimated units with RES1 fallback",
                "manufactured_home": "RES2 fallback when estimated units do not imply multi-unit",
                "small_apt": "2-4 summed estimated units, or RES3A-B fallback",
                "medium_apt": "5-19 summed estimated units, or RES3C-D fallback",
                "large_apt": "20+ summed estimated units, or RES3E-F fallback",
            },
            "service_location_type_counts": _count_values(locations, "service_location_type"),
            "estimated_housing_units_total": float(locations["resunits_total"].sum()),
        },
        "road_access": road_audit,
        "stage_semantics": {
            "latent_service_location": "eligible physical delivery stop in the City Logistics Environment",
            "active_customer": "not generated in this command",
            "package_count_demand_time_window": "deferred to Stage 2 instance construction",
        },
        "outputs": {
            "service_boundary": str(frozen_boundary_path),
            "query_tiles": str(query_tiles_path),
            "raw_feature_stream_tiles": str(raw_dir / "*.geojsonseq.gz"),
            "ordinary_residential_records": str(records_path),
            "latent_service_locations": str(locations_path),
            "density_grid": str(density_path),
        },
        "output_sha256": {
            "service_boundary": sha256_file(frozen_boundary_path),
            "query_tiles": sha256_file(query_tiles_path),
            "ordinary_residential_records": sha256_file(records_path),
            "latent_service_locations": sha256_file(locations_path),
            "density_grid": sha256_file(density_path),
        },
        "known_limitations": [
            "NSI occupancy, units, and structure attributes are modeled estimates, not parcel-level truth.",
            "A 200 m road-access threshold is shown only for sensitivity review and is not frozen.",
            "A road anchor is not yet an entrance or curb-access observation.",
            "No package, demand, service-time, time-window, weekday/weekend, or active-customer draw is performed.",
        ],
    }
    manifest_path = options.output_dir / "customer_cle_manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def verify_nsi_customer_cle(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "customer_cle_manifest.json"
    if not manifest_path.exists():
        return {"passed": False, "errors": [f"Missing {manifest_path}"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []

    output_names = {
        "service_boundary": "service_boundary.geojson",
        "query_tiles": "nsi_query_tiles.geojson",
        "ordinary_residential_records": "nsi_ordinary_residential_records.parquet",
        "latent_service_locations": "latent_service_locations.parquet",
        "density_grid": "latent_service_density_500m.geojson",
    }
    for label, filename in output_names.items():
        path = output_dir / filename
        if not path.exists():
            errors.append(f"Missing {filename}")
            continue
        expected = manifest.get("output_sha256", {}).get(label)
        if expected and sha256_file(path) != expected:
            errors.append(f"SHA-256 mismatch for {filename}")

    raw_dir = output_dir / "raw_tiles"
    tile_downloads = manifest.get("source", {}).get("tile_downloads", [])
    for item in tile_downloads:
        raw_path = raw_dir / Path(item["path"]).name
        if not raw_path.exists():
            errors.append(f"Missing raw tile {raw_path.name}")
        elif sha256_file(raw_path) != item.get("response_sha256"):
            errors.append(f"SHA-256 mismatch for raw tile {raw_path.name}")

    location_path = output_dir / "latent_service_locations.parquet"
    boundary_path = output_dir / "service_boundary.geojson"
    if location_path.exists() and boundary_path.exists():
        locations = gpd.read_parquet(location_path)
        expected_count = manifest.get("grouping", {}).get("latent_service_location_count")
        if len(locations) != expected_count:
            errors.append(
                f"Latent-location row count {len(locations)} != manifest {expected_count}"
            )
        if locations["latent_service_location_id"].duplicated().any():
            errors.append("Duplicate latent_service_location_id")
        if bool(locations["active_customer"].astype(bool).any()):
            errors.append("Stage 1 latent table unexpectedly contains active customers")
        if locations["road_access_distance_m"].isna().any() or bool(
            (locations["road_access_distance_m"] < 0).any()
        ):
            errors.append("Invalid road-access distance")
        service_boundary = gpd.read_file(boundary_path).geometry.union_all()
        if not bool(locations.geometry.map(service_boundary.covers).all()):
            errors.append("Location outside frozen service boundary")
        actual_types = _count_values(locations, "service_location_type")
        if actual_types != manifest.get("classification", {}).get("service_location_type_counts"):
            errors.append("Service-location type counts differ from manifest")
        for threshold in (25, 50, 100, 200, 500):
            actual = int((locations["road_access_distance_m"] <= threshold).sum())
            expected = (
                manifest.get("road_access", {})
                .get("threshold_sensitivity", {})
                .get(str(threshold), {})
                .get("count")
            )
            if actual != expected:
                errors.append(f"Road-access count at {threshold} m {actual} != manifest {expected}")

    graph_path = Path(manifest.get("road_graph", {}).get("path", ""))
    if graph_path.exists():
        if sha256_file(graph_path) != manifest.get("road_graph", {}).get("sha256"):
            errors.append("Operational graph SHA-256 differs from manifest")
    else:
        warnings.append("Operational graph path is unavailable; graph hash was not rechecked")

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "checked_raw_tile_count": len(tile_downloads),
        "checked_latent_location_count": (len(locations) if "locations" in locals() else None),
    }
