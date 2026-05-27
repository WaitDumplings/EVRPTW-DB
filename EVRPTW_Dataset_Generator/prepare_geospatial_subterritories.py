from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

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
LATENT_CUSTOMER_SOURCE_KEY = "latent_customer_csv"
LATENT_CUSTOMER_REL = "normalized/latent_customer.csv"
CUSTOMER_SEED_COLUMNS = ["community_id", "tract", "block_group", "lon", "lat", "x_km", "y_km", "occupancy"]
DEPOT_COLUMNS = ["candidate_id", "lon", "lat", "x_km", "y_km", "source", "source_id", "category"]
LATENT_CUSTOMER_COLUMNS = [
    "customer_id",
    "community_id",
    "lon",
    "lat",
    "x_km",
    "y_km",
    "snap_lon",
    "snap_lat",
    "snap_x_km",
    "snap_y_km",
    "snap_edge_u",
    "snap_edge_v",
    "snap_node_id",
    "snap_distance_km",
    "connector_distance_km",
    "occupancy_weight",
    "source",
]
PROJECTED_CRS = "EPSG:5070"
WGS84_CRS = "EPSG:4326"
ROAD_COLOR = "#59616a"
COMMUNITY_SEED_COLOR = "#2563eb"
REMOTE_SEED_COLOR = "#9d174d"
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slice large Geo-AC-v1 county containers into sub-county service territories."
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        default=GENERATOR_ROOT / "configs/geo_ac_v1_us10.with_sources.yaml",
    )
    parser.add_argument(
        "--slice-config",
        type=Path,
        default=GENERATOR_ROOT / "configs/geo_ac_v1_na_us20_slices.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "EVRPTW_Dataset" / "Geo_AC_v1" / "source_data_na_us20",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=GENERATOR_ROOT / "configs/geo_ac_v1_na_us20.with_sources.yaml",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite generated CSV/QA files.")
    parser.add_argument("--skip-maps", action="store_true", help="Write CSVs and QA without PNG maps.")
    return parser.parse_args()


def _require_deps() -> dict[str, Any]:
    modules: dict[str, Any] = {}
    missing = []
    for name in ["geopandas", "matplotlib", "networkx", "numpy", "pandas", "pyproj", "shapely", "yaml"]:
        try:
            modules[name] = __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        install = "conda run -n maojie python -m pip install geopandas shapely pyproj requests osmnx duckdb pyarrow matplotlib"
        raise RuntimeError(f"Missing geospatial slicing dependencies: {missing}. Install with: {install}")
    return modules


def _write_yaml(path: Path, data: dict[str, Any], yaml_mod: Any) -> None:
    class NoAliasDumper(yaml_mod.SafeDumper):
        def ignore_aliases(self, data: Any) -> bool:
            return True

    ensure_dir(path.parent)
    path.write_text(
        yaml_mod.dump(data, Dumper=NoAliasDumper, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return str(path)


def _resolve_path(path: str | Path, base: Path) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    return (base / raw).resolve()


def _source_root(spec: dict[str, Any], config_path: Path) -> Path:
    if "data_root" not in spec:
        raise ValueError(f"{spec.get('territory_id')} has no data_root.")
    return _resolve_path(spec["data_root"], config_path.parent)


def _source_file(spec: dict[str, Any], config_path: Path, key: str) -> Path:
    root = _source_root(spec, config_path)
    rel = (spec.get("source_files", {}) or REQUIRED_CSVS).get(key)
    if rel is None:
        raise ValueError(f"{spec.get('territory_id')} has no source_files.{key}.")
    return _resolve_path(rel, root)


def _read_county_boundary(parent_root: Path, gpd: Any) -> Any:
    preview = parent_root / "qa" / "preview_layers.geojson"
    if not preview.exists():
        raise FileNotFoundError(f"Missing county preview boundary: {preview}")
    layers = gpd.read_file(preview).to_crs(WGS84_CRS)
    if "layer" in layers.columns:
        county = layers[layers["layer"] == "county"].copy()
    else:
        county = layers[layers.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].head(1).copy()
    if county.empty:
        raise ValueError(f"No county boundary layer found in {preview}.")
    return county.iloc[[0]].copy()


def _largest_polygon_component(county_wgs: Any, gpd: Any) -> Any:
    geometry = county_wgs.geometry.iloc[0]
    parts = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    largest = max(parts, key=lambda geom: geom.area)
    out = county_wgs.iloc[[0]].copy()
    out.geometry = gpd.GeoSeries([largest], crs=county_wgs.crs)
    return out


def _anchor_cells(county_wgs: Any, children: list[dict[str, Any]], deps: dict[str, Any]) -> list[dict[str, Any]]:
    gpd = deps["geopandas"]
    shapely_geom = __import__("shapely.geometry", fromlist=["MultiPoint"])
    shapely_ops = __import__("shapely.ops", fromlist=["voronoi_diagram"])

    anchors = gpd.GeoDataFrame(
        children,
        geometry=gpd.points_from_xy(
            [float(child["anchor_lonlat"][0]) for child in children],
            [float(child["anchor_lonlat"][1]) for child in children],
        ),
        crs=WGS84_CRS,
    ).to_crs(PROJECTED_CRS)
    county_proj = county_wgs.to_crs(PROJECTED_CRS).geometry.iloc[0]
    points = shapely_geom.MultiPoint(list(anchors.geometry))
    diagram = shapely_ops.voronoi_diagram(points, envelope=county_proj.buffer(100_000), edges=False)
    polygons = [geom for geom in diagram.geoms if not geom.is_empty]

    cells: list[dict[str, Any]] = []
    used: set[int] = set()
    for idx, child in enumerate(children):
        point = anchors.geometry.iloc[idx]
        candidate_idx = None
        for poly_idx, polygon in enumerate(polygons):
            if poly_idx in used:
                continue
            if polygon.covers(point) or polygon.distance(point) < 1e-6:
                candidate_idx = poly_idx
                break
        if candidate_idx is None:
            distances = [
                (poly_idx, polygon.distance(point))
                for poly_idx, polygon in enumerate(polygons)
                if poly_idx not in used
            ]
            candidate_idx = min(distances, key=lambda item: item[1])[0]
        used.add(candidate_idx)
        cell_proj = polygons[candidate_idx].intersection(county_proj)
        if cell_proj.is_empty:
            raise ValueError(f"Empty sliced cell for {child['territory_id']}.")
        cell_wgs = gpd.GeoSeries([cell_proj], crs=PROJECTED_CRS).to_crs(WGS84_CRS).iloc[0]
        out = dict(child)
        out["geometry_projected"] = cell_proj
        out["geometry_wgs84"] = cell_wgs
        out["anchor_x_km"] = float(point.x / 1000.0)
        out["anchor_y_km"] = float(point.y / 1000.0)
        cells.append(out)
    return cells


def _geometry_parts(geometry: Any) -> list[Any]:
    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    return [geometry]


def _update_cell_geometry(cell: dict[str, Any], geometry_wgs: Any, gpd: Any) -> None:
    cell["geometry_wgs84"] = geometry_wgs
    cell["geometry_projected"] = gpd.GeoSeries([geometry_wgs], crs=WGS84_CRS).to_crs(PROJECTED_CRS).iloc[0]


def _apply_component_reassignments(
    cells: list[dict[str, Any]],
    group: dict[str, Any],
    deps: dict[str, Any],
) -> list[dict[str, Any]]:
    reassignments = group.get("component_reassignments", []) or []
    if not reassignments:
        return cells
    gpd = deps["geopandas"]
    shapely_geom = __import__("shapely.geometry", fromlist=["box"])
    shapely_ops = __import__("shapely.ops", fromlist=["unary_union"])
    by_id = {str(cell["territory_id"]): cell for cell in cells}
    for rule in reassignments:
        from_id = str(rule["from_territory_id"])
        to_id = str(rule["to_territory_id"])
        if from_id not in by_id or to_id not in by_id:
            raise ValueError(f"Invalid component reassignment {from_id} -> {to_id}.")
        bbox = rule.get("bbox_lonlat")
        if not bbox or len(bbox) != 4:
            raise ValueError(f"Component reassignment {from_id} -> {to_id} requires bbox_lonlat.")
        selector = shapely_geom.box(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        from_parts = []
        moved_parts = []
        for part in _geometry_parts(by_id[from_id]["geometry_wgs84"]):
            representative = part.representative_point()
            if selector.covers(representative):
                moved_parts.append(part)
            else:
                from_parts.append(part)
        if not moved_parts:
            continue
        from_geometry = shapely_ops.unary_union(from_parts)
        to_geometry = shapely_ops.unary_union(_geometry_parts(by_id[to_id]["geometry_wgs84"]) + moved_parts)
        if from_geometry.is_empty or to_geometry.is_empty:
            raise ValueError(f"Component reassignment {from_id} -> {to_id} produced an empty cell.")
        _update_cell_geometry(by_id[from_id], from_geometry, gpd)
        _update_cell_geometry(by_id[to_id], to_geometry, gpd)
    return cells


def _point_mask(df: Any, polygon_wgs: Any, gpd: Any) -> Any:
    points = gpd.GeoSeries(gpd.points_from_xy(df["lon"], df["lat"]), crs=WGS84_CRS)
    return points.within(polygon_wgs) | points.touches(polygon_wgs)


def _read_nodes(path: Path, pd: Any) -> Any:
    return pd.read_csv(path, dtype={"node_id": str})


def _read_edges(path: Path, pd: Any) -> Any:
    return pd.read_csv(path, dtype={"u": str, "v": str})


def _largest_component_node_ids(nodes: Any, edges: Any, nx: Any) -> tuple[set[str], float]:
    graph = nx.Graph()
    graph.add_nodes_from(nodes["node_id"].astype(str).tolist())
    graph.add_edges_from(edges[["u", "v"]].astype(str).itertuples(index=False, name=None))
    if graph.number_of_nodes() == 0:
        return set(), 0.0
    components = list(nx.connected_components(graph))
    if not components:
        return set(), 0.0
    largest = max(components, key=len)
    return set(largest), float(len(largest) / max(graph.number_of_nodes(), 1))


def _clip_roads(
    parent_spec: dict[str, Any],
    source_config: Path,
    polygon_wgs: Any,
    normalized_dir: Path,
    deps: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    pd = deps["pandas"]
    gpd = deps["geopandas"]
    nx = deps["networkx"]
    nodes = _read_nodes(_source_file(parent_spec, source_config, "road_nodes_csv"), pd)
    edges = _read_edges(_source_file(parent_spec, source_config, "road_edges_csv"), pd)
    mask = _point_mask(nodes, polygon_wgs, gpd)
    selected_nodes = nodes[mask.to_numpy()].copy()
    selected_ids = set(selected_nodes["node_id"].astype(str))
    selected_edges = edges[
        edges["u"].astype(str).isin(selected_ids) & edges["v"].astype(str).isin(selected_ids)
    ].copy()
    _, component_share = _largest_component_node_ids(selected_nodes, selected_edges, nx)
    if selected_nodes.empty or selected_edges.empty:
        raise ValueError(f"Road slicing produced an empty graph for {parent_spec['territory_id']}.")
    selected_nodes.to_csv(normalized_dir / "road_nodes.csv", index=False)
    selected_edges.to_csv(normalized_dir / "road_edges.csv", index=False)
    summary = {
        "road_node_count": int(len(selected_nodes)),
        "road_edge_count": int(len(selected_edges)),
        "road_largest_component_node_share_after_slice": float(component_share),
        "road_total_edge_length_km": float(pd.to_numeric(selected_edges["length_km"], errors="coerce").fillna(0).sum()),
    }
    return selected_nodes, selected_edges, summary


def _clip_point_csv(
    parent_spec: dict[str, Any],
    source_config: Path,
    key: str,
    out_name: str,
    polygon_wgs: Any,
    normalized_dir: Path,
    deps: dict[str, Any],
) -> Any:
    pd = deps["pandas"]
    gpd = deps["geopandas"]
    df = pd.read_csv(_source_file(parent_spec, source_config, key))
    if df.empty:
        df.to_csv(normalized_dir / out_name, index=False)
        return df
    clipped = df[_point_mask(df, polygon_wgs, gpd).to_numpy()].copy()
    clipped.to_csv(normalized_dir / out_name, index=False)
    return clipped


def _append_fallback_depots(
    depots: Any,
    road_nodes: Any,
    child: dict[str, Any],
    min_count: int,
    pd: Any,
    np_mod: Any,
    seeds: Any | None = None,
    max_seed_distance_km: float = 25.0,
) -> tuple[Any, int]:
    if len(depots) >= min_count:
        return depots, 0
    rows = depots.to_dict("records")
    existing_ids = {str(row.get("candidate_id", "")) for row in rows}
    need = int(min_count - len(rows))
    nodes = road_nodes.copy()
    if seeds is not None and not seeds.empty and {"x_km", "y_km"}.issubset(seeds.columns):
        nearest_seed = _nearest_distances(
            nodes[["x_km", "y_km"]].to_numpy(dtype=float),
            seeds[["x_km", "y_km"]].to_numpy(dtype=float),
            np_mod,
        )
        nodes["nearest_seed_km"] = nearest_seed
        seed_supported = nodes[nodes["nearest_seed_km"] <= float(max_seed_distance_km)].copy()
        if not seed_supported.empty:
            nodes = seed_supported
    anchor_x = float(child.get("anchor_x_km", pd.to_numeric(nodes["x_km"], errors="coerce").median()))
    anchor_y = float(child.get("anchor_y_km", pd.to_numeric(nodes["y_km"], errors="coerce").median()))
    nodes["dx"] = pd.to_numeric(nodes["x_km"], errors="coerce") - anchor_x
    nodes["dy"] = pd.to_numeric(nodes["y_km"], errors="coerce") - anchor_y
    nodes["anchor_distance_km"] = (nodes["dx"] ** 2 + nodes["dy"] ** 2) ** 0.5
    ordered = nodes.sort_values("anchor_distance_km")
    chosen: list[Any] = []
    min_spacing_km = 5.0
    for _, row in ordered.iterrows():
        x = float(row["x_km"])
        y = float(row["y_km"])
        if all(math.hypot(x - float(prev["x_km"]), y - float(prev["y_km"])) >= min_spacing_km for prev in chosen):
            chosen.append(row)
        if len(chosen) >= need:
            break
    if len(chosen) < need:
        for _, row in ordered.iterrows():
            if str(row["node_id"]) not in {str(prev["node_id"]) for prev in chosen}:
                chosen.append(row)
            if len(chosen) >= need:
                break

    for idx, row in enumerate(chosen[:need]):
        candidate_id = f"fallback_{idx:03d}"
        while candidate_id in existing_ids:
            idx += 1
            candidate_id = f"fallback_{idx:03d}"
        existing_ids.add(candidate_id)
        rows.append(
            {
                "candidate_id": candidate_id,
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "x_km": float(row["x_km"]),
                "y_km": float(row["y_km"]),
                "source": "fallback_subterritory_road_node",
                "source_id": str(row["node_id"]),
                "category": "fallback_subterritory_road_node",
            }
        )
    return pd.DataFrame(rows, columns=DEPOT_COLUMNS), need


def _nearest_distances(points: Any, targets: Any, np_mod: Any, chunk_size: int = 512) -> Any:
    points_arr = np_mod.asarray(points, dtype=float)
    targets_arr = np_mod.asarray(targets, dtype=float)
    if points_arr.size == 0 or targets_arr.size == 0:
        return np_mod.full(points_arr.shape[0], np_mod.inf, dtype=float)
    out = np_mod.full(points_arr.shape[0], np_mod.inf, dtype=float)
    for start in range(0, points_arr.shape[0], chunk_size):
        block = points_arr[start : start + chunk_size]
        diff = block[:, None, :] - targets_arr[None, :, :]
        dist = np_mod.sqrt(np_mod.sum(diff * diff, axis=2))
        out[start : start + len(block)] = np_mod.min(dist, axis=1)
    return out


def _annotate_seed_outliers(seeds: Any, np_mod: Any, threshold_km: float = 15.0) -> dict[str, Any]:
    if seeds.empty or not {"x_km", "y_km"}.issubset(seeds.columns):
        seeds["nearest_seed_km"] = []
        seeds["is_remote_seed_outlier"] = []
        return {"remote_seed_outlier_count": 0, "max_seed_nearest_neighbor_km": 0.0}
    if len(seeds) == 1:
        nearest = np_mod.asarray([np_mod.inf], dtype=float)
    else:
        points = seeds[["x_km", "y_km"]].to_numpy(dtype=float)
        nearest = np_mod.full(len(seeds), np_mod.inf, dtype=float)
        for start in range(0, len(points), 512):
            block = points[start : start + 512]
            diff = block[:, None, :] - points[None, :, :]
            dist = np_mod.sqrt(np_mod.sum(diff * diff, axis=2))
            for local_idx in range(len(block)):
                dist[local_idx, start + local_idx] = np_mod.inf
            nearest[start : start + len(block)] = np_mod.min(dist, axis=1)
    seeds["nearest_seed_km"] = nearest
    seeds["is_remote_seed_outlier"] = nearest >= float(threshold_km)
    return {
        "remote_seed_outlier_count": int(seeds["is_remote_seed_outlier"].sum()),
        "remote_seed_outlier_threshold_km": float(threshold_km),
        "max_seed_nearest_neighbor_km": float(np_mod.nanmax(nearest[np_mod.isfinite(nearest)])) if np_mod.isfinite(nearest).any() else 0.0,
    }


def _annotate_depot_outliers(depots: Any, seeds: Any, np_mod: Any, threshold_km: float = 25.0) -> dict[str, Any]:
    if depots.empty or seeds.empty or not {"x_km", "y_km"}.issubset(depots.columns) or not {"x_km", "y_km"}.issubset(seeds.columns):
        depots["nearest_seed_km"] = []
        depots["is_remote_depot_outlier"] = []
        return {"remote_depot_outlier_count": 0, "max_depot_to_seed_km": 0.0}
    nearest = _nearest_distances(
        depots[["x_km", "y_km"]].to_numpy(dtype=float),
        seeds[["x_km", "y_km"]].to_numpy(dtype=float),
        np_mod,
    )
    depots["nearest_seed_km"] = nearest
    depots["is_remote_depot_outlier"] = nearest >= float(threshold_km)
    return {
        "remote_depot_outlier_count": int(depots["is_remote_depot_outlier"].sum()),
        "remote_depot_outlier_threshold_km": float(threshold_km),
        "max_depot_to_seed_km": float(np_mod.nanmax(nearest[np_mod.isfinite(nearest)])) if np_mod.isfinite(nearest).any() else 0.0,
    }


def _standard_columns(df: Any, columns: list[str]) -> Any:
    keep = [col for col in columns if col in df.columns]
    return df[keep].copy()


def _filter_sparse_remote_seeds(seeds: Any, filter_cfg: dict[str, Any], np_mod: Any) -> tuple[Any, Any, dict[str, Any]]:
    seeds = seeds.copy()
    threshold = float(filter_cfg.get("remote_seed_nearest_neighbor_km", 15.0))
    before = _annotate_seed_outliers(seeds, np_mod, threshold)
    remove = bool(filter_cfg.get("remove_sparse_remote_community_seeds", True))
    if not remove or "is_remote_seed_outlier" not in seeds.columns:
        return _standard_columns(seeds, CUSTOMER_SEED_COLUMNS), seeds.iloc[0:0].copy(), {
            **before,
            "removed_remote_seed_count": 0,
            "customer_seed_count_before_remote_filter": int(len(seeds)),
            "customer_seed_count_after_remote_filter": int(len(seeds)),
        }
    remote = seeds[seeds["is_remote_seed_outlier"].astype(bool)].copy()
    kept = seeds[~seeds["is_remote_seed_outlier"].astype(bool)].copy()
    if kept.empty:
        kept = seeds.copy()
        remote = seeds.iloc[0:0].copy()
    kept_standard = _standard_columns(kept, CUSTOMER_SEED_COLUMNS)
    return kept_standard, remote, {
        "remote_seed_outlier_threshold_km": threshold,
        "remote_seed_outlier_count_before_filter": int(before.get("remote_seed_outlier_count", 0)),
        "removed_remote_seed_count": int(len(remote)),
        "customer_seed_count_before_remote_filter": int(len(seeds)),
        "customer_seed_count_after_remote_filter": int(len(kept_standard)),
    }


def _filter_remote_depots(depots: Any, seeds: Any, filter_cfg: dict[str, Any], np_mod: Any) -> tuple[Any, Any, dict[str, Any]]:
    depots = depots.copy()
    threshold = float(filter_cfg.get("remote_depot_to_seed_km", 25.0))
    before = _annotate_depot_outliers(depots, seeds, np_mod, threshold)
    remove = bool(filter_cfg.get("remove_remote_depots_after_seed_filter", True))
    if not remove or "is_remote_depot_outlier" not in depots.columns:
        return _standard_columns(depots, DEPOT_COLUMNS), depots.iloc[0:0].copy(), {
            **before,
            "removed_remote_depot_count": 0,
            "depot_candidate_count_before_remote_filter": int(len(depots)),
            "depot_candidate_count_after_remote_filter": int(len(depots)),
        }
    remote = depots[depots["is_remote_depot_outlier"].astype(bool)].copy()
    kept = depots[~depots["is_remote_depot_outlier"].astype(bool)].copy()
    kept_standard = _standard_columns(kept, DEPOT_COLUMNS)
    return kept_standard, remote, {
        "remote_depot_outlier_threshold_km": threshold,
        "remote_depot_outlier_count_before_filter": int(before.get("remote_depot_outlier_count", 0)),
        "removed_remote_depot_count": int(len(remote)),
        "depot_candidate_count_before_remote_filter": int(len(depots)),
        "depot_candidate_count_after_remote_filter": int(len(kept_standard)),
    }


def _filter_manual_depot_exclusions(
    depots: Any,
    territory_id: str,
    filter_cfg: dict[str, Any],
    pd: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    zones = ((filter_cfg.get("manual_depot_exclusions", {}) or {}).get(territory_id, []) or [])
    if depots.empty or not zones:
        return _standard_columns(depots.copy(), DEPOT_COLUMNS), depots.iloc[0:0].copy(), {
            "manual_depot_exclusion_zone_count": int(len(zones)),
            "manual_removed_depot_count": 0,
        }
    remove_mask = pd.Series(False, index=depots.index)
    active_zones = 0
    for zone in zones:
        bbox = zone.get("bbox_lonlat")
        if not bbox or len(bbox) != 4:
            continue
        lon_min, lat_min, lon_max, lat_max = [float(value) for value in bbox]
        zone_mask = depots["lon"].between(lon_min, lon_max) & depots["lat"].between(lat_min, lat_max)
        remove_mask = remove_mask | zone_mask
        active_zones += 1
    removed = depots[remove_mask].copy()
    kept = depots[~remove_mask].copy()
    return _standard_columns(kept, DEPOT_COLUMNS), removed, {
        "manual_depot_exclusion_zone_count": int(active_zones),
        "manual_removed_depot_count": int(len(removed)),
    }


def _latent_customer_count(occupancy_total: float, policy: dict[str, Any]) -> int:
    per_customer = float(policy.get("housing_units_per_latent_customer", 50))
    minimum = int(policy.get("min", 20_000))
    maximum = int(policy.get("max", 50_000))
    raw = occupancy_total / max(per_customer, 1.0)
    rounded = int(round(raw / 1000.0) * 1000)
    return int(max(minimum, min(maximum, rounded)))


def _stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _latent_counts_by_occupancy(seeds: Any, total: int, pd: Any, np_mod: Any) -> Any:
    occupancy = pd.to_numeric(seeds.get("occupancy", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    weights = occupancy.to_numpy(dtype=float)
    positive = weights > 0
    counts = np_mod.zeros(len(seeds), dtype=int)
    if int(total) <= 0 or len(seeds) == 0:
        return counts
    if not positive.any():
        counts[:] = int(total) // max(len(seeds), 1)
        counts[: int(total) - int(counts.sum())] += 1
        return counts
    if int(total) >= int(positive.sum()):
        counts[positive] = 1
        remaining = int(total) - int(counts.sum())
    else:
        remaining = int(total)
    probs = weights / max(float(weights.sum()), 1e-12)
    base = np_mod.floor(probs * remaining).astype(int)
    counts += base
    shortfall = int(total) - int(counts.sum())
    if shortfall > 0:
        frac = probs * remaining - base
        order = np_mod.argsort(-frac, kind="mergesort")
        counts[order[:shortfall]] += 1
    return counts


def _road_edge_table(road_nodes: Any, road_edges: Any, pd: Any) -> Any:
    node_cols = road_nodes[["node_id", "lon", "lat", "x_km", "y_km"]].copy()
    node_cols["node_id"] = node_cols["node_id"].astype(str)
    edges = road_edges.copy()
    edges["u"] = edges["u"].astype(str)
    edges["v"] = edges["v"].astype(str)
    merged = (
        edges.merge(
            node_cols.rename(
                columns={"node_id": "u", "lon": "u_lon", "lat": "u_lat", "x_km": "u_x", "y_km": "u_y"}
            ),
            on="u",
            how="inner",
        )
        .merge(
            node_cols.rename(
                columns={"node_id": "v", "lon": "v_lon", "lat": "v_lat", "x_km": "v_x", "y_km": "v_y"}
            ),
            on="v",
            how="inner",
        )
        .copy()
    )
    merged["length_km"] = pd.to_numeric(merged["length_km"], errors="coerce")
    merged = merged[merged["length_km"].notna() & (merged["length_km"] > 0)].copy()
    return merged.reset_index(drop=True)


def _generate_latent_customers(
    territory_id: str,
    seeds: Any,
    road_nodes: Any,
    road_edges: Any,
    policy: dict[str, Any],
    normalized_dir: Path,
    deps: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    pd = deps["pandas"]
    np_mod = deps["numpy"]
    pyproj = deps["pyproj"]
    latent_total = _latent_customer_count(
        float(pd.to_numeric(seeds.get("occupancy", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()),
        policy,
    )
    edge_table = _road_edge_table(road_nodes, road_edges, pd)
    min_len = float(policy.get("min_road_edge_length_km", 0.015))
    max_len = float(policy.get("max_road_edge_length_km", 0.45))
    edge_table = edge_table[(edge_table["length_km"] >= min_len) & (edge_table["length_km"] <= max_len)].copy()
    if edge_table.empty:
        edge_table = _road_edge_table(road_nodes, road_edges, pd)
    counts = _latent_counts_by_occupancy(seeds, latent_total, pd, np_mod)
    seed_xy = seeds[["x_km", "y_km"]].to_numpy(dtype=float)
    u = edge_table[["u_x", "u_y"]].to_numpy(dtype=float)
    v = edge_table[["v_x", "v_y"]].to_numpy(dtype=float)
    uv = v - u
    seg_len = np_mod.linalg.norm(uv, axis=1)
    unit = uv / np_mod.maximum(seg_len[:, None], 1e-12)
    normal = np_mod.column_stack([-unit[:, 1], unit[:, 0]])
    mid = 0.5 * (u + v)
    edge_lengths = edge_table["length_km"].to_numpy(dtype=float)
    road_radius = float(policy.get("road_radius_km", 0.85))
    sigma = float(policy.get("weight_sigma_km", 0.38))
    min_access = float(policy.get("min_access_km", 0.004))
    max_access = float(policy.get("max_access_km", 0.020))
    rng = np_mod.random.default_rng(_stable_seed(f"{territory_id}:latent_customers:{latent_total}"))
    transformer = pyproj.Transformer.from_crs(PROJECTED_CRS, WGS84_CRS, always_xy=True)
    rows: list[dict[str, Any]] = []

    for seed_idx, count in enumerate(counts.tolist()):
        if count <= 0:
            continue
        center = seed_xy[seed_idx]
        dist = np_mod.linalg.norm(mid - center[None, :], axis=1)
        local_idx = np_mod.flatnonzero(dist <= road_radius)
        if local_idx.size == 0:
            local_idx = np_mod.asarray([int(np_mod.argmin(dist))], dtype=int)
        weights = edge_lengths[local_idx] * np_mod.exp(-0.5 * (dist[local_idx] / max(sigma, 1e-9)) ** 2)
        weights = np_mod.maximum(weights, 1e-12)
        weights = weights / max(float(weights.sum()), 1e-12)
        chosen_local = rng.choice(local_idx, size=int(count), replace=True, p=weights)
        t = rng.uniform(0.02, 0.98, size=int(count))
        side = rng.choice(np_mod.asarray([-1.0, 1.0]), size=int(count))
        access = rng.uniform(min_access, max_access, size=int(count))
        along_noise = rng.normal(0.0, 0.004, size=int(count))
        snap_xy = u[chosen_local] + t[:, None] * uv[chosen_local]
        customer_xy = snap_xy + side[:, None] * access[:, None] * normal[chosen_local] + along_noise[:, None] * unit[chosen_local]
        u_dist = t * seg_len[chosen_local]
        v_dist = (1.0 - t) * seg_len[chosen_local]
        use_u = u_dist <= v_dist
        snap_node_id = np_mod.where(
            use_u,
            edge_table.iloc[chosen_local]["u"].to_numpy(dtype=str),
            edge_table.iloc[chosen_local]["v"].to_numpy(dtype=str),
        )
        connector = access + np_mod.minimum(u_dist, v_dist)
        lon, lat = transformer.transform(customer_xy[:, 0] * 1000.0, customer_xy[:, 1] * 1000.0)
        snap_lon, snap_lat = transformer.transform(snap_xy[:, 0] * 1000.0, snap_xy[:, 1] * 1000.0)
        community_id = str(seeds.iloc[seed_idx]["community_id"])
        occupancy = float(pd.to_numeric(pd.Series([seeds.iloc[seed_idx].get("occupancy", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        for local_pos in range(int(count)):
            edge_row = edge_table.iloc[int(chosen_local[local_pos])]
            rows.append(
                {
                    "customer_id": f"{territory_id}_lc_{len(rows):07d}",
                    "community_id": community_id,
                    "lon": float(lon[local_pos]),
                    "lat": float(lat[local_pos]),
                    "x_km": float(customer_xy[local_pos, 0]),
                    "y_km": float(customer_xy[local_pos, 1]),
                    "snap_lon": float(snap_lon[local_pos]),
                    "snap_lat": float(snap_lat[local_pos]),
                    "snap_x_km": float(snap_xy[local_pos, 0]),
                    "snap_y_km": float(snap_xy[local_pos, 1]),
                    "snap_edge_u": str(edge_row["u"]),
                    "snap_edge_v": str(edge_row["v"]),
                    "snap_node_id": str(snap_node_id[local_pos]),
                    "snap_distance_km": float(access[local_pos]),
                    "connector_distance_km": float(connector[local_pos]),
                    "occupancy_weight": occupancy,
                    "source": "road_frontage_from_community_occupancy",
                }
            )
    latent = pd.DataFrame(rows, columns=LATENT_CUSTOMER_COLUMNS)
    latent.to_csv(normalized_dir / "latent_customer.csv", index=False)
    connector = pd.to_numeric(latent.get("connector_distance_km", pd.Series(dtype=float)), errors="coerce")
    snap = pd.to_numeric(latent.get("snap_distance_km", pd.Series(dtype=float)), errors="coerce")
    return latent, {
        "latent_customer_count": int(len(latent)),
        "latent_customer_policy": dict(policy),
        "latent_customer_housing_units_per_customer": float(policy.get("housing_units_per_latent_customer", 50.0)),
        "latent_customer_snap_p50_km": float(snap.quantile(0.50)) if len(snap) else float("nan"),
        "latent_customer_snap_p90_km": float(snap.quantile(0.90)) if len(snap) else float("nan"),
        "latent_customer_connector_p50_km": float(connector.quantile(0.50)) if len(connector) else float("nan"),
        "latent_customer_connector_p90_km": float(connector.quantile(0.90)) if len(connector) else float("nan"),
    }


def _territory_area_km2(polygon_proj: Any) -> float:
    return float(polygon_proj.area / 1_000_000.0)


def _write_preview_layers(
    path: Path,
    boundary_wgs: Any,
    seeds: Any,
    chargers: Any,
    depots: Any,
    child: dict[str, Any],
    deps: dict[str, Any],
) -> None:
    gpd = deps["geopandas"]
    pd = deps["pandas"]
    frames = [
        gpd.GeoDataFrame(
            [{"layer": "service_territory", "territory_id": child["territory_id"]}],
            geometry=[boundary_wgs],
            crs=WGS84_CRS,
        )
    ]
    if not seeds.empty:
        seed_props = seeds[["community_id", "occupancy"]].copy()
        seed_props["layer"] = "customer_seed"
        seed_props["territory_id"] = child["territory_id"]
        frames.append(
            gpd.GeoDataFrame(
                seed_props,
                geometry=gpd.points_from_xy(seeds["lon"], seeds["lat"]),
                crs=WGS84_CRS,
            )
        )
    if not chargers.empty:
        charger_cols = ["station_id", "name", "network", "level2_count", "dc_fast_count", "status", "access"]
        charger_props = chargers[[col for col in charger_cols if col in chargers.columns]].copy()
        charger_props["layer"] = "charging_station"
        charger_props["territory_id"] = child["territory_id"]
        frames.append(
            gpd.GeoDataFrame(
                charger_props,
                geometry=gpd.points_from_xy(chargers["lon"], chargers["lat"]),
                crs=WGS84_CRS,
            )
        )
    if not depots.empty:
        depot_props = depots[[col for col in ["candidate_id", "source", "category"] if col in depots.columns]].copy()
        depot_props["layer"] = "depot_candidate"
        depot_props["territory_id"] = child["territory_id"]
        frames.append(
            gpd.GeoDataFrame(
                depot_props,
                geometry=gpd.points_from_xy(depots["lon"], depots["lat"]),
                crs=WGS84_CRS,
            )
        )
    ensure_dir(path.parent)
    pd.concat(frames, ignore_index=True).to_file(path, driver="GeoJSON")


def _write_qa_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# Geo-AC-v1 Subterritory QA: {summary['territory_id']}",
        "",
        f"- Parent territory: `{summary.get('parent_territory_id', 'unchanged')}`",
        f"- Split method: `{summary.get('split_method', 'unchanged')}`",
        f"- Area: {summary.get('territory_area_km2', 0):.1f} km2",
        f"- Occupied housing units: {summary.get('occupancy_total', 0):.0f}",
        f"- Customer seeds: {summary.get('customer_seed_count', 0)}",
        f"- Latent customers: {summary.get('latent_customer_count', summary.get('latent_customer_pool_size', 0))}",
        f"- Removed sparse remote customer seeds: {summary.get('removed_remote_seed_count', 0)}",
        f"- Road nodes / edges: {summary.get('road_node_count', 0)} / {summary.get('road_edge_count', 0)}",
        f"- Public charging stations: {summary.get('charging_station_count', 0)}",
        f"- Depot candidates: {summary.get('depot_candidate_count', 0)}",
        f"- Removed remote depot candidates: {summary.get('removed_remote_depot_count', 0)}",
        f"- Removed manual-QA depot candidates: {summary.get('manual_removed_depot_count', 0)}",
        f"- Removed depot candidates total: {summary.get('removed_depot_count_total', 0)}",
        f"- Fallback depot candidates added: {summary.get('fallback_depot_count', 0)}",
        f"- Latent customer pool size: {summary.get('latent_customer_pool_size', 0)}",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _base_child_spec(
    parent_spec: dict[str, Any],
    child: dict[str, Any],
    child_root: Path,
    config_out: Path,
    summary: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    spec = dict(parent_spec)
    spec["territory_id"] = child["territory_id"]
    spec["display_name"] = child.get("display_name", child["territory_id"])
    spec["service_area_name"] = child.get("service_area_name", child["territory_id"])
    spec["parent_territory_id"] = parent_spec["territory_id"]
    spec["split_method"] = "anchored_voronoi_clipped_to_county"
    spec["anchor_lonlat"] = [float(child["anchor_lonlat"][0]), float(child["anchor_lonlat"][1])]
    spec["data_root"] = _relative_or_absolute(child_root.resolve(), config_out.parent.resolve())
    spec["source_files"] = {**REQUIRED_CSVS, LATENT_CUSTOMER_SOURCE_KEY: LATENT_CUSTOMER_REL}
    spec["latent_customer_pool_size"] = int(
        summary.get("latent_customer_count", _latent_customer_count(float(summary.get("occupancy_total", 0.0)), policy))
    )
    spec["cs_candidate_pool_size"] = int(max(40, min(160, summary.get("charging_station_count", 0))))
    spec["depot_candidate_count"] = int(child.get("depot_candidate_count", parent_spec.get("depot_candidate_count", 8)))
    spec["subterritory_source"] = {
        "parent_territory_id": parent_spec["territory_id"],
        "split_method": "anchored_voronoi_clipped_to_county",
        "anchor_lonlat": spec["anchor_lonlat"],
        "mainland_only": bool(child.get("mainland_only", False)),
    }
    filters = dict(spec.get("data_filters", {}) or {})
    filters["service_territory_split"] = "county_clipped_anchor_voronoi"
    spec["data_filters"] = filters
    return spec


def _slice_child(
    parent_spec: dict[str, Any],
    source_config: Path,
    child: dict[str, Any],
    output_root: Path,
    config_out: Path,
    policy: dict[str, Any],
    filter_cfg: dict[str, Any],
    deps: dict[str, Any],
    force: bool,
) -> tuple[dict[str, Any], dict[str, Any], Any, Any, Any, Any, Any]:
    pd = deps["pandas"]
    np_mod = deps["numpy"]
    child_root = output_root / child["territory_id"]
    normalized_dir = ensure_dir(child_root / "normalized")
    qa_dir = ensure_dir(child_root / "qa")
    if any((normalized_dir / name).exists() for name in ["road_nodes.csv", "road_edges.csv"]) and not force:
        raise FileExistsError(f"{normalized_dir} already exists. Re-run with --force to overwrite.")

    road_nodes, road_edges, road_summary = _clip_roads(parent_spec, source_config, child["geometry_wgs84"], normalized_dir, deps)
    seeds = _clip_point_csv(
        parent_spec,
        source_config,
        "customer_seed_csv",
        "customer_seed.csv",
        child["geometry_wgs84"],
        normalized_dir,
        deps,
    )
    chargers = _clip_point_csv(
        parent_spec,
        source_config,
        "charging_station_csv",
        "charging_station.csv",
        child["geometry_wgs84"],
        normalized_dir,
        deps,
    )
    depots = _clip_point_csv(
        parent_spec,
        source_config,
        "depot_candidate_csv",
        "depot_candidate.csv",
        child["geometry_wgs84"],
        normalized_dir,
        deps,
    )
    seeds, removed_seeds, seed_filter_summary = _filter_sparse_remote_seeds(seeds, filter_cfg, np_mod)
    seeds.to_csv(normalized_dir / "customer_seed.csv", index=False)
    latent_customers, latent_summary = _generate_latent_customers(
        child["territory_id"],
        seeds,
        road_nodes,
        road_edges,
        policy,
        normalized_dir,
        deps,
    )
    depots, removed_depots_pre_fallback, depot_filter_summary = _filter_remote_depots(depots, seeds, filter_cfg, np_mod)
    depots, removed_manual_depots_pre_fallback, manual_depot_summary = _filter_manual_depot_exclusions(
        depots,
        child["territory_id"],
        filter_cfg,
        pd,
    )
    fallback_count = 0
    if bool(filter_cfg.get("replenish_depots_after_filter", True)):
        depots, fallback_count = _append_fallback_depots(
            depots,
            road_nodes,
            child,
            int(child.get("depot_candidate_count", parent_spec.get("depot_candidate_count", 8))),
            pd,
            np_mod,
            seeds,
            float(filter_cfg.get("remote_depot_to_seed_km", 25.0)),
        )
    depots, removed_depots_post_fallback, depot_post_filter_summary = _filter_remote_depots(depots, seeds, filter_cfg, np_mod)
    depots, removed_manual_depots_post_fallback, manual_depot_post_summary = _filter_manual_depot_exclusions(
        depots,
        child["territory_id"],
        filter_cfg,
        pd,
    )
    removed_remote_depots = pd.concat([removed_depots_pre_fallback, removed_depots_post_fallback], ignore_index=True)
    removed_manual_depots = pd.concat(
        [removed_manual_depots_pre_fallback, removed_manual_depots_post_fallback],
        ignore_index=True,
    )
    depots.to_csv(normalized_dir / "depot_candidate.csv", index=False)
    seed_outlier_summary = _annotate_seed_outliers(seeds, np_mod, float(filter_cfg.get("remote_seed_nearest_neighbor_km", 15.0)))
    depot_outlier_summary = _annotate_depot_outliers(depots, seeds, np_mod, float(filter_cfg.get("remote_depot_to_seed_km", 25.0)))
    occupancy_total = float(pd.to_numeric(seeds.get("occupancy", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    summary = {
        "territory_id": child["territory_id"],
        "display_name": child.get("display_name", child["territory_id"]),
        "parent_territory_id": parent_spec["territory_id"],
        "split_method": "anchored_voronoi_clipped_to_county",
        "mainland_only": bool(child.get("mainland_only", False)),
        "anchor_lonlat": [float(child["anchor_lonlat"][0]), float(child["anchor_lonlat"][1])],
        "territory_area_km2": _territory_area_km2(child["geometry_projected"]),
        "occupancy_total": occupancy_total,
        "positive_occupancy_seed_count": int((pd.to_numeric(seeds.get("occupancy", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum()),
        "charging_station_count": int(len(chargers)),
        "fallback_depot_count": int(fallback_count),
        "latent_customer_pool_size": int(len(latent_customers)),
        **latent_summary,
        **seed_filter_summary,
        **depot_filter_summary,
        "depot_candidate_count_after_post_fallback_filter": int(len(depots)),
        "manual_depot_exclusion_zone_count": int(manual_depot_summary.get("manual_depot_exclusion_zone_count", 0)),
        "manual_removed_depot_count": int(len(removed_manual_depots)),
        "removed_depot_count_total": int(len(removed_remote_depots) + len(removed_manual_depots)),
        **seed_outlier_summary,
        **depot_outlier_summary,
        **road_summary,
        "customer_seed_count": int(len(seeds)),
        "removed_remote_seed_count": int(len(removed_seeds)),
        "depot_candidate_count": int(len(depots)),
        "removed_remote_depot_count": int(len(removed_remote_depots)),
    }
    _write_json(qa_dir / "qa_summary.json", summary)
    _write_qa_report(qa_dir / "qa_report.md", summary)
    _write_preview_layers(qa_dir / "preview_layers.geojson", child["geometry_wgs84"], seeds, chargers, depots, child, deps)
    spec = _base_child_spec(parent_spec, child, child_root, config_out, summary, policy)
    return spec, summary, seeds, chargers, depots, road_nodes, road_edges


def _copy_unchanged(
    spec: dict[str, Any],
    source_config: Path,
    output_root: Path,
    config_out: Path,
    policy: dict[str, Any],
    filter_cfg: dict[str, Any],
    deps: dict[str, Any],
    force: bool,
) -> tuple[dict[str, Any], dict[str, Any], Any, Any, Any, Any, Any]:
    pd = deps["pandas"]
    np_mod = deps["numpy"]
    parent_root = _source_root(spec, source_config)
    child_root = output_root / spec["territory_id"]
    normalized_dir = ensure_dir(child_root / "normalized")
    qa_dir = ensure_dir(child_root / "qa")
    if any((normalized_dir / path.name).exists() for path in (parent_root / "normalized").glob("*.csv")) and not force:
        raise FileExistsError(f"{normalized_dir} already exists. Re-run with --force to overwrite.")
    for key, rel in REQUIRED_CSVS.items():
        shutil.copy2(_source_file(spec, source_config, key), child_root / rel)
    for qa_name in ["preview_layers.geojson", "qa_report.md"]:
        source = parent_root / "qa" / qa_name
        if source.exists():
            shutil.copy2(source, qa_dir / qa_name)
    source_summary_path = parent_root / "qa" / "qa_summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8")) if source_summary_path.exists() else {}
    seeds = pd.read_csv(child_root / REQUIRED_CSVS["customer_seed_csv"])
    chargers = pd.read_csv(child_root / REQUIRED_CSVS["charging_station_csv"])
    depots = pd.read_csv(child_root / REQUIRED_CSVS["depot_candidate_csv"])
    road_nodes = _read_nodes(child_root / REQUIRED_CSVS["road_nodes_csv"], pd)
    road_edges = _read_edges(child_root / REQUIRED_CSVS["road_edges_csv"], pd)
    seeds, removed_seeds, seed_filter_summary = _filter_sparse_remote_seeds(seeds, filter_cfg, np_mod)
    seeds.to_csv(child_root / REQUIRED_CSVS["customer_seed_csv"], index=False)
    latent_customers, latent_summary = _generate_latent_customers(
        spec["territory_id"],
        seeds,
        road_nodes,
        road_edges,
        policy,
        normalized_dir,
        deps,
    )
    depots, removed_depots_pre_fallback, depot_filter_summary = _filter_remote_depots(depots, seeds, filter_cfg, np_mod)
    depots, removed_manual_depots_pre_fallback, manual_depot_summary = _filter_manual_depot_exclusions(
        depots,
        spec["territory_id"],
        filter_cfg,
        pd,
    )
    fallback_count = 0
    if bool(filter_cfg.get("replenish_depots_after_filter", True)):
        fallback_child = {
            "anchor_x_km": float(pd.to_numeric(seeds["x_km"], errors="coerce").median()) if not seeds.empty else 0.0,
            "anchor_y_km": float(pd.to_numeric(seeds["y_km"], errors="coerce").median()) if not seeds.empty else 0.0,
        }
        depots, fallback_count = _append_fallback_depots(
            depots,
            road_nodes,
            fallback_child,
            int(spec.get("depot_candidate_count", 8)),
            pd,
            np_mod,
            seeds,
            float(filter_cfg.get("remote_depot_to_seed_km", 25.0)),
        )
    depots, removed_depots_post_fallback, depot_post_filter_summary = _filter_remote_depots(depots, seeds, filter_cfg, np_mod)
    depots, removed_manual_depots_post_fallback, manual_depot_post_summary = _filter_manual_depot_exclusions(
        depots,
        spec["territory_id"],
        filter_cfg,
        pd,
    )
    removed_remote_depots = pd.concat([removed_depots_pre_fallback, removed_depots_post_fallback], ignore_index=True)
    removed_manual_depots = pd.concat(
        [removed_manual_depots_pre_fallback, removed_manual_depots_post_fallback],
        ignore_index=True,
    )
    depots.to_csv(child_root / REQUIRED_CSVS["depot_candidate_csv"], index=False)
    seed_outlier_summary = _annotate_seed_outliers(seeds, np_mod, float(filter_cfg.get("remote_seed_nearest_neighbor_km", 15.0)))
    depot_outlier_summary = _annotate_depot_outliers(depots, seeds, np_mod, float(filter_cfg.get("remote_depot_to_seed_km", 25.0)))
    occupancy_total = float(pd.to_numeric(seeds.get("occupancy", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    summary = {
        **source_summary,
        "territory_id": spec["territory_id"],
        "split_method": "unchanged_county_container",
        "occupancy_total": occupancy_total,
        "charging_station_count": int(len(chargers)),
        "fallback_depot_count": int(fallback_count),
        "road_node_count": int(len(road_nodes)),
        "road_edge_count": int(len(road_edges)),
        "latent_customer_pool_size": int(len(latent_customers)),
        **latent_summary,
        **seed_filter_summary,
        **depot_filter_summary,
        "depot_candidate_count_after_post_fallback_filter": int(len(depots)),
        "manual_depot_exclusion_zone_count": int(manual_depot_summary.get("manual_depot_exclusion_zone_count", 0)),
        "manual_removed_depot_count": int(len(removed_manual_depots)),
        "removed_depot_count_total": int(len(removed_remote_depots) + len(removed_manual_depots)),
        **seed_outlier_summary,
        **depot_outlier_summary,
        "customer_seed_count": int(len(seeds)),
        "removed_remote_seed_count": int(len(removed_seeds)),
        "depot_candidate_count": int(len(depots)),
        "removed_remote_depot_count": int(len(removed_remote_depots)),
    }
    _write_json(qa_dir / "qa_summary.json", summary)
    out_spec = dict(spec)
    out_spec["data_root"] = _relative_or_absolute(child_root.resolve(), config_out.parent.resolve())
    out_spec["source_files"] = {**REQUIRED_CSVS, LATENT_CUSTOMER_SOURCE_KEY: LATENT_CUSTOMER_REL}
    out_spec["latent_customer_pool_size"] = summary["latent_customer_pool_size"]
    county_wgs = _read_county_boundary(parent_root, deps["geopandas"])
    _write_preview_layers(
        qa_dir / "preview_layers.geojson",
        county_wgs.geometry.iloc[0],
        seeds,
        chargers,
        depots,
        {"territory_id": spec["territory_id"]},
        deps,
    )
    return out_spec, summary, seeds, chargers, depots, road_nodes, road_edges


def _road_segments(road_nodes: Any, road_edges: Any, max_edges: int) -> list[list[tuple[float, float]]]:
    if road_nodes is None or road_edges is None or road_nodes.empty or road_edges.empty:
        return []
    edges = road_edges[["u", "v"]].dropna().copy()
    if len(edges) > max_edges:
        edges = edges.sample(n=int(max_edges), random_state=20260526).sort_index()
    coords = road_nodes[["node_id", "lon", "lat"]].copy()
    coords["node_id"] = coords["node_id"].astype(str)
    coords = coords.set_index("node_id")
    merged = (
        edges.astype(str)
        .merge(coords.rename(columns={"lon": "u_lon", "lat": "u_lat"}), left_on="u", right_index=True, how="inner")
        .merge(coords.rename(columns={"lon": "v_lon", "lat": "v_lat"}), left_on="v", right_index=True, how="inner")
    )
    if merged.empty:
        return []
    return [
        [(float(row.u_lon), float(row.u_lat)), (float(row.v_lon), float(row.v_lat))]
        for row in merged.itertuples(index=False)
    ]


def _plot_roads(ax: Any, item: dict[str, Any], max_edges: int, label: str | None = None) -> bool:
    segments = _road_segments(item.get("road_nodes"), item.get("road_edges"), max_edges)
    if not segments:
        return False
    LineCollection = __import__("matplotlib.collections", fromlist=["LineCollection"]).LineCollection
    lines = LineCollection(
        segments,
        colors=ROAD_COLOR,
        linewidths=0.18,
        alpha=0.24,
        zorder=2,
        label=label,
    )
    ax.add_collection(lines)
    return True


def _plot_group_map(
    parent_id: str,
    county_wgs: Any,
    child_items: list[dict[str, Any]],
    map_dir: Path,
    deps: dict[str, Any],
) -> Path:
    gpd = deps["geopandas"]
    plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(12, 10))
    county_wgs.boundary.plot(ax=ax, color="#202020", linewidth=1.0)
    seen_roads = False
    seen_seed = False
    seen_chargers = False
    seen_real_depot = False
    seen_fallback_depot = False
    seen_remote_seed = False
    seen_remote_depot = False
    for idx, item in enumerate(child_items):
        color = colors[idx % len(colors)]
        gpd.GeoSeries([item["boundary_wgs"]], crs=WGS84_CRS).plot(ax=ax, color=color, alpha=0.16, edgecolor=color, linewidth=1.5)
        if _plot_roads(ax, item, max_edges=25_000, label=None if seen_roads else "Road network sample"):
            seen_roads = True
        chargers = item.get("chargers")
        if chargers is not None and not chargers.empty:
            ax.scatter(
                chargers["lon"],
                chargers["lat"],
                marker="o",
                s=13,
                color="#2ca25f",
                edgecolor="white",
                linewidth=0.25,
                alpha=0.78,
                label=None if seen_chargers else "Public charging station",
                zorder=4,
            )
            seen_chargers = True
        if not item["seeds"].empty:
            ax.scatter(
                item["seeds"]["lon"],
                item["seeds"]["lat"],
                s=3,
                color=COMMUNITY_SEED_COLOR,
                alpha=0.62,
                linewidths=0,
                label=None if seen_seed else "Community seeds",
                zorder=3,
            )
            seen_seed = True
            if "is_remote_seed_outlier" in item["seeds"].columns:
                remote_seeds = item["seeds"][item["seeds"]["is_remote_seed_outlier"].astype(bool)]
                if not remote_seeds.empty:
                    ax.scatter(
                        remote_seeds["lon"],
                        remote_seeds["lat"],
                        marker="x",
                        s=36,
                        color=REMOTE_SEED_COLOR,
                        linewidth=0.9,
                        label=None if seen_remote_seed else "Sparse remote seed",
                        zorder=6,
                    )
                    seen_remote_seed = True
        if not item["depots"].empty:
            source = item["depots"].get("source", "")
            fallback = source.astype(str).str.contains("fallback", case=False, na=False)
            real_depots = item["depots"][~fallback]
            fallback_depots = item["depots"][fallback]
            if not real_depots.empty:
                ax.scatter(
                    real_depots["lon"],
                    real_depots["lat"],
                    marker="^",
                    s=70,
                    color="#d62728",
                    edgecolor="white",
                    linewidth=0.6,
                    alpha=0.98,
                    label=None if seen_real_depot else "Open depot candidate",
                    zorder=5,
                )
                seen_real_depot = True
            if not fallback_depots.empty:
                ax.scatter(
                    fallback_depots["lon"],
                    fallback_depots["lat"],
                    marker="^",
                    s=70,
                    color="#ff9f1c",
                    edgecolor="#111111",
                    linewidth=0.6,
                    alpha=0.98,
                    label=None if seen_fallback_depot else "Fallback depot candidate",
                    zorder=5,
                )
                seen_fallback_depot = True
            if "is_remote_depot_outlier" in item["depots"].columns:
                remote_depots = item["depots"][item["depots"]["is_remote_depot_outlier"].astype(bool)]
                if not remote_depots.empty:
                    ax.scatter(
                        remote_depots["lon"],
                        remote_depots["lat"],
                        marker="o",
                        s=155,
                        facecolors="none",
                        edgecolors="#111111",
                        linewidth=1.2,
                        label=None if seen_remote_depot else "Depot far from seeds",
                        zorder=6,
                    )
                    seen_remote_depot = True
        label_point = gpd.GeoSeries([item["boundary_wgs"]], crs=WGS84_CRS).representative_point().iloc[0]
        ax.text(label_point.x, label_point.y, item["service_area_name"], fontsize=8, ha="center", va="center")
    ax.set_title(f"{parent_id}: sliced service territories, roads, chargers, depots, community seeds")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    ensure_dir(map_dir)
    path = map_dir / f"{parent_id}_sliced_overview.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_single_map(
    item: dict[str, Any],
    map_dir: Path,
    deps: dict[str, Any],
) -> Path:
    gpd = deps["geopandas"]
    plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
    fig, ax = plt.subplots(figsize=(9, 8))
    gpd.GeoSeries([item["boundary_wgs"]], crs=WGS84_CRS).plot(ax=ax, color="#d8ecff", alpha=0.55, edgecolor="#24517a", linewidth=1.4)
    _plot_roads(ax, item, max_edges=70_000, label="Road network sample")
    chargers = item.get("chargers")
    if chargers is not None and not chargers.empty:
        size = 24 if len(chargers) < 500 else 13
        ax.scatter(
            chargers["lon"],
            chargers["lat"],
            marker="o",
            s=size,
            color="#2ca25f",
            edgecolor="white",
            linewidth=0.35,
            alpha=0.78,
            label=f"Public charging stations (n={len(chargers)})",
            zorder=4,
        )
    if not item["seeds"].empty:
        size = 4 if len(item["seeds"]) < 500 else 2
        ax.scatter(
            item["seeds"]["lon"],
            item["seeds"]["lat"],
            s=size,
            color=COMMUNITY_SEED_COLOR,
            alpha=0.66,
            linewidths=0,
            label=f"Community seeds (n={len(item['seeds'])})",
            zorder=3,
        )
        if "is_remote_seed_outlier" in item["seeds"].columns:
            remote_seeds = item["seeds"][item["seeds"]["is_remote_seed_outlier"].astype(bool)]
            if not remote_seeds.empty:
                ax.scatter(
                    remote_seeds["lon"],
                    remote_seeds["lat"],
                    marker="x",
                    s=42,
                    color=REMOTE_SEED_COLOR,
                    linewidth=1.0,
                    label=f"Sparse remote seeds (n={len(remote_seeds)})",
                    zorder=6,
                )
    if not item["depots"].empty:
        source = item["depots"].get("source", "")
        fallback = source.astype(str).str.contains("fallback", case=False, na=False)
        real_depots = item["depots"][~fallback]
        fallback_depots = item["depots"][fallback]
        if not real_depots.empty:
            ax.scatter(
                real_depots["lon"],
                real_depots["lat"],
                marker="^",
                s=95,
                color="#d62728",
                edgecolor="white",
                linewidth=0.7,
                label=f"Open depot candidates (n={len(real_depots)})",
                zorder=5,
            )
        if not fallback_depots.empty:
            ax.scatter(
                fallback_depots["lon"],
                fallback_depots["lat"],
                marker="^",
                s=95,
                color="#ff9f1c",
                edgecolor="#111111",
                linewidth=0.7,
                label=f"Fallback depot candidates (n={len(fallback_depots)})",
                zorder=5,
            )
        if "is_remote_depot_outlier" in item["depots"].columns:
            remote_depots = item["depots"][item["depots"]["is_remote_depot_outlier"].astype(bool)]
            if not remote_depots.empty:
                ax.scatter(
                    remote_depots["lon"],
                    remote_depots["lat"],
                    marker="o",
                    s=190,
                    facecolors="none",
                    edgecolors="#111111",
                    linewidth=1.3,
                    label=f"Depot far from seeds (n={len(remote_depots)})",
                    zorder=6,
                )
    ax.set_title(f"{item['territory_id']}: roads, chargers, depot candidates, community seeds")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    ensure_dir(map_dir)
    path = map_dir / f"{item['territory_id']}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_contact_sheet(image_paths: list[Path], output_path: Path, title: str, columns: int, deps: dict[str, Any]) -> Path:
    if not image_paths:
        return output_path
    plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
    rows = int(math.ceil(len(image_paths) / max(columns, 1)))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 5.2, rows * 4.2))
    flat_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, path in zip(flat_axes, image_paths):
        ax.imshow(plt.imread(path))
        ax.set_title(path.stem, fontsize=8)
        ax.axis("off")
    for ax in flat_axes[len(image_paths) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def _write_map_gallery(map_dir: Path, qa_rows: list[dict[str, Any]]) -> Path:
    overview_maps = sorted(path for path in map_dir.glob("*_sliced_overview.png") if "contact_sheet" not in path.name)
    territory_maps = sorted(
        path
        for path in map_dir.glob("*.png")
        if not path.name.endswith("_sliced_overview.png") and "contact_sheet" not in path.name
    )
    summary_by_id = {str(row.get("territory_id")): row for row in qa_rows}

    def card(path: Path) -> str:
        stem = path.stem
        summary = summary_by_id.get(stem, {})
        bits = []
        if summary:
            bits = [
                f"seeds={int(summary.get('customer_seed_count', 0))}",
                f"latent={int(summary.get('latent_customer_count', summary.get('latent_customer_pool_size', 0)))}",
                f"removed_seeds={int(summary.get('removed_remote_seed_count', 0))}",
                f"depots={int(summary.get('depot_candidate_count', 0))}",
                f"removed_depots={int(summary.get('removed_remote_depot_count', 0))}",
                f"manual_removed={int(summary.get('manual_removed_depot_count', 0))}",
                f"removed_total={int(summary.get('removed_depot_count_total', 0))}",
                f"fallback={int(summary.get('fallback_depot_count', 0))}",
                f"chargers={int(summary.get('charging_station_count', 0))}",
            ]
        caption = " | ".join(bits)
        return (
            '<figure class="card">'
            f'<a href="{html.escape(path.name)}"><img src="{html.escape(path.name)}" alt="{html.escape(stem)}"></a>'
            f"<figcaption><strong>{html.escape(stem)}</strong><br>{html.escape(caption)}</figcaption>"
            "</figure>"
        )

    body = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>Geo-AC-v1 NA-US-20 Map QA Gallery</title>",
            "<style>",
            "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;color:#1f2933}",
            ".legend{display:flex;gap:18px;align-items:center;margin:12px 0 24px}",
            ".swatch{display:inline-block;width:14px;height:14px;clip-path:polygon(50% 0,0 100%,100% 100%);margin-right:6px;vertical-align:-2px}",
            ".open{background:#d62728}.fallback{background:#ff9f1c;border:1px solid #111}",
            ".dot{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:6px;vertical-align:-1px}.charger{background:#2ca25f}",
            ".seed{background:#2563eb}",
            ".line{display:inline-block;width:28px;height:0;border-top:2px solid #59616a;margin-right:6px;vertical-align:4px}",
            ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}",
            ".card{border:1px solid #d8dee9;border-radius:8px;padding:10px;margin:0;background:#fff}",
            ".card img{width:100%;height:auto;display:block;border:1px solid #edf2f7}",
            "figcaption{font-size:13px;line-height:1.4;margin-top:8px}",
            "h1,h2{margin-bottom:8px}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Geo-AC-v1 NA-US-20 Map QA Gallery</h1>",
            '<div class="legend"><span><span class="swatch open"></span>Open depot candidate</span>'
            '<span><span class="swatch fallback"></span>Fallback depot candidate</span>'
            '<span><span class="dot charger"></span>Public charging station</span>'
            '<span><span class="dot seed"></span>Community seed</span>'
            '<span><span class="line"></span>Road network sample</span>'
            "<span>Dark red x: sparse remote seed</span>"
            "<span>Black ring: depot far from seeds</span></div>",
            "<h2>County Split Overviews</h2>",
            '<div class="grid">',
            "\n".join(card(path) for path in overview_maps),
            "</div>",
            "<h2>Individual Territory Maps</h2>",
            '<div class="grid">',
            "\n".join(card(path) for path in territory_maps),
            "</div>",
            "</body>",
            "</html>",
        ]
    )
    path = map_dir / "index.html"
    path.write_text(body, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    deps = _require_deps()
    yaml_mod = deps["yaml"]
    source_config = args.source_config.resolve()
    slice_config = args.slice_config.resolve()
    output_root = ensure_dir(args.output_root.resolve())
    config_out = args.config_out.resolve()
    source_cfg = load_yaml(source_config)
    split_cfg = load_yaml(slice_config)
    source_specs = {str(spec["territory_id"]): spec for spec in source_cfg.get("territories", [])}
    policy = split_cfg.get("latent_customer_policy", {}) or {}
    filter_cfg = split_cfg.get("outlier_filters", {}) or {}
    territories: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    maps: list[Path] = []
    map_dir = ensure_dir(output_root / "qa_maps")

    split_parent_ids = {str(group["parent_territory_id"]) for group in split_cfg.get("slice_groups", [])}
    for group in split_cfg.get("slice_groups", []):
        parent_id = str(group["parent_territory_id"])
        if parent_id not in source_specs:
            raise ValueError(f"Slice parent {parent_id} not found in {source_config}.")
        parent_spec = source_specs[parent_id]
        parent_root = _source_root(parent_spec, source_config)
        county_wgs = _read_county_boundary(parent_root, deps["geopandas"])
        if bool(group.get("mainland_only", False)):
            county_wgs = _largest_polygon_component(county_wgs, deps["geopandas"])
        cells = _apply_component_reassignments(_anchor_cells(county_wgs, group.get("children", []), deps), group, deps)
        child_items = []
        for child in cells:
            child["mainland_only"] = bool(group.get("mainland_only", False))
            spec, summary, seeds, chargers, depots, road_nodes, road_edges = _slice_child(
                parent_spec,
                source_config,
                child,
                output_root,
                config_out,
                policy,
                filter_cfg,
                deps,
                args.force,
            )
            territories.append(spec)
            qa_rows.append(summary)
            item = {
                "territory_id": child["territory_id"],
                "service_area_name": child.get("service_area_name", child["territory_id"]),
                "boundary_wgs": child["geometry_wgs84"],
                "seeds": seeds,
                "chargers": chargers,
                "depots": depots,
                "road_nodes": road_nodes,
                "road_edges": road_edges,
            }
            child_items.append(item)
            if not args.skip_maps:
                maps.append(_plot_single_map(item, map_dir, deps))
            print(json.dumps({"territory": child["territory_id"], "normalized_dir": str(output_root / child["territory_id"] / "normalized")}, sort_keys=True))
        if not args.skip_maps:
            maps.append(_plot_group_map(parent_id, county_wgs, child_items, map_dir, deps))

    for territory_id in split_cfg.get("unchanged_territories", []):
        territory_id = str(territory_id)
        if territory_id in split_parent_ids:
            continue
        if territory_id not in source_specs:
            raise ValueError(f"Unchanged territory {territory_id} not found in {source_config}.")
        spec, summary, seeds, chargers, depots, road_nodes, road_edges = _copy_unchanged(
            source_specs[territory_id],
            source_config,
            output_root,
            config_out,
            policy,
            filter_cfg,
            deps,
            args.force,
        )
        territories.append(spec)
        qa_rows.append(summary)
        if not args.skip_maps:
            parent_root = _source_root(source_specs[territory_id], source_config)
            county_wgs = _read_county_boundary(parent_root, deps["geopandas"])
            item = {
                "territory_id": territory_id,
                "service_area_name": spec.get("display_name", territory_id),
                "boundary_wgs": county_wgs.geometry.iloc[0],
                "seeds": seeds,
                "chargers": chargers,
                "depots": depots,
                "road_nodes": road_nodes,
                "road_edges": road_edges,
            }
            maps.append(_plot_single_map(item, map_dir, deps))
        print(json.dumps({"territory": territory_id, "normalized_dir": str(output_root / territory_id / "normalized")}, sort_keys=True))

    out_cfg = dict(source_cfg)
    out_cfg["profile_name"] = split_cfg.get("profile_name", "Geo-AC-v1 / NA-US-20")
    out_cfg["dataset_version"] = split_cfg.get("dataset_version", source_cfg.get("dataset_version", "Geo-AC-v1"))
    out_cfg["territories"] = territories
    out_cfg["territory_split"] = {
        "source_config": _relative_or_absolute(source_config, config_out.parent),
        "slice_config": _relative_or_absolute(slice_config, config_out.parent),
        "split_method": split_cfg.get("split_method", "anchored_voronoi_clipped_to_county"),
        "territory_count": len(territories),
        "latent_customer_policy": policy,
        "outlier_filters": filter_cfg,
    }
    _write_yaml(config_out, out_cfg, yaml_mod)
    gallery_path = None
    if not args.skip_maps:
        overview_maps = sorted(path for path in map_dir.glob("*_sliced_overview.png") if "contact_sheet" not in path.name)
        territory_maps = sorted(
            path
            for path in map_dir.glob("*.png")
            if not path.name.endswith("_sliced_overview.png") and "contact_sheet" not in path.name
        )
        maps.append(
            _plot_contact_sheet(
                overview_maps,
                map_dir / "all_sliced_overviews_contact_sheet.png",
                "Geo-AC-v1 NA-US-20 county split overviews",
                2,
                deps,
            )
        )
        maps.append(
            _plot_contact_sheet(
                territory_maps,
                map_dir / "all_territory_maps_contact_sheet.png",
                "Geo-AC-v1 NA-US-20 individual service territories",
                4,
                deps,
            )
        )
        gallery_path = _write_map_gallery(map_dir, qa_rows)
    summary_path = output_root / "na_us20_summary.json"
    _write_json(
        summary_path,
        {
            "profile_name": out_cfg["profile_name"],
            "territory_count": len(territories),
            "territories": qa_rows,
            "maps": [str(path) for path in maps],
            "map_gallery": str(gallery_path) if gallery_path else None,
            "config": str(config_out),
        },
    )
    print(json.dumps({"config_out": str(config_out), "summary": str(summary_path), "territory_count": len(territories)}, indent=2))


if __name__ == "__main__":
    main()
