from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import shapely
from shapely.geometry import mapping

from evrptw_cle.nsi import _as_bool, _parse_multivalue

MAJOR_ROADS = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
}


def _line_coordinates(geometry) -> list[list[list[float]]]:
    if geometry.geom_type == "LineString":
        return [[[round(float(x), 5), round(float(y), 5)] for x, y in geometry.coords]]
    if geometry.geom_type == "MultiLineString":
        return [
            [[round(float(x), 5), round(float(y), 5)] for x, y in line.coords]
            for line in geometry.geoms
        ]
    return []


def build_payload(
    customer_dir: Path,
    graph_file: Path,
    boundary_file: Path,
) -> dict:
    manifest = json.loads(
        (customer_dir / "customer_cle_manifest.json").read_text(encoding="utf-8")
    )
    density = gpd.read_file(customer_dir / "latent_service_density_500m.geojson")
    points = density.geometry.representative_point()
    cells = []
    for row, point in zip(density.itertuples(index=False), points, strict=True):
        cells.append(
            [
                round(float(point.x), 5),
                round(float(point.y), 5),
                int(row.service_location_count),
                int(row.house),
                int(row.manufactured_home),
                int(row.small_apt),
                int(row.medium_apt),
                int(row.large_apt),
                int(row.access_le_50m),
                int(row.access_le_100m),
                int(row.access_le_200m),
            ]
        )

    boundary = gpd.read_file(boundary_file).to_crs("EPSG:32611")
    boundary_geometry = boundary.geometry.union_all().simplify(120).buffer(0)
    boundary_wgs84 = (
        gpd.GeoSeries([boundary_geometry], crs="EPSG:32611").to_crs("EPSG:4326").iloc[0]
    )

    graph = ox.load_graphml(graph_file)
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True, fill_edge_geometry=True)
    major = edges[
        edges.apply(
            lambda row: (
                not _as_bool(row.get("transit_only", False))
                and any(value in MAJOR_ROADS for value in _parse_multivalue(row.get("highway")))
            ),
            axis=1,
        )
    ].copy()
    major = major.to_crs("EPSG:32611")
    major.geometry = major.geometry.simplify(80)
    major = major.to_crs("EPSG:4326")
    seen = set()
    road_lines = []
    for geometry in major.geometry:
        key = shapely.normalize(geometry).wkb_hex
        if key in seen:
            continue
        seen.add(key)
        road_lines.extend(_line_coordinates(geometry))

    locations = gpd.read_parquet(
        customer_dir / "latent_service_locations.parquet",
        columns=[
            "latent_service_location_id",
            "service_location_type",
            "resunits_total",
            "road_access_distance_m",
            "geometry",
        ],
    )
    outliers = locations[locations["road_access_distance_m"] > 200].sort_values(
        "road_access_distance_m", ascending=False
    )
    outlier_points = [
        [
            round(float(row.geometry.x), 5),
            round(float(row.geometry.y), 5),
            round(float(row.road_access_distance_m), 1),
            str(row.service_location_type),
            int(row.resunits_total),
        ]
        for row in outliers.itertuples()
    ]
    return {
        "boundary": mapping(boundary_wgs84),
        "roads": road_lines,
        "cells": cells,
        "outliers": outlier_points,
        "summary": {
            "locations": int(manifest["grouping"]["latent_service_location_count"]),
            "typeCounts": manifest["classification"]["service_location_type_counts"],
            "thresholds": manifest["road_access"]["threshold_sensitivity"],
            "p95": manifest["road_access"]["distance_quantiles_m"]["p95"],
            "max": manifest["road_access"]["distance_quantiles_m"]["max"],
            "mixedUse": manifest["grouping"]["mixed_use_location_count"],
            "estimatedUnits": manifest["classification"]["estimated_housing_units_total"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--customer-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(args.customer_dir, args.graph, args.boundary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
