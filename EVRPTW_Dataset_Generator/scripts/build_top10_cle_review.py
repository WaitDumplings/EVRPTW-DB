#!/usr/bin/env python3
"""Build a compact interactive review of all ten final cle layers."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, MultiLineString, mapping

ROAD_CLASSES = {"motorway", "trunk", "primary", "secondary"}


def _normalize_highways(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(item) for item in value}
    if value is None:
        return set()
    return {str(value)}


def _iter_lines(geometry: Any) -> Iterable[LineString]:
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms


def _road_context(graph_path: Path, land: Any, max_segments: int) -> list[list[list[float]]]:
    graph = ox.load_graphml(graph_path)
    edges = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True).reset_index()
    edges = edges.loc[
        edges["highway"].map(lambda value: bool(_normalize_highways(value) & ROAD_CLASSES))
    ].sort_values("length", ascending=False)
    roads: list[list[list[float]]] = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    for geometry in edges.geometry:
        for line in _iter_lines(geometry.intersection(land)):
            coordinates = [
                [round(x, 5), round(y, 5)]
                for x, y in line.simplify(0.001, preserve_topology=False).coords
            ]
            if len(coordinates) < 2:
                continue
            forward = tuple((point[0], point[1]) for point in coordinates)
            key = min(forward, tuple(reversed(forward)))
            if key in seen:
                continue
            seen.add(key)
            roads.append(coordinates)
            if len(roads) >= max_segments:
                return roads
    return roads


def _density_points(path: Path) -> list[list[float | int]]:
    frame = gpd.read_file(path).to_crs("EPSG:4326")
    points = frame.geometry.representative_point()
    return [
        [round(float(point.x), 4), round(float(point.y), 4), int(count)]
        for point, count in zip(points, frame["service_location_count"], strict=True)
    ]


def _depot_points(path: Path, eligible_column: str) -> list[list[Any]]:
    columns = [
        "longitude",
        "latitude",
        "facility_name",
        "facility_area_m2",
        "road_access_distance_m",
        eligible_column,
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame.loc[frame[eligible_column].astype(bool)]
    return [
        [
            round(float(row.longitude), 6),
            round(float(row.latitude), 6),
            str(row.facility_name) if pd.notna(row.facility_name) else "",
            round(float(row.facility_area_m2), 1),
            round(float(row.road_access_distance_m), 1),
        ]
        for row in frame.itertuples(index=False)
    ]


def _charger_points(path: Path) -> list[list[Any]]:
    columns = [
        "Longitude",
        "Latitude",
        "Station Name",
        "reference_charge_mode",
        "charger_candidate_eligible",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame.loc[frame["charger_candidate_eligible"].astype(bool)]
    return [
        [
            round(float(row["Longitude"]), 6),
            round(float(row["Latitude"]), 6),
            str(row["Station Name"]) if pd.notna(row["Station Name"]) else "",
            str(row["reference_charge_mode"]),
        ]
        for _, row in frame.iterrows()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("analysis/cle_repairs/top10-v1"),
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("templates/top10_cle_review.fragment.html"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-road-segments", type=int, default=100)
    parser.add_argument(
        "--boundary-simplify-degrees",
        type=float,
        default=0.003,
        help="Rendering-only boundary simplification; source boundary is unchanged.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    index = json.loads((root / "top10_cle_index.json").read_text())
    cities = []
    for item in index["cities"]:
        slug = item["city_slug"]
        cle = root / "cles" / slug
        boundary_frame = gpd.read_file(
            cle / "boundary/service_boundary.geojson"
        ).to_crs("EPSG:4326")
        land = boundary_frame.geometry.union_all()
        infrastructure = cle / "infrastructure"
        cities.append(
            {
                "slug": slug,
                "label": item["city_label"],
                "counts": {
                    "service": int(item["latent_service_locations"]),
                    "chargers": int(item["charger_candidates"]),
                    "strict": int(item["strict_depot_candidates"]),
                    "optional": int(item["optional_depot_candidates"]),
                    "depots": int(item["depot_candidates"]),
                },
                "boundary": mapping(
                    land.simplify(
                        args.boundary_simplify_degrees,
                        preserve_topology=True,
                    )
                ),
                "roads": _road_context(
                    root / "cities" / slug / "graph_operational.graphml",
                    land,
                    args.max_road_segments,
                ),
                "density": _density_points(
                    root / "customers" / slug / "latent_service_density_500m.geojson"
                ),
                "strict": _depot_points(
                    infrastructure / "depots.parquet",
                    "strict_depot_candidate_eligible",
                ),
                "optional": _depot_points(
                    infrastructure / "depots.parquet",
                    "optional_depot_candidate_eligible",
                ),
                "chargers": _charger_points(infrastructure / "chargers.parquet"),
            }
        )
    payload = json.dumps({"cities": cities}, ensure_ascii=False, separators=(",", ":"))
    template = args.template.read_text(encoding="utf-8")
    if "__CLE_DATA__" not in template:
        raise ValueError("Visualization template is missing its data placeholder")
    output = template.replace("__CLE_DATA__", payload.replace("</", "<\\/"))
    if any(token in output.lower() for token in ("<!doctype", "<html", "<head", "<body")):
        raise ValueError("Visualization output must remain an HTML fragment")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    size = args.output.stat().st_size
    if size >= 2_000_000:
        raise ValueError(f"Visualization fragment exceeds 2 MB: {size:,} bytes")
    print(json.dumps({"output": str(args.output.resolve()), "bytes": size, "cities": 10}))


if __name__ == "__main__":
    main()
