from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pyarrow.dataset as ds
from shapely import from_wkb
from shapely.geometry import Point, mapping

MAJOR_ROADS = {"motorway", "trunk", "primary"}


def _highway_values(value) -> set[str]:
    if isinstance(value, list):
        return {str(item) for item in value}
    return {str(value)}


def _rounded_lines(graph_path: Path) -> list[list[list[float]]]:
    graph = ox.load_graphml(graph_path)
    seen = set()
    lines = []
    for _, _, _, data in graph.edges(keys=True, data=True):
        if not (_highway_values(data.get("highway")) & MAJOR_ROADS):
            continue
        geometry = data.get("geometry")
        if geometry is None:
            continue
        simplified = geometry.simplify(0.00015, preserve_topology=True)
        key = simplified.wkb
        reverse_key = type(simplified)(list(simplified.coords)[::-1]).wkb
        if key in seen or reverse_key in seen:
            continue
        seen.add(key)
        lines.append([[round(x, 5), round(y, 5)] for x, y in simplified.coords])
    return lines


def _city_payload(project_root: Path, slug: str) -> dict:
    city_dir = project_root / "data" / "buildings" / slug
    summary = json.loads((city_dir / "building_summary.json").read_text())
    boundary_frame = gpd.read_file(
        project_root / "boundaries" / "top10" / slug / "land_boundary.geojson"
    )
    boundary = boundary_frame.geometry.union_all().simplify(0.0001, preserve_topology=True)
    grid = gpd.read_file(city_dir / "building_density_grid.geojson")
    centers = grid.geometry.representative_point()
    density = [
        [
            round(point.x, 5),
            round(point.y, 5),
            int(row.building_count),
            round(float(row.mean_footprint_area_m2), 1),
        ]
        for point, row in zip(centers, grid.itertuples(), strict=True)
    ]

    focus_row = grid.loc[grid.building_count.idxmax()]
    min_x, min_y, max_x, max_y = focus_row.geometry.bounds
    parquet = ds.dataset(city_dir / "footprints", format="parquet")
    filter_expression = (
        (ds.field("location_lon") >= min_x)
        & (ds.field("location_lon") <= max_x)
        & (ds.field("location_lat") >= min_y)
        & (ds.field("location_lat") <= max_y)
    )
    table = parquet.to_table(
        columns=["geometry", "location_lon", "location_lat"], filter=filter_expression
    )
    footprints = []
    rows = zip(
        table.column("geometry").to_pylist(),
        table.column("location_lon").to_pylist(),
        table.column("location_lat").to_pylist(),
        strict=True,
    )
    for geometry_wkb, longitude, latitude in rows:
        if not focus_row.geometry.covers(Point(longitude, latitude)):
            continue
        geometry = from_wkb(geometry_wkb).simplify(0.000005, preserve_topology=True)
        if geometry.geom_type != "Polygon":
            continue
        footprints.append(
            [
                [[round(x, 6), round(y, 6)] for x, y in ring.coords]
                for ring in [geometry.exterior, *geometry.interiors]
            ]
        )

    quantiles = summary["area"]["quantiles_m2"]
    return {
        "slug": slug,
        "label": summary["city_label"],
        "count": summary["building_count"],
        "median_area": round(quantiles["p50"], 1),
        "p95_area": round(quantiles["p95"], 1),
        "boundary": mapping(boundary),
        "roads": _rounded_lines(
            project_root / "data" / "cities" / slug / "graph_operational.graphml"
        ),
        "density": density,
        "focus": {
            "bbox": [round(value, 6) for value in (min_x, min_y, max_x, max_y)],
            "building_count": int(focus_row.building_count),
            "footprints": footprints,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "cities": [
            _city_payload(args.project_root, "los-angeles"),
            _city_payload(args.project_root, "san-francisco"),
        ]
    }
    args.output.write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
