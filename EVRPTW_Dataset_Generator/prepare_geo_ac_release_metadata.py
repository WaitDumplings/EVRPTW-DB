from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "EVRPTW_Dataset" / "Geo_AC_v1" / "source_data_na_us20"
DEFAULT_EVAL_ROOT = REPO_ROOT / "EVRPTW_Dataset" / "Geo_AC_v1" / "eval_standard_20"
SCALE_INSTANCE_COUNT = 20
SCALES = (5, 15, 50, 100)

NORMALIZED_SCHEMAS = {
    "road_nodes.csv": ["node_id", "lon", "lat", "x_km", "y_km"],
    "road_edges.csv": ["u", "v", "length_km", "source"],
    "customer_seed.csv": ["community_id", "tract", "block_group", "lon", "lat", "x_km", "y_km", "occupancy"],
    "latent_customer.csv": [
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
    ],
    "charging_station.csv": [
        "station_id",
        "name",
        "lon",
        "lat",
        "x_km",
        "y_km",
        "network",
        "level2_count",
        "dc_fast_count",
        "status",
        "access",
    ],
    "depot_candidate.csv": ["candidate_id", "lon", "lat", "x_km", "y_km", "source", "source_id", "category"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write release metadata for Geo-AC-v1 NA-US-20.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--instances-per-territory-scale", type=int, default=SCALE_INSTANCE_COUNT)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)


def _count_pickle_instances(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        import pickle

        with path.open("rb") as f:
            payload = pickle.load(f)
            if isinstance(payload, dict) and "num_instances" in payload:
                return int(payload["num_instances"])
            if isinstance(payload, dict) and "instances" in payload:
                return len(payload["instances"])
            if isinstance(payload, list):
                return len(payload)
            count = 0
            while True:
                try:
                    pickle.load(f)
                    count += 1
                except EOFError:
                    return count
    except Exception:
        return None


def _summary_rows(source_root: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    by_id = {item["territory_id"]: item for item in summary.get("territories", [])}
    territory_dirs = sorted(path for path in source_root.iterdir() if (path / "normalized").exists())
    for territory_dir in territory_dirs:
        tid = territory_dir.name
        item = by_id.get(tid, {})
        normalized = territory_dir / "normalized"
        rows.append(
            {
                "territory_id": tid,
                "display_name": item.get("display_name", tid),
                "parent_territory_id": item.get("parent_territory_id", ""),
                "split_method": item.get("split_method", "county_container"),
                "road_nodes": item.get("road_node_count", _count_rows(normalized / "road_nodes.csv")),
                "road_edges": item.get("road_edge_count", _count_rows(normalized / "road_edges.csv")),
                "customer_seeds": item.get("customer_seed_count", _count_rows(normalized / "customer_seed.csv")),
                "latent_customers": item.get("latent_customer_count", _count_rows(normalized / "latent_customer.csv")),
                "charging_stations": item.get("charging_station_count", _count_rows(normalized / "charging_station.csv")),
                "depot_candidates": item.get("depot_candidate_count", _count_rows(normalized / "depot_candidate.csv")),
                "fallback_depots": item.get("fallback_depot_count", ""),
                "occupied_housing_units": int(float(item.get("occupancy_total", 0) or 0)),
                "area_km2": round(float(item.get("territory_area_km2", 0) or 0), 3),
                "latent_connector_p50_km": round(float(item.get("latent_customer_connector_p50_km", 0) or 0), 5),
                "latent_connector_p90_km": round(float(item.get("latent_customer_connector_p90_km", 0) or 0), 5),
            }
        )
    return rows


def _file_inventory(source_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for territory_dir in sorted(path for path in source_root.iterdir() if (path / "normalized").exists()):
        for rel in [f"normalized/{name}" for name in NORMALIZED_SCHEMAS] + [
            "qa/qa_summary.json",
            "qa/qa_report.md",
            "qa/preview_layers.geojson",
        ]:
            path = territory_dir / rel
            rows.append(
                {
                    "territory_id": territory_dir.name,
                    "relative_path": f"{territory_dir.name}/{rel}",
                    "bytes": path.stat().st_size if path.exists() else 0,
                    "rows": _count_rows(path) if path.suffix == ".csv" else "",
                    "exists": bool(path.exists()),
                }
            )
    return rows


def _eval_inventory(eval_root: Path, territories: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for territory_id in territories:
        for scale in SCALES:
            instance_path = eval_root / "eval" / territory_id / f"Cus_{scale}" / "instances.pkl"
            rows.append(
                {
                    "territory_id": territory_id,
                    "scale": scale,
                    "instances_path": str(instance_path.relative_to(eval_root)) if instance_path.exists() else "",
                    "instances": _count_pickle_instances(instance_path),
                    "exists": bool(instance_path.exists()),
                }
            )
    return rows


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = ["territory_id", "road_nodes", "road_edges", "customer_seeds", "latent_customers", "charging_stations", "depot_candidates"]
    out = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("|" + "|".join(str(row.get(header, "")) for header in headers) + "|")
    return "\n".join(out)


def _write_source_readme(source_root: Path, rows: list[dict[str, Any]], instances_per: int, eval_present: bool) -> None:
    total_instances = len(rows) * len(SCALES) * int(instances_per)
    text = f"""# Geo-AC-v1 / NA-US-20 Source Data

This directory is the source-data release for the Geo-AC-v1 NA-US-20
real-geography, semi-synthetic EVRPTW benchmark.

Each territory contains normalized CSV inputs:

- `road_nodes.csv`
- `road_edges.csv`
- `customer_seed.csv`
- `latent_customer.csv`
- `charging_station.csv`
- `depot_candidate.csv`

The spatial layer is derived from public geospatial data: Census TIGER/Line,
ACS occupied housing units, OSM/OSMnx roads, AFDC/NREL public EV charging
stations, and OSM/Overture depot-like industrial/logistics features. Amazon
calibration is used only for daily operating attributes such as demand, service
time, time windows, and activation rates.

The fixed standard evaluation split is `20 territories x 4 scales x
{instances_per} instances = {total_instances:,}` instances. Eval files are
present in the sibling `eval_standard_20` directory: `{eval_present}`.

## Territory Table

{_markdown_table(rows)}

## Metadata Files

- `metadata/territory_table.csv`
- `metadata/dataset_summary.json`
- `metadata/source_versions.json`
- `metadata/normalized_schema.json`
- `metadata/file_inventory.csv`

The GitHub repository intentionally does not track this directory because it is
large generated data. Publish it through a dataset hosting service or release
artifact.
"""
    (source_root / "README.md").write_text(text, encoding="utf-8")


def _write_eval_readme(eval_root: Path, rows: list[dict[str, Any]], instances_per: int) -> None:
    if not eval_root.exists():
        return
    text = f"""# Geo-AC-v1 / NA-US-20 eval_standard_20

This directory stores the fixed evaluation split for Geo-AC-v1 NA-US-20.

Layout:

```text
service_territories/<territory_id>/service_territory_pool.pkl
eval/<territory_id>/Cus_5/instances.pkl
eval/<territory_id>/Cus_15/instances.pkl
eval/<territory_id>/Cus_50/instances.pkl
eval/<territory_id>/Cus_100/instances.pkl
```

Each territory-scale pair contains {instances_per} operating-day instances. The
full split contains {len(rows) * len(SCALES) * int(instances_per):,} instances.
The spatial service-territory source CSVs are stored in the sibling
`source_data_na_us20` directory.
"""
    (eval_root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    eval_root = args.eval_root.resolve()
    summary_path = source_root / "na_us20_summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else {"territories": []}
    if isinstance(summary.get("config"), str):
        summary["config"] = "EVRPTW_Dataset_Generator/configs/geo_ac_v1_na_us20.with_sources.yaml"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    territory_rows = _summary_rows(source_root, summary)
    territory_ids = [row["territory_id"] for row in territory_rows]
    metadata_dir = source_root / "metadata"

    totals = {
        "road_nodes": sum(int(row["road_nodes"]) for row in territory_rows),
        "road_edges": sum(int(row["road_edges"]) for row in territory_rows),
        "customer_seeds": sum(int(row["customer_seeds"]) for row in territory_rows),
        "latent_customers": sum(int(row["latent_customers"]) for row in territory_rows),
        "charging_stations": sum(int(row["charging_stations"]) for row in territory_rows),
        "depot_candidates": sum(int(row["depot_candidates"]) for row in territory_rows),
    }
    eval_inventory = _eval_inventory(eval_root, territory_ids)
    eval_present = bool(eval_inventory) and all(bool(row["exists"]) for row in eval_inventory)

    _write_csv(
        metadata_dir / "territory_table.csv",
        territory_rows,
        [
            "territory_id",
            "display_name",
            "parent_territory_id",
            "split_method",
            "road_nodes",
            "road_edges",
            "customer_seeds",
            "latent_customers",
            "charging_stations",
            "depot_candidates",
            "fallback_depots",
            "occupied_housing_units",
            "area_km2",
            "latent_connector_p50_km",
            "latent_connector_p90_km",
        ],
    )
    _write_csv(metadata_dir / "file_inventory.csv", _file_inventory(source_root), ["territory_id", "relative_path", "bytes", "rows", "exists"])
    _write_csv(metadata_dir / "eval_inventory.csv", eval_inventory, ["territory_id", "scale", "instances_path", "instances", "exists"])
    _write_json(metadata_dir / "normalized_schema.json", {"csv_schemas": NORMALIZED_SCHEMAS})
    _write_json(
        metadata_dir / "source_versions.json",
        {
            "census_tiger_line": "TIGER/Line 2025",
            "acs": "ACS 2024 5-year B25002_002E occupied housing units",
            "roads": "OSMnx/OpenStreetMap drive network; TIGER roads where recorded by territory QA",
            "chargers": "NREL/AFDC public alternative fuel stations API",
            "depots": "OSM/Overture warehouse, logistics, freight, and industrial features with marked fallback candidates",
            "customer_generation": "occupancy-weighted road-frontage latent customer placement",
        },
    )
    _write_json(
        metadata_dir / "dataset_summary.json",
        {
            "dataset_family": "EVRPTW-D",
            "dataset_version": "Geo-AC-v1",
            "profile_name": summary.get("profile_name", "Geo-AC-v1 / NA-US-20"),
            "territory_count": len(territory_rows),
            "territory_ids": territory_ids,
            "totals": totals,
            "standard_eval": {
                "scales": list(SCALES),
                "instances_per_territory_scale": int(args.instances_per_territory_scale),
                "total_instances": len(territory_rows) * len(SCALES) * int(args.instances_per_territory_scale),
                "eval_root_exists": eval_root.exists(),
                "eval_files_complete": eval_present,
            },
            "release_layout": {
                "source_data": ".",
                "eval_standard_20": "../eval_standard_20",
                "qa_maps": "qa_maps",
                "qa_maps_latent": "qa_maps_latent",
            },
            "spatial_framing": "real-geography semi-synthetic EVRPTW",
            "calibration_note": "Amazon calibration is used for daily operating attributes only, not customer/depot/charger locations.",
        },
    )
    _write_source_readme(source_root, territory_rows, int(args.instances_per_territory_scale), eval_present)
    _write_eval_readme(eval_root, territory_rows, int(args.instances_per_territory_scale))
    print(json.dumps({"source_root": str(source_root), "metadata_dir": str(metadata_dir), "territories": len(territory_rows), "eval_complete": eval_present}, indent=2))


if __name__ == "__main__":
    main()
