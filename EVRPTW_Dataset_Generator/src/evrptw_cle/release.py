from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .util import sha256_file, write_json


def build_release_index(preset_file: Path, city_root: Path, output_file: Path) -> dict[str, Any]:
    preset = json.loads(preset_file.read_text(encoding="utf-8"))
    city_records = []
    for item in preset["cities"]:
        city_dir = city_root / item["slug"]
        manifest_file = city_dir / "manifest.json"
        graph_file = city_dir / "graph_all.graphml"
        if not manifest_file.exists() or not graph_file.exists():
            city_records.append(
                {
                    "slug": item["slug"],
                    "query": item["query"],
                    "census_place_geoid": item.get("census_place_geoid"),
                    "status": "missing",
                }
            )
            continue
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        operational = manifest.get("operational_connectivity") or {}
        operational_graph_name = manifest.get("operational_graph")
        operational_graph_file = (
            city_dir / operational_graph_name if operational_graph_name else None
        )
        city_records.append(
            {
                "slug": item["slug"],
                "query": item["query"],
                "census_place_geoid": item.get("census_place_geoid"),
                "status": "complete",
                "graph_nodes": manifest["connectivity"]["node_count"],
                "graph_directed_edges": manifest["connectivity"]["directed_edge_count"],
                "weak_component_count": manifest["connectivity"]["weak_component_count"],
                "largest_weak_component_node_share": manifest["connectivity"][
                    "largest_weak_component_node_share"
                ],
                "graph_bytes": graph_file.stat().st_size,
                "graph_sha256": sha256_file(graph_file),
                "manifest_sha256": sha256_file(manifest_file),
                "operational_graph": operational_graph_name,
                "operational_status": (
                    "complete"
                    if operational_graph_file is not None and operational_graph_file.exists()
                    else "missing"
                ),
                "operational_selected_buffer_km": operational.get("selected_buffer_km"),
                "operational_city_node_coverage": operational.get("city_node_coverage"),
                "operational_city_road_length_coverage": operational.get(
                    "city_physical_road_length_coverage"
                ),
                "operational_transit_only_nodes": operational.get("transit_only_node_count"),
                "operational_graph_bytes": (
                    operational_graph_file.stat().st_size
                    if operational_graph_file is not None and operational_graph_file.exists()
                    else None
                ),
                "operational_graph_sha256": (
                    sha256_file(operational_graph_file)
                    if operational_graph_file is not None and operational_graph_file.exists()
                    else None
                ),
            }
        )
    payload = {
        "schema": "evrptw_cle_release_index_v2",
        "preset_id": preset["preset_id"],
        "selection_semantics": preset["selection_semantics"],
        "city_count": len(city_records),
        "complete_city_count": sum(record["status"] == "complete" for record in city_records),
        "cities": city_records,
    }
    write_json(output_file, payload)
    return payload
