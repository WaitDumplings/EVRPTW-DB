from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import osmnx as ox
import pandas as pd

from .util import sha256_file


def verify_city_output(city_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = city_dir / "manifest.json"
    if not manifest_path.exists():
        return {"passed": False, "errors": ["missing manifest.json"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for relative_path, expected in manifest.get("checksums", {}).items():
        path = city_dir / relative_path
        if not path.exists():
            errors.append(f"missing checksum target: {relative_path}")
        elif sha256_file(path) != expected:
            errors.append(f"checksum mismatch: {relative_path}")

    graph_path = city_dir / manifest.get("all_graph", "graph_all.graphml")
    components_path = city_dir / "components.csv"
    if graph_path.exists():
        graph = ox.load_graphml(graph_path)
        expected = manifest.get("connectivity", {})
        if not graph.is_directed() or not graph.is_multigraph():
            errors.append("complete graph is not a directed MultiDiGraph")
        if graph.number_of_nodes() != expected.get("node_count"):
            errors.append("graph node count differs from manifest")
        if graph.number_of_edges() != expected.get("directed_edge_count"):
            errors.append("graph edge count differs from manifest")
    if components_path.exists():
        components = pd.read_csv(components_path)
        expected_count = manifest.get("connectivity", {}).get("weak_component_count")
        if len(components) != expected_count:
            errors.append("components.csv row count differs from manifest")

    selected = city_dir / manifest.get("selected_graph", "")
    if not selected.exists():
        errors.append("selected graph referenced by manifest is missing")

    operational_summary = manifest.get("operational_connectivity")
    operational_name = manifest.get("operational_graph")
    if operational_summary is not None:
        if not operational_name:
            errors.append("operational connectivity exists but operational graph is not named")
        else:
            operational_path = city_dir / operational_name
            if not operational_path.exists():
                errors.append("operational graph is missing")
            else:
                operational = ox.load_graphml(operational_path)
                if not operational.is_directed() or not operational.is_multigraph():
                    errors.append("operational graph is not a directed MultiDiGraph")
                if nx.number_weakly_connected_components(operational) != 1:
                    errors.append("operational graph is not one weak component")
                if operational.number_of_nodes() != operational_summary.get(
                    "operational_node_count"
                ):
                    errors.append("operational graph node count differs from manifest")
                if operational.number_of_edges() != operational_summary.get(
                    "operational_directed_edge_count"
                ):
                    errors.append("operational graph edge count differs from manifest")
        if operational_summary.get("city_node_coverage", 0.0) < operational_summary.get(
            "min_city_node_coverage", 1.0
        ):
            errors.append("operational city-node coverage gate failed")
        if operational_summary.get(
            "city_physical_road_length_coverage", 0.0
        ) < operational_summary.get("min_city_physical_road_length_coverage", 1.0):
            errors.append("operational city-road-length coverage gate failed")
    return {"passed": not errors, "errors": errors, "city_dir": str(city_dir)}
