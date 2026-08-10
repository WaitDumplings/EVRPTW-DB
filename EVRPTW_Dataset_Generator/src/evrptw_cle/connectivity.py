from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import networkx as nx
import osmnx as ox
import pandas as pd

COMPONENT_POLICIES = ("all", "largest_weak")


@dataclass(frozen=True)
class ConnectivityAudit:
    components: pd.DataFrame
    summary: dict[str, Any]


def _node_key(node: Hashable) -> str:
    return str(node)


def ordered_weak_components(graph: nx.MultiDiGraph) -> list[set[Hashable]]:
    components = [set(component) for component in nx.weakly_connected_components(graph)]
    return sorted(components, key=lambda values: (-len(values), min(map(_node_key, values))))


def physical_undirected_graph(graph: nx.MultiDiGraph) -> nx.MultiGraph:
    """Use OSMnx geometry-aware deduplication, with a generic-graph test fallback."""
    if "crs" in graph.graph:
        return ox.convert.to_undirected(graph)
    return graph.to_undirected()


def audit_and_label(graph: nx.MultiDiGraph) -> ConnectivityAudit:
    if not graph.is_directed() or not graph.is_multigraph():
        raise TypeError("Connectivity audit requires a directed MultiDiGraph")
    if len(graph) == 0:
        raise ValueError("Cannot audit an empty graph")

    weak_components = ordered_weak_components(graph)
    records = []
    total_nodes = graph.number_of_nodes()
    total_edges = graph.number_of_edges()
    undirected = physical_undirected_graph(graph)
    total_physical_length_m = sum(
        float(data.get("length", 0.0) or 0.0)
        for _, _, _, data in undirected.edges(keys=True, data=True)
    )

    for rank, nodes in enumerate(weak_components, start=1):
        component_id = f"W{rank:04d}"
        nx.set_node_attributes(graph, {node: component_id for node in nodes}, "weak_component_id")
        nx.set_node_attributes(graph, {node: rank for node in nodes}, "weak_component_rank")
        subgraph = graph.subgraph(nodes)
        undirected_subgraph = undirected.subgraph(nodes)
        physical_length_m = sum(
            float(data.get("length", 0.0) or 0.0)
            for _, _, _, data in undirected_subgraph.edges(keys=True, data=True)
        )
        for u, v, key in subgraph.edges(keys=True):
            graph.edges[u, v, key]["weak_component_id"] = component_id
            graph.edges[u, v, key]["weak_component_rank"] = rank

        xs = [float(graph.nodes[node].get("x", float("nan"))) for node in nodes]
        ys = [float(graph.nodes[node].get("y", float("nan"))) for node in nodes]
        edge_count = subgraph.number_of_edges()
        records.append(
            {
                "component_id": component_id,
                "rank": rank,
                "node_count": len(nodes),
                "directed_edge_count": edge_count,
                "physical_road_length_m": physical_length_m,
                "node_share": len(nodes) / total_nodes,
                "edge_share": edge_count / max(total_edges, 1),
                "physical_road_length_share": physical_length_m / max(total_physical_length_m, 1.0),
                "is_largest": rank == 1,
                "min_lon": min(xs),
                "min_lat": min(ys),
                "max_lon": max(xs),
                "max_lat": max(ys),
            }
        )

    strong_sizes = sorted(
        (len(component) for component in nx.strongly_connected_components(graph)), reverse=True
    )
    frame = pd.DataFrame.from_records(records)
    summary = {
        "directed_multigraph": True,
        "node_count": total_nodes,
        "directed_edge_count": total_edges,
        "physical_road_length_m": total_physical_length_m,
        "weak_component_count": len(weak_components),
        "largest_weak_component_nodes": int(frame.iloc[0]["node_count"]),
        "largest_weak_component_node_share": float(frame.iloc[0]["node_share"]),
        "strong_component_count": len(strong_sizes),
        "largest_strong_component_nodes": strong_sizes[0],
        "largest_strong_component_node_share": strong_sizes[0] / total_nodes,
        "filtering_basis": "weak connectivity",
        "strong_connectivity_role": "reported as a directionality diagnostic; never used for default filtering",
    }
    return ConnectivityAudit(frame, summary)


def apply_component_policy(graph: nx.MultiDiGraph, policy: str) -> nx.MultiDiGraph:
    if policy not in COMPONENT_POLICIES:
        raise ValueError(f"Unknown component policy {policy!r}; choose from {COMPONENT_POLICIES}")
    if policy == "all":
        return graph.copy()
    largest_nodes = ordered_weak_components(graph)[0]
    return graph.subgraph(largest_nodes).copy()
