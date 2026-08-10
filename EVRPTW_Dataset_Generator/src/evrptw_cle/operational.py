from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import networkx as nx
import osmnx as ox

from .connectivity import ConnectivityAudit, audit_and_label


@dataclass(frozen=True)
class OperationalPolicy:
    buffer_ladder_km: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)
    min_node_coverage: float = 0.99
    min_road_length_coverage: float = 0.995
    micro_component_node_threshold: int = 10
    micro_component_length_km_threshold: float = 1.0

    def validate(self) -> None:
        if not self.buffer_ladder_km:
            raise ValueError("buffer_ladder_km cannot be empty")
        if self.buffer_ladder_km[0] != 0.0:
            raise ValueError("buffer_ladder_km must start at 0")
        if tuple(sorted(set(self.buffer_ladder_km))) != self.buffer_ladder_km:
            raise ValueError("buffer_ladder_km must be unique and increasing")
        if not 0.0 < self.min_node_coverage <= 1.0:
            raise ValueError("min_node_coverage must be in (0, 1]")
        if not 0.0 < self.min_road_length_coverage <= 1.0:
            raise ValueError("min_road_length_coverage must be in (0, 1]")
        if self.micro_component_node_threshold < 1:
            raise ValueError("micro_component_node_threshold must be positive")
        if self.micro_component_length_km_threshold <= 0:
            raise ValueError("micro_component_length_km_threshold must be positive")


@dataclass(frozen=True)
class OperationalSelection:
    graph: nx.MultiDiGraph
    audit: ConnectivityAudit
    summary: dict[str, Any]


class OperationalCoverageError(RuntimeError):
    """Raised when the available real-OSM envelope cannot meet the frozen gates."""


def _buffer_geometry(boundary: gpd.GeoDataFrame, buffer_km: float):
    if buffer_km == 0.0:
        return boundary.geometry.iloc[0]
    local_crs = boundary.estimate_utm_crs()
    projected = boundary.to_crs(local_crs)
    return (
        projected.set_geometry(projected.geometry.buffer(buffer_km * 1_000.0))
        .to_crs("EPSG:4326")
        .geometry.iloc[0]
    )


def _root_component(
    graph: nx.MultiDiGraph,
    city_nodes: set[Hashable],
) -> set[Hashable]:
    components = list(nx.weakly_connected_components(graph))
    if not components:
        raise ValueError("Cannot select an operational component from an empty graph")
    return set(
        max(
            components,
            key=lambda nodes: (
                len(city_nodes.intersection(nodes)),
                len(nodes),
                min(map(str, nodes)),
            ),
        )
    )


def _covered_physical_length_m(
    raw_audit: ConnectivityAudit,
    raw_graph: nx.MultiDiGraph,
    covered_city_nodes: set[Hashable],
) -> float:
    covered_component_ids = {
        raw_graph.nodes[node]["weak_component_id"] for node in covered_city_nodes
    }
    covered = raw_audit.components[raw_audit.components["component_id"].isin(covered_component_ids)]
    return float(covered["physical_road_length_m"].sum())


def _micro_component_summary(
    raw_audit: ConnectivityAudit,
    policy: OperationalPolicy,
) -> dict[str, Any]:
    components = raw_audit.components
    mask = (components["node_count"] < policy.micro_component_node_threshold) & (
        components["physical_road_length_m"] < policy.micro_component_length_km_threshold * 1_000.0
    )
    micro = components[mask]
    total_nodes = int(components["node_count"].sum())
    total_length = float(components["physical_road_length_m"].sum())
    return {
        "definition": (
            f"node_count < {policy.micro_component_node_threshold} AND "
            f"physical_road_length_km < {policy.micro_component_length_km_threshold}"
        ),
        "component_count": len(micro),
        "node_count": int(micro["node_count"].sum()),
        "node_share": float(micro["node_count"].sum() / max(total_nodes, 1)),
        "physical_road_length_m": float(micro["physical_road_length_m"].sum()),
        "physical_road_length_share": float(
            micro["physical_road_length_m"].sum() / max(total_length, 1.0)
        ),
        "role": "auto-skippable only; the raw city graph always preserves these components",
    }


def select_operational_graph(
    envelope_graph: nx.MultiDiGraph,
    raw_city_graph: nx.MultiDiGraph,
    raw_audit: ConnectivityAudit,
    boundary: gpd.GeoDataFrame,
    policy: OperationalPolicy,
) -> OperationalSelection:
    """Choose the smallest real-OSM routing envelope that meets coverage gates."""
    policy.validate()
    city_nodes = set(raw_city_graph.nodes)
    total_city_nodes = len(city_nodes)
    total_physical_length_m = float(raw_audit.summary["physical_road_length_m"])
    trials: list[dict[str, Any]] = []
    selected: tuple[float, nx.MultiDiGraph, set[Hashable], dict[str, Any]] | None = None

    for buffer_km in policy.buffer_ladder_km:
        polygon = _buffer_geometry(boundary, buffer_km)
        candidate = ox.truncate.truncate_graph_polygon(
            envelope_graph,
            polygon,
            truncate_by_edge=False,
        )
        root_nodes = _root_component(candidate, city_nodes)
        covered_city_nodes = root_nodes.intersection(city_nodes)
        covered_length_m = _covered_physical_length_m(
            raw_audit,
            raw_city_graph,
            covered_city_nodes,
        )
        trial = {
            "buffer_km": buffer_km,
            "candidate_node_count": candidate.number_of_nodes(),
            "candidate_weak_component_count": nx.number_weakly_connected_components(candidate),
            "root_total_node_count": len(root_nodes),
            "covered_city_node_count": len(covered_city_nodes),
            "city_node_coverage": len(covered_city_nodes) / total_city_nodes,
            "covered_city_physical_road_length_m": covered_length_m,
            "city_physical_road_length_coverage": covered_length_m
            / max(total_physical_length_m, 1.0),
        }
        trial["coverage_gate_passed"] = (
            trial["city_node_coverage"] >= policy.min_node_coverage
            and trial["city_physical_road_length_coverage"] >= policy.min_road_length_coverage
        )
        trials.append(trial)
        if trial["coverage_gate_passed"]:
            selected = (buffer_km, candidate, root_nodes, trial)
            break

    if selected is None:
        last = trials[-1]
        raise OperationalCoverageError(
            "No actual-OSM routing envelope met the operational coverage gates: "
            f"node={last['city_node_coverage']:.6f} required={policy.min_node_coverage:.6f}, "
            "road_length="
            f"{last['city_physical_road_length_coverage']:.6f} "
            f"required={policy.min_road_length_coverage:.6f}"
        )

    buffer_km, _, root_nodes, selected_trial = selected
    operational = envelope_graph.subgraph(root_nodes).copy()
    for node in operational.nodes:
        inside_city = node in city_nodes
        operational.nodes[node]["inside_city"] = inside_city
        operational.nodes[node]["transit_only"] = not inside_city
        operational.nodes[node]["service_location_eligible"] = inside_city
        operational.nodes[node]["raw_weak_component_id"] = (
            str(raw_city_graph.nodes[node].get("weak_component_id", ""))
            if inside_city
            else "OUTSIDE_CITY"
        )
    for u, v, key in operational.edges(keys=True):
        transit_only = u not in city_nodes or v not in city_nodes
        operational.edges[u, v, key]["transit_only"] = transit_only

    operational_audit = audit_and_label(operational)
    transit_nodes = sum(bool(data["transit_only"]) for _, data in operational.nodes(data=True))
    transit_edges = sum(
        bool(data["transit_only"]) for _, _, _, data in operational.edges(keys=True, data=True)
    )
    summary = {
        "schema": "evrptw_operational_connectivity_v1",
        "passed": True,
        "selected_buffer_km": buffer_km,
        "buffer_ladder_km": list(policy.buffer_ladder_km),
        "min_city_node_coverage": policy.min_node_coverage,
        "min_city_physical_road_length_coverage": policy.min_road_length_coverage,
        "city_node_count": total_city_nodes,
        "city_physical_road_length_m": total_physical_length_m,
        "covered_city_node_count": selected_trial["covered_city_node_count"],
        "city_node_coverage": selected_trial["city_node_coverage"],
        "covered_city_physical_road_length_m": selected_trial[
            "covered_city_physical_road_length_m"
        ],
        "city_physical_road_length_coverage": selected_trial["city_physical_road_length_coverage"],
        "operational_node_count": operational.number_of_nodes(),
        "operational_directed_edge_count": operational.number_of_edges(),
        "transit_only_node_count": transit_nodes,
        "transit_only_directed_edge_count": transit_edges,
        "weak_component_count": operational_audit.summary["weak_component_count"],
        "largest_strong_component_node_share": operational_audit.summary[
            "largest_strong_component_node_share"
        ],
        "micro_components": _micro_component_summary(raw_audit, policy),
        "protected_anchor_roundtrip_gate": {
            "required_coverage": 1.0,
            "status": "deferred_until_depot_charger_and_service_locations_exist",
        },
        "connector_semantics": (
            "actual OSM drive edges only; outside-city nodes are transit-only; "
            "no synthetic connector edges"
        ),
        "trials": trials,
    }
    return OperationalSelection(operational, operational_audit, summary)
