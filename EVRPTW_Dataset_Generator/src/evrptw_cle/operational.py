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
    auto_skip_component_node_threshold: int = 100
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
        if self.auto_skip_component_node_threshold < 1:
            raise ValueError("auto_skip_component_node_threshold must be positive")
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


def _component_skip_gate(
    raw_audit: ConnectivityAudit,
    raw_graph: nx.MultiDiGraph,
    covered_city_nodes: set[Hashable],
    policy: OperationalPolicy,
) -> dict[str, Any]:
    """Evaluate the preregistered small-isolated-component fallback.

    The primary gate always uses every raw city component.  Only after the full
    real-OSM buffer ladder fails may still-uncovered components with fewer than
    ``auto_skip_component_node_threshold`` nodes be removed from the effective
    coverage denominator.  Raw coverage is retained separately for audit.
    """

    covered_component_ids = {
        raw_graph.nodes[node]["weak_component_id"] for node in covered_city_nodes
    }
    components = raw_audit.components
    uncovered = components[~components["component_id"].isin(covered_component_ids)]
    skipped = uncovered[
        uncovered["node_count"] < policy.auto_skip_component_node_threshold
    ]
    retained_uncovered = uncovered[
        uncovered["node_count"] >= policy.auto_skip_component_node_threshold
    ]
    total_nodes = int(components["node_count"].sum())
    total_length_m = float(components["physical_road_length_m"].sum())
    skipped_nodes = int(skipped["node_count"].sum())
    skipped_length_m = float(skipped["physical_road_length_m"].sum())
    covered_nodes = len(covered_city_nodes)
    covered_length_m = _covered_physical_length_m(
        raw_audit, raw_graph, covered_city_nodes
    )
    effective_node_denominator = max(total_nodes - skipped_nodes, 1)
    effective_length_denominator = max(total_length_m - skipped_length_m, 1.0)
    node_coverage = covered_nodes / effective_node_denominator
    road_coverage = covered_length_m / effective_length_denominator
    passed = (
        retained_uncovered.empty
        and node_coverage >= policy.min_node_coverage
        and road_coverage >= policy.min_road_length_coverage
    )
    return {
        "definition": (
            "after exhausting the real-OSM buffer ladder, exclude only still-uncovered "
            f"weak components with node_count < {policy.auto_skip_component_node_threshold} "
            "from the effective coverage denominator"
        ),
        "component_node_threshold_exclusive": policy.auto_skip_component_node_threshold,
        "uncovered_component_count": int(len(uncovered)),
        "auto_skipped_component_count": int(len(skipped)),
        "auto_skipped_component_ids": skipped["component_id"].astype(str).tolist(),
        "auto_skipped_node_count": skipped_nodes,
        "auto_skipped_node_share": skipped_nodes / max(total_nodes, 1),
        "auto_skipped_physical_road_length_m": skipped_length_m,
        "auto_skipped_physical_road_length_share": skipped_length_m
        / max(total_length_m, 1.0),
        "retained_uncovered_component_count": int(len(retained_uncovered)),
        "retained_uncovered_component_ids": retained_uncovered["component_id"]
        .astype(str)
        .tolist(),
        "effective_city_node_count": int(effective_node_denominator),
        "effective_city_physical_road_length_m": effective_length_denominator,
        "effective_city_node_coverage": node_coverage,
        "effective_city_physical_road_length_coverage": road_coverage,
        "passed": bool(passed),
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
    selected: tuple[float, nx.MultiDiGraph, set[Hashable], dict[str, Any], str] | None = None

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
        trial["small_component_fallback"] = _component_skip_gate(
            raw_audit, raw_city_graph, covered_city_nodes, policy
        )
        trials.append(trial)
        if trial["coverage_gate_passed"]:
            selected = (buffer_km, candidate, root_nodes, trial, "raw_coverage")
            break

    if selected is None:
        fallback_trials = [
            trial for trial in trials if trial["small_component_fallback"]["passed"]
        ]
        if fallback_trials:
            selected_trial = max(
                fallback_trials,
                key=lambda trial: (
                    trial["city_node_coverage"],
                    trial["city_physical_road_length_coverage"],
                    -trial["buffer_km"],
                ),
            )
            buffer_km = float(selected_trial["buffer_km"])
            polygon = _buffer_geometry(boundary, buffer_km)
            candidate = ox.truncate.truncate_graph_polygon(
                envelope_graph,
                polygon,
                truncate_by_edge=False,
            )
            root_nodes = _root_component(candidate, city_nodes)
            selected = (
                buffer_km,
                candidate,
                root_nodes,
                selected_trial,
                "small_isolated_component_fallback",
            )
        else:
            last = trials[-1]
            fallback = last["small_component_fallback"]
            raise OperationalCoverageError(
                "No actual-OSM routing envelope met the operational coverage gates: "
                f"node={last['city_node_coverage']:.6f} required={policy.min_node_coverage:.6f}, "
                "road_length="
                f"{last['city_physical_road_length_coverage']:.6f} "
                f"required={policy.min_road_length_coverage:.6f}; "
                "small-component fallback retained "
                f"{fallback['retained_uncovered_component_count']} uncovered component(s) "
                "at or above the node threshold"
            )

    buffer_km, _, root_nodes, selected_trial, gate_mode = selected
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
        "coverage_gate_mode": gate_mode,
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
        "coverage_gate_city_node_coverage": (
            selected_trial["city_node_coverage"]
            if gate_mode == "raw_coverage"
            else selected_trial["small_component_fallback"][
                "effective_city_node_coverage"
            ]
        ),
        "coverage_gate_city_physical_road_length_coverage": (
            selected_trial["city_physical_road_length_coverage"]
            if gate_mode == "raw_coverage"
            else selected_trial["small_component_fallback"][
                "effective_city_physical_road_length_coverage"
            ]
        ),
        "small_isolated_component_fallback": selected_trial[
            "small_component_fallback"
        ],
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
