"""Directed round-trip eligibility for road-anchored CLE locations.

The operational road graph is intentionally allowed to retain small directed
components for provenance.  Benchmark locations, however, must be able to
leave and return to the reference service component.  This module labels
virtual edge projections without materializing millions of access nodes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import networkx as nx
import pandas as pd


@dataclass(frozen=True)
class DirectedComponentIndex:
    """Stable SCC labels for one directed operational graph."""

    node_to_scc: dict[str, str]
    scc_sizes: dict[str, int]
    reference_scc_id: str
    graph_node_count: int

    @property
    def reference_scc_node_count(self) -> int:
        return self.scc_sizes[self.reference_scc_id]

    @property
    def reference_scc_node_share(self) -> float:
        return self.reference_scc_node_count / self.graph_node_count


def build_directed_component_index(
    graph: nx.MultiDiGraph,
) -> DirectedComponentIndex:
    """Return deterministic size-ranked SCC labels; S0001 is the reference SCC."""

    components = sorted(
        ({str(node) for node in component} for component in nx.strongly_connected_components(graph)),
        key=lambda component: (-len(component), min(component)),
    )
    if not components:
        raise ValueError("Operational graph has no directed strong component")
    node_to_scc: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for rank, component in enumerate(components, start=1):
        scc_id = f"S{rank:04d}"
        sizes[scc_id] = len(component)
        node_to_scc.update({node: scc_id for node in component})
    return DirectedComponentIndex(
        node_to_scc=node_to_scc,
        scc_sizes=sizes,
        reference_scc_id="S0001",
        graph_node_count=graph.number_of_nodes(),
    )


def projection_scc_id(
    directed_projection_offsets: str,
    index: DirectedComponentIndex,
    *,
    endpoint_tolerance: float = 1e-10,
) -> str:
    """Infer the SCC inherited by an exact directed-edge projection.

    An interior split node inherits an existing SCC only when both endpoints
    of at least one referenced directed edge are already in that SCC.  An
    endpoint projection inherits the endpoint SCC.  Ambiguous or cross-SCC
    interior projections are explicitly unresolved rather than assumed safe.
    """

    inherited: set[str] = set()
    for ref in json.loads(directed_projection_offsets):
        u = str(ref["u"])
        v = str(ref["v"])
        fraction = float(ref["projection_fraction_from_u"])
        if fraction <= endpoint_tolerance:
            inherited.add(index.node_to_scc[u])
        elif fraction >= 1.0 - endpoint_tolerance:
            inherited.add(index.node_to_scc[v])
        elif index.node_to_scc[u] == index.node_to_scc[v]:
            inherited.add(index.node_to_scc[u])
    if len(inherited) == 1:
        return next(iter(inherited))
    return "S_UNRESOLVED"


def label_projection_connectivity(
    frame: pd.DataFrame,
    index: DirectedComponentIndex,
) -> pd.DataFrame:
    """Add SCC and protected round-trip fields to a road-anchored table."""

    if "directed_projection_offsets" not in frame:
        raise ValueError("Projection table lacks directed_projection_offsets")
    result = frame.copy()
    unique_offsets = result["directed_projection_offsets"].drop_duplicates()
    label_by_offsets = {
        value: projection_scc_id(str(value), index) for value in unique_offsets
    }
    result["anchor_scc_id"] = result["directed_projection_offsets"].map(
        label_by_offsets
    )
    result["reference_scc_id"] = index.reference_scc_id
    result["reference_scc_node_count"] = index.reference_scc_node_count
    result["reference_scc_node_share"] = index.reference_scc_node_share
    result["anchor_in_reference_scc"] = result["anchor_scc_id"].eq(
        index.reference_scc_id
    )
    result["protected_roundtrip_eligible"] = result[
        "anchor_in_reference_scc"
    ]
    result["protected_roundtrip_status"] = result[
        "protected_roundtrip_eligible"
    ].map(
        {
            True: "passed_reference_scc",
            False: "quarantine_outside_reference_scc",
        }
    )
    return result


def connectivity_summary(
    frame: pd.DataFrame,
    index: DirectedComponentIndex,
) -> dict[str, Any]:
    """Return compact, paper-table-ready protected-anchor counts."""

    eligible = frame["protected_roundtrip_eligible"].astype(bool)
    return {
        "reference_scc_id": index.reference_scc_id,
        "road_graph_node_count": index.graph_node_count,
        "road_graph_strong_component_count": len(index.scc_sizes),
        "reference_scc_node_count": index.reference_scc_node_count,
        "reference_scc_node_share": index.reference_scc_node_share,
        "anchor_count": len(frame),
        "roundtrip_eligible_anchor_count": int(eligible.sum()),
        "roundtrip_quarantined_anchor_count": int((~eligible).sum()),
        "roundtrip_eligible_anchor_share": float(eligible.mean()) if len(frame) else 0.0,
        "policy": (
            "retain all source locations; default benchmark candidates must inherit "
            "the reference directed SCC"
        ),
    }
