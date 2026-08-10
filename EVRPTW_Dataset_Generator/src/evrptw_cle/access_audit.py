"""Local-route QA for virtual service-node materialization."""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from .service_access import materialize_active_service_graph
from .speed_audit import _set_scenario_weights
from .util import sha256_file, write_json


def _path_costs(graph: nx.MultiDiGraph, path: list[Any]) -> tuple[float, float]:
    distance_m = 0.0
    time_s = 0.0
    for u, v in pairwise(path):
        attributes = min(
            graph[u][v].values(), key=lambda attrs: float(attrs["scenario_time_s"])
        )
        distance_m += float(attributes["length"])
        time_s += float(attributes["scenario_time_s"])
    return distance_m, time_s


def _select_same_edge_pairs(
    candidates: pd.DataFrame, *, seed: int, pair_count: int
) -> list[dict[str, str]]:
    eligible = candidates.loc[
        candidates["geometry_evidence_tier"].isin(
            {"G1_containment", "G2_near_area_consistent"}
        )
        & (pd.to_numeric(candidates["road_access_distance_m"], errors="coerce") <= 200.0)
        & (pd.to_numeric(candidates["directed_edge_ref_count"], errors="coerce") >= 2)
    ].copy()
    pairs = []
    for physical_edge_id, group in eligible.groupby("physical_edge_id", sort=False):
        ordered = group.sort_values("road_projection_fraction_from_physical_start")
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        if first["road_projection_node_id"] == last["road_projection_node_id"]:
            continue
        pairs.append(
            {
                "physical_edge_id": str(physical_edge_id),
                "a": str(first["latent_service_location_id"]),
                "b": str(last["latent_service_location_id"]),
                "a_type": str(first["service_location_type"]),
                "b_type": str(last["service_location_type"]),
            }
        )
    if len(pairs) < pair_count:
        raise ValueError(f"Only {len(pairs)} eligible same-edge pairs; need {pair_count}")
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(pairs), size=pair_count, replace=False)
    return [pairs[int(index)] for index in selected]


def build_service_access_local_route_audit(
    *,
    graph_path: Path,
    access_candidates_path: Path,
    service_nodes_path: Path,
    projection_nodes_path: Path,
    connectors_path: Path,
    scenarios_path: Path,
    output_dir: Path,
    scenario_id: str | None = None,
    connector_speed_kph: float = 10.0,
    seed: int = 20270805,
    pair_count: int = 12,
) -> dict[str, Any]:
    """Prove that close stops use exact partial directed edges, not endpoint snaps."""

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_parquet(access_candidates_path)
    pairs = _select_same_edge_pairs(candidates, seed=seed, pair_count=pair_count)
    active = {value for pair in pairs for value in (pair["a"], pair["b"])}
    service_nodes = gpd.read_parquet(service_nodes_path)
    projection_nodes = gpd.read_parquet(projection_nodes_path)
    connectors = pd.read_parquet(connectors_path)
    scenarios = pd.read_parquet(scenarios_path)
    if scenario_id is None:
        scenario_id = str(min(scenarios["scenario_id"].unique()))
    scenario = scenarios.loc[scenarios["scenario_id"].astype(str) == scenario_id]
    if scenario.empty:
        raise KeyError(f"Unknown scenario_id {scenario_id}")

    graph = ox.load_graphml(graph_path)
    _set_scenario_weights(graph, scenario)
    materialized, materialization = materialize_active_service_graph(
        graph=graph,
        service_nodes=service_nodes,
        projection_nodes=projection_nodes,
        connectors=connectors,
        active_service_location_ids=active,
        connector_speed_kph=connector_speed_kph,
        proportional_edge_attributes=("length", "scenario_time_s"),
    )
    service_id = dict(
        zip(
            service_nodes["latent_service_location_id"].astype(str),
            service_nodes["service_access_node_id"].astype(str),
            strict=True,
        )
    )
    records = []
    for index, pair in enumerate(pairs):
        for direction, origin, destination in (
            ("a_to_b", pair["a"], pair["b"]),
            ("b_to_a", pair["b"], pair["a"]),
        ):
            path = nx.shortest_path(
                materialized,
                service_id[origin],
                service_id[destination],
                weight="scenario_time_s",
                method="dijkstra",
            )
            distance_m, time_s = _path_costs(materialized, path)
            records.append(
                {
                    "pair_id": f"local-{index:03d}",
                    "direction": direction,
                    "physical_edge_id": pair["physical_edge_id"],
                    "origin_id": origin,
                    "destination_id": destination,
                    "origin_type": pair["a_type"] if direction == "a_to_b" else pair["b_type"],
                    "destination_type": pair["b_type"] if direction == "a_to_b" else pair["a_type"],
                    "distance_m": distance_m,
                    "time_s": time_s,
                    "path_node_count": len(path),
                }
            )
    routes = pd.DataFrame(records)
    paired = routes.pivot(index="pair_id", columns="direction", values="time_s")
    asymmetry = (paired["a_to_b"] - paired["b_to_a"]).abs() / (
        0.5 * (paired["a_to_b"] + paired["b_to_a"])
    )
    csv_path = output_dir / "service_access_local_routes.csv"
    routes.to_csv(csv_path, index=False)
    report = {
        "schema": "evrptw_service_access_local_route_audit_v1",
        "status": "passed_exact_virtual_access_pilot",
        "generated_utc": datetime.now(UTC).isoformat(),
        "scenario_id": scenario_id,
        "sampling": {
            "seed": seed,
            "same_physical_edge_pair_count": pair_count,
            "directed_route_count": len(routes),
            "active_service_location_count": len(active),
            "eligibility": "G1/G2 candidate, <=200 m sensitivity band, reciprocal directed refs",
        },
        "materialization": materialization,
        "results": {
            "route_distance_m_p50": float(routes["distance_m"].median()),
            "route_time_s_p50": float(routes["time_s"].median()),
            "absolute_directional_time_difference_median_share": float(asymmetry.median()),
            "all_routes_positive_finite": bool(
                np.isfinite(routes[["distance_m", "time_s"]].to_numpy()).all()
                and (routes[["distance_m", "time_s"]] > 0).all().all()
            ),
        },
        "connector_policy": {
            "speed_kph": connector_speed_kph,
            "symmetry": "equal_both_directions",
            "status": "engineering QA value; official instance policy not frozen",
        },
        "inputs": {
            "graph_sha256": sha256_file(graph_path),
            "access_candidates_sha256": sha256_file(access_candidates_path),
            "service_nodes_sha256": sha256_file(service_nodes_path),
            "projection_nodes_sha256": sha256_file(projection_nodes_path),
            "connectors_sha256": sha256_file(connectors_path),
            "scenarios_sha256": sha256_file(scenarios_path),
        },
        "outputs": {"route_samples": csv_path.name},
        "output_sha256": {"route_samples": sha256_file(csv_path)},
    }
    if not report["results"]["all_routes_positive_finite"]:
        raise RuntimeError("Local access route QA produced invalid costs")
    write_json(output_dir / "service_access_local_route_audit.json", report)
    return report
