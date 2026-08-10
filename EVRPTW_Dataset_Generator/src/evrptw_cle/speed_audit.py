"""Route-level QA for static operational speed scenario banks."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import pandas as pd
import pyarrow.parquet as pq

from .util import sha256_file, write_json


def _resolve_node(graph, value: Any):
    lookup = getattr(graph, "_string_node_lookup", None)
    if lookup is None:
        lookup = {str(node): node for node in graph.nodes}
        graph._string_node_lookup = lookup
    return lookup.get(str(value))


def _resolve_key(graph, u, v, value: Any):
    if u not in graph or v not in graph[u]:
        return None
    for key in graph[u][v]:
        if str(key) == str(value):
            return key
    return None


def _set_scenario_weights(graph, scenario: pd.DataFrame) -> None:
    missing = 0
    for row in scenario.itertuples(index=False):
        u = _resolve_node(graph, row.edge_u)
        v = _resolve_node(graph, row.edge_v)
        key = _resolve_key(graph, u, v, row.edge_key) if u is not None and v is not None else None
        if key is None:
            missing += 1
            continue
        graph[u][v][key]["scenario_time_s"] = float(row.travel_time_s)
        graph[u][v][key]["scenario_speed_kph"] = float(row.speed_kph)
    if missing:
        raise ValueError(f"Scenario rows did not resolve to {missing} graph edges")


def _best_edge(graph, u, v, weight: str) -> dict[str, Any]:
    edges = graph[u][v]
    return min(
        edges.values(),
        key=lambda attributes: (
            float(attributes.get(weight, math.inf)),
            str(attributes.get("osmid", "")),
        ),
    )


def _path_costs(graph, path: list[Any]) -> tuple[float, float]:
    distance_m = 0.0
    time_s = 0.0
    for u, v in pairwise(path):
        attributes = _best_edge(graph, u, v, "scenario_time_s")
        distance_m += float(attributes["length"])
        time_s += float(attributes["scenario_time_s"])
    return distance_m, time_s


def _distance_path_time(graph, path: list[Any]) -> tuple[float, float]:
    distance_m = 0.0
    time_s = 0.0
    for u, v in pairwise(path):
        attributes = _best_edge(graph, u, v, "length")
        distance_m += float(attributes["length"])
        time_s += float(attributes["scenario_time_s"])
    return distance_m, time_s


def _path_hash(path: list[Any]) -> str:
    return hashlib.sha256("|".join(str(node) for node in path).encode("utf-8")).hexdigest()


def _select_od_pairs(
    graph,
    locations: pd.DataFrame,
    depots: pd.DataFrame,
    *,
    seed: int,
    pair_count: int,
) -> list[dict[str, Any]]:
    strong_component_by_node: dict[Any, int] = {}
    for component_id, component in enumerate(nx.strongly_connected_components(graph)):
        for node in component:
            strong_component_by_node[node] = component_id

    eligible_depots = depots.loc[depots["depot_candidate_eligible"].astype(bool)].copy()
    eligible_depots = eligible_depots.sort_values("candidate_rank")
    depot_rows_by_component: defaultdict[int, list[tuple[str, Any]]] = defaultdict(list)
    for row in eligible_depots.itertuples(index=False):
        node = _resolve_node(graph, row.road_anchor_node)
        if node is not None:
            depot_rows_by_component[strong_component_by_node[node]].append(
                (str(row.candidate_id), node)
            )
    if not depot_rows_by_component:
        raise ValueError("No depot candidate anchor resolves to the operational graph")

    if "cle_candidate_eligible" in locations.columns:
        location_eligible = locations["cle_candidate_eligible"].astype(bool)
    else:
        required = {"geometry_evidence_tier", "road_access_distance_m"}
        missing = required - set(locations.columns)
        if missing:
            raise ValueError(f"Location audit table lacks eligibility columns: {sorted(missing)}")
        location_eligible = locations["geometry_evidence_tier"].isin(
            {"G1_containment", "G2_near_area_consistent"}
        ) & (pd.to_numeric(locations["road_access_distance_m"], errors="coerce") <= 200.0)
    eligible_locations = locations.loc[location_eligible].copy()
    eligible_locations["resolved_node"] = pd.Series(
        [_resolve_node(graph, value) for value in eligible_locations["edge_u"]],
        index=eligible_locations.index,
        dtype=object,
    )
    eligible_locations = eligible_locations.dropna(subset=["resolved_node"])
    eligible_locations = eligible_locations.drop_duplicates("resolved_node")
    eligible_locations["strong_component_id"] = eligible_locations["resolved_node"].map(
        strong_component_by_node
    )
    eligible_locations = eligible_locations.loc[
        eligible_locations["strong_component_id"].isin(depot_rows_by_component)
    ].copy()
    if len(eligible_locations) < pair_count:
        raise ValueError(
            "Not enough distinct customer road anchors sharing a directed strong "
            "component with a depot candidate for route audit"
        )
    sampled = eligible_locations.sample(n=pair_count, random_state=seed).reset_index(drop=True)
    pairs = []
    for index, row in sampled.iterrows():
        matching_depots = depot_rows_by_component[int(row["strong_component_id"])]
        depot_id, depot_node = matching_depots[index % min(3, len(matching_depots))]
        pairs.append(
            {
                "pair_id": f"pair-{index:03d}",
                "depot_id": depot_id,
                "customer_id": str(row["latent_service_location_id"]),
                "depot_node": depot_node,
                "customer_node": row["resolved_node"],
                "customer_type": str(row["service_location_type"]),
            }
        )
    return pairs


def _render_summary(
    scenario_frame: pd.DataFrame,
    route_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    scenario_means = (
        scenario_frame.assign(weighted=lambda x: x["speed_kph"] * x["length_m"])
        .groupby("scenario_id")
        .agg(weighted_speed=("weighted", "sum"), length=("length_m", "sum"))
    )
    scenario_means["mean_speed_kph"] = scenario_means["weighted_speed"] / scenario_means["length"]
    scenario_means["mean_speed_kph"].plot.bar(ax=axes[0, 0], color="#2878b5")
    axes[0, 0].set_title("Length-weighted edge speed")
    axes[0, 0].set_ylabel("km/h")
    axes[0, 0].tick_params(axis="x", rotation=25)

    route_frame.boxplot(column="route_average_speed_kph", by="scenario_id", ax=axes[0, 1])
    axes[0, 1].set_title("Sampled fastest-route average speed")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("km/h")
    axes[0, 1].tick_params(axis="x", rotation=25)

    paired = route_frame.pivot_table(
        index=["scenario_id", "pair_id"], columns="direction", values="fastest_time_s"
    ).dropna()
    relative = (paired["depot_to_customer"] - paired["customer_to_depot"]).abs() / (
        0.5 * (paired["depot_to_customer"] + paired["customer_to_depot"])
    )
    axes[1, 0].hist(relative * 100.0, bins=12, color="#f28e2b", edgecolor="white")
    axes[1, 0].set_title("Directed OD time asymmetry")
    axes[1, 0].set_xlabel("absolute directional difference (%)")
    axes[1, 0].set_ylabel("OD-scenario count")

    path_counts = route_frame.groupby("directed_od_id")["fastest_path_hash"].nunique()
    path_counts.value_counts().sort_index().plot.bar(ax=axes[1, 1], color="#59a14f")
    axes[1, 1].set_title("Unique fastest paths across scenarios")
    axes[1, 1].set_xlabel("unique path count")
    axes[1, 1].set_ylabel("directed ODs")
    fig.suptitle("Static operational speed pilot QA", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_static_speed_route_audit(
    *,
    graph_path: Path,
    locations_path: Path,
    depots_path: Path,
    scenarios_path: Path,
    output_dir: Path,
    seed: int = 20270805,
    pair_count: int = 12,
) -> dict[str, Any]:
    """Audit route-level effects without treating pilot speeds as observed traffic."""

    output_dir.mkdir(parents=True, exist_ok=True)
    graph = ox.load_graphml(graph_path)
    location_schema = set(pq.read_schema(locations_path).names)
    location_columns = {
        "edge_u",
        "latent_service_location_id",
        "service_location_type",
    }
    if "cle_candidate_eligible" in location_schema:
        location_columns.add("cle_candidate_eligible")
    else:
        location_columns.update({"geometry_evidence_tier", "road_access_distance_m"})
    missing_location_columns = location_columns - location_schema
    if missing_location_columns:
        raise ValueError(
            "Location audit table lacks route-sampling columns: "
            f"{sorted(missing_location_columns)}"
        )
    # Large-city access ledgers can contain hundreds of thousands of rich rows.
    # Route QA only needs anchor and eligibility fields, so avoid materializing
    # geometry, directed-edge ledgers, and connector metadata into memory.
    locations = pd.read_parquet(locations_path, columns=sorted(location_columns))
    depots = pd.read_parquet(depots_path)
    scenarios = pd.read_parquet(scenarios_path)
    pairs = _select_od_pairs(graph, locations, depots, seed=seed, pair_count=pair_count)

    distance_paths: dict[str, list[Any]] = {}
    for pair in pairs:
        for direction, source, target in (
            ("depot_to_customer", pair["depot_node"], pair["customer_node"]),
            ("customer_to_depot", pair["customer_node"], pair["depot_node"]),
        ):
            directed_od_id = f"{pair['pair_id']}|{direction}"
            distance_paths[directed_od_id] = nx.shortest_path(
                graph, source=source, target=target, weight="length", method="dijkstra"
            )

    records: list[dict[str, Any]] = []
    for scenario_id, scenario in scenarios.groupby("scenario_id", sort=True):
        _set_scenario_weights(graph, scenario)
        day_type = str(scenario["day_type"].iloc[0])
        for pair in pairs:
            for direction, source, target in (
                ("depot_to_customer", pair["depot_node"], pair["customer_node"]),
                ("customer_to_depot", pair["customer_node"], pair["depot_node"]),
            ):
                directed_od_id = f"{pair['pair_id']}|{direction}"
                fastest_path = nx.shortest_path(
                    graph,
                    source=source,
                    target=target,
                    weight="scenario_time_s",
                    method="dijkstra",
                )
                fastest_distance_m, fastest_time_s = _path_costs(graph, fastest_path)
                distance_path_m, distance_path_time_s = _distance_path_time(
                    graph, distance_paths[directed_od_id]
                )
                records.append(
                    {
                        "scenario_id": str(scenario_id),
                        "day_type": day_type,
                        "pair_id": pair["pair_id"],
                        "directed_od_id": directed_od_id,
                        "direction": direction,
                        "depot_id": pair["depot_id"],
                        "customer_id": pair["customer_id"],
                        "customer_type": pair["customer_type"],
                        "distance_optimal_distance_m": distance_path_m,
                        "distance_optimal_time_s": distance_path_time_s,
                        "fastest_distance_m": fastest_distance_m,
                        "fastest_time_s": fastest_time_s,
                        "route_average_speed_kph": (fastest_distance_m / fastest_time_s) * 3.6,
                        "fastest_distance_premium_pct": (
                            (fastest_distance_m / distance_path_m - 1.0) * 100.0
                        ),
                        "fastest_time_saving_pct": (
                            (1.0 - fastest_time_s / distance_path_time_s) * 100.0
                        ),
                        "fastest_path_hash": _path_hash(fastest_path),
                        "fastest_path_node_count": len(fastest_path),
                    }
                )
    routes = pd.DataFrame(records)
    paired = routes.pivot_table(
        index=["scenario_id", "pair_id"], columns="direction", values="fastest_time_s"
    ).dropna()
    relative_asymmetry = (
        (paired["depot_to_customer"] - paired["customer_to_depot"]).abs()
        / (0.5 * (paired["depot_to_customer"] + paired["customer_to_depot"]))
    )
    path_counts = routes.groupby("directed_od_id")["fastest_path_hash"].nunique()
    report = {
        "schema": "evrptw_static_speed_route_audit_v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "status": "passed_pilot_route_qa_not_external_calibration",
        "inputs": {
            "graph": {"path": str(graph_path.resolve()), "sha256": sha256_file(graph_path)},
            "locations": {
                "path": str(locations_path.resolve()),
                "sha256": sha256_file(locations_path),
            },
            "depots": {"path": str(depots_path.resolve()), "sha256": sha256_file(depots_path)},
            "scenarios": {
                "path": str(scenarios_path.resolve()),
                "sha256": sha256_file(scenarios_path),
            },
        },
        "sampling": {
            "seed": seed,
            "customer_depot_reachability_rule": "same_directed_strong_component",
            "physical_od_pair_count": pair_count,
            "directed_od_count": int(routes["directed_od_id"].nunique()),
            "scenario_count": int(routes["scenario_id"].nunique()),
            "route_record_count": len(routes),
        },
        "results": {
            "route_average_speed_kph": {
                "p10": float(routes["route_average_speed_kph"].quantile(0.10)),
                "p50": float(routes["route_average_speed_kph"].median()),
                "p90": float(routes["route_average_speed_kph"].quantile(0.90)),
            },
            "absolute_directional_time_difference": {
                "median_share": float(relative_asymmetry.median()),
                "p90_share": float(relative_asymmetry.quantile(0.90)),
                "nonzero_share": float((relative_asymmetry > 1e-9).mean()),
            },
            "directed_ods_with_scenario_dependent_fastest_path_share": float(
                (path_counts > 1).mean()
            ),
            "fastest_path_distance_premium_pct_p50": float(
                routes["fastest_distance_premium_pct"].median()
            ),
            "fastest_path_time_saving_pct_p50": float(
                routes["fastest_time_saving_pct"].median()
            ),
        },
        "interpretation_limit": (
            "This checks internal directed-routing behavior only. It does not validate pilot "
            "variation parameters against observed local traffic."
        ),
        "routing_note": (
            "QA uses deterministic NetworkX Dijkstra on the existing MultiDiGraph. The formal "
            "path catalog still needs edge-key-level tie-breaking and turn-cost integration."
        ),
    }
    csv_path = output_dir / "static_speed_route_samples.csv"
    json_path = output_dir / "static_speed_route_audit.json"
    png_path = output_dir / "static_speed_route_audit.png"
    routes.to_csv(csv_path, index=False)
    _render_summary(scenarios, routes, png_path)
    report["outputs"] = {
        "route_samples": csv_path.name,
        "summary_plot": png_path.name,
    }
    report["output_sha256"] = {
        "route_samples": sha256_file(csv_path),
        "summary_plot": sha256_file(png_path),
    }
    write_json(json_path, report)
    return report
