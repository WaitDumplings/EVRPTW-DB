"""Amazon-structured multi-region customer activation for Stage 2."""

from __future__ import annotations

import math
import resource
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .rounding import (
    balanced_cell_partition,
    controlled_matrix_round,
    largest_remainder,
    stable_u64,
)


class SpatialActivationError(ValueError):
    """A sampled family cannot satisfy the frozen spatial contract."""

    def __init__(self, code: str, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class SpatialActivationResult:
    customers: pd.DataFrame
    assignment: pd.DataFrame
    radial_baseline: pd.DataFrame
    metadata: dict[str, Any]


def _quota_matrix(
    structure_targets: pd.DataFrame,
    *,
    customer_count: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    required = {"route_id", "radial_decile", "station_to_stop_time_s"}
    missing = required - set(structure_targets.columns)
    if missing:
        raise ValueError(f"Structure targets lack columns: {sorted(missing)}")
    source = structure_targets.loc[
        structure_targets["radial_decile"].between(0, 9)
    ].copy()
    if len(source) < customer_count:
        raise SpatialActivationError(
            "PF2_STRUCTURE_UNSUPPORTED",
            f"structure source contains {len(source)} stops for N={customer_count}",
        )
    route_ids = sorted(source["route_id"].astype(str).unique())
    counts = (
        source.assign(route_id=source["route_id"].astype(str))
        .groupby(["route_id", "radial_decile"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=route_ids, columns=range(10), fill_value=0)
        .to_numpy(dtype=float)
    )
    scale_factor = float(customer_count / counts.sum())
    if scale_factor > 1.0 + 1e-12:
        raise SpatialActivationError(
            "PF2_STRUCTURE_UPSCALING_FORBIDDEN",
            f"structure scaling factor {scale_factor:.6f} exceeds one",
        )
    fractional = counts * scale_factor
    row_targets = largest_remainder(
        fractional.sum(axis=1),
        total=customer_count,
        seed=seed,
        namespace="structure_row_margin",
        labels=route_ids,
    )
    column_targets = largest_remainder(
        fractional.sum(axis=0),
        total=customer_count,
        seed=seed,
        namespace="structure_column_margin",
        labels=[f"d{index}" for index in range(10)],
    )
    rounded = controlled_matrix_round(
        fractional,
        row_targets=row_targets,
        column_targets=column_targets,
        seed=seed,
        namespace="structure_route_decile",
        row_labels=route_ids,
        column_labels=[f"d{index}" for index in range(10)],
    )
    records: list[dict[str, Any]] = []
    target_times: list[float] = []
    dropped: list[str] = []
    for row_index, route_id in enumerate(route_ids):
        route_total = int(rounded[row_index].sum())
        if not route_total:
            dropped.append(route_id)
            continue
        for decile in range(10):
            quota = int(rounded[row_index, decile])
            if not quota:
                continue
            cell = source.loc[
                source["route_id"].astype(str).eq(route_id)
                & source["radial_decile"].eq(decile)
            ].copy()
            cell["_rank"] = [
                stable_u64(seed, "structure_target", route_id, decile, value)
                for value in cell.get("template_id", cell.index.astype(str))
            ]
            chosen = cell.sort_values(
                ["_rank", "station_to_stop_time_s"], kind="stable"
            ).head(quota)
            if len(chosen) != quota:
                raise AssertionError("Rounded structure quota exceeds observed cell support")
            target_times.extend(chosen["station_to_stop_time_s"].astype(float).tolist())
            records.append(
                {
                    "region_id": f"region_{row_index:03d}",
                    "structure_route_id": route_id,
                    "radial_decile": decile,
                    "quota": quota,
                    "region_quota": route_total,
                }
            )
    quotas = pd.DataFrame.from_records(records)
    if int(quotas["quota"].sum()) != customer_count:
        raise AssertionError("Region-decile quotas do not sum to N")
    return (
        quotas,
        {
            "scale_factor": scale_factor,
            "source_stop_count": len(source),
            "source_route_count": len(route_ids),
            "retained_region_count": int(quotas["region_id"].nunique()),
            "routes_dropped_by_rounding": dropped,
            "row_margins_exact": True,
            "column_margins_exact": True,
            "radial_decile_targets": [int(value) for value in column_targets],
        },
        np.asarray(target_times, dtype=float),
    )


def _community_graph(adjacency: pd.DataFrame, community_ids: set[str]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(community_ids))
    if adjacency.empty:
        return graph
    required = {"source_community_id", "target_community_id"}
    missing = required - set(adjacency.columns)
    if missing:
        raise ValueError(f"Community adjacency lacks columns: {sorted(missing)}")
    time_column = (
        "crossing_time_s" if "crossing_time_s" in adjacency else "crossing_length_m"
    )
    # Preserve zero-customer transit communities. They cannot become region
    # seeds, but they may be the only real-road bridge between two customer
    # communities.
    graph.add_nodes_from(
        sorted(
            set(adjacency["source_community_id"].astype(str))
            | set(adjacency["target_community_id"].astype(str))
        )
    )
    for row in adjacency.itertuples(index=False):
        source = str(row.source_community_id)
        target = str(row.target_community_id)
        if source == target:
            continue
        value = float(getattr(row, time_column))
        previous = graph.get_edge_data(source, target, {}).get("weight", math.inf)
        if value < previous:
            graph.add_edge(source, target, weight=value)
    return graph


def _seed_decile(region_quotas: pd.DataFrame) -> int:
    ordered = region_quotas.sort_values("radial_decile")
    threshold = float(ordered["quota"].sum()) / 2.0
    return int(
        ordered.loc[ordered["quota"].cumsum().ge(threshold), "radial_decile"].iloc[0]
    )


def _fallback_deciles(target: int) -> list[int]:
    result = [target]
    for delta in range(1, 10):
        inner = target - delta
        outer = target + delta
        if inner >= 0:
            result.append(inner)
        if outer <= 9:
            result.append(outer)
    return result


def _symmetrized_distances(
    graph: nx.DiGraph,
    reverse: nx.DiGraph,
    source: str,
) -> dict[str, float]:
    forward = nx.single_source_dijkstra_path_length(graph, source, weight="weight")
    backward = nx.single_source_dijkstra_path_length(reverse, source, weight="weight")
    return {
        target: (float(forward[target]) + float(backward[target])) / 2.0
        for target in forward.keys() & backward.keys()
    }


def _choose_region_seeds(
    customers: pd.DataFrame,
    quotas: pd.DataFrame,
    graph: nx.DiGraph,
    *,
    seed: int,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    capacity = (
        customers.groupby(["community_id", "radial_decile"], observed=True)
        .size()
        .rename("capacity")
    )
    region_sizes = quotas.groupby("region_id", sort=False)["quota"].sum().to_dict()
    route_ids = quotas.groupby("region_id", sort=False)["structure_route_id"].first().to_dict()
    order = sorted(
        region_sizes,
        key=lambda region: (
            -int(region_sizes[region]),
            stable_u64(seed, "region_seed_order", route_ids[region]),
            region,
        ),
    )
    reverse = graph.reverse(copy=False)
    selected: dict[str, str] = {}
    distance_cache: dict[str, dict[str, float]] = {}
    fallback_events: list[dict[str, Any]] = []
    for region_index, region in enumerate(order):
        if progress_callback is not None:
            progress_callback(
                "region_seed_selection.progress",
                {
                    "status": "progress",
                    "region_index": int(region_index),
                    "region_count": int(len(order)),
                    "selected_seed_count": int(len(selected)),
                },
            )
        target = _seed_decile(quotas.loc[quotas["region_id"].eq(region)])
        candidates: list[str] = []
        chosen_decile = target
        for decile in _fallback_deciles(target):
            candidates = sorted(
                community
                for community in graph.nodes
                if int(capacity.get((community, decile), 0)) > 0
                and community not in set(selected.values())
            )
            if candidates:
                chosen_decile = decile
                break
        if not candidates:
            raise SpatialActivationError(
                "REGION_SEED_UNAVAILABLE",
                f"no seed community is available for {region}",
            )
        if not selected:
            chosen = min(
                candidates,
                key=lambda community: (
                    -int(capacity.get((community, chosen_decile), 0)),
                    stable_u64(seed, "first_region_seed", region, community),
                    community,
                ),
            )
        else:
            for existing in selected.values():
                distance_cache.setdefault(
                    existing, _symmetrized_distances(graph, reverse, existing)
                )
            chosen = max(
                candidates,
                key=lambda community: (
                    min(
                        distance_cache[existing].get(community, -math.inf)
                        for existing in selected.values()
                    ),
                    -stable_u64(seed, "maxmin_region_seed", region, community),
                    community,
                ),
            )
        selected[region] = chosen
        if progress_callback is not None:
            progress_callback(
                "region_seed_selection.progress",
                {
                    "status": "progress",
                    "region_index": int(region_index + 1),
                    "region_count": int(len(order)),
                    "selected_seed_count": int(len(selected)),
                },
            )
        if chosen_decile != target:
            fallback_events.append(
                {
                    "region_id": region,
                    "target_decile": target,
                    "selected_decile": chosen_decile,
                    "seed_community_id": chosen,
                }
            )
    return selected, fallback_events


def _region_unmet_cells(
    region: str,
    communities: set[str],
    quotas: pd.DataFrame,
    customers: pd.DataFrame,
) -> set[int]:
    target = {
        int(row.radial_decile): int(row.quota)
        for row in quotas.loc[quotas["region_id"].eq(region)].itertuples(index=False)
    }
    capacity = (
        customers.loc[customers["community_id"].isin(communities)]
        .groupby("radial_decile", observed=True)
        .size()
        .to_dict()
    )
    return {
        decile for decile, quota in target.items() if int(capacity.get(decile, 0)) < quota
    }


def _neighbor_score(
    region: str,
    candidate: str,
    regions: dict[str, set[str]],
    unmet: set[int],
    customers: pd.DataFrame,
    graph: nx.DiGraph,
    seed: int,
) -> tuple[int, float, int, str]:
    deciles = set(
        customers.loc[customers["community_id"].eq(candidate), "radial_decile"].astype(int)
    )
    helps = bool(deciles & unmet)
    crossing: list[float] = []
    for member in regions[region]:
        if graph.has_edge(member, candidate):
            crossing.append(float(graph[member][candidate]["weight"]))
        if graph.has_edge(candidate, member):
            crossing.append(float(graph[candidate][member]["weight"]))
    return (
        0 if helps else 1,
        min(crossing) if crossing else math.inf,
        stable_u64(seed, "region_growth", region, candidate),
        candidate,
    )


def _grow_regions_reference(
    customers: pd.DataFrame,
    quotas: pd.DataFrame,
    graph: nx.DiGraph,
    seeds: dict[str, str],
    *,
    seed: int,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    decision_trace: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, set[str]], int]:
    """Original DataFrame-scan implementation retained for differential tests."""

    regions = {region: {community} for region, community in seeds.items()}
    growth_steps = 0
    maximum_steps = max(1, len(graph) * len(regions))
    order = sorted(
        regions,
        key=lambda region: (
            -int(quotas.loc[quotas["region_id"].eq(region), "quota"].sum()),
            stable_u64(seed, "region_growth_order", region),
        ),
    )
    growth_pass = 0
    last_reported_growth = -1
    while growth_steps < maximum_steps:
        changed = False
        all_satisfied = True
        for region_index, region in enumerate(order):
            if (
                progress_callback is not None
                and growth_steps != last_reported_growth
                and (growth_steps < 5 or growth_steps % 100 == 0)
            ):
                progress_callback(
                    "region_growth.progress",
                    {
                        "status": "progress",
                        "growth_pass": int(growth_pass),
                        "growth_steps": int(growth_steps),
                        "maximum_steps": int(maximum_steps),
                        "region_index": int(region_index),
                        "region_count": int(len(order)),
                    },
                )
                last_reported_growth = growth_steps
            unmet = _region_unmet_cells(region, regions[region], quotas, customers)
            if not unmet:
                continue
            all_satisfied = False
            neighbors: set[str] = set()
            for community in regions[region]:
                neighbors.update(graph.successors(community))
                neighbors.update(graph.predecessors(community))
            neighbors -= regions[region]
            if not neighbors:
                continue
            chosen = min(
                neighbors,
                key=lambda candidate: _neighbor_score(
                    region,
                    candidate,
                    regions,
                    unmet,
                    customers,
                    graph,
                    seed,
                ),
            )
            if decision_trace is not None:
                decision_trace.append(
                    {
                        "growth_pass": int(growth_pass),
                        "growth_step": int(growth_steps),
                        "region_id": region,
                        "unmet_deciles": sorted(unmet),
                        "chosen_community_id": chosen,
                        "selection_key": _neighbor_score(
                            region,
                            chosen,
                            regions,
                            unmet,
                            customers,
                            graph,
                            seed,
                        ),
                    }
                )
            regions[region].add(chosen)
            growth_steps += 1
            changed = True
        growth_pass += 1
        if all_satisfied:
            return regions, growth_steps
        if not changed:
            break
    deficits = {
        region: sorted(_region_unmet_cells(region, communities, quotas, customers))
        for region, communities in regions.items()
    }
    deficits = {key: value for key, value in deficits.items() if value}
    raise SpatialActivationError(
        "REGION_GROWTH_EXHAUSTED",
        "community graph was exhausted before all region-decile capacities were met",
        {"unmet_region_deciles": deficits, "growth_steps": growth_steps},
    )


def _growth_capacity_tables(
    customers: pd.DataFrame,
    quotas: pd.DataFrame,
) -> tuple[
    dict[Any, dict[int, int]],
    dict[Any, set[int]],
    dict[str, dict[int, int]],
    dict[str, int],
]:
    """Build exact family-local sufficient statistics with one customer groupby."""

    community_capacity: dict[Any, dict[int, int]] = {}
    community_support: dict[Any, set[int]] = {}
    grouped = customers.groupby(
        ["community_id", "radial_decile"], observed=True, sort=False
    ).size()
    for (community, decile), count in grouped.items():
        decile_id = int(decile)
        community_capacity.setdefault(community, {})[decile_id] = int(count)
        community_support.setdefault(community, set()).add(decile_id)

    region_quota: dict[str, dict[int, int]] = {}
    region_total: dict[str, int] = {}
    for row in quotas.itertuples(index=False):
        region = str(row.region_id)
        region_quota.setdefault(region, {})[int(row.radial_decile)] = int(row.quota)
        region_total[region] = region_total.get(region, 0) + int(row.quota)
    return community_capacity, community_support, region_quota, region_total


def _incident_crossing_minimums(graph: nx.DiGraph) -> dict[str, dict[str, float]]:
    """Return the exact minimum directed crossing weight for each adjacent pair."""

    result: dict[str, dict[str, float]] = {str(node): {} for node in graph.nodes}
    for source_raw, target_raw, data in graph.edges(data=True):
        source = str(source_raw)
        target = str(target_raw)
        if source == target:
            continue
        weight = float(data["weight"])
        previous = result[source].get(target, math.inf)
        reverse_previous = result[target].get(source, math.inf)
        pair_minimum = min(weight, previous, reverse_previous)
        result[source][target] = pair_minimum
        result[target][source] = pair_minimum
    return result


def _grow_regions(
    customers: pd.DataFrame,
    quotas: pd.DataFrame,
    graph: nx.DiGraph,
    seeds: dict[str, str],
    *,
    seed: int,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    decision_trace: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, set[str]], int]:
    """Grow regions with exact incremental capacity, frontier and crossing caches."""

    (
        community_capacity,
        community_support,
        region_quota,
        region_total,
    ) = _growth_capacity_tables(customers, quotas)
    incident_minimums = _incident_crossing_minimums(graph)
    regions = {region: {community} for region, community in seeds.items()}
    region_capacity: dict[str, dict[int, int]] = {}
    frontiers: dict[str, set[str]] = {}
    crossing_minimums: dict[str, dict[str, float]] = {}
    for region, members in regions.items():
        capacity: dict[int, int] = {}
        frontier: set[str] = set()
        crossing: dict[str, float] = {}
        for member in members:
            for decile, count in community_capacity.get(member, {}).items():
                capacity[decile] = capacity.get(decile, 0) + count
            for candidate, weight in incident_minimums.get(member, {}).items():
                if candidate in members:
                    continue
                frontier.add(candidate)
                crossing[candidate] = min(crossing.get(candidate, math.inf), weight)
        region_capacity[region] = capacity
        frontiers[region] = frontier
        crossing_minimums[region] = crossing

    growth_steps = 0
    maximum_steps = max(1, len(graph) * len(regions))
    order = sorted(
        regions,
        key=lambda region: (
            -int(region_total[region]),
            stable_u64(seed, "region_growth_order", region),
        ),
    )
    growth_pass = 0
    last_reported_growth = -1
    while growth_steps < maximum_steps:
        changed = False
        all_satisfied = True
        for region_index, region in enumerate(order):
            if (
                progress_callback is not None
                and growth_steps != last_reported_growth
                and (growth_steps < 5 or growth_steps % 100 == 0)
            ):
                progress_callback(
                    "region_growth.progress",
                    {
                        "status": "progress",
                        "growth_pass": int(growth_pass),
                        "growth_steps": int(growth_steps),
                        "maximum_steps": int(maximum_steps),
                        "region_index": int(region_index),
                        "region_count": int(len(order)),
                    },
                )
                last_reported_growth = growth_steps
            unmet = {
                decile
                for decile, quota in region_quota[region].items()
                if int(region_capacity[region].get(decile, 0)) < quota
            }
            if not unmet:
                continue
            all_satisfied = False
            neighbors = frontiers[region]
            if not neighbors:
                continue

            def selection_key(candidate: str) -> tuple[int, float, int, str]:
                return (
                    0 if community_support.get(candidate, set()) & unmet else 1,
                    crossing_minimums[region].get(candidate, math.inf),
                    stable_u64(seed, "region_growth", region, candidate),
                    candidate,
                )

            chosen = min(neighbors, key=selection_key)
            chosen_key = selection_key(chosen)
            if decision_trace is not None:
                decision_trace.append(
                    {
                        "growth_pass": int(growth_pass),
                        "growth_step": int(growth_steps),
                        "region_id": region,
                        "unmet_deciles": sorted(unmet),
                        "chosen_community_id": chosen,
                        "selection_key": chosen_key,
                    }
                )
            regions[region].add(chosen)
            for decile, count in community_capacity.get(chosen, {}).items():
                region_capacity[region][decile] = (
                    region_capacity[region].get(decile, 0) + count
                )
            neighbors.discard(chosen)
            crossing_minimums[region].pop(chosen, None)
            for candidate, weight in incident_minimums.get(chosen, {}).items():
                if candidate in regions[region]:
                    continue
                neighbors.add(candidate)
                crossing_minimums[region][candidate] = min(
                    crossing_minimums[region].get(candidate, math.inf),
                    weight,
                )
            growth_steps += 1
            changed = True
        growth_pass += 1
        if all_satisfied:
            return regions, growth_steps
        if not changed:
            break
    deficits = {
        region: sorted(
            decile
            for decile, quota in region_quota[region].items()
            if int(region_capacity[region].get(decile, 0)) < quota
        )
        for region in regions
    }
    deficits = {key: value for key, value in deficits.items() if value}
    raise SpatialActivationError(
        "REGION_GROWTH_EXHAUSTED",
        "community graph was exhausted before all region-decile capacities were met",
        {"unmet_region_deciles": deficits, "growth_steps": growth_steps},
    )


def _weighted_key(seed: int, region: str, customer_id: str, units: float) -> float:
    integer = stable_u64(seed, "weighted_customer", region, customer_id)
    uniform = (integer + 1.0) / (2**64 + 1.0)
    return -math.log(uniform) / max(float(units), 1.0)


def _assign_customers(
    customers: pd.DataFrame,
    quotas: pd.DataFrame,
    regions: dict[str, set[str]],
    *,
    seed: int,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> pd.DataFrame:
    def begin(stage: str, **details: Any) -> tuple[float, float]:
        if progress_callback is not None:
            progress_callback(stage, {"status": "started", **details})
        return time.perf_counter(), time.process_time()

    def finish(
        stage: str,
        started: tuple[float, float],
        *,
        status: str = "completed",
        **details: Any,
    ) -> None:
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if progress_callback is not None:
            progress_callback(
                stage,
                {
                    "status": status,
                    "wall_seconds": time.perf_counter() - started[0],
                    "cpu_seconds": time.process_time() - started[1],
                    "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
                    **details,
                },
            )

    stage_started = begin(
        "global_assignment.graph_build",
        quota_cell_count=len(quotas),
        territory_count=len(customers),
    )
    graph = nx.DiGraph()
    total = int(quotas["quota"].sum())
    graph.add_node("source", demand=-total)
    graph.add_node("sink", demand=total)
    customer_lookup = customers.set_index("latent_service_location_id", drop=False)
    candidate_edge_count = 0
    for cell_index, row in enumerate(quotas.itertuples(index=False)):
        cell = f"cell:{row.region_id}:{int(row.radial_decile)}"
        graph.add_node(cell, demand=0)
        graph.add_edge("source", cell, capacity=int(row.quota), weight=0)
        candidates = customers.loc[
            customers["community_id"].isin(regions[str(row.region_id)])
            & customers["radial_decile"].eq(int(row.radial_decile))
        ]
        for candidate in candidates.itertuples(index=False):
            customer_id = str(candidate.latent_service_location_id)
            node = f"customer:{customer_id}"
            if node not in graph:
                graph.add_node(node, demand=0)
                graph.add_edge(node, "sink", capacity=1, weight=0)
            priority = _weighted_key(
                seed,
                str(row.region_id),
                customer_id,
                float(candidate.residential_units),
            )
            graph.add_edge(cell, node, capacity=1, weight=round(priority * 1e9))
            candidate_edge_count += 1
        if progress_callback is not None and (
            cell_index == 0 or (cell_index + 1) % 10 == 0 or cell_index + 1 == len(quotas)
        ):
            progress_callback(
                "global_assignment.graph_build.progress",
                {
                    "status": "progress",
                    "completed_quota_cells": int(cell_index + 1),
                    "quota_cell_count": len(quotas),
                    "candidate_edge_count": int(candidate_edge_count),
                    "graph_node_count": graph.number_of_nodes(),
                },
            )
    finish(
        "global_assignment.graph_build",
        stage_started,
        graph_node_count=graph.number_of_nodes(),
        graph_edge_count=graph.number_of_edges(),
        candidate_edge_count=candidate_edge_count,
    )
    stage_started = begin(
        "global_assignment.feasibility",
        graph_node_count=graph.number_of_nodes(),
        graph_edge_count=graph.number_of_edges(),
        required_flow=total,
    )
    feasible_flow = int(
        nx.maximum_flow_value(
            graph,
            "source",
            "sink",
            capacity="capacity",
            flow_func=nx.algorithms.flow.shortest_augmenting_path,
        )
    )
    if feasible_flow != total:
        finish(
            "global_assignment.feasibility",
            stage_started,
            status="failed",
            feasible_flow=feasible_flow,
            required_flow=total,
            error_code="GLOBAL_CUSTOMER_ASSIGNMENT_INFEASIBLE",
        )
        raise SpatialActivationError(
            "GLOBAL_CUSTOMER_ASSIGNMENT_INFEASIBLE",
            "overlapping regions cannot satisfy all quotas with globally unique locations",
        )
    finish(
        "global_assignment.feasibility",
        stage_started,
        feasible_flow=feasible_flow,
        required_flow=total,
    )
    stage_started = begin(
        "global_assignment.min_cost_flow",
        graph_node_count=graph.number_of_nodes(),
        graph_edge_count=graph.number_of_edges(),
        required_flow=total,
    )
    try:
        flow = nx.min_cost_flow(graph)
    except nx.NetworkXUnfeasible as error:
        finish(
            "global_assignment.min_cost_flow",
            stage_started,
            status="failed",
            error_code="GLOBAL_CUSTOMER_ASSIGNMENT_INFEASIBLE",
        )
        raise SpatialActivationError(
            "GLOBAL_CUSTOMER_ASSIGNMENT_INFEASIBLE",
            "overlapping regions cannot satisfy all quotas with globally unique locations",
        ) from error
    finish("global_assignment.min_cost_flow", stage_started)
    stage_started = begin("global_assignment.result_extract")
    records: list[dict[str, Any]] = []
    quota_lookup = quotas.set_index(["region_id", "radial_decile"])
    for row in quotas.itertuples(index=False):
        cell = f"cell:{row.region_id}:{int(row.radial_decile)}"
        chosen = sorted(
            node.removeprefix("customer:")
            for node, value in flow[cell].items()
            if node.startswith("customer:") and int(value) == 1
        )
        if len(chosen) != int(row.quota):
            raise AssertionError("Global customer assignment did not fill a quota cell")
        for customer_id in chosen:
            customer = customer_lookup.loc[customer_id]
            records.append(
                {
                    "latent_service_location_id": customer_id,
                    "sampling_cluster_id": str(row.region_id),
                    "structure_route_id": str(row.structure_route_id),
                    "activation_decile": int(row.radial_decile),
                    "community_id": str(customer["community_id"]),
                    "residential_units": int(customer["residential_units"]),
                    "depot_running_time_s": float(customer["depot_running_time_s"]),
                }
            )
    assignment = pd.DataFrame.from_records(records)
    if len(assignment) != total or assignment["latent_service_location_id"].duplicated().any():
        raise AssertionError("Global assignment violated exact count or uniqueness")
    actual = assignment.groupby(
        ["sampling_cluster_id", "activation_decile"], observed=True
    ).size()
    for key, target in quota_lookup["quota"].items():
        if int(actual.get(key, 0)) != int(target):
            raise AssertionError(f"Activation quota mismatch for {key}")
    finish(
        "global_assignment.result_extract",
        stage_started,
        assignment_count=len(assignment),
    )
    return assignment


def _assign_with_competition_expansion_reference(
    customers: pd.DataFrame,
    quotas: pd.DataFrame,
    regions: dict[str, set[str]],
    graph: nx.DiGraph,
    *,
    seed: int,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    decision_trace: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Original competition expansion retained for differential tests."""

    expansions = 0
    maximum_expansions = max(1, len(graph) * len(regions))
    order = sorted(
        regions,
        key=lambda region: (
            -int(quotas.loc[quotas["region_id"].eq(region), "quota"].sum()),
            stable_u64(seed, "competition_expansion_order", region),
        ),
    )
    while True:
        try:
            return (
                _assign_customers(
                    customers,
                    quotas,
                    regions,
                    seed=seed,
                    progress_callback=progress_callback,
                ),
                expansions,
            )
        except SpatialActivationError as error:
            if error.code != "GLOBAL_CUSTOMER_ASSIGNMENT_INFEASIBLE":
                raise
        if expansions >= maximum_expansions:
            raise SpatialActivationError(
                "GLOBAL_ASSIGNMENT_EXPANSION_EXHAUSTED",
                "global assignment remained infeasible after graph-wide expansion",
            )
        changed = False
        for region in order:
            target_deciles = set(
                quotas.loc[quotas["region_id"].eq(region), "radial_decile"].astype(int)
            )
            neighbors: set[str] = set()
            for community in regions[region]:
                neighbors.update(graph.successors(community))
                neighbors.update(graph.predecessors(community))
            neighbors -= regions[region]
            if not neighbors:
                continue
            chosen = min(
                neighbors,
                key=lambda candidate: _neighbor_score(
                    region,
                    candidate,
                    regions,
                    target_deciles,
                    customers,
                    graph,
                    seed,
                ),
            )
            if decision_trace is not None:
                decision_trace.append(
                    {
                        "expansion_step": int(expansions),
                        "region_id": region,
                        "chosen_community_id": chosen,
                        "selection_key": _neighbor_score(
                            region,
                            chosen,
                            regions,
                            target_deciles,
                            customers,
                            graph,
                            seed,
                        ),
                    }
                )
            regions[region].add(chosen)
            expansions += 1
            changed = True
        if not changed:
            raise SpatialActivationError(
                "GLOBAL_ASSIGNMENT_EXPANSION_EXHAUSTED",
                "all regions cover their reachable graph but assignment is infeasible",
            )


def _assign_with_competition_expansion(
    customers: pd.DataFrame,
    quotas: pd.DataFrame,
    regions: dict[str, set[str]],
    graph: nx.DiGraph,
    *,
    seed: int,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    decision_trace: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Expand competition regions with exact incremental graph-local caches."""

    _, community_support, region_quota, region_total = _growth_capacity_tables(
        customers, quotas
    )
    incident_minimums = _incident_crossing_minimums(graph)
    frontiers: dict[str, set[str]] = {}
    crossing_minimums: dict[str, dict[str, float]] = {}
    for region, members in regions.items():
        frontier: set[str] = set()
        crossing: dict[str, float] = {}
        for member in members:
            for candidate, weight in incident_minimums.get(member, {}).items():
                if candidate in members:
                    continue
                frontier.add(candidate)
                crossing[candidate] = min(crossing.get(candidate, math.inf), weight)
        frontiers[region] = frontier
        crossing_minimums[region] = crossing

    expansions = 0
    expansion_round = 0
    maximum_expansions = max(1, len(graph) * len(regions))
    order = sorted(
        regions,
        key=lambda region: (
            -int(region_total[region]),
            stable_u64(seed, "competition_expansion_order", region),
        ),
    )
    while True:
        if progress_callback is not None:
            progress_callback(
                "global_assignment.competition_round",
                {
                    "status": "started",
                    "competition_round": int(expansion_round),
                    "competition_expansions": int(expansions),
                },
            )

        def attempt_progress(stage: str, details: Mapping[str, Any]) -> None:
            if progress_callback is not None:
                progress_callback(
                    stage,
                    {
                        **dict(details),
                        "competition_round": int(expansion_round),
                        "competition_expansions": int(expansions),
                    },
                )

        try:
            assignment = _assign_customers(
                customers,
                quotas,
                regions,
                seed=seed,
                progress_callback=attempt_progress,
            )
            if progress_callback is not None:
                progress_callback(
                    "global_assignment.competition_round",
                    {
                        "status": "completed",
                        "competition_round": int(expansion_round),
                        "competition_expansions": int(expansions),
                    },
                )
            return assignment, expansions
        except SpatialActivationError as error:
            if error.code != "GLOBAL_CUSTOMER_ASSIGNMENT_INFEASIBLE":
                raise
        if expansions >= maximum_expansions:
            raise SpatialActivationError(
                "GLOBAL_ASSIGNMENT_EXPANSION_EXHAUSTED",
                "global assignment remained infeasible after graph-wide expansion",
            )
        changed = False
        for region in order:
            target_deciles = set(region_quota[region])
            neighbors = frontiers[region]
            if not neighbors:
                continue

            def selection_key(candidate: str) -> tuple[int, float, int, str]:
                return (
                    0 if community_support.get(candidate, set()) & target_deciles else 1,
                    crossing_minimums[region].get(candidate, math.inf),
                    stable_u64(seed, "region_growth", region, candidate),
                    candidate,
                )

            chosen = min(neighbors, key=selection_key)
            if decision_trace is not None:
                decision_trace.append(
                    {
                        "expansion_step": int(expansions),
                        "region_id": region,
                        "chosen_community_id": chosen,
                        "selection_key": selection_key(chosen),
                    }
                )
            regions[region].add(chosen)
            neighbors.discard(chosen)
            crossing_minimums[region].pop(chosen, None)
            for candidate, weight in incident_minimums.get(chosen, {}).items():
                if candidate in regions[region]:
                    continue
                neighbors.add(candidate)
                crossing_minimums[region][candidate] = min(
                    crossing_minimums[region].get(candidate, math.inf),
                    weight,
                )
            expansions += 1
            changed = True
        if not changed:
            raise SpatialActivationError(
                "GLOBAL_ASSIGNMENT_EXPANSION_EXHAUSTED",
                "all regions cover their reachable graph but assignment is infeasible",
            )
        expansion_round += 1



def _region_first_partition(
    frame: pd.DataFrame,
    *,
    child_sizes: list[int],
    seed: int,
    namespace: str,
    community_graph: nx.DiGraph | None,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    """Partition exact children while preserving whole structure regions first."""

    children: list[list[pd.DataFrame]] = [[] for _ in child_sizes]
    remaining = np.asarray(child_sizes, dtype=int)
    split_regions: set[str] = set()
    groups = []
    for region_id, group in frame.groupby("sampling_cluster_id", sort=True):
        groups.append(
            (
                -len(group),
                stable_u64(seed, namespace, "region", str(region_id)),
                str(region_id),
                group.copy(),
            )
        )
    for _, _, region_id, group in sorted(groups):
        fitting = np.flatnonzero(remaining >= len(group))
        if len(fitting):
            child = min(
                map(int, fitting),
                key=lambda index: (-int(remaining[index]), index),
            )
            children[child].append(group)
            remaining[child] -= len(group)
            continue

        split_regions.add(region_id)
        target = np.zeros(len(child_sizes), dtype=int)
        left = len(group)
        for child in sorted(range(len(child_sizes)), key=lambda index: (-remaining[index], index)):
            take = min(left, int(remaining[child]))
            target[child] = take
            left -= take
            if not left:
                break
        if left:
            raise AssertionError("Region-first split could not satisfy remaining capacities")
        decile_groups = list(group.groupby("activation_decile", sort=True, observed=True))
        allocation = balanced_cell_partition(
            np.asarray([len(rows) for _, rows in decile_groups], dtype=int),
            target,
            seed=seed,
            namespace=f"{namespace}:split:{region_id}",
            cell_labels=[f"d{int(decile)}" for decile, _ in decile_groups],
            child_labels=[f"child_{index}" for index in range(len(child_sizes))],
        )
        for decile_index, (decile, rows) in enumerate(decile_groups):
            ordered = rows.copy()
            ordered["_region_first_rank"] = [
                stable_u64(
                    seed,
                    namespace,
                    region_id,
                    int(decile),
                    customer_id,
                )
                for customer_id in ordered["latent_service_location_id"].astype(str)
            ]
            ordered = ordered.sort_values(
                ["_region_first_rank", "latent_service_location_id"], kind="stable"
            ).drop(columns="_region_first_rank")
            cursor = 0
            for child, count in enumerate(allocation[decile_index]):
                count = int(count)
                if count:
                    children[child].append(ordered.iloc[cursor : cursor + count])
                cursor += count
        remaining -= target

    result = [
        pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0].copy()
        for parts in children
    ]
    if remaining.any() or [len(child) for child in result] != child_sizes:
        raise AssertionError("Region-first child sizes are not exact")
    parent_ids = set(frame["latent_service_location_id"].astype(str))
    child_ids = [set(child["latent_service_location_id"].astype(str)) for child in result]
    if set.union(*child_ids) != parent_ids or sum(map(len, child_ids)) != len(parent_ids):
        raise AssertionError("Region-first children are not a disjoint exact union")

    occurrences: dict[str, int] = {}
    for child in result:
        for region in set(child["sampling_cluster_id"].astype(str)):
            occurrences[region] = occurrences.get(region, 0) + 1
    split_actual = {region for region, count in occurrences.items() if count > 1}
    node_reports = []
    for child_index, child in enumerate(result):
        region_sizes = child["sampling_cluster_id"].astype(str).value_counts()
        shares = region_sizes.to_numpy(dtype=float) / len(child)
        communities = set(child["community_id"].astype(str))
        component_count = 1
        if community_graph is not None and communities:
            component_count = nx.number_weakly_connected_components(
                community_graph.subgraph(communities)
            )
        node_reports.append(
            {
                "child_index": child_index,
                "child_size": len(child),
                "region_count": int(len(region_sizes)),
                "parent_regions_touched": sorted(region_sizes.index.tolist()),
                "split_region_count": int(sum(region in split_actual for region in region_sizes.index)),
                "fragmentation_score": int(
                    sum(max(0, occurrences[region] - 1) for region in region_sizes.index)
                ),
                "largest_region_share": float(shares.max()),
                "region_hhi": float(np.square(shares).sum()),
                "road_community_component_count": int(component_count),
            }
        )
    return result, {
        "namespace": namespace,
        "child_sizes": child_sizes,
        "split_region_ids": sorted(split_actual),
        "split_region_count": len(split_actual),
        "fragmentation_score": int(sum(max(0, count - 1) for count in occurrences.values())),
        "children": node_reports,
    }


def nested_customer_order(
    assignment: pd.DataFrame,
    *,
    customer_count: int,
    seed: int,
    community_graph: nx.DiGraph | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Order parent customers so contiguous views encode the region-first tree."""

    levels: list[dict[str, Any]] = []
    if customer_count == 1000:
        cus500, report = _region_first_partition(
            assignment,
            child_sizes=[500, 500],
            seed=seed,
            namespace="tree_1000_to_500",
            community_graph=community_graph,
        )
        levels.append(report)
        leaves: list[pd.DataFrame] = []
        for group_index, group in enumerate(cus500):
            cus100, report = _region_first_partition(
                group,
                child_sizes=[100] * 5,
                seed=seed,
                namespace=f"tree_500_{group_index}_to_100",
                community_graph=community_graph,
            )
            levels.append(report)
            for node_index, node in enumerate(cus100):
                pair, report = _region_first_partition(
                    node,
                    child_sizes=[50, 50],
                    seed=seed,
                    namespace=f"tree_100_{group_index}_{node_index}_to_50",
                    community_graph=community_graph,
                )
                leaves.extend(pair)
                levels.append(report)
        ordered = pd.concat(leaves, ignore_index=True)
        shape = {"cus500_nodes": 2, "cus100_nodes": 10, "cus50_nodes": 20}
    elif customer_count == 2000:
        controls, report = _region_first_partition(
            assignment,
            child_sizes=[1000, 1000],
            seed=seed,
            namespace="scalability_2000_to_1000",
            community_graph=community_graph,
        )
        levels.append(report)
        ordered = pd.concat(controls, ignore_index=True)
        shape = {"paired_cus1000_control_nodes": 2}
    else:
        ordered = assignment.copy().reset_index(drop=True)
        shape = {"leaf_only": True}
    if ordered["latent_service_location_id"].duplicated().any() or len(ordered) != customer_count:
        raise AssertionError("Nested customer order violated parent invariants")
    return ordered, {
        "policy": "deterministic_region_first_nested_partition_v1",
        "parent_customer_count": customer_count,
        "union_exact": True,
        "pairwise_disjoint": True,
        "child_sizes_exact": True,
        "partition_levels": levels,
        **shape,
    }


def _radial_baseline(
    customers: pd.DataFrame,
    targets: list[int],
    *,
    seed: int,
) -> pd.DataFrame:
    chosen: list[pd.DataFrame] = []
    for decile, count in enumerate(targets):
        cell = customers.loc[customers["radial_decile"].eq(decile)].copy()
        if len(cell) < count:
            raise SpatialActivationError(
                "RADIAL_BASELINE_UNSUPPORTED",
                f"decile {decile} contains {len(cell)} locations for target {count}",
            )
        cell["_baseline_key"] = [
            _weighted_key(seed, f"baseline_d{decile}", customer_id, units)
            for customer_id, units in zip(
                cell["latent_service_location_id"].astype(str),
                cell["residential_units"].astype(float),
            )
        ]
        chosen.append(
            cell.sort_values(
                ["_baseline_key", "latent_service_location_id"], kind="stable"
            ).head(count)
        )
    return pd.concat(chosen, ignore_index=True).drop(columns="_baseline_key")


def activate_spatial_customers(
    customers: pd.DataFrame,
    community_adjacency: pd.DataFrame,
    structure_targets: pd.DataFrame,
    *,
    customer_count: int,
    seed: int,
    region_redraw_cap: int = 3,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> SpatialActivationResult:
    """Activate one exact parent set using the frozen Step-6 proposal."""

    performance_profile: list[dict[str, Any]] = []

    def begin(stage: str, **details: Any) -> tuple[float, float]:
        if progress_callback is not None:
            progress_callback(stage, {"status": "started", **details})
        return time.perf_counter(), time.process_time()

    def finish(
        stage: str,
        started: tuple[float, float],
        *,
        status: str = "completed",
        **details: Any,
    ) -> None:
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        event = {
            "stage": stage,
            "status": status,
            "wall_seconds": time.perf_counter() - started[0],
            "cpu_seconds": time.process_time() - started[1],
            "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
            **details,
        }
        performance_profile.append(event)
        if progress_callback is not None:
            progress_callback(stage, event)

    required = {
        "latent_service_location_id",
        "community_id",
        "residential_units",
        "radial_decile",
        "depot_running_time_s",
    }
    missing = required - set(customers.columns)
    if missing:
        raise ValueError(f"Spatial customer pool lacks columns: {sorted(missing)}")
    if len(customers) < customer_count:
        raise SpatialActivationError(
            "TERRITORY_TOO_SMALL",
            f"territory has {len(customers)} eligible customers for N={customer_count}",
        )
    if customers["latent_service_location_id"].duplicated().any():
        raise ValueError("Spatial customer pool contains duplicate IDs")
    stage_started = begin("quota_matrix", territory_count=len(customers))
    quotas, quota_metadata, target_times = _quota_matrix(
        structure_targets,
        customer_count=customer_count,
        seed=seed,
    )
    finish(
        "quota_matrix",
        stage_started,
        quota_cell_count=len(quotas),
        region_count=int(quotas["region_id"].nunique()),
    )
    target_columns = quota_metadata["radial_decile_targets"]
    available_columns = (
        customers.groupby("radial_decile", observed=True).size().reindex(range(10), fill_value=0)
    )
    unsupported = {
        decile: {"available": int(available_columns.iloc[decile]), "target": int(target)}
        for decile, target in enumerate(target_columns)
        if int(available_columns.iloc[decile]) < int(target)
    }
    if unsupported:
        raise SpatialActivationError(
            "SPATIAL_QUOTA_UNSUPPORTED",
            "territory lacks customer capacity in one or more radial deciles",
            {"decile_deficits": unsupported},
        )
    community_ids = set(customers["community_id"].astype(str))
    if not community_adjacency.empty:
        community_ids |= set(community_adjacency["source_community_id"].astype(str))
        community_ids |= set(community_adjacency["target_community_id"].astype(str))
    stage_started = begin(
        "community_graph",
        adjacency_row_count=len(community_adjacency),
        community_id_count=len(community_ids),
    )
    graph = _community_graph(community_adjacency, community_ids)
    finish(
        "community_graph",
        stage_started,
        graph_node_count=graph.number_of_nodes(),
        graph_edge_count=graph.number_of_edges(),
    )
    if graph.number_of_edges() == 0 and len(community_ids) > 1:
        raise SpatialActivationError(
            "COMMUNITY_ADJACENCY_EMPTY",
            "multi-community territory has no road-derived adjacency edges",
        )
    last_error: SpatialActivationError | None = None
    assignment: pd.DataFrame | None = None
    seeds: dict[str, str] = {}
    fallback_events: list[dict[str, Any]] = []
    growth_steps = 0
    competition_expansions = 0
    used_attempt = -1
    for attempt in range(region_redraw_cap + 1):
        attempt_seed = int(stable_u64(seed, "region_redraw", attempt) % (2**63 - 1))
        try:
            stage_started = begin(
                "region_seed_selection",
                redraw_attempt=attempt,
                region_count=int(quotas["region_id"].nunique()),
            )
            seeds, fallback_events = _choose_region_seeds(
                customers,
                quotas,
                graph,
                seed=attempt_seed,
                progress_callback=progress_callback,
            )
            finish(
                "region_seed_selection",
                stage_started,
                redraw_attempt=attempt,
                selected_seed_count=len(seeds),
            )
            stage_started = begin(
                "region_growth",
                redraw_attempt=attempt,
                graph_node_count=graph.number_of_nodes(),
            )
            regions, growth_steps = _grow_regions(
                customers,
                quotas,
                graph,
                seeds,
                seed=attempt_seed,
                progress_callback=progress_callback,
            )
            finish(
                "region_growth",
                stage_started,
                redraw_attempt=attempt,
                growth_steps=growth_steps,
            )
            stage_started = begin(
                "global_customer_assignment",
                redraw_attempt=attempt,
                region_count=len(regions),
            )
            assignment, competition_expansions = _assign_with_competition_expansion(
                customers,
                quotas,
                regions,
                graph,
                seed=attempt_seed,
                progress_callback=progress_callback,
            )
            finish(
                "global_customer_assignment",
                stage_started,
                redraw_attempt=attempt,
                assignment_count=len(assignment),
                competition_expansions=competition_expansions,
            )
            used_attempt = attempt
            break
        except SpatialActivationError as error:
            finish(
                "region_redraw_attempt",
                stage_started,
                status="failed",
                redraw_attempt=attempt,
                error_code=error.code,
            )
            last_error = error
    if assignment is None:
        raise SpatialActivationError(
            "REGION_REDRAW_EXHAUSTED",
            f"all {region_redraw_cap + 1} region attempts failed; last={last_error}",
            {"last_error_code": last_error.code if last_error else None},
        )
    stage_started = begin("nested_customer_order", assignment_count=len(assignment))
    ordered_assignment, tree_metadata = nested_customer_order(
        assignment,
        customer_count=customer_count,
        seed=seed,
        community_graph=graph,
    )
    finish("nested_customer_order", stage_started, ordered_count=len(ordered_assignment))
    stage_started = begin("selected_customer_join")
    customer_lookup = customers.set_index("latent_service_location_id", drop=False)
    selected = customer_lookup.loc[
        ordered_assignment["latent_service_location_id"].astype(str)
    ].reset_index(drop=True)
    selected = selected.merge(
        ordered_assignment[
            [
                "latent_service_location_id",
                "sampling_cluster_id",
                "structure_route_id",
                "activation_decile",
            ]
        ],
        on="latent_service_location_id",
        how="left",
        validate="one_to_one",
    )
    finish("selected_customer_join", stage_started, selected_count=len(selected))
    stage_started = begin("radial_baseline")
    radial_baseline = _radial_baseline(
        customers,
        target_columns,
        seed=int(stable_u64(seed, "radial_baseline") % (2**63 - 1)),
    )
    finish("radial_baseline", stage_started, baseline_count=len(radial_baseline))
    region_sizes = (
        ordered_assignment.groupby("sampling_cluster_id", sort=True)
        .size()
        .astype(int)
        .to_dict()
    )
    metadata = {
        "schema": "evrptw_spatial_activation_report_v3",
        "policy": "amazon_stationday_multi_region_community_graph_v3",
        "customer_count": customer_count,
        "eligible_territory_count": len(customers),
        "quota": quota_metadata,
        "target_radial_time_s": target_times.tolist(),
        "region_count": len(region_sizes),
        "region_sizes": {str(key): int(value) for key, value in region_sizes.items()},
        "region_seed_communities": seeds,
        "seed_fallback_events": fallback_events,
        "seed_fallback_count": len(fallback_events),
        "region_redraw_cap": int(region_redraw_cap),
        "region_attempts_used": int(used_attempt + 1),
        "region_redraw_count": int(used_attempt),
        "community_growth_steps": int(growth_steps),
        "assignment_competition_expansions": int(competition_expansions),
        "reservation_unit": "latent_customer_id",
        "transit_communities_shareable": True,
        "communities_shareable_across_regions": True,
        "global_customer_uniqueness": True,
        "view_tree": tree_metadata,
        "radial_baseline_count": len(radial_baseline),
        "performance_profile": performance_profile,
    }
    return SpatialActivationResult(
        customers=selected,
        assignment=ordered_assignment,
        radial_baseline=radial_baseline,
        metadata=metadata,
    )
