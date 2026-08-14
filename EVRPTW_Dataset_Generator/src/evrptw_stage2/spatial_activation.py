"""Amazon-structured multi-region customer activation for Stage 2."""

from __future__ import annotations

import math
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
    for region in order:
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


def _grow_regions(
    customers: pd.DataFrame,
    quotas: pd.DataFrame,
    graph: nx.DiGraph,
    seeds: dict[str, str],
    *,
    seed: int,
) -> tuple[dict[str, set[str]], int]:
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
    while growth_steps < maximum_steps:
        changed = False
        all_satisfied = True
        for region in order:
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
            regions[region].add(chosen)
            growth_steps += 1
            changed = True
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
) -> pd.DataFrame:
    graph = nx.DiGraph()
    total = int(quotas["quota"].sum())
    graph.add_node("source", demand=-total)
    graph.add_node("sink", demand=total)
    customer_lookup = customers.set_index("latent_service_location_id", drop=False)
    for row in quotas.itertuples(index=False):
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
    try:
        flow = nx.min_cost_flow(graph)
    except nx.NetworkXUnfeasible as error:
        raise SpatialActivationError(
            "GLOBAL_CUSTOMER_ASSIGNMENT_INFEASIBLE",
            "overlapping regions cannot satisfy all quotas with globally unique locations",
        ) from error
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
    return assignment


def _assign_with_competition_expansion(
    customers: pd.DataFrame,
    quotas: pd.DataFrame,
    regions: dict[str, set[str]],
    graph: nx.DiGraph,
    *,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    """Expand regions when overlapping reservations block global assignment."""

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
            return _assign_customers(customers, quotas, regions, seed=seed), expansions
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
            regions[region].add(chosen)
            expansions += 1
            changed = True
        if not changed:
            raise SpatialActivationError(
                "GLOBAL_ASSIGNMENT_EXPANSION_EXHAUSTED",
                "all regions cover their reachable graph but assignment is infeasible",
            )


def _partition_ids(
    frame: pd.DataFrame,
    *,
    child_sizes: list[int],
    seed: int,
    namespace: str,
) -> list[pd.DataFrame]:
    cell_columns = ["sampling_cluster_id", "activation_decile"]
    grouped = list(frame.groupby(cell_columns, sort=True, observed=True))
    cell_labels = [f"{key[0]}:d{int(key[1])}" for key, _ in grouped]
    allocation = balanced_cell_partition(
        np.asarray([len(group) for _, group in grouped], dtype=int),
        np.asarray(child_sizes, dtype=int),
        seed=seed,
        namespace=namespace,
        cell_labels=cell_labels,
        child_labels=[f"child_{index}" for index in range(len(child_sizes))],
    )
    children: list[list[pd.DataFrame]] = [[] for _ in child_sizes]
    for cell_index, ((_, group), label) in enumerate(zip(grouped, cell_labels)):
        ordered = group.copy()
        ordered["_partition_rank"] = [
            stable_u64(seed, namespace, label, customer_id)
            for customer_id in ordered["latent_service_location_id"].astype(str)
        ]
        ordered = ordered.sort_values(
            ["_partition_rank", "latent_service_location_id"], kind="stable"
        ).drop(columns="_partition_rank")
        cursor = 0
        for child_index, count in enumerate(allocation[cell_index]):
            count = int(count)
            children[child_index].append(ordered.iloc[cursor : cursor + count])
            cursor += count
        if cursor != len(ordered):
            raise AssertionError("Cell partition failed to consume every customer")
    result = [
        pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0].copy()
        for parts in children
    ]
    ids = [set(child["latent_service_location_id"].astype(str)) for child in result]
    if set.union(*ids) != set(frame["latent_service_location_id"].astype(str)):
        raise AssertionError("Partition child union differs from parent")
    if sum(map(len, ids)) != len(set.union(*ids)):
        raise AssertionError("Partition children are not disjoint")
    if [len(child) for child in result] != child_sizes:
        raise AssertionError("Partition child size mismatch")
    return result


def nested_customer_order(
    assignment: pd.DataFrame,
    *,
    customer_count: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Order parent customers so contiguous views encode the frozen tree."""

    if customer_count == 1000:
        cus500 = _partition_ids(
            assignment,
            child_sizes=[500, 500],
            seed=seed,
            namespace="tree_1000_to_500",
        )
        leaves: list[pd.DataFrame] = []
        for group_index, group in enumerate(cus500):
            cus100 = _partition_ids(
                group,
                child_sizes=[100] * 5,
                seed=seed,
                namespace=f"tree_500_{group_index}_to_100",
            )
            for node_index, node in enumerate(cus100):
                leaves.extend(
                    _partition_ids(
                        node,
                        child_sizes=[50, 50],
                        seed=seed,
                        namespace=f"tree_100_{group_index}_{node_index}_to_50",
                    )
                )
        ordered = pd.concat(leaves, ignore_index=True)
        shape = {"cus500_nodes": 2, "cus100_nodes": 10, "cus50_nodes": 20}
    elif customer_count == 2000:
        controls = _partition_ids(
            assignment,
            child_sizes=[1000, 1000],
            seed=seed,
            namespace="scalability_2000_to_1000",
        )
        ordered = pd.concat(controls, ignore_index=True)
        shape = {"paired_cus1000_control_nodes": 2}
    else:
        ordered = assignment.copy().reset_index(drop=True)
        shape = {"leaf_only": True}
    if ordered["latent_service_location_id"].duplicated().any() or len(ordered) != customer_count:
        raise AssertionError("Nested customer order violated parent invariants")
    return ordered, {
        "policy": "region_decile_controlled_rounding_tree_v1",
        "parent_customer_count": customer_count,
        "union_exact": True,
        "pairwise_disjoint": True,
        "child_sizes_exact": True,
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
) -> SpatialActivationResult:
    """Activate one exact parent set using the frozen Step-6 proposal."""

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
    quotas, quota_metadata, target_times = _quota_matrix(
        structure_targets,
        customer_count=customer_count,
        seed=seed,
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
    graph = _community_graph(community_adjacency, community_ids)
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
            seeds, fallback_events = _choose_region_seeds(
                customers, quotas, graph, seed=attempt_seed
            )
            regions, growth_steps = _grow_regions(
                customers, quotas, graph, seeds, seed=attempt_seed
            )
            assignment, competition_expansions = _assign_with_competition_expansion(
                customers,
                quotas,
                regions,
                graph,
                seed=attempt_seed,
            )
            used_attempt = attempt
            break
        except SpatialActivationError as error:
            last_error = error
    if assignment is None:
        raise SpatialActivationError(
            "REGION_REDRAW_EXHAUSTED",
            f"all {region_redraw_cap + 1} region attempts failed; last={last_error}",
            {"last_error_code": last_error.code if last_error else None},
        )
    ordered_assignment, tree_metadata = nested_customer_order(
        assignment,
        customer_count=customer_count,
        seed=seed,
    )
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
    radial_baseline = _radial_baseline(
        customers,
        target_columns,
        seed=int(stable_u64(seed, "radial_baseline") % (2**63 - 1)),
    )
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
    }
    return SpatialActivationResult(
        customers=selected,
        assignment=ordered_assignment,
        radial_baseline=radial_baseline,
        metadata=metadata,
    )
