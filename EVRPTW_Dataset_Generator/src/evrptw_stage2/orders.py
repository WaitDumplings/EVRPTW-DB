"""View-level packages, volume, service time, time windows, and feasibility."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, maximum_bipartite_matching

from .rounding import stable_u64

FULL_CS_TO_DEPOT_CACHE_CONTRACT = {
    "semantics": "full_departure_cs_to_depot_fastest_feasible_time_v2",
    "time_matrix": "running_time_shortest_matrix_s",
    "energy_source": (
        "running_time_path_distance_km_times_specific_energy_consumption_kwh_per_km"
    ),
    "intermediate_full_charge_policy": "full_charge_linear_v1",
    "allows_multiple_cs_hops": True,
    "origin_cs_charge_time_included": False,
    "depot_charge_time_included": False,
}


@dataclass(frozen=True)
class ViewAttributes:
    package_counts: np.ndarray
    demands_cm3: np.ndarray
    service_time_s: np.ndarray
    time_windows_s: np.ndarray
    feasible_arrival_time_s: np.ndarray
    feasible_return_duration_s: np.ndarray
    feasibility_requires_charging: np.ndarray
    feasibility_charging_visit_count: np.ndarray
    feasibility_inbound_full_state_terminal_index: np.ndarray
    feasibility_first_post_customer_charger_terminal_index: np.ndarray
    feasibility_energy_margin_kwh: np.ndarray
    full_cs_to_depot_time_s: np.ndarray
    order_sampling_attempts: np.ndarray
    report: dict[str, Any]


@dataclass(frozen=True)
class SingleCustomerCertificates:
    arrival_elapsed_s: np.ndarray
    return_duration_s: np.ndarray
    requires_charging: np.ndarray
    charging_visit_count: np.ndarray
    inbound_full_state_terminal_index: np.ndarray
    first_post_customer_charger_terminal_index: np.ndarray
    customer_transition_energy_margin_kwh: np.ndarray
    full_cs_to_depot_time_s: np.ndarray


@dataclass(frozen=True)
class FullStateRouteCache:
    """Shortest-time caches between full-battery infrastructure states.

    Position zero is the depot and the remaining positions are charging
    stations.  ``to_depot_s[q]`` is the shortest duration from station ``q``
    while full to the depot, including any required intermediate CS visits and
    their full-charge durations.
    """

    terminal_indices: np.ndarray
    transition_cost_s: np.ndarray
    from_depot_s: np.ndarray
    to_depot_s: np.ndarray
    from_depot_predecessor: np.ndarray
    to_depot_reverse_predecessor: np.ndarray


def _certificates_for_view(
    *,
    customer_count: int,
    running_time_matrix_s: np.ndarray,
    running_time_path_distance_matrix_km: np.ndarray,
    charging_power_kw: np.ndarray,
    profile: Mapping[str, Any],
) -> SingleCustomerCertificates:
    return _single_customer_certificates(
        customer_count=customer_count,
        running_time_matrix_s=running_time_matrix_s,
        running_time_path_distance_matrix_km=running_time_path_distance_matrix_km,
        specific_energy_consumption_kwh_per_km=float(
            profile["energy"]["specific_energy_consumption_kwh_per_km"]
        ),
        charging_power_kw=np.asarray(charging_power_kw, dtype=np.float32),
        battery_capacity_kwh=float(profile["energy"]["battery_capacity_kwh"]),
        charging_power_derating_factor=float(
            profile["charging"]["charging_power_derating_factor"]
        ),
    )


def match_amazon_order_templates(
    *,
    customer_count: int,
    order_sources: list[tuple[dict[str, Any], pd.DataFrame]],
    matching_seed: int,
    operating_start_s: int,
    operating_end_s: int,
    running_time_matrix_s: np.ndarray,
    running_time_path_distance_matrix_km: np.ndarray,
    charging_power_kw: np.ndarray,
    profile: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Find a covering customer-template matching from an admissible source.

    Every customer receives one distinct observed Amazon stop template.  A
    template edge exists only when the single-customer certificate can serve
    its time window and return within the operating horizon.
    """

    certificates = _certificates_for_view(
        customer_count=customer_count,
        running_time_matrix_s=running_time_matrix_s,
        running_time_path_distance_matrix_km=running_time_path_distance_matrix_km,
        charging_power_kw=charging_power_kw,
        profile=profile,
    )
    arrival = float(operating_start_s) + certificates.arrival_elapsed_s.astype(float)
    if not np.isfinite(arrival).all() or not np.isfinite(
        certificates.return_duration_s
    ).all():
        raise ValueError("Customer parent fails structural energy reachability before orders")
    cargo_capacity = float(profile["vehicle"]["cargo_capacity_cm3"])
    source_attempts: list[dict[str, Any]] = []
    for source_rank, (source, raw_templates) in enumerate(order_sources):
        templates = raw_templates.copy()
        templates["_rank"] = [
            stable_u64(matching_seed, "order_template", template_id)
            for template_id in templates["template_id"].astype(str)
        ]
        templates = templates.sort_values(["_rank", "template_id"], kind="stable").drop(
            columns="_rank"
        )
        template_count = len(templates)
        if template_count < customer_count:
            source_attempts.append(
                {
                    "source_rank": source_rank,
                    "source_mode": source["order_source_mode"],
                    "template_count": template_count,
                    "matched_customer_count": 0,
                    "status": "insufficient_template_count",
                }
            )
            continue
        tw_start = templates["tw_start_s"].to_numpy(dtype=float)
        tw_end = templates["tw_end_s"].to_numpy(dtype=float)
        service = templates["service_time_s"].to_numpy(dtype=float)
        demand = templates["demand_cm3"].to_numpy(dtype=float)
        row_indices: list[np.ndarray] = []
        column_indices: list[np.ndarray] = []
        for customer_start in range(0, customer_count, 128):
            customer_stop = min(customer_count, customer_start + 128)
            block_arrival = arrival[customer_start:customer_stop, None]
            service_start = np.maximum(block_arrival, tw_start[None, :])
            feasible = (
                (service_start <= tw_end[None, :] + 1e-6)
                & (
                    service_start
                    + service[None, :]
                    + certificates.return_duration_s[customer_start:customer_stop, None]
                    <= float(operating_end_s) + 1e-6
                )
                & (demand[None, :] <= cargo_capacity + 1e-6)
            )
            block_rows, block_columns = np.nonzero(feasible)
            row_indices.append(block_rows + customer_start)
            column_indices.append(block_columns)
        rows = np.concatenate(row_indices) if row_indices else np.zeros(0, dtype=int)
        columns = (
            np.concatenate(column_indices) if column_indices else np.zeros(0, dtype=int)
        )
        graph = csr_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, columns)),
            shape=(customer_count, template_count),
        )
        matching = maximum_bipartite_matching(graph, perm_type="column")
        matched_count = int(np.sum(matching >= 0))
        source_attempts.append(
            {
                "source_rank": source_rank,
                "source_mode": source["order_source_mode"],
                "station_code": source["station_code"],
                "station_day_ids": list(source["station_day_ids"]),
                "template_count": template_count,
                "feasible_edge_count": int(graph.nnz),
                "matched_customer_count": matched_count,
                "status": "accepted" if matched_count == customer_count else "hall_failure",
            }
        )
        if matched_count != customer_count:
            continue
        matched = templates.iloc[matching].reset_index(drop=True)
        matched.insert(0, "parent_customer_position", np.arange(customer_count, dtype=int))
        def bias_summary(frame: pd.DataFrame) -> dict[str, Any]:
            width = frame["tw_end_s"].to_numpy(dtype=float) - frame[
                "tw_start_s"
            ].to_numpy(dtype=float)
            return {
                "count": len(frame),
                "tw_presence_rate": float(frame["tw_was_specified"].astype(bool).mean()),
                "tw_width_s_mean": float(width.mean()),
                "tw_width_s_p50": float(np.quantile(width, 0.50)),
                "package_count_mean": float(frame["package_count"].mean()),
                "package_count_p90": float(frame["package_count"].quantile(0.90)),
                "demand_cm3_mean": float(frame["demand_cm3"].mean()),
                "demand_cm3_p90": float(frame["demand_cm3"].quantile(0.90)),
                "service_time_s_mean": float(frame["service_time_s"].mean()),
                "service_time_s_p90": float(frame["service_time_s"].quantile(0.90)),
            }

        return matched, {
            "policy": "amazon_stationday_covering_bipartite_matching_v1",
            "selected_order_source_mode": source["order_source_mode"],
            "selected_station_code": source["station_code"],
            "selected_station_day_ids": list(source["station_day_ids"]),
            "generation_track": source.get("generation_track"),
            "source_attempt_count": source_rank + 1,
            "source_attempts": source_attempts,
            "customer_count": customer_count,
            "template_reuse": False,
            "hall_coverage_complete": True,
            "matching_bias_audit": {
                "eligible_pool": bias_summary(templates),
                "matched_templates": bias_summary(matched),
            },
        }
    raise ValueError(
        "ORDER_SOURCE_EXHAUSTED: no admissible single-day or same-station composite "
        f"covers all customers; attempts={source_attempts}"
    )


def _build_full_state_route_cache(
    *,
    customer_count: int,
    running_time_matrix_s: np.ndarray,
    running_time_energy_matrix_kwh: np.ndarray,
    charging_power_kw: np.ndarray,
    battery_capacity_kwh: float,
    charging_power_derating_factor: float,
) -> FullStateRouteCache:
    """Cache directed depot/CS travel under full-charge semantics.

    An arc ending at a CS restores the energy consumed since its full-state
    origin.  Reverse Dijkstra from the depot therefore gives every full CS's
    fastest return, with multi-hop charging when a direct return is impossible.
    A separate forward cache is required because the road matrices are
    directed and a customer need not be reachable directly from the depot.
    """
    terminal_count = running_time_matrix_s.shape[0]
    charger_nodes = np.arange(1 + customer_count, terminal_count, dtype=np.int32)
    full_nodes = np.concatenate([np.asarray([0], dtype=np.int32), charger_nodes])
    full_time = running_time_matrix_s[np.ix_(full_nodes, full_nodes)].astype(float)
    full_energy = running_time_energy_matrix_kwh[
        np.ix_(full_nodes, full_nodes)
    ].astype(float)

    transition_cost = np.full_like(full_time, np.inf, dtype=float)
    np.fill_diagonal(transition_cost, 0.0)
    for destination in range(1, len(full_nodes)):
        allowed = full_energy[:, destination] <= battery_capacity_kwh + 1e-9
        recharge_s = (
            full_energy[:, destination]
            / (
                float(charging_power_kw[destination - 1])
                * charging_power_derating_factor
            )
            * 3600.0
        )
        transition_cost[allowed, destination] = (
            full_time[allowed, destination] + recharge_s[allowed]
        )
    depot_allowed = full_energy[:, 0] <= battery_capacity_kwh + 1e-9
    transition_cost[depot_allowed, 0] = full_time[depot_allowed, 0]
    transition_cost[0, 0] = 0.0

    from_depot, from_predecessor = dijkstra(
        transition_cost,
        directed=True,
        indices=0,
        return_predecessors=True,
    )
    to_depot, to_reverse_predecessor = dijkstra(
        transition_cost.T,
        directed=True,
        indices=0,
        return_predecessors=True,
    )
    return FullStateRouteCache(
        terminal_indices=full_nodes,
        transition_cost_s=transition_cost,
        from_depot_s=np.asarray(from_depot, dtype=float),
        to_depot_s=np.asarray(to_depot, dtype=float),
        from_depot_predecessor=np.asarray(from_predecessor),
        to_depot_reverse_predecessor=np.asarray(to_reverse_predecessor),
    )







def _single_customer_certificates(
    *,
    customer_count: int,
    running_time_matrix_s: np.ndarray,
    running_time_path_distance_matrix_km: np.ndarray,
    specific_energy_consumption_kwh_per_km: float,
    charging_power_kw: np.ndarray,
    battery_capacity_kwh: float,
    charging_power_derating_factor: float,
) -> SingleCustomerCertificates:
    """Construct one-customer routes with optional full-charge CS visits.

    All paths are evaluated on the already materialized directed running-time
    terminal closure. A depot or charging station is a full-battery state. The
    customer does not reset energy, so the incoming and outgoing customer legs
    must jointly fit between two consecutive full-battery states.
    """
    terminal_count = running_time_matrix_s.shape[0]
    if running_time_path_distance_matrix_km.shape != running_time_matrix_s.shape:
        raise ValueError("Running-time path-distance matrix shape mismatch")
    if specific_energy_consumption_kwh_per_km <= 0.0:
        raise ValueError("Specific energy consumption must be positive")
    running_time_energy_matrix_kwh = (
        running_time_path_distance_matrix_km.astype(float)
        * float(specific_energy_consumption_kwh_per_km)
    )
    charger_count = terminal_count - customer_count - 1
    if charger_count <= 0:
        raise ValueError("Feasibility certificate requires at least one charging station")
    if charging_power_kw.shape != (charger_count,):
        raise ValueError(
            "charging_power_kw shape mismatch: "
            f"got {charging_power_kw.shape}, expected {(charger_count,)}"
        )
    if np.any(charging_power_kw <= 0.0):
        raise ValueError("All effective charging powers must be positive")
    if not 0.0 < charging_power_derating_factor <= 1.0:
        raise ValueError("charging_power_derating_factor must be in (0, 1]")

    customer_nodes = np.arange(1, 1 + customer_count, dtype=np.int32)
    charger_nodes = np.arange(1 + customer_count, terminal_count, dtype=np.int32)
    cache = _build_full_state_route_cache(
        customer_count=customer_count,
        running_time_matrix_s=running_time_matrix_s,
        running_time_energy_matrix_kwh=running_time_energy_matrix_kwh,
        charging_power_kw=charging_power_kw,
        battery_capacity_kwh=battery_capacity_kwh,
        charging_power_derating_factor=charging_power_derating_factor,
    )
    full_nodes = cache.terminal_indices

    def predecessor_charger_count(predecessors: np.ndarray, node: int) -> int:
        count = 0
        current = int(node)
        while current != 0:
            if current < 0:
                raise ValueError("Broken predecessor chain in feasibility certificate")
            count += 1
            current = int(predecessors[current])
        return count

    arrival_elapsed = np.full(customer_count, np.inf, dtype=float)
    return_duration = np.full(customer_count, np.inf, dtype=float)
    requires_charging = np.zeros(customer_count, dtype=bool)
    charging_visit_count = np.zeros(customer_count, dtype=np.int16)
    inbound_full_state_terminal_index = np.full(customer_count, -1, dtype=np.int32)
    first_post_customer_charger_terminal_index = np.full(
        customer_count, -1, dtype=np.int32
    )
    transition_margin = np.full(customer_count, -np.inf, dtype=float)

    for position, customer_node in enumerate(customer_nodes):
        best_total = np.inf
        best_arrival = np.inf
        best_return = np.inf
        best_requires_charging = False
        best_charging_visit_count = 0
        best_inbound_terminal = -1
        best_post_terminal = -1
        best_margin = -np.inf
        for source_position, source_node in enumerate(full_nodes):
            inbound_energy = float(
                running_time_energy_matrix_kwh[source_node, customer_node]
            )
            if (
                not np.isfinite(cache.from_depot_s[source_position])
                or inbound_energy > battery_capacity_kwh + 1e-9
            ):
                continue
            arrival = float(cache.from_depot_s[source_position]) + float(
                running_time_matrix_s[source_node, customer_node]
            )

            direct_energy = inbound_energy + float(
                running_time_energy_matrix_kwh[customer_node, 0]
            )
            candidate_return = np.inf
            candidate_margin = -np.inf
            candidate_post_charge = False
            candidate_post_position = -1
            if direct_energy <= battery_capacity_kwh + 1e-9:
                candidate_return = float(running_time_matrix_s[customer_node, 0])
                candidate_margin = battery_capacity_kwh - direct_energy

            outbound_energy = running_time_energy_matrix_kwh[
                customer_node, charger_nodes
            ].astype(float)
            combined_energy = inbound_energy + outbound_energy
            allowed = (
                (combined_energy <= battery_capacity_kwh + 1e-9)
                & np.isfinite(cache.to_depot_s[1:])
            )
            if allowed.any():
                recharge_s = (
                    combined_energy[allowed]
                    / (
                        charging_power_kw[allowed].astype(float)
                        * charging_power_derating_factor
                    )
                    * 3600.0
                )
                via_cs = (
                    running_time_matrix_s[customer_node, charger_nodes[allowed]].astype(float)
                    + recharge_s
                    + cache.to_depot_s[1:][allowed]
                )
                local = int(np.argmin(via_cs))
                if float(via_cs[local]) < candidate_return:
                    allowed_positions = np.flatnonzero(allowed)
                    candidate_post_position = int(allowed_positions[local]) + 1
                    candidate_return = float(via_cs[local])
                    candidate_margin = battery_capacity_kwh - float(combined_energy[allowed][local])
                    candidate_post_charge = True

            total = arrival + candidate_return
            if total < best_total:
                best_total = total
                best_arrival = arrival
                best_return = candidate_return
                best_requires_charging = source_position != 0 or candidate_post_charge
                pre_visit_count = predecessor_charger_count(
                    cache.from_depot_predecessor, source_position
                )
                post_visit_count = 0
                if candidate_post_charge:
                    post_visit_count = 1 + max(
                        0,
                        predecessor_charger_count(
                            cache.to_depot_reverse_predecessor,
                            candidate_post_position,
                        )
                        - 1,
                    )
                best_charging_visit_count = pre_visit_count + post_visit_count
                best_inbound_terminal = int(source_node)
                best_post_terminal = (
                    int(full_nodes[candidate_post_position])
                    if candidate_post_charge
                    else -1
                )
                best_margin = candidate_margin

        if not np.isfinite(best_total):
            continue
        arrival_elapsed[position] = best_arrival
        return_duration[position] = best_return
        requires_charging[position] = best_requires_charging
        charging_visit_count[position] = best_charging_visit_count
        inbound_full_state_terminal_index[position] = best_inbound_terminal
        first_post_customer_charger_terminal_index[position] = best_post_terminal
        transition_margin[position] = best_margin

    return SingleCustomerCertificates(
        arrival_elapsed_s=arrival_elapsed.astype(np.float32),
        return_duration_s=return_duration.astype(np.float32),
        requires_charging=requires_charging,
        charging_visit_count=charging_visit_count,
        inbound_full_state_terminal_index=inbound_full_state_terminal_index,
        first_post_customer_charger_terminal_index=(
            first_post_customer_charger_terminal_index
        ),
        customer_transition_energy_margin_kwh=transition_margin.astype(np.float32),
        full_cs_to_depot_time_s=cache.to_depot_s[1:].astype(np.float32),
    )



def build_view_attributes_from_amazon(
    customer_rows: pd.DataFrame,
    order_templates: pd.DataFrame,
    *,
    day_type: str,
    operating_start_s: int,
    operating_end_s: int,
    running_time_matrix_s: np.ndarray,
    running_time_path_distance_matrix_km: np.ndarray,
    charging_power_kw: np.ndarray,
    profile: Mapping[str, Any],
    order_source_report: Mapping[str, Any],
) -> ViewAttributes:
    """Attach already matched Amazon templates and recompute view certificates."""

    customer_count = len(customer_rows)
    if len(order_templates) != customer_count:
        raise ValueError("Amazon template count differs from view customer count")
    templates = order_templates.reset_index(drop=True)
    certificates = _certificates_for_view(
        customer_count=customer_count,
        running_time_matrix_s=running_time_matrix_s,
        running_time_path_distance_matrix_km=running_time_path_distance_matrix_km,
        charging_power_kw=charging_power_kw,
        profile=profile,
    )
    package_counts = templates["package_count"].to_numpy(dtype=np.int32)
    demands = templates["demand_cm3"].to_numpy(dtype=np.float32)
    service = templates["service_time_s"].to_numpy(dtype=np.float32)
    windows = templates[["tw_start_s", "tw_end_s"]].to_numpy(dtype=np.float32)
    feasible_arrival = float(operating_start_s) + certificates.arrival_elapsed_s
    service_start = np.maximum(feasible_arrival, windows[:, 0])
    time_feasible = (
        (service_start <= windows[:, 1] + 1e-6)
        & (
            service_start + service + certificates.return_duration_s
            <= float(operating_end_s) + 1e-6
        )
    )
    energy_feasible = np.isfinite(certificates.arrival_elapsed_s) & np.isfinite(
        certificates.return_duration_s
    )
    capacity_feasible = demands <= float(profile["vehicle"]["cargo_capacity_cm3"]) + 1e-6
    if not (time_feasible & energy_feasible & capacity_feasible).all():
        raise ValueError(
            "Inherited Amazon template fails a descendant-view certificate: "
            f"time={int((~time_feasible).sum())}, "
            f"energy={int((~energy_feasible).sum())}, "
            f"capacity={int((~capacity_feasible).sum())}"
        )
    cached_returns = certificates.full_cs_to_depot_time_s.astype(float)
    finite_cached_returns = cached_returns[np.isfinite(cached_returns)]
    tight = (windows[:, 0] > operating_start_s) | (windows[:, 1] < operating_end_s)
    certificate_cs = set(
        certificates.inbound_full_state_terminal_index[
            certificates.inbound_full_state_terminal_index > customer_count
        ].astype(int)
    ) | set(
        certificates.first_post_customer_charger_terminal_index[
            certificates.first_post_customer_charger_terminal_index > customer_count
        ].astype(int)
    )
    soc_tolerance_kwh = 1e-6
    report = {
        "schema": "cle_evrptw_view_attribute_report_v3",
        "day_type": day_type,
        "customer_count": customer_count,
        "order_attribute_source": "amazon_last_mile_2021_observed_stop_templates",
        "order_template_inheritance": True,
        "package_count_mean": float(package_counts.mean()),
        "package_count_p90": float(np.quantile(package_counts, 0.90)),
        "demand_cm3_mean": float(demands.mean()),
        "demand_cm3_p90": float(np.quantile(demands, 0.90)),
        "service_time_s_mean": float(service.mean()),
        "service_time_s_p90": float(np.quantile(service, 0.90)),
        "time_windows": {
            "tight_window_count": int(tight.sum()),
            "actual_tight_window_rate": float(tight.mean()),
            "feasibility_clipping_applied": False,
        },
        "order_matching": dict(order_source_report),
        "energy": {
            "model_id": str(profile["energy"]["model_id"]),
            "specific_energy_consumption_kwh_per_km": float(
                profile["energy"]["specific_energy_consumption_kwh_per_km"]
            ),
            "stored_energy_matrices": False,
        },
        "full_cs_to_depot_cache": {
            "semantics": FULL_CS_TO_DEPOT_CACHE_CONTRACT["semantics"],
            "charging_station_count": len(cached_returns),
            "finite_return_count": len(finite_cached_returns),
            "unreachable_return_count": int((~np.isfinite(cached_returns)).sum()),
            "minimum_time_s": (
                float(finite_cached_returns.min()) if len(finite_cached_returns) else None
            ),
            "median_time_s": (
                float(np.median(finite_cached_returns)) if len(finite_cached_returns) else None
            ),
            "maximum_time_s": (
                float(finite_cached_returns.max()) if len(finite_cached_returns) else None
            ),
        },
        "feasibility_gate": {
            "policy": "fixed_amazon_template_single_customer_certificate_v1",
            "time_feasible_count": int(time_feasible.sum()),
            "energy_feasible_count": int(energy_feasible.sum()),
            "capacity_feasible_count": int(capacity_feasible.sum()),
            "requires_charging_count": int(certificates.requires_charging.sum()),
            "maximum_charging_visit_count": int(certificates.charging_visit_count.max()),
            "minimum_customer_transition_energy_margin_kwh": float(
                certificates.customer_transition_energy_margin_kwh.min()
            ),
            "passed": True,
        },
        "charging_usage_diagnostics": {
            "certificate_used_cs_count": len(certificate_cs),
            "solution_cs_visit_count": None,
            "binding_energy_count": int(
                np.count_nonzero(
                    certificates.customer_transition_energy_margin_kwh
                    <= soc_tolerance_kwh
                )
            ),
            "soc_tolerance_kwh": soc_tolerance_kwh,
            "baseline_solver_run": False,
        },
    }
    return ViewAttributes(
        package_counts=package_counts,
        demands_cm3=demands,
        service_time_s=service,
        time_windows_s=windows,
        feasible_arrival_time_s=feasible_arrival.astype(np.float32),
        feasible_return_duration_s=certificates.return_duration_s,
        feasibility_requires_charging=certificates.requires_charging,
        feasibility_charging_visit_count=certificates.charging_visit_count,
        feasibility_inbound_full_state_terminal_index=(
            certificates.inbound_full_state_terminal_index
        ),
        feasibility_first_post_customer_charger_terminal_index=(
            certificates.first_post_customer_charger_terminal_index
        ),
        feasibility_energy_margin_kwh=(
            certificates.customer_transition_energy_margin_kwh
        ),
        full_cs_to_depot_time_s=certificates.full_cs_to_depot_time_s,
        order_sampling_attempts=np.ones(customer_count, dtype=np.int16),
        report=report,
    )
