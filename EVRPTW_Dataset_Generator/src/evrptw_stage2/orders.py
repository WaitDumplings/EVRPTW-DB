"""View-level packages, volume, service time, time windows, and feasibility."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra


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


def _build_full_state_route_cache(
    *,
    customer_count: int,
    running_time_matrix_s: np.ndarray,
    running_time_energy_matrix_kwh: np.ndarray,
    charging_power_kw: np.ndarray,
    battery_capacity_kwh: float,
    charging_efficiency: float,
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
            / (float(charging_power_kw[destination - 1]) * charging_efficiency)
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


def _truncated_normal(
    rng: np.random.Generator,
    *,
    mean: float,
    std: float,
    low: float,
    high: float,
) -> float:
    if std <= 0.0:
        return float(np.clip(mean, low, high))
    for _ in range(64):
        value = float(rng.normal(mean, std))
        if low <= value <= high:
            return value
    return float(np.clip(mean, low, high))


def _sample_packages_and_volume(
    customers: pd.DataFrame,
    *,
    day_type: str,
    profile: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    activation = profile["customer_activation"]
    per_unit_probability = float(activation["per_unit_order_probability"][day_type])
    units = pd.to_numeric(customers["residential_units"], errors="coerce").fillna(1)
    units = units.clip(lower=1, upper=5000).to_numpy(dtype=np.int32)
    ordering_units = rng.binomial(units, per_unit_probability).astype(np.int32)
    zero = ordering_units == 0
    while zero.any():
        ordering_units[zero] = rng.binomial(units[zero], per_unit_probability)
        zero = ordering_units == 0

    package_cfg = profile["packages"]
    extra_cfg = package_cfg["extra_packages_per_ordering_unit"]
    mean_extra = float(extra_cfg["mean"])
    dispersion = float(extra_cfg["dispersion"])
    probability = dispersion / (dispersion + mean_extra)
    extra = np.asarray(
        [
            rng.negative_binomial(dispersion * int(active_units), probability)
            for active_units in ordering_units
        ],
        dtype=np.int32,
    )
    max_packages = int(package_cfg["max_packages_per_location"])
    package_counts = np.clip(ordering_units + extra, 1, max_packages).astype(np.int32)

    volume_cfg = package_cfg["per_package_volume_lognormal"]
    median = float(volume_cfg["median_cm3"])
    sigma = float(volume_cfg["sigma"])
    maximum = float(volume_cfg["max_package_volume_cm3"])
    demands = np.empty(len(customers), dtype=np.float32)
    for index, count in enumerate(package_counts):
        parcels = rng.lognormal(math_log(median), sigma, size=int(count))
        demands[index] = float(np.clip(parcels, 1.0, maximum).sum())
    return ordering_units, package_counts, demands


def math_log(value: float) -> float:
    if value <= 0.0:
        raise ValueError("Lognormal median must be positive")
    return float(np.log(value))


def _sample_service_time(
    demands_cm3: np.ndarray,
    package_counts: np.ndarray,
    *,
    profile: Mapping[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    cfg = profile["service_time"]
    deterministic = (
        float(cfg["base_seconds"])
        + float(cfg["beta_package_seconds"]) * package_counts.astype(float)
        + float(cfg["beta_volume_seconds_per_cm3"]) * demands_cm3.astype(float)
    )
    sigma = float(cfg["lognormal_noise_sigma"])
    noise = rng.lognormal(-0.5 * sigma**2, sigma, size=len(deterministic))
    values = deterministic * noise
    return np.clip(values, float(cfg["min_seconds"]), float(cfg["max_seconds"])).astype(
        np.float32
    )


def _sample_time_windows(
    *,
    customer_count: int,
    day_type: str,
    service_time_s: np.ndarray,
    feasible_arrival_time_s: np.ndarray,
    feasible_return_duration_s: np.ndarray,
    operating_start_s: int,
    operating_end_s: int,
    profile: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = profile["time_window"]
    windows = np.column_stack(
        [
            np.full(customer_count, operating_start_s, dtype=np.float32),
            np.full(customer_count, operating_end_s, dtype=np.float32),
        ]
    )
    beta = cfg["presence_rate_beta"][day_type]
    rate = float(rng.beta(float(beta["alpha"]), float(beta["beta"])))
    count = min(customer_count, round(rate * customer_count))
    earliest = feasible_arrival_time_s
    latest = operating_end_s - feasible_return_duration_s - service_time_s
    feasible = np.flatnonzero(earliest <= latest)
    if count <= 0 or not len(feasible):
        return windows, {"target_rate": rate, "tight_window_count": 0}
    service_rank = np.argsort(np.argsort(service_time_s)).astype(float)
    service_rank /= max(customer_count - 1, 1)
    weights = 1.0 + 1.5 * service_rank[feasible]
    weights /= weights.sum()
    chosen = rng.choice(feasible, size=min(count, len(feasible)), replace=False, p=weights)
    strain_share = float(cfg["realistic_strain_share"][day_type])
    labels = np.where(rng.random(len(chosen)) < strain_share, "strain", "loose")
    for customer_index, label in zip(chosen, labels):
        specification = cfg["profiles"][day_type][str(label)]
        width_h = _truncated_normal(
            rng,
            mean=float(specification["width_mean_h"]),
            std=float(specification["width_std_h"]),
            low=float(specification["width_min_h"]),
            high=float(specification["width_max_h"]),
        )
        center_h = float(
            np.clip(
                rng.normal(
                    float(specification["center_mean_h"]),
                    float(specification["center_std_h"]),
                ),
                operating_start_s / 3600.0,
                operating_end_s / 3600.0,
            )
        )
        start = center_h * 3600.0 - width_h * 1800.0
        end = center_h * 3600.0 + width_h * 1800.0
        start = max(float(earliest[customer_index]), start, float(operating_start_s))
        end = min(float(latest[customer_index]), end, float(operating_end_s))
        if end > start:
            windows[customer_index] = (start, end)
    tight = (windows[:, 0] > operating_start_s) | (windows[:, 1] < operating_end_s)
    return windows, {
        "target_rate": rate,
        "target_tight_window_count": count,
        "tight_window_count": int(tight.sum()),
        "actual_tight_window_rate": float(tight.mean()),
    }


def _single_customer_certificates(
    *,
    customer_count: int,
    running_time_matrix_s: np.ndarray,
    running_time_energy_matrix_kwh: np.ndarray,
    charging_power_kw: np.ndarray,
    battery_capacity_kwh: float,
    charging_efficiency: float,
) -> SingleCustomerCertificates:
    """Construct one-customer routes with optional full-charge CS visits.

    All paths are evaluated on the already materialized directed running-time
    terminal closure. A depot or charging station is a full-battery state. The
    customer does not reset energy, so the incoming and outgoing customer legs
    must jointly fit between two consecutive full-battery states.
    """
    terminal_count = running_time_matrix_s.shape[0]
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
    if not 0.0 < charging_efficiency <= 1.0:
        raise ValueError("charging_efficiency must be in (0, 1]")

    customer_nodes = np.arange(1, 1 + customer_count, dtype=np.int32)
    charger_nodes = np.arange(1 + customer_count, terminal_count, dtype=np.int32)
    cache = _build_full_state_route_cache(
        customer_count=customer_count,
        running_time_matrix_s=running_time_matrix_s,
        running_time_energy_matrix_kwh=running_time_energy_matrix_kwh,
        charging_power_kw=charging_power_kw,
        battery_capacity_kwh=battery_capacity_kwh,
        charging_efficiency=charging_efficiency,
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
                    / (charging_power_kw[allowed].astype(float) * charging_efficiency)
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
    )


def build_view_attributes(
    customer_rows: pd.DataFrame,
    *,
    day_type: str,
    package_seed: int,
    service_time_seed: int,
    time_window_seed: int,
    operating_start_s: int,
    operating_end_s: int,
    running_time_matrix_s: np.ndarray,
    running_time_energy_matrix_kwh: np.ndarray,
    charging_power_kw: np.ndarray,
    profile: Mapping[str, Any],
) -> ViewAttributes:
    customer_count = len(customer_rows)
    expected_terminal_count = int(running_time_matrix_s.shape[0])
    if running_time_matrix_s.shape != (expected_terminal_count, expected_terminal_count):
        raise ValueError("Running-time matrix must be square")
    if running_time_energy_matrix_kwh.shape != running_time_matrix_s.shape:
        raise ValueError("Running-time energy matrix shape mismatch")
    if expected_terminal_count <= customer_count:
        raise ValueError("View matrix must include one depot and its charging stations")

    package_rng = np.random.default_rng(package_seed)
    ordering_units, package_counts, demands = _sample_packages_and_volume(
        customer_rows,
        day_type=day_type,
        profile=profile,
        rng=package_rng,
    )
    service = _sample_service_time(
        demands,
        package_counts,
        profile=profile,
        rng=np.random.default_rng(service_time_seed),
    )
    battery = float(profile["energy"]["battery_capacity_kwh"])
    charging_efficiency = float(profile["charging"]["charging_efficiency"])
    certificates = _single_customer_certificates(
        customer_count=customer_count,
        running_time_matrix_s=running_time_matrix_s,
        running_time_energy_matrix_kwh=running_time_energy_matrix_kwh,
        charging_power_kw=np.asarray(charging_power_kw, dtype=np.float32),
        battery_capacity_kwh=battery,
        charging_efficiency=charging_efficiency,
    )
    feasible_arrival_time = float(operating_start_s) + certificates.arrival_elapsed_s
    windows, time_window_report = _sample_time_windows(
        customer_count=customer_count,
        day_type=day_type,
        service_time_s=service,
        feasible_arrival_time_s=feasible_arrival_time,
        feasible_return_duration_s=certificates.return_duration_s,
        operating_start_s=operating_start_s,
        operating_end_s=operating_end_s,
        profile=profile,
        rng=np.random.default_rng(time_window_seed),
    )

    arrival = np.maximum(feasible_arrival_time, windows[:, 0])
    route_time_feasible = (
        (arrival <= windows[:, 1] + 1e-6)
        & (
            arrival
            + service
            + certificates.return_duration_s
            <= float(operating_end_s) + 1e-6
        )
    )
    route_energy_feasible = np.isfinite(certificates.arrival_elapsed_s) & np.isfinite(
        certificates.return_duration_s
    )
    cargo_capacity = float(profile["vehicle"]["cargo_capacity_cm3"])
    capacity_feasible = demands <= cargo_capacity + 1e-6
    sufficient_feasible = route_time_feasible & route_energy_feasible & capacity_feasible
    if not sufficient_feasible.all():
        raise ValueError(
            "View failed the unlimited-fleet direct-service sufficient feasibility gate: "
            f"time={int((~route_time_feasible).sum())}, "
            f"energy={int((~route_energy_feasible).sum())}, "
            f"capacity={int((~capacity_feasible).sum())}"
        )
    report = {
        "schema": "cle_evrptw_view_attribute_report_v1",
        "day_type": day_type,
        "customer_count": customer_count,
        "package_count_mean": float(package_counts.mean()),
        "package_count_p90": float(np.quantile(package_counts, 0.90)),
        "ordering_unit_count_mean": float(ordering_units.mean()),
        "demand_cm3_mean": float(demands.mean()),
        "demand_cm3_p90": float(np.quantile(demands, 0.90)),
        "service_time_s_mean": float(service.mean()),
        "service_time_s_p90": float(np.quantile(service, 0.90)),
        "time_windows": time_window_report,
        "feasibility_gate": {
            "policy": (
                "unlimited_fleet_individual_service_with_optional_full_charge_"
                "sufficient_condition_v1"
            ),
            "time_feasible_count": int(route_time_feasible.sum()),
            "energy_feasible_count": int(route_energy_feasible.sum()),
            "capacity_feasible_count": int(capacity_feasible.sum()),
            "requires_charging_count": int(certificates.requires_charging.sum()),
            "maximum_charging_visit_count": int(certificates.charging_visit_count.max()),
            "minimum_customer_transition_energy_margin_kwh": float(
                certificates.customer_transition_energy_margin_kwh.min()
            ),
            "passed": bool(sufficient_feasible.all()),
        },
    }
    return ViewAttributes(
        package_counts=package_counts,
        demands_cm3=demands,
        service_time_s=service,
        time_windows_s=windows.astype(np.float32),
        feasible_arrival_time_s=feasible_arrival_time.astype(np.float32),
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
        report=report,
    )
