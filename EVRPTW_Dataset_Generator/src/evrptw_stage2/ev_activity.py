"""Deterministic route-level EV activity audit for the 140-family pilot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .orders import FullStateRouteCache, _build_full_state_route_cache
from .progress import atomic_write_json


REPORT_SCHEMA = "stage2_primary_view_ev_activity_audit_v1"
PRIMARY_SCALES = frozenset({"cus100", "cus500", "cus1000"})
TOLERANCE = 1e-6


@dataclass(frozen=True)
class RouteHeuristicResult:
    passed: bool
    customer_count: int
    served_customer_count: int
    vehicle_count: int
    charging_station_visit_count: int
    route_count_exceeding_battery_without_charging: int
    minimum_soc_kwh: float
    total_charging_time_s: float
    total_distance_km: float
    failure_reason: str | None = None


def _return_plan(
    *,
    current: int,
    current_time: float,
    soc: float,
    time_matrix: np.ndarray,
    energy_matrix: np.ndarray,
    chargers: np.ndarray,
    charging_power_kw: np.ndarray,
    derating: float,
    battery: float,
    horizon_end: float,
    cache: FullStateRouteCache,
) -> tuple[float, int | None, float] | None:
    options: list[tuple[float, int | None, float]] = []
    direct_energy = float(energy_matrix[current, 0])
    if direct_energy <= soc + TOLERANCE:
        finish = current_time + float(time_matrix[current, 0])
        if finish <= horizon_end + TOLERANCE:
            options.append((finish, None, 0.0))
    for offset, charger in enumerate(chargers):
        to_charger = float(energy_matrix[current, charger])
        if to_charger > soc + TOLERANCE or not np.isfinite(cache.to_depot_s[offset + 1]):
            continue
        residual = max(0.0, soc - to_charger)
        charge_time = (battery - residual) / (
            float(charging_power_kw[offset]) * derating
        ) * 3600.0
        finish = (
            current_time
            + float(time_matrix[current, charger])
            + charge_time
            + float(cache.to_depot_s[offset + 1])
        )
        if finish <= horizon_end + TOLERANCE:
            options.append((finish, int(charger), float(charge_time)))
    return (
        min(options, key=lambda item: (item[0], -1 if item[1] is None else item[1]))
        if options
        else None
    )


def _charger_choice(
    *,
    current: int,
    current_time: float,
    soc: float,
    unserved: np.ndarray,
    remaining_capacity: float,
    time_matrix: np.ndarray,
    energy_matrix: np.ndarray,
    demands: np.ndarray,
    service: np.ndarray,
    windows: np.ndarray,
    chargers: np.ndarray,
    charging_power_kw: np.ndarray,
    derating: float,
    battery: float,
    horizon_end: float,
    escape_energy: np.ndarray,
) -> tuple[int, float] | None:
    candidates: list[tuple[float, int, float]] = []
    customer_nodes = np.arange(1, len(unserved) + 1, dtype=int)
    for offset, charger in enumerate(chargers):
        leg_energy = float(energy_matrix[current, charger])
        if charger == current or leg_energy > soc + TOLERANCE:
            continue
        arrival = current_time + float(time_matrix[current, charger])
        remaining_soc = max(0.0, soc - leg_energy)
        charge_time = (battery - remaining_soc) / (
            float(charging_power_kw[offset]) * derating
        ) * 3600.0
        departure = arrival + charge_time
        travel = time_matrix[charger, customer_nodes]
        start = np.maximum(departure + travel, windows[:, 0])
        feasible = (
            unserved
            & (demands <= remaining_capacity + TOLERANCE)
            & (energy_matrix[charger, customer_nodes] + escape_energy <= battery + TOLERANCE)
            & (start + service + time_matrix[customer_nodes, 0] <= horizon_end + TOLERANCE)
        )
        if feasible.any():
            first_service = float(np.min(start[feasible]))
            candidates.append((first_service, int(charger), float(charge_time)))
    if not candidates:
        return None
    _, charger, charge_time = min(candidates)
    return charger, charge_time


def run_deterministic_route_heuristic(
    *,
    time_matrix_s: np.ndarray,
    distance_matrix_km: np.ndarray,
    demands_cm3: np.ndarray,
    service_time_s: np.ndarray,
    time_windows_s: np.ndarray,
    charging_power_kw: np.ndarray,
    battery_capacity_kwh: float,
    cargo_capacity_cm3: float,
    specific_energy_kwh_per_km: float,
    charging_derating_factor: float,
    horizon_start_s: float,
    horizon_end_s: float,
    use_energy_constraints: bool,
) -> RouteHeuristicResult:
    """Build deterministic nearest-feasible routes with optional full-charge stops."""

    time_matrix = np.asarray(time_matrix_s, dtype=float)
    distance_matrix = np.asarray(distance_matrix_km, dtype=float)
    demands = np.asarray(demands_cm3, dtype=float)
    service = np.asarray(service_time_s, dtype=float)
    windows = np.asarray(time_windows_s, dtype=float)
    power = np.asarray(charging_power_kw, dtype=float)
    n = len(demands)
    chargers = np.arange(1 + n, time_matrix.shape[0], dtype=int)
    if time_matrix.shape != distance_matrix.shape or time_matrix.shape[0] != 1 + n + len(power):
        raise ValueError("terminal/matrix/attribute shape mismatch")
    if np.any(power <= 0.0) or battery_capacity_kwh <= 0.0:
        raise ValueError("battery and charging power must be positive")
    energy = distance_matrix * float(specific_energy_kwh_per_km)
    customer_nodes = np.arange(1, n + 1, dtype=int)
    if len(chargers):
        escape = np.minimum(
            energy[customer_nodes, 0],
            np.min(energy[np.ix_(customer_nodes, chargers)], axis=1),
        )
    else:
        escape = energy[customer_nodes, 0]
    cache = _build_full_state_route_cache(
        customer_count=n,
        running_time_matrix_s=time_matrix,
        running_time_energy_matrix_kwh=energy,
        charging_power_kw=power,
        battery_capacity_kwh=float(battery_capacity_kwh),
        charging_power_derating_factor=float(charging_derating_factor),
    )

    unserved = np.ones(n, dtype=bool)
    vehicles = visits = binding_routes = served = 0
    total_distance = total_charging_time = 0.0
    minimum_soc = float(battery_capacity_kwh)
    while unserved.any():
        vehicles += 1
        current = 0
        now = float(horizon_start_s)
        soc = float(battery_capacity_kwh)
        remaining_capacity = float(cargo_capacity_cm3)
        route_customer_count = 0
        route_energy_without_reset = 0.0
        while True:
            travel = time_matrix[current, customer_nodes]
            service_start = np.maximum(now + travel, windows[:, 0])
            feasible = (
                unserved
                & (demands <= remaining_capacity + TOLERANCE)
                & (service_start <= windows[:, 1] + TOLERANCE)
                & (service_start + service + time_matrix[customer_nodes, 0] <= horizon_end_s + TOLERANCE)
            )
            if use_energy_constraints:
                feasible &= energy[current, customer_nodes] + escape <= soc + TOLERANCE
            if feasible.any():
                indices = sorted(
                    np.flatnonzero(feasible),
                    key=lambda i: (float(travel[i]), float(windows[i, 1]), int(i)),
                )
                chosen_offset = None
                for candidate in indices:
                    if not use_energy_constraints:
                        chosen_offset = int(candidate)
                        break
                    candidate_node = int(candidate + 1)
                    candidate_soc = soc - float(energy[current, candidate_node])
                    candidate_time = (
                        max(
                            now + float(time_matrix[current, candidate_node]),
                            float(windows[candidate, 0]),
                        )
                        + float(service[candidate])
                    )
                    if _return_plan(
                        current=candidate_node,
                        current_time=candidate_time,
                        soc=candidate_soc,
                        time_matrix=time_matrix,
                        energy_matrix=energy,
                        chargers=chargers,
                        charging_power_kw=power,
                        derating=float(charging_derating_factor),
                        battery=float(battery_capacity_kwh),
                        horizon_end=float(horizon_end_s),
                        cache=cache,
                    ) is not None:
                        chosen_offset = int(candidate)
                        break
                if chosen_offset is None:
                    feasible[:] = False
            if feasible.any():
                node = int(chosen_offset + 1)
                leg_distance = float(distance_matrix[current, node])
                leg_energy = float(energy[current, node])
                total_distance += leg_distance
                route_energy_without_reset += leg_energy
                soc -= leg_energy
                minimum_soc = min(minimum_soc, soc)
                now = max(now + float(time_matrix[current, node]), float(windows[chosen_offset, 0])) + float(service[chosen_offset])
                remaining_capacity -= float(demands[chosen_offset])
                unserved[chosen_offset] = False
                served += 1
                route_customer_count += 1
                current = node
                continue

            if use_energy_constraints and unserved.any():
                choice = _charger_choice(
                    current=current,
                    current_time=now,
                    soc=soc,
                    unserved=unserved,
                    remaining_capacity=remaining_capacity,
                    time_matrix=time_matrix,
                    energy_matrix=energy,
                    demands=demands,
                    service=service,
                    windows=windows,
                    chargers=chargers,
                    charging_power_kw=power,
                    derating=float(charging_derating_factor),
                    battery=float(battery_capacity_kwh),
                    horizon_end=float(horizon_end_s),
                    escape_energy=escape,
                )
                if choice is not None:
                    charger, charge_time = choice
                    leg_distance = float(distance_matrix[current, charger])
                    leg_energy = float(energy[current, charger])
                    total_distance += leg_distance
                    route_energy_without_reset += leg_energy
                    soc -= leg_energy
                    minimum_soc = min(minimum_soc, soc)
                    now += float(time_matrix[current, charger]) + charge_time
                    total_charging_time += charge_time
                    visits += 1
                    soc = float(battery_capacity_kwh)
                    current = charger
                    continue

            if route_customer_count == 0:
                return RouteHeuristicResult(
                    False, n, served, vehicles, visits, binding_routes,
                    float(minimum_soc), float(total_charging_time), float(total_distance),
                    "no_individually_feasible_unserved_customer",
                )

            direct_energy = float(energy[current, 0])
            direct_time = float(time_matrix[current, 0])
            if not use_energy_constraints and now + direct_time <= horizon_end_s + TOLERANCE:
                total_distance += float(distance_matrix[current, 0])
                route_energy_without_reset += direct_energy
                soc -= direct_energy
                minimum_soc = min(minimum_soc, soc)
                now += direct_time
            elif use_energy_constraints:
                plan = _return_plan(
                    current=current,
                    current_time=now,
                    soc=soc,
                    time_matrix=time_matrix,
                    energy_matrix=energy,
                    chargers=chargers,
                    charging_power_kw=power,
                    derating=float(charging_derating_factor),
                    battery=float(battery_capacity_kwh),
                    horizon_end=float(horizon_end_s),
                    cache=cache,
                )
                if plan is None:
                    return RouteHeuristicResult(
                        False, n, served, vehicles, visits, binding_routes,
                        float(minimum_soc), float(total_charging_time), float(total_distance),
                        "route_cannot_return_to_depot_with_energy_and_time",
                    )
                _, charger, charge_time = plan
                if charger is None:
                    total_distance += float(distance_matrix[current, 0])
                    route_energy_without_reset += float(energy[current, 0])
                    soc -= float(energy[current, 0])
                    minimum_soc = min(minimum_soc, soc)
                else:
                    total_distance += float(distance_matrix[current, charger])
                    route_energy_without_reset += float(energy[current, charger])
                    soc -= float(energy[current, charger])
                    minimum_soc = min(minimum_soc, soc)
                    total_charging_time += charge_time
                    visits += 1
                    soc = float(battery_capacity_kwh)
                    full_position = int(np.flatnonzero(cache.terminal_indices == charger)[0])
                    while full_position != 0:
                        next_position = int(cache.to_depot_reverse_predecessor[full_position])
                        if next_position < 0:
                            raise RuntimeError("broken multi-hop CS-to-depot predecessor chain")
                        origin = int(cache.terminal_indices[full_position])
                        destination = int(cache.terminal_indices[next_position])
                        leg_energy = float(energy[origin, destination])
                        total_distance += float(distance_matrix[origin, destination])
                        route_energy_without_reset += leg_energy
                        soc -= leg_energy
                        minimum_soc = min(minimum_soc, soc)
                        if next_position != 0:
                            destination_power = float(power[next_position - 1])
                            recharge = (battery_capacity_kwh - max(0.0, soc)) / (
                                destination_power * charging_derating_factor
                            ) * 3600.0
                            total_charging_time += recharge
                            visits += 1
                            soc = float(battery_capacity_kwh)
                        full_position = next_position
            if route_energy_without_reset > battery_capacity_kwh + TOLERANCE:
                binding_routes += 1
            break
    return RouteHeuristicResult(
        True, n, served, vehicles, visits, binding_routes,
        float(minimum_soc), float(total_charging_time), float(total_distance), None,
    )


def audit_primary_pilot_views(
    *,
    instance_root: Path,
    code_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    family_root = instance_root / "materialized" / "families"
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(family_root.glob("*/family_manifest.json")):
        family_dir = manifest_path.parent
        family = json.loads(manifest_path.read_text(encoding="utf-8"))
        terminal_count = int(family["terminal_count"])
        time_parent = np.load(family_dir / family["matrix_files"]["running_time_shortest_matrix_s"], mmap_mode="r", allow_pickle=False)
        distance_parent = np.load(family_dir / family["matrix_files"]["running_time_path_distance_km"], mmap_mode="r", allow_pickle=False)
        if time_parent.shape != (terminal_count, terminal_count):
            raise ValueError(f"{family['family_id']}: parent time matrix shape mismatch")
        for view_path in sorted((family_dir / "views").glob("*/view_manifest.json")):
            view = json.loads(view_path.read_text(encoding="utf-8"))
            if str(view["scale_id"]) not in PRIMARY_SCALES:
                continue
            view_dir = view_path.parent
            indices = np.load(view_dir / view["terminal_parent_indices"], allow_pickle=False)
            time_matrix = np.asarray(time_parent[np.ix_(indices, indices)])
            distance_matrix = np.asarray(distance_parent[np.ix_(indices, indices)])
            with np.load(view_dir / view["customer_attributes"], allow_pickle=False) as data:
                demands = data["demands_cm3"]
                service = data["service_time_s"]
                windows = data["time_windows_s"]
            with np.load(view_dir / view["charging_attributes"], allow_pickle=False) as data:
                power = data["charging_power_kw"]
            common = dict(
                time_matrix_s=time_matrix,
                distance_matrix_km=distance_matrix,
                demands_cm3=demands,
                service_time_s=service,
                time_windows_s=windows,
                charging_power_kw=power,
                battery_capacity_kwh=float(view["vehicle"]["battery_capacity_kwh"]),
                cargo_capacity_cm3=float(view["vehicle"]["cargo_capacity_cm3"]),
                specific_energy_kwh_per_km=float(view["vehicle"]["specific_energy_consumption_kwh_per_km"]),
                charging_derating_factor=float(view["charging_policy"]["charging_power_derating_factor"]),
                horizon_start_s=float(view["operating_horizon_s"][0]),
                horizon_end_s=float(view["operating_horizon_s"][1]),
            )
            no_energy = run_deterministic_route_heuristic(**common, use_energy_constraints=False)
            energy = run_deterministic_route_heuristic(**common, use_energy_constraints=True)
            effect = bool(
                abs(energy.total_distance_km - no_energy.total_distance_km) > 1e-5
                or energy.vehicle_count != no_energy.vehicle_count
            )
            rows.append({
                "family_id": str(family["family_id"]),
                "view_id": str(view["view_id"]),
                "city_slug": str(family["city_slug"]),
                "scale_id": str(view["scale_id"]),
                "day_type": str(view["day_type"]),
                "passed": no_energy.passed and energy.passed,
                "no_energy_ablation": no_energy.__dict__,
                "energy_aware": energy.__dict__,
                "battery_binding_route_count_without_cs": no_energy.route_count_exceeding_battery_without_charging,
                "energy_ablation_effect": effect,
                "distance_delta_km": float(energy.total_distance_km - no_energy.total_distance_km),
                "vehicle_count_delta": int(energy.vehicle_count - no_energy.vehicle_count),
            })
    total_visits = sum(row["energy_aware"]["charging_station_visit_count"] for row in rows)
    binding_routes = sum(row["battery_binding_route_count_without_cs"] for row in rows)
    effect_views = sum(bool(row["energy_ablation_effect"]) for row in rows)
    incomplete = [row for row in rows if not row["passed"]]
    all_zero = total_visits == 0 and binding_routes == 0 and effect_views == 0
    return {
        "schema": REPORT_SCHEMA,
        "status": "passed" if rows and not incomplete and not all_zero else "failed",
        "passed": bool(rows and not incomplete and not all_zero),
        "scope": "all_primary_views_in_existing_140_family_pilot",
        "heuristic": "deterministic_nearest_feasible_full_charge_route_heuristic_v1",
        "primary_scales": sorted(PRIMARY_SCALES),
        "view_count": len(rows),
        "completed_view_count": len(rows) - len(incomplete),
        "charging_station_visit_count": int(total_visits),
        "battery_binding_route_count_without_cs": int(binding_routes),
        "energy_ablation_effect_view_count": int(effect_views),
        "minimum_energy_aware_soc_kwh": min((row["energy_aware"]["minimum_soc_kwh"] for row in rows), default=None),
        "total_charging_time_s": float(sum(row["energy_aware"]["total_charging_time_s"] for row in rows)),
        "degenerate_all_zero_stop_triggered": all_zero,
        "stop_condition": "cs_visits_and_battery_binding_routes_and_energy_ablation_effect_are_all_zero",
        "incomplete_views": [{"family_id": row["family_id"], "view_id": row["view_id"], "no_energy_failure": row["no_energy_ablation"]["failure_reason"], "energy_failure": row["energy_aware"]["failure_reason"]} for row in incomplete],
        "rows": rows,
        "code_provenance": dict(code_provenance or {}),
        "hash_validation_performed": False,
    }


def write_primary_pilot_ev_activity_audit(*, output: Path, **kwargs: Any) -> dict[str, Any]:
    report = audit_primary_pilot_views(**kwargs)
    atomic_write_json(output, report)
    return report
