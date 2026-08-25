from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from evrptw_core.schema import EVRPTWInstance


@dataclass(frozen=True)
class ChargingProfile:
    power_kw: np.ndarray
    power_factor: float
    power_source: str
    power_factor_source: str


def resolve_charging_profile(instance: EVRPTWInstance) -> ChargingProfile:
    m = instance.num_charging_stations
    raw_power = instance.raw.get("charging_power_kw")
    power_source = "charging_power_kw"
    if raw_power is None:
        raw_power = instance.cs_activation.get("charging_power_kw")
        power_source = "cs_activation.charging_power_kw"

    policy = instance.raw.get("charging_policy", {})
    if "charging_power_derating_factor" in policy:
        if "charging_efficiency" in policy:
            raise ValueError("charging policy cannot define both derating and efficiency")
        power_factor = float(policy["charging_power_derating_factor"])
        factor_source = "charging_policy.charging_power_derating_factor"
    elif "charging_efficiency" in policy:
        power_factor = float(policy["charging_efficiency"])
        factor_source = "charging_policy.charging_efficiency_legacy"
    elif "charging_power_derating_factor" in instance.vehicle:
        power_factor = float(instance.vehicle["charging_power_derating_factor"])
        factor_source = "vehicle.charging_power_derating_factor"
    else:
        power_factor = float(instance.vehicle.get("charging_efficiency", 1.0))
        factor_source = "vehicle.charging_efficiency_legacy_default"
    if not 0.0 < power_factor <= 1.0:
        raise ValueError(
            "charging power factor must be in (0, 1], "
            f"got {power_factor}"
        )

    if raw_power is None and m:
        raise ValueError(
            "CLE-backed Stage-2 charging stations require charging_power_kw"
        )

    power = np.asarray(
        np.empty(0, dtype=np.float64) if raw_power is None else raw_power,
        dtype=np.float64,
    )
    if power.shape != (m,):
        raise ValueError(
            f"charging_power_kw must have shape {(m,)}, got {power.shape}"
        )
    if np.any(~np.isfinite(power)) or np.any(power <= 0.0):
        raise ValueError("charging_power_kw must contain finite positive values")
    return ChargingProfile(
        power_kw=power,
        power_factor=power_factor,
        power_source=power_source,
        power_factor_source=factor_source,
    )


def validate_routes(
    instance: EVRPTWInstance,
    routes: list[list[int]],
    *,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Independently replay routes under the current Stage-2 resource contract."""

    n = instance.num_customers
    terminal_count = instance.num_terminals
    first_cs = n + 1
    profile = resolve_charging_profile(instance)
    distance = np.asarray(instance.distance_matrix_km, dtype=float)
    travel = _running_time_matrix(instance)
    energy = _running_time_energy_matrix(instance)
    battery_capacity = float(instance.vehicle["battery_capacity_kwh"])
    cargo_capacity = float(instance.vehicle["cargo_capacity_cm3"])

    violations: list[str] = []
    customer_visits = np.zeros(n, dtype=np.int32)
    total_distance = 0.0
    total_charge_time = 0.0
    charging_visits = 0
    route_metrics: list[dict[str, Any]] = []

    for route_index, route_raw in enumerate(routes):
        route = [int(node) for node in route_raw]
        if len(route) < 3 or route[0] != 0 or route[-1] != 0:
            violations.append(f"route {route_index} must start and end at depot 0")
            continue
        if any(node < 0 or node >= terminal_count for node in route):
            violations.append(f"route {route_index} contains an invalid terminal")
            continue

        current_time = float(instance.working_start_s)
        remaining_battery = battery_capacity
        load = 0.0
        route_distance = 0.0
        route_charge_time = 0.0

        for arc_index, (origin, destination) in enumerate(zip(route, route[1:])):
            arc_distance = float(distance[origin, destination])
            arc_time = float(travel[origin, destination])
            arc_energy = float(energy[origin, destination])
            if not all(np.isfinite(value) for value in (arc_distance, arc_time, arc_energy)):
                violations.append(
                    f"route {route_index} arc {arc_index} has a non-finite metric"
                )
                break
            route_distance += arc_distance
            current_time += arc_time
            remaining_battery -= arc_energy
            if remaining_battery < -tolerance:
                violations.append(
                    f"route {route_index} arc {origin}->{destination} exceeds battery"
                )

            if 1 <= destination <= n:
                customer = destination - 1
                customer_visits[customer] += 1
                service_start = max(
                    current_time,
                    float(instance.tw_s[customer, 0]),
                )
                if service_start > float(instance.tw_s[customer, 1]) + tolerance:
                    violations.append(
                        f"route {route_index} reaches customer {destination} after its TW"
                    )
                current_time = service_start + float(instance.service_time_s[customer])
                load += float(instance.demands_cm3[customer])
                if load > cargo_capacity + tolerance:
                    violations.append(f"route {route_index} exceeds cargo capacity")
            elif destination >= first_cs:
                station = destination - first_cs
                missing_energy = max(0.0, battery_capacity - remaining_battery)
                charge_time = (
                    missing_energy
                    / (profile.power_factor * float(profile.power_kw[station]))
                    * 3600.0
                )
                current_time += charge_time
                route_charge_time += charge_time
                charging_visits += 1
                remaining_battery = battery_capacity

        if current_time > float(instance.working_end_s) + tolerance:
            violations.append(f"route {route_index} returns after the operating horizon")
        total_distance += route_distance
        total_charge_time += route_charge_time
        route_metrics.append(
            {
                "route_index": route_index,
                "distance_km": route_distance,
                "end_time_s": current_time,
                "load_cm3": load,
                "remaining_battery_kwh": remaining_battery,
                "charging_time_s": route_charge_time,
            }
        )

    missing = np.flatnonzero(customer_visits == 0) + 1
    duplicate = np.flatnonzero(customer_visits > 1) + 1
    if missing.size:
        violations.append(f"customers not served exactly once: {missing.tolist()}")
    if duplicate.size:
        violations.append(f"customers served more than once: {duplicate.tolist()}")

    return {
        "passed": not violations,
        "violations": violations,
        "objective_distance_km": total_distance,
        "charging_visit_count": charging_visits,
        "total_charging_time_s": total_charge_time,
        "charging_power_source": profile.power_source,
        "charging_power_factor_source": profile.power_factor_source,
        "charging_power_derating_factor": profile.power_factor,
        "route_metrics": route_metrics,
    }


def _running_time_matrix(instance: EVRPTWInstance) -> np.ndarray:
    matrix = instance.raw.get("running_time_shortest_matrix_s")
    if matrix is None:
        matrix = instance.raw_travel_time_matrix_s
    if matrix is None:
        effective_speed = float(
            instance.speed_profile.get("effective_speed_kmh")
            or instance.vehicle.get("design_speed_kmh")
            or 40.0
        )
        matrix = (
            np.asarray(instance.distance_matrix_km, dtype=float)
            / max(effective_speed, 1e-9)
            * 3600.0
        )
    return np.asarray(matrix, dtype=float)


def _running_time_energy_matrix(instance: EVRPTWInstance) -> np.ndarray:
    matrix = instance.raw.get("running_time_path_energy_kwh")
    if matrix is None:
        matrix = instance.energy_matrix_kwh
    if matrix is None:
        consumption = float(
            instance.vehicle.get(
                "consumption_kwh_per_km",
                instance.vehicle.get("specific_energy_consumption_kwh_per_km", 0.404),
            )
        )
        matrix = np.asarray(instance.distance_matrix_km, dtype=float) * consumption
    return np.asarray(matrix, dtype=float)
