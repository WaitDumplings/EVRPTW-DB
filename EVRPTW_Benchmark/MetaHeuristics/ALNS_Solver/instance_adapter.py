from __future__ import annotations

from typing import Any

import numpy as np

from evrptw_core.schema import EVRPTWInstance, merge_route_sequences
from benchmark_common import (
    certificate_singleton_routes,
    charging_profile,
    running_time_energy_matrix_kwh,
    running_time_matrix_s,
)


def to_alns_tensor_instance(instance: EVRPTWInstance) -> dict[str, Any]:
    """Convert the canonical pickle schema into the ALNS tensor-style dict.

    ALNS internally uses minutes for time, km for distance, kWh for battery,
    and the canonical node order [depot, customers, charging stations]. The
    objective remains total road distance in km, matching the exact solver.
    """
    battery_capacity = float(instance.vehicle.get("battery_capacity_kwh", 100.0))
    charging_power_kw, charging_power_factor, charging_power_source = charging_profile(instance)
    effective_speed_kmh = float(
        instance.speed_profile.get("effective_speed_kmh")
        or instance.vehicle.get("design_speed_kmh")
        or 40.0
    )
    effective_speed_km_per_min = effective_speed_kmh / 60.0

    distance_matrix_km = np.asarray(instance.distance_matrix_km, dtype=np.float64)
    time_matrix_min = running_time_matrix_s(instance) / 60.0
    energy_matrix_kwh = running_time_energy_matrix_kwh(instance)
    verified_certificate_routes = certificate_singleton_routes(instance)

    return {
        "instance_id": instance.instance_id,
        "depot": np.asarray(instance.depot, dtype=np.float64).reshape(1, 2),
        "customers": np.asarray(instance.customers, dtype=np.float64),
        "charging_stations": np.asarray(instance.charging_stations, dtype=np.float64),
        "customer_demand": np.asarray(instance.demands_cm3, dtype=np.float64),
        "customer_service": np.asarray(instance.service_time_s, dtype=np.float64),
        "tw": np.asarray(instance.tw_s, dtype=np.float64),
        "distance_matrix_km": distance_matrix_km,
        "time_matrix_min": time_matrix_min,
        "energy_matrix_kwh": energy_matrix_kwh,
        "charging_power_kw": charging_power_kw,
        "charging_power_derating_factor": charging_power_factor,
        "charging_power_source": charging_power_source,
        # Only the reconstructed routes are suitable for a warm start: the
        # helper has independently replayed them under the canonical contract.
        # The raw generator witness is retained for diagnostics only.
        "certificate_singleton_routes": verified_certificate_routes,
        "feasibility_certificate": instance.raw.get("feasibility_certificate"),
        "env": {
            "instance_startTime": float(instance.working_start_s),
            "instance_endTime": float(instance.working_end_s),
            "battery_capacity": battery_capacity,
            "loading_capacity": float(instance.vehicle.get("cargo_capacity_cm3", np.inf)),
            "consumption_per_distance": float(instance.vehicle.get("consumption_kwh_per_km", 0.404)),
            # Legacy scalar fallback is retained for non-station code paths;
            # Stage 2 charging always uses the per-station power vector above.
            "charging_speed": float("inf"),
            "speed": effective_speed_km_per_min,
        },
    }


def flatten_routes(routes: list[list[int]]) -> list[int]:
    return merge_route_sequences(routes)
