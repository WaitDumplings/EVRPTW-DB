from __future__ import annotations

from collections import namedtuple
from typing import Any

import numpy as np

from evrptw_core.schema import EVRPTWInstance, merge_route_sequences
from benchmark_common import (
    charging_profile,
    running_time_energy_matrix_kwh,
    running_time_matrix_s,
)


Customer = namedtuple("Customer", ["id", "type", "x", "y", "demand", "ready", "due", "service"])
Station = namedtuple("Station", Customer._fields)
Depot = namedtuple("Depot", Customer._fields)


class Route:
    def __init__(self, nodes=None):
        self.nodes = list(nodes) if nodes is not None else []
        self.load = 0.0
        self.time = 0.0
        self.fuel = 0.0


class VNSTInstance:
    def __init__(self) -> None:
        self.instance_id = ""
        self.depot = None
        self.customers = []
        self.stations = []
        self.vehicle_params = {
            "fuel_cap": None,
            "load_cap": None,
            "consump_rate": None,
            "velocity": None,
        }
        self.dist_matrix = None
        self.time_matrix = None
        self.energy_matrix = None
        self.terminal_order = []
        self.station_charging_power_kw = {}


def to_vnst_instance(instance: EVRPTWInstance) -> VNSTInstance:
    """Convert canonical pickle schema to the legacy VNS-TS object model.

    VNS-TS expects node order [depot, stations, customers] in its internal
    distance matrix, while the benchmark schema uses [depot, customers,
    stations]. Node ids remain benchmark-terminal ids so returned routes are
    already compatible with EVRPTW_Core.
    """
    out = VNSTInstance()
    out.instance_id = instance.instance_id

    depot_xy = np.asarray(instance.depot, dtype=float).reshape(2)
    out.depot = Depot(
        0,
        "d",
        float(depot_xy[0]),
        float(depot_xy[1]),
        0.0,
        float(instance.working_start_s),
        float(instance.working_end_s),
        0.0,
    )

    out.stations = []
    for station_idx, xy in enumerate(np.asarray(instance.charging_stations, dtype=float), start=1):
        terminal_id = instance.num_customers + station_idx
        out.stations.append(
            Station(
                int(terminal_id),
                "f",
                float(xy[0]),
                float(xy[1]),
                0.0,
                float(instance.working_start_s),
                float(instance.working_end_s),
                0.0,
            )
        )

    out.customers = []
    customers = np.asarray(instance.customers, dtype=float)
    demands = np.asarray(instance.demands_cm3, dtype=float)
    tw = np.asarray(instance.tw_s, dtype=float)
    service = np.asarray(instance.service_time_s, dtype=float)
    for customer_idx, xy in enumerate(customers, start=1):
        out.customers.append(
            Customer(
                int(customer_idx),
                "c",
                float(xy[0]),
                float(xy[1]),
                float(demands[customer_idx - 1]),
                float(tw[customer_idx - 1, 0]),
                float(tw[customer_idx - 1, 1]),
                float(service[customer_idx - 1]),
            )
        )

    battery_capacity = float(instance.vehicle.get("battery_capacity_kwh", 100.0))
    charging_power_kw, charging_efficiency, charging_power_source = charging_profile(instance)
    effective_speed_kmh = float(
        instance.speed_profile.get("effective_speed_kmh")
        or instance.vehicle.get("design_speed_kmh")
        or 40.0
    )

    out.vehicle_params = {
        "fuel_cap": battery_capacity,
        "load_cap": float(instance.vehicle.get("cargo_capacity_cm3", np.inf)),
        "consump_rate": float(instance.vehicle.get("consumption_kwh_per_km", 0.404)),
        "charging_efficiency": charging_efficiency,
        "charging_power_source": charging_power_source,
        "velocity": effective_speed_kmh / 3600.0,
    }

    canonical_distance = np.asarray(instance.distance_matrix_km, dtype=np.float64)
    station_ids = list(range(instance.num_customers + 1, instance.num_terminals))
    customer_ids = list(range(1, instance.num_customers + 1))
    order = [0] + station_ids + customer_ids
    out.terminal_order = order
    out.dist_matrix = canonical_distance[np.ix_(order, order)]
    out.time_matrix = running_time_matrix_s(instance)[np.ix_(order, order)]
    out.energy_matrix = running_time_energy_matrix_kwh(instance)[np.ix_(order, order)]
    out.station_charging_power_kw = {
        int(instance.num_customers + station_offset + 1): float(power)
        for station_offset, power in enumerate(charging_power_kw)
    }
    return out


def routes_to_terminal_ids(solution: list[Route] | None) -> list[list[int]]:
    if solution is None:
        return []
    routes: list[list[int]] = []
    for route in solution:
        ids = [int(getattr(node, "id", node)) for node in getattr(route, "nodes", route)]
        if not ids:
            continue
        if not any(node_id != 0 for node_id in ids):
            continue
        if ids[0] != 0:
            ids.insert(0, 0)
        if ids[-1] != 0:
            ids.append(0)
        routes.append(ids)
    return routes


def route_distance_km(routes: list[list[int]], instance: EVRPTWInstance) -> float:
    distance = np.asarray(instance.distance_matrix_km, dtype=float)
    total = 0.0
    for route in routes:
        for i in range(len(route) - 1):
            total += float(distance[int(route[i]), int(route[i + 1])])
    return total


def flatten_routes(routes: list[list[int]]) -> list[int]:
    return merge_route_sequences(routes)
