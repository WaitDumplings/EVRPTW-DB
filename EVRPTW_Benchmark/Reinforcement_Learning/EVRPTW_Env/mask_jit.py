from __future__ import annotations

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without numba installed.
    njit = None
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _charge_time_s_jit(
        battery_used_kwh: float,
        battery_capacity_kwh: float,
        full_charge_time_s: float,
        station_node: int,
        station_start: int,
        charging_power_kw: np.ndarray,
        charging_power_derating_factor: float,
        station_power_full: bool,
        legacy_fixed_full: bool,
    ) -> float:
        if legacy_fixed_full:
            return full_charge_time_s
        if station_power_full:
            station_offset = station_node - station_start
            usable_power_kw = (
                charging_power_kw[station_offset]
                * charging_power_derating_factor
            )
            energy_added_kwh = battery_used_kwh
            if energy_added_kwh < 0.0:
                energy_added_kwh = 0.0
            elif energy_added_kwh > battery_capacity_kwh:
                energy_added_kwh = battery_capacity_kwh
            return 3600.0 * energy_added_kwh / usable_power_kw
        ratio = battery_used_kwh / max(battery_capacity_kwh, 1e-12)
        if ratio < 0.0:
            ratio = 0.0
        elif ratio > 1.0:
            ratio = 1.0
        return ratio * full_charge_time_s


    @njit(cache=True)
    def _direct_depot_feasible_jit(
        start: int,
        current_time_s: float,
        battery_used_kwh: float,
        travel_time_s: np.ndarray,
        energy_kwh: np.ndarray,
        battery_capacity_kwh: float,
        working_end_s: float,
    ) -> bool:
        battery_after = battery_used_kwh + energy_kwh[start, 0]
        arrival = current_time_s + travel_time_s[start, 0]
        return battery_after <= battery_capacity_kwh + 1e-9 and arrival <= working_end_s + 1e-9


    @njit(cache=True)
    def _route_local_stop_to_depot_times_jit(
        station_start: int,
        num_nodes: int,
        unavailable_stations: np.ndarray,
        travel_time_s: np.ndarray,
        energy_kwh: np.ndarray,
        battery_capacity_kwh: float,
        full_charge_time_s: float,
        charging_power_kw: np.ndarray,
        charging_power_derating_factor: float,
        station_power_full: bool,
        legacy_fixed_full: bool,
        allow_consecutive_station_actions: bool,
    ) -> np.ndarray:
        """Policy-consistent return times after removing route-used stations."""

        distance = np.full(num_nodes, np.inf, dtype=np.float64)
        settled = np.zeros(num_nodes, dtype=np.bool_)
        distance[0] = 0.0
        if not allow_consecutive_station_actions:
            for station in range(station_start, num_nodes):
                if unavailable_stations[station]:
                    continue
                if energy_kwh[station, 0] <= battery_capacity_kwh + 1e-9:
                    distance[station] = travel_time_s[station, 0]
            return distance

        # The stop graph contains only the depot and charging stations.  A
        # dense O(M^2) Dijkstra is quicker here than allocating a heap in every
        # trajectory and remains small for the benchmark's M=50 stations.
        for _ in range(num_nodes - station_start + 1):
            successor = -1
            best = np.inf
            if not settled[0]:
                successor = 0
                best = distance[0]
            for station in range(station_start, num_nodes):
                if unavailable_stations[station] or settled[station]:
                    continue
                if distance[station] < best:
                    successor = station
                    best = distance[station]
            if successor < 0 or not np.isfinite(best):
                break
            settled[successor] = True

            # Relax incoming full-battery station -> successor edges.  Charging
            # at the successor is part of that edge unless it is the depot.
            for predecessor in range(station_start, num_nodes):
                if (
                    unavailable_stations[predecessor]
                    or settled[predecessor]
                    or predecessor == successor
                ):
                    continue
                leg_energy = energy_kwh[predecessor, successor]
                if leg_energy > battery_capacity_kwh + 1e-9:
                    continue
                edge_time = travel_time_s[predecessor, successor]
                if successor != 0:
                    edge_time += _charge_time_s_jit(
                        leg_energy,
                        battery_capacity_kwh,
                        full_charge_time_s,
                        successor,
                        station_start,
                        charging_power_kw,
                        charging_power_derating_factor,
                        station_power_full,
                        legacy_fixed_full,
                    )
                candidate = best + edge_time
                if candidate + 1e-12 < distance[predecessor]:
                    distance[predecessor] = candidate
        return distance


    @njit(cache=True)
    def _refresh_route_return_times_jit(
        route_return_time_s: np.ndarray,
        dirty_trajectories: np.ndarray,
        station_start: int,
        num_nodes: int,
        cs_visited_current_route: np.ndarray,
        travel_time_s: np.ndarray,
        energy_kwh: np.ndarray,
        battery_capacity_kwh: float,
        full_charge_time_s: float,
        charging_power_kw: np.ndarray,
        charging_power_derating_factor: float,
        station_power_full: bool,
        legacy_fixed_full: bool,
        allow_consecutive_station_actions: bool,
    ) -> None:
        for trajectory in range(cs_visited_current_route.shape[0]):
            if not dirty_trajectories[trajectory]:
                continue
            route_return_time_s[trajectory] = (
                _route_local_stop_to_depot_times_jit(
                    station_start,
                    num_nodes,
                    cs_visited_current_route[trajectory],
                    travel_time_s,
                    energy_kwh,
                    battery_capacity_kwh,
                    full_charge_time_s,
                    charging_power_kw,
                    charging_power_derating_factor,
                    station_power_full,
                    legacy_fixed_full,
                    allow_consecutive_station_actions,
                )
            )


    @njit(cache=True)
    def _can_return_to_depot_jit(
        start: int,
        current_time_s: float,
        battery_used_kwh: float,
        station_start: int,
        num_nodes: int,
        travel_time_s: np.ndarray,
        energy_kwh: np.ndarray,
        route_return_time_s: np.ndarray,
        battery_capacity_kwh: float,
        full_charge_time_s: float,
        charging_power_kw: np.ndarray,
        charging_power_derating_factor: float,
        working_end_s: float,
        station_power_full: bool,
        legacy_fixed_full: bool,
    ) -> bool:
        if start == 0:
            return True
        if battery_used_kwh + energy_kwh[start, 0] <= battery_capacity_kwh + 1e-9:
            return current_time_s + travel_time_s[start, 0] <= working_end_s + 1e-9

        for first in range(station_start, num_nodes):
            if first == start or not np.isfinite(route_return_time_s[first]):
                continue
            battery_at_first = battery_used_kwh + energy_kwh[start, first]
            if battery_at_first > battery_capacity_kwh + 1e-9:
                continue
            time_at_first = current_time_s + travel_time_s[start, first]
            depart_first = time_at_first + _charge_time_s_jit(
                battery_at_first,
                battery_capacity_kwh,
                full_charge_time_s,
                first,
                station_start,
                charging_power_kw,
                charging_power_derating_factor,
                station_power_full,
                legacy_fixed_full,
            )
            if depart_first + route_return_time_s[first] <= working_end_s + 1e-9:
                return True
        return False


    @njit(cache=True)
    def _compute_action_mask_jit(
        n_traj: int,
        num_nodes: int,
        num_customers: int,
        station_start: int,
        last: np.ndarray,
        visited: np.ndarray,
        cs_visited_current_route: np.ndarray,
        route_return_time_s: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        served_customers: np.ndarray,
        route_has_customer: np.ndarray,
        current_time_s: np.ndarray,
        battery_used_kwh: np.ndarray,
        load_cm3: np.ndarray,
        demand_cm3: np.ndarray,
        service_time_s: np.ndarray,
        tw_s: np.ndarray,
        travel_time_s: np.ndarray,
        energy_kwh: np.ndarray,
        battery_capacity_kwh: float,
        cargo_capacity_cm3: float,
        full_charge_time_s: float,
        charging_power_kw: np.ndarray,
        charging_power_derating_factor: float,
        working_end_s: float,
        station_power_full: bool,
        legacy_fixed_full: bool,
        allow_consecutive_station_actions: bool,
    ) -> np.ndarray:
        mask = np.zeros((n_traj, num_nodes), dtype=np.bool_)
        for t in range(n_traj):
            if terminated[t] or truncated[t]:
                mask[t, 0] = True
                continue

            start = int(last[t])
            all_served = served_customers[t] == num_customers
            trajectory_return_time_s = route_return_time_s[t]
            if all_served:
                if start == 0 or _direct_depot_feasible_jit(
                    start,
                    current_time_s[t],
                    battery_used_kwh[t],
                    travel_time_s,
                    energy_kwh,
                    battery_capacity_kwh,
                    working_end_s,
                ):
                    mask[t, 0] = True
                # Fall through so a charging-assisted return remains available.

            if (
                start != 0
                and route_has_customer[t]
                and _direct_depot_feasible_jit(
                    start,
                    current_time_s[t],
                    battery_used_kwh[t],
                    travel_time_s,
                    energy_kwh,
                    battery_capacity_kwh,
                    working_end_s,
                )
            ):
                mask[t, 0] = True

            for customer in range(1, 1 + num_customers):
                if visited[t, customer]:
                    continue
                battery_after = battery_used_kwh[t] + energy_kwh[start, customer]
                if battery_after > battery_capacity_kwh + 1e-9:
                    continue
                if load_cm3[t] + demand_cm3[customer] > cargo_capacity_cm3 + 1e-9:
                    continue
                arrival = current_time_s[t] + travel_time_s[start, customer]
                ready = tw_s[customer, 0]
                due = tw_s[customer, 1]
                service_start = arrival
                service_start = max(service_start, ready)
                service_departure = service_start + service_time_s[customer]
                if service_start > due + 1e-9 or service_departure > working_end_s + 1e-9:
                    continue
                if _can_return_to_depot_jit(
                    customer,
                    service_departure,
                    battery_after,
                    station_start,
                    num_nodes,
                    travel_time_s,
                    energy_kwh,
                    trajectory_return_time_s,
                    battery_capacity_kwh,
                    full_charge_time_s,
                    charging_power_kw,
                    charging_power_derating_factor,
                    working_end_s,
                    station_power_full,
                    legacy_fixed_full,
                ):
                    mask[t, customer] = True

            if allow_consecutive_station_actions or start < station_start:
                for station in range(station_start, num_nodes):
                    if station == start or cs_visited_current_route[t, station]:
                        continue
                    battery_after = battery_used_kwh[t] + energy_kwh[start, station]
                    if battery_after > battery_capacity_kwh + 1e-9:
                        continue
                    arrival = current_time_s[t] + travel_time_s[start, station]
                    departure = arrival + _charge_time_s_jit(
                        battery_after,
                        battery_capacity_kwh,
                        full_charge_time_s,
                        station,
                        station_start,
                        charging_power_kw,
                        charging_power_derating_factor,
                        station_power_full,
                        legacy_fixed_full,
                    )
                    if departure > working_end_s + 1e-9:
                        continue
                    if (
                        np.isfinite(trajectory_return_time_s[station])
                        and departure + trajectory_return_time_s[station]
                        <= working_end_s + 1e-9
                    ):
                        mask[t, station] = True

            if (
                not np.any(mask[t])
                and start != 0
                and _direct_depot_feasible_jit(
                    start,
                    current_time_s[t],
                    battery_used_kwh[t],
                    travel_time_s,
                    energy_kwh,
                    battery_capacity_kwh,
                    working_end_s,
                )
            ):
                mask[t, 0] = True
        return mask


def compute_action_mask_jit(
    *,
    n_traj: int,
    num_nodes: int,
    num_customers: int,
    station_start: int,
    last: np.ndarray,
    visited: np.ndarray,
    cs_visited_current_route: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    served_customers: np.ndarray,
    route_has_customer: np.ndarray,
    current_time_s: np.ndarray,
    battery_used_kwh: np.ndarray,
    load_cm3: np.ndarray,
    demand_cm3: np.ndarray,
    service_time_s: np.ndarray,
    tw_s: np.ndarray,
    travel_time_s: np.ndarray,
    energy_kwh: np.ndarray,
    battery_capacity_kwh: float,
    cargo_capacity_cm3: float,
    full_charge_time_s: float,
    charging_power_kw: np.ndarray,
    charging_power_derating_factor: float,
    working_end_s: float,
    station_power_full: bool,
    legacy_fixed_full: bool,
    allow_consecutive_station_actions: bool = True,
    route_return_time_s: np.ndarray | None = None,
    # Retained as ignored keyword-only compatibility inputs for callers of the
    # former static-witness API.  Route-local witnesses are always recomputed
    # when ``route_return_time_s`` is not supplied.
    stop_to_depot_time_s: np.ndarray | None = None,
) -> np.ndarray:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    del stop_to_depot_time_s
    if route_return_time_s is None:
        route_return_time_s = np.full(
            (int(n_traj), int(num_nodes)), np.inf, dtype=np.float64
        )
        dirty = np.ones(int(n_traj), dtype=bool)
        _refresh_route_return_times_jit(
            route_return_time_s,
            dirty,
            int(station_start),
            int(num_nodes),
            cs_visited_current_route,
            travel_time_s,
            energy_kwh,
            float(battery_capacity_kwh),
            float(full_charge_time_s),
            charging_power_kw,
            float(charging_power_derating_factor),
            bool(station_power_full),
            bool(legacy_fixed_full),
            bool(allow_consecutive_station_actions),
        )
    return _compute_action_mask_jit(
        int(n_traj),
        int(num_nodes),
        int(num_customers),
        int(station_start),
        last,
        visited,
        cs_visited_current_route,
        route_return_time_s,
        terminated,
        truncated,
        served_customers,
        route_has_customer,
        current_time_s,
        battery_used_kwh,
        load_cm3,
        demand_cm3,
        service_time_s,
        tw_s,
        travel_time_s,
        energy_kwh,
        float(battery_capacity_kwh),
        float(cargo_capacity_cm3),
        float(full_charge_time_s),
        charging_power_kw,
        float(charging_power_derating_factor),
        float(working_end_s),
        bool(station_power_full),
        bool(legacy_fixed_full),
        bool(allow_consecutive_station_actions),
    )


def refresh_route_return_times_jit(
    *,
    route_return_time_s: np.ndarray,
    dirty_trajectories: np.ndarray,
    station_start: int,
    num_nodes: int,
    cs_visited_current_route: np.ndarray,
    travel_time_s: np.ndarray,
    energy_kwh: np.ndarray,
    battery_capacity_kwh: float,
    full_charge_time_s: float,
    charging_power_kw: np.ndarray,
    charging_power_derating_factor: float,
    station_power_full: bool,
    legacy_fixed_full: bool,
    allow_consecutive_station_actions: bool = True,
) -> None:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available")
    _refresh_route_return_times_jit(
        route_return_time_s,
        dirty_trajectories,
        int(station_start),
        int(num_nodes),
        cs_visited_current_route,
        travel_time_s,
        energy_kwh,
        float(battery_capacity_kwh),
        float(full_charge_time_s),
        charging_power_kw,
        float(charging_power_derating_factor),
        bool(station_power_full),
        bool(legacy_fixed_full),
        bool(allow_consecutive_station_actions),
    )


__all__ = [
    "NUMBA_AVAILABLE",
    "compute_action_mask_jit",
    "refresh_route_return_times_jit",
]
