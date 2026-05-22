from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Optional

import numpy as np

from evrptw_core.schema import EVRPTWInstance, merge_route_sequences


@dataclass(frozen=True)
class Transition:
    path: list[int]
    arrival_time_s: float
    arrival_battery_kwh: float
    cost_s: float


class GreedyEVRPTWSolver:
    """Road-distance-native constructive EVRP-TW-D baseline.

    The solver is deterministic. It repeatedly dispatches a vehicle from the
    depot, greedily serves the nearest customer that is feasible under load,
    time-window, battery, charging, and return-to-depot constraints, then
    starts a new vehicle when no additional customer can be inserted.

    Full-charge semantics match the exact/ALNS benchmark layer: arriving at a
    charging station triggers a full charge before departure.
    """

    name = "greedy_constructive"

    def __init__(self, instance: EVRPTWInstance, customer_order: str = "nearest"):
        if customer_order not in {"nearest", "earliest_due", "hybrid"}:
            raise ValueError(f"Unsupported customer_order: {customer_order}")
        self.instance = instance
        self.customer_order = customer_order
        self.distance = np.asarray(instance.distance_matrix_km, dtype=np.float64)
        self.n_customers = int(instance.num_customers)
        self.n_stations = int(instance.num_charging_stations)
        self.num_nodes = int(instance.num_terminals)
        self.depot = 0
        self.customer_nodes = list(range(1, 1 + self.n_customers))
        self.station_nodes = list(range(1 + self.n_customers, self.num_nodes))
        self.stop_nodes = [self.depot] + self.station_nodes

        self.speed_kmh = float(
            instance.speed_profile.get("effective_speed_kmh")
            or instance.vehicle.get("design_speed_kmh")
            or 40.0
        )
        self.speed_km_per_s = max(self.speed_kmh / 3600.0, 1e-12)
        self.battery_capacity = float(instance.vehicle.get("battery_capacity_kwh", 100.0))
        self.energy_per_km = float(instance.vehicle.get("consumption_kwh_per_km", 0.404))
        self.cargo_capacity = float(instance.vehicle.get("cargo_capacity_cm3", np.inf))
        self.full_charge_time_s = float(instance.vehicle.get("full_charge_time_s", 0.0))
        self.working_start_s = float(instance.working_start_s)
        self.working_end_s = float(instance.working_end_s)

        self.demands = np.asarray(instance.demands_cm3, dtype=np.float64)
        self.service_time_s = np.asarray(instance.service_time_s, dtype=np.float64)
        self.tw_s = np.asarray(instance.tw_s, dtype=np.float64)
        self.stop_adj = self._build_stop_adjacency()

    def travel_time_s(self, i: int, j: int) -> float:
        return float(self.distance[i, j]) / self.speed_km_per_s

    def energy_kwh(self, i: int, j: int) -> float:
        return float(self.distance[i, j]) * self.energy_per_km

    def route_distance_km(self, routes: list[list[int]]) -> float:
        total = 0.0
        for route in routes:
            for idx in range(len(route) - 1):
                total += float(self.distance[route[idx], route[idx + 1]])
        return total

    def _build_stop_adjacency(self) -> dict[int, list[tuple[int, float]]]:
        adjacency: dict[int, list[tuple[int, float]]] = {node: [] for node in self.stop_nodes}
        for i in self.stop_nodes:
            for j in self.stop_nodes:
                if i == j:
                    continue
                if self.energy_kwh(i, j) > self.battery_capacity + 1e-9:
                    continue
                charge_time = self.full_charge_time_s if j in self.station_nodes else 0.0
                adjacency[i].append((j, self.travel_time_s(i, j) + charge_time))
        return adjacency

    def _shortest_stop_path(self, start: int, target: int) -> tuple[list[int], float] | None:
        if start == target:
            return [start], 0.0
        heap: list[tuple[float, int]] = [(0.0, start)]
        dist = {start: 0.0}
        prev: dict[int, int] = {}
        while heap:
            cur_cost, cur = heapq.heappop(heap)
            if cur == target:
                break
            if cur_cost > dist.get(cur, math.inf) + 1e-12:
                continue
            for nxt, edge_cost in self.stop_adj.get(cur, []):
                cand = cur_cost + edge_cost
                if cand + 1e-12 < dist.get(nxt, math.inf):
                    dist[nxt] = cand
                    prev[nxt] = cur
                    heapq.heappush(heap, (cand, nxt))
        if target not in dist:
            return None
        path = [target]
        cur = target
        while cur != start:
            cur = prev[cur]
            path.append(cur)
        path.reverse()
        return path, float(dist[target])

    def _append_path(self, route: list[int], path: list[int]) -> None:
        if not path:
            return
        if not route:
            route.extend(path)
        elif route[-1] == path[0]:
            route.extend(path[1:])
        else:
            route.extend(path)

    def _transition_between(
        self,
        start: int,
        target: int,
        current_time_s: float,
        current_battery_kwh: float,
    ) -> Optional[Transition]:
        direct_energy = self.energy_kwh(start, target)
        direct_time = self.travel_time_s(start, target)
        if direct_energy <= current_battery_kwh + 1e-9:
            return Transition(
                path=[start, target],
                arrival_time_s=current_time_s + direct_time,
                arrival_battery_kwh=current_battery_kwh - direct_energy,
                cost_s=direct_time,
            )

        best: Optional[Transition] = None
        for first_station in self.station_nodes:
            e1 = self.energy_kwh(start, first_station)
            if e1 > current_battery_kwh + 1e-9:
                continue
            t1 = self.travel_time_s(start, first_station) + self.full_charge_time_s

            # first_station may directly serve target after full charge.
            e_last = self.energy_kwh(first_station, target)
            if e_last <= self.battery_capacity + 1e-9:
                cost = t1 + self.travel_time_s(first_station, target)
                candidate = Transition(
                    path=[start, first_station, target],
                    arrival_time_s=current_time_s + cost,
                    arrival_battery_kwh=self.battery_capacity - e_last,
                    cost_s=cost,
                )
                if best is None or candidate.cost_s < best.cost_s - 1e-12:
                    best = candidate

            for last_station in self.station_nodes:
                if last_station == first_station:
                    continue
                stop_plan = self._shortest_stop_path(first_station, last_station)
                if stop_plan is None:
                    continue
                stop_path, stop_cost = stop_plan
                e_last = self.energy_kwh(last_station, target)
                if e_last > self.battery_capacity + 1e-9:
                    continue
                cost = t1 + stop_cost + self.travel_time_s(last_station, target)
                path = [start] + stop_path + [target]
                candidate = Transition(
                    path=path,
                    arrival_time_s=current_time_s + cost,
                    arrival_battery_kwh=self.battery_capacity - e_last,
                    cost_s=cost,
                )
                if best is None or candidate.cost_s < best.cost_s - 1e-12:
                    best = candidate
        return best

    def _return_to_depot(
        self,
        current_node: int,
        current_time_s: float,
        current_battery_kwh: float,
    ) -> Optional[Transition]:
        plan = self._transition_between(current_node, self.depot, current_time_s, current_battery_kwh)
        if plan is None:
            return None
        if plan.arrival_time_s > self.working_end_s + 1e-9:
            return None
        return plan

    def _candidate_to_customer(
        self,
        current_node: int,
        current_time_s: float,
        current_battery_kwh: float,
        current_load: float,
        customer_node: int,
    ) -> Optional[tuple[float, Transition, Transition, float, float]]:
        customer_idx = customer_node - 1
        demand = float(self.demands[customer_idx])
        if current_load + demand > self.cargo_capacity + 1e-9:
            return None

        transition = self._transition_between(current_node, customer_node, current_time_s, current_battery_kwh)
        if transition is None:
            return None

        ready, due = map(float, self.tw_s[customer_idx])
        service_start = max(transition.arrival_time_s, ready)
        if service_start > due + 1e-9:
            return None
        departure_time = service_start + float(self.service_time_s[customer_idx])
        if departure_time > self.working_end_s + 1e-9:
            return None

        return_plan = self._return_to_depot(customer_node, departure_time, transition.arrival_battery_kwh)
        if return_plan is None:
            return None

        score = self._candidate_score(current_node, customer_node, transition, due)
        return score, transition, return_plan, departure_time, current_load + demand

    def _candidate_score(self, current_node: int, customer_node: int, transition: Transition, due: float) -> float:
        if self.customer_order == "nearest":
            return float(self.distance[current_node, customer_node])
        if self.customer_order == "earliest_due":
            return float(due)
        return float(self.distance[current_node, customer_node]) + 1e-4 * float(due)

    def solve(self) -> tuple[list[list[int]], dict[str, int | bool]]:
        unserved = set(self.customer_nodes)
        routes: list[list[int]] = []
        failed_vehicle_starts = 0

        while unserved:
            route = [self.depot]
            current_node = self.depot
            current_time = self.working_start_s
            current_battery = self.battery_capacity
            current_load = 0.0
            visited_this_vehicle = 0
            cached_return: Optional[Transition] = None

            while unserved:
                best = None
                for customer_node in sorted(unserved):
                    candidate = self._candidate_to_customer(
                        current_node,
                        current_time,
                        current_battery,
                        current_load,
                        customer_node,
                    )
                    if candidate is None:
                        continue
                    if best is None or candidate[0] < best[0][0] - 1e-12:
                        best = (candidate, customer_node)

                if best is None:
                    break

                (score, transition, return_plan, departure_time, new_load), customer_node = best
                self._append_path(route, transition.path)
                unserved.remove(customer_node)
                current_node = customer_node
                current_time = departure_time
                current_battery = transition.arrival_battery_kwh
                current_load = new_load
                cached_return = return_plan
                visited_this_vehicle += 1

            if current_node != self.depot:
                return_plan = cached_return or self._return_to_depot(current_node, current_time, current_battery)
                if return_plan is None:
                    break
                self._append_path(route, return_plan.path)

            if visited_this_vehicle == 0:
                failed_vehicle_starts += 1
                break

            routes.append(route)

        visited_all = not unserved
        return routes, {
            "visited_all": bool(visited_all),
            "unvisited_count": int(len(unserved)),
            "failed_vehicle_starts": int(failed_vehicle_starts),
        }


def flatten_routes(routes: list[list[int]]) -> list[int]:
    return merge_route_sequences(routes)
