from __future__ import annotations

import heapq
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from gymnasium import Env, spaces

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from evrptw_core.schema import EVRPTWInstance, merge_route_sequences


@dataclass(frozen=True)
class Transition:
    path: list[int]
    travel_time_s: float
    distance_km: float
    battery_used_after_kwh: float


class EVRPTWVectorEnv(Env):
    """Gymnasium-style vectorized EVRP-TW-D environment.

    The environment batches multiple trajectories for one canonical instance,
    which matches POMO-style rollout requirements. It intentionally returns
    vector-valued ``reward``, ``terminated``, and ``truncated`` arrays with shape
    ``(n_traj,)``.

    Node convention:
    - 0: depot
    - 1..N: customers
    - N+1..N+M: active charging stations

    Battery state follows the legacy DRL convention: ``current_battery`` is the
    fraction of battery already consumed since the last full charge. A station
    action recharges to full and resets battery used to zero.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        instance: EVRPTWInstance | None = None,
        n_traj: int = 1,
        reward_mode: str = "distance",
        invalid_action_penalty: float = -10.0,
        success_bonus: float = 0.0,
        max_steps_factor: int = 4,
        charging_mode: str = "station_power_full",
        matrix_mode: str = "canonical",
        normalize_reward: bool = True,
        reward_distance_scale_km: float | None = None,
        reward_distance_scale_mode: str = "single_customer_repair_median",
    ) -> None:
        super().__init__()
        if reward_mode not in {"distance", "distance_success"}:
            raise ValueError(f"Unsupported reward_mode: {reward_mode}")
        if charging_mode not in {"station_power_full", "legacy_proportional_full", "legacy_fixed_full"}:
            raise ValueError(f"Unsupported charging_mode: {charging_mode}")
        if matrix_mode not in {"canonical", "legacy_derived"}:
            raise ValueError(f"Unsupported matrix_mode: {matrix_mode}")

        self.instance: EVRPTWInstance | None = None
        self.n_traj = int(n_traj)
        self.reward_mode = reward_mode
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.success_bonus = float(success_bonus)
        self.max_steps_factor = int(max_steps_factor)
        self.charging_mode = charging_mode
        self.matrix_mode = matrix_mode
        self.normalize_reward = bool(normalize_reward)
        self.reward_distance_scale_km_override = reward_distance_scale_km
        self.reward_distance_scale_mode = str(reward_distance_scale_mode)
        valid_scale_modes = {
            "max_edge",
            "single_customer_repair_sum",
            "single_customer_repair_mean",
            "single_customer_repair_median",
        }
        if self.reward_distance_scale_mode not in valid_scale_modes:
            raise ValueError(f"reward_distance_scale_mode must be one of {sorted(valid_scale_modes)}")
        self.reward_distance_scale_km = 1.0

        self._rng = np.random.default_rng()
        if instance is not None:
            self.set_instance(instance)
        else:
            self.observation_space = spaces.Dict({})
            self.action_space = spaces.MultiDiscrete([1] * self.n_traj)

    def set_instance(self, instance: EVRPTWInstance) -> None:
        self.instance = instance
        self.num_customers = int(instance.num_customers)
        self.num_stations = int(instance.num_charging_stations)
        self.num_nodes = int(instance.num_terminals)
        self.depot = 0
        self.customer_start = 1
        self.station_start = 1 + self.num_customers
        self.customer_nodes = np.arange(1, 1 + self.num_customers, dtype=np.int32)
        self.station_nodes = np.arange(self.station_start, self.num_nodes, dtype=np.int32)
        self.stop_nodes = [0] + [int(x) for x in self.station_nodes]

        self.distance_km = np.asarray(instance.distance_matrix_km, dtype=np.float64)
        self.reward_distance_scale_km = self._compute_reward_distance_scale_km()
        self.coords_raw = np.vstack(
            [
                np.asarray(instance.depot, dtype=np.float64).reshape(1, 2),
                np.asarray(instance.customers, dtype=np.float64),
                np.asarray(instance.charging_stations, dtype=np.float64),
            ]
        )

        self.battery_capacity_kwh = float(instance.vehicle.get("battery_capacity_kwh", 100.0))
        self.cargo_capacity_cm3 = float(instance.vehicle.get("cargo_capacity_cm3", np.inf))
        self.full_charge_time_s = float(instance.vehicle.get("full_charge_time_s", 0.0))
        self.travel_time_s, self.travel_time_source = self._resolve_travel_time_matrix(instance)
        self.energy_kwh, self.energy_source = self._resolve_energy_matrix(instance)
        self.charging_power_kw, self.charging_power_source = self._resolve_charging_power(instance)
        charging_policy = instance.raw.get("charging_policy", {})
        self.charging_power_derating_factor = float(
            charging_policy.get(
                "charging_power_derating_factor",
                instance.vehicle.get("charging_power_derating_factor", 1.0),
            )
        )
        if not 0.0 < self.charging_power_derating_factor <= 1.0:
            raise ValueError("charging_power_derating_factor must be in (0, 1]")
        self.working_start_s = float(instance.working_start_s)
        self.working_end_s = float(instance.working_end_s)
        self.horizon_s = max(float(self.working_end_s - self.working_start_s), 1.0)

        demand_c = np.asarray(instance.demands_cm3, dtype=np.float64)
        service_c = np.asarray(instance.service_time_s, dtype=np.float64)
        tw_c = np.asarray(instance.tw_s, dtype=np.float64)
        self.demand_cm3 = np.concatenate([np.zeros(1), demand_c, np.zeros(self.num_stations)])
        self.service_time_s = np.concatenate([np.zeros(1), service_c, np.zeros(self.num_stations)])
        self.tw_s = np.vstack(
            [
                np.array([[self.working_start_s, self.working_end_s]], dtype=np.float64),
                tw_c,
                np.tile(
                    np.array([[self.working_start_s, self.working_end_s]], dtype=np.float64),
                    (self.num_stations, 1),
                ),
            ]
        )

        self.stop_adj = self._build_stop_adjacency()
        self.max_steps = max(1, self.max_steps_factor * self.num_nodes)
        self._build_spaces()

    def _compute_reward_distance_scale_km(self) -> float:
        if self.reward_distance_scale_km_override is not None:
            return max(float(self.reward_distance_scale_km_override), 1e-9)
        finite_dist = self.distance_km[np.isfinite(self.distance_km)]
        if finite_dist.size == 0:
            return 1.0
        if self.reward_distance_scale_mode == "max_edge":
            return max(float(finite_dist.max()), 1e-9)
        customer_nodes = np.arange(1, 1 + self.num_customers, dtype=np.int32)
        if customer_nodes.size == 0:
            return max(float(finite_dist.max()), 1e-9)
        repair = self.distance_km[0, customer_nodes] + self.distance_km[customer_nodes, 0]
        repair = repair[np.isfinite(repair)]
        if repair.size == 0:
            return max(float(finite_dist.max()), 1e-9)
        if self.reward_distance_scale_mode == "single_customer_repair_mean":
            return max(float(repair.mean()), 1e-9)
        if self.reward_distance_scale_mode == "single_customer_repair_median":
            return max(float(np.median(repair)), 1e-9)
        return max(float(repair.sum()), 1e-9)

    def _build_spaces(self) -> None:
        n = self.num_nodes
        obs_dict = {
            "cus_loc": spaces.Box(0.0, 1.0, shape=(self.num_customers, 2), dtype=np.float32),
            "depot_loc": spaces.Box(0.0, 1.0, shape=(1, 2), dtype=np.float32),
            "rs_loc": spaces.Box(0.0, 1.0, shape=(self.num_stations, 2), dtype=np.float32),
            "demand": spaces.Box(0.0, np.inf, shape=(n,), dtype=np.float32),
            "time_window": spaces.Box(0.0, np.inf, shape=(n, 2), dtype=np.float32),
            "service_time": spaces.Box(0.0, np.inf, shape=(n,), dtype=np.float32),
            "charging_power": spaces.Box(0.0, np.inf, shape=(n,), dtype=np.float32),
            "remaining_demand": spaces.Box(
                0.0,
                np.inf,
                shape=(self.n_traj, n),
                dtype=np.float32,
            ),
            "action_mask": spaces.MultiBinary([self.n_traj, n]),
            "last_node_idx": spaces.MultiDiscrete([n] * self.n_traj),
            "current_load": spaces.Box(0.0, np.inf, shape=(self.n_traj,), dtype=np.float32),
            "current_battery": spaces.Box(0.0, np.inf, shape=(self.n_traj,), dtype=np.float32),
            # Canonical rollouts never expose a negative value.  The open
            # lower bound supports the DRL-TS Stage-1 training wrapper, which
            # deliberately measures and penalizes temporary energy deficits.
            "remaining_battery": spaces.Box(
                -np.inf,
                np.inf,
                shape=(self.n_traj,),
                dtype=np.float32,
            ),
            "current_time": spaces.Box(0.0, np.inf, shape=(self.n_traj,), dtype=np.float32),
            "remaining_vehicle_ratio": spaces.Box(
                0.0,
                1.0,
                shape=(self.n_traj,),
                dtype=np.float32,
            ),
            "visited_customers_ratio": spaces.Box(0.0, 1.0, shape=(self.n_traj, 1), dtype=np.float32),
            "visited_customers_raio": spaces.Box(0.0, 1.0, shape=(self.n_traj, 1), dtype=np.float32),
            "remain_feasible_customers_ratio": spaces.Box(0.0, 1.0, shape=(self.n_traj, 1), dtype=np.float32),
            "remain_feasible_customers_raio": spaces.Box(0.0, 1.0, shape=(self.n_traj, 1), dtype=np.float32),
            "battery_capacity": spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
            "loading_capacity": spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
        }
        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.MultiDiscrete([n] * self.n_traj)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if options:
            if "instance" in options:
                self.set_instance(options["instance"])
            if "n_traj" in options and int(options["n_traj"]) != self.n_traj:
                self.n_traj = int(options["n_traj"])
                if self.instance is None:
                    raise ValueError("Cannot update n_traj before setting an instance.")
                self._build_spaces()
        if self.instance is None:
            raise ValueError("EVRPTWVectorEnv requires an EVRPTWInstance before reset.")

        self.step_count = 0
        self.last = np.zeros(self.n_traj, dtype=np.int32)
        self.current_time_s = np.full(self.n_traj, self.working_start_s, dtype=np.float64)
        self.load_cm3 = np.zeros(self.n_traj, dtype=np.float64)
        self.battery_used_kwh = np.zeros(self.n_traj, dtype=np.float64)
        self.visited = np.zeros((self.n_traj, self.num_nodes), dtype=bool)
        self.visited[:, 0] = True
        self.served_customers = np.zeros(self.n_traj, dtype=np.int32)
        self.objective_distance_km = np.zeros(self.n_traj, dtype=np.float64)
        self.vehicle_count = np.zeros(self.n_traj, dtype=np.int32)
        self.route_has_customer = np.zeros(self.n_traj, dtype=bool)
        self.terminated = np.zeros(self.n_traj, dtype=bool)
        self.truncated = np.zeros(self.n_traj, dtype=bool)
        self.invalid_action = np.zeros(self.n_traj, dtype=bool)
        self.routes: list[list[list[int]]] = [[] for _ in range(self.n_traj)]
        self.current_routes: list[list[int]] = [[0] for _ in range(self.n_traj)]

        obs = self._make_observation()
        info = self._make_info(obs["action_mask"])
        return obs, info

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.int64).reshape(self.n_traj)
        mask_before = self._compute_action_mask()
        reward = np.zeros(self.n_traj, dtype=np.float32)
        self.invalid_action.fill(False)

        for t in range(self.n_traj):
            if self.terminated[t] or self.truncated[t]:
                continue
            destination = int(action_arr[t])
            if destination < 0 or destination >= self.num_nodes or not mask_before[t, destination]:
                self.invalid_action[t] = True
                self.truncated[t] = True
                reward[t] += self.invalid_action_penalty
                continue
            reward[t] += self._apply_action(t, destination)

        self.step_count += 1
        if self.step_count >= self.max_steps:
            unfinished = ~self.terminated
            self.truncated[unfinished] = True

        obs = self._make_observation()
        action_mask = obs["action_mask"]
        no_action = (~action_mask.any(axis=1)) & (~self.terminated) & (~self.truncated)
        if np.any(no_action):
            self.truncated[no_action] = True
            reward[no_action] += self.invalid_action_penalty
            obs = self._make_observation()
            action_mask = obs["action_mask"]

        info = self._make_info(action_mask)
        return obs, reward, self.terminated.copy(), self.truncated.copy(), info

    def _apply_action(self, traj_idx: int, destination: int) -> float:
        start = int(self.last[traj_idx])
        dist = float(self.distance_km[start, destination])
        energy = float(self.energy_kwh[start, destination])
        travel_time = float(self.travel_time_s[start, destination])
        self.objective_distance_km[traj_idx] += dist
        self.current_time_s[traj_idx] += travel_time
        self.battery_used_kwh[traj_idx] += energy
        reward = -dist / self.reward_distance_scale_km if self.normalize_reward else -dist

        self._append_route_node(traj_idx, destination)

        if self._is_customer(destination):
            ready, _ = self.tw_s[destination]
            self.current_time_s[traj_idx] = max(self.current_time_s[traj_idx], float(ready))
            self.current_time_s[traj_idx] += float(self.service_time_s[destination])
            self.load_cm3[traj_idx] += float(self.demand_cm3[destination])
            self.visited[traj_idx, destination] = True
            self.served_customers[traj_idx] += 1
            self.route_has_customer[traj_idx] = True
        elif self._is_station(destination):
            self.current_time_s[traj_idx] += self._charge_time_s(
                self.battery_used_kwh[traj_idx], destination
            )
            self.battery_used_kwh[traj_idx] = 0.0
            self.visited[traj_idx, destination] = True
        elif destination == 0:
            self._close_route_at_depot(traj_idx)

        self.last[traj_idx] = int(destination)

        if destination == 0 and self.served_customers[traj_idx] == self.num_customers:
            self.terminated[traj_idx] = True
            if self.reward_mode == "distance_success":
                reward += self.success_bonus
        return float(reward)

    def _append_route_node(self, traj_idx: int, node: int) -> None:
        route = self.current_routes[traj_idx]
        if not route:
            route.append(0)
        if route[-1] != node:
            route.append(int(node))

    def _close_route_at_depot(self, traj_idx: int) -> None:
        if self.route_has_customer[traj_idx]:
            route = self.current_routes[traj_idx]
            if route[-1] != 0:
                route.append(0)
            self.routes[traj_idx].append(route)
            self.vehicle_count[traj_idx] += 1
        self.current_routes[traj_idx] = [0]
        self.route_has_customer[traj_idx] = False
        self.current_time_s[traj_idx] = self.working_start_s
        self.load_cm3[traj_idx] = 0.0
        self.battery_used_kwh[traj_idx] = 0.0

    def _compute_action_mask(self) -> np.ndarray:
        mask = np.zeros((self.n_traj, self.num_nodes), dtype=bool)
        for t in range(self.n_traj):
            if self.terminated[t] or self.truncated[t]:
                mask[t, 0] = True
                continue
            start = int(self.last[t])
            all_served = self.served_customers[t] == self.num_customers

            if all_served:
                if start == 0 or self._direct_depot_feasible(t):
                    mask[t, 0] = True
                continue

            if start != 0 and self.route_has_customer[t] and self._direct_depot_feasible(t):
                mask[t, 0] = True

            for customer in self.customer_nodes:
                c = int(customer)
                if self.visited[t, c]:
                    continue
                if self._customer_action_feasible(t, c):
                    mask[t, c] = True

            for station in self.station_nodes:
                s = int(station)
                if s == start:
                    continue
                if self._station_action_feasible(t, s):
                    mask[t, s] = True
        return mask

    def _customer_action_feasible(self, traj_idx: int, customer: int) -> bool:
        start = int(self.last[traj_idx])
        energy = float(self.energy_kwh[start, customer])
        battery_after = float(self.battery_used_kwh[traj_idx] + energy)
        if battery_after > self.battery_capacity_kwh + 1e-9:
            return False
        if self.load_cm3[traj_idx] + self.demand_cm3[customer] > self.cargo_capacity_cm3 + 1e-9:
            return False
        arrival = float(self.current_time_s[traj_idx] + self.travel_time_s[start, customer])
        ready, due = self.tw_s[customer]
        service_start = max(arrival, float(ready))
        service_departure = service_start + float(self.service_time_s[customer])
        if service_start > float(due) + 1e-9 or service_departure > self.working_end_s + 1e-9:
            return False
        return self._can_return_to_depot(customer, service_departure, battery_after)

    def _station_action_feasible(self, traj_idx: int, station: int) -> bool:
        start = int(self.last[traj_idx])
        battery_after = float(self.battery_used_kwh[traj_idx] + self.energy_kwh[start, station])
        if battery_after > self.battery_capacity_kwh + 1e-9:
            return False
        arrival = float(self.current_time_s[traj_idx] + self.travel_time_s[start, station])
        departure = arrival + self._charge_time_s(battery_after, station)
        if departure > self.working_end_s + 1e-9:
            return False
        return self._can_return_to_depot(station, departure, 0.0)

    def _direct_depot_feasible(self, traj_idx: int) -> bool:
        start = int(self.last[traj_idx])
        battery_after = float(self.battery_used_kwh[traj_idx] + self.energy_kwh[start, 0])
        arrival = float(self.current_time_s[traj_idx] + self.travel_time_s[start, 0])
        return battery_after <= self.battery_capacity_kwh + 1e-9 and arrival <= self.working_end_s + 1e-9

    def _can_return_to_depot(self, start: int, current_time_s: float, battery_used_kwh: float) -> bool:
        if start == 0:
            return True
        if battery_used_kwh + self.energy_kwh[start, 0] <= self.battery_capacity_kwh + 1e-9:
            return current_time_s + self.travel_time_s[start, 0] <= self.working_end_s + 1e-9
        for first_station in self.station_nodes:
            first = int(first_station)
            battery_at_first = battery_used_kwh + self.energy_kwh[start, first]
            if battery_at_first > self.battery_capacity_kwh + 1e-9:
                continue
            time_at_first = current_time_s + self.travel_time_s[start, first]
            depart_first = time_at_first + self._charge_time_s(battery_at_first, first)
            stop_plan = self._shortest_stop_time(first, 0)
            if stop_plan is None:
                continue
            if depart_first + stop_plan <= self.working_end_s + 1e-9:
                return True
        return False

    def _build_stop_adjacency(self) -> dict[int, list[tuple[int, float]]]:
        adjacency: dict[int, list[tuple[int, float]]] = {node: [] for node in self.stop_nodes}
        for i in self.stop_nodes:
            for j in self.stop_nodes:
                if i == j:
                    continue
                energy = float(self.energy_kwh[i, j])
                if energy > self.battery_capacity_kwh + 1e-9:
                    continue
                charge_time = self._charge_time_s(energy, j) if self._is_station(j) else 0.0
                adjacency[i].append((j, float(self.travel_time_s[i, j]) + charge_time))
        return adjacency

    def _shortest_stop_time(self, start: int, target: int) -> float | None:
        heap: list[tuple[float, int]] = [(0.0, int(start))]
        dist = {int(start): 0.0}
        while heap:
            cost, node = heapq.heappop(heap)
            if node == target:
                return float(cost)
            if cost > dist.get(node, math.inf) + 1e-12:
                continue
            for nxt, edge_cost in self.stop_adj.get(node, []):
                cand = cost + edge_cost
                if cand + 1e-12 < dist.get(nxt, math.inf):
                    dist[nxt] = cand
                    heapq.heappush(heap, (cand, nxt))
        return None

    def _charge_time_s(self, battery_used_kwh: float, station_node: int) -> float:
        if self.charging_mode == "legacy_fixed_full":
            return self.full_charge_time_s
        if self.charging_mode == "legacy_proportional_full":
            used_ratio = max(
                0.0,
                min(
                    float(battery_used_kwh)
                    / max(self.battery_capacity_kwh, 1e-12),
                    1.0,
                ),
            )
            return used_ratio * self.full_charge_time_s
        if not self._is_station(station_node):
            raise ValueError(f"charging target is not a station: {station_node}")
        station_offset = int(station_node) - self.station_start
        usable_power_kw = (
            float(self.charging_power_kw[station_offset])
            * self.charging_power_derating_factor
        )
        if usable_power_kw <= 0.0 or not math.isfinite(usable_power_kw):
            raise ValueError("station charging power must be finite and positive")
        energy_added_kwh = max(
            0.0,
            min(float(battery_used_kwh), self.battery_capacity_kwh),
        )
        return 3600.0 * energy_added_kwh / usable_power_kw

    def _make_observation(self) -> dict[str, np.ndarray]:
        action_mask = self._compute_action_mask()
        feasible_customer_count = action_mask[:, self.customer_start:self.station_start].sum(axis=1, keepdims=True)
        visited_ratio = (self.served_customers.astype(np.float32) / max(float(self.num_customers), 1.0))[:, None]
        remain_feasible_ratio = (feasible_customer_count.astype(np.float32) / max(float(self.num_customers), 1.0))

        coords = self._normalized_coords()
        demand_norm = (self.demand_cm3 / max(self.cargo_capacity_cm3, 1e-12)).astype(np.float32)
        tw_norm = ((self.tw_s - self.working_start_s) / self.horizon_s).astype(np.float32)
        service_norm = (self.service_time_s / self.horizon_s).astype(np.float32)
        charging_power = np.zeros(self.num_nodes, dtype=np.float32)
        if self.num_stations:
            power_scale = max(float(np.max(self.charging_power_kw)), 1e-12)
            charging_power[self.station_start :] = (
                self.charging_power_kw / power_scale
            ).astype(np.float32)
        current_battery = (self.battery_used_kwh / max(self.battery_capacity_kwh, 1e-12)).astype(np.float32)
        remaining = (1.0 - current_battery).astype(np.float32)
        current_load = (self.load_cm3 / max(self.cargo_capacity_cm3, 1e-12)).astype(np.float32)
        current_time = ((self.current_time_s - self.working_start_s) / self.horizon_s).astype(np.float32)
        remaining_demand = np.broadcast_to(
            demand_norm,
            (self.n_traj, self.num_nodes),
        ).copy()
        remaining_demand[self.visited] = 0.0
        remaining_vehicle_ratio = np.maximum(
            0.0,
            (self.num_customers - self.vehicle_count.astype(np.float32))
            / max(float(self.num_customers), 1.0),
        )

        return {
            "cus_loc": coords[self.customer_start:self.station_start].astype(np.float32),
            "depot_loc": coords[0:1].astype(np.float32),
            "rs_loc": coords[self.station_start:].astype(np.float32),
            "demand": demand_norm,
            "time_window": tw_norm,
            "service_time": service_norm,
            "charging_power": charging_power,
            "remaining_demand": remaining_demand,
            "action_mask": action_mask,
            "last_node_idx": self.last.copy(),
            "current_load": current_load,
            "current_battery": current_battery,
            "remaining_battery": remaining,
            "current_time": current_time,
            "remaining_vehicle_ratio": remaining_vehicle_ratio,
            # Model-facing state is normalized; physical capacities stay in
            # self.battery_capacity_kwh / self.cargo_capacity_cm3 for dynamics.
            "battery_capacity": np.array([1.0], dtype=np.float32),
            "loading_capacity": np.array([1.0], dtype=np.float32),
            "visited_customers_ratio": visited_ratio,
            "visited_customers_raio": visited_ratio,
            "remain_feasible_customers_ratio": remain_feasible_ratio,
            "remain_feasible_customers_raio": remain_feasible_ratio,
        }

    def _make_info(self, action_mask: np.ndarray) -> dict[str, Any]:
        success = self.terminated & (self.served_customers == self.num_customers) & (self.last == 0)
        return {
            "action_mask": action_mask.copy(),
            "objective_distance_km": self.objective_distance_km.copy(),
            "vehicle_count": self.vehicle_count.copy(),
            "success": success.copy(),
            "served_customers": self.served_customers.copy(),
            "invalid_action": self.invalid_action.copy(),
            "travel_time_source": self.travel_time_source,
            "energy_source": self.energy_source,
            "charging_power_source": self.charging_power_source,
            "routes": self.get_routes(),
            "route_sequence": [merge_route_sequences(routes) for routes in self.get_routes()],
        }

    def _normalized_coords(self) -> np.ndarray:
        lo = self.coords_raw.min(axis=0)
        hi = self.coords_raw.max(axis=0)
        scale = np.maximum(hi - lo, 1e-6)
        return (self.coords_raw - lo) / scale

    def get_routes(self) -> list[list[list[int]]]:
        out: list[list[list[int]]] = []
        for t in range(self.n_traj):
            routes = [list(route) for route in self.routes[t]]
            if self.route_has_customer[t] and self.current_routes[t]:
                route = list(self.current_routes[t])
                if route[-1] != 0:
                    route.append(0)
                routes.append(route)
            out.append(routes)
        return out

    def _is_customer(self, node: int) -> bool:
        return self.customer_start <= int(node) < self.station_start

    def _is_station(self, node: int) -> bool:
        return self.station_start <= int(node) < self.num_nodes

    def _resolve_travel_time_matrix(
        self, instance: EVRPTWInstance
    ) -> tuple[np.ndarray, str]:
        matrix = instance.shortest_time_matrix_s
        source = "EVRPTWInstance.shortest_time_matrix_s"
        if matrix is None:
            matrix = instance.raw.get("running_time_shortest_matrix_s")
            source = "raw.running_time_shortest_matrix_s"
        if matrix is None:
            if self.matrix_mode == "canonical":
                raise ValueError(
                    "canonical RL evaluation requires running_time_shortest_matrix_s"
                )
            speed_kmh = float(
                instance.speed_profile.get("effective_speed_kmh")
                or instance.vehicle.get("design_speed_kmh")
                or 40.0
            )
            matrix = self.distance_km / max(speed_kmh / 3600.0, 1e-12)
            source = "legacy_distance_divided_by_scalar_speed"
        return self._validate_terminal_matrix(matrix, "travel time"), source

    def _resolve_energy_matrix(
        self, instance: EVRPTWInstance
    ) -> tuple[np.ndarray, str]:
        matrix = instance.energy_matrix_kwh
        source = "EVRPTWInstance.energy_matrix_kwh"
        if matrix is None:
            matrix = instance.raw.get("running_time_path_energy_kwh")
            source = "raw.running_time_path_energy_kwh"
        if matrix is None:
            if self.matrix_mode == "canonical":
                raise ValueError(
                    "canonical RL evaluation requires running_time_path_energy_kwh"
                )
            rate = float(
                instance.vehicle.get(
                    "consumption_kwh_per_km",
                    instance.vehicle.get(
                        "specific_energy_consumption_kwh_per_km", 0.404
                    ),
                )
            )
            matrix = self.distance_km * rate
            source = "legacy_objective_distance_times_consumption"
        return self._validate_terminal_matrix(matrix, "energy"), source

    def _resolve_charging_power(
        self, instance: EVRPTWInstance
    ) -> tuple[np.ndarray, str]:
        raw_power = instance.raw.get("charging_power_kw")
        source = "raw.charging_power_kw"
        if raw_power is None:
            raw_power = instance.cs_activation.get("charging_power_kw")
            source = "cs_activation.charging_power_kw"
        if raw_power is None:
            if self.num_stations == 0:
                return np.zeros(0, dtype=np.float64), "no_charging_stations"
            if self.charging_mode == "station_power_full":
                raise ValueError(
                    "canonical RL evaluation requires per-station charging_power_kw"
                )
            return np.ones(self.num_stations, dtype=np.float64), "legacy_unused"
        power = np.asarray(raw_power, dtype=np.float64)
        if power.shape != (self.num_stations,):
            raise ValueError(
                "charging_power_kw must have shape "
                f"{(self.num_stations,)}, got {power.shape}"
            )
        if np.any(~np.isfinite(power)) or np.any(power <= 0.0):
            raise ValueError("charging_power_kw must contain finite positive values")
        return power, source

    def _validate_terminal_matrix(self, value: Any, label: str) -> np.ndarray:
        matrix = np.asarray(value, dtype=np.float64)
        expected = (self.num_nodes, self.num_nodes)
        if matrix.shape != expected:
            raise ValueError(f"{label} matrix must have shape {expected}, got {matrix.shape}")
        if np.any(~np.isfinite(matrix)) or np.any(matrix < -1e-9):
            raise ValueError(f"{label} matrix must contain finite non-negative values")
        return matrix


__all__ = ["EVRPTWVectorEnv", "Transition"]
