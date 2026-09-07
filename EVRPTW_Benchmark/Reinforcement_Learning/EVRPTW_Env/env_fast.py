from __future__ import annotations

from typing import Any

import numpy as np

from .env import EVRPTWVectorEnv, Transition
from .mask_jit import (
    NUMBA_AVAILABLE,
    compute_action_mask_jit,
    refresh_route_return_times_jit,
)


class EVRPTWVectorEnvFast(EVRPTWVectorEnv):
    """Experimental drop-in optimized EVRP-TW-D env.

    This class intentionally lives beside the reference implementation so the
    reference dynamics remain available. It keeps the exact EVRP transition
    semantics while removing repeated work that dominates larger Cus/CS settings:

    - cache the previous action mask and reuse it to validate the next action;
    - compute route-local station return witnesses inside the mask kernel;
    - cache static observation arrays that do not change during a rollout;
    - optionally compute action masks through a numba JIT array kernel;
    - optionally return light training info without route reconstruction.
    """

    def __init__(self, *args: Any, info_level: str = "full", use_jit_mask: bool = True, **kwargs: Any) -> None:
        if info_level not in {"full", "light"}:
            raise ValueError("info_level must be 'full' or 'light'")
        self.info_level = info_level
        self.use_jit_mask = bool(use_jit_mask and NUMBA_AVAILABLE)
        self._current_action_mask: np.ndarray | None = None
        self._route_return_time_cache: np.ndarray | None = None
        self._route_return_station_state_cache: np.ndarray | None = None
        self._static_obs_cache: dict[str, np.ndarray] | None = None
        super().__init__(*args, **kwargs)

    def set_instance(self, instance):
        super().set_instance(instance)
        self._build_static_observation_cache()
        self._current_action_mask = None
        self._route_return_time_cache = None
        self._route_return_station_state_cache = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._route_return_time_cache = None
        self._route_return_station_state_cache = None
        obs, info = super().reset(seed=seed, options=options)
        self._current_action_mask = np.asarray(obs["action_mask"], dtype=bool).copy()
        return obs, info

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.int64).reshape(self.n_traj)
        if self._current_action_mask is None:
            mask_before = self._compute_action_mask()
        else:
            mask_before = self._current_action_mask
        reward = np.zeros(self.n_traj, dtype=np.float32)
        self.invalid_action.fill(False)

        for t in range(self.n_traj):
            if self.terminated[t] or self.truncated[t]:
                continue
            destination = int(action_arr[t])
            if destination < 0 or destination >= self.num_nodes or not mask_before[t, destination]:
                self.invalid_action[t] = True
                self.truncated[t] = True
                self.failure_reason[t] = "invalid_action"
                reward[t] += self.invalid_action_penalty
                continue
            reward[t] += self._apply_action(t, destination)

        self.step_count += 1
        if self.step_count >= self.max_steps:
            unfinished = ~self.terminated
            self.truncated[unfinished] = True
            self.failure_reason[
                unfinished & (self.failure_reason == "in_progress")
            ] = "environment_step_limit"

        obs = self._make_observation()
        action_mask = obs["action_mask"]
        no_action = (~action_mask.any(axis=1)) & (~self.terminated) & (~self.truncated)
        if np.any(no_action):
            self.truncated[no_action] = True
            self.failure_reason[no_action] = "no_feasible_action"
            reward[no_action] += self.invalid_action_penalty
            obs = self._make_observation()
            action_mask = obs["action_mask"]

        self._current_action_mask = np.asarray(action_mask, dtype=bool).copy()
        info = self._make_info(action_mask)
        return obs, reward, self.terminated.copy(), self.truncated.copy(), info

    def _compute_action_mask(self) -> np.ndarray:
        if not self.use_jit_mask:
            return super()._compute_action_mask()
        route_return_time_s = self._cached_route_return_times()
        return compute_action_mask_jit(
            n_traj=self.n_traj,
            num_nodes=self.num_nodes,
            num_customers=self.num_customers,
            station_start=self.station_start,
            last=self.last,
            visited=self.visited,
            cs_visited_current_route=self.cs_visited_current_route,
            route_return_time_s=route_return_time_s,
            terminated=self.terminated,
            truncated=self.truncated,
            served_customers=self.served_customers,
            route_has_customer=self.route_has_customer,
            current_time_s=self.current_time_s,
            battery_used_kwh=self.battery_used_kwh,
            load_cm3=self.load_cm3,
            demand_cm3=self.demand_cm3,
            service_time_s=self.service_time_s,
            tw_s=self.tw_s,
            travel_time_s=self.travel_time_s,
            energy_kwh=self.energy_kwh,
            battery_capacity_kwh=self.battery_capacity_kwh,
            cargo_capacity_cm3=self.cargo_capacity_cm3,
            full_charge_time_s=self.full_charge_time_s,
            charging_power_kw=self.charging_power_kw,
            charging_power_derating_factor=self.charging_power_derating_factor,
            working_end_s=self.working_end_s,
            station_power_full=self.charging_mode == "station_power_full",
            legacy_fixed_full=self.charging_mode == "legacy_fixed_full",
            allow_consecutive_station_actions=(
                self.allow_consecutive_station_actions
            ),
        )

    def _cached_route_return_times(self) -> np.ndarray:
        expected_shape = (self.n_traj, self.num_nodes)
        if (
            self._route_return_time_cache is None
            or self._route_return_time_cache.shape != expected_shape
            or self._route_return_station_state_cache is None
            or self._route_return_station_state_cache.shape != expected_shape
        ):
            self._route_return_time_cache = np.full(
                expected_shape, np.inf, dtype=np.float64
            )
            self._route_return_station_state_cache = np.logical_not(
                self.cs_visited_current_route
            )

        station_state = self._route_return_station_state_cache
        return_time = self._route_return_time_cache
        dirty = np.any(
            station_state != self.cs_visited_current_route, axis=1
        ) & (~self.terminated) & (~self.truncated)
        if np.any(dirty):
            refresh_route_return_times_jit(
                route_return_time_s=return_time,
                dirty_trajectories=dirty,
                station_start=self.station_start,
                num_nodes=self.num_nodes,
                cs_visited_current_route=self.cs_visited_current_route,
                travel_time_s=self.travel_time_s,
                energy_kwh=self.energy_kwh,
                battery_capacity_kwh=self.battery_capacity_kwh,
                full_charge_time_s=self.full_charge_time_s,
                charging_power_kw=self.charging_power_kw,
                charging_power_derating_factor=self.charging_power_derating_factor,
                station_power_full=self.charging_mode == "station_power_full",
                legacy_fixed_full=self.charging_mode == "legacy_fixed_full",
                allow_consecutive_station_actions=(
                    self.allow_consecutive_station_actions
                ),
            )
            station_state[dirty] = self.cs_visited_current_route[dirty]
        return return_time

    def _build_static_observation_cache(self) -> None:
        coords = self._normalized_coords().astype(np.float32)
        demand_norm = (self.demand_cm3 / max(self.cargo_capacity_cm3, 1e-12)).astype(np.float32)
        tw_norm = ((self.tw_s - self.working_start_s) / self.horizon_s).astype(np.float32)
        service_norm = (self.service_time_s / self.horizon_s).astype(np.float32)
        self._static_obs_cache = {
            "cus_loc": coords[self.customer_start:self.station_start],
            "depot_loc": coords[0:1],
            "rs_loc": coords[self.station_start:],
            "demand": demand_norm,
            "time_window": tw_norm,
            "service_time": service_norm,
            "charging_power": self._normalized_charging_power(),
            "charging_time_ratio": self._charging_time_ratio(),
            "battery_capacity": np.array([1.0], dtype=np.float32),
            "loading_capacity": np.array([1.0], dtype=np.float32),
        }

    def _make_observation(self) -> dict[str, np.ndarray]:
        action_mask = self._compute_action_mask()
        self._current_action_mask = np.asarray(action_mask, dtype=bool).copy()
        feasible_customer_count = action_mask[:, self.customer_start:self.station_start].sum(axis=1, keepdims=True)
        visited_ratio = (self.served_customers.astype(np.float32) / max(float(self.num_customers), 1.0))[:, None]
        remain_feasible_ratio = feasible_customer_count.astype(np.float32) / max(float(self.num_customers), 1.0)

        static = self._static_obs_cache
        if static is None:
            self._build_static_observation_cache()
            static = self._static_obs_cache
        assert static is not None
        current_battery = (self.battery_used_kwh / max(self.battery_capacity_kwh, 1e-12)).astype(np.float32)
        remaining = (1.0 - current_battery).astype(np.float32)
        current_load = (self.load_cm3 / max(self.cargo_capacity_cm3, 1e-12)).astype(np.float32)
        current_time = ((self.current_time_s - self.working_start_s) / self.horizon_s).astype(np.float32)
        remaining_demand = np.broadcast_to(
            static["demand"],
            (self.n_traj, self.num_nodes),
        ).copy()
        remaining_demand[self.visited] = 0.0
        remaining_vehicle_ratio = np.maximum(
            0.0,
            (self.num_customers - self.vehicle_count.astype(np.float32))
            / max(float(self.num_customers), 1.0),
        )

        return {
            "cus_loc": static["cus_loc"],
            "depot_loc": static["depot_loc"],
            "rs_loc": static["rs_loc"],
            "demand": static["demand"],
            "time_window": static["time_window"],
            "service_time": static["service_time"],
            "charging_power": static["charging_power"],
            "charging_time_ratio": static["charging_time_ratio"],
            "remaining_demand": remaining_demand,
            "action_mask": action_mask,
            "last_node_idx": self.last.copy(),
            "current_load": current_load,
            "current_battery": current_battery,
            "remaining_battery": remaining,
            "current_time": current_time,
            "remaining_vehicle_ratio": remaining_vehicle_ratio,
            "battery_capacity": static["battery_capacity"],
            "loading_capacity": static["loading_capacity"],
            "visited_customers_ratio": visited_ratio,
            "visited_customers_raio": visited_ratio,
            "remain_feasible_customers_ratio": remain_feasible_ratio,
            "remain_feasible_customers_raio": remain_feasible_ratio,
        }

    def _make_info(self, action_mask: np.ndarray) -> dict[str, Any]:
        success = self.terminated & (self.served_customers == self.num_customers) & (self.last == 0)
        info: dict[str, Any] = {
            "action_mask": action_mask.copy(),
            "objective_distance_km": self.objective_distance_km.copy(),
            **self._objective_info(),
            "vehicle_count": self.vehicle_count.copy(),
            "success": success.copy(),
            "served_customers": self.served_customers.copy(),
            "invalid_action": self.invalid_action.copy(),
            "failure_reason": self.failure_reason.copy(),
            "travel_time_source": self.travel_time_source,
            "energy_source": self.energy_source,
            "charging_power_source": self.charging_power_source,
        }
        if self.info_level == "full":
            routes = self.get_routes()
            info["routes"] = routes
            from evrptw_core.schema import merge_route_sequences

            info["route_sequence"] = [merge_route_sequences(route_set) for route_set in routes]
        return info

    def _normalized_charging_power(self) -> np.ndarray:
        out = np.zeros(self.num_nodes, dtype=np.float32)
        if self.num_stations:
            scale = max(float(np.max(self.charging_power_kw)), 1e-12)
            out[self.station_start :] = (self.charging_power_kw / scale).astype(
                np.float32
            )
        return out

    def _charging_time_ratio(self) -> np.ndarray:
        out = np.zeros(self.num_nodes, dtype=np.float32)
        if self.num_stations:
            usable_power_kw = np.maximum(
                self.charging_power_kw * self.charging_power_derating_factor,
                1e-12,
            )
            out[self.station_start :] = (
                3600.0 * self.battery_capacity_kwh / usable_power_kw / self.horizon_s
            ).astype(np.float32)
        return out


__all__ = ["EVRPTWVectorEnvFast", "Transition"]
