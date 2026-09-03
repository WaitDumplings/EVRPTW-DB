from __future__ import annotations

from typing import Any

import numpy as np

from ..EVRPTW_Env import EVRPTWVectorEnvFast


class DRLTSSoftConstraintEnv(EVRPTWVectorEnvFast):
    """Paper Stage-1 environment with benchmark-normalized violations.

    Customer uniqueness and the paper tour rules remain hard. Cargo, time
    window, and battery violations are allowed and accumulated as normalized
    benchmark-adapter penalties. This class is training-only; evaluation uses
    :class:`DRLTSHardConstraintEnv` and the independent route verifier.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["use_jit_mask"] = False
        super().__init__(*args, **kwargs)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        observation, info = super().reset(seed=seed, options=options)
        self.capacity_violation = np.zeros(self.n_traj, dtype=np.float64)
        self.time_violation = np.zeros(self.n_traj, dtype=np.float64)
        self.energy_violation = np.zeros(self.n_traj, dtype=np.float64)
        return observation, self._with_violation_info(info)

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.int64).reshape(self.n_traj)
        mask = self._compute_action_mask()
        for trajectory, destination in enumerate(action_arr):
            if self.terminated[trajectory] or self.truncated[trajectory]:
                continue
            if destination < 0 or destination >= self.num_nodes:
                continue
            if not mask[trajectory, destination]:
                continue
            capacity, time_window, energy = self._normalized_violations(
                trajectory,
                int(destination),
            )
            self.capacity_violation[trajectory] += capacity
            self.time_violation[trajectory] += time_window
            self.energy_violation[trajectory] += energy
        observation, reward, terminated, truncated, info = super().step(action_arr)
        return (
            observation,
            reward,
            terminated,
            truncated,
            self._with_violation_info(info),
        )

    def _compute_action_mask(self) -> np.ndarray:
        mask = np.zeros((self.n_traj, self.num_nodes), dtype=bool)
        for trajectory in range(self.n_traj):
            if self.terminated[trajectory] or self.truncated[trajectory]:
                mask[trajectory, 0] = True
                continue
            start = int(self.last[trajectory])
            if self.served_customers[trajectory] == self.num_customers:
                mask[trajectory, 0] = True
                continue
            if start != 0:
                mask[trajectory, 0] = True
            for customer in self.customer_nodes:
                node = int(customer)
                if not self.visited[trajectory, node]:
                    mask[trajectory, node] = True
            if self._is_customer(start):
                mask[trajectory, self.station_nodes] = True
        return mask

    def _normalized_violations(
        self,
        trajectory: int,
        destination: int,
    ) -> tuple[float, float, float]:
        start = int(self.last[trajectory])
        capacity = 0.0
        if self._is_customer(destination):
            remaining_capacity = (
                self.cargo_capacity_cm3 - self.load_cm3[trajectory]
            )
            capacity = max(
                float(self.demand_cm3[destination] - remaining_capacity),
                0.0,
            ) / max(self.cargo_capacity_cm3, 1e-12)
        arrival = float(
            self.current_time_s[trajectory]
            + self.travel_time_s[start, destination]
        )
        due = float(self.tw_s[destination, 1])
        time_window = max(arrival - due, 0.0) / self.horizon_s
        remaining_energy = (
            self.battery_capacity_kwh - self.battery_used_kwh[trajectory]
        )
        energy = max(
            float(self.energy_kwh[start, destination] - remaining_energy),
            0.0,
        ) / max(self.battery_capacity_kwh, 1e-12)
        return capacity, time_window, energy

    def _charge_time_s(self, battery_used_kwh: float, station_node: int) -> float:
        if self.charging_mode != "station_power_full":
            return super()._charge_time_s(battery_used_kwh, station_node)
        if not self._is_station(station_node):
            raise ValueError(f"charging target is not a station: {station_node}")
        station_offset = int(station_node) - self.station_start
        usable_power_kw = (
            float(self.charging_power_kw[station_offset])
            * self.charging_power_derating_factor
        )
        return 3600.0 * max(float(battery_used_kwh), 0.0) / usable_power_kw

    def _with_violation_info(self, info: dict[str, Any]) -> dict[str, Any]:
        return {
            **info,
            "capacity_violation_normalized": self.capacity_violation.copy(),
            "time_violation_normalized": self.time_violation.copy(),
            "energy_violation_normalized": self.energy_violation.copy(),
        }


__all__ = ["DRLTSSoftConstraintEnv"]
