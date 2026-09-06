from __future__ import annotations

from typing import Any

import numpy as np

from ..EVRPTW_Env import EVRPTWVectorEnvFast


SOFT_VIOLATION_CONTRACT_ID = "drl_ts_soft_auxiliary_v1"
SOFT_VIOLATION_DENOMINATOR = "num_customers"
SOFT_VIOLATION_AGGREGATION = (
    "clipped_normalized_excess_sum_div_fixed_customers_then_component_clip"
)
SOFT_VIOLATION_APPLICABILITY = (
    "stage1_soft_only_capacity_on_customer_time_energy_on_all_valid_travel"
)
DEFAULT_SOFT_VIOLATION_STEP_CLIP = 1.0
DEFAULT_SOFT_VIOLATION_COMPONENT_CLIP = 1.0


class DRLTSSoftConstraintEnv(EVRPTWVectorEnvFast):
    """Paper Stage-1 environment with benchmark-normalized violations.

    Customer uniqueness and the paper tour rules remain hard. Cargo, time
    window, and battery violations are allowed and accumulated as normalized
    benchmark-adapter penalties. This class is training-only; evaluation uses
    :class:`DRLTSHardConstraintEnv` and the independent route verifier.
    """

    def __init__(
        self,
        *args: Any,
        soft_violation_contract_id: str = SOFT_VIOLATION_CONTRACT_ID,
        soft_violation_step_clip: float = DEFAULT_SOFT_VIOLATION_STEP_CLIP,
        soft_violation_denominator: str = SOFT_VIOLATION_DENOMINATOR,
        **kwargs: Any,
    ) -> None:
        if str(soft_violation_contract_id) != SOFT_VIOLATION_CONTRACT_ID:
            raise ValueError(
                "unsupported DRL-TS soft violation contract: "
                f"{soft_violation_contract_id!r}"
            )
        if str(soft_violation_denominator) != SOFT_VIOLATION_DENOMINATOR:
            raise ValueError(
                "DRL-TS soft violation denominator must be num_customers"
            )
        step_clip = float(soft_violation_step_clip)
        if not np.isfinite(step_clip) or step_clip <= 0.0:
            raise ValueError("soft violation step clip must be finite and positive")
        self.soft_violation_contract_id = SOFT_VIOLATION_CONTRACT_ID
        self.soft_violation_step_clip = step_clip
        self.soft_violation_denominator = SOFT_VIOLATION_DENOMINATOR
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
        self.capacity_violation_clipped_sum = np.zeros(
            self.n_traj, dtype=np.float64
        )
        self.time_violation_clipped_sum = np.zeros(self.n_traj, dtype=np.float64)
        self.energy_violation_clipped_sum = np.zeros(
            self.n_traj, dtype=np.float64
        )
        self.capacity_violation_applicable_transitions = np.zeros(
            self.n_traj, dtype=np.int64
        )
        self.time_violation_applicable_transitions = np.zeros(
            self.n_traj, dtype=np.int64
        )
        self.energy_violation_applicable_transitions = np.zeros(
            self.n_traj, dtype=np.int64
        )
        return observation, self._with_violation_info(info)

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.int64).reshape(self.n_traj)
        # The fast parent stores the observation's pre-action mask. Penalties
        # and the actual transition must use that same soft-rule mask.
        mask = self._current_action_mask
        if mask is None:
            mask = self._compute_action_mask()
            self._current_action_mask = mask
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
            self._record_normalized_violations(
                trajectory,
                int(destination),
                capacity,
                time_window,
                energy,
            )
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
        active = ~(
            self.terminated
            | self.truncated
            | (self.served_customers == self.num_customers)
        )
        mask[:, 0] = ~active | (self.last != 0)
        mask[:, self.customer_nodes] = (
            active[:, None] & ~self.visited[:, self.customer_nodes]
        )
        at_customer = (self.last >= self.customer_start) & (
            self.last < self.station_start
        )
        mask[:, self.station_nodes] = (
            (active & at_customer)[:, None]
            & ~self.cs_visited_current_route[:, self.station_nodes]
        )
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

    def _record_normalized_violations(
        self,
        trajectory: int,
        destination: int,
        capacity: float,
        time_window: float,
        energy: float,
    ) -> None:
        """Record raw diagnostics and bounded per-transition soft costs.

        Raw sums preserve the historical audit signal.  The clipped sums are
        the only values eligible for the Stage-1 training auxiliary. Capacity
        applies only to customer arrivals; time and energy apply to every valid
        travel transition.  Their fixed customer-count denominator is applied
        later in :mod:`rollout`, so adding zero-violation depot/station travel
        can never dilute a violation already incurred.
        """

        values = np.asarray((capacity, time_window, energy), dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise RuntimeError(
                "DRL-TS produced a non-finite or negative normalized violation"
            )
        self.capacity_violation[trajectory] += float(capacity)
        self.time_violation[trajectory] += float(time_window)
        self.energy_violation[trajectory] += float(energy)
        clip = self.soft_violation_step_clip
        if self._is_customer(destination):
            self.capacity_violation_clipped_sum[trajectory] += min(
                float(capacity), clip
            )
            self.capacity_violation_applicable_transitions[trajectory] += 1
        self.time_violation_clipped_sum[trajectory] += min(
            float(time_window), clip
        )
        self.energy_violation_clipped_sum[trajectory] += min(float(energy), clip)
        self.time_violation_applicable_transitions[trajectory] += 1
        self.energy_violation_applicable_transitions[trajectory] += 1

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
            "soft_violation_contract_id": self.soft_violation_contract_id,
            "soft_violation_step_clip": self.soft_violation_step_clip,
            "soft_violation_denominator": self.soft_violation_denominator,
            "capacity_violation_normalized": self.capacity_violation.copy(),
            "time_violation_normalized": self.time_violation.copy(),
            "energy_violation_normalized": self.energy_violation.copy(),
            "capacity_violation_raw_sum": self.capacity_violation.copy(),
            "time_violation_raw_sum": self.time_violation.copy(),
            "energy_violation_raw_sum": self.energy_violation.copy(),
            "capacity_violation_clipped_sum": (
                self.capacity_violation_clipped_sum.copy()
            ),
            "time_violation_clipped_sum": self.time_violation_clipped_sum.copy(),
            "energy_violation_clipped_sum": (
                self.energy_violation_clipped_sum.copy()
            ),
            "capacity_violation_applicable_transitions": (
                self.capacity_violation_applicable_transitions.copy()
            ),
            "time_violation_applicable_transitions": (
                self.time_violation_applicable_transitions.copy()
            ),
            "energy_violation_applicable_transitions": (
                self.energy_violation_applicable_transitions.copy()
            ),
        }


__all__ = [
    "DEFAULT_SOFT_VIOLATION_COMPONENT_CLIP",
    "DEFAULT_SOFT_VIOLATION_STEP_CLIP",
    "DRLTSSoftConstraintEnv",
    "SOFT_VIOLATION_CONTRACT_ID",
    "SOFT_VIOLATION_DENOMINATOR",
    "SOFT_VIOLATION_AGGREGATION",
    "SOFT_VIOLATION_APPLICABILITY",
]
