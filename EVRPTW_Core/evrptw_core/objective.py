"""Versioned objective profiles shared by exact and heuristic benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .schema import EVRPTWInstance


COST_DISTANCE_SOURCE = "running_time_path_distance_km"
LEGACY_DISTANCE_SOURCE = "distance_matrix_km"


@dataclass(frozen=True)
class ObjectiveConfig:
    mode: str = "distance"
    profile_id: str = "distance_v1"
    electricity_price_usd_per_kwh: float = 0.0
    consumption_kwh_per_km: float = 0.0
    vehicle_fixed_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"distance", "energy_vehicle_cost"}:
            raise ValueError(f"unsupported objective mode: {self.mode}")
        if not self.profile_id:
            raise ValueError("objective profile_id must be nonempty")
        for name in (
            "electricity_price_usd_per_kwh",
            "consumption_kwh_per_km",
            "vehicle_fixed_cost_usd",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if self.is_cost and self.distance_unit_cost <= 0.0:
            raise ValueError(
                "cost objective requires positive electricity price and consumption"
            )

    @property
    def is_cost(self) -> bool:
        return self.mode == "energy_vehicle_cost"

    @property
    def unit(self) -> str:
        return "USD" if self.is_cost else "km"

    @property
    def distance_unit_cost(self) -> float:
        if not self.is_cost:
            return 1.0
        return self.electricity_price_usd_per_kwh * self.consumption_kwh_per_km

    @property
    def vehicle_unit_cost(self) -> float:
        return self.vehicle_fixed_cost_usd if self.is_cost else 0.0

    def value(self, distance_km: Any, vehicles_started: Any) -> Any:
        return (
            self.distance_unit_cost * distance_km
            + self.vehicle_unit_cost * vehicles_started
        )

    def fields(self, distance_km: float, vehicles_started: int) -> dict[str, Any]:
        distance = float(distance_km)
        vehicles = int(vehicles_started)
        electricity = self.distance_unit_cost * distance if self.is_cost else None
        fixed = self.vehicle_unit_cost * vehicles if self.is_cost else None
        return {
            "objective_mode": self.mode,
            "objective_profile_id": self.profile_id,
            "objective_unit": self.unit,
            "objective_value": float(self.value(distance, vehicles)),
            "objective_cost_usd": (
                float(electricity + fixed) if self.is_cost else None
            ),
            "electricity_cost_usd": electricity,
            "vehicle_cost_usd": fixed,
            "vehicles_started": vehicles,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_objective(path: str | Path) -> ObjectiveConfig:
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    selected = payload.get("objective", payload)
    if not isinstance(selected, dict):
        raise ValueError("objective profile must contain an object mapping")
    return ObjectiveConfig(**selected)


def select_objective_distance(
    instance: EVRPTWInstance, objective: ObjectiveConfig
) -> EVRPTWInstance:
    """Select the solver/replay distance without modifying the source instance.

    New nonlearning monetary runs price the distance travelled on the fastest
    time path, the same path used for travel time and battery energy. Explicit
    legacy distance runs retain the shortest-distance matrix. This is an
    opt-in instance adapter, not a global change to archived or DRL inputs.

    A missing fastest-time-path matrix is an error, including for Euclidean
    inputs: their adapter must explicitly supply the equivalent matrix. We
    never infer road path equivalence from coordinates or silently substitute
    shortest-distance paths. Reapplying the adapter is safe, and changing back
    to distance mode recovers the original matrix.
    """

    raw = dict(instance.raw)
    original_distance = raw.get(
        "shortest_distance_matrix_km", instance.distance_matrix_km
    )
    source = COST_DISTANCE_SOURCE if objective.is_cost else LEGACY_DISTANCE_SOURCE
    if objective.is_cost:
        selected = raw.get(COST_DISTANCE_SOURCE)
        if selected is None:
            raise ValueError(
                f"cost objective for {instance.instance_id} requires "
                f"{COST_DISTANCE_SOURCE}; refusing a shortest-distance fallback"
            )
    else:
        selected = original_distance
    distance = np.asarray(selected)
    expected_shape = (instance.num_terminals, instance.num_terminals)
    if distance.shape != expected_shape:
        raise ValueError(
            f"{source} must have shape {expected_shape}, got {distance.shape}"
        )
    if not np.issubdtype(distance.dtype, np.number) or np.iscomplexobj(distance):
        raise ValueError(f"{source} must contain real numeric distances")
    # Positive infinity can denote an unreachable arc, which the solvers and
    # independent replay already reject. NaN and negative distances cannot.
    if np.any(np.isnan(distance)) or np.any(distance < 0.0):
        raise ValueError(f"{source} contains NaN or negative distances")
    metadata = dict(instance.metadata)
    metric_contract = dict(metadata.get("metric_contract", {}))
    metric_contract["objective"] = source
    metadata.update(
        objective_distance_source=source,
        objective_profile_id=objective.profile_id,
        metric_contract=metric_contract,
    )
    raw.update(
        shortest_distance_matrix_km=original_distance,
        distance_matrix_km=distance,
        metadata=metadata,
    )
    return replace(instance, distance_matrix_km=distance, metadata=metadata, raw=raw)
