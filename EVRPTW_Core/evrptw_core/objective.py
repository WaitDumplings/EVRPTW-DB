"""Versioned objective profiles shared by exact and heuristic benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any


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
