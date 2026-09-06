"""Versioned benchmark objectives, independent of policy architecture and shaping."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ObjectiveConfig:
    mode: str = "distance"
    profile_id: str = "distance_v1"
    electricity_price_usd_per_kwh: float = 0.1341
    consumption_kwh_per_km: float = 100.0 / 257.0
    vehicle_fixed_cost_usd: float = 33.56

    def __post_init__(self) -> None:
        if self.mode not in {"distance", "energy_vehicle_cost"}:
            raise ValueError(f"unsupported objective mode: {self.mode}")
        if not self.profile_id:
            raise ValueError("objective profile_id must be nonempty")
        for name in (
            "electricity_price_usd_per_kwh", "consumption_kwh_per_km",
            "vehicle_fixed_cost_usd",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if self.is_cost and self.distance_unit_cost <= 0:
            raise ValueError("cost objective requires positive electricity price and consumption")

    @property
    def is_cost(self) -> bool:
        return self.mode == "energy_vehicle_cost"

    @property
    def unit(self) -> str:
        return "USD" if self.is_cost else "km"

    @property
    def distance_unit_cost(self) -> float:
        return (
            self.electricity_price_usd_per_kwh * self.consumption_kwh_per_km
            if self.is_cost else 1.0
        )

    @property
    def vehicle_unit_cost(self) -> float:
        return self.vehicle_fixed_cost_usd if self.is_cost else 0.0

    def value(self, distance: Any, vehicles: Any) -> Any:
        # Arithmetic supports scalar, NumPy and PyTorch values without detaching.
        return self.distance_unit_cost * distance + self.vehicle_unit_cost * vehicles

    def reward_scale(self, distance_scale_km: float, num_customers: int, scale_mode: str) -> float:
        distance_scale = float(distance_scale_km)
        if not math.isfinite(distance_scale) or distance_scale <= 0:
            raise ValueError("reward distance scale must be finite and positive")
        if not self.is_cost:
            return distance_scale
        # A singleton repair sum contains N dispatches; a mean/median one.
        # Use training-pool scales unchanged, converting the entire reference
        # workload to the active objective's units, not only its distance term.
        dispatches = int(num_customers) if str(scale_mode).removeprefix("dataset_") == "single_customer_repair_sum" else 1
        return float(self.value(distance_scale, dispatches))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fields(self, distance: float, vehicles: int) -> dict[str, Any]:
        distance = float(distance)
        vehicles = int(vehicles)
        electricity = self.electricity_price_usd_per_kwh * self.consumption_kwh_per_km * distance
        fixed = self.vehicle_fixed_cost_usd * vehicles
        return {
            "objective_mode": self.mode,
            "objective_profile_id": self.profile_id,
            "objective_unit": self.unit,
            "objective_value": float(self.value(distance, vehicles)),
            "objective_cost_usd": electricity + fixed if self.is_cost else None,
            "electricity_cost_usd": electricity if self.is_cost else None,
            "vehicle_cost_usd": fixed if self.is_cost else None,
            "vehicles_started": vehicles,
        }


def resolve_objective(config: Any = None) -> ObjectiveConfig:
    if config is None:
        return ObjectiveConfig()
    if isinstance(config, ObjectiveConfig):
        return config
    # Legacy entrypoints can import the same package under a shorter module
    # prefix. Revalidate its serialized fields rather than relying on identity.
    if callable(getattr(config, "to_dict", None)):
        return ObjectiveConfig(**config.to_dict())
    if isinstance(config, (str, Path)):
        return load_objective(config)
    if isinstance(config, dict):
        if "objective" in config:
            config = config["objective"]
        return ObjectiveConfig(**config)
    raise TypeError("objective must be a config mapping, ObjectiveConfig, or JSON path")


def load_objective(path: str | Path) -> ObjectiveConfig:
    source = Path(path)
    if not source.is_absolute() and not source.exists():
        source = REPO_ROOT / source
    with source.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    return resolve_objective(payload.get("objective", payload))


def objective_from_args(args: Any) -> ObjectiveConfig:
    selected = getattr(args, "objective", None)
    if selected is None:
        selected = getattr(args, "objective_config", None)
    return resolve_objective(selected)


def objective_from_checkpoint(checkpoint: dict[str, Any], override: Any = None) -> ObjectiveConfig:
    saved_args = checkpoint.get("args", {}) or {}
    if not isinstance(saved_args, dict):
        saved_args = vars(saved_args)
    snapshots = [
        checkpoint.get("objective_config"),
        (checkpoint.get("config", {}) or {}).get("objective"),
        saved_args.get("objective"),
    ]
    # A legacy CLI filepath alone never defines checkpoint training semantics.
    if isinstance(saved_args.get("objective_config"), dict):
        snapshots.append(saved_args["objective_config"])
    resolved = []
    for snapshot in snapshots:
        if snapshot is None:
            continue
        if isinstance(snapshot, (str, Path)):
            raise ValueError("checkpoint objective requires resolved values, not a mutable config path")
        resolved.append(resolve_objective(snapshot))
    result = resolved[0] if resolved else ObjectiveConfig()
    if any(snapshot.to_dict() != result.to_dict() for snapshot in resolved[1:]):
        raise ValueError("checkpoint objective snapshots disagree")
    if override is not None and resolve_objective(override).to_dict() != result.to_dict():
        raise ValueError("checkpoint objective mismatch; do not relabel or resume a different objective")
    return result


def route_dispatch_count(routes: list[list[int]]) -> int:
    return sum(
        int(int(origin) == 0 and int(destination) != 0)
        for route in routes for origin, destination in zip(route, route[1:])
    )


__all__ = [
    "ObjectiveConfig", "resolve_objective", "load_objective", "objective_from_args",
    "objective_from_checkpoint", "route_dispatch_count",
]
