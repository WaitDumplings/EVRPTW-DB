"""Optional operations profile used by the U.S. Stage-2 reference adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_reference_profile(path: str | Path, *, official: bool = False) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "cle_evrptw_us_reference_instance_profile_v1":
        raise ValueError(f"Unsupported Stage-2 reference profile: {payload.get('schema')!r}")
    if official and not bool(payload.get("official_generation_eligible", False)):
        raise ValueError(
            "The selected operations profile is still a development calibration and cannot "
            "produce official benchmark instances."
        )
    weights = payload["day_type"]
    if (int(weights["weekday_weight"]), int(weights["weekend_weight"])) != (5, 2):
        raise ValueError("The V1 U.S. profile must use the frozen weekday/weekend ratio 5:2")
    energy = payload["energy"]
    battery = float(energy["battery_capacity_kwh"])
    range_km = float(energy["nominal_range_km"])
    specific = float(energy["specific_energy_consumption_kwh_per_km"])
    if abs(specific - battery / range_km) > 1e-9:
        raise ValueError("Linear energy coefficient must equal battery_capacity / nominal_range")
    if payload["vehicle"]["cargo_capacity_cm3"] != 18_500_000.0:
        raise ValueError("The Delivery 700 reference cargo capacity must be 18.5 m3")
    charging = payload["charging"]
    if float(charging["dc_vehicle_cap_kw"]) != 100.0:
        raise ValueError("The V1 Rivian reference DC charging cap must be 100 kW")
    if float(charging["ac_l2_vehicle_cap_kw"]) != 11.0:
        raise ValueError("The V1 Rivian reference AC L2 charging cap must be 11 kW")
    return payload
