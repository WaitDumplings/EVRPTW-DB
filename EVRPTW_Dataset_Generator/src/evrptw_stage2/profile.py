"""Versioned operations profile for the U.S. Stage-2 reference adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROFILE_SCHEMA_V2 = "cle_evrptw_us_reference_instance_profile_v2"
DELETED_V1_KEYS = frozenset(
    {
        "customer_activation",
        "packages",
        "service_time",
        "time_window",
        "feasibility",
    }
)


def load_reference_profile(path: str | Path, *, official: bool = False) -> dict[str, Any]:
    profile_path = Path(path)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if payload.get("schema") != PROFILE_SCHEMA_V2:
        raise ValueError(f"Unsupported Stage-2 reference profile: {payload.get('schema')!r}")
    deleted = sorted(DELETED_V1_KEYS.intersection(payload))
    if deleted:
        raise ValueError(f"Stage-2 profile v2 rejects deleted V1 keys: {deleted}")
    if official:
        promotion = payload.get("acceptance_promotion", {})
        report_hash = str(promotion.get("pilot_report_sha256", "")).lower()
        acceptance_hash = str(promotion.get("acceptance_config_sha256", "")).lower()
        if (
            payload.get("profile_status") != "release_calibrated"
            or not bool(payload.get("official_generation_eligible", False))
            or promotion.get("schema") != "evrptw_profile_acceptance_promotion_v1"
            or not str(promotion.get("pilot_report_id", ""))
            or len(report_hash) != 64
            or any(character not in "0123456789abcdef" for character in report_hash)
            or len(acceptance_hash) != 64
            or any(character not in "0123456789abcdef" for character in acceptance_hash)
            or not str(promotion.get("advisor_signoff_id", ""))
        ):
            raise ValueError(
                "Official generation requires a release_calibrated profile promoted from "
                "a passing pilot report, frozen acceptance config, and advisor sign-off"
            )
    weights = payload["day_type"]
    if (int(weights["weekday_weight"]), int(weights["weekend_weight"])) != (5, 2):
        raise ValueError("The V2 U.S. profile must use the frozen weekday/weekend ratio 5:2")
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
        raise ValueError("The V2 Rivian reference DC charging cap must be 100 kW")
    if float(charging["ac_l2_vehicle_cap_kw"]) != 11.0:
        raise ValueError("The V2 Rivian reference AC L2 charging cap must be 11 kW")
    if "charging_efficiency" in charging:
        raise ValueError("Stage-2 profile v2 rejects charging_efficiency")
    if float(charging.get("charging_power_derating_factor", -1.0)) != 0.90:
        raise ValueError("Canonical charging_power_derating_factor must equal 0.90")
    if charging.get("missing_station_power_policy") != "national_mode_median_or_error":
        raise ValueError("V2 missing station power must use national mode median or error")
    medians = charging.get("national_mode_medians_kw", {})
    if set(medians) != {"ac_level2", "dc_fast"} or any(
        float(value) <= 0.0 for value in medians.values()
    ):
        raise ValueError("V2 requires positive frozen AC L2 and DC-fast national medians")
    registry_name = charging.get("national_mode_median_registry")
    registry_sha256 = charging.get("national_mode_median_registry_sha256")
    if not registry_name or not registry_sha256:
        raise ValueError("V2 charging-power median registry and SHA256 are required")
    registry_path = profile_path.parent / str(registry_name)
    if not registry_path.is_file():
        raise ValueError(f"Charging-power median registry is missing: {registry_path}")
    registry_bytes = registry_path.read_bytes()
    actual_sha256 = hashlib.sha256(registry_bytes).hexdigest()
    if actual_sha256 != str(registry_sha256):
        raise ValueError("Charging-power median registry SHA256 mismatch")
    registry = json.loads(registry_bytes)
    if registry.get("schema") != "evrptw_national_charging_power_medians_v1":
        raise ValueError("Unsupported charging-power median registry schema")
    registry_medians = registry.get("national_mode_medians_kw", {})
    if {key: float(value) for key, value in medians.items()} != {
        key: float(value) for key, value in registry_medians.items()
    }:
        raise ValueError("Profile charging-power medians do not match the frozen registry")
    turn = payload["turn_penalty"]
    if any(float(turn[key]) != 0.0 for key in ("right_turn_s", "left_turn_s", "u_turn_s")):
        raise ValueError("Canonical V2 turn penalties must all be zero")
    adapter = payload["optional_adapters"]["geometry_turn_penalty_v1"]
    if (
        float(adapter["right_turn_s"]),
        float(adapter["left_turn_s"]),
        float(adapter["u_turn_s"]),
    ) != (3.0, 8.0, 20.0):
        raise ValueError("geometry_turn_penalty_v1 must remain 3/8/20 seconds")
    return payload
