"""Family-level directed road speeds and speed-sensitive reference energy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .planning import derive_seed


def _group_log_multiplier(
    values: pd.Series,
    *,
    sigma: float,
    seed: int,
    namespace: str,
) -> np.ndarray:
    if sigma <= 0.0:
        return np.ones(len(values), dtype=float)
    labels = values.fillna("__missing__").astype(str)
    unique = sorted(labels.unique())
    rng = np.random.default_rng(derive_seed(seed, namespace))
    draws = np.exp(rng.normal(-0.5 * sigma**2, sigma, size=len(unique)))
    mapping = dict(zip(unique, draws))
    return labels.map(mapping).to_numpy(dtype=float)


def _speed_sensitive_consumption(
    speed_kph: np.ndarray,
    energy_config: Mapping[str, Any],
) -> np.ndarray:
    reference = float(energy_config["reference_consumption_kwh_per_km"])
    reference_speed = float(energy_config["reference_speed_kph"])
    rolling = float(energy_config["rolling_share"])
    aerodynamic = float(energy_config["aerodynamic_share"])
    auxiliary = float(energy_config["auxiliary_share"])
    ratio = np.maximum(speed_kph, 1e-6) / reference_speed
    return reference * (rolling + aerodynamic * ratio**2 + auxiliary / ratio)


def auxiliary_power_kw(energy_config: Mapping[str, Any]) -> float:
    """Auxiliary load implied by the reference energy anchor."""

    return (
        float(energy_config["reference_consumption_kwh_per_km"])
        * float(energy_config["auxiliary_share"])
        * float(energy_config["reference_speed_kph"])
    )


def build_family_road_state(
    directed_speeds: pd.DataFrame,
    *,
    day_type: str,
    road_state_seed: int,
    profile: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "edge_u",
        "edge_v",
        "edge_key",
        "edge_id",
        "physical_segment_id",
        "corridor_id",
        "direction_id",
        "length_m",
        "operating_mode",
        "legal_speed_kph",
        "reference_speed_kph",
    }
    missing = required - set(directed_speeds.columns)
    if missing:
        raise ValueError(f"Directed-speed layer is missing columns: {sorted(missing)}")
    cfg = profile["road_state"]
    mode_cfg = cfg["factor_by_day_and_mode"][day_type]
    frame = directed_speeds.copy().reset_index(drop=True)
    factors = np.empty(len(frame), dtype=float)
    mode_baselines: dict[str, float] = {}
    for mode in ("H", "M", "U"):
        mask = frame["operating_mode"].astype(str).eq(mode).to_numpy()
        if not mask.any():
            continue
        values = mode_cfg[mode]
        rng = np.random.default_rng(derive_seed(road_state_seed, "mode", mode))
        baseline = float(
            np.clip(
                rng.normal(float(values["mean"]), float(values["std"])),
                float(values["min"]),
                float(values["max"]),
            )
        )
        mode_baselines[mode] = baseline
        factors[mask] = baseline
    unknown_modes = sorted(set(frame["operating_mode"].astype(str)) - {"H", "M", "U"})
    if unknown_modes:
        raise ValueError(f"Unsupported operating modes: {unknown_modes}")

    factors *= _group_log_multiplier(
        frame["corridor_id"],
        sigma=float(cfg["corridor_log_sigma"]),
        seed=road_state_seed,
        namespace="corridor",
    )
    factors *= _group_log_multiplier(
        frame["physical_segment_id"],
        sigma=float(cfg["physical_segment_log_sigma"]),
        seed=road_state_seed,
        namespace="physical_segment",
    )
    # Edge IDs are directional, so this component creates reproducible A->B / B->A variation.
    factors *= _group_log_multiplier(
        frame["edge_id"],
        sigma=float(cfg["direction_log_sigma"]),
        seed=road_state_seed,
        namespace="directed_edge",
    )
    for mode in ("H", "M", "U"):
        mask = frame["operating_mode"].astype(str).eq(mode).to_numpy()
        values = mode_cfg[mode]
        factors[mask] = np.clip(
            factors[mask], float(values["min"]), float(values["max"])
        )
    reference = pd.to_numeric(frame["reference_speed_kph"], errors="coerce").to_numpy()
    legal = pd.to_numeric(frame["legal_speed_kph"], errors="coerce").to_numpy()
    if not np.isfinite(reference).all() or not np.isfinite(legal).all():
        raise ValueError("CLE speed layer contains missing/non-finite reference or legal speeds")
    instance_speed = np.minimum(legal, reference * factors)
    instance_speed = np.maximum(instance_speed, float(cfg["minimum_speed_kph"]))
    length_m = pd.to_numeric(frame["length_m"], errors="coerce").to_numpy()
    frame["day_type"] = day_type
    frame["road_state_factor"] = factors.astype(np.float32)
    frame["instance_speed_kph"] = instance_speed.astype(np.float32)
    frame["edge_travel_time_s"] = (length_m / (instance_speed / 3.6)).astype(np.float32)
    consumption = _speed_sensitive_consumption(instance_speed, profile["energy"])
    frame["edge_energy_kwh_per_km"] = consumption.astype(np.float32)
    frame["edge_energy_kwh"] = (consumption * length_m / 1000.0).astype(np.float32)

    directional = frame.groupby("physical_segment_id")["instance_speed_kph"].agg(
        ["count", "min", "max"]
    )
    comparable = directional.loc[directional["count"] >= 2]
    asymmetric = (comparable["max"] - comparable["min"]) > 1e-6
    report = {
        "schema": "cle_evrptw_family_road_state_report_v1",
        "model_id": str(cfg["model_id"]),
        "day_type": day_type,
        "road_state_seed": int(road_state_seed),
        "directed_edge_count": len(frame),
        "mode_baseline_factors": mode_baselines,
        "speed_kph_min": float(instance_speed.min()),
        "speed_kph_median": float(np.median(instance_speed)),
        "speed_kph_max": float(instance_speed.max()),
        "legal_cap_binding_fraction": float(np.mean(instance_speed >= legal - 1e-6)),
        "comparable_bidirectional_physical_segment_count": len(comparable),
        "asymmetric_speed_physical_segment_fraction": (
            float(asymmetric.mean()) if len(asymmetric) else 0.0
        ),
        "energy_model_id": str(profile["energy"]["model_id"]),
        "auxiliary_power_kw": auxiliary_power_kw(profile["energy"]),
    }
    return frame, report


def connector_costs(
    length_m: float,
    *,
    profile: Mapping[str, Any],
) -> tuple[float, float, float]:
    """Return connector distance km, time s, and energy kWh."""

    speed = float(profile["road_state"]["connector_reference_speed_kph"])
    distance_km = float(length_m) / 1000.0
    time_s = distance_km / speed * 3600.0
    consumption = float(
        _speed_sensitive_consumption(np.asarray([speed]), profile["energy"])[0]
    )
    return distance_km, time_s, consumption * distance_km
