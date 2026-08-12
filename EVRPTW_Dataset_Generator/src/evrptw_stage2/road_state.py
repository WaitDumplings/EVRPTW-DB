"""Family-level directed road speeds for the U.S. reference adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .planning import derive_seed


def specific_energy_consumption_kwh_per_km(
    profile: Mapping[str, Any],
) -> float:
    """Return the frozen constant-distance energy coefficient ``h``."""

    return float(profile["energy"]["specific_energy_consumption_kwh_per_km"])


def build_family_road_state(
    directed_speeds: pd.DataFrame,
    *,
    day_type: str,
    road_state_seed: int,
    profile: Mapping[str, Any],
    moves_road_type_baseline_factors: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "edge_u",
        "edge_v",
        "edge_key",
        "edge_id",
        "physical_segment_id",
        "length_m",
        "operating_mode",
        "legal_speed_kph",
        "reference_speed_kph",
    }
    missing = required - set(directed_speeds.columns)
    if missing:
        raise ValueError(f"Directed-speed layer is missing columns: {sorted(missing)}")
    cfg = profile["road_state"]
    moves_cfg = cfg["factor_by_day_and_moves_road_type"][day_type]
    mode_to_moves = {str(k): str(v) for k, v in cfg["mode_to_moves_road_type"].items()}
    frame = directed_speeds.copy().reset_index(drop=True)
    factors = np.empty(len(frame), dtype=float)
    road_type_baselines: dict[str, float] = {}
    modes = frame["operating_mode"].astype(str)
    unknown_modes = sorted(set(modes) - set(mode_to_moves))
    if unknown_modes:
        raise ValueError(f"Unsupported operating modes: {unknown_modes}")
    moves_types = modes.map(mode_to_moves)
    expected_moves_types = set(map(str, moves_types.unique()))
    supplied_baselines = (
        None
        if moves_road_type_baseline_factors is None
        else {str(key): float(value) for key, value in moves_road_type_baseline_factors.items()}
    )
    if supplied_baselines is not None:
        missing_baselines = expected_moves_types - set(supplied_baselines)
        extra_baselines = set(supplied_baselines) - expected_moves_types
        if missing_baselines or extra_baselines:
            raise ValueError(
                "Stored MOVES road-type baselines do not match the CLE road-state modes; "
                f"missing={sorted(missing_baselines)}, extra={sorted(extra_baselines)}"
            )
        if not all(np.isfinite(value) for value in supplied_baselines.values()):
            raise ValueError("Stored MOVES road-type baselines must be finite")
    for moves_road_type in sorted(expected_moves_types):
        mask = moves_types.eq(moves_road_type).to_numpy()
        values = moves_cfg[moves_road_type]
        if supplied_baselines is None:
            rng = np.random.default_rng(
                derive_seed(road_state_seed, "moves_road_type", moves_road_type)
            )
            baseline = float(
                np.clip(
                    rng.normal(float(values["mean"]), float(values["std"])),
                    float(values["min"]),
                    float(values["max"]),
                )
            )
        else:
            baseline = supplied_baselines[moves_road_type]
            if not float(values["min"]) <= baseline <= float(values["max"]):
                raise ValueError(
                    f"Stored baseline {baseline} for {moves_road_type!r} is outside "
                    f"the profile range [{values['min']}, {values['max']}]"
                )
        road_type_baselines[moves_road_type] = baseline
        factors[mask] = baseline
    for moves_road_type in sorted(moves_types.unique()):
        mask = moves_types.eq(moves_road_type).to_numpy()
        values = moves_cfg[moves_road_type]
        factors[mask] = np.clip(
            factors[mask], float(values["min"]), float(values["max"])
        )
    reference = pd.to_numeric(frame["reference_speed_kph"], errors="coerce").to_numpy()
    legal = pd.to_numeric(frame["legal_speed_kph"], errors="coerce").to_numpy()
    if not np.isfinite(reference).all() or not np.isfinite(legal).all():
        raise ValueError("CLE speed layer contains missing/non-finite reference or legal speeds")
    if np.any(legal <= 0.0) or np.any(reference <= 0.0):
        raise ValueError("CLE legal and reference speeds must be positive")
    instance_speed = np.minimum(
        legal,
        np.maximum(reference * factors, float(cfg["minimum_speed_kph"])),
    )
    length_m = pd.to_numeric(frame["length_m"], errors="coerce").to_numpy()
    frame["day_type"] = day_type
    frame["moves_road_type"] = moves_types
    frame["road_state_factor"] = factors.astype(np.float32)
    frame["instance_speed_kph"] = instance_speed.astype(np.float32)
    frame["edge_travel_time_s"] = (length_m / (instance_speed / 3.6)).astype(np.float32)

    directional = frame.groupby("physical_segment_id")["instance_speed_kph"].agg(
        ["count", "min", "max"]
    )
    comparable = directional.loc[directional["count"] >= 2]
    asymmetric = (comparable["max"] - comparable["min"]) > 1e-6
    report = {
        "schema": "cle_evrptw_family_road_state_report_v2",
        "model_id": str(cfg["model_id"]),
        "day_type": day_type,
        "road_state_seed": int(road_state_seed),
        "directed_edge_count": len(frame),
        "mode_to_moves_road_type": mode_to_moves,
        "moves_road_type_baseline_factors": road_type_baselines,
        "baseline_factor_source": (
            "stored_family_manifest"
            if supplied_baselines is not None
            else "deterministic_seed_sampling"
        ),
        "factor_granularity": "day_type_x_moves_road_type",
        "additional_random_edge_factors": False,
        "speed_kph_min": float(instance_speed.min()),
        "speed_kph_median": float(np.median(instance_speed)),
        "speed_kph_max": float(instance_speed.max()),
        "legal_cap_binding_fraction": float(np.mean(instance_speed >= legal - 1e-6)),
        "comparable_bidirectional_physical_segment_count": len(comparable),
        "asymmetric_speed_physical_segment_fraction": (
            float(asymmetric.mean()) if len(asymmetric) else 0.0
        ),
        "energy_model_id": str(profile["energy"]["model_id"]),
        "specific_energy_consumption_kwh_per_km": (
            specific_energy_consumption_kwh_per_km(profile)
        ),
    }
    return frame, report


def connector_costs(
    length_m: float,
    *,
    profile: Mapping[str, Any],
) -> tuple[float, float]:
    """Return connector distance km and time s under the fixed U profile."""

    speed = float(profile["road_state"]["connector_reference_speed_kph"])
    distance_km = float(length_m) / 1000.0
    time_s = distance_km / speed * 3600.0
    return distance_km, time_s
