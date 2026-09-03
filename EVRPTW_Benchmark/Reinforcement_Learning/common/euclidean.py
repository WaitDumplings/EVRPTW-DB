from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from evrptw_core.schema import EVRPTWInstance


EUCLIDEAN_MANIFEST_SCHEMA = "drl_euclidean_calibration_manifest_v1"


def haversine_matrix_km(coordinates_lon_lat: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates_lon_lat, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (n, 2) in longitude/latitude order")
    lon = np.deg2rad(coordinates[:, 0])
    lat = np.deg2rad(coordinates[:, 1])
    dlon = lon[None, :] - lon[:, None]
    dlat = lat[None, :] - lat[:, None]
    value = np.sin(dlat / 2.0) ** 2 + (
        np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    distance = 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))
    np.fill_diagonal(distance, 0.0)
    return distance.astype(np.float32)


def load_euclidean_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema") != EUCLIDEAN_MANIFEST_SCHEMA:
        raise ValueError("Euclidean calibration manifest schema mismatch")
    speeds = manifest.get("speed_kmh_by_day_type", {})
    if set(speeds) != {"weekday", "weekend"}:
        raise ValueError("Euclidean manifest must freeze weekday/weekend speeds")
    if any(not np.isfinite(float(value)) or float(value) <= 0.0 for value in speeds.values()):
        raise ValueError("Euclidean speeds must be finite and positive")
    if manifest.get("calibration_split") != "train":
        raise ValueError("Euclidean speeds must be calibrated on train only")
    return manifest


def euclidean_instance(
    instance: EVRPTWInstance,
    manifest: dict[str, Any],
) -> EVRPTWInstance:
    day_type = str(instance.day_type)
    speed_kmh = float(manifest["speed_kmh_by_day_type"][day_type])
    coordinates = np.vstack(
        [
            np.asarray(instance.depot).reshape(1, 2),
            np.asarray(instance.customers),
            np.asarray(instance.charging_stations),
        ]
    )
    distance = haversine_matrix_km(coordinates)
    travel_time = (distance / (speed_kmh / 3600.0)).astype(np.float32)
    rate = float(
        instance.vehicle.get(
            "specific_energy_consumption_kwh_per_km",
            instance.vehicle.get("consumption_kwh_per_km", 0.404),
        )
    )
    energy = (distance * rate).astype(np.float32)
    customer_count = instance.num_customers
    station_start = 1 + customer_count
    cs_to_depot = travel_time[station_start:, 0].astype(np.float32, copy=True)
    raw = dict(instance.raw)
    raw.update(
        {
            "distance_matrix_km": distance,
            "running_time_shortest_matrix_s": travel_time,
            "running_time_path_distance_km": distance,
            "running_time_path_energy_kwh": energy,
            "distance_path_travel_time_s": travel_time,
            "full_cs_to_depot_time_s": cs_to_depot,
        }
    )
    metadata = dict(instance.metadata)
    metadata.update(
        {
            "training_representation": "E",
            "euclidean_calibration_schema": manifest["schema"],
            "euclidean_speed_day_type": day_type,
            "euclidean_speed_kmh": speed_kmh,
        }
    )
    return replace(
        instance,
        distance_matrix_km=distance,
        raw_travel_time_matrix_s=travel_time,
        ev_transition_time_matrix_s=travel_time,
        shortest_time_matrix_s=travel_time,
        energy_matrix_kwh=energy,
        cs_time_to_depot_s=cs_to_depot,
        speed_profile={
            "matrix_source": "euclidean_training_representation",
            "effective_speed_kmh": speed_kmh,
        },
        metadata=metadata,
        raw=raw,
    )


__all__ = [
    "EUCLIDEAN_MANIFEST_SCHEMA",
    "euclidean_instance",
    "haversine_matrix_km",
    "load_euclidean_manifest",
]
