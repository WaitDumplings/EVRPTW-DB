from __future__ import annotations

import numpy as np

from evrptw_core.schema import EVRPTWInstance
from EVRPTW_Benchmark.Reinforcement_Learning.common.euclidean import (
    EUCLIDEAN_MANIFEST_SCHEMA,
    euclidean_instance,
    haversine_matrix_km,
)


def _instance() -> EVRPTWInstance:
    coordinates = np.array([[-118.25, 34.05], [-118.24, 34.06], [-118.23, 34.04]])
    graph = np.array([[0, 5, 8], [7, 0, 4], [9, 6, 0]], dtype=np.float32)
    return EVRPTWInstance.from_dict(
        {
            "instance_id": "v1",
            "day_type": "weekday",
            "working_start_s": 0,
            "working_end_s": 36000,
            "depot": coordinates[0],
            "customers": coordinates[1:2],
            "charging_stations": coordinates[2:],
            "distance_matrix_km": graph,
            "demands_cm3": [1],
            "package_counts": [1],
            "service_time_s": [30],
            "tw_s": [[0, 36000]],
            "cs_time_to_depot_s": [100],
            "vehicle": {"specific_energy_consumption_kwh_per_km": 0.5},
            "shortest_time_matrix_s": graph * 100,
            "energy_matrix_kwh": graph * 0.5,
            "charging_policy": {},
        }
    )


def test_haversine_matrix_is_symmetric_with_zero_diagonal() -> None:
    matrix = haversine_matrix_km(np.array([[-118.25, 34.05], [-118.24, 34.06]]))
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    assert matrix[0, 1] > 0.0


def test_euclidean_instance_replaces_distance_time_energy_only() -> None:
    source = _instance()
    manifest = {
        "schema": EUCLIDEAN_MANIFEST_SCHEMA,
        "speed_kmh_by_day_type": {"weekday": 30.0, "weekend": 35.0},
    }
    converted = euclidean_instance(source, manifest)
    assert converted.instance_id == source.instance_id
    assert np.allclose(converted.distance_matrix_km, converted.distance_matrix_km.T)
    assert np.allclose(converted.shortest_time_matrix_s, converted.distance_matrix_km / (30 / 3600))
    assert np.allclose(converted.energy_matrix_kwh, converted.distance_matrix_km * 0.5)
    assert converted.metadata["training_representation"] == "E"
