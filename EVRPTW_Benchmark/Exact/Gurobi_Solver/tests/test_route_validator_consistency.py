from __future__ import annotations

import unittest

import numpy as np

from evrptw_core.schema import EVRPTWInstance
from route_validator import (
    validate_routes as validate_exact_routes,
)
from benchmark_common import (
    validate_routes as validate_metaheuristic_routes,
)


def _instance() -> EVRPTWInstance:
    """Two customers and two chargers with generous resource bounds.

    All non-diagonal arcs consume 1 kWh over 1 km in 60 seconds. Chargers
    3 and 4 have effective powers of 18 and 36 kW respectively, so each
    replenished kWh takes 200 or 100 seconds. Loose constraints ensure
    malformed routes fail because of their structure, not their resources.
    """
    distance = np.ones((5, 5), dtype=np.float32)
    np.fill_diagonal(distance, 0.0)
    travel = distance * 60.0
    return EVRPTWInstance.from_dict(
        {
            "instance_id": "route_structure_consistency",
            "working_start_s": 0,
            "working_end_s": 10_000,
            "depot": [0.0, 0.0],
            "customers": [[1.0, 0.0], [2.0, 0.0]],
            "charging_stations": [[3.0, 0.0], [4.0, 0.0]],
            "distance_matrix_km": distance,
            "demands_cm3": [1.0, 1.0],
            "package_counts": [1, 1],
            "service_time_s": [10.0, 10.0],
            "tw_s": [[0.0, 9_000.0], [0.0, 9_000.0]],
            "cs_time_to_depot_s": [60.0, 60.0],
            "vehicle": {
                "battery_capacity_kwh": 10.0,
                "cargo_capacity_cm3": 10.0,
            },
            "running_time_shortest_matrix_s": travel,
            "running_time_path_energy_kwh": distance.copy(),
            "charging_power_kw": [36.0, 72.0],
            "charging_policy": {"charging_power_derating_factor": 0.5},
        }
    )


class RouteValidatorConsistencyTests(unittest.TestCase):
    def test_rejects_invalid_structure_with_complete_customer_coverage(self) -> None:
        instance = _instance()
        invalid_cases = [
            ([[0, 1, 0, 2, 0]], "internal depot visit"),
            ([[0, 1, 2, 0], [0, 3, 0]], "contains no customer"),
        ]
        validators = [
            ("exact_and_rl", validate_exact_routes),
            ("metaheuristic", validate_metaheuristic_routes),
        ]
        for routes, expected_violation in invalid_cases:
            # Both candidates already serve every customer exactly once. In
            # particular, a charger-only route cannot hide behind missing coverage.
            self.assertEqual(
                sorted(node for route in routes for node in route if 1 <= node <= 2),
                [1, 2],
            )
            for name, validate in validators:
                with self.subTest(validator=name, violation=expected_violation):
                    result = validate(instance, routes)
                    self.assertIs(result["passed"], False)
                    self.assertTrue(
                        any(
                            expected_violation in message
                            for message in result["violations"]
                        ),
                        result["violations"],
                    )

    def test_valid_routes_have_consistent_recomputed_metrics(self) -> None:
        instance = _instance()
        valid_cases = [
            ("separate_vehicle_routes", [[0, 1, 0], [0, 2, 0]], 4.0, 0, 0.0),
            ("same_vehicle_station_revisit", [[0, 3, 1, 3, 2, 0]], 5.0, 2, 600.0),
            (
                "different_vehicle_station_revisit",
                [[0, 3, 1, 0], [0, 3, 2, 0]],
                6.0,
                2,
                400.0,
            ),
            ("different_station_powers", [[0, 3, 1, 4, 2, 0]], 5.0, 2, 400.0),
        ]
        for name, routes, distance_km, charging_visits, charging_time_s in valid_cases:
            with self.subTest(case=name):
                results = [
                    validate_exact_routes(instance, routes),
                    validate_metaheuristic_routes(instance, routes),
                ]
                for result in results:
                    self.assertIs(result["passed"], True, result["violations"])
                    self.assertEqual(result["violations"], [])
                    self.assertAlmostEqual(result["objective_distance_km"], distance_km)
                    self.assertEqual(result["charging_visit_count"], charging_visits)
                    self.assertAlmostEqual(
                        result["total_charging_time_s"], charging_time_s
                    )

                for metric in (
                    "objective_distance_km",
                    "charging_visit_count",
                    "total_charging_time_s",
                ):
                    self.assertAlmostEqual(results[0][metric], results[1][metric])


if __name__ == "__main__":
    unittest.main()
