from __future__ import annotations

from typing import Any

import numpy as np

from EVRPTW_Benchmark.Exact.Gurobi_Solver.route_validator import validate_routes


def select_min_verified_distance(
    instance,
    info: dict[str, Any],
) -> tuple[int, list[list[int]], dict[str, Any]]:
    """Return the shortest environment-successful candidate that verifies."""

    success = np.asarray(info["success"], dtype=bool)
    objective = np.asarray(info["objective_distance_km"], dtype=np.float64)
    served = np.asarray(info["served_customers"], dtype=np.int64)
    successful = np.flatnonzero(success)
    for selected in successful[np.argsort(objective[successful])]:
        routes = info["routes"][int(selected)]
        verification = validate_routes(instance, routes)
        if verification["passed"]:
            return int(selected), routes, verification

    candidates = np.flatnonzero(served == served.max())
    selected = int(candidates[np.argmin(objective[candidates])])
    routes = info["routes"][selected]
    return selected, routes, validate_routes(instance, routes)


__all__ = ["select_min_verified_distance"]
