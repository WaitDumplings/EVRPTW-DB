from __future__ import annotations

from typing import Any

import numpy as np

from EVRPTW_Benchmark.Exact.Gurobi_Solver.route_validator import validate_routes
from .objective import resolve_objective, route_dispatch_count
from .action_constraints import ACTION_CONSTRAINT_CONTRACT_ID, consecutive_cs_arcs


def _verify_drl_routes(instance, routes) -> dict[str, Any]:
    verification = dict(validate_routes(instance, routes))
    forbidden_arcs = consecutive_cs_arcs(instance, routes)
    physical_passed = bool(verification["passed"])
    verification.update(
        physical_verifier_passed=physical_passed,
        drl_policy_passed=not forbidden_arcs,
        action_constraint_contract_id=ACTION_CONSTRAINT_CONTRACT_ID,
        passed=physical_passed and not forbidden_arcs,
    )
    if forbidden_arcs:
        verification["violations"] = list(verification.get("violations", [])) + [
            f"DRL action constraint forbids consecutive CS visits: route {route}, {origin}->{destination}"
            for route, origin, destination in forbidden_arcs
        ]
    return verification


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
        verification = _verify_drl_routes(instance, routes)
        if verification["passed"]:
            return int(selected), routes, verification

    candidates = np.flatnonzero(served == served.max())
    selected = int(candidates[np.argmin(objective[candidates])])
    routes = info["routes"][selected]
    return selected, routes, _verify_drl_routes(instance, routes)


def select_min_verified_objective(
    instance,
    info: dict[str, Any],
    objective_config=None,
) -> tuple[int, list[list[int]], dict[str, Any]]:
    """Feasibility first, then the active objective, recomputed from exported routes.

    In the cost track, candidate ordering must not trust an environment's route
    count or an old distance-only ranking. Replaying the matrix sums is cheap;
    the full resource verifier is still called in sorted order until one passes.
    """
    config = resolve_objective(
        objective_config if objective_config is not None else info.get("objective_config")
    )
    if (objective_config is not None and info.get("objective_config") is not None
            and resolve_objective(info["objective_config"]).to_dict() != config.to_dict()):
        raise ValueError("candidate/environment objective config does not match evaluation objective")
    if not config.is_cost:
        selected, routes, verification = select_min_verified_distance(instance, info)
        verification = dict(verification)
        verification.update(config.fields(verification["objective_distance_km"], route_dispatch_count(routes)))
        return selected, routes, verification

    success = np.asarray(info["success"], dtype=bool)
    served = np.asarray(info["served_customers"], dtype=np.int64)
    matrix = np.asarray(instance.distance_matrix_km, dtype=np.float64)
    scores = np.full(success.shape, np.inf, dtype=np.float64)
    for index in np.flatnonzero(success):
        routes = info["routes"][int(index)]
        distance = 0.0
        for route in routes:
            nodes = np.asarray(route, dtype=np.int64)
            if np.any(nodes < 0) or np.any(nodes >= matrix.shape[0]):
                distance = np.inf
                break
            distance += float(matrix[nodes[:-1], nodes[1:]].sum())
        scores[index] = config.value(distance, route_dispatch_count(routes))
    successful = np.flatnonzero(success)
    for index in successful[np.argsort(scores[successful], kind="stable")]:
        routes = info["routes"][int(index)]
        verification = _verify_drl_routes(instance, routes)
        if verification["passed"]:
            verification = dict(verification)
            verification.update(config.fields(verification["objective_distance_km"], route_dispatch_count(routes)))
            return int(index), routes, verification

    # Failed candidates remain failures even if route export virtually appends
    # a depot return. Their incurred accounting still includes open dispatches.
    candidates = np.flatnonzero(served == served.max())
    distances = np.asarray(info["objective_distance_km"], dtype=np.float64)
    dispatches = (
        np.asarray(info["vehicles_started"], dtype=np.int64)
        if "vehicles_started" in info else np.asarray([
            route_dispatch_count(routes) for routes in info["routes"]
        ], dtype=np.int64)
    )
    incurred = config.value(distances, dispatches)
    selected = int(candidates[np.argmin(incurred[candidates])])
    routes = info["routes"][selected]
    verification = _verify_drl_routes(instance, routes)
    verification["route_verifier_passed"] = bool(verification["passed"])
    verification["passed"] = False
    verification["violations"] = list(verification.get("violations", []))
    verification["violations"].append("candidate did not complete successfully in the environment")
    actual_distance = float(distances[selected])
    actual_dispatches = int(dispatches[selected])
    verification.update(config.fields(actual_distance, actual_dispatches))
    verification["objective_distance_km"] = actual_distance
    return selected, routes, verification


__all__ = ["select_min_verified_distance", "select_min_verified_objective"]
