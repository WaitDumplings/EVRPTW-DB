from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
ALNS_ROOT = REPO_ROOT / "EVRPTW_Benchmark" / "MetaHeuristics" / "ALNS_Solver"
SOLVER_PATH = ALNS_ROOT / "solver.py"
sys.path.insert(0, str(ALNS_ROOT))

# This is the source hash of the pre-optimization solver in commit a5c4505.
# HEAD comparison is useful while reviewing the optimization, but it is only a
# pre-commit aid: after the optimization is committed, HEAD no longer contains
# this oracle.  The golden tests below remain active in source archives and in
# repositories whose HEAD has advanced.
PREOPTIMIZATION_SOLVER_SHA256 = (
    "11971ed930e69eca495e6b3911161df63a54c17154349db1a098419f297ebca5"
)
SEARCH_GOLDEN_SHA256 = (
    "d6989acedefdd8c816574acce3bf4814ad418ebea87d81b1b0661d7fae753ab2"
)
OPERATOR_GOLDEN_SHA256 = (
    "a419ac5d0efa53dc1261a4e64e7df782ed21e089194303c35493b9ea1c3fbf4f"
)


def _load_current_solver_module():
    spec = importlib.util.spec_from_file_location("alns_optimized_test_module", SOLVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_head_solver_module():
    """Load the pre-optimization HEAD solver when it is still available."""

    relative = SOLVER_PATH.relative_to(REPO_ROOT).as_posix()
    try:
        source = subprocess.check_output(
            ["git", "show", f"HEAD:{relative}"],
            cwd=REPO_ROOT,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"pre-optimization ALNS HEAD oracle is unavailable: {exc}")
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if source_sha256 != PREOPTIMIZATION_SOLVER_SHA256:
        pytest.skip(
            "HEAD no longer contains the pre-optimization ALNS oracle; "
            "the persistent golden tests cover the committed solver"
        )
    module = types.ModuleType("alns_checked_in_oracle")
    module.__file__ = f"HEAD:{relative}"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _tensor_instance(n_customers: int = 8, n_stations: int = 3) -> dict:
    n = 1 + n_customers + n_stations
    distance = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            distance[i, j] = (
                0.0 if i == j else abs(i - j) + 0.01 * i + 0.001 * j
            )
    travel = distance * 1.7
    energy = distance * 0.05
    return {
        "depot": np.array([[0.0, 0.0]], dtype=np.float64),
        "customers": np.array(
            [[float(i), float(i % 3)] for i in range(1, n_customers + 1)],
            dtype=np.float64,
        ),
        "charging_stations": np.array(
            [[float(n_customers + i), -1.0] for i in range(1, n_stations + 1)],
            dtype=np.float64,
        ),
        "customer_demand": np.ones(n_customers, dtype=np.float64),
        "customer_service": np.full(n_customers, 30.0, dtype=np.float64),
        "tw": np.array([[0.0, 36_000.0]] * n_customers, dtype=np.float64),
        "distance_matrix_km": distance,
        "time_matrix_min": travel,
        "energy_matrix_kwh": energy,
        "charging_power_kw": np.array([11.0, 50.0, 100.0], dtype=np.float64)[
            :n_stations
        ],
        "charging_efficiency": 0.9,
        "env": {
            "instance_startTime": 0.0,
            "instance_endTime": 36_000.0,
            "battery_capacity": 100.0,
            # Keep the mechanical-optimization oracle away from the exact
            # capacity boundary.  That boundary has its own regression test
            # below, so a correctness fix there does not invalidate unrelated
            # route/RNG golden trajectories.
            "loading_capacity": 3.25,
            "consumption_per_distance": 0.05,
            "charging_speed": float("inf"),
            "speed": 1.0,
        },
    }


def _new_solver(module, seed: int = 71):
    solver = module.ALNS_Solver(_tensor_instance(), seed=seed, format="tensor")
    # Exercise every operator family in a short deterministic run.
    solver.max_iters = 35
    solver.NSR = 7
    solver.NC = 4
    solver.NS = 7
    solver.NRR = 9
    solver.nRR = 2
    return solver


def _golden_normalize(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, dict):
        return [
            [_golden_normalize(key), _golden_normalize(item)]
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        ]
    if isinstance(value, (list, tuple)):
        return [_golden_normalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_golden_normalize(item) for item in value), key=repr)
    return value


def _golden_sha256(value) -> str:
    payload = json.dumps(
        _golden_normalize(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _search_state_payload(solver, result) -> dict:
    return {
        "result": result,
        "cur_iter": solver.cur_iter,
        "global_value": solver.global_value,
        "temperature": solver.temperature,
        "terminated_by_time_limit": solver.terminated_by_time_limit,
        "current_routes": solver.current_routes,
        "best_routes": solver.best_routes,
        "visited": solver.visited,
        "weights": {
            "cr": solver.cr_weights,
            "ci": solver.ci_weights,
            "sr": solver.sr_weights,
            "si": solver.si_weights,
        },
        "scores": {
            "cr": solver.cr_scores,
            "ci": solver.ci_scores,
            "sr": solver.sr_scores,
            "si": solver.si_scores,
        },
        "uses": {
            "cr": solver.cr_uses,
            "ci": solver.ci_uses,
            "sr": solver.sr_uses,
            "si": solver.si_uses,
        },
        "attribute_frequency": dict(solver.attribute_frequency),
        "attribute_total": solver.attribute_total,
        "rng_state": solver.rng.getstate(),
        "station_indices": solver.station_indices,
        "station_charge_minutes_per_kwh": solver.station_charge_minutes_per_kwh,
    }


def _operator_payload(module) -> dict:
    shaw_routes = [
        [0, 1, 2, 0],
        [0, 3, 1, 4, 0],
        [0, 5, 6, 7, 8, 0],
    ]
    shaw = {}
    for family in ("shaw", "proximity", "demand", "time"):
        solver = _new_solver(module, seed=931)
        output = solver._cr_shaw_family(
            [list(route) for route in shaw_routes], family
        )
        shaw[family] = {"output": output, "rng_state": solver.rng.getstate()}

    insertion_routes = [[0, 2, 3, 0], [0, 4, 5, 0], [0, 6, 7, 8, 0]]
    insertion = {}
    for mode in ("distance", "time"):
        for allowed_routes in (None, {1}):
            solver = _new_solver(module)
            options = solver._all_customer_insertions(
                [list(route) for route in insertion_routes],
                1,
                mode=mode,
                allowed_routes=allowed_routes,
            )
            key = f"{mode}:{'all' if allowed_routes is None else 'route1'}"
            insertion[key] = {
                "options": options,
                "rng_state": solver.rng.getstate(),
            }
    return {"shaw": shaw, "insertion": insertion}


def _assert_search_state_equal(reference, optimized) -> None:
    scalar_fields = (
        "cur_iter",
        "global_value",
        "temperature",
        "terminated_by_time_limit",
        "attribute_total",
    )
    mapping_fields = (
        "cr_weights",
        "ci_weights",
        "sr_weights",
        "si_weights",
        "cr_scores",
        "ci_scores",
        "sr_scores",
        "si_scores",
        "cr_uses",
        "ci_uses",
        "sr_uses",
        "si_uses",
        "attribute_frequency",
    )
    for name in scalar_fields:
        assert getattr(optimized, name) == getattr(reference, name), name
    for name in mapping_fields:
        assert dict(getattr(optimized, name)) == dict(getattr(reference, name)), name
    assert optimized.current_routes == reference.current_routes
    assert optimized.best_routes == reference.best_routes
    assert optimized.visited == reference.visited
    assert optimized.rng.getstate() == reference.rng.getstate()


def test_fixed_iteration_search_matches_persistent_golden_state() -> None:
    """Permanent oracle that works after commit and without a Git checkout."""

    current_module = _load_current_solver_module()
    solver = _new_solver(current_module)
    with contextlib.redirect_stdout(io.StringIO()):
        result = solver.solve(delta_iters=35)

    assert solver.algorithm_profile_id == "alns_stage2_scalable_v2"
    assert result == [1, 2, 0, 4, 6, 5, 0, 8, 7, 3, 0]
    assert solver.global_value.hex() == "0x1.032b020c49ba6p+5"
    assert _golden_sha256(_search_state_payload(solver, result)) == SEARCH_GOLDEN_SHA256


def test_operator_order_and_rng_match_persistent_golden_state() -> None:
    """Protect Shaw/removal and insertion ordering when Git is unavailable."""

    current_module = _load_current_solver_module()
    assert _golden_sha256(_operator_payload(current_module)) == OPERATOR_GOLDEN_SHA256


def test_fixed_iteration_search_is_reproducible_for_scalable_profile() -> None:
    current_module = _load_current_solver_module()
    reference = _new_solver(current_module)
    repeated = _new_solver(current_module)

    with contextlib.redirect_stdout(io.StringIO()):
        reference_result = reference.solve(delta_iters=35)
        repeated_result = repeated.solve(delta_iters=35)

    assert repeated_result == reference_result
    _assert_search_state_equal(reference, repeated)


def test_route_structure_rejects_internal_depot_and_invalid_terminal() -> None:
    solver = _new_solver(_load_current_solver_module())
    assert not solver.is_route_feasible([0, 1, 0, 2, 0])
    assert not solver.is_route_feasible([0, 1, len(solver.nodes), 0])


@pytest.mark.parametrize("family", ["shaw", "proximity", "demand", "time"])
def test_shaw_family_preserves_first_route_and_rng_semantics(family: str) -> None:
    current_module = _load_current_solver_module()
    oracle_module = _load_head_solver_module()
    routes = [
        [0, 1, 2, 0],
        [0, 3, 1, 4, 0],  # duplicate customer 1 exercises first-route semantics
        [0, 5, 6, 7, 8, 0],
    ]
    reference = _new_solver(oracle_module, seed=931)
    optimized = _new_solver(current_module, seed=931)

    expected = reference._cr_shaw_family([list(route) for route in routes], family)
    actual = optimized._cr_shaw_family([list(route) for route in routes], family)

    assert actual == expected
    assert optimized.rng.getstate() == reference.rng.getstate()


@pytest.mark.parametrize("mode", ["distance", "time"])
@pytest.mark.parametrize("allowed_routes", [None, {1}])
def test_scalable_insertion_enumeration_is_deterministic_feasible_and_exact(
    mode: str,
    allowed_routes: set[int] | None,
) -> None:
    current_module = _load_current_solver_module()
    routes = [[0, 2, 3, 0], [0, 4, 5, 0], [0, 6, 7, 8, 0]]
    first = _new_solver(current_module)
    repeated = _new_solver(current_module)

    actual = first._all_customer_insertions(
        [list(route) for route in routes],
        1,
        mode=mode,
        allowed_routes=allowed_routes,
    )
    repeated_actual = repeated._all_customer_insertions(
        [list(route) for route in routes],
        1,
        mode=mode,
        allowed_routes=allowed_routes,
    )

    assert actual == repeated_actual
    assert first.rng.getstate() == repeated.rng.getstate()
    assert actual
    for route_index, candidate, delta in actual:
        assert first.is_route_feasible(candidate)
        assert candidate.count(1) == 1
        if route_index == len(routes):
            expected_delta = (
                first._route_total_time(candidate)
                if mode == "time"
                else first._route_distance(candidate)
            )
        else:
            assert allowed_routes is None or route_index in allowed_routes
            metric = first._route_total_time if mode == "time" else first._route_distance
            expected_delta = metric(candidate) - metric(routes[route_index])
        assert delta == pytest.approx(expected_delta, abs=1e-12)


def test_candidate_evaluation_scans_solution_feasibility_once() -> None:
    module = _load_current_solver_module()
    solver = _new_solver(module)
    candidate = [[0, 1, 2, 0], [0, 3, 4, 0], [0, 5, 6, 0], [0, 7, 8, 0]]
    solver.global_value = float("inf")
    calls = 0
    original = solver.is_solution_feasible

    def counted(routes):
        nonlocal calls
        calls += 1
        return original(routes)

    solver.is_solution_feasible = counted
    accepted, reward, candidate_distance = solver._evaluate_candidate(
        candidate, current_distance=float("inf")
    )

    assert accepted
    assert reward == solver.r1
    assert candidate_distance == sum(solver._route_distance(route) for route in candidate)
    assert calls == 1


def test_cached_immutable_helpers_match_recomputation() -> None:
    module = _load_current_solver_module()
    solver = _new_solver(module)

    assert solver.station_indices == sorted(solver.station_index_set)
    assert solver.station_index_set == set(solver.station_indices)
    assert solver.max_distance == float(solver.dist_matrix.max())
    assert solver.customer_zone_map == solver._build_zone_map(solver.customer_indices)


def _multi_hop_tensor_instance(*, with_certificate: bool = False) -> dict:
    # Canonical tensor order: depot 0, customer 1, stations 2 and 3.  Customer
    # 1 cannot be reached directly or through only one station; the feasible
    # singleton route is 0->2->3->1->0.
    n = 4
    distance = np.full((n, n), 100.0, dtype=np.float64)
    energy = np.full((n, n), 100.0, dtype=np.float64)
    travel = np.full((n, n), 10.0, dtype=np.float64)
    np.fill_diagonal(distance, 0.0)
    np.fill_diagonal(energy, 0.0)
    np.fill_diagonal(travel, 0.0)
    for origin, destination, value in (
        (0, 1, 30.0),
        (0, 2, 6.0),
        (2, 1, 15.0),
        (2, 3, 6.0),
        (3, 1, 4.0),
        (1, 0, 4.0),
    ):
        energy[origin, destination] = value
        distance[origin, destination] = value
    instance = {
        "depot": np.array([[0.0, 0.0]], dtype=np.float64),
        "customers": np.array([[1.0, 0.0]], dtype=np.float64),
        "charging_stations": np.array([[2.0, 0.0], [3.0, 0.0]], dtype=np.float64),
        "customer_demand": np.array([1.0], dtype=np.float64),
        "customer_service": np.array([0.0], dtype=np.float64),
        "tw": np.array([[0.0, 100_000.0]], dtype=np.float64),
        "distance_matrix_km": distance,
        "time_matrix_min": travel,
        "energy_matrix_kwh": energy,
        "charging_power_kw": np.array([50.0, 100.0], dtype=np.float64),
        "charging_efficiency": 1.0,
        "env": {
            "instance_startTime": 0.0,
            "instance_endTime": 100_000.0,
            "battery_capacity": 10.0,
            "loading_capacity": 10.0,
            "consumption_per_distance": 1.0,
            "charging_speed": float("inf"),
            "speed": 1.0,
        },
    }
    if with_certificate:
        instance["certificate_singleton_routes"] = [[0, 2, 3, 1, 0]]
    return instance


def test_progressive_station_repair_supports_multiple_charging_visits() -> None:
    module = _load_current_solver_module()
    solver = module.ALNS_Solver(
        _multi_hop_tensor_instance(), seed=7, format="tensor"
    )

    route = solver._make_single_customer_route(1)

    assert route == [0, 2, 3, 1, 0]
    assert solver.is_route_feasible(route)
    assert solver._simulate_route(route)["feasible"]


def test_replayed_certificate_singletons_are_normalized_by_customer() -> None:
    module = _load_current_solver_module()
    instance = _multi_hop_tensor_instance(with_certificate=True)
    # Input order is irrelevant; on a one-customer fixture this also documents
    # the canonical route representation expected from the shared adapter.
    solver = module.ALNS_Solver(instance, seed=7, format="tensor")

    routes = solver._construct_singleton_solution()

    assert routes == [[0, 2, 3, 1, 0]]
    assert solver.singleton_source == "stage2_certificate_replayed"
    assert solver.is_solution_feasible(routes)


def test_invalid_certificate_is_never_trusted_or_published() -> None:
    module = _load_current_solver_module()
    instance = _multi_hop_tensor_instance()
    instance["certificate_singleton_routes"] = [[0, 1, 0]]
    solver = module.ALNS_Solver(instance, seed=7, format="tensor")

    routes = solver._construct_singleton_solution()

    assert routes == [[0, 2, 3, 1, 0]]
    assert solver.singleton_source == "solver_repair"
    assert solver.is_solution_feasible(routes)


def test_constructor_budget_falls_back_only_to_complete_solution() -> None:
    module = _load_current_solver_module()
    solver = module.ALNS_Solver(
        _tensor_instance(n_customers=100), seed=2026, format="tensor"
    )
    solver.initial_construction_budget_s = 0.0
    observed: list[list[list[int]]] = []

    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve(
            delta_iters=0,
            time_limit_s=60.0,
            incumbent_callback=lambda _elapsed, _objective, routes: observed.append(
                [list(route) for route in routes]
            ),
        )

    assert observed
    assert all(solver.is_solution_feasible(routes) for routes in observed)
    assert len(
        {
            customer
            for route in observed[0]
            for customer in route
            if customer in solver.customer_to_mask
        }
    ) == solver.n_customers
    assert solver.initial_construction_stats["budget_exhausted"] is True


def test_route_at_exact_vehicle_capacity_is_feasible() -> None:
    module = _load_current_solver_module()
    instance = _tensor_instance(n_customers=1)
    capacity = float(instance["env"]["loading_capacity"])
    instance["customer_demand"] = np.asarray([capacity], dtype=float)
    solver = module.ALNS_Solver(instance, seed=2026, format="tensor")

    assert solver.is_route_feasible([0, 1, 0])

    solver.nodes[1]["demand"] = capacity + 2e-9
    solver._demand_cache.clear()
    assert not solver.is_route_feasible([0, 1, 0])


def test_memoization_cache_is_bounded() -> None:
    module = _load_current_solver_module()
    solver = _new_solver(module)
    cache: dict[int, int] = {}

    for value in range(20):
        solver._cache_store(cache, value, value, limit=4)

    assert len(cache) <= 4
    assert cache[19] == 19


@pytest.mark.parametrize(
    "route",
    [
        [0, 1, 2, 0],
        [0, 1, 2, 3, 0],
        [0, 1, 2, 3, 4, 0],
        [0, 1, 2, 3, 4, 5, 0],
    ],
)
def test_compact_feasibility_matches_detailed_simulation(route: list[int]) -> None:
    module = _load_current_solver_module()
    solver = _new_solver(module)

    solver._clear_caches()
    compact_first = solver._is_route_schedule_feasible(route)
    detailed_after = solver._simulate_route(route)["feasible"]
    solver._clear_caches()
    detailed_first = solver._simulate_route(route)["feasible"]
    compact_after = solver._is_route_schedule_feasible(route)

    assert compact_first == detailed_after == detailed_first == compact_after
