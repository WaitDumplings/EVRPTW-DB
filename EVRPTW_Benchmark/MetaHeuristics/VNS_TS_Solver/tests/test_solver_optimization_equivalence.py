from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
META_ROOT = REPO_ROOT / "EVRPTW_Benchmark" / "MetaHeuristics"
SOLVER_ROOT = META_ROOT / "VNS_TS_Solver"
SOLVER_PATH = SOLVER_ROOT / "solver.py"

# HEAD is an optional review-time oracle only.  These persistent goldens keep
# validating the optimized solver after it is committed or distributed without
# the repository's .git directory.
PREOPTIMIZATION_SOLVER_SHA256 = (
    "f83815fc4e4599aa427df716faf08843828ecde9bbb795f90c21e25de64d5d38"
)
CANDIDATE_GOLDEN_SHA256 = (
    "e21564238cd2594489d54c4d92f24ff435e55b6192b22d28e651af629d60865c"
)
SEARCH_GOLDEN_SHA256 = (
    "1bf78b7951748d69fdf0241afbfa44047bf4b67c017c7eccd746d54ff3291339"
)

for path in (
    SOLVER_ROOT,
    META_ROOT,
    REPO_ROOT / "EVRPTW_Core",
    REPO_ROOT / "EVRPTW_Dataset_Generator" / "src",
):
    sys.path.insert(0, str(path))

from vnst_adapter import Customer, Depot, Route, Station, VNSTInstance  # noqa: E402
from vnst_adapter import to_vnst_instance  # noqa: E402
from evrptw_core.schema import EVRPTWInstance  # noqa: E402


def _load_current_solver_class():
    spec = importlib.util.spec_from_file_location("vnst_solver_optimized", SOLVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VNSTSolver


def _load_head_solver_class():
    """Load the pre-optimization HEAD solver when it is still available."""
    try:
        source = subprocess.check_output(
            [
                "git",
                "show",
                "HEAD:EVRPTW_Benchmark/MetaHeuristics/VNS_TS_Solver/solver.py",
            ],
            cwd=REPO_ROOT,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"pre-optimization VNS-TS HEAD oracle is unavailable: {exc}")
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if source_sha256 != PREOPTIMIZATION_SOLVER_SHA256:
        pytest.skip(
            "HEAD no longer contains the pre-optimization VNS-TS oracle; "
            "the persistent golden tests cover the committed solver"
        )
    old_signature = "    def _candidate_moves_fast(self, solution):\n"
    assert source.count(old_signature) == 1
    source = source.replace(
        old_signature,
        "    def _ranked_candidate_moves_fast(self, solution):\n",
        1,
    )
    old_return = (
        "        return [move for _, move in moves]\n\n"
        "    def _apply_fast_move(self, solution, move):\n"
    )
    assert source.count(old_return) == 1
    source = source.replace(
        old_return,
        "        return moves\n\n"
        "    def _candidate_moves_fast(self, solution):\n"
        "        return [move for _, move in self._ranked_candidate_moves_fast(solution)]\n\n"
        "    def _apply_fast_move(self, solution, move):\n",
        1,
    )
    module = types.ModuleType("vnst_solver_head_oracle")
    module.__file__ = "<git-show-HEAD:VNS_TS_Solver/solver.py>"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module.VNSTSolver


CurrentSolver = _load_current_solver_class()


def _make_instance() -> VNSTInstance:
    """Canonical ids with legacy [depot, stations, customers] matrix order."""
    instance = VNSTInstance()
    instance.depot = Depot(0, "d", 0.0, 0.0, 0.0, 0.0, 2_000.0, 0.0)
    instance.customers = [
        Customer(node_id, "c", float(node_id), 0.0, 1.0, 0.0, 2_000.0, 1.0)
        for node_id in range(1, 7)
    ]
    instance.stations = [
        Station(7, "f", 7.0, 0.0, 0.0, 0.0, 2_000.0, 0.0),
        Station(8, "f", 8.0, 0.0, 0.0, 0.0, 2_000.0, 0.0),
    ]
    instance.vehicle_params = {
        "fuel_cap": 100.0,
        "load_cap": 4.0,
        "consump_rate": 1.0,
        "velocity": 1.0,
        "charging_efficiency": 1.0,
    }
    instance.station_charging_power_kw = {7: 11.0, 8: 100.0}

    size = 9
    canonical_distance = np.empty((size, size), dtype=np.float64)
    for origin in range(size):
        for destination in range(size):
            canonical_distance[origin, destination] = (
                0.0
                if origin == destination
                else float(((origin * 3 + destination * 5) % 4) + 1)
            )
    canonical_time = np.ones((size, size), dtype=np.float64)
    np.fill_diagonal(canonical_time, 0.0)
    canonical_energy = np.ones((size, size), dtype=np.float64)
    np.fill_diagonal(canonical_energy, 0.0)
    # Both stations receive exactly 20 kWh of spent energy on these routes.
    canonical_energy[0, 1] = 10.0
    canonical_energy[1, 7] = 10.0
    canonical_energy[1, 8] = 10.0

    legacy_order = [0, 7, 8, 1, 2, 3, 4, 5, 6]
    legacy_index = np.ix_(legacy_order, legacy_order)
    instance.terminal_order = legacy_order
    instance.dist_matrix = canonical_distance[legacy_index]
    instance.time_matrix = canonical_time[legacy_index]
    instance.energy_matrix = canonical_energy[legacy_index]
    return instance


def _solution(instance: VNSTInstance) -> list[Route]:
    depot = instance.depot
    customers = instance.customers
    fast_station = instance.stations[1]
    return [
        Route([depot, customers[0], fast_station, customers[1], customers[2], depot]),
        Route([depot, customers[3], customers[4], customers[5], depot]),
    ]


def _clone_solution(solution: list[Route]) -> list[Route]:
    return [Route(list(route.nodes)) for route in solution]


def _normalized(value):
    if hasattr(value, "id"):
        return ("node", int(value.id))
    if isinstance(value, tuple):
        return tuple(_normalized(item) for item in value)
    return value


def _ranked_signature(ranked):
    return [(float(proxy).hex(), _normalized(move)) for proxy, move in ranked]


def _route_ids(solution):
    return [[int(node.id) for node in route.nodes] for route in solution]


def _configure(solver):
    solver.move_candidate_limit = 80
    solver.route_neighbor_limit = 2
    solver.position_neighbor_limit = 3
    solver.exchange_neighbor_limit = 4
    solver.station_candidate_limit = 2
    solver.tabu_iter = 4
    solver.global_value = 1e10
    solver.global_solution = None
    return solver


def _golden_normalize(value):
    if isinstance(value, np.generic):
        value = value.item()
    if hasattr(value, "id"):
        return {"node_id": int(value.id), "node_type": str(value.type)}
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


def _candidate_golden_payload() -> dict:
    instance = _make_instance()
    solution = _solution(instance)
    solver = _configure(CurrentSolver(instance))
    nodes = [instance.depot, *instance.customers, *instance.stations]
    terminal_metrics = [
        (
            int(origin.id),
            int(destination.id),
            solver._dist(origin, destination),
            float(solver.time_cost(origin, destination)),
            solver.fuel_consumption(origin, destination),
        )
        for origin in nodes
        for destination in nodes
    ]
    ranked = solver._ranked_candidate_moves_fast(_clone_solution(solution))
    customer = instance.customers[0]
    slow = Route([instance.depot, customer, instance.stations[0], instance.depot])
    fast = Route([instance.depot, customer, instance.stations[1], instance.depot])
    return {
        "terminal_metrics": terminal_metrics,
        "ranked_candidates": _ranked_signature(ranked),
        "candidate_moves": [
            _normalized(move)
            for move in solver._candidate_moves_fast(_clone_solution(solution))
        ],
        "slow_time_violation": solver.time_violation(slow),
        "fast_time_violation": solver.time_violation(fast),
        "slow_charge_s": solver.charging_time(instance.stations[0], 20.0),
        "fast_charge_s": solver.charging_time(instance.stations[1], 20.0),
        "station_power_kw": instance.station_charging_power_kw,
    }


def _run_fixed_outer_search(solver_class=CurrentSolver):
    instance = _make_instance()
    solver = _configure(solver_class(instance, predefine_route_number=2))
    setattr(solver, "η_feas", 3)
    setattr(solver, "η_dist", 3)
    solver.tabu_iter = 2
    solver.k_max = 3
    random.seed(731)
    result = solver.solve()
    return solver, result, random.getstate()


def _search_golden_payload(solver, result, rng_state) -> dict:
    return {
        "result": _route_ids(result),
        "global_solution": _route_ids(solver.global_solution),
        "global_value": solver.global_value,
        "rng_state": rng_state,
        "attribute_frequency": dict(solver.attribute_frequency),
        "attribute_total": solver.attribute_total,
        "penalty_weights": (solver.alpha, solver.beta, solver.gamma),
        "temperature": solver.temp,
        "tabu_list": list(solver.tabu_list),
        "station_reinsert_tabu": solver.StationReIn_tabu_list,
    }


def test_candidate_order_matrix_contract_and_power_are_deterministic() -> None:
    first = _candidate_golden_payload()
    second = _candidate_golden_payload()
    assert _golden_sha256(first) == _golden_sha256(second)
    assert first["slow_time_violation"] is True
    assert first["fast_time_violation"] is False
    assert first["slow_charge_s"] > first["fast_charge_s"]


def test_fixed_budget_search_is_reproducible() -> None:
    first = _run_fixed_outer_search()
    second = _run_fixed_outer_search()
    assert _golden_sha256(_search_golden_payload(*first)) == _golden_sha256(
        _search_golden_payload(*second)
    )
    assert first[0].is_solution_feasible(first[0].global_solution)


def test_canonical_matrix_lookup_and_dual_power_match_source_contract() -> None:
    instance = _make_instance()
    new = _configure(CurrentSolver(instance))
    nodes = [instance.depot, *instance.customers, *instance.stations]
    legacy_index = {
        int(terminal_id): position
        for position, terminal_id in enumerate(instance.terminal_order)
    }

    assert new._direct_terminal_index
    for origin in nodes:
        for destination in nodes:
            row = legacy_index[int(origin.id)]
            column = legacy_index[int(destination.id)]
            assert new._dist(origin, destination).hex() == float(
                instance.dist_matrix[row, column]
            ).hex()
            assert float(new.time_cost(origin, destination)).hex() == float(
                instance.time_matrix[row, column]
            ).hex()
            assert new.fuel_consumption(origin, destination).hex() == float(
                instance.energy_matrix[row, column]
            ).hex()

    customer = instance.customers[0]
    slow = Route([instance.depot, customer, instance.stations[0], instance.depot])
    fast = Route([instance.depot, customer, instance.stations[1], instance.depot])
    assert new.time_violation(slow) is True
    assert new.time_violation(fast) is False
    assert new.charging_time(instance.stations[0], 20.0) > new.charging_time(
        instance.stations[1], 20.0
    )


def test_ranked_fast_candidates_are_deterministic_unique_and_bounded() -> None:
    instance = _make_instance()
    solution = _solution(instance)
    new = _configure(CurrentSolver(instance))

    first = new._ranked_candidate_moves_fast(_clone_solution(solution))
    second = new._ranked_candidate_moves_fast(_clone_solution(solution))
    assert _ranked_signature(first) == _ranked_signature(second)
    assert [proxy for proxy, _ in first] == sorted(proxy for proxy, _ in first)
    assert len(first) <= new._effective_fast_limits(len(instance.customers))[-1]
    assert len({_normalized(move) for _, move in first}) == len(first)
    assert any(move[0] == "relocate" for _, move in first)
    assert any(move[0] == "exchange" for _, move in first)


def test_iteration_local_route_cache_preserves_cost_and_reuses_routes() -> None:
    instance = _make_instance()
    new = _configure(CurrentSolver(instance))
    solution = _solution(instance)
    moves = new._candidate_moves_fast(_clone_solution(solution))[:20]
    candidates = [
        new._apply_fast_move(_clone_solution(solution), move) for move in moves
    ]
    assert all(candidate is not None for candidate in candidates)
    expected_values = [
        new.generalized_cost(candidate, False, False, False)
        for candidate in candidates
    ]
    cache = {}

    original = new.is_route_feasible
    route_checks = 0

    def counted(route):
        nonlocal route_checks
        route_checks += 1
        return original(route)

    new.is_route_feasible = counted
    for new_candidate, expected_value in zip(candidates, expected_values):
        assert new_candidate is not None
        new_value = new.generalized_cost(
            new_candidate,
            False,
            False,
            False,
            route_feasibility_cache=cache,
        )
        assert float(new_value).hex() == float(expected_value).hex()

    assert route_checks < len(moves) * len(solution)
    assert cache


def test_fixed_iteration_fast_tabu_is_reproducible_and_feasible() -> None:
    instance = _make_instance()
    initial = _solution(instance)
    first = _configure(CurrentSolver(instance))
    second = _configure(CurrentSolver(instance))

    random.seed(20260811)
    first_result = first._tabu_search_fast(_clone_solution(initial))
    first_rng_state = random.getstate()

    random.seed(20260811)
    second_result = second._tabu_search_fast(_clone_solution(initial))
    second_rng_state = random.getstate()

    assert second_rng_state == first_rng_state
    assert _route_ids(second_result) == _route_ids(first_result)
    assert float(second.global_value).hex() == float(first.global_value).hex()
    assert _route_ids(second.global_solution) == _route_ids(first.global_solution)
    assert second.attribute_total == first.attribute_total
    assert dict(second.attribute_frequency) == dict(first.attribute_frequency)
    assert second.is_solution_feasible(second.global_solution)


def test_fixed_budget_outer_solve_is_reproducible() -> None:
    instance = _make_instance()
    first = _configure(CurrentSolver(instance, predefine_route_number=2))
    second = _configure(CurrentSolver(instance, predefine_route_number=2))
    for solver in (first, second):
        setattr(solver, "η_feas", 3)
        setattr(solver, "η_dist", 3)
        solver.tabu_iter = 2
        solver.k_max = 3

    random.seed(731)
    first_result = first.solve()
    first_rng_state = random.getstate()

    random.seed(731)
    second_result = second.solve()
    second_rng_state = random.getstate()

    assert second_rng_state == first_rng_state
    assert _route_ids(second_result) == _route_ids(first_result)
    assert float(second.global_value).hex() == float(first.global_value).hex()


def test_duplicate_incumbent_callback_is_suppressed_but_improvement_is_reported(
    monkeypatch,
) -> None:
    instance = _make_instance()
    solver = _configure(CurrentSolver(instance))
    solver._solve_start = 1.0
    initial = _solution(instance)
    improved = _clone_solution(initial)
    improved[0].nodes[1], improved[1].nodes[1] = (
        improved[1].nodes[1],
        improved[0].nodes[1],
    )
    solver.global_solution = initial
    monkeypatch.setattr("time.perf_counter", lambda: 2.0)
    events = []
    solver._incumbent_callback = lambda elapsed, objective, routes: events.append(
        (objective, routes)
    )

    solver.global_value = 100.0
    solver._report_incumbent()
    solver._report_incumbent()
    solver.global_solution = improved
    solver.global_value = 20.0
    solver._report_incumbent()
    solver._report_incumbent()

    assert [objective for objective, _ in events] == [21.0, 17.0]


def test_feasibility_boundary_uses_consistent_numerical_tolerance() -> None:
    instance = _make_instance()
    solver = _configure(CurrentSolver(instance))
    depot = instance.depot
    customer = instance.customers[0]

    exact_load = customer._replace(
        demand=float(instance.vehicle_params["load_cap"]),
        due=1.0,
    )
    route = Route([depot, exact_load, depot])
    # Make both arcs land exactly on the fuel boundary and the customer arrival
    # exactly on its due time.
    origin = solver.node_id[depot.id]
    target = solver.node_id[exact_load.id]
    solver.energy_matrix[origin, target] = instance.vehicle_params["fuel_cap"]
    solver.energy_matrix[target, origin] = 0.0
    solver.time_matrix[origin, target] = 1.0
    solver.time_matrix[target, origin] = 0.0

    assert solver.is_route_feasible(route)

    over_load = exact_load._replace(
        demand=float(instance.vehicle_params["load_cap"]) + 2e-9
    )
    assert solver.load_violation(Route([depot, over_load, depot]))
    solver.energy_matrix[origin, target] = (
        float(instance.vehicle_params["fuel_cap"]) + 2e-9
    )
    assert solver.battery_violation(route)
    solver.energy_matrix[origin, target] = instance.vehicle_params["fuel_cap"]
    solver.time_matrix[origin, target] = 1.0 + 2e-9
    assert solver.time_violation(route)


def test_diversification_history_is_bounded_and_deterministic() -> None:
    instance = _make_instance()
    first = _configure(CurrentSolver(instance))
    second = _configure(CurrentSolver(instance))
    first._history_max_entries = second._history_max_entries = 2
    solution = _solution(instance)
    for _ in range(10):
        first.update_diversification_history(solution)
        second.update_diversification_history(solution)
    assert len(first.attribute_frequency) <= 6
    assert dict(first.attribute_frequency) == dict(second.attribute_frequency)
    assert first.attribute_total == second.attribute_total


def test_singleton_warm_start_is_feasible_and_reported_before_search(monkeypatch) -> None:
    instance = _make_instance()
    solver = _configure(CurrentSolver(instance))
    warm_start = solver.singleton_warm_start()
    assert warm_start is not None
    assert solver.is_solution_feasible(warm_start)
    assert len(warm_start) == len(instance.customers)

    events = []
    monkeypatch.setattr(solver, "initial_solution", lambda: _solution(instance))
    solver.eta_feas = 0
    solver.eta_dist = 0
    solver.solve(incumbent_callback=lambda elapsed, objective, routes: events.append(routes))
    assert events
    assert len(events[0]) == len(instance.customers)


def test_certificate_singleton_warm_start_supports_multi_hop_charging() -> None:
    instance = _make_instance()
    # One customer uses a station hop. The remaining direct routes keep the
    # whole six-customer witness complete and independently solver-feasible.
    instance.certificate_singleton_routes = [
        [0, 1, 8, 0],
        [0, 2, 0],
        [0, 3, 0],
        [0, 4, 0],
        [0, 5, 0],
        [0, 6, 0],
    ]
    solver = _configure(CurrentSolver(instance))
    warm_start = solver.singleton_warm_start()
    assert warm_start is not None
    assert _route_ids(warm_start)[0] == [0, 1, 8, 0]
    assert solver.is_solution_feasible(warm_start)


def test_route_structure_requires_canonical_depot_boundaries() -> None:
    instance = _make_instance()
    solver = _configure(CurrentSolver(instance))
    depot = instance.depot
    customer = instance.customers[0]
    station = instance.stations[0]

    assert solver.is_route_feasible(Route([depot, customer, depot]))
    assert not solver.is_route_feasible(Route([]))
    assert not solver.is_route_feasible(Route([depot, depot]))
    assert not solver.is_route_feasible(Route([customer, depot, depot]))
    assert not solver.is_route_feasible(Route([depot, customer, station]))
    assert not solver.is_route_feasible(Route([depot, station, depot]))
    assert not solver.is_route_feasible(Route([depot, customer, depot, depot]))
    assert not solver.is_solution_feasible([])


def test_time_penalty_propagates_lateness_without_rewinding_clock() -> None:
    instance = _make_instance()
    first = instance.customers[0]._replace(due=0.0, service=1.0)
    second = instance.customers[1]._replace(due=0.0, service=1.0)
    solver = _configure(CurrentSolver(instance))
    route = Route([instance.depot, first, second, instance.depot])

    # Travel is one second on every off-diagonal arc: first arrives at 1 and
    # second at 3 after service, so cumulative lateness is 1 + 3.
    assert solver.time_penalty(route) == 4.0


def test_malformed_fast_candidate_cannot_become_global_incumbent(monkeypatch) -> None:
    instance = _make_instance()
    solver = _configure(CurrentSolver(instance))
    solver.tabu_iter = 1
    valid = _solution(instance)
    malformed = _clone_solution(valid)
    malformed[0].nodes.pop()

    monkeypatch.setattr(
        solver,
        "_ranked_candidate_moves_fast",
        lambda solution: [(0.0, ("malformed",))],
    )
    monkeypatch.setattr(
        solver,
        "_apply_fast_move",
        lambda solution, move: _clone_solution(malformed),
    )
    result = solver._tabu_search_fast(_clone_solution(valid))

    assert solver.is_solution_feasible(result)
    assert solver.is_solution_feasible(solver.global_solution)
    assert _route_ids(solver.global_solution) == _route_ids(valid)
    assert solver.global_value == solver.generalized_cost(
        valid,
        penalty_value=False,
        p_div_value=False,
        allow_infeasible=False,
    )


def test_singleton_best_fit_is_complete_feasible_and_deterministic() -> None:
    instance = _make_instance()
    first = _configure(CurrentSolver(instance))
    second = _configure(CurrentSolver(instance))
    first.initial_construction_budget_s = second.initial_construction_budget_s = 5.0

    first_singleton = first.singleton_warm_start()
    second_singleton = second.singleton_warm_start()
    first_result = first._consolidate_singleton_solution(first_singleton)
    second_result = second._consolidate_singleton_solution(second_singleton)

    assert first.is_solution_feasible(first_result)
    assert second.is_solution_feasible(second_result)
    assert _route_ids(first_result) == _route_ids(second_result)
    assert len(first_result) < len(first_singleton)
    assert first.initial_construction_stats["result_route_count"] == len(first_result)
    assert first.initial_construction_stats["merged_customer_count"] > 0


def test_singleton_best_fit_timeout_returns_complete_fallback() -> None:
    instance = _make_instance()
    solver = _configure(CurrentSolver(instance))
    singleton = solver.singleton_warm_start()
    solver.initial_construction_budget_s = 0.0

    result = solver._consolidate_singleton_solution(singleton)

    assert solver.is_solution_feasible(result)
    assert len(result) == len(singleton)
    assert solver.initial_construction_stats["budget_exhausted"] is True
    assert solver.initial_construction_stats["merged_customer_count"] == 0


def test_singleton_best_fit_accepts_multi_hop_certificate_routes() -> None:
    instance = _make_instance()
    instance.certificate_singleton_routes = [
        [0, 7, 8, 1, 0],
        [0, 2, 0],
        [0, 3, 0],
        [0, 4, 0],
        [0, 5, 0],
        [0, 6, 0],
    ]
    solver = _configure(CurrentSolver(instance))
    solver.initial_construction_budget_s = 5.0
    singleton = solver.singleton_warm_start()

    result = solver._consolidate_singleton_solution(singleton)

    assert solver.singleton_source == "stage2_certificate_replayed"
    assert solver.is_solution_feasible(result)
    assert {node.id for route in result for node in route.nodes if node.type == "c"} == {
        customer.id for customer in instance.customers
    }


def test_effective_fast_policy_is_exposed_and_scale_adaptive() -> None:
    solver = _configure(CurrentSolver(_make_instance()))
    small = solver.effective_fast_policy(50)
    medium = solver.effective_fast_policy(100)
    large = solver.effective_fast_policy(500)
    assert small["version"] == "adaptive_nearest_best_fit_v3"
    assert small["move_candidate_limit"] >= medium["move_candidate_limit"]
    assert medium["move_candidate_limit"] >= large["move_candidate_limit"]
    assert large["route_neighbor_limit"] <= small["route_neighbor_limit"]


@pytest.mark.parametrize("customer_count", [1, 50, 100, 499, 500, 2000])
def test_preexecution_contract_exactly_matches_runtime_fast_policy(
    customer_count: int,
    monkeypatch,
) -> None:
    fake_solver_module = types.ModuleType("solver")
    fake_solver_module.VNSTSolver = CurrentSolver
    monkeypatch.setitem(sys.modules, "solver", fake_solver_module)
    runner_spec = importlib.util.spec_from_file_location(
        f"vns_contract_policy_{customer_count}", SOLVER_ROOT / "run_vns_ts.py"
    )
    assert runner_spec is not None and runner_spec.loader is not None
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)

    solver = CurrentSolver(
        _make_instance(),
        move_candidate_limit=40,
        route_neighbor_limit=4,
        position_neighbor_limit=4,
        exchange_neighbor_limit=6,
        station_candidate_limit=5,
    )
    runtime = solver.effective_fast_policy(customer_count)
    contract = runner.contract_effective_fast_policy(
        customer_count=customer_count,
        move_candidate_limit=40,
        route_neighbor_limit=4,
        position_neighbor_limit=4,
        exchange_neighbor_limit=6,
        station_candidate_limit=5,
    )
    assert contract == {
        key: runtime[key]
        for key in (
            "version",
            "move_candidate_limit",
            "route_neighbor_limit",
            "position_neighbor_limit",
            "exchange_neighbor_limit",
            "station_candidate_limit",
        )
    }
    assert runner.contract_algorithm_profile_id("fast") == (
        "vns_ts_stage2_adaptive_fast_v4"
    )
    assert runner.contract_algorithm_profile_id("full") == (
        "vns_ts_stage2_full_enumeration_v3"
    )


def test_reported_global_objective_is_recomputed_from_paired_routes() -> None:
    solver = _configure(CurrentSolver(_make_instance()))
    routes = _solution(solver.instance)
    expected = solver.generalized_cost(
        routes,
        penalty_value=False,
        p_div_value=False,
        allow_infeasible=False,
    )
    observed = []
    solver.global_solution = solver.clone_solution_shallow(routes)
    solver.global_value = expected + 123.0
    solver._solve_start = 0.0
    solver._incumbent_callback = (
        lambda elapsed, value, route_ids: observed.append((value, route_ids))
    )

    solver._report_incumbent()

    assert solver.global_value == expected
    assert observed[0][0] == expected
    assert observed[0][1] == [
        [node.id for node in route.nodes] for route in routes
    ]


def test_neighbor_cutoff_uses_stable_customer_order_for_exact_ties() -> None:
    instance = _make_instance()
    instance.dist_matrix.fill(1.0)
    np.fill_diagonal(instance.dist_matrix, 0.0)
    solver = CurrentSolver(instance, route_neighbor_limit=1, exchange_neighbor_limit=1)
    customer_ids = [customer.id for customer in instance.customers]
    for customer_id in customer_ids:
        expected = tuple(other for other in customer_ids if other != customer_id)
        assert solver._customer_neighbor_ids[customer_id] == expected


def test_stage2_adapter_keeps_canonical_matrices_without_permutation() -> None:
    distance = np.arange(16, dtype=np.float64).reshape(4, 4)
    travel = distance + 100.0
    energy = distance + 200.0
    canonical = EVRPTWInstance.from_dict(
        {
            "instance_id": "canonical_adapter_test",
            "region_id": "test",
            "mother_board_id": "test",
            "operating_day_id": "test",
            "day_type": "weekday",
            "working_start_s": 0.0,
            "working_end_s": 10_000.0,
            "depot": [0.0, 0.0],
            "customers": [[1.0, 0.0], [2.0, 0.0]],
            "charging_stations": [[3.0, 0.0]],
            "distance_matrix_km": distance,
            "demands_cm3": [1.0, 1.0],
            "package_counts": [1, 1],
            "service_time_s": [0.0, 0.0],
            "tw_s": [[0.0, 10_000.0], [0.0, 10_000.0]],
            "cs_time_to_depot_s": [0.0],
            "vehicle": {
                "battery_capacity_kwh": 1_000.0,
                "cargo_capacity_cm3": 100.0,
                "charging_efficiency": 1.0,
            },
            "running_time_shortest_matrix_s": travel,
            "running_time_path_energy_kwh": energy,
            "charging_power_kw": [100.0],
            "charging_policy": {"charging_efficiency": 1.0},
        }
    )
    adapted = to_vnst_instance(canonical)
    solver = CurrentSolver(adapted)
    assert adapted.terminal_order == [0, 1, 2, 3]
    np.testing.assert_array_equal(solver.dist_matrix, distance)
    np.testing.assert_array_equal(solver.time_matrix, travel)
    np.testing.assert_array_equal(solver.energy_matrix, energy)


def test_vns_runner_discards_structurally_invalid_callback_before_recording(
    monkeypatch,
) -> None:
    """The independent canonical replay remains the final publication gate."""

    canonical = EVRPTWInstance.from_dict(
        {
            "instance_id": "vns-invalid-callback-gate",
            "region_id": "test",
            "mother_board_id": "test",
            "operating_day_id": "test",
            "day_type": "weekday",
            "working_start_s": 0.0,
            "working_end_s": 10_000.0,
            "depot": [0.0, 0.0],
            "customers": [[1.0, 0.0]],
            "charging_stations": [],
            "distance_matrix_km": [[0.0, 1.0], [1.0, 0.0]],
            "demands_cm3": [1.0],
            "package_counts": [1],
            "service_time_s": [0.0],
            "tw_s": [[0.0, 10_000.0]],
            "cs_time_to_depot_s": [],
            "vehicle": {
                "battery_capacity_kwh": 100.0,
                "cargo_capacity_cm3": 100.0,
                "charging_efficiency": 1.0,
            },
            "running_time_shortest_matrix_s": [[0.0, 1.0], [1.0, 0.0]],
            "running_time_path_energy_kwh": [[0.0, 1.0], [1.0, 0.0]],
            "charging_power_kw": [],
            "charging_policy": {"charging_efficiency": 1.0},
        }
    )

    class FakeSolver:
        def __init__(self, _instance, **kwargs):
            self.terminated_by_time_limit = False
            self.search_mode = kwargs["search_mode"]
            for key, value in kwargs.items():
                setattr(self, key, value)

        def solve(self, *, time_limit_s, incumbent_callback):  # noqa: ARG002
            incumbent_callback(0.0, 0.0, [[1, 0, 0]])
            incumbent_callback(0.0, 0.0, [[0, 1, 0]])

        def effective_fast_policy(self):
            return {
                "version": "test-policy",
                "move_candidate_limit": self.move_candidate_limit,
                "route_neighbor_limit": self.route_neighbor_limit,
                "position_neighbor_limit": self.position_neighbor_limit,
                "exchange_neighbor_limit": self.exchange_neighbor_limit,
                "station_candidate_limit": self.station_candidate_limit,
            }

    fake_solver_module = types.ModuleType("solver")
    fake_solver_module.VNSTSolver = FakeSolver
    fake_adapter_module = types.ModuleType("vnst_adapter")
    fake_adapter_module.to_vnst_instance = lambda instance: instance
    monkeypatch.setitem(sys.modules, "solver", fake_solver_module)
    monkeypatch.setitem(sys.modules, "vnst_adapter", fake_adapter_module)
    runner_spec = importlib.util.spec_from_file_location(
        "vns_invalid_callback_runner", SOLVER_ROOT / "run_vns_ts.py"
    )
    assert runner_spec is not None and runner_spec.loader is not None
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    monkeypatch.setattr(
        runner,
        "load_input_task",
        lambda task: (
            canonical,
            {
                "file": "/tmp/view_index.parquet",
                "family_id": "mf-test",
                "city_slug": "test",
                "split_id": "test",
                "track_id": "core",
                "scale_id": "Cus1",
            },
        ),
    )
    monkeypatch.setattr(
        runner,
        "validate_instance_structure",
        lambda instance: types.SimpleNamespace(success=True, errors=[]),
    )
    monkeypatch.setattr(
        runner,
        "charging_profile",
        lambda instance: (np.asarray([], dtype=np.float32), 1.0, "none"),
    )
    task = {
        "seed": 7,
        "time_limit_s": 1.0,
        "checkpoints_s": (1.0,),
        "predefine_route_number": 5,
        "eta_feas": 0,
        "eta_dist": 0,
        "eta_dist_requested": 0,
        "search_budget_mode": "fixed_iterations",
        "tabu_iter": 1,
        "tabu_tenure": 1,
        "k_max": 1,
        "search_mode": "fast",
        "move_candidate_limit": 10,
        "route_neighbor_limit": 2,
        "position_neighbor_limit": 2,
        "exchange_neighbor_limit": 2,
        "station_candidate_limit": 1,
        "verbose": False,
        "save_traceback": True,
        "stage2_task": {
            "view_id": canonical.instance_id,
            "index_path": "/tmp/view_index.parquet",
            "family_id": "mf-test",
            "city_slug": "test",
            "split_id": "test",
            "track_id": "core",
            "scale_id": "Cus1",
        },
    }

    result = runner.solve_one(task)

    summary = result["summary_row"]
    assert summary["status"] == "COMPLETED_WITH_INCUMBENT"
    assert json.loads(summary["routes_json"]) == [[0, 1, 0]]
    assert summary["route_validation_passed"] is True
    assert summary["incumbent_replay_misses"] == 2
    assert summary["incumbent_replay_hits"] == 1
