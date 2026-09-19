"""Regression cases where distance-only proposal ranking disagrees with money."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_FIXTURE_PATH = Path(__file__).with_name("test_solver_optimization_equivalence.py")
_spec = importlib.util.spec_from_file_location("vns_cost_fixtures", _FIXTURE_PATH)
_fixtures = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_fixtures)
VNSTSolver = _fixtures.CurrentSolver
Route = _fixtures.Route


def make_solver(*, fixed=413.6331536717643, alpha=0.15175097276264593, mode="fast"):
    instance = _fixtures._make_instance()
    instance.vehicle_params["load_cap"] = 100.0
    return VNSTSolver(instance, distance_unit_cost=alpha, vehicle_fixed_cost=fixed, search_mode=mode)


def monetary_value(solver, routes):
    """Independent scalar sum, not solver generalized_cost or delta helper."""
    return sum(
        solver.vehicle_fixed_cost + solver.distance_unit_cost * sum(
            float(solver.dist_matrix[a.id, b.id]) for a, b in zip(route.nodes, route.nodes[1:])
        ) for route in routes
    )


@pytest.mark.parametrize("move", [
    ("relocate", 0, 1, 1, 2),
    ("relocate", 0, 1, 0, 4),
    ("relocate_new", 0, 1),
    ("exchange", 0, 1, 1, 2),
    ("exchange", 0, 1, 0, 3),
    ("two_opt", 0, 1, 2, 2),
    ("station_remove", 0, 2),
    ("station_insert", 1, 2, "station"),
])
def test_move_delta_equals_actual_selected_routes_objective(move):
    solver = make_solver()
    solution = _fixtures._solution(solver.instance)
    if move[-1] == "station":
        move = (*move[:-1], solver.instance.stations[0])
    before_ids = _fixtures._route_ids(solution)
    candidate = solver._apply_fast_move(solution, move)
    expected = monetary_value(solver, candidate) - monetary_value(solver, solution)
    assert solver._local_move_cost_delta(solution, move) == pytest.approx(expected)
    assert _fixtures._route_ids(solution) == before_ids


def test_singleton_merge_saves_dispatch_even_when_distance_increases():
    solver = make_solver(fixed=100.0, alpha=0.1)
    d = solver.instance.depot
    a, b = solver.instance.customers[:2]
    solver.instance.customers = [a, b]
    solver.all_customer_ids = frozenset((a.id, b.id))
    solver.dist_matrix[:] = 30.0
    np.fill_diagonal(solver.dist_matrix, 0)
    solver.dist_matrix[d.id, :] = solver.dist_matrix[:, d.id] = 1.0
    solver.dist_matrix[0, 0] = 0
    solution = [Route([d, a, d]), Route([d, b, d])]
    move = ("relocate", 0, 1, 1, 1)
    assert solver._local_move_cost_delta(solution, move) == pytest.approx(-97.2)
    ranked = solver._ranked_candidate_moves_fast(solution)
    assert ranked[0][0] == pytest.approx(-97.2)
    assert ranked[0][1][0] == "relocate"
    # A geometric shortlist must not silently discard monetary-improving moves.
    solver._customer_neighbor_ids = {}
    assert solver._ranked_candidate_moves_fast(solution) == ranked


def test_opening_route_includes_fixed_charge_and_source_savings():
    solver = make_solver(fixed=100.0, alpha=0.1)
    d = solver.instance.depot
    a, b = solver.instance.customers[:2]
    solver.dist_matrix[:] = 1.0
    np.fill_diagonal(solver.dist_matrix, 0)
    solution = [Route([d, a, b, d])]
    assert solver._local_move_cost_delta(solution, ("relocate_new", 0, 1)) == pytest.approx(100.1)


@pytest.mark.parametrize("fixed,expected_routes", [(10.0, 2), (2000.0, 1)])
def test_consolidation_compares_merge_cost_with_separate_route(fixed, expected_routes):
    solver = make_solver(fixed=fixed, alpha=1.0)
    d = solver.instance.depot
    a, b = solver.instance.customers[:2]
    solver.instance.customers = [a, b]
    solver.all_customer_ids = frozenset((a.id, b.id))
    solver.dist_matrix[:] = 1000.0
    np.fill_diagonal(solver.dist_matrix, 0)
    solver.dist_matrix[0, :] = solver.dist_matrix[:, 0] = 1.0
    solver.dist_matrix[0, 0] = 0
    singleton = [Route([d, a, d]), Route([d, b, d])]
    solver.initial_construction_budget_s = 5.0
    result = solver._consolidate_singleton_solution(singleton)
    assert len(result) == expected_routes
    assert monetary_value(solver, result) <= monetary_value(solver, singleton)
    assert solver.is_solution_feasible(result)


def test_full_search_scores_post_cleanup_dispatch_count():
    solver = make_solver(mode="full")
    d = solver.instance.depot
    station = solver.instance.stations[1]
    a = solver.instance.customers[0]
    solver.all_customer_ids = frozenset((a.id,))
    empty_source = Route([d, station, d])
    kept = Route([d, a, d])
    value = solver.generalized_cost([empty_source, kept], False, False, False)
    assert value == pytest.approx(monetary_value(solver, [kept]))
    assert _fixtures._route_ids([empty_source]) == [[0, station.id, 0]]


def test_all_retained_candidates_rank_by_exact_monetary_change():
    solver = make_solver()
    solution = _fixtures._solution(solver.instance)
    ranked = solver._ranked_candidate_moves_fast(solution)
    assert ranked
    assert [score for score, _ in ranked] == sorted(score for score, _ in ranked)
    for score, move in ranked:
        candidate = solver._apply_fast_move(solution, move)
        assert score == pytest.approx(monetary_value(solver, candidate) - monetary_value(solver, solution))
