"""Cost-based proposal pruning and incumbent retention regressions."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
SOLVER_ROOT = ROOT / 'EVRPTW_Benchmark/MetaHeuristics/VNS_TS_Solver'
for path in (SOLVER_ROOT, SOLVER_ROOT.parent, ROOT / 'EVRPTW_Core', ROOT / 'EVRPTW_Dataset_Generator/src'):
    sys.path.insert(0, str(path))

from evrptw_core.objective import load_objective
from vnst_adapter import Customer, Depot, Route, Station, VNSTInstance

spec = importlib.util.spec_from_file_location('vns_cost_solver', SOLVER_ROOT / 'solver.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VNSTSolver = module.VNSTSolver
OBJECTIVE = load_objective(ROOT / 'EVRPTW_Benchmark/Reinforcement_Learning/configs/rivian_energy_vehicle_cost_v2.json')


def make(xs, *, stations=(), **kwargs):
    instance = VNSTInstance()
    instance.depot = Depot(0, 'd', 0, 0, 0, 0, 1e9, 0)
    instance.customers = [Customer(i, 'c', x, 0, 1, 0, 1e9, 0) for i, x in enumerate(xs, 1)]
    instance.stations = [Station(i, 'f', x, 0, 0, 0, 1e9, 0) for i, x in enumerate(stations, len(xs) + 1)]
    instance.station_charging_power_kw = {station.id: 100. for station in instance.stations}
    instance.vehicle_params = {'fuel_cap': 1e9, 'load_cap': 1e9, 'velocity': 1, 'consump_rate': 0.01}
    coordinates = np.array([0, *xs, *stations], dtype=float)
    instance.dist_matrix = np.abs(coordinates[:, None] - coordinates[None, :])
    instance.time_matrix = instance.dist_matrix.copy()
    instance.energy_matrix = 0.01 * instance.dist_matrix
    instance.terminal_order = list(range(len(coordinates)))
    solver = VNSTSolver(instance, distance_unit_cost=OBJECTIVE.distance_unit_cost,
                       vehicle_fixed_cost=OBJECTIVE.vehicle_unit_cost, **kwargs)
    return solver, [instance.depot, *instance.customers, *instance.stations]


def cost(solver, solution):
    return solver.generalized_cost(solution, False, False, False)


def ids(solution):
    return tuple(tuple(node.id for node in route.nodes) for route in solution)


def test_customer_relocate_closing_a_vehicle_survives_fast_pruning():
    solver, n = make([1.] * 100 + [100.], move_candidate_limit=40)
    initial = [Route([n[0], n[i], n[i + 1], n[0]]) for i in range(1, 101, 2)]
    initial.append(Route([n[0], n[101], n[0]]))
    moves = solver._ranked_candidate_moves_fast(initial)
    closures = [(delta, move) for delta, move in moves if move[0] == 'relocate' and move[1] == 50 and move[3] != 50]
    assert closures, 'The old insertion-distance ranking discarded every singleton closure.'
    for delta, move in closures:
        candidate = solver._apply_fast_move(initial, move)
        assert len(candidate) == len(initial) - 1
        assert delta == pytest.approx(cost(solver, candidate) - cost(solver, initial))
        assert delta < -400
    assert len(moves) <= solver._effective_fast_limits(101)[-1]
    # Opening a vehicle remains available for feasibility recovery, despite its fee.
    assert any(move[0] == 'relocate_new' for _, move in moves)


@pytest.mark.parametrize('cost_mode', [True, False])
def test_all_retained_proxies_equal_objective_change_on_directed_network(cost_mode):
    solver, n = make([1., 2., 3., 4.], stations=[1.5, 3.5], move_candidate_limit=0,
                     route_neighbor_limit=4, position_neighbor_limit=99, exchange_neighbor_limit=99)
    if not cost_mode:
        solver.distance_unit_cost, solver.vehicle_fixed_cost = 1.0, 0.0
    rng = np.random.default_rng(812)
    solver.dist_matrix = rng.uniform(1, 10, (7, 7))
    np.fill_diagonal(solver.dist_matrix, 0)
    # Several customer-free prefixes/tails exercise two-opt route removal.
    initial = [Route([n[0], n[5], n[1], n[2], n[0]]),
               Route([n[0], n[3], n[6], n[0]]), Route([n[0], n[4], n[0]])]
    before_ids = ids(initial)
    baseline = solver.generalized_cost(initial, False, False, True)
    ranked = solver._ranked_candidate_moves_fast(initial)
    kinds = {move[0] for _, move in ranked}
    assert {'relocate', 'relocate_new', 'exchange', 'two_opt', 'station_remove'} <= kinds
    for delta, move in ranked:
        candidate = solver._apply_fast_move(initial, move)
        actual = solver.generalized_cost(candidate, False, False, True) - baseline
        assert delta == pytest.approx(actual, abs=1e-10), move
    assert ids(initial) == before_ids
    assert any(move[0] == 'two_opt' and len(solver._apply_fast_move(initial, move)) < len(initial)
               for _, move in ranked)


def test_station_insert_proxy_uses_cost_coefficient_and_can_repair_battery():
    solver, n = make([2.], stations=[1.], move_candidate_limit=40)
    solver.instance.vehicle_params['fuel_cap'] = 3.0
    solver.energy_matrix = solver.dist_matrix.copy()
    # A positive distance detour with independently feasible energy arcs.
    solver.dist_matrix[1, 2] = solver.dist_matrix[2, 0] = 1.5
    initial = [Route([n[0], n[1], n[0]])]
    proposals = solver._ranked_candidate_moves_fast(initial)
    insertions = [(delta, move) for delta, move in proposals if move[0] == 'station_insert']
    assert insertions
    assert all(delta > 0 for delta, _ in insertions)
    for delta, move in insertions:
        candidate = solver._apply_fast_move(initial, move)
        assert solver.is_solution_feasible(candidate)
        assert delta == pytest.approx(solver.generalized_cost(candidate, False, False, True)
                                      - solver.generalized_cost(initial, False, False, True))


def test_full_scores_empty_route_cleanup_before_selecting_the_move():
    solver, n = make([0.01, 100., -100., 101., -101.], search_mode='full')
    initial = [Route([n[0], n[1], n[0]]), Route([n[0], n[2], n[3], n[4], n[5], n[0]])]
    before_ids = ids(initial)
    solver.global_value = cost(solver, initial)
    solver.global_solution = solver.clone_solution_shallow(initial)
    solver.tabu_iter = 1
    actual_cost = solver.generalized_cost
    evaluated = []

    def checking(candidate, *args, **kwargs):
        assert all(any(node.type == 'c' for node in route.nodes) for route in candidate)
        evaluated.append(ids(candidate))
        return actual_cost(candidate, *args, **kwargs)

    solver.generalized_cost = checking
    selected = solver._tabu_search(initial)
    assert solver.is_solution_feasible(selected)
    assert len(selected) == 1
    # Any single-vehicle tour here beats even the distance-optimal two-vehicle tour.
    assert cost(solver, selected) < 600
    assert ids(initial) == before_ids  # apply/rollback did not damage caller-owned routes
    assert any(len(candidate) == 1 for candidate in evaluated)
    assert solver.global_value == cost(solver, selected)


def test_fast_saves_feasible_candidate_when_an_infeasible_move_wins():
    solver = VNSTSolver.__new__(VNSTSolver)
    solver.clone_solution_shallow = lambda routes: list(routes)
    solver._time_limit_reached = lambda: False
    solver._decay_station_tabu = lambda: None
    solver._candidate_moves_fast = lambda solution: ['feasible_improvement', 'cheap_infeasible']
    solver._move_tabu_key = lambda move: move
    solver._apply_fast_move = lambda solution, move: [move]
    solver.solution_fix = lambda solution: solution
    trajectory = []
    solver.update_diversification_history = lambda solution: trajectory.append(list(solution))
    events = []
    solver._report_incumbent = lambda: events.append((solver.global_value, list(solver.global_solution)))
    solver.tabu_tenure, solver.tabu_iter = 30, 1
    solver.global_value, solver.global_solution = 1000., ['initial']

    def evaluate(solution, penalty_value=True, p_div_value=True, allow_infeasible=True, **kwargs):
        return {'initial': 1000., 'feasible_improvement': 900.,
                'cheap_infeasible': 800. if allow_infeasible else 1e10}[solution[0]]

    solver.generalized_cost = evaluate
    selected = solver._tabu_search_fast(['initial'])
    assert trajectory == [['cheap_infeasible']]
    assert selected == ['feasible_improvement']
    assert solver.global_solution == selected
    assert solver.global_value == 900.
    assert events == [(900., ['feasible_improvement'])]


def test_full_saves_feasible_candidate_when_an_infeasible_move_wins(monkeypatch):
    solver, n = make([1., 2., 3.], search_mode='full')
    initial = [Route([n[0], n[1], n[0]]), Route([n[0], n[2], n[3], n[0]])]
    good = ((0, 2, 1, 3, 0),)
    bad = ((0, 1, 2, 3, 0),)
    trajectory, events = [], []

    def evaluate(solution, penalty_value=True, p_div_value=True, allow_infeasible=True, **kwargs):
        key = ids(solution)
        if key == good:
            return 900.
        if key == bad:
            return 800. if allow_infeasible else 1e10
        return 1000.

    monkeypatch.setattr(solver, 'generalized_cost', evaluate)
    monkeypatch.setattr(solver, '_report_incumbent', lambda: events.append(ids(solver.global_solution)))
    monkeypatch.setattr(solver, 'update_diversification_history', lambda solution: trajectory.append(ids(solution)))
    solver.tabu_iter = 1
    solver.global_value, solver.global_solution = 1000., solver.clone_solution_shallow(initial)
    result = solver._tabu_search(initial)
    assert trajectory == [bad]
    assert ids(result) == good
    assert ids(solver.global_solution) == good
    assert solver.global_value == 900.
    assert events == [good]


def test_cli_defaults_to_v2_and_distance_requires_explicit_choice(monkeypatch):
    fake_solver = types.ModuleType('solver')
    fake_solver.VNSTSolver = VNSTSolver
    monkeypatch.setitem(sys.modules, 'solver', fake_solver)
    runner_spec = importlib.util.spec_from_file_location('vns_cost_runner', SOLVER_ROOT / 'run_vns_ts.py')
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    assert runner.resolve_objective_config(None) == OBJECTIVE
    assert runner.resolve_objective_config('') == OBJECTIVE
    assert runner.resolve_objective_config(runner.DEFAULT_OBJECTIVE_CONFIG) == OBJECTIVE
    legacy = runner.resolve_objective_config('distance')
    assert legacy.profile_id == 'distance_v1'
    assert legacy.value(12, 2) == 12
