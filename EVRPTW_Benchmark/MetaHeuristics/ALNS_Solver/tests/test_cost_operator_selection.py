from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("alns_cost_operators", Path(__file__).resolve().parents[1] / "solver.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ALNS_Solver = MODULE.ALNS_Solver


def make_solver(*, monetary=True, top_k=None):
    n = 5  # depot, three customers, one station
    distances = np.full((n, n), 100.0)
    np.fill_diagonal(distances, 0.0)
    data = {
        "depot": np.array([[0.0, 0.0]]),
        "customers": np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        "charging_stations": np.array([[4.0, 0.0]]),
        "customer_demand": np.ones(3), "customer_service": np.zeros(3),
        "tw": np.array([[0.0, 60000.0]] * 3),
        "distance_matrix_km": distances,
        "time_matrix_min": np.ones((n, n)),
        "energy_matrix_kwh": np.zeros((n, n)),
        "charging_power_kw": np.array([50.0]),
        "env": {"instance_startTime": 0.0, "instance_endTime": 60000.0,
                "battery_capacity": 100.0, "loading_capacity": 100.0,
                "consumption_per_distance": 0.0, "charging_speed": float("inf"), "speed": 1.0},
    }
    solver = ALNS_Solver(data, format="tensor", seed=2026,
                         distance_unit_cost=0.15 if monetary else 1.0,
                         vehicle_fixed_cost=413.0 if monetary else 0.0)
    solver.customer_top_k = top_k
    return solver


@pytest.mark.parametrize("top_k", [None, 2])
@pytest.mark.parametrize("mode", ["distance", "time"])
def test_insertion_prices_new_vehicle_even_when_it_is_shorter(top_k, mode):
    solver = make_solver(top_k=top_k)
    solver.dist_matrix[0, 1] = solver.dist_matrix[1, 0] = 10.0
    solver.dist_matrix[0, 2] = solver.dist_matrix[2, 0] = 1.0
    solver.dist_matrix[1, 2] = solver.dist_matrix[2, 1] = 20.0
    options = solver._all_customer_insertions([[0, 1, 0]], 2, mode=mode)
    new_option = next(item for item in options if item[0] == 1)
    existing_option = min((item for item in options if item[0] == 0), key=lambda item: item[2])
    assert new_option[2] == pytest.approx(413.0 + 0.15 * 2)
    assert existing_option[2] == pytest.approx(0.15 * 11)
    assert min(options, key=lambda item: item[2])[0] == 0


def test_explicit_distance_mode_still_prefers_shorter_new_route():
    solver = make_solver(monetary=False)
    solver.dist_matrix[0, 1] = solver.dist_matrix[1, 0] = 10.0
    solver.dist_matrix[0, 2] = solver.dist_matrix[2, 0] = 1.0
    solver.dist_matrix[1, 2] = solver.dist_matrix[2, 1] = 20.0
    assert solver._best_customer_insertion([[0, 1, 0]], 2)[0] == 1


def test_worst_removal_accounts_for_eliminated_vehicle():
    solver = make_solver()
    solver.dist_matrix[0, 1] = solver.dist_matrix[1, 0] = 1.0
    solver._num_customers_to_remove = lambda: 1
    solver._random_customer_removal_mode = lambda: "customer_only"
    solver._select_ranked_with_noise = lambda scored, q, determinism: [entry[1] for entry in scored[:q]]
    remaining, removed = solver._cr_worst_distance([[0, 1, 0], [0, 2, 3, 0]])
    assert removed == [1]  # 413.3 USD saved, versus 15 USD for either other customer.
    assert remaining == [[0, 2, 3, 0]]


def test_regret_missing_alternative_uses_monetary_units():
    solver = make_solver()
    solver.max_distance = 100.0
    def alternatives(routes, customer, mode):
        if customer == 1:
            return [(len(routes), [0, 1, 0], 10.0)]
        return [(len(routes), [0, 2, 0], 0.0), (len(routes), [0, 2, 0], 200.0)]
    solver._all_customer_insertions = alternatives
    assert solver._ci_regret2([], [1, 2])[0] == [0, 1, 0]


def test_pruning_station_cannot_raise_fastest_path_cost():
    solver = make_solver()
    solver.dist_matrix[0, 1] = 1.0
    solver.dist_matrix[1, 4] = solver.dist_matrix[4, 0] = 1.0
    solver.dist_matrix[1, 0] = 50.0
    assert solver._prune_redundant_stations([0, 1, 4, 0]) == [0, 1, 4, 0]
    solver.dist_matrix[1, 0] = 1.0
    solver._clear_caches()
    assert solver._prune_redundant_stations([0, 1, 4, 0]) == [0, 1, 0]


def test_cost_acceptance_rewards_one_vehicle_despite_longer_distance():
    solver = make_solver()
    solver.dist_matrix[0, 1] = solver.dist_matrix[1, 0] = 1.0
    solver.dist_matrix[0, 2] = solver.dist_matrix[2, 0] = 1.0
    solver.dist_matrix[1, 2] = 100.0
    solver.customer_indices = [1, 2]
    old_routes = [[0, 1, 0], [0, 2, 0]]
    solver.global_value = solver.objective_value(old_routes)
    accepted, reward, value = solver._evaluate_candidate([[0, 1, 2, 0]], solver.global_value)
    assert accepted and reward == solver.r1
    assert value == pytest.approx(413.0 + 0.15 * 102)
    assert value < solver.global_value


def test_initial_consolidation_declines_more_expensive_nonmetric_merge():
    solver = make_solver()
    solver.dist_matrix[0, 1] = solver.dist_matrix[1, 0] = 1.0
    solver.dist_matrix[0, 2] = solver.dist_matrix[2, 0] = 1.0
    solver.dist_matrix[1, 2] = solver.dist_matrix[2, 1] = 10000.0
    solver.customer_indices = [1, 2]
    solver.singleton_source = "test_verified_singletons"
    routes = [[0, 1, 0], [0, 2, 0]]
    constructed = solver._construct_initial_solution(singleton_routes=routes, use_wall_clock_budget=False)
    assert constructed == routes


@pytest.mark.parametrize("monetary", [True, False])
def test_time_named_repair_shortlists_by_usd_in_cost_mode(monetary):
    solver = make_solver(monetary=monetary, top_k=1)
    solver.dist_matrix[0, 1] = solver.dist_matrix[1, 0] = 10.0
    solver.dist_matrix[0, 2] = solver.dist_matrix[2, 1] = 30.0
    solver.dist_matrix[1, 2] = solver.dist_matrix[2, 0] = 1.0
    solver.time_matrix[1, 2] = solver.time_matrix[2, 0] = 100.0
    # Before customer 1 is temporally cheaper; after it is monetarily cheaper.
    options = solver._all_customer_insertions(
        [[0, 1, 0]], 2, mode="time", include_new_route=False
    )
    assert len(options) == 1
    assert options[0][1] == ([0, 1, 2, 0] if monetary else [0, 2, 1, 0])
    if monetary:
        assert options[0][2] == pytest.approx(-1.2)


def test_initial_route_shortlist_uses_insertion_cost_not_nearest_node():
    solver = make_solver()
    solver.initial_merge_candidate_limit = 1
    solver.initial_exact_insertion_limit = 2
    solver.dist_matrix[0, 1] = solver.dist_matrix[1, 0] = 1.0
    solver.dist_matrix[0, 2] = solver.dist_matrix[2, 0] = 1000.0
    solver.dist_matrix[1, 2] = solver.dist_matrix[2, 1] = 10000.0
    solver.dist_matrix[1, 3] = solver.dist_matrix[3, 1] = 1.0
    solver.singleton_source = "test_verified_singletons"
    # Customers 1 and 2 cannot profitably merge. Customer 3 is nearer to 1,
    # but replacing 2's costly depot arc saves more money.
    routes = solver._construct_initial_solution(
        singleton_routes=[[0, 1, 0], [0, 2, 0], [0, 3, 0]],
        use_wall_clock_budget=False,
    )
    assert routes[0] == [0, 1, 0]
    assert len(routes) == 2
    assert set(routes[1]) == {0, 2, 3}
    assert solver.objective_value(routes) < solver.objective_value(
        [[0, 1, 3, 0], [0, 2, 0]]
    )
