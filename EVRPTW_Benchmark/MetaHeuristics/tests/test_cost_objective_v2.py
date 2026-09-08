from __future__ import annotations

from types import SimpleNamespace

import pytest

from evrptw_core.objective import load_objective
from benchmark_common import IncumbentEventRecorder
from solver import ALNS_Solver


PROFILE = (
    "EVRPTW_Benchmark/Reinforcement_Learning/configs/"
    "rivian_energy_vehicle_cost_v2.json"
)


def test_frozen_cost_profile_matches_published_coefficients() -> None:
    objective = load_objective(PROFILE)

    assert objective.profile_id == "rivian_energy_vehicle_cost_v2"
    assert objective.unit == "USD"
    assert objective.distance_unit_cost == pytest.approx(0.151750972762646)
    assert objective.vehicle_unit_cost == pytest.approx(413.6331536717643)
    assert objective.value(100.0, 2) == pytest.approx(
        0.151750972762646 * 100.0 + 413.6331536717643 * 2
    )


def test_incumbent_recorder_ranks_cost_not_raw_distance() -> None:
    objective = load_objective(PROFILE)
    recorder = IncumbentEventRecorder((300.0, 1800.0), 1800.0)

    two_vehicle_fields = objective.fields(10.0, 2)
    recorder.observe(
        1.0,
        two_vehicle_fields["objective_value"],
        [[0, 1, 0], [0, 2, 0]],
        objective_distance_km=10.0,
        objective_fields=two_vehicle_fields,
    )
    one_vehicle_fields = objective.fields(20.0, 1)
    recorder.observe(
        2.0,
        one_vehicle_fields["objective_value"],
        [[0, 1, 2, 0]],
        objective_distance_km=20.0,
        objective_fields=one_vehicle_fields,
    )

    best = recorder.best_event
    assert best is not None
    assert best["objective_distance_km"] == 20.0
    assert best["vehicles_started"] == 1
    assert best["objective_value"] == pytest.approx(
        one_vehicle_fields["objective_value"]
    )


def test_alns_internal_objective_includes_vehicle_fixed_cost() -> None:
    objective = load_objective(PROFILE)
    solver = ALNS_Solver.__new__(ALNS_Solver)
    solver.distance_unit_cost = objective.distance_unit_cost
    solver.vehicle_fixed_cost = objective.vehicle_unit_cost
    solver.is_solution_feasible = lambda routes: True
    solver._route_distance = lambda route: float(route[0])

    value = solver.objective_value([[10.0], [20.0]])

    assert value == pytest.approx(objective.value(30.0, 2))


def test_vnsts_internal_objective_includes_vehicle_fixed_cost() -> None:
    # Import by explicit package path to avoid the two historical solver.py
    # modules sharing one short import name in the same pytest process.
    from VNS_TS_Solver.solver import VNSTSolver

    objective = load_objective(PROFILE)
    solver = VNSTSolver.__new__(VNSTSolver)
    solver.distance_unit_cost = objective.distance_unit_cost
    solver.vehicle_fixed_cost = objective.vehicle_unit_cost
    solver._direct_terminal_index = True
    solver.dist_matrix = __import__("numpy").asarray(
        [[0.0, 10.0], [20.0, 0.0]], dtype=float
    )
    route = SimpleNamespace(
        nodes=[SimpleNamespace(id=0), SimpleNamespace(id=1), SimpleNamespace(id=0)]
    )

    value = solver.generalized_cost(
        [route],
        penalty_value=False,
        p_div_value=False,
        allow_infeasible=True,
    )

    assert value == pytest.approx(objective.value(30.0, 1))
