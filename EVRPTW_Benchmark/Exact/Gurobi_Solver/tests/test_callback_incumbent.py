"""Exercise incumbent/candidate ordering without running an optimizer."""
from __future__ import annotations

import pytest
from gurobipy import GRB

from gurobi_solver import GurobiEVRPTWSolver, GurobiSolverConfig


class CallbackModel:
    def __init__(self, runtime, *, objective=None, best=None, bound=0.0, route_tag=1):
        self.values = {
            GRB.Callback.RUNTIME: runtime,
            GRB.Callback.MIPSOL_OBJ: objective,
            GRB.Callback.MIPSOL_OBJBST: best,
            GRB.Callback.MIPSOL_OBJBND: bound,
            GRB.Callback.MIP_OBJBST: best,
            GRB.Callback.MIP_OBJBND: bound,
        }
        self.route_tag = route_tag
        self.route_reads = 0

    def cbGet(self, key):
        return self.values[key]

    def cbGetSolution(self, variables):
        self.route_reads += 1
        return [self.route_tag] * len(variables)


def setup_callback(monkeypatch, checkpoints=()):
    solver = GurobiEVRPTWSolver(GurobiSolverConfig(checkpoints_s=checkpoints))
    # Route tags distinguish equal-objective candidate route identities.
    monkeypatch.setattr(
        solver, '_extract_routes_from_arc_values',
        lambda node_map, values: [[0, int(values[(0, 1)]), 0]],
    )
    monkeypatch.setattr(solver, '_route_distance_km', lambda routes, instance: 1.0)
    trace = solver._new_trace()
    callback = solver._make_callback(trace, object(), {(0, 1): object()}, object())
    return trace, callback


def test_worse_and_tied_candidates_preserve_first_incumbent_route(monkeypatch):
    trace, callback = setup_callback(monkeypatch)
    callback(CallbackModel(1, objective=100, best=GRB.INFINITY, route_tag=1), GRB.Callback.MIPSOL)
    worse = CallbackModel(2, objective=110, best=100, route_tag=2)
    tied = CallbackModel(3, objective=100, best=100, route_tag=3)
    callback(worse, GRB.Callback.MIPSOL)
    callback(tied, GRB.Callback.MIPSOL)
    assert trace['last_incumbent']['routes'] == [[0, 1, 0]]
    assert trace['last_best_obj'] == 100
    assert len(trace['incumbent_events']) == 1
    assert worse.route_reads == tied.route_reads == 0
    callback(CallbackModel(4, objective=90, best=100, route_tag=4), GRB.Callback.MIPSOL)
    assert trace['last_incumbent']['routes'] == [[0, 4, 0]]
    assert [event['objective_value'] for event in trace['incumbent_events']] == [100, 90]


def test_prior_mip_best_does_not_hide_later_matching_better_route(monkeypatch):
    trace, callback = setup_callback(monkeypatch)
    callback(CallbackModel(1, objective=100, route_tag=1), GRB.Callback.MIPSOL)
    callback(CallbackModel(2, best=80), GRB.Callback.MIP)
    worse_than_global = CallbackModel(3, objective=90, best=80, route_tag=2)
    callback(worse_than_global, GRB.Callback.MIPSOL)
    assert trace['last_best_obj'] == 80
    assert trace['last_incumbent']['objective_value'] == 100
    assert worse_than_global.route_reads == 0
    callback(CallbackModel(4, objective=80, best=80, route_tag=3), GRB.Callback.MIPSOL)
    assert trace['last_incumbent']['objective_value'] == 80
    assert trace['last_incumbent']['routes'] == [[0, 3, 0]]
    assert [event['objective_value'] for event in trace['incumbent_events']] == [100, 80]
    # A delayed/stale scalar notification cannot worsen the global best.
    callback(CallbackModel(5, best=90), GRB.Callback.MIP)
    assert trace['last_best_obj'] == 80


def test_candidate_callback_best_can_reject_first_nonincumbent_candidate(monkeypatch):
    trace, callback = setup_callback(monkeypatch)
    rejected = CallbackModel(1, objective=110, best=100)
    callback(rejected, GRB.Callback.MIPSOL)
    assert trace['last_incumbent'] is None
    assert trace['last_best_obj'] == 100
    assert rejected.route_reads == 0
    callback(CallbackModel(2, objective=100, best=100, route_tag=2), GRB.Callback.MIPSOL)
    assert trace['last_incumbent']['routes'] == [[0, 2, 0]]


def test_checkpoint_causality_and_equal_time_flush_survive_rejected_candidates(monkeypatch):
    trace, callback = setup_callback(monkeypatch, checkpoints=(60, 100, 110, 120))
    callback(CallbackModel(100, objective=100, bound=70, route_tag=1), GRB.Callback.MIPSOL)
    at_60, at_100 = trace['checkpoint_snapshots']
    assert not at_60['has_incumbent']
    assert at_100['objective_value'] == 100
    callback(CallbackModel(120, objective=110, best=100, bound=80, route_tag=2), GRB.Callback.MIPSOL)
    at_110, at_120 = trace['checkpoint_snapshots'][2:]
    assert at_110['objective_value'] == at_120['objective_value'] == 100
    assert at_110['best_bound'] == 70
    assert at_120['best_bound'] == 80
    assert at_120['mip_gap'] == pytest.approx(0.2)
    assert at_110['routes'] == at_120['routes'] == [[0, 1, 0]]
    assert trace['first_feasible_time_s'] == 100


def test_nonfinite_candidate_never_replaces_saved_incumbent(monkeypatch):
    trace, callback = setup_callback(monkeypatch, checkpoints=(2,))
    callback(CallbackModel(1, objective=100, bound=70), GRB.Callback.MIPSOL)
    callback(CallbackModel(2, objective=GRB.INFINITY, best=100, bound=80), GRB.Callback.MIPSOL)
    assert trace['last_incumbent']['objective_value'] == 100
    assert trace['checkpoint_snapshots'][0]['objective_value'] == 100
    assert trace['checkpoint_snapshots'][0]['best_bound'] == 80
