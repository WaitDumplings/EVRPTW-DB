"""Regressions for cost-aware repair and checkpoint score units."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
META_ROOT = REPO / "EVRPTW_Benchmark/MetaHeuristics"
ALNS_ROOT = META_ROOT / "ALNS_Solver"
for path in (REPO / "EVRPTW_Core", REPO / "EVRPTW_Dataset_Generator/src", META_ROOT, ALNS_ROOT):
    sys.path.insert(0, str(path))

from evrptw_core.objective import load_objective
from solver import ALNS_Solver

COST = load_objective(
    REPO / "EVRPTW_Benchmark/Reinforcement_Learning/configs/rivian_energy_vehicle_cost_v2.json"
)
BASE = [[0, 1, 2, 0]]
MERGED = [[0, 1, 3, 2, 0]]
SPLIT = [[0, 1, 2, 0], [0, 3, 0]]


def make_solver(*, cost: bool = True, **kwargs) -> ALNS_Solver:
    def node(identifier, x, ready, due, kind="c"):
        return dict(id=identifier, type=kind, x=x, y=0,
                    demand=int(kind == "c"), ready=ready, due=due, service=0)

    instance = {
        "depot": node("D0", 0, 0, 100, "d"),
        "customers": [node("A", 10, 0, 10), node("B", 11, 33, 34),
                      node("C", -1, 21, 22)],
        "stations": [],
        "vehicle": dict(Q=1000, C=100, r=.01, g=1, v=1),
    }
    if cost:
        kwargs.setdefault("distance_unit_cost", COST.distance_unit_cost)
        kwargs.setdefault("vehicle_fixed_cost", COST.vehicle_unit_cost)
    return ALNS_Solver(instance, **kwargs)


def populated_checkpoint(*, cost=True):
    solver = make_solver(cost=cost)
    solver.current_routes = copy.deepcopy(SPLIT)
    solver.best_routes = copy.deepcopy(SPLIT)
    solver.global_value = solver.objective_value(SPLIT)
    solver.temperature = 8.75
    solver.cur_iter = 17
    return solver.get_checkpoint()


@pytest.mark.parametrize("top_k", [None, 8])
@pytest.mark.parametrize("operator", ["_ci_greedy", "_ci_regret2", "_ci_regret3"])
def test_cost_repair_prefers_longer_one_vehicle_solution(top_k, operator):
    solver = make_solver()
    solver.customer_top_k = top_k
    assert solver.is_solution_feasible(MERGED)
    assert solver.is_solution_feasible(SPLIT)
    assert sum(map(solver._route_distance, MERGED)) == 44
    assert sum(map(solver._route_distance, SPLIT)) == 24
    assert solver.objective_value(MERGED) == pytest.approx(420.310196)
    assert solver.objective_value(SPLIT) == pytest.approx(830.908331)
    assert getattr(solver, operator)(copy.deepcopy(BASE), [3]) == MERGED


@pytest.mark.parametrize("top_k", [None, 8])
def test_insertion_scores_equal_full_cost_change_including_vehicle_fee(top_k):
    solver = make_solver()
    solver.customer_top_k = top_k
    base_cost = COST.value(22, 1)
    options = solver._all_customer_insertions(BASE, 3)
    assert {ridx for ridx, _, _ in options} == {0, 1}
    for ridx, route, score in options:
        candidate = copy.deepcopy(BASE)
        if ridx == len(candidate):
            candidate.append(route)
        else:
            candidate[ridx] = route
        assert score == pytest.approx(solver.objective_value(candidate) - base_cost)


@pytest.mark.parametrize("top_k", [None, 8])
def test_repair_can_still_open_a_cheaper_route_with_small_vehicle_fee(top_k):
    solver = make_solver(distance_unit_cost=2, vehicle_fixed_cost=.1)
    solver.customer_top_k = top_k
    assert solver.objective_value(SPLIT) < solver.objective_value(MERGED)
    assert solver._ci_greedy(copy.deepcopy(BASE), [3]) == SPLIT


@pytest.mark.parametrize("top_k", [None, 8])
def test_explicit_distance_and_time_exploration_keep_their_rankings(top_k):
    distance_solver, cost_solver = make_solver(cost=False), make_solver()
    distance_solver.customer_top_k = cost_solver.customer_top_k = top_k
    assert distance_solver._ci_greedy(copy.deepcopy(BASE), [3]) == SPLIT
    assert cost_solver._all_customer_insertions(BASE, 3, mode="time") == (
        distance_solver._all_customer_insertions(BASE, 3, mode="time")
    )


def test_cost_acceptance_does_not_accept_shorter_but_more_expensive_routes():
    solver = make_solver()
    solver.global_value = solver.objective_value(MERGED)
    solver.temperature = 1e-8
    accepted, reward, value = solver._evaluate_candidate(SPLIT, solver.global_value)
    assert not accepted and reward == solver.r4
    assert value == pytest.approx(COST.value(24, 2))


def test_checkpoint_roundtrip_preserves_cost_best_and_temperature_units():
    checkpoint = populated_checkpoint()
    solver = make_solver(checkpoint=checkpoint)
    assert solver.get_checkpoint() == checkpoint
    accepted, reward, value = solver._evaluate_candidate(
        MERGED, solver.objective_value(solver.current_routes)
    )
    assert accepted and reward == solver.r1
    assert value == pytest.approx(COST.value(44, 1))
    assert solver.temperature == 8.75


@pytest.mark.parametrize("overrides", [
    dict(distance_unit_cost=1, vehicle_fixed_cost=0),
    dict(distance_unit_cost=.2), dict(vehicle_fixed_cost=123),
])
def test_cross_objective_checkpoint_is_rejected_before_mutation(overrides):
    solver = make_solver(**overrides)
    before = solver.get_checkpoint()
    with pytest.raises(ValueError, match="objective contract"):
        solver.load_checkpoint(populated_checkpoint())
    assert solver.get_checkpoint() == before


def test_legacy_distance_checkpoint_is_validated_and_can_resume_distance_only():
    legacy = populated_checkpoint(cost=False)
    legacy.pop("objective_contract")
    legacy.pop("algorithm_profile_id")
    solver = make_solver(cost=False, checkpoint=legacy)
    assert solver.global_value == 24
    assert solver.temperature == 8.75
    with pytest.raises(ValueError, match="legacy checkpoint.*distance_v1"):
        make_solver(checkpoint=legacy)


def test_unlabelled_cost_checkpoint_is_not_misread_as_legacy_distance():
    legacy = populated_checkpoint()
    legacy.pop("objective_contract")
    with pytest.raises(ValueError, match="best value disagrees"):
        make_solver(cost=False, checkpoint=legacy)


@pytest.mark.parametrize("bad_value", [24, float("inf"), float("nan")])
def test_checkpoint_rejects_stale_or_nonfinite_best_value(bad_value):
    checkpoint = populated_checkpoint()
    checkpoint["global_value"] = bad_value
    with pytest.raises(ValueError, match="best value disagrees"):
        make_solver(checkpoint=checkpoint)


@pytest.mark.parametrize("bad_temperature", [-1, float("inf"), float("nan")])
def test_checkpoint_rejects_invalid_annealing_temperature(bad_temperature):
    checkpoint = populated_checkpoint()
    checkpoint["temperature"] = bad_temperature
    with pytest.raises(ValueError, match="temperature must"):
        make_solver(checkpoint=checkpoint)


def test_uninitialized_checkpoint_roundtrip_and_ambiguous_legacy_temperature():
    checkpoint = make_solver().get_checkpoint()
    assert make_solver(checkpoint=checkpoint).global_value == float("inf")
    checkpoint = make_solver(cost=False).get_checkpoint()
    checkpoint.pop("objective_contract")
    checkpoint["temperature"] = 1
    with pytest.raises(ValueError, match="temperature units cannot"):
        make_solver(cost=False, checkpoint=checkpoint)


@pytest.mark.parametrize("explicit_distance", [False, True])
def test_cli_defaults_to_cost_but_allows_explicit_distance(
    tmp_path, monkeypatch, capsys, explicit_distance
):
    spec = importlib.util.spec_from_file_location("alns_cost_cli_test", ALNS_ROOT / "run_alns.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "build_input_tasks", lambda *args, **kwargs: [])
    arguments = ["run_alns.py", "--dataset_path", "unused", "--save_path", str(tmp_path / "out")]
    if explicit_distance:
        profile = tmp_path / "distance_v1.json"
        profile.write_text(json.dumps({"objective": {"mode": "distance", "profile_id": "distance_v1"}}))
        arguments.extend(["--objective_config", str(profile)])
    monkeypatch.setattr(sys, "argv", arguments)
    module.main()
    expected = "distance_v1" if explicit_distance else COST.profile_id
    assert f"objective={expected}" in capsys.readouterr().out
    assert module.ALGORITHM_PROFILE_ID == make_solver().algorithm_profile_id
    assert module.ALGORITHM_PROFILE_ID != "alns_stage2_scalable_v2"
