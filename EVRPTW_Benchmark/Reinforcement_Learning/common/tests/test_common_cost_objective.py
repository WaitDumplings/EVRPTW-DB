from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env.tests.test_objective_cost import (
    economic_instance,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.evaluation import (
    select_min_verified_distance,
    select_min_verified_objective,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import (
    ObjectiveConfig,
    objective_from_checkpoint,
    resolve_objective,
    route_dispatch_count,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_protocol import (
    validation_key,
    verified_validation,
)


def cost_config():
    return ObjectiveConfig(
        mode="energy_vehicle_cost", profile_id="rivian_energy_vehicle_cost_v1"
    )


def selection_instance():
    instance = economic_instance()
    distance = instance.distance_matrix_km.copy()
    # One vehicle is longer (34 km) than two vehicles (10 km), but still cheaper
    # under the versioned fixed fee. Resource matrices remain canonical inputs.
    distance[1, 2] = 30.0
    return replace(instance, distance_matrix_km=distance)


def candidate_info(routes, *, success=None, served=None, distance=None, vehicles=None, objective=None):
    objective = objective or cost_config()
    count = len(routes)
    distance = np.asarray(distance if distance is not None else np.ones(count), dtype=np.float64)
    vehicles = np.asarray(
        vehicles if vehicles is not None else [route_dispatch_count(route_set) for route_set in routes],
        dtype=np.int64,
    )
    return {
        "success": np.asarray(success if success is not None else [True] * count),
        "served_customers": np.asarray(served if served is not None else [2] * count),
        "objective_distance_km": distance,
        "vehicles_started": vehicles,
        "objective_value": objective.value(distance, vehicles),
        "objective_config": objective.to_dict(),
        "routes": routes,
    }


def test_versioned_cost_scalar_array_tensor_units_and_gradients():
    objective = cost_config()
    coefficient = 0.1341 * (100.0 / 257.0)
    assert objective.unit == "USD"
    assert objective.distance_unit_cost == pytest.approx(coefficient)
    assert objective.vehicle_unit_cost == pytest.approx(33.56)
    assert objective.value(100.0, 2) == pytest.approx(coefficient * 100 + 67.12)
    distance = np.asarray([1.0, 10.0])
    vehicles = np.asarray([1, 2])
    np.testing.assert_allclose(objective.value(distance, vehicles), coefficient * distance + 33.56 * vehicles)
    tensor_distance = torch.tensor(distance, requires_grad=True)
    tensor_vehicles = torch.tensor(vehicles, dtype=torch.float64, requires_grad=True)
    value = objective.value(tensor_distance, tensor_vehicles)
    value.sum().backward()
    torch.testing.assert_close(tensor_distance.grad, torch.full_like(tensor_distance, coefficient))
    torch.testing.assert_close(tensor_vehicles.grad, torch.full_like(tensor_vehicles, 33.56))
    fields = objective.fields(100.0, 2)
    assert fields["objective_cost_usd"] == pytest.approx(
        fields["electricity_cost_usd"] + fields["vehicle_cost_usd"]
    )
    assert fields["objective_value"] == fields["objective_cost_usd"]


@pytest.mark.parametrize("field,value", [
    ("mode", "unversioned-cost"),
    ("profile_id", ""),
    ("electricity_price_usd_per_kwh", -1),
    ("electricity_price_usd_per_kwh", float("nan")),
    ("consumption_kwh_per_km", float("inf")),
    ("vehicle_fixed_cost_usd", -1),
    ("vehicle_fixed_cost_usd", float("nan")),
    ("consumption_kwh_per_km", 0),
])
def test_objective_rejects_invalid_economic_contract(field, value):
    payload = cost_config().to_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        resolve_objective(payload)


@pytest.mark.parametrize("mode,dispatches", [
    ("max_edge", 1),
    ("single_customer_repair_mean", 1),
    ("single_customer_repair_median", 1),
    ("single_customer_repair_sum", 7),
    ("dataset_single_customer_repair_sum", 7),
])
def test_reference_scale_converts_distance_and_matching_reference_dispatches(mode, dispatches):
    objective = cost_config()
    assert objective.reward_scale(120.0, 7, mode) == pytest.approx(objective.value(120.0, dispatches))
    assert ObjectiveConfig().reward_scale(120.0, 7, mode) == 120.0


def test_legacy_and_checkpoint_contracts_are_explicit_and_not_silently_relabelled():
    legacy = objective_from_checkpoint({"args": {"objective_config": Path("mutable-old-file.json")}})
    assert legacy.mode == "distance"
    assert legacy.unit == "km"
    assert legacy.fields(10, 3)["objective_cost_usd"] is None
    assert legacy.value(10, 999) == 10
    with pytest.raises(ValueError, match="mismatch"):
        objective_from_checkpoint({}, override=cost_config())
    frozen = {"objective_config": cost_config().to_dict()}
    assert objective_from_checkpoint(frozen) == cost_config()
    changed = replace(cost_config(), vehicle_fixed_cost_usd=1)
    with pytest.raises(ValueError, match="mismatch"):
        objective_from_checkpoint(frozen, override=changed)
    # Older shorter package imports may have a different dataclass identity.
    aliased = SimpleNamespace(to_dict=cost_config().to_dict)
    assert resolve_objective(aliased) == cost_config()


def test_dispatch_count_counts_depot_departures_not_route_containers_or_station_visits():
    assert route_dispatch_count([[0, 1, 0, 2, 0], [0, 3, 4, 0]]) == 3
    assert route_dispatch_count([[0], [0, 0], [0, 0, 3, 4]]) == 1


def test_cost_selection_replays_cost_not_environment_distance_or_vehicle_labels():
    instance = selection_instance()
    routes = [[[0, 1, 0], [0, 2, 0]], [[0, 1, 2, 0]]]
    # Deliberately wrong legacy metadata: selected cost must come from replay.
    info = candidate_info(routes, distance=[1, 999], vehicles=[0, 999])
    selected, selected_routes, verification = select_min_verified_objective(instance, info)
    assert selected == 1
    assert selected_routes == routes[1]
    assert verification["passed"]
    assert verification["objective_distance_km"] == 34
    assert verification["vehicles_started"] == 1
    assert verification["objective_value"] == pytest.approx(cost_config().value(34, 1))
    # The compatibility wrapper still uses historical distance ranking.
    legacy_info = {**info, "objective_config": ObjectiveConfig().to_dict()}
    legacy_selected, _, _ = select_min_verified_objective(instance, legacy_info)
    assert legacy_selected == 0


@pytest.mark.parametrize("invalid_route", [[[0, 1, 0]], [[0, -1, 0]], [[0, 99, 0]]])
def test_cost_selection_skips_cheaper_environment_success_that_fails_verifier(invalid_route):
    info = candidate_info([invalid_route, [[0, 1, 2, 0]]])
    selected, _, verification = select_min_verified_objective(selection_instance(), info)
    assert selected == 1
    assert verification["passed"]


def test_paid_charger_only_trip_and_interior_depot_departures_count_in_replayed_cost():
    instance = economic_instance()
    info = candidate_info([[[0, 3, 0], [0, 1, 0, 2, 0]]])
    _, _, verification = select_min_verified_objective(instance, info)
    assert verification["passed"]
    assert verification["vehicles_started"] == 3
    assert verification["objective_distance_km"] == 19
    assert verification["vehicle_cost_usd"] == pytest.approx(3 * 33.56)


def test_cost_failure_cannot_become_verified_success_through_virtual_depot_return():
    info = candidate_info(
        [[[0, 1, 2, 0]]], success=[False], served=[2], distance=[32], vehicles=[1],
    )
    selected, _, verification = select_min_verified_objective(selection_instance(), info)
    assert selected == 0
    assert not verification["passed"]
    assert verification["route_verifier_passed"]  # Virtual final edge was feasible.
    assert verification["objective_distance_km"] == 32  # Not replayed 34 km.
    assert verification["objective_value"] == pytest.approx(cost_config().value(32, 1))
    summary = verified_validation([selection_instance()], lambda *_: info, seed=3)
    assert summary["complete_and_feasible"] == 0
    assert summary["mean_verified_objective"] is None
    assert summary["mean_verified_cost_usd"] is None
    assert not summary["rows"][0]["environment_success"]
    assert not summary["rows"][0]["verifier_passed"]


def test_failure_fallback_prioritizes_service_then_incurred_cost_including_open_vehicles():
    info = candidate_info(
        [[[0, 1, 0]], [[0, 1, 0], [0, 2, 0]], [[0, 1, 2, 0]]],
        success=[False, False, False], served=[1, 2, 2],
        distance=[0.1, 10, 32], vehicles=[1, 2, 1],
    )
    selected, _, verification = select_min_verified_objective(selection_instance(), info)
    assert selected == 2
    assert not verification["passed"]
    assert verification["vehicles_started"] == 1
    assert verification["objective_distance_km"] == 32


@pytest.mark.parametrize("cached_total", [None, [-1.0, 9999.0]])
@pytest.mark.parametrize("keep_vehicle_ledger", [False, True])
def test_failure_fallback_recomputes_incurred_cost_when_cached_total_is_missing_or_stale(
    cached_total, keep_vehicle_ledger,
):
    info = candidate_info(
        [[[0, 1, 0], [0, 2, 0]], [[0, 1, 2, 0]]],
        success=[False, False], served=[2, 2], distance=[10, 32], vehicles=[2, 1],
    )
    if cached_total is None:
        info.pop("objective_value")
    else:
        info["objective_value"] = np.asarray(cached_total)
    if not keep_vehicle_ledger:
        info.pop("vehicles_started")
    selected, _, verification = select_min_verified_objective(selection_instance(), info)
    assert selected == 1
    assert not verification["passed"]
    assert verification["objective_distance_km"] == 32
    assert verification["vehicles_started"] == 1
    assert verification["objective_value"] == pytest.approx(cost_config().value(32, 1))


def test_validation_averages_only_verified_candidates_and_ranks_feasibility_then_usd():
    instance = selection_instance()
    successful = candidate_info([[[0, 1, 2, 0]]], distance=[34], vehicles=[1])
    failed = candidate_info([[[0, 1, 2, 0]]], success=[False], distance=[32], vehicles=[1])
    summary = verified_validation(
        [instance, instance], lambda _instance, seed: successful if seed == 5 else failed, seed=5,
    )
    assert summary["objective_unit"] == "USD"
    assert summary["complete_and_feasible_rate"] == 0.5
    assert summary["mean_verified_distance_km"] == 34
    assert summary["mean_verified_vehicle_count"] == 1
    assert summary["mean_verified_objective"] == pytest.approx(cost_config().value(34, 1))
    assert summary["mean_verified_cost_usd"] == summary["mean_verified_objective"]
    assert summary["mean_verified_cost_usd"] == pytest.approx(
        summary["mean_verified_electricity_cost_usd"] + summary["mean_verified_vehicle_cost_usd"]
    )
    more_feasible = {**summary, "complete_and_feasible_rate": 1.0, "mean_verified_objective": 1e6}
    assert validation_key(more_feasible) > validation_key(summary)
    cheaper = {**summary, "mean_verified_objective": 1.0, "mean_verified_distance_km": 1e6}
    assert validation_key(cheaper) > validation_key(summary)
    assert validation_key({"complete_and_feasible_rate": 1, "mean_verified_distance_km": 5}) == (1, -5)


def test_distance_wrapper_keeps_existing_selection_verifier_fields_and_kilometre_units():
    info = candidate_info(
        [[[0, 1, 0], [0, 2, 0]], [[0, 1, 2, 0]]],
        distance=[10, 34], objective=ObjectiveConfig(),
    )
    expected_index, expected_routes, expected_verification = select_min_verified_distance(selection_instance(), info)
    index, routes, verification = select_min_verified_objective(selection_instance(), info)
    assert index == expected_index
    assert routes == expected_routes
    for key, value in expected_verification.items():
        assert verification[key] == value
    assert verification["objective_unit"] == "km"
    assert verification["objective_value"] == verification["objective_distance_km"]
    assert verification["objective_cost_usd"] is None


def test_validation_rejects_mixed_objective_configs_in_one_cohort():
    info = candidate_info([[[0, 1, 2, 0]]])
    changed = replace(cost_config(), vehicle_fixed_cost_usd=1.0)
    second = {**info, "objective_config": changed.to_dict()}
    with pytest.raises(ValueError, match="mixes objective"):
        verified_validation([selection_instance(), selection_instance()], lambda _, seed: info if seed == 9 else second, seed=9)


@pytest.mark.parametrize("override", [ObjectiveConfig(), replace(cost_config(), vehicle_fixed_cost_usd=1.0)])
def test_explicit_objective_must_match_info_declared_contract(override):
    info = candidate_info([[[0, 1, 2, 0]]])
    with pytest.raises(ValueError, match="objective.*match"):
        select_min_verified_objective(selection_instance(), info, override)
    with pytest.raises(ValueError, match="objective.*match"):
        verified_validation([selection_instance()], lambda *_: info, seed=1, objective_config=override)
