from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "EVRPTW_Core"))

from EVRPTW_Benchmark.Exact.Gurobi_Solver.route_validator import validate_routes
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env.tests.test_canonical_contract import canonical_instance
from EVRPTW_Benchmark.Reinforcement_Learning.common.action_constraints import (
    ACTION_CONSTRAINT_CONTRACT_ID, consecutive_cs_arcs, require_checkpoint_action_contract,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.candidate_protocol import merge_single_candidate_infos
from EVRPTW_Benchmark.Reinforcement_Learning.common.evaluation import select_min_verified_objective
from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import ObjectiveConfig
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_protocol import verified_validation


def two_station_instance():
    instance = canonical_instance()
    matrix = np.ones((4, 4), dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    return replace(
        instance, charging_stations=np.asarray([[0.5, 0.0], [0.6, 0.0]]),
        distance_matrix_km=matrix, shortest_time_matrix_s=matrix * 10,
        energy_matrix_kwh=matrix * 0.1, cs_time_to_depot_s=np.asarray([10.0, 10.0]),
        raw={**instance.raw, "charging_power_kw": np.asarray([20.0, 20.0])},
    )


@pytest.mark.parametrize("mode", ["distance", "energy_vehicle_cost"])
def test_drl_rejects_physically_valid_consecutive_cs_without_changing_physical_validator(mode):
    instance = two_station_instance()
    prohibited = [[0, 1, 2, 3, 0]]
    allowed = [[0, 2, 1, 3, 0]]
    assert validate_routes(instance, prohibited)["passed"]
    assert validate_routes(instance, allowed)["passed"]
    info = {
        "success": np.asarray([True, True]), "served_customers": np.asarray([1, 1]),
        "objective_distance_km": np.asarray([4.0, 4.0]),
        "routes": [prohibited, allowed], "vehicles_started": np.asarray([1, 1]),
    }
    config = ObjectiveConfig(mode=mode)
    selected, routes, verification = select_min_verified_objective(instance, info, config)
    assert selected == 1
    assert routes == allowed
    assert verification["passed"] and verification["physical_verifier_passed"]
    assert verification["drl_policy_passed"]
    assert verification["action_constraint_contract_id"] == ACTION_CONSTRAINT_CONTRACT_ID
    for key in ("success", "served_customers", "objective_distance_km", "vehicles_started", "routes"):
        info[key] = info[key][:1]
    _, _, verification = select_min_verified_objective(instance, info, config)
    assert verification["physical_verifier_passed"]
    assert not verification["drl_policy_passed"]
    assert not verification["passed"]
    assert any("consecutive CS" in item for item in verification["violations"])
    summary = verified_validation([instance], lambda *_: info, seed=1234, objective_config=config)
    assert summary["complete_and_feasible_rate"] == 0.0
    assert summary["mean_verified_objective"] is None
    assert summary["action_constraint_contract_id"] == ACTION_CONSTRAINT_CONTRACT_ID


def test_station_constraint_checks_adjacent_service_terminals_not_entire_routes():
    instance = two_station_instance()
    assert consecutive_cs_arcs(instance, [[0, 2, 1, 3, 0]]) == []
    assert consecutive_cs_arcs(instance, [[0, 2, 0], [0, 3, 0]]) == []
    assert consecutive_cs_arcs(instance, [[0, 1, 2, 2, 3, 0]]) == [(0, 2, 2), (0, 2, 3)]


@pytest.mark.parametrize("container", ["top", "args", "config"])
def test_checkpoint_policy_is_explicit_and_consistent(container):
    field = "action_constraint_contract_id"
    checkpoint = {field: ACTION_CONSTRAINT_CONTRACT_ID} if container == "top" else {
        container: {field: ACTION_CONSTRAINT_CONTRACT_ID}
    }
    require_checkpoint_action_contract(checkpoint)
    checkpoint["config" if container != "config" else "args"] = {field: "legacy_cs_chain"}
    with pytest.raises(ValueError, match="action constraint contract mismatch"):
        require_checkpoint_action_contract(checkpoint)


def test_missing_checkpoint_policy_is_not_silently_upgraded():
    with pytest.raises(ValueError, match="action constraint contract mismatch"):
        require_checkpoint_action_contract({})


def test_candidate_merge_preserves_scalar_action_policy_and_rejects_mixing():
    one = {"action_constraint_contract_id": ACTION_CONSTRAINT_CONTRACT_ID, "success": np.asarray([True])}
    merged = merge_single_candidate_infos([one, one])
    assert merged["action_constraint_contract_id"] == ACTION_CONSTRAINT_CONTRACT_ID
    assert merged["success"].tolist() == [True, True]
    with pytest.raises(ValueError):
        merge_single_candidate_infos([one, {**one, "action_constraint_contract_id": "old"}])
