"""Regression contracts for final D_time training without relabelling archives."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import (
    objective_from_args, objective_from_checkpoint, select_objective_instance,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.evaluation import select_min_verified_objective
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_protocol import verified_validation
from EVRPTW_Benchmark.Reinforcement_Learning.common import stage2_data
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import EVRPTWVectorEnv, EVRPTWVectorEnvFast
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.env import DRLTSHardConstraintEnv
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.soft_env import DRLTSSoftConstraintEnv
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.rollout import normalized_edge_matrices
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env.tests.test_objective_cost import economic_instance, cost_objective
from EVRPTW_Benchmark.Reinforcement_Learning.common.tests.test_common_cost_objective import candidate_info


SOURCE = "running_time_path_distance_km"


def dtime_objective():
    return replace(cost_objective(), objective_distance_source=SOURCE)


def divergent_instance():
    original = economic_instance()
    dtime = original.distance_matrix_km.copy()
    # The shortest-distance order 0->1->2->0 costs 8 km. Under the actual
    # driven path its long middle arc reverses the two one-vehicle candidates.
    dtime[1, 2] = 80
    dtime[2, 1] = 1
    return replace(original, raw={**original.raw, SOURCE: dtime})


def test_legacy_snapshots_stay_identical_and_dtime_resume_mismatch_is_rejected():
    old = cost_objective()
    saved = old.to_dict()
    assert "objective_distance_source" not in saved
    assert objective_from_checkpoint({"objective_config": saved}).to_dict() == saved
    new = dtime_objective()
    assert new.to_dict()["objective_distance_source"] == SOURCE
    with pytest.raises(ValueError, match="objective mismatch"):
        objective_from_checkpoint({"objective_config": saved}, override=new)
    assert objective_from_args(SimpleNamespace(objective=saved, objective_distance_source=SOURCE)) == new
    assert objective_from_checkpoint({"objective_config": new.to_dict()}) == new


def test_mapping_is_nonmutating_and_can_explicitly_restore_legacy_distance():
    original = divergent_instance()
    mapped = select_objective_instance(original, dtime_objective())
    np.testing.assert_equal(mapped.distance_matrix_km, original.raw[SOURCE])
    assert original.distance_matrix_km[1, 2] == 4
    assert mapped.distance_matrix_km[1, 2] == 80
    assert mapped.energy_matrix_kwh is original.energy_matrix_kwh
    assert mapped.shortest_time_matrix_s is original.shortest_time_matrix_s
    assert mapped.metadata["objective_distance_source"] == SOURCE
    restored = select_objective_instance(mapped, cost_objective())
    np.testing.assert_equal(restored.distance_matrix_km, original.distance_matrix_km)
    assert SOURCE not in original.metadata


def test_no_silent_shortest_distance_fallback():
    with pytest.raises(ValueError, match="requires running_time_path_distance_km"):
        select_objective_instance(economic_instance(), dtime_objective())


@pytest.mark.parametrize("env_cls", [EVRPTWVectorEnv, EVRPTWVectorEnvFast, DRLTSHardConstraintEnv, DRLTSSoftConstraintEnv])
def test_all_policy_environment_paths_price_dtime_but_keep_resource_matrices(env_cls):
    instance = divergent_instance()
    kwargs = {"use_jit_mask": False} if issubclass(env_cls, EVRPTWVectorEnvFast) else {}
    env = env_cls(instance, n_traj=1, objective_config=dtime_objective(), normalize_reward=False, **kwargs)
    obs, _ = env.reset(seed=1)
    np.testing.assert_equal(env.distance_km, instance.raw[SOURCE])
    np.testing.assert_equal(env.energy_kwh, instance.energy_matrix_kwh)
    np.testing.assert_equal(env.travel_time_s, instance.shortest_time_matrix_s)
    relations, _, _ = normalized_edge_matrices([env])
    np.testing.assert_allclose(relations[0], instance.raw[SOURCE] / env.reward_distance_scale_km)
    for action in [1, 2, 0]:
        _, _, _, _, info = env.step(np.array([action]))
    assert info["success"][0]
    assert info["objective_distance_km"][0] == 84
    assert info["objective_cost_usd"][0] == pytest.approx(dtime_objective().value(84, 1))
    assert info["objective_config"]["objective_distance_source"] == SOURCE


def test_selection_and_periodic_validation_recompute_same_dtime_objective():
    original = divergent_instance()
    routes = [[[0, 1, 2, 0]], [[0, 2, 1, 0]]]
    objective = dtime_objective()
    # Environment metadata is deliberately inconsistent; verifier computes the
    # two route costs from D_time, then selects candidate 1 (7 rather than 84).
    info = candidate_info(routes, distance=[8, 9], objective=objective)
    index, selected, verification = select_min_verified_objective(original, info, objective)
    assert index == 1
    assert selected == routes[1]
    assert verification["passed"]
    assert verification["objective_distance_km"] == 7
    assert verification["objective_value"] == pytest.approx(objective.value(7, 1))
    result = verified_validation([original], lambda instance, seed: info, seed=123, objective_config=objective, cuda_rng_devices=[])
    assert result["complete_and_feasible"] == 1
    assert result["mean_verified_cost_usd"] == pytest.approx(objective.value(7, 1))
    assert result["objective_config"]["objective_distance_source"] == SOURCE


def test_training_pool_calibration_and_cached_instances_use_new_matrix(monkeypatch, tmp_path):
    instance = divergent_instance()
    task = SimpleNamespace(view_id="one", scale_label="Cus100", split_id="train", track_id="train", city_slug="test", terminal_count=5, customer_count=2)
    monkeypatch.setattr(stage2_data, "is_synthetic_index", lambda path: False)
    monkeypatch.setattr(stage2_data, "read_stage2_tasks", lambda *a, **kw: [task])
    monkeypatch.setattr(stage2_data, "load_stage2_instance", lambda task: instance)
    pool = stage2_data.Stage2TaskPool(tmp_path / "index.parquet", objective_config=dtime_objective())
    mapped = pool.instance(task)
    assert pool.instance(task) is mapped
    assert mapped.metadata["objective_distance_source"] == SOURCE
    assert pool.reward_distance_scale_km("max_edge") == 80
    assert instance.distance_matrix_km.max() == 5
