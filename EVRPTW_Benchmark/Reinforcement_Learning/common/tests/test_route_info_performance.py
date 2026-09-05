from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import (
    _instance,
)
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.env import DRLTSHardConstraintEnv
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.soft_env import DRLTSSoftConstraintEnv
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import EVRPTWVectorEnvFast
from EVRPTW_Benchmark.Reinforcement_Learning.common import protocol_entrypoints
from EVRPTW_Benchmark.Reinforcement_Learning.common.evaluation import (
    select_min_verified_distance,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.route_info import finalize_route_infos


def _assert_info_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        if isinstance(expected[key], np.ndarray):
            np.testing.assert_array_equal(actual[key], expected[key])
        else:
            assert actual[key] == expected[key]


@pytest.mark.parametrize(
    "env_cls", [EVRPTWVectorEnvFast, DRLTSHardConstraintEnv, DRLTSSoftConstraintEnv]
)
@pytest.mark.parametrize("actions", [[], [1], [1, 3], [1, 3, 2, 0]])
def test_final_only_route_info_matches_full_at_any_rollout_limit(
    env_cls, actions, monkeypatch
):
    full = env_cls(_instance(), n_traj=3, info_level="full", use_jit_mask=False)
    light = env_cls(_instance(), n_traj=3, info_level="light", use_jit_mask=False)
    route_exports = []
    original_get_routes = light.get_routes

    def tracked_routes():
        route_exports.append(1)
        return original_get_routes()

    monkeypatch.setattr(light, "get_routes", tracked_routes)
    _, full_info = full.reset(seed=7)
    _, light_info = light.reset(seed=7)
    for action in actions:
        destinations = np.full(3, action, dtype=np.int64)
        full_step = full.step(destinations)
        light_step = light.step(destinations)
        for actual, expected in zip(light_step[1:4], full_step[1:4]):
            np.testing.assert_array_equal(actual, expected)
        full_info, light_info = full_step[-1], light_step[-1]
    assert route_exports == []
    assert "routes" not in light_info

    finalized = finalize_route_infos([light], [light_info])[0]
    assert route_exports == [1]
    assert "routes" not in light_info  # Do not mutate retained light info.
    _assert_info_equal(finalized, full_info)
    for key, value in light_info.items():
        assert finalized[key] is value  # Includes soft-penalty and source fields.
    assert select_min_verified_distance(_instance(), finalized) == (
        select_min_verified_distance(_instance(), full_info)
    )


def test_final_route_info_checks_environment_count():
    with pytest.raises(ValueError, match="one final info"):
        finalize_route_infos([object()], [])


@pytest.mark.parametrize(
    ("runner", "module_name"),
    [
        (protocol_entrypoints.run_am, "AM_EVRPTW"),
        (protocol_entrypoints.run_evrptw_rl, "EVRPTW_RL"),
        (protocol_entrypoints.run_drl_ts, "DRL_TS"),
    ],
)
@pytest.mark.parametrize("validation_decode", ["greedy", "sampling"])
def test_protocol_training_baseline_is_light_validation_exports_final_once(
    runner, module_name, validation_decode, monkeypatch
):
    instance = _instance()
    observations = []
    export_count = []
    rollout_results = []
    get_routes = EVRPTWVectorEnvFast.get_routes

    def tracked_routes(env):
        export_count.append(1)
        return get_routes(env)

    monkeypatch.setattr(EVRPTWVectorEnvFast, "get_routes", tracked_routes)

    def scripted_rollout(_policy, envs, **kwargs):
        assert all(env.info_level == "light" for env in envs)
        is_actor = len(observations) in {0, 2}
        assert kwargs["compute_log_likelihood"] is is_actor
        infos = []
        for env in envs:
            _, info = env.reset(seed=kwargs["seed"])
            for destination in [1, 3, 2, 0]:
                _, _, _, _, info = env.step(np.full(env.n_traj, destination))
            assert "routes" not in info
            infos.append(info)
        observations.append((kwargs["decode_type"], envs[0].n_traj))
        result = SimpleNamespace(infos=infos, runtime_s=3.0)
        rollout_results.append(result)
        return result

    module = importlib.import_module(
        f"EVRPTW_Benchmark.Reinforcement_Learning.{module_name}.rollout"
    )
    monkeypatch.setattr(module, "rollout", scripted_rollout)

    def inspect_callbacks(**callbacks):
        for soft in [False, True]:
            actor = callbacks["make_actor"]([instance], soft, 11)
            baseline = callbacks["make_baseline"](object(), [instance], soft, 11)
            assert all("routes" not in info for info in actor.infos + baseline.infos)
            assert actor.runtime_s == baseline.runtime_s == 3.0
        assert not export_count
        ticks = iter([10.0, 10.25])
        monkeypatch.setattr(
            protocol_entrypoints, "time", SimpleNamespace(perf_counter=lambda: next(ticks))
        )
        final = callbacks["validation_solve"](object(), instance, 17)
        assert rollout_results[-1].runtime_s == 3.25
        assert len(export_count) == 1
        assert len(final["routes"]) == candidates
        assert select_min_verified_distance(instance, final)[2]["passed"]

    monkeypatch.setattr(
        protocol_entrypoints, "train_reinforce_data_passes", inspect_callbacks
    )
    candidates = 100 if validation_decode == "sampling" else 1
    args = SimpleNamespace(
        training_rollout_steps=80,
        validation_decode_type=validation_decode,
        validation_candidates=candidates,
        samples_per_instance=1,
        incomplete_penalty_km=100.0,
        incomplete_penalty=100.0,
        station_visit_penalty=0.3,
        capacity_penalty=1.0,
        time_penalty=1.0,
        energy_penalty=1.0,
        batch_size=1,
        soft_stage_fraction=0.5,
    )
    pool = SimpleNamespace(
        reward_distance_scale_km=lambda _mode: 1.0,
        reward_scale_metadata={},
    )
    runner(args, pool, object(), object())
    assert observations[-1] == (validation_decode, candidates)
