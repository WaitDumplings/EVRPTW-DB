from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from evrptw_core.schema import EVRPTWInstance

from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import (
    EVRPTWVectorEnv,
    EVRPTWVectorEnvFast,
)
from EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.env import (
    DRLTSHardConstraintEnv,
)


def _ffp_instance(energy_kwh: np.ndarray, *, instance_id: str) -> EVRPTWInstance:
    """One-customer test instance with the remaining nodes as stations."""

    energy = np.asarray(energy_kwh, dtype=np.float32)
    num_nodes = int(energy.shape[0])
    assert energy.shape == (num_nodes, num_nodes)
    num_stations = num_nodes - 2
    travel_time = energy * 10.0
    np.fill_diagonal(travel_time, 0.0)
    distance = energy.copy()
    np.fill_diagonal(distance, 0.0)
    return EVRPTWInstance(
        instance_id=instance_id,
        region_id="test",
        mother_board_id="test",
        operating_day_id="test",
        day_type="weekday",
        working_start_s=0.0,
        working_end_s=100_000.0,
        depot=np.asarray([0.0, 0.0], dtype=np.float32),
        customers=np.asarray([[1.0, 0.0]], dtype=np.float32),
        charging_stations=np.stack(
            [
                np.linspace(0.25, 0.75, num_stations, dtype=np.float32),
                np.zeros(num_stations, dtype=np.float32),
            ],
            axis=1,
        ),
        distance_matrix_km=distance,
        demands_cm3=np.asarray([1.0], dtype=np.float32),
        package_counts=np.asarray([1], dtype=np.int32),
        service_time_s=np.asarray([0.0], dtype=np.float32),
        tw_s=np.asarray([[0.0, 100_000.0]], dtype=np.float32),
        cs_time_to_depot_s=travel_time[2:, 0].copy(),
        vehicle={
            "battery_capacity_kwh": 10.0,
            "cargo_capacity_cm3": 10.0,
            "specific_energy_consumption_kwh_per_km": 1.0,
        },
        shortest_time_matrix_s=travel_time,
        energy_matrix_kwh=energy,
        raw={
            "charging_power_kw": np.full(num_stations, 3600.0, dtype=np.float32),
            "charging_policy": {"charging_power_derating_factor": 1.0},
        },
    )


def _visited_intermediate_instance() -> EVRPTWInstance:
    # Nodes: depot, customer, station A, station B.  Once B has already been
    # used, the static A -> B -> depot witness is no longer executable.
    energy = np.full((4, 4), 50.0, dtype=np.float32)
    np.fill_diagonal(energy, 0.0)
    energy[0, 3] = 2.0
    energy[3, 0] = 2.0
    energy[3, 1] = 2.0
    energy[1, 2] = 7.0
    energy[2, 3] = 6.0
    return _ffp_instance(energy, instance_id="ffp_visited_intermediate")


def _consecutive_station_chain_instance() -> EVRPTWInstance:
    # The only return after serving the customer is customer -> A -> B -> depot.
    energy = np.full((4, 4), 50.0, dtype=np.float32)
    np.fill_diagonal(energy, 0.0)
    energy[0, 1] = 2.0
    energy[1, 2] = 7.0
    energy[2, 3] = 6.0
    energy[3, 0] = 6.0
    return _ffp_instance(energy, instance_id="ffp_consecutive_station_chain")


def _pre_customer_station_instance() -> EVRPTWInstance:
    # Reaching the customer requires a depot -> station prefix. Once at the
    # station the customer is feasible, so the normal no-empty-route gate must
    # keep depot masked rather than introduce a two-step charger-only loop.
    energy = np.full((3, 3), 50.0, dtype=np.float32)
    np.fill_diagonal(energy, 0.0)
    energy[0, 2] = 2.0
    energy[2, 0] = 2.0
    energy[2, 1] = 2.0
    energy[1, 0] = 2.0
    return _ffp_instance(energy, instance_id="ffp_pre_customer_station")


def _randomized_ffp_instance(seed: int) -> EVRPTWInstance:
    rng = np.random.default_rng(seed)
    num_customers = 3
    num_stations = 5
    num_nodes = 1 + num_customers + num_stations
    energy = rng.uniform(1.0, 14.0, size=(num_nodes, num_nodes)).astype(
        np.float32
    )
    np.fill_diagonal(energy, 0.0)
    # Keep one reusable route-local hub physically reachable from the depot so
    # every reset state has at least one legal action.  Other randomized edges
    # still exercise direct, one-station, and multi-station return witnesses.
    hub = num_nodes - 1
    energy[0, hub] = 3.0
    energy[hub, 0] = 3.0
    travel_time = energy * rng.uniform(
        7.0, 13.0, size=(num_nodes, num_nodes)
    ).astype(np.float32)
    np.fill_diagonal(travel_time, 0.0)
    return EVRPTWInstance(
        instance_id=f"ffp_random_{seed}",
        region_id="test",
        mother_board_id="test",
        operating_day_id="test",
        day_type="weekday",
        working_start_s=0.0,
        working_end_s=100_000.0,
        depot=np.asarray([0.0, 0.0], dtype=np.float32),
        customers=rng.random((num_customers, 2), dtype=np.float32),
        charging_stations=rng.random((num_stations, 2), dtype=np.float32),
        distance_matrix_km=energy.copy(),
        demands_cm3=np.ones(num_customers, dtype=np.float32),
        package_counts=np.ones(num_customers, dtype=np.int32),
        service_time_s=np.zeros(num_customers, dtype=np.float32),
        tw_s=np.tile(
            np.asarray([[0.0, 100_000.0]], dtype=np.float32),
            (num_customers, 1),
        ),
        cs_time_to_depot_s=travel_time[1 + num_customers :, 0].copy(),
        vehicle={
            "battery_capacity_kwh": 10.0,
            "cargo_capacity_cm3": 3.0,
            "specific_energy_consumption_kwh_per_km": 1.0,
        },
        shortest_time_matrix_s=travel_time,
        energy_matrix_kwh=energy,
        raw={
            "charging_power_kw": rng.uniform(
                500.0, 1500.0, size=num_stations
            ).astype(np.float32),
            "charging_policy": {"charging_power_derating_factor": 0.9},
        },
    )


def _env_pair(instance: EVRPTWInstance):
    reference = EVRPTWVectorEnv(instance, n_traj=1, max_steps_factor=20)
    fast = EVRPTWVectorEnvFast(
        instance,
        n_traj=1,
        max_steps_factor=20,
        use_jit_mask=True,
    )
    reference_obs, _ = reference.reset(seed=31)
    fast_obs, _ = fast.reset(seed=31)
    np.testing.assert_array_equal(
        reference_obs["action_mask"], fast_obs["action_mask"]
    )
    return reference, fast, reference_obs, fast_obs


def _assert_every_admitted_action_has_a_successor(env: EVRPTWVectorEnv) -> None:
    """Exercise the FFP invariant without mutating the supplied state."""

    mask = env._compute_action_mask()[0]
    assert mask.any()
    for action in np.flatnonzero(mask):
        branch = copy.deepcopy(env)
        # The fast environment validates against its cached pre-action mask.
        if isinstance(branch, EVRPTWVectorEnvFast):
            branch._current_action_mask = mask.reshape(1, -1).copy()
        observation, _, terminated, truncated, info = branch.step(
            np.asarray([action], dtype=np.int64)
        )
        assert info["failure_reason"][0] != "no_feasible_action"
        assert not truncated[0], info["failure_reason"][0]
        if not terminated[0] and not truncated[0]:
            assert observation["action_mask"][0].any()


def test_ffp_removes_visited_station_from_the_entire_return_witness() -> None:
    instance = _visited_intermediate_instance()
    reference, fast, reference_obs, fast_obs = _env_pair(instance)
    station_a, station_b = 2, 3
    assert reference_obs["action_mask"][0, station_b]

    for env in (reference, fast):
        observation, _, terminated, truncated, info = env.step(
            np.asarray([station_b], dtype=np.int64)
        )
        assert not terminated[0]
        assert not truncated[0]
        assert info["failure_reason"][0] == "in_progress"
        # A charger-only route can always close when its direct depot leg is
        # feasible; previously route_has_customer incorrectly blocked this.
        assert observation["action_mask"][0, 0]
        # Static FFP sees A -> B -> depot, but B is already unavailable on this
        # route, so admitting the customer would create a dead end.
        assert not observation["action_mask"][0, 1]

    np.testing.assert_array_equal(
        reference._compute_action_mask(), fast._compute_action_mask()
    )
    assert reference._shortest_stop_time(station_a, 0) is not None
    assert np.isinf(reference._route_local_stop_to_depot_times(0)[station_a])
    _assert_every_admitted_action_has_a_successor(reference)
    _assert_every_admitted_action_has_a_successor(fast)


def test_ffp_preserves_legal_consecutive_unvisited_station_chain() -> None:
    instance = _consecutive_station_chain_instance()
    reference, fast, reference_obs, fast_obs = _env_pair(instance)
    customer, station_a, station_b = 1, 2, 3
    assert reference_obs["action_mask"][0, customer]
    assert fast_obs["action_mask"][0, customer]

    for expected_action in (customer, station_a, station_b, 0):
        for env in (reference, fast):
            _assert_every_admitted_action_has_a_successor(env)
            assert env._compute_action_mask()[0, expected_action]
        reference_obs, _, reference_done, reference_truncated, reference_info = (
            reference.step(np.asarray([expected_action], dtype=np.int64))
        )
        fast_obs, _, fast_done, fast_truncated, fast_info = fast.step(
            np.asarray([expected_action], dtype=np.int64)
        )
        np.testing.assert_array_equal(
            reference_obs["action_mask"], fast_obs["action_mask"]
        )
        np.testing.assert_array_equal(reference_done, fast_done)
        np.testing.assert_array_equal(reference_truncated, fast_truncated)
        np.testing.assert_array_equal(
            reference_info["failure_reason"], fast_info["failure_reason"]
        )

    assert reference_done[0] and fast_done[0]
    assert not reference_truncated[0] and not fast_truncated[0]
    assert reference_info["routes"][0] == [[0, 1, 2, 3, 0]]
    assert fast_info["routes"][0] == [[0, 1, 2, 3, 0]]


def test_drl_ts_hard_ffp_does_not_use_a_forbidden_two_station_witness() -> None:
    instance = _consecutive_station_chain_instance()
    customer = 1

    # The shared default remains permissive for TERRAN and can execute the
    # customer -> A -> B -> depot witness.
    shared = EVRPTWVectorEnv(instance, n_traj=1)
    shared_observation, _ = shared.reset(seed=32)
    assert shared.allow_consecutive_station_actions is True
    assert shared_observation["action_mask"][0, customer]

    # Both reference Python and fast/JIT paths must apply the restricted
    # witness before exposing the customer action.
    restricted_reference = EVRPTWVectorEnv(
        instance,
        n_traj=1,
        allow_consecutive_station_actions=False,
    )
    restricted_fast = EVRPTWVectorEnvFast(
        instance,
        n_traj=1,
        use_jit_mask=True,
        allow_consecutive_station_actions=False,
    )
    reference_observation, _ = restricted_reference.reset(seed=32)
    fast_observation, _ = restricted_fast.reset(seed=32)
    np.testing.assert_array_equal(
        reference_observation["action_mask"], fast_observation["action_mask"]
    )
    assert not reference_observation["action_mask"][0, customer]
    assert restricted_reference.truncated[0]
    assert restricted_fast.truncated[0]
    assert restricted_reference.failure_reason[0] == "no_feasible_action"
    assert restricted_fast.failure_reason[0] == "no_feasible_action"
    assert reference_observation["action_mask"][0, 0]
    assert fast_observation["action_mask"][0, 0]

    # DRL-TS hard mode freezes the restricted witness policy and retains its
    # paper mask: no station action directly from the depot.
    for use_jit_mask in (False, True):
        hard = DRLTSHardConstraintEnv(
            instance,
            n_traj=1,
            use_jit_mask=use_jit_mask,
        )
        observation, _ = hard.reset(seed=32)
        assert hard.allow_consecutive_station_actions is False
        assert hard.truncated[0]
        assert hard.failure_reason[0] == "no_feasible_action"
        assert observation["action_mask"][0, 0]
        assert not observation["action_mask"][0, customer]
        assert not observation["action_mask"][0, hard.station_nodes].any()


def test_charger_only_depot_escape_is_exposed_only_to_avoid_empty_mask() -> None:
    reference, fast, reference_obs, fast_obs = _env_pair(
        _pre_customer_station_instance()
    )
    customer, station = 1, 2
    assert reference_obs["action_mask"][0, station]
    assert fast_obs["action_mask"][0, station]

    for env in (reference, fast):
        observation, _, terminated, truncated, _ = env.step(
            np.asarray([station], dtype=np.int64)
        )
        assert not terminated[0]
        assert not truncated[0]
        assert observation["action_mask"][0, customer]
        assert not observation["action_mask"][0, 0]


def test_python_and_jit_masks_preserve_ffp_successor_property_on_valid_states() -> None:
    for seed in range(4):
        rng = np.random.default_rng(10_000 + seed)
        reference, fast, _, _ = _env_pair(_randomized_ffp_instance(seed))
        for _ in range(24):
            reference_mask = reference._compute_action_mask()
            fast_mask = fast._compute_action_mask()
            np.testing.assert_array_equal(reference_mask, fast_mask)
            _assert_every_admitted_action_has_a_successor(reference)
            _assert_every_admitted_action_has_a_successor(fast)

            choices = np.flatnonzero(reference_mask[0])
            non_depot = choices[choices != 0]
            if non_depot.size and rng.random() < 0.8:
                choices = non_depot
            action = int(rng.choice(choices))
            reference_obs, _, reference_done, reference_truncated, reference_info = (
                reference.step(np.asarray([action], dtype=np.int64))
            )
            fast_obs, _, fast_done, fast_truncated, fast_info = fast.step(
                np.asarray([action], dtype=np.int64)
            )
            np.testing.assert_array_equal(
                reference_obs["action_mask"], fast_obs["action_mask"]
            )
            np.testing.assert_array_equal(reference_done, fast_done)
            np.testing.assert_array_equal(reference_truncated, fast_truncated)
            np.testing.assert_array_equal(
                reference_info["failure_reason"], fast_info["failure_reason"]
            )
            assert reference_info["failure_reason"][0] != "no_feasible_action"
            if reference_done[0] or reference_truncated[0]:
                break
