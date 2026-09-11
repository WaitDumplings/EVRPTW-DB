"""Regression cases for observed TERRAN/canonical data-contract mismatches."""
import numpy as np

from EVRPTW_Benchmark.Reinforcement_Learning.common.terran_synthetic_feasibility import (
    native_singleton_candidate_feasible,
)


def feasible(customer, stops, ready=0, due=100, battery=7, service=1, horizon=100):
    return native_singleton_candidate_feasible(
        np.asarray(customer), np.asarray(stops), ready, due,
        velocity=1, consumption=1, battery=battery, charge_rate=10,
        service=service, horizon=horizon,
    )


def test_upstream_inverted_window_is_rejected_before_route_admission():
    assert not feasible([1, 0], [[0, 0], [2, 0]], ready=276.20415657468163,
                        due=240, battery=79.69, service=10, horizon=240)


def test_waiting_until_window_open_must_leave_time_for_service_and_return():
    # The previous 2*travel+service check accepts this, despite mandatory waiting.
    assert not feasible([1, 0], [[0, 0], [2, 0]], ready=98, due=100,
                        battery=79.69, service=2, horizon=100)
    assert feasible([1, 0], [[0, 0], [2, 0]], ready=95, due=100,
                    battery=79.69, service=2, horizon=100)


def test_repeated_station_only_witness_is_not_executable_by_canonical_env():
    # Depot->station->customer->same station->depot is physically possible,
    # but canonical trajectories cannot visit one station twice within a route.
    assert not feasible([9, 0], [[0, 0], [6, 0]])
    # A distinct return station supplies a valid canonical witness.
    assert feasible([9, 0], [[0, 0], [6, 0], [6.5, 0]])
