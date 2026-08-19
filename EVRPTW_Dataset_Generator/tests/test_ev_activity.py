import numpy as np

from evrptw_stage2.ev_activity import run_deterministic_route_heuristic


def _case(distance: np.ndarray, battery: float):
    n = 2
    return run_deterministic_route_heuristic(
        time_matrix_s=distance * 60.0,
        distance_matrix_km=distance,
        demands_cm3=np.array([1.0, 1.0]),
        service_time_s=np.array([0.0, 0.0]),
        time_windows_s=np.array([[0.0, 100000.0], [0.0, 100000.0]]),
        charging_power_kw=np.array([100.0]),
        battery_capacity_kwh=battery,
        cargo_capacity_cm3=10.0,
        specific_energy_kwh_per_km=1.0,
        charging_derating_factor=1.0,
        horizon_start_s=0.0,
        horizon_end_s=100000.0,
        use_energy_constraints=True,
    )


def test_energy_aware_heuristic_visits_charger_when_route_binds():
    # depot=0, customers=1/2, charger=3.  The charger is on the return corridor.
    distance = np.array(
        [[0, 4, 5, 3], [4, 0, 4, 2], [5, 4, 0, 2], [3, 2, 2, 0]],
        dtype=float,
    )
    result = _case(distance, battery=8.0)
    assert result.passed
    assert result.charging_station_visit_count > 0
    assert result.minimum_soc_kwh >= -1e-6


def test_energy_aware_heuristic_has_no_activity_when_battery_is_nonbinding():
    distance = np.array(
        [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],
        dtype=float,
    )
    result = _case(distance, battery=100.0)
    assert result.passed
    assert result.charging_station_visit_count == 0
    assert result.vehicle_count == 1
