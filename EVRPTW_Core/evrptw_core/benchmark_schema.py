"""Shared output contracts for the EVRPTW benchmark runners."""

from __future__ import annotations


UNIFIED_TIME_TRACE_SCHEMA = "evrptw_benchmark_time_trace_v2"

# The public benchmark launchers record best-so-far solutions after 5 and 30
# minutes. A solver may support additional checkpoints when invoked
# directly, but every emitted trace uses the same column contract below.
FROZEN_BENCHMARK_CHECKPOINTS_S = (300.0, 1800.0)

UNIFIED_TIME_TRACE_FIELDNAMES = [
    "instance_id",
    "file",
    "family_id",
    "solver_name",
    "algorithm_profile_id",
    "seed",
    "seed_scheme",
    "run_contract_fingerprint",
    "checkpoint_s",
    "elapsed_s",
    "reached_checkpoint",
    "status",
    "benchmark_status",
    "has_incumbent",
    "first_feasible_time_s",
    "incumbent_event_time_s",
    "objective_distance_km",
    "objective_mode",
    "objective_profile_id",
    "objective_unit",
    "objective_value",
    "objective_cost_usd",
    "electricity_cost_usd",
    "vehicle_cost_usd",
    "vehicles_started",
    "best_bound",
    "mip_gap",
    "vehicle_count",
    "routes_json",
    "route_sequence_json",
    "checkpoint_solution_path",
    "source",
    "errors",
    "route_validation_passed",
    "route_validation_json",
    "diagnostic_objective_distance_km",
    "diagnostic_routes_json",
]
