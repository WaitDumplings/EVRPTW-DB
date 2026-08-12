from __future__ import annotations

from pathlib import Path

import pytest

import gurobi_solver as gurobi_module
from evrptw_core.schema import EVRPTWSolution
from gurobi_solver import (
    STANDARD_BENCHMARK_CHECKPOINTS_S,
    GurobiEVRPTWSolver,
    GurobiSolverConfig,
)
from run_gurobi import (
    append_time_rows,
    is_terminal_summary_row,
    resolve_time_schedule,
)


def _snapshot(
    solver: GurobiEVRPTWSolver,
    *,
    elapsed_s: float,
    objective: float | None,
    routes: list[list[int]],
    status: str = "RUNNING",
) -> dict[str, object]:
    return solver._make_snapshot(
        checkpoint_s=None,
        elapsed_s=elapsed_s,
        reached_checkpoint=True,
        solver_status=status,
        objective_distance_km=objective,
        best_bound=objective,
        routes=routes,
        source="test",
    )


def test_default_schedule_is_the_published_five_checkpoints() -> None:
    checkpoints, time_limit_s = resolve_time_schedule(tuple(), None)

    assert checkpoints == STANDARD_BENCHMARK_CHECKPOINTS_S
    assert time_limit_s == 7200.0


def test_short_explicit_smoke_schedule_does_not_expand_to_two_hours() -> None:
    checkpoints, time_limit_s = resolve_time_schedule(tuple(), 30.0)

    assert checkpoints == (30.0,)
    assert time_limit_s == 30.0


def test_cs_copies_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        GurobiSolverConfig(cs_copies=0)


def test_late_first_incumbent_is_not_backfilled_into_earlier_checkpoint() -> None:
    solver = GurobiEVRPTWSolver(
        GurobiSolverConfig(checkpoints_s=(60.0, 300.0))
    )
    trace = solver._new_trace()

    # This is the pre-event half of a callback at t=100: checkpoint 60 must
    # be frozen before the newly reported incumbent is installed.
    solver._record_due_checkpoints(
        trace,
        100.0,
        "RUNNING",
        include_equal=False,
    )
    trace["last_incumbent"] = _snapshot(
        solver,
        elapsed_s=100.0,
        objective=12.0,
        routes=[[0, 1, 0]],
    )
    final = _snapshot(
        solver,
        elapsed_s=100.0,
        objective=12.0,
        routes=[[0, 1, 0]],
        status="OPTIMAL",
    )
    solver._finalize_checkpoints(trace, final, 100.0, "OPTIMAL")

    at_60, at_300 = trace["checkpoint_snapshots"]
    assert at_60["has_incumbent"] is False
    assert at_60["objective_distance_km"] is None
    assert at_300["has_incumbent"] is True
    assert at_300["objective_distance_km"] == 12.0
    assert at_300["routes"] == [[0, 1, 0]]
    assert at_300["reached_checkpoint"] is False
    assert at_300["source"] == "final_after_early_stop"


def test_early_optimum_forward_fills_all_later_checkpoints() -> None:
    solver = GurobiEVRPTWSolver(
        GurobiSolverConfig(checkpoints_s=STANDARD_BENCHMARK_CHECKPOINTS_S)
    )
    trace = solver._new_trace()
    final = _snapshot(
        solver,
        elapsed_s=20.0,
        objective=9.0,
        routes=[[0, 1, 0]],
        status="OPTIMAL",
    )

    solver._finalize_checkpoints(trace, final, 20.0, "OPTIMAL")
    solver._annotate_checkpoint_statuses(
        trace,
        final_benchmark_status="COMPLETED_OPTIMAL",
        final_has_incumbent=True,
        terminal_budget_s=7200.0,
    )

    snapshots = trace["checkpoint_snapshots"]
    assert [row["checkpoint_s"] for row in snapshots] == list(
        STANDARD_BENCHMARK_CHECKPOINTS_S
    )
    assert all(row["objective_distance_km"] == 9.0 for row in snapshots)
    assert all(row["routes"] == [[0, 1, 0]] for row in snapshots)
    assert all(row["reached_checkpoint"] is False for row in snapshots)
    assert all(row["source"] == "final_after_early_stop" for row in snapshots)


def test_two_hour_run_without_incumbent_is_explicitly_unfinished() -> None:
    solver = GurobiEVRPTWSolver(
        GurobiSolverConfig(checkpoints_s=STANDARD_BENCHMARK_CHECKPOINTS_S)
    )
    trace = solver._new_trace()
    final = _snapshot(
        solver,
        elapsed_s=7200.0,
        objective=None,
        routes=[],
        status="TIME_LIMIT",
    )

    solver._finalize_checkpoints(trace, final, 7200.0, "TIME_LIMIT")
    solver._annotate_checkpoint_statuses(
        trace,
        final_benchmark_status="UNFINISHED_NO_INCUMBENT",
        final_has_incumbent=False,
        terminal_budget_s=7200.0,
    )

    snapshots = trace["checkpoint_snapshots"]
    assert len(snapshots) == 5
    assert all(row["has_incumbent"] is False for row in snapshots)
    assert all(row["routes"] == [] for row in snapshots)
    assert snapshots[-1]["checkpoint_s"] == 7200.0
    assert snapshots[-1]["benchmark_status"] == "UNFINISHED_NO_INCUMBENT"


def test_custom_early_checkpoint_is_not_mislabeled_as_terminal() -> None:
    solver = GurobiEVRPTWSolver(
        GurobiSolverConfig(checkpoints_s=(60.0,), time_limit_s=7200.0)
    )
    trace = solver._new_trace()
    final = _snapshot(
        solver,
        elapsed_s=7200.0,
        objective=None,
        routes=[],
        status="TIME_LIMIT",
    )

    solver._finalize_checkpoints(trace, final, 7200.0, "TIME_LIMIT")
    solver._validate_checkpoint_snapshots(object(), trace)
    solver._annotate_checkpoint_statuses(
        trace,
        final_benchmark_status="UNFINISHED_NO_INCUMBENT",
        final_has_incumbent=False,
        terminal_budget_s=7200.0,
    )

    assert trace["checkpoint_snapshots"][0]["benchmark_status"] == (
        "NO_INCUMBENT_YET"
    )


def test_invalid_checkpoint_route_is_diagnostic_only(monkeypatch: pytest.MonkeyPatch) -> None:
    solver = GurobiEVRPTWSolver(
        GurobiSolverConfig(checkpoints_s=(60.0,))
    )
    trace = solver._new_trace()
    trace["checkpoint_snapshots"] = [
        solver._make_snapshot(
            checkpoint_s=60.0,
            elapsed_s=60.0,
            reached_checkpoint=True,
            solver_status="RUNNING",
            objective_distance_km=12.0,
            best_bound=10.0,
            routes=[[0, 1, 0]],
            source="checkpoint_incumbent",
        )
    ]
    monkeypatch.setattr(
        gurobi_module,
        "validate_routes",
        lambda _instance, _routes: {
            "passed": False,
            "violations": ["battery exceeded"],
            "objective_distance_km": 12.0,
        },
    )

    solver._validate_checkpoint_snapshots(object(), trace)
    solver._annotate_checkpoint_statuses(
        trace,
        final_benchmark_status="INVALID_INCUMBENT",
        final_has_incumbent=True,
        terminal_budget_s=7200.0,
    )

    snapshot = trace["checkpoint_snapshots"][0]
    assert snapshot["route_validation_passed"] is False
    assert snapshot["benchmark_status"] == "INVALID_INCUMBENT"
    assert snapshot["has_incumbent"] is False
    assert snapshot["objective_distance_km"] is None
    assert snapshot["routes"] == []
    assert snapshot["diagnostic_objective_distance_km"] == 12.0
    assert snapshot["diagnostic_routes"] == [[0, 1, 0]]


def test_invalid_checkpoint_does_not_write_feasible_solution(
    tmp_path: Path,
) -> None:
    snapshot = {
        "checkpoint_s": 60.0,
        "elapsed_s": 60.0,
        "reached_checkpoint": True,
        "solver_status": "RUNNING",
        "benchmark_status": "INVALID_INCUMBENT",
        "has_incumbent": False,
        "objective_distance_km": None,
        "best_bound": 10.0,
        "mip_gap": None,
        "vehicle_count": None,
        "routes": [],
        "route_sequence": [],
        "source": "checkpoint_incumbent_invalid",
        "route_validation_passed": False,
        "route_validation": {
            "passed": False,
            "violations": ["battery exceeded"],
        },
        "diagnostic_objective_distance_km": 12.0,
        "diagnostic_routes": [[0, 1, 0]],
    }
    solution = EVRPTWSolution(
        instance_id="test",
        solver_name="test",
        routes=[],
        objective_distance_km=None,
        vehicle_count=None,
        runtime_s=60.0,
        feasible=False,
        metadata={"checkpoint_snapshots": [snapshot]},
    )
    rows: list[dict[str, object]] = []

    append_time_rows(
        rows,
        tmp_path / "view_index.parquet",
        "test",
        solution,
        tmp_path / "checkpoints",
    )

    assert rows[0]["checkpoint_solution_path"] == ""
    assert rows[0]["has_incumbent"] is False
    assert rows[0]["routes_json"] == "[]"
    assert rows[0]["diagnostic_routes_json"] == "[[0, 1, 0]]"
    assert not (tmp_path / "checkpoints").exists()


def test_skip_completed_uses_explicit_benchmark_terminal_states() -> None:
    assert is_terminal_summary_row(
        {"benchmark_status": "COMPLETED_WITH_INCUMBENT", "status_name": "TIME_LIMIT"}
    )
    assert is_terminal_summary_row(
        {"benchmark_status": "UNFINISHED_NO_INCUMBENT", "status_name": "TIME_LIMIT"}
    )
    assert not is_terminal_summary_row(
        {"benchmark_status": "UNFINISHED_NO_INCUMBENT", "status_name": "INTERRUPTED"}
    )
    assert not is_terminal_summary_row(
        {"benchmark_status": "INVALID_INCUMBENT", "status_name": "OPTIMAL"}
    )
