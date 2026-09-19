from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from dataclasses import replace

import pytest
from gurobipy import GRB
from evrptw_core.objective import load_objective, select_objective_distance
from gurobi_solver import GurobiEVRPTWSolver, GurobiSolverConfig
from route_validator import validate_routes
from run_single_cus500 import (
    ProgressRecorder,
    LoggedSolver,
    relative_gap,
    finite,
    choose_task,
)
from stage2_adapter import Stage2ViewTask
from test_stage2_gurobi import _charging_fixture


def cost_config(**overrides):
    obj = load_objective(
        "EVRPTW_Benchmark/Reinforcement_Learning/configs/rivian_energy_vehicle_cost_v2.json"
    )
    cfg = GurobiSolverConfig(
        time_limit_s=None,
        mip_gap=None,
        threads=None,
        output_flag=0,
        checkpoints_s=(),
        cs_copies=1,
        objective_mode=obj.mode,
        objective_profile_id=obj.profile_id,
        electricity_price_usd_per_kwh=obj.electricity_price_usd_per_kwh,
        consumption_kwh_per_km=obj.consumption_kwh_per_km,
        vehicle_fixed_cost_usd=obj.vehicle_fixed_cost_usd,
    )
    return obj, replace(cfg, **overrides)


def test_true_gurobi_default_parameters_dtime_and_durable_logs(tmp_path):
    obj, cfg = cost_config()
    rec = ProgressRecorder(tmp_path, 0.001)
    manifest = {}
    solver = LoggedSolver(cfg, tmp_path, rec, manifest)
    try:
        instance = _charging_fixture()
        original = instance.distance_matrix_km.copy()
        solution = solver.solve(instance)
        final = rec.final(solver.model, solution.metadata["gurobi_status_name"])
        assert solver.model.Params.Threads == solver.model.getParamInfo("Threads")[5] == 0
        assert solver.model.Params.MIPGap == solver.model.getParamInfo("MIPGap")[5] == 1e-4
        assert math.isinf(solver.model.Params.TimeLimit)
        assert solution.feasible and solution.objective_distance_km == 6.0
        assert (instance.distance_matrix_km == original).all()
        assert solution.metadata["distance_matrix_source"] == "running_time_path_distance_km"
        assert solution.metadata["objective_cost_usd"] == pytest.approx(obj.value(6.0, 1))
        assert final["objective_value_usd"] == pytest.approx(solver.model.ObjVal)
        assert final["mip_gap"] == solver.model.MIPGap
        assert final["gurobi_runtime_s"] == solver.model.Runtime
        assert final["wall_runtime_s"] >= final["gurobi_runtime_s"]
        assert rec.error is None and (tmp_path / "gurobi.log").is_file()
        assert json.loads((tmp_path / "latest_progress.json").read_text()) == final
        with (tmp_path / "progress.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert (
            rows[0]["event"] == "model_build_start"
            and rows[0]["objective_value_usd"] == ""
            and rows[0]["mip_gap"] == ""
        )
        assert (
            rows[-1]["event"] == "final"
            and float(rows[-1]["objective_value_usd"]) == final["objective_value_usd"]
        )
    finally:
        rec.close()
        if solver.model is not None:
            solver.model.dispose()


def test_optimal_tolerance_does_not_fabricate_zero_gap(monkeypatch):
    obj, cfg = cost_config(checkpoints_s=(30.0,))
    solver = GurobiEVRPTWSolver(cfg)
    original = solver._safe_model_float

    def measured(model, attr):
        if attr == "MIPGap":
            return 5e-5
        if attr == "ObjBound":
            return float(model.ObjVal) * (1.0 - 5e-5)
        return original(model, attr)

    monkeypatch.setattr(solver, "_safe_model_float", measured)
    try:
        solution = solver.solve(_charging_fixture())
        assert solution.metadata["gurobi_status_name"] == "OPTIMAL"
        assert solution.metadata["mip_gap"] == 5e-5
        assert solution.metadata["best_bound"] < solution.metadata["objective_cost_usd"]
        assert solution.metadata["checkpoint_snapshots"][0]["mip_gap"] == pytest.approx(5e-5)
    finally:
        if solver.model is not None:
            solver.model.dispose()


def test_gap_handles_no_incumbent_zero_and_infinity():
    assert finite(GRB.INFINITY) is None
    assert finite(-GRB.INFINITY) is None
    assert relative_gap(None, 0) == (None, "no_incumbent")
    assert relative_gap(10, None) == (None, "bound_unavailable")
    assert relative_gap(0, 0) == (0, "finite")
    assert relative_gap(0, 2) == (None, "infinite")
    assert relative_gap(100, 90) == (0.1, "finite")


def test_mipsol_candidate_is_not_misreported_as_incumbent(tmp_path):
    rec = ProgressRecorder(tmp_path)

    class MockModel:
        def cbGet(self, code):
            return {
                GRB.Callback.RUNTIME: 1.0,
                GRB.Callback.MIPSOL_OBJBST: 100.0,
                GRB.Callback.MIPSOL_OBJ: 110.0,
                GRB.Callback.MIPSOL_OBJBND: 90.0,
                GRB.Callback.MIPSOL_NODCNT: 1.0,
                GRB.Callback.MIPSOL_SOLCNT: 2.0,
            }[code]

        def terminate(self):
            raise AssertionError("unexpected terminate")

    try:
        rec.callback(MockModel(), GRB.Callback.MIPSOL)
        assert rec.error is None
        row = json.loads((tmp_path / "latest_progress.json").read_text())
        assert (
            row["objective_value_usd"] == 100
            and row["candidate_objective_usd"] == 110
            and row["mip_gap"] == 0.1
        )
    finally:
        rec.close()


def test_replay_rejects_internal_depot_and_customer_free_trip():
    instance = _charging_fixture()
    assert any(
        "internal depot" in s
        for s in validate_routes(instance, [[0, 2, 0, 1, 0]])["violations"]
    )
    assert any(
        "serves no customer" in s
        for s in validate_routes(instance, [[0, 2, 1, 0], [0, 2, 0]])["violations"]
    )


def test_seeded_selection_is_one_test_cus500_independent_of_input_order():
    def task(n, scale="Cus500", split="test"):
        return Stage2ViewTask(
            "/index", "/family", f"id{n}", f"f{n}", "cohort", split,
            "T1", "city", scale, int(scale[3:]), 50, n,
        )

    tasks = [task(0, "Cus100"), task(1), task(2), task(3, "Cus500", "train")]
    a, c = choose_task(tasks, 1234)
    b, _ = choose_task(tasks[::-1], 1234)
    assert a == b and len(c) == 2 and a.scale_label == "Cus500" and a.split_id == "test"
    with pytest.raises(ValueError, match="Duplicate"):
        choose_task([task(1), task(1)], 0)


def test_every_gap_change_is_flushed_even_within_one_heartbeat(tmp_path, monkeypatch):
    # Freeze the wall clock: none of these writes can be caused by the timer.
    monkeypatch.setattr("run_single_cus500.time.perf_counter", lambda: 100.0)
    rec = ProgressRecorder(tmp_path, interval_s=60.0)

    class MockModel:
        def __init__(self, runtime, objective, bound):
            self.runtime = runtime
            self.objective = objective
            self.bound = bound

        def cbGet(self, code):
            return {
                GRB.Callback.RUNTIME: self.runtime,
                GRB.Callback.MIP_OBJBST: self.objective,
                GRB.Callback.MIP_OBJBND: self.bound,
                GRB.Callback.MIP_NODCNT: 1.0,
                GRB.Callback.MIP_SOLCNT: int(self.objective < GRB.INFINITY),
            }[code]

        def terminate(self):
            raise AssertionError("unexpected terminate")

    try:
        for runtime, objective, bound in [
            (0.01, GRB.INFINITY, 50.0),
            (0.02, 100.0, 50.0),  # First incumbent: gap becomes finite.
            (0.03, 100.0, 70.0),  # Bound changes without MIPSOL.
            (0.04, 80.0, 70.0),   # Incumbent changes without timer elapsing.
            (0.05, 80.0, 70.0),   # Identical state: no extra heartbeat yet.
            (0.06, 40.0, 35.0),   # Same gap, different objective/bound: retain.
        ]:
            rec.callback(MockModel(runtime, objective, bound), GRB.Callback.MIP)
        assert rec.error is None
        # Read before closing to establish that each row was flushed.
        with (tmp_path / "progress.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert [float(row["gurobi_runtime_s"]) for row in rows] == [
            0.01, 0.02, 0.03, 0.04, 0.06
        ]
        assert rows[0]["mip_gap"] == rows[0]["objective_value_usd"] == ""
        assert [float(row["mip_gap"]) for row in rows[1:]] == [
            0.5, 0.3, 0.125, 0.125
        ]
        assert [float(row["objective_value_usd"]) for row in rows[1:]] == [
            100.0, 100.0, 80.0, 40.0
        ]
        assert [float(row["mip_gap_percent"]) for row in rows[1:]] == [
            50.0, 30.0, 12.5, 12.5
        ]
    finally:
        rec.close()
