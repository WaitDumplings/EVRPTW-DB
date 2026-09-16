from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gurobipy import GRB, GurobiError

from evrptw_core.objective import load_objective
from evrptw_core.schema import EVRPTWInstance
from gurobi_solver import GurobiEVRPTWSolver, GurobiSolverConfig, NodeMap
import run_gurobi
from run_contract import build_run_contract, ensure_run_contract


PROFILE = run_gurobi.DEFAULT_OBJECTIVE_CONFIG


def _cost_config(**overrides):
    objective = load_objective(PROFILE)
    fields = dict(
        time_limit_s=10., checkpoints_s=(10.,), cs_copies=1, threads=1,
        objective_mode=objective.mode, objective_profile_id=objective.profile_id,
        electricity_price_usd_per_kwh=objective.electricity_price_usd_per_kwh,
        consumption_kwh_per_km=objective.consumption_kwh_per_km,
        vehicle_fixed_cost_usd=objective.vehicle_fixed_cost_usd,
    )
    return GurobiSolverConfig(**(fields | overrides))


def _instance(distance=None, capacity=10.):
    if distance is None:
        # Road-distance matrices evaluated along the fastest paths need not
        # obey the distance triangle inequality. This fixture distinguishes
        # distance minimization (two cars, 4 km) from cost (one car, 7 km).
        distance = np.array([[0., 1., 1.], [1., 0., 5.], [1., 5., 0.]])
    n = len(distance) - 1
    return EVRPTWInstance.from_dict({
        'instance_id': 'objective_regression', 'region_id': 'test',
        'mother_board_id': 'test', 'operating_day_id': 'test',
        'day_type': 'weekday', 'working_start_s': 0, 'working_end_s': 1e6,
        'depot': np.zeros(2), 'customers': np.zeros((n, 2)),
        'charging_stations': np.zeros((0, 2)),
        'distance_matrix_km': distance, 'demands_cm3': np.ones(n),
        'package_counts': np.ones(n), 'service_time_s': np.ones(n),
        'tw_s': np.tile([0., 1e6], (n, 1)), 'cs_time_to_depot_s': np.array([]),
        'vehicle': {'battery_capacity_kwh': 1e6, 'cargo_capacity_cm3': capacity,
                    'consumption_kwh_per_km': .404},
        'running_time_shortest_matrix_s': distance,
        'running_time_path_energy_kwh': distance * .1,
    })


def _solve(solver, instance):
    try:
        return solver.solve(instance)
    except GurobiError as exc:
        if 'license' in str(exc).lower() or 'hostid' in str(exc).lower():
            pytest.skip(f'Gurobi license unavailable: {exc}')
        raise


def test_real_cost_model_prefers_longer_route_with_one_vehicle():
    instance = _instance()
    cost_solver = GurobiEVRPTWSolver(_cost_config(tie_break_vehicle_count=True))
    cost = _solve(cost_solver, instance)
    distance = _solve(GurobiEVRPTWSolver(GurobiSolverConfig(
        time_limit_s=10., checkpoints_s=(10.,), threads=1,
    )), instance)
    assert cost.feasible and distance.feasible
    assert (cost.objective_distance_km, cost.vehicle_count) == (7., 1)
    assert (distance.objective_distance_km, distance.vehicle_count) == (4., 2)
    assert cost.metadata['objective_value'] == pytest.approx(
        load_objective(PROFILE).value(7., 1)
    )
    assert cost_solver.model.ObjVal == pytest.approx(cost.metadata['objective_value'])
    assert cost.metadata['tie_break_applied'] is False
    assert cost.metadata['is_certified_optimal'] is True
    assert all(row['objective_value'] == pytest.approx(cost.metadata['objective_value'])
               for row in cost.metadata['checkpoint_snapshots'])


def test_real_mipsol_pool_never_replaces_best_with_worse_or_equal_solution():
    solver = GurobiEVRPTWSolver(_cost_config(checkpoints_s=()))
    instance = _instance()
    try:
        model, node_map, x, *_ = solver._build_model(instance)
    except GurobiError as exc:
        if 'license' in str(exc).lower() or 'hostid' in str(exc).lower():
            pytest.skip(f'Gurobi license unavailable: {exc}')
        raise
    # Deliberately request additional solutions to exercise the documented
    # non-improving MIPSOL case with actual Gurobi callbacks.
    model.Params.PoolSearchMode = 2
    model.Params.PoolSolutions = 4
    trace = solver._new_trace()
    record = solver._make_callback(trace, node_map, x, instance)
    reported = []
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            reported.append(model.cbGet(GRB.Callback.MIPSOL_OBJ))
        record(model, where)
    model.optimize(callback)
    assert len(reported) >= 2
    assert any(value >= min(reported[:i]) for i, value in enumerate(reported) if i)
    assert trace['last_incumbent']['objective_value'] == pytest.approx(min(reported))
    events = [row['objective_value'] for row in trace['incumbent_events']]
    assert all(new < old for old, new in zip(events, events[1:]))
    model.dispose()


def test_mipsol_worse_and_equal_events_preserve_checkpoint_causality():
    solver = GurobiEVRPTWSolver(GurobiSolverConfig(checkpoints_s=(15., 20., 30.)))
    instance = _instance()
    node_map = NodeMap([0, 1, 2, 0], [1, 2], [], 0, 3)
    x = {(0, 1): object(), (1, 3): object(), (0, 2): object(),
         (2, 3): object(), (1, 2): object()}
    trace = solver._new_trace()
    callback = solver._make_callback(trace, node_map, x, instance)
    class CallbackModel:
        runtime = 10.
        objective = 4.
        def cbGet(self, what):
            return {GRB.Callback.RUNTIME: self.runtime,
                    GRB.Callback.MIPSOL_OBJ: self.objective,
                    GRB.Callback.MIPSOL_OBJBND: 1.}[what]
        def cbGetSolution(self, variables):
            return [1., 1., 1., 1., 0.] if self.objective == 4. else [1., 0., 0., 1., 1.]
    model = CallbackModel()
    callback(model, GRB.Callback.MIPSOL)
    model.runtime, model.objective = 20., 7.
    callback(model, GRB.Callback.MIPSOL)
    model.runtime, model.objective = 35., 4.
    callback(model, GRB.Callback.MIPSOL)
    assert trace['last_incumbent']['elapsed_s'] == 10.
    assert trace['first_feasible_time_s'] == 10.
    assert len(trace['incumbent_events']) == 1
    assert [row['objective_value'] for row in trace['checkpoint_snapshots']] == [4., 4., 4.]
    assert all(row['routes'] == [[0, 1, 0], [0, 2, 0]]
               for row in trace['checkpoint_snapshots'])


def test_real_positive_gap_is_not_published_as_zero_gap_or_certified_optimal(tmp_path):
    class LooseGapSolver(GurobiEVRPTWSolver):
        def _build_model(self, instance):
            result = super()._build_model(instance)
            result[0].Params.Presolve = 0
            result[0].Params.Heuristics = 1
            result[0].Params.Cuts = 0
            return result
    coordinates = np.random.default_rng(42).uniform(0, 10, size=(9, 2))
    coordinates[0] = 0
    distance = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    solver = LooseGapSolver(_cost_config(mip_gap=.99))
    solution = _solve(solver, _instance(distance, capacity=4.))
    assert solution.feasible
    assert solution.metadata['gurobi_status_name'] == 'OPTIMAL'
    assert solver.model.MIPGap > 0.01
    assert solution.metadata['mip_gap'] == pytest.approx(solver.model.MIPGap)
    assert solution.metadata['best_bound'] == pytest.approx(solver.model.ObjBound)
    assert solution.metadata['is_certified_optimal'] is False
    assert solution.metadata['benchmark_status'] == 'COMPLETED_WITH_INCUMBENT'
    checkpoint = solution.metadata['checkpoint_snapshots'][-1]
    assert checkpoint['best_bound'] == pytest.approx(solver.model.ObjBound)
    assert checkpoint['mip_gap'] == pytest.approx(solver.model.MIPGap)
    info = dict(instance_key='x', split='test', scale='Cus8', instance_id='x', region_id='x')
    reference = run_gurobi.write_reference_route(tmp_path, info, solution, 'test', 10.)
    assert json.loads(reference.read_text())['is_certified_optimal'] is False


def _contract(tmp_path, *, config=None, version='test'):
    index = tmp_path / 'view_index.parquet'
    if not index.exists():
        index.write_bytes(b'index content')
    family = tmp_path / 'family'
    family.mkdir(exist_ok=True)
    manifest = family / 'family_manifest.json'
    if not manifest.exists():
        manifest.write_text('{"matrix_identity":"test"}')
    return build_run_contract(
        solver_config=config or _cost_config(),
        tasks=[SimpleNamespace(index_path=str(index), family_dir=str(family))],
        solver_version=version, algorithm_profile_id=run_gurobi.GUROBI_ALGORITHM_PROFILE_ID,
    )


@pytest.mark.parametrize('change', [
    {'vehicle_fixed_cost_usd': 1.}, {'time_limit_s': 20.}, {'checkpoints_s': (5., 10.)},
    {'cs_copies': 2}, {'tie_break_vehicle_count': True}, {'mip_gap': .1}, {'threads': 2},
    {'objective_mode': 'distance', 'objective_profile_id': 'distance_v1'},
])
def test_resume_contract_rejects_changed_objective_or_budget(tmp_path, change):
    output = tmp_path / 'output'
    contract = _contract(tmp_path)
    ensure_run_contract(output, contract)
    ensure_run_contract(output, contract)
    changed = _contract(tmp_path, config=replace(_cost_config(), **change))
    with pytest.raises(ValueError, match='choose a new --save_path'):
        ensure_run_contract(output, changed)


def test_resume_contract_rejects_changed_data_or_gurobi_version(tmp_path):
    output = tmp_path / 'output'
    original = _contract(tmp_path)
    ensure_run_contract(output, original)
    with pytest.raises(ValueError, match='choose a new --save_path'):
        ensure_run_contract(output, _contract(tmp_path, version='new'))
    (tmp_path / 'view_index.parquet').write_bytes(b'different instances')
    with pytest.raises(ValueError, match='choose a new --save_path'):
        ensure_run_contract(output, _contract(tmp_path))


def test_resume_refuses_legacy_results_without_contract(tmp_path):
    output = tmp_path / 'output'
    output.mkdir()
    summary = output / 'gurobi_summary.csv'
    previous = 'instance_id,status_name,objective_mode\nx,OPTIMAL,distance\n'
    summary.write_text(previous)
    with pytest.raises(ValueError, match='no run contract'):
        ensure_run_contract(output, _contract(tmp_path))
    assert summary.read_text() == previous
    assert not (output / 'run_contract.json').exists()


def test_cli_defaults_to_cost_and_keeps_explicit_distance(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_gurobi, 'read_stage2_tasks', lambda *args, **kwargs: [])
    with pytest.raises(ValueError, match='No Stage-2'):
        run_gurobi.main(['--dataset_path', str(tmp_path), '--save_path', str(tmp_path / 'cost')])
    assert 'objective=rivian_energy_vehicle_cost_v2' in capsys.readouterr().out
    distance_path = tmp_path / 'distance.json'
    distance_path.write_text('{"mode":"distance", "profile_id":"distance_v1"}')
    with pytest.raises(ValueError, match='No Stage-2'):
        run_gurobi.main(['--dataset_path', str(tmp_path), '--save_path', str(tmp_path / 'distance'),
                         '--objective_config', str(distance_path)])
    assert 'objective=distance_v1' in capsys.readouterr().out


def test_checkpoint_replay_rejects_model_cost_with_wrong_vehicle_count():
    solver = GurobiEVRPTWSolver(_cost_config())
    instance = _instance()
    trace = solver._new_trace()
    # Distance is correct, but the claimed cost incorrectly pays for one car.
    snapshot = solver._make_snapshot(
        checkpoint_s=10., elapsed_s=10., reached_checkpoint=True,
        solver_status='RUNNING', objective_distance_km=4., best_bound=0.,
        objective_value=float(solver.objective_config.value(4., 1)),
        routes=[[0, 1, 0], [0, 2, 0]], source='test',
    )
    trace['checkpoint_snapshots'] = [snapshot]
    solver._validate_checkpoint_snapshots(instance, trace)
    assert snapshot['route_validation_passed'] is False
    assert snapshot['has_incumbent'] is False
    assert snapshot['objective_value'] is None
    assert 'cost/vehicle count' in snapshot['route_validation']['violations'][-1]


def test_model_construction_time_is_deducted_and_never_backfills_checkpoint():
    import time
    class SlowBuildSolver(GurobiEVRPTWSolver):
        def _build_model(self, instance):
            result = super()._build_model(instance)
            time.sleep(.05)
            return result
    solver = SlowBuildSolver(_cost_config(time_limit_s=2., checkpoints_s=(.01, 2.)))
    solution = _solve(solver, _instance())
    assert solution.feasible
    metadata = solution.metadata
    assert metadata['model_build_runtime_s'] >= .05
    assert metadata['stage1_optimization_started'] is True
    assert solver.model.Params.TimeLimit <= 2. - metadata['model_build_runtime_s']
    assert metadata['first_feasible_time_s'] >= metadata['model_build_runtime_s']
    assert metadata['checkpoint_snapshots'][0]['has_incumbent'] is False
    assert metadata['checkpoint_snapshots'][1]['has_incumbent'] is True
    assert metadata['stage1_elapsed_s'] >= metadata['model_build_runtime_s']


def test_exhausted_construction_budget_does_not_start_optimization():
    import time
    class ExhaustedBuildSolver(GurobiEVRPTWSolver):
        def _build_model(self, instance):
            result = super()._build_model(instance)
            time.sleep(.03)
            return result
    solver = ExhaustedBuildSolver(_cost_config(time_limit_s=.01, checkpoints_s=(.01,)))
    solution = _solve(solver, _instance())
    assert not solution.feasible
    assert solver.model.Status == GRB.LOADED
    assert solution.metadata['gurobi_status_name'] == 'TIME_LIMIT'
    assert solution.metadata['benchmark_status'] == 'UNFINISHED_NO_INCUMBENT'
    assert solution.metadata['stage1_optimization_started'] is False
    assert solution.metadata['stage1_optimization_runtime_s'] == 0.
    assert solution.metadata['checkpoint_snapshots'][0]['has_incumbent'] is False
