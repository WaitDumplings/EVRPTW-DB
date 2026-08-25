from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gurobipy import GRB, Model, quicksum

from evrptw_core.schema import EVRPTWInstance, EVRPTWSolution, merge_route_sequences
from route_validator import resolve_charging_profile, validate_routes


STANDARD_BENCHMARK_CHECKPOINTS_S = (60.0, 300.0, 900.0, 3600.0, 7200.0)


@dataclass(frozen=True)
class GurobiSolverConfig:
    time_limit_s: float = 7200.0
    mip_gap: float = 0.0
    cs_copies: int = 2
    output_flag: int = 0
    checkpoints_s: tuple[float, ...] = field(
        default_factory=lambda: STANDARD_BENCHMARK_CHECKPOINTS_S
    )
    record_incumbent_events: bool = True
    tie_break_vehicle_count: bool = False
    distance_tolerance_abs: float = 1e-6
    distance_tolerance_rel: float = 1e-8
    threads: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.cs_copies, bool) or int(self.cs_copies) != self.cs_copies:
            raise ValueError("cs_copies must be an integer")
        if int(self.cs_copies) < 1:
            raise ValueError("cs_copies must be at least 1")


@dataclass(frozen=True)
class NodeMap:
    solver_to_terminal: list[int]
    customer_nodes: list[int]
    cs_nodes: list[int]
    start_depot: int
    end_depot: int


class GurobiEVRPTWSolver:
    """Small-scale exact EVRP-TW-D solver for canonical and Stage-2 views.

    The model is an arc-flow MILP with duplicated charging-station nodes. Depot
    and charging stations reset the battery to full before departure.  A CS
    copy inherits its physical station's effective power, and full-charge time
    is linear in the energy missing on arrival.

    Benchmark logging is callback-based: every checkpoint stores the incumbent
    route sequence available at or before that runtime, plus the current best
    bound/gap when Gurobi exposes them.
    """

    name = "gurobi_exact_arcflow"

    def __init__(self, config: GurobiSolverConfig | None = None):
        self.config = config or GurobiSolverConfig()
        self.model: Model | None = None
        self.node_map: NodeMap | None = None
        self.x: dict[tuple[int, int], Any] = {}

    def solve(self, instance: EVRPTWInstance) -> EVRPTWSolution:
        start = time.perf_counter()
        model, node_map, x, distance_expr, vehicle_expr, metric_metadata = self._build_model(instance)
        self.model = model
        self.node_map = node_map
        self.x = x

        trace = self._new_trace()
        callback = self._make_callback(trace, node_map, x)
        model.optimize(callback)

        stage1_status = int(model.Status)
        stage1_status_name = self._status_name(stage1_status)
        stage1_runtime_s = self._safe_model_float(model, "Runtime")
        if stage1_runtime_s is None:
            stage1_runtime_s = time.perf_counter() - start
        stage1_has_solution = model.SolCount > 0
        stage1_best_distance = self._safe_model_float(model, "ObjVal") if stage1_has_solution else None
        stage1_best_bound = self._safe_model_float(model, "ObjBound")
        stage1_mip_gap = self._safe_model_float(model, "MIPGap") if stage1_has_solution else None

        # Freeze the distance-objective trace before an optional secondary
        # vehicle-count optimization changes the model objective and solution.
        stage1_routes = self._extract_routes(node_map, x) if stage1_has_solution else []
        stage1_route_distance = (
            self._route_distance_km(stage1_routes, instance)
            if stage1_has_solution
            else None
        )
        if stage1_has_solution and trace["first_feasible_time_s"] is None:
            trace["first_feasible_time_s"] = stage1_runtime_s
        stage1_final_snapshot = self._make_snapshot(
            checkpoint_s=None,
            elapsed_s=stage1_runtime_s,
            reached_checkpoint=True,
            solver_status=stage1_status_name,
            objective_distance_km=stage1_route_distance,
            best_bound=(
                stage1_route_distance
                if stage1_status == GRB.OPTIMAL
                and stage1_route_distance is not None
                else stage1_best_bound
            ),
            routes=stage1_routes,
            source="primary_final",
        )
        if stage1_status == GRB.OPTIMAL and stage1_route_distance is not None:
            stage1_final_snapshot["mip_gap"] = 0.0
        elif stage1_mip_gap is not None:
            stage1_final_snapshot["mip_gap"] = stage1_mip_gap
        self._finalize_checkpoints(
            trace,
            stage1_final_snapshot,
            stage1_runtime_s,
            stage1_status_name,
        )

        tie_break_applied = False
        tie_break_skipped_no_time = False
        distance_tolerance = None
        tie_break_status = None
        tie_break_status_name = None

        if (
            self.config.tie_break_vehicle_count
            and stage1_has_solution
            and stage1_status == GRB.OPTIMAL
            and stage1_best_distance is not None
        ):
            remaining_time_s = max(
                0.0,
                float(self.config.time_limit_s) - (time.perf_counter() - start),
            )
            if remaining_time_s > 1e-6:
                tie_break_applied = True
                distance_tolerance = max(
                    float(self.config.distance_tolerance_abs),
                    float(self.config.distance_tolerance_rel)
                    * abs(float(stage1_best_distance)),
                )
                model.addConstr(
                    distance_expr
                    <= float(stage1_best_distance) + distance_tolerance,
                    name="distance_optimal_tolerance",
                )
                model.setObjective(vehicle_expr, GRB.MINIMIZE)
                model.Params.TimeLimit = remaining_time_s
                model.optimize()
                tie_break_status = int(model.Status)
                tie_break_status_name = self._status_name(tie_break_status)
            else:
                tie_break_skipped_no_time = True

        runtime = time.perf_counter() - start

        # The published status/bound/gap belong to the primary distance
        # objective.  A secondary vehicle-count solve has its own metadata and
        # must not downgrade a proven distance optimum to TIME_LIMIT.
        status = stage1_status
        status_name = stage1_status_name
        # Published distance results always use the frozen primary solution.
        # An optional secondary solve may inspect vehicle count, but its route
        # must not replace the distance-optimal incumbent or alter its gap.
        has_solution = stage1_has_solution
        routes = stage1_routes
        objective = stage1_route_distance
        route_validation = (
            validate_routes(instance, routes) if has_solution else None
        )
        best_bound = stage1_best_bound
        mip_gap = stage1_mip_gap
        feasible = (
            status
            in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL, GRB.INTERRUPTED}
            and has_solution
            and bool(route_validation and route_validation["passed"])
        )
        violations: dict[str, Any] = {}
        if not feasible:
            violations["gurobi_status"] = status
        if route_validation is not None and not route_validation["passed"]:
            violations["route_validation"] = route_validation["violations"]
        if (
            route_validation is not None
            and objective is not None
            and not math.isclose(
                float(objective),
                float(route_validation["objective_distance_km"]),
                rel_tol=1e-7,
                abs_tol=1e-6,
            )
        ):
            feasible = False
            violations["objective_distance_mismatch"] = {
                "solver_route_sum": objective,
                "independent_replay": route_validation[
                    "objective_distance_km"
                ],
            }

        benchmark_status = self._benchmark_status(
            status=status,
            has_solution=has_solution,
            feasible=bool(feasible),
        )
        benchmark_completed = bool(has_solution and feasible)
        self._validate_checkpoint_snapshots(instance, trace)
        self._annotate_checkpoint_statuses(
            trace,
            final_benchmark_status=benchmark_status,
            final_has_incumbent=bool(stage1_has_solution),
            terminal_budget_s=float(self.config.time_limit_s),
        )

        return EVRPTWSolution(
            instance_id=instance.instance_id,
            solver_name=self.name,
            routes=routes,
            objective_distance_km=objective,
            vehicle_count=len(routes) if routes else None,
            runtime_s=runtime,
            feasible=bool(feasible),
            violations=violations,
            metadata={
                "gurobi_status": status,
                "gurobi_status_name": status_name,
                "benchmark_status": benchmark_status,
                "benchmark_completed": benchmark_completed,
                "has_incumbent": bool(has_solution),
                "mip_gap": 0.0 if status == GRB.OPTIMAL and objective is not None else mip_gap,
                "best_bound": objective if status == GRB.OPTIMAL and objective is not None else best_bound,
                "cs_copies": int(self.config.cs_copies),
                "node_count_with_copies": len(node_map.solver_to_terminal),
                "checkpoints_s": list(trace["checkpoints_s"]),
                "first_feasible_time_s": trace["first_feasible_time_s"],
                "checkpoint_snapshots": trace["checkpoint_snapshots"],
                "incumbent_events": trace["incumbent_events"],
                "tie_break_vehicle_count": bool(self.config.tie_break_vehicle_count),
                "tie_break_applied": bool(tie_break_applied),
                "tie_break_skipped_no_time": bool(tie_break_skipped_no_time),
                "stage1_best_distance_km": stage1_best_distance,
                "stage1_optimization_runtime_s": stage1_runtime_s,
                "wall_runtime_s": runtime,
                "stage1_gurobi_status": stage1_status,
                "stage1_gurobi_status_name": stage1_status_name,
                "stage1_mip_gap": stage1_mip_gap,
                "stage1_best_bound": stage1_best_bound,
                "distance_tolerance": distance_tolerance,
                "tie_break_gurobi_status": tie_break_status,
                "tie_break_gurobi_status_name": tie_break_status_name,
                "tie_break_solution_discarded": bool(tie_break_applied),
                "route_validation": route_validation,
                **metric_metadata,
            },
        )

    def _build_model(self, instance: EVRPTWInstance) -> tuple[Model, NodeMap, dict[tuple[int, int], Any], Any, Any, dict[str, Any]]:
        n = instance.num_customers
        m = instance.num_charging_stations
        cs_copies = int(self.config.cs_copies) if m else 0

        solver_to_terminal = [0]
        solver_to_terminal.extend(range(1, n + 1))
        for _ in range(cs_copies):
            solver_to_terminal.extend(range(n + 1, n + 1 + m))
        solver_to_terminal.append(0)

        start_depot = 0
        end_depot = len(solver_to_terminal) - 1
        customer_nodes = list(range(1, n + 1))
        cs_nodes = list(range(n + 1, end_depot))
        node_map = NodeMap(solver_to_terminal, customer_nodes, cs_nodes, start_depot, end_depot)

        terminals = np.asarray(solver_to_terminal, dtype=int)
        distance, travel_s, energy_kwh, metric_metadata = self._resolve_arc_metric_matrices(instance, terminals)

        battery_capacity = float(instance.vehicle.get("battery_capacity_kwh", 100.0))
        cargo_capacity = float(instance.vehicle.get("cargo_capacity_cm3", np.inf))
        charging = resolve_charging_profile(instance)

        demand = np.zeros(len(solver_to_terminal), dtype=float)
        service = np.zeros(len(solver_to_terminal), dtype=float)
        ready = np.full(len(solver_to_terminal), float(instance.working_start_s), dtype=float)
        due = np.full(len(solver_to_terminal), float(instance.working_end_s), dtype=float)
        for local, customer_node in enumerate(customer_nodes):
            demand[customer_node] = float(instance.demands_cm3[local])
            service[customer_node] = float(instance.service_time_s[local])
            ready[customer_node] = float(instance.tw_s[local, 0])
            due[customer_node] = float(instance.tw_s[local, 1])

        charging_power_by_node: dict[int, float] = {}
        maximum_charge_time_by_node: dict[int, float] = {}
        for cs_node in cs_nodes:
            terminal = solver_to_terminal[cs_node]
            physical_station = terminal - (n + 1)
            power = float(charging.power_kw[physical_station])
            charging_power_by_node[cs_node] = power
            maximum_charge_time_by_node[cs_node] = (
                battery_capacity / (charging.power_factor * power) * 3600.0
            )

        model = Model(f"EVRPTW_{instance.instance_id}")
        model.Params.TimeLimit = float(self.config.time_limit_s)
        model.Params.MIPGap = float(self.config.mip_gap)
        model.Params.OutputFlag = int(self.config.output_flag)
        if self.config.threads is not None:
            model.Params.Threads = int(self.config.threads)

        recharge_nodes = {start_depot, *cs_nodes}
        end_nodes = set(customer_nodes + cs_nodes + [end_depot])
        start_nodes = set([start_depot] + customer_nodes + cs_nodes)
        arcs: list[tuple[int, int]] = []
        for i in start_nodes:
            for j in end_nodes:
                if i == j:
                    continue
                if i == start_depot and j == end_depot:
                    continue
                if i in cs_nodes and j in cs_nodes and solver_to_terminal[i] == solver_to_terminal[j]:
                    continue
                if not np.isfinite(distance[i, j]) or not np.isfinite(travel_s[i, j]) or not np.isfinite(energy_kwh[i, j]):
                    continue
                if energy_kwh[i, j] > battery_capacity + 1e-7:
                    continue
                arcs.append((i, j))

        x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
        tau = model.addVars(range(len(solver_to_terminal)), lb=ready.tolist(), ub=due.tolist(), vtype=GRB.CONTINUOUS, name="tau")
        load = model.addVars(range(len(solver_to_terminal)), lb=0.0, ub=cargo_capacity, vtype=GRB.CONTINUOUS, name="load")
        battery = model.addVars(range(len(solver_to_terminal)), lb=0.0, ub=battery_capacity, vtype=GRB.CONTINUOUS, name="battery")
        charge_time = {
            node: model.addVar(
                lb=0.0,
                ub=maximum_charge_time_by_node[node],
                vtype=GRB.CONTINUOUS,
                name=f"charge_time[{node}]",
            )
            for node in cs_nodes
        }

        distance_expr = quicksum(float(distance[i, j]) * x[i, j] for i, j in arcs)
        model.setObjective(distance_expr, GRB.MINIMIZE)

        incoming = {node: [] for node in range(len(solver_to_terminal))}
        outgoing = {node: [] for node in range(len(solver_to_terminal))}
        for i, j in arcs:
            outgoing[i].append((i, j))
            incoming[j].append((i, j))

        for c in customer_nodes:
            model.addConstr(quicksum(x[a] for a in incoming[c]) == 1, name=f"customer_in_{c}")
            model.addConstr(quicksum(x[a] for a in outgoing[c]) == 1, name=f"customer_out_{c}")

        for f in cs_nodes:
            visit = quicksum(x[a] for a in incoming[f])
            model.addConstr(visit == quicksum(x[a] for a in outgoing[f]), name=f"cs_flow_{f}")
            model.addConstr(visit <= 1, name=f"cs_visit_{f}")
            max_charge_s = maximum_charge_time_by_node[f]
            seconds_per_kwh = 3600.0 / (
                charging.power_factor * charging_power_by_node[f]
            )
            exact_charge_s = seconds_per_kwh * (
                battery_capacity - battery[f]
            )
            model.addConstr(
                charge_time[f] <= max_charge_s * visit,
                name=f"charge_zero_if_unvisited_{f}",
            )
            model.addConstr(
                charge_time[f]
                <= exact_charge_s + max_charge_s * (1 - visit),
                name=f"charge_time_upper_{f}",
            )
            model.addConstr(
                charge_time[f]
                >= exact_charge_s - max_charge_s * (1 - visit),
                name=f"charge_time_lower_{f}",
            )

        vehicle_expr = quicksum(x[a] for a in outgoing[start_depot])
        model.addConstr(vehicle_expr == quicksum(x[a] for a in incoming[end_depot]), name="depot_balance")
        model.addConstr(vehicle_expr >= 1, name="at_least_one_route")
        model.addConstr(tau[start_depot] == float(instance.working_start_s), name="start_time")
        model.addConstr(load[start_depot] == 0.0, name="start_load")
        model.addConstr(battery[start_depot] == battery_capacity, name="start_battery")

        horizon = float(instance.working_end_s - instance.working_start_s)
        max_arc_time = float(np.nanmax(travel_s[np.isfinite(travel_s)])) if np.any(np.isfinite(travel_s)) else 0.0
        maximum_charge_time_s = max(maximum_charge_time_by_node.values(), default=0.0)
        big_m_time = max(1.0, horizon + max_arc_time + float(np.max(service)) + maximum_charge_time_s + 1.0)
        big_m_load = max(1.0, cargo_capacity + float(np.sum(instance.demands_cm3)) + 1.0)
        max_arc_energy = float(np.nanmax(energy_kwh[np.isfinite(energy_kwh)])) if np.any(np.isfinite(energy_kwh)) else 0.0
        big_m_battery = max(1.0, battery_capacity + max_arc_energy + 1.0)

        for i, j in arcs:
            departure_charge_time = charge_time[i] if i in charge_time else 0.0
            model.addConstr(
                tau[j] >= tau[i] + float(service[i]) + departure_charge_time + float(travel_s[i, j]) - big_m_time * (1 - x[i, j]),
                name=f"time_{i}_{j}",
            )
            model.addConstr(
                load[j] >= load[i] + float(demand[j]) - big_m_load * (1 - x[i, j]),
                name=f"load_{i}_{j}",
            )
            energy = float(energy_kwh[i, j])
            if i in recharge_nodes:
                arrival_battery = battery_capacity - energy
            else:
                arrival_battery = battery[i] - energy

            if j == end_depot:
                if i not in recharge_nodes:
                    model.addConstr(
                        battery[i]
                        >= energy - big_m_battery * (1 - x[i, j]),
                        name=f"battery_to_depot_{i}_{j}",
                    )
            else:
                model.addConstr(
                    battery[j]
                    <= arrival_battery + big_m_battery * (1 - x[i, j]),
                    name=f"battery_upper_{i}_{j}",
                )
                model.addConstr(
                    battery[j]
                    >= arrival_battery - big_m_battery * (1 - x[i, j]),
                    name=f"battery_lower_{i}_{j}",
                )

        metric_metadata.update(
            {
                "charging_time_model": "arrival_soc_station_power_linear_full_charge_derated_v2",
                "charging_power_source": charging.power_source,
                "charging_power_factor_source": charging.power_factor_source,
                "charging_power_derating_factor": charging.power_factor,
                "charging_power_min_kw": (
                    float(np.min(charging.power_kw)) if charging.power_kw.size else None
                ),
                "charging_power_max_kw": (
                    float(np.max(charging.power_kw)) if charging.power_kw.size else None
                ),
                "maximum_zero_to_full_charge_time_s": maximum_charge_time_s,
                "battery_propagation": "conditional_equality_except_shared_end_depot",
            }
        )

        return model, node_map, x, distance_expr, vehicle_expr, metric_metadata

    def _resolve_arc_metric_matrices(
        self,
        instance: EVRPTWInstance,
        terminals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Return the current objective, running-time, and running-path energy matrices."""

        distance = instance.distance_matrix_km[np.ix_(terminals, terminals)].astype(float)

        stage2_travel = instance.raw.get("running_time_shortest_matrix_s")
        if stage2_travel is not None:
            travel_s = np.asarray(stage2_travel, dtype=float)[np.ix_(terminals, terminals)]
            travel_time_source = "running_time_shortest_matrix_s"
        elif instance.raw_travel_time_matrix_s is not None:
            travel_s = instance.raw_travel_time_matrix_s[np.ix_(terminals, terminals)].astype(float)
            travel_time_source = "raw_travel_time_matrix_s"
        else:
            effective_speed = float(
                instance.speed_profile.get("effective_speed_kmh")
                or instance.vehicle.get("design_speed_kmh")
                or 40.0
            )
            travel_s = distance / max(effective_speed, 1e-9) * 3600.0
            travel_time_source = "distance_over_effective_speed"

        stage2_energy = instance.raw.get("running_time_path_energy_kwh")
        if stage2_energy is not None:
            energy_kwh = np.asarray(stage2_energy, dtype=float)[np.ix_(terminals, terminals)]
            energy_source = "running_time_path_energy_kwh"
        elif instance.energy_matrix_kwh is not None:
            energy_kwh = instance.energy_matrix_kwh[np.ix_(terminals, terminals)].astype(float)
            energy_source = "energy_matrix_kwh"
        else:
            consumption = float(instance.vehicle.get("consumption_kwh_per_km", 0.404))
            energy_kwh = distance * consumption
            energy_source = "distance_times_vehicle_consumption"

        metric_metadata = {
            "distance_matrix_source": "distance_matrix_km",
            "travel_time_matrix_source": travel_time_source,
            "energy_matrix_source": energy_source,
            "ev_transition_time_matrix_available": instance.ev_transition_time_matrix_s is not None,
            "shortest_time_matrix_available": instance.shortest_time_matrix_s is not None,
            "distance_asymmetry_max_km": self._max_asymmetry(distance),
            "travel_time_asymmetry_max_s": self._max_asymmetry(travel_s),
            "energy_asymmetry_max_kwh": self._max_asymmetry(energy_kwh),
        }
        return distance, travel_s, energy_kwh, metric_metadata

    @staticmethod
    def _max_asymmetry(matrix: np.ndarray) -> float | None:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return None
        with np.errstate(invalid="ignore"):
            diff = np.abs(matrix - matrix.T)
        finite = diff[np.isfinite(diff)]
        if finite.size == 0:
            return None
        return float(np.max(finite))

    @staticmethod
    def _route_distance_km(routes: list[list[int]], instance: EVRPTWInstance) -> float:
        distance = np.asarray(instance.distance_matrix_km, dtype=float)
        total = 0.0
        for route in routes:
            for i in range(len(route) - 1):
                total += float(distance[int(route[i]), int(route[i + 1])])
        return total

    def _new_trace(self) -> dict[str, Any]:
        checkpoints = tuple(sorted({float(t) for t in self.config.checkpoints_s if float(t) >= 0.0}))
        return {
            "checkpoints_s": checkpoints,
            "next_checkpoint_index": 0,
            "first_feasible_time_s": None,
            "last_incumbent": None,
            "last_best_bound": None,
            "last_best_obj": None,
            "checkpoint_snapshots": [],
            "incumbent_events": [],
        }

    def _make_callback(self, trace: dict[str, Any], node_map: NodeMap, x: dict[tuple[int, int], Any]):
        arcs = list(x.keys())
        x_vars = [x[arc] for arc in arcs]

        def callback(model: Model, where: int) -> None:
            runtime = self._callback_float(model, GRB.Callback.RUNTIME)
            if runtime is None:
                return

            if where == GRB.Callback.MIPSOL:
                # A solution reported at runtime r must never be back-filled
                # into a checkpoint t < r. Flush those checkpoints using the
                # state observed by the preceding callback first.
                self._record_due_checkpoints(
                    trace,
                    runtime,
                    "RUNNING",
                    include_equal=False,
                )
                objective = self._callback_float(model, GRB.Callback.MIPSOL_OBJ)
                best_bound = self._callback_float(model, GRB.Callback.MIPSOL_OBJBND)
                if best_bound is None:
                    best_bound = trace.get("last_best_bound")
                values = model.cbGetSolution(x_vars)
                arc_values = {arc: float(value) for arc, value in zip(arcs, values)}
                routes = self._extract_routes_from_arc_values(node_map, arc_values)
                snapshot = self._make_snapshot(
                    checkpoint_s=None,
                    elapsed_s=runtime,
                    reached_checkpoint=True,
                    solver_status="RUNNING",
                    objective_distance_km=objective,
                    best_bound=best_bound,
                    routes=routes,
                    source="incumbent",
                )
                trace["last_incumbent"] = snapshot
                trace["last_best_obj"] = objective
                if best_bound is not None:
                    trace["last_best_bound"] = best_bound
                if trace["first_feasible_time_s"] is None:
                    trace["first_feasible_time_s"] = runtime
                if self.config.record_incumbent_events:
                    event = dict(snapshot)
                    event.pop("routes", None)
                    event.pop("route_sequence", None)
                    trace["incumbent_events"].append(event)
                self._record_due_checkpoints(
                    trace,
                    runtime,
                    "RUNNING",
                    include_equal=True,
                )

            elif where == GRB.Callback.MIP:
                self._record_due_checkpoints(
                    trace,
                    runtime,
                    "RUNNING",
                    include_equal=False,
                )
                best_obj = self._callback_float(model, GRB.Callback.MIP_OBJBST)
                best_bound = self._callback_float(model, GRB.Callback.MIP_OBJBND)
                if best_obj is not None and math.isfinite(best_obj):
                    trace["last_best_obj"] = best_obj
                if best_bound is not None and math.isfinite(best_bound):
                    trace["last_best_bound"] = best_bound
                self._record_due_checkpoints(
                    trace,
                    runtime,
                    "RUNNING",
                    include_equal=True,
                )

        return callback

    def _record_due_checkpoints(
        self,
        trace: dict[str, Any],
        elapsed_s: float,
        solver_status: str,
        *,
        include_equal: bool = True,
    ) -> None:
        checkpoints = trace["checkpoints_s"]
        while trace["next_checkpoint_index"] < len(checkpoints):
            checkpoint_s = checkpoints[trace["next_checkpoint_index"]]
            if checkpoint_s > elapsed_s or (
                not include_equal and checkpoint_s == elapsed_s
            ):
                break
            trace["checkpoint_snapshots"].append(
                self._snapshot_at_checkpoint(
                    trace,
                    checkpoint_s,
                    elapsed_s,
                    True,
                    solver_status,
                )
            )
            trace["next_checkpoint_index"] += 1

    def _finalize_checkpoints(self, trace: dict[str, Any], final_snapshot: dict[str, Any], runtime_s: float, solver_status: str) -> None:
        checkpoints = trace["checkpoints_s"]

        # Preserve causality for a checkpoint crossed between the last callback
        # and optimize() returning. Only a checkpoint exactly at the final
        # runtime may consume the final state.
        self._record_due_checkpoints(
            trace,
            runtime_s,
            solver_status,
            include_equal=False,
        )
        if final_snapshot.get("has_incumbent"):
            trace["last_incumbent"] = final_snapshot
            trace["last_best_obj"] = final_snapshot.get("objective_distance_km")
        if final_snapshot.get("best_bound") is not None:
            trace["last_best_bound"] = final_snapshot.get("best_bound")
        self._record_due_checkpoints(
            trace,
            runtime_s,
            solver_status,
            include_equal=True,
        )

        while trace["next_checkpoint_index"] < len(checkpoints):
            checkpoint_s = checkpoints[trace["next_checkpoint_index"]]
            trace["checkpoint_snapshots"].append(
                self._snapshot_at_checkpoint(
                    trace,
                    checkpoint_s,
                    runtime_s,
                    False,
                    solver_status,
                )
            )
            trace["next_checkpoint_index"] += 1

    def _snapshot_at_checkpoint(
        self,
        trace: dict[str, Any],
        checkpoint_s: float,
        elapsed_s: float,
        reached_checkpoint: bool,
        solver_status: str,
    ) -> dict[str, Any]:
        incumbent = trace.get("last_incumbent")
        if incumbent is None:
            return self._make_snapshot(
                checkpoint_s=checkpoint_s,
                elapsed_s=checkpoint_s if reached_checkpoint else elapsed_s,
                reached_checkpoint=reached_checkpoint,
                solver_status=solver_status,
                objective_distance_km=None,
                best_bound=trace.get("last_best_bound"),
                routes=[],
                source=(
                    "checkpoint_no_incumbent"
                    if reached_checkpoint
                    else "final_no_incumbent"
                ),
            )

        best_bound = trace.get("last_best_bound")
        if best_bound is None:
            best_bound = incumbent.get("best_bound")
        return self._make_snapshot(
            checkpoint_s=checkpoint_s,
            elapsed_s=checkpoint_s if reached_checkpoint else elapsed_s,
            reached_checkpoint=reached_checkpoint,
            solver_status=solver_status,
            objective_distance_km=incumbent.get("objective_distance_km"),
            best_bound=best_bound,
            routes=incumbent.get("routes", []),
            source="checkpoint_incumbent" if reached_checkpoint else "final_after_early_stop",
        )

    @staticmethod
    def _benchmark_status(
        *,
        status: int,
        has_solution: bool,
        feasible: bool,
    ) -> str:
        if has_solution and feasible:
            return (
                "COMPLETED_OPTIMAL"
                if status == GRB.OPTIMAL
                else "COMPLETED_WITH_INCUMBENT"
            )
        if has_solution:
            return "INVALID_INCUMBENT"
        if status == GRB.TIME_LIMIT:
            return "UNFINISHED_NO_INCUMBENT"
        if status in {GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED}:
            return "NO_FEASIBLE_SOLUTION"
        return "UNFINISHED_NO_INCUMBENT"

    @staticmethod
    def _validate_checkpoint_snapshots(
        instance: EVRPTWInstance,
        trace: dict[str, Any],
    ) -> None:
        """Replay every distinct checkpoint route before it is published."""

        validation_cache: dict[tuple[tuple[int, ...], ...], dict[str, Any]] = {}
        for snapshot in trace.get("checkpoint_snapshots", []):
            routes = [list(map(int, route)) for route in snapshot.get("routes", [])]
            if not routes:
                snapshot["route_validation_passed"] = None
                snapshot["route_validation"] = None
                continue

            cache_key = tuple(tuple(route) for route in routes)
            cached_validation = validation_cache.get(cache_key)
            if cached_validation is None:
                try:
                    cached_validation = dict(validate_routes(instance, routes))
                except Exception as exc:  # preserve a bad snapshot diagnostically
                    cached_validation = {
                        "passed": False,
                        "violations": [
                            f"route replay raised {type(exc).__name__}: {exc}"
                        ],
                    }
                validation_cache[cache_key] = cached_validation

            validation = dict(cached_validation)
            validation["violations"] = list(validation.get("violations", []))
            claimed_objective = snapshot.get("objective_distance_km")
            replay_objective = validation.get("objective_distance_km")
            if (
                validation.get("passed")
                and claimed_objective is not None
                and replay_objective is not None
                and not math.isclose(
                    float(claimed_objective),
                    float(replay_objective),
                    rel_tol=1e-7,
                    abs_tol=1e-6,
                )
            ):
                validation["passed"] = False
                validation["violations"].append(
                    "checkpoint objective does not match independent replay"
                )

            snapshot["route_validation"] = validation
            snapshot["route_validation_passed"] = bool(validation.get("passed"))
            if validation.get("passed"):
                continue

            # Keep the rejected incumbent for debugging, but never expose it
            # through the benchmark's feasible objective/route columns.
            snapshot["diagnostic_objective_distance_km"] = snapshot.get(
                "objective_distance_km"
            )
            snapshot["diagnostic_routes"] = routes
            snapshot["diagnostic_route_sequence"] = snapshot.get(
                "route_sequence", []
            )
            snapshot["diagnostic_mip_gap"] = snapshot.get("mip_gap")
            snapshot["has_incumbent"] = False
            snapshot["objective_distance_km"] = None
            snapshot["mip_gap"] = None
            snapshot["vehicle_count"] = None
            snapshot["routes"] = []
            snapshot["route_sequence"] = []
            snapshot["source"] = f"{snapshot.get('source', 'checkpoint')}_invalid"

    @staticmethod
    def _annotate_checkpoint_statuses(
        trace: dict[str, Any],
        *,
        final_benchmark_status: str,
        final_has_incumbent: bool,
        terminal_budget_s: float,
    ) -> None:
        snapshots = trace.get("checkpoint_snapshots", [])
        for snapshot in snapshots:
            if snapshot.get("route_validation_passed") is False:
                snapshot["benchmark_status"] = "INVALID_INCUMBENT"
            elif snapshot.get("has_incumbent"):
                snapshot["benchmark_status"] = "INCUMBENT_AVAILABLE"
            elif not snapshot.get("reached_checkpoint"):
                snapshot["benchmark_status"] = final_benchmark_status
            elif (
                not final_has_incumbent
                and float(snapshot.get("checkpoint_s") or 0.0)
                >= terminal_budget_s - 1e-6
            ):
                snapshot["benchmark_status"] = final_benchmark_status
            else:
                snapshot["benchmark_status"] = "NO_INCUMBENT_YET"

    def _make_snapshot(
        self,
        checkpoint_s: float | None,
        elapsed_s: float,
        reached_checkpoint: bool,
        solver_status: str,
        objective_distance_km: float | None,
        best_bound: float | None,
        routes: list[list[int]],
        source: str,
    ) -> dict[str, Any]:
        gap = self._relative_gap(objective_distance_km, best_bound)
        return {
            "checkpoint_s": checkpoint_s,
            "elapsed_s": float(elapsed_s),
            "reached_checkpoint": bool(reached_checkpoint),
            "solver_status": solver_status,
            "has_incumbent": bool(routes),
            "objective_distance_km": objective_distance_km,
            "best_bound": best_bound,
            "mip_gap": gap,
            "vehicle_count": len(routes) if routes else None,
            "routes": routes,
            "route_sequence": self._flatten_routes(routes),
            "source": source,
        }

    def _extract_routes(self, node_map: NodeMap, x: dict[tuple[int, int], Any]) -> list[list[int]]:
        return self._extract_routes_from_arc_values(node_map, {arc: float(var.X) for arc, var in x.items()})

    def _extract_routes_from_arc_values(self, node_map: NodeMap, arc_values: dict[tuple[int, int], float]) -> list[list[int]]:
        outgoing: dict[int, list[int]] = {}
        for (i, j), value in arc_values.items():
            if value > 0.5:
                outgoing.setdefault(i, []).append(j)

        routes: list[list[int]] = []
        starts = sorted(outgoing.get(node_map.start_depot, []))
        for first in starts:
            route_solver_nodes = [node_map.start_depot, first]
            current = first
            seen = {node_map.start_depot}
            while current != node_map.end_depot:
                if current in seen:
                    break
                seen.add(current)
                next_nodes = sorted(outgoing.get(current, []))
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                route_solver_nodes.append(nxt)
                current = nxt

            mapped = []
            for solver_node in route_solver_nodes:
                terminal = int(node_map.solver_to_terminal[solver_node])
                if solver_node == node_map.end_depot:
                    terminal = 0
                if not mapped or mapped[-1] != terminal:
                    mapped.append(terminal)
            if mapped and mapped[-1] != 0:
                mapped.append(0)
            routes.append(mapped)
        return routes

    @staticmethod
    def _flatten_routes(routes: list[list[int]]) -> list[int]:
        return merge_route_sequences(routes)

    @staticmethod
    def _relative_gap(objective: float | None, bound: float | None) -> float | None:
        if objective is None or bound is None:
            return None
        if not math.isfinite(objective) or not math.isfinite(bound):
            return None
        denom = max(abs(objective), 1e-9)
        return float(abs(objective - bound) / denom)

    @staticmethod
    def _callback_float(model: Model, what: int) -> float | None:
        try:
            value = float(model.cbGet(what))
        except Exception:
            return None
        return value if math.isfinite(value) else None

    @staticmethod
    def _safe_model_float(model: Model, attr: str) -> float | None:
        try:
            value = float(getattr(model, attr))
        except Exception:
            return None
        return value if math.isfinite(value) else None

    @staticmethod
    def _status_name(status: int) -> str:
        names: dict[int, str] = {}
        for name in (
            "OPTIMAL",
            "INFEASIBLE",
            "INF_OR_UNBD",
            "UNBOUNDED",
            "CUTOFF",
            "ITERATION_LIMIT",
            "NODE_LIMIT",
            "TIME_LIMIT",
            "SOLUTION_LIMIT",
            "INTERRUPTED",
            "NUMERIC",
            "SUBOPTIMAL",
            "INPROGRESS",
            "USER_OBJ_LIMIT",
        ):
            value = getattr(GRB, name, None)
            if value is not None:
                names[int(value)] = name
        return names.get(status, str(status))
