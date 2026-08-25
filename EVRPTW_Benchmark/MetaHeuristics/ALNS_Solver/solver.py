from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from itertools import islice
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


def merge_routes(routes: list[list[int]]) -> list[int]:
    """
    Merge routes like:
    [[0, ..., 0], [0, ..., 0], ...]
    into one sequence, removing duplicated 0 between adjacent routes.

    Example:
    [[0,1,2,0],[0,3,4,0]] -> [0,1,2,0,3,4,0]
    """
    if not routes:
        return []

    merged = routes[0].copy()

    for route in routes[1:]:
        if not route:
            continue

        if merged and merged[-1] == 0 and route[0] == 0:
            merged.extend(route[1:])
        else:
            merged.extend(route)

    return merged[1:]


def assemble_nodes(instance, order="depot-customer-station"):
    """
    Return:
        nodes: list[dict]

    order:
        - "depot-station-customer"
        - "depot-customer-station"
    """
    depot = instance["depot"]
    customers = instance.get("customers", [])
    charging_stations = instance.get("charging_stations", [])
    env = instance["env"]

    working_start = float(env["instance_startTime"]) / 60.0
    working_end = float(env["instance_endTime"]) / 60.0

    def to_2d(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return arr.reshape(0, 2)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    depot = to_2d(depot)
    customers = to_2d(customers)
    charging_stations = to_2d(charging_stations)

    def get_customer_field(candidates, n, default=0.0):
        for key in candidates:
            if key in instance:
                arr = np.asarray(instance[key], dtype=float).reshape(-1)
                if len(arr) != n:
                    raise ValueError(
                        f"Field '{key}' length = {len(arr)}, but number of customers = {n}"
                    )
                return arr
        return np.full(n, default, dtype=float)

    n_customers = len(customers)

    customer_demands = get_customer_field(
        ["customer_demand", "demands", "demand"], n_customers, default=0.0
    )
    customer_service = get_customer_field(
        ["customer_service", "service_times", "service_time", "service"],
        n_customers,
        default=0.0,
    )

    if len(depot) != 1:
        raise ValueError(f"Expect exactly 1 depot, but got {len(depot)}")

    depot_node = {
        "id": "D0",
        "type": "d",
        "x": float(depot[0, 0]),
        "y": float(depot[0, 1]),
        "demand": 0.0,
        "ready": working_start,
        "due": working_end,
        "service": 0.0,
    }

    customer_nodes = []
    for i in range(n_customers):
        customer_nodes.append(
            {
                "id": f"C{i}",
                "type": "c",
                "x": float(customers[i, 0]),
                "y": float(customers[i, 1]),
                "demand": float(customer_demands[i]),
                "ready": float(instance["tw"][i][0]) / 60.0,
                "due": float(instance["tw"][i][1]) / 60.0,
                "service": float(customer_service[i]) / 60.0,
            }
        )

    station_nodes = []
    for i in range(len(charging_stations)):
        station_nodes.append(
            {
                "id": f"S{i+1}",
                "type": "f",
                "x": float(charging_stations[i, 0]),
                "y": float(charging_stations[i, 1]),
                "demand": 0.0,
                "ready": working_start,
                "due": working_end,
                "service": 0.0,
            }
        )

    nodes = [depot_node]
    if order == "depot-station-customer":
        nodes.extend(station_nodes)
        nodes.extend(customer_nodes)
    elif order == "depot-customer-station":
        nodes.extend(customer_nodes)
        nodes.extend(station_nodes)
    else:
        raise ValueError("order must be 'depot-station-customer' or 'depot-customer-station'")

    return nodes


class ALNS_Solver:
    """
    Keskin & Catay (2016)-style ALNS refactored for:
      - EVRPTW
      - full recharging only
      - objective = minimal feasible total distance

    Internal route representation:
      route = [0, ..., 0]
      where node indices follow:
        0 = depot
        1..N = customers
        N+1 .. N+S = charging stations
    """

    def __init__(
        self,
        instance: Dict[str, Any],
        seed: int = 1234,
        format=None,
        checkpoint: Optional[Dict[str, Any]] = None,
    ):
        self.instance = instance
        self.rng = random.Random(seed)
        self.format = format

        if format == "tensor":
            self.n_stations = len(instance.get("charging_stations", []))
            self.n_customers = len(instance.get("customers", []))
            self.nodes = assemble_nodes(instance, order="depot-customer-station")
            self.depot = self.nodes[0]

            self.customer_indices = list(range(1, 1 + self.n_customers))
            self.station_indices = list(range(1 + self.n_customers, len(self.nodes)))
            self.customer_to_mask = {idx: k for k, idx in enumerate(self.customer_indices)}

            self.vehicle = {
                "Q": instance["env"]["battery_capacity"],
                "C": instance["env"]["loading_capacity"],
                "r": instance["env"]["consumption_per_distance"],
                "g": (
                    0.0
                    if math.isinf(float(instance["env"]["charging_speed"]))
                    else 1.0 / float(instance["env"]["charging_speed"])
                ),
                "v": instance["env"]["speed"],
            }

            self.Q = float(self.vehicle["Q"])
            self.C = float(self.vehicle["C"])
            self.r = float(self.vehicle["r"])
            self.g = float(self.vehicle["g"])
            self.v = float(self.vehicle["v"])

        else:
            self.depot = instance["depot"]
            self.stations = instance.get("stations", [])
            self.customers = instance.get("customers", [])
            self.vehicle = instance["vehicle"]

            self.Q = float(self.vehicle["Q"])
            self.C = float(self.vehicle["C"])
            self.r = float(self.vehicle["r"])
            self.g = float(self.vehicle["g"])
            self.v = float(self.vehicle["v"])

            self.nodes: List[Dict[str, Any]] = [self.depot] + self.customers + self.stations
            self.n_stations = len(self.stations)
            self.n_customers = len(self.customers)

            self.customer_indices = list(range(1, 1 + self.n_customers))
            self.station_indices = list(range(1 + self.n_customers, len(self.nodes)))
            self.customer_to_mask = {idx: k for k, idx in enumerate(self.customer_indices)}

        coords = np.array([[node["x"], node["y"]] for node in self.nodes], dtype=float)
        if format == "tensor" and "distance_matrix_km" in instance:
            self.dist_matrix = np.asarray(instance["distance_matrix_km"], dtype=float)
            expected_shape = (len(self.nodes), len(self.nodes))
            if self.dist_matrix.shape != expected_shape:
                raise ValueError(
                    f"distance_matrix_km shape {self.dist_matrix.shape} does not match node shape {expected_shape}"
                )
        else:
            diff = coords[:, None, :] - coords[None, :, :]
            self.dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        if format == "tensor" and "time_matrix_min" in instance:
            self.time_matrix = np.asarray(instance["time_matrix_min"], dtype=float)
            expected_shape = (len(self.nodes), len(self.nodes))
            if self.time_matrix.shape != expected_shape:
                raise ValueError(
                    f"time_matrix_min shape {self.time_matrix.shape} does not match node shape {expected_shape}"
                )
        else:
            self.time_matrix = self.dist_matrix / max(1e-12, self.v)

        if format == "tensor" and "energy_matrix_kwh" in instance:
            self.energy_matrix = np.asarray(instance["energy_matrix_kwh"], dtype=float)
            expected_shape = (len(self.nodes), len(self.nodes))
            if self.energy_matrix.shape != expected_shape:
                raise ValueError(
                    f"energy_matrix_kwh shape {self.energy_matrix.shape} does not match node shape {expected_shape}"
                )
        else:
            self.energy_matrix = self.dist_matrix * self.r

        # The matrices are immutable for the lifetime of a solver.  Keeping the
        # maximum avoids rescanning the full O(n^2) distance matrix in every
        # regret-insertion round.
        self.max_distance = float(self.dist_matrix.max())

        self.station_charge_minutes_per_kwh: Dict[int, float] = {}
        self.certificate_singleton_routes = instance.get("certificate_singleton_routes")
        self.feasibility_certificate = instance.get("feasibility_certificate")
        if format == "tensor":
            power = np.asarray(instance.get("charging_power_kw", []), dtype=float)
            power_factor = float(
                instance.get(
                    "charging_power_derating_factor",
                    instance.get("charging_efficiency", 1.0),
                )
            )
            if power.shape != (self.n_stations,):
                raise ValueError(
                    f"charging_power_kw shape {power.shape} does not match {(self.n_stations,)}"
                )
            if not 0.0 < power_factor <= 1.0:
                raise ValueError("charging power factor must be in (0, 1]")
            if np.any(~np.isfinite(power)) or np.any(power <= 0.0):
                raise ValueError("charging_power_kw must contain finite positive values")
            self.station_charge_minutes_per_kwh = {
                node: 60.0 / (power_factor * float(power[offset]))
                for offset, node in enumerate(self.station_indices)
            }

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.cur_iter = 1
        if self.format == "tensor":
            self.max_iters = 200
            self.NC = 20
            self.NS = 100
            self.NSR = 40
            self.NRR = 200
            self.nRR = 50
        else:
            self.max_iters = 3000
            self.NC = 200
            self.NS = 2000
            self.NSR = 60
            self.NRR = 4000
            self.nRR = 1000

        self.min_remove_customers = max(1, min(int(0.10 * max(1, self.n_customers)), 30))
        self.max_remove_customers = max(
            self.min_remove_customers,
            min(int(0.40 * max(1, self.n_customers)), 60),
        )
        self.route_removal_upper_ratio = 0.40

        self.reaction_factor = 0.2

        self.temperature: Optional[float] = None
        self.cooling_rate = 0.9994
        self.initial_temp_control = 0.2

        self.r1 = 30.0
        self.r2 = 20.0
        self.r3 = 25.0
        self.r4 = 0.0

        self.phi1 = 7.0
        self.phi2 = 13.0
        self.phi3 = 1.0
        self.phi4 = 0.25

        self.worst_determinism = 5
        self.shaw_deteminism = 6
        self.n_zones = 9

        # Preserve ``station_indices`` as a list because its iteration order is
        # part of deterministic neighbourhood enumeration.  Membership checks,
        # however, are frequent enough to warrant a separate O(1) lookup set.
        self.station_index_set = set(self.station_indices)
        self.customer_zone_map = self._build_zone_map(self.customer_indices)

        # ------------------------------------------------------------------
        # Operator pools
        # ------------------------------------------------------------------
        self.cr_ops = {
            "random_customer": self._cr_random_customer,
            "worst_distance": self._cr_worst_distance,
            "worst_time": self._cr_worst_time,
            "shaw": self._cr_shaw,
            "proximity": self._cr_proximity,
            "demand_based": self._cr_demand_based,
            "time_based": self._cr_time_based,
            "zone_removal": self._cr_zone_removal,
            "random_route": self._cr_random_route,
            "greedy_route": self._cr_greedy_route,
        }

        self.ci_ops = {
            "greedy": self._ci_greedy,
            "regret2": self._ci_regret2,
            "regret3": self._ci_regret3,
            "time_based": self._ci_time_based,
            "zone_insertion": self._ci_zone_insertion,
        }

        self.sr_ops = {
            "random_station": self._sr_random_station,
            "worst_distance_station": self._sr_worst_distance_station,
            "worst_charge_usage_station": self._sr_worst_charge_usage_station,
            "full_charge_station": self._sr_full_charge_station,
        }

        self.si_ops = {
            "gsi": self._si_greedy_station_insertion,
            "gsi_comparison": self._si_greedy_station_insertion_with_comparison,
            "best_station": self._si_best_station_insertion,
        }

        self.cr_weights = {k: 1.0 for k in self.cr_ops}
        self.ci_weights = {k: 1.0 for k in self.ci_ops}
        self.sr_weights = {k: 1.0 for k in self.sr_ops}
        self.si_weights = {k: 1.0 for k in self.si_ops}

        self.cr_scores = {k: 0.0 for k in self.cr_ops}
        self.ci_scores = {k: 0.0 for k in self.ci_ops}
        self.sr_scores = {k: 0.0 for k in self.sr_ops}
        self.si_scores = {k: 0.0 for k in self.si_ops}

        self.cr_uses = {k: 0 for k in self.cr_ops}
        self.ci_uses = {k: 0 for k in self.ci_ops}
        self.sr_uses = {k: 0 for k in self.sr_ops}
        self.si_uses = {k: 0 for k in self.si_ops}

        self.attribute_frequency = defaultdict(int)
        self.attribute_total = 0
        self.lambda_div = 0.0

        # ------------------------------------------------------------------
        # Caches
        # ------------------------------------------------------------------
        self._sim_cache: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        self._feasibility_cache: Dict[Tuple[int, ...], bool] = {}
        self._dist_cache: Dict[Tuple[int, ...], float] = {}
        self._demand_cache: Dict[Tuple[int, ...], float] = {}
        self._time_cache: Dict[Tuple[int, ...], float] = {}
        self._best_station_arc_cache: Dict[Tuple[Tuple[int, ...], int], Optional[List[int]]] = {}

        self.heavy_postprocess_interval = 200

        # Construction is deliberately bounded: a complete singleton solution
        # is available almost immediately on certified Stage-2 instances, after
        # which a deterministic best-fit pass may improve vehicle consolidation
        # without delaying the first published incumbent indefinitely.
        if self.n_customers >= 500:
            self.initial_construction_budget_s = 5.0
            self.initial_merge_candidate_limit = 12
            self.initial_exact_insertion_limit = 6
            self.customer_top_k = 8
            self.simulation_cache_limit = 12_000
        elif self.n_customers >= 100:
            self.initial_construction_budget_s = 8.0
            self.initial_merge_candidate_limit = 20
            self.initial_exact_insertion_limit = 12
            self.customer_top_k = 16
            self.simulation_cache_limit = 20_000
        else:
            self.initial_construction_budget_s = 12.0
            self.initial_merge_candidate_limit = None
            self.initial_exact_insertion_limit = 24
            self.customer_top_k = None
            self.simulation_cache_limit = 30_000
        self.scalar_cache_limit = 50_000
        self.station_cache_limit = 12_000
        self.initial_construction_strategy = "singleton_best_fit_v1"
        self.algorithm_profile_id = "alns_stage2_scalable_v2"
        self.initial_construction_stats: Dict[str, Any] = {}
        self.singleton_source = "solver_repair"
        self._singleton_route_cache: Dict[int, List[int]] = {}

        # ------------------------------------------------------------------
        # Search state
        # ------------------------------------------------------------------
        self.current_routes: List[List[int]] = []
        self.best_routes: List[List[int]] = []
        self.global_value: float = float("inf")
        self.visited: List[bool] = [False] * self.n_customers
        self.terminated_by_time_limit = False

        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

    def get_checkpoint(self) -> Dict[str, Any]:
        """
        Export current solver state so we can resume later.
        """
        return {
            "cur_iter": int(self.cur_iter),
            "max_iters": int(self.max_iters),

            "current_routes": [list(r) for r in self.current_routes],
            "best_routes": [list(r) for r in self.best_routes],
            "global_value": float(self.global_value),
            "temperature": None if self.temperature is None else float(self.temperature),

            "cr_weights": dict(self.cr_weights),
            "ci_weights": dict(self.ci_weights),
            "sr_weights": dict(self.sr_weights),
            "si_weights": dict(self.si_weights),

            "cr_scores": dict(self.cr_scores),
            "ci_scores": dict(self.ci_scores),
            "sr_scores": dict(self.sr_scores),
            "si_scores": dict(self.si_scores),

            "cr_uses": dict(self.cr_uses),
            "ci_uses": dict(self.ci_uses),
            "sr_uses": dict(self.sr_uses),
            "si_uses": dict(self.si_uses),

            "attribute_frequency": dict(self.attribute_frequency),
            "attribute_total": int(self.attribute_total),

            "visited": list(self.visited),
        }

    def get_run_metadata(self) -> Dict[str, Any]:
        """Return the effective, scale-adaptive algorithm profile for outputs."""

        return {
            "algorithm_profile_id": self.algorithm_profile_id,
            "initial_construction_strategy": self.initial_construction_strategy,
            "initial_construction": dict(self.initial_construction_stats),
            "singleton_source": self.singleton_source,
            "customer_insertion_exact_candidate_limit": self.customer_top_k,
            "initial_merge_candidate_limit": self.initial_merge_candidate_limit,
            "initial_exact_insertion_limit": self.initial_exact_insertion_limit,
            "initial_construction_budget_s": self.initial_construction_budget_s,
            "station_repair": "progressive_multi_hop_full_replay_v1",
            "route_feasibility": "compact_cached_exact_v1",
            "simulation_cache_limit": self.simulation_cache_limit,
            "scalar_cache_limit": self.scalar_cache_limit,
            "station_cache_limit": self.station_cache_limit,
            "published_incumbent_validation": "solver_full_then_runner_independent_replay",
        }


    def load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Restore solver state from a previous checkpoint.
        """
        self.cur_iter = int(checkpoint.get("cur_iter", 1))

        # optional: allow checkpoint to overwrite max_iters if present
        if "max_iters" in checkpoint:
            self.max_iters = int(checkpoint["max_iters"])

        self.current_routes = [list(r) for r in checkpoint.get("current_routes", [])]
        self.best_routes = [list(r) for r in checkpoint.get("best_routes", [])]

        self.global_value = float(checkpoint.get("global_value", float("inf")))

        self.temperature = checkpoint.get("temperature", None)
        if self.temperature is not None:
            self.temperature = float(self.temperature)

        self.cr_weights = dict(checkpoint.get("cr_weights", self.cr_weights))
        self.ci_weights = dict(checkpoint.get("ci_weights", self.ci_weights))
        self.sr_weights = dict(checkpoint.get("sr_weights", self.sr_weights))
        self.si_weights = dict(checkpoint.get("si_weights", self.si_weights))

        self.cr_scores = dict(checkpoint.get("cr_scores", self.cr_scores))
        self.ci_scores = dict(checkpoint.get("ci_scores", self.ci_scores))
        self.sr_scores = dict(checkpoint.get("sr_scores", self.sr_scores))
        self.si_scores = dict(checkpoint.get("si_scores", self.si_scores))

        self.cr_uses = dict(checkpoint.get("cr_uses", self.cr_uses))
        self.ci_uses = dict(checkpoint.get("ci_uses", self.ci_uses))
        self.sr_uses = dict(checkpoint.get("sr_uses", self.sr_uses))
        self.si_uses = dict(checkpoint.get("si_uses", self.si_uses))

        self.attribute_frequency = defaultdict(
            int, checkpoint.get("attribute_frequency", {})
        )
        self.attribute_total = int(checkpoint.get("attribute_total", 0))

        self.visited = list(checkpoint.get("visited", self.visited))

    # ======================================================================
    # Public API
    # ======================================================================
    def solve(
        self,
        initial_routes: Optional[List[List[int]]] = None,
        delta_iters: Optional[int] = None,
        resume: bool = False,
        time_limit_s: Optional[float] = None,
        incumbent_callback: Optional[Callable[[float, float, List[List[int]]], None]] = None,
    ):
        """
        Three modes:
        1) fresh run:        resume=False, initial_routes=None
        2) warm start:       resume=False, initial_routes=...
        3) checkpoint resume: resume=True
        """

        solve_start = time.perf_counter()
        deadline = (
            None
            if time_limit_s is None
            else solve_start + max(0.0, float(time_limit_s))
        )
        self.terminated_by_time_limit = False

        def report_incumbent() -> None:
            if incumbent_callback is None or not np.isfinite(self.global_value):
                return
            incumbent_callback(
                time.perf_counter() - solve_start,
                float(self.global_value),
                [list(route) for route in self.best_routes],
            )

        # ------------------------------------------------------------
        # Build / restore starting search state
        # ------------------------------------------------------------
        if resume:
            if not self.current_routes:
                raise ValueError("resume=True but checkpoint has empty current_routes.")
            current = [list(r) for r in self.current_routes]
            initial_value = self.objective_value(current)

            if self.temperature is None:
                base_val = self.global_value if np.isfinite(self.global_value) else max(1.0, initial_value)
                self.temperature = self._initial_temperature(base_val)

        elif initial_routes is not None:
            current = [list(r) for r in initial_routes]
            current = self._postprocess_solution(current)
            initial_value = self.objective_value(current)

            if not self.is_solution_feasible(current):
                self.current_routes = [list(r) for r in current]
                self.best_routes = [list(r) for r in current]
                self.global_value = float("inf")
                self.visited = self._served_mask(current)
                return merge_routes(current) if self.format == "tensor" else self._export_routes(current)

            self.current_routes = [list(r) for r in current]
            self.best_routes = [list(r) for r in current]
            self.global_value = self.objective_value(current)
            self.temperature = self._initial_temperature(self.global_value)
            self.cur_iter = 1
            report_incumbent()

        else:
            # Publish the complete singleton fallback before spending any time
            # on consolidation.  It is an ordinary fully checked solution, not
            # a partial incumbent, so a large instance gets a valid checkpoint
            # route even if construction consumes the remaining time budget.
            singleton = self._construct_singleton_solution()
            if not self.is_solution_feasible(singleton):
                self.current_routes = [list(r) for r in singleton]
                self.best_routes = [list(r) for r in singleton]
                self.global_value = float("inf")
                self.visited = self._served_mask(singleton)
                return merge_routes(singleton) if self.format == "tensor" else self._export_routes(singleton)
            singleton_value = self.objective_value(singleton)
            self.current_routes = [list(r) for r in singleton]
            self.best_routes = [list(r) for r in singleton]
            self.global_value = singleton_value
            self.cur_iter = 1
            report_incumbent()

            current = self._construct_initial_solution(
                deadline=deadline,
                singleton_routes=singleton,
            )
            current = self._postprocess_solution(current)
            initial_value = self.objective_value(current)

            if not self.is_solution_feasible(current):
                current = [list(r) for r in singleton]
                initial_value = singleton_value

            self.current_routes = [list(r) for r in current]
            if initial_value + 1e-9 < self.global_value:
                self.best_routes = [list(r) for r in current]
                self.global_value = initial_value
                report_incumbent()
            self.temperature = self._initial_temperature(initial_value)
            self.cur_iter = 1

        # ------------------------------------------------------------
        # Decide how far to run this time
        # ------------------------------------------------------------
        start_iter = int(self.cur_iter)

        if delta_iters is None:
            target_iter = int(self.max_iters)
        else:
            target_iter = min(int(self.max_iters), start_iter + int(delta_iters) - 1)

        # already maxed out
        if start_iter > self.max_iters or start_iter > target_iter:
            self.visited = self._served_mask(self.best_routes)
            return merge_routes(self.best_routes) if self.format == "tensor" else self._export_routes(self.best_routes)

        current_distance = self.objective_value(current)
        route_removal_burst = 0

        for it in (range(start_iter, target_iter + 1)):
            if deadline is not None and time.perf_counter() >= deadline:
                self.terminated_by_time_limit = True
                break
            reward = self.r4
            accepted = False

            if it % self.NSR == 0:
                sr_name = self._roulette(self.sr_weights)
                si_name = self._roulette(self.si_weights)

                partial, _ = self.sr_ops[sr_name]([list(r) for r in current])
                candidate = self._repair_all_routes_with_si(partial, si_name)
                candidate = self._light_postprocess_solution(candidate)

                accepted, reward, cand_obj = self._evaluate_candidate(
                    candidate, current_distance
                )

                self.sr_uses[sr_name] += 1
                self.si_uses[si_name] += 1
                self.sr_scores[sr_name] += reward
                self.si_scores[si_name] += reward

            else:
                if it % self.NRR == 0:
                    route_removal_burst = self.nRR

                if route_removal_burst > 0:
                    cr_name = self.rng.choice(["random_route", "greedy_route"])
                    route_removal_burst -= 1
                else:
                    cr_candidates = {
                        k: w for k, w in self.cr_weights.items()
                        if k not in {"random_route", "greedy_route"}
                    }
                    cr_name = self._roulette(cr_candidates)

                ci_name = self._roulette(self.ci_weights)

                partial, removed_customers = self.cr_ops[cr_name]([list(r) for r in current])
                partial = self._repair_all_routes_with_si(partial, "gsi")
                candidate = self.ci_ops[ci_name](partial, removed_customers)
                candidate = self._light_postprocess_solution(candidate)

                accepted, reward, cand_obj = self._evaluate_candidate(
                    candidate, current_distance
                )

                self.cr_uses[cr_name] += 1
                self.ci_uses[ci_name] += 1
                self.cr_scores[cr_name] += reward
                self.ci_scores[ci_name] += reward

            if accepted:
                if cand_obj + 1e-9 < self.global_value:
                    candidate = self._postprocess_solution(candidate)
                    cand_obj = self.objective_value(candidate)

                current = [list(r) for r in candidate]
                current_distance = cand_obj
                self.update_diversification_history(current)

                if current_distance + 1e-9 < self.global_value:
                    self.global_value = current_distance
                    self.best_routes = [list(r) for r in current]
                    report_incumbent()

            self.temperature *= self.cooling_rate

            if it % self.NC == 0:
                self._update_weights(self.cr_weights, self.cr_scores, self.cr_uses)
                self._update_weights(self.ci_weights, self.ci_scores, self.ci_uses)

            if it % self.NS == 0:
                self._update_weights(self.sr_weights, self.sr_scores, self.sr_uses)
                self._update_weights(self.si_weights, self.si_scores, self.si_uses)

            self._maybe_clear_caches(it)

            # keep checkpoint state updated every iteration
            self.current_routes = [list(r) for r in current]
            self.cur_iter = it + 1

            if deadline is not None and time.perf_counter() >= deadline:
                self.terminated_by_time_limit = True
                break

        print(
            f"Initial value: {initial_value}, iter {start_iter}->{target_iter}, best value: {self.global_value}",
            flush=True
        )
        self.visited = self._served_mask(self.best_routes)
        return merge_routes(self.best_routes) if self.format == "tensor" else self._export_routes(self.best_routes)

    # ======================================================================
    # Objective / acceptance
    # ======================================================================
    def objective_value(self, routes: List[List[int]]) -> float:
        if not self.is_solution_feasible(routes):
            return float("inf")
        return sum(self._route_distance(r) for r in routes)

    def _evaluate_candidate(
        self,
        candidate: List[List[int]],
        current_distance: float,
    ) -> Tuple[bool, float, float]:
        if not self.is_solution_feasible(candidate):
            return False, self.r4, float("inf")

        # Feasibility was established immediately above.  Summing the route
        # distances directly preserves objective_value's reduction order while
        # avoiding a second full solution-feasibility scan.
        cand_dist = sum(self._route_distance(route) for route in candidate)

        if cand_dist + 1e-9 < self.global_value:
            return True, self.r1, cand_dist
        if cand_dist + 1e-9 < current_distance:
            return True, self.r2, cand_dist
        if self._accept_sa(cand_dist, current_distance):
            return True, self.r3, cand_dist
        return False, self.r4, cand_dist

    def _accept_sa(self, new_dist: float, old_dist: float) -> bool:
        if new_dist <= old_dist + 1e-9:
            return True
        delta = new_dist - old_dist
        prob = math.exp(-delta / max(1e-12, self.temperature))
        return self.rng.random() < prob

    def _initial_temperature(self, initial_distance: float) -> float:
        delta = max(1e-6, self.initial_temp_control * max(1.0, initial_distance))
        return -delta / math.log(0.5)

    # ======================================================================
    # Feasibility
    # ======================================================================
    def is_solution_feasible(self, routes: List[List[int]]) -> bool:
        seen = set()
        for route in routes:
            if not self.is_route_feasible(route):
                return False
            for x in route:
                if x in self.customer_to_mask:
                    if x in seen:
                        return False
                    seen.add(x)
        return len(seen) == len(self.customer_indices)

    def is_route_feasible(self, route: List[int]) -> bool:
        if not route or route[0] != 0 or route[-1] != 0:
            return False
        if any(node < 0 or node >= len(self.nodes) for node in route):
            return False
        if 0 in route[1:-1]:
            return False
        if not self._has_customer(route):
            return False
        # A route whose demand is exactly the vehicle capacity is feasible.
        # Use the same numerical tolerance as the insertion pre-filters; the
        # previous ``C - eps`` comparison incorrectly rejected full loads.
        if self._route_demand(route) > self.C + 1e-9:
            return False
        return self._is_route_schedule_feasible(route)

    def _is_route_schedule_feasible(self, route: List[int]) -> bool:
        """Compact route replay for hot feasibility-only evaluations."""

        key = self._route_key(route)
        if key in self._feasibility_cache:
            return self._feasibility_cache[key]
        detailed = self._sim_cache.get(key)
        if detailed is not None:
            feasible = bool(detailed["feasible"])
            self._cache_store(
                self._feasibility_cache, key, feasible, self.scalar_cache_limit
            )
            return feasible

        current_time = max(0.0, float(self.depot["ready"]))
        battery = self.Q
        for pos in range(1, len(route)):
            origin = route[pos - 1]
            destination = route[pos]
            current_time += float(self.time_matrix[origin, destination])
            battery -= float(self.energy_matrix[origin, destination])
            node = self.nodes[destination]
            start = max(current_time, float(node["ready"]))
            if battery < -1e-9 or start > float(node["due"]) + 1e-9:
                self._cache_store(
                    self._feasibility_cache, key, False, self.scalar_cache_limit
                )
                return False
            if destination in self.station_index_set:
                recharge_amount = self.Q - battery
                minutes_per_kwh = self.station_charge_minutes_per_kwh.get(
                    destination, self.g
                )
                current_time = start + recharge_amount * minutes_per_kwh
                battery = self.Q
            elif destination == 0:
                current_time = start
            else:
                current_time = start + float(node["service"])

        feasible = current_time <= float(self.depot["due"]) + 1e-9
        self._cache_store(
            self._feasibility_cache, key, feasible, self.scalar_cache_limit
        )
        return feasible

    def _simulate_route(self, route: List[int]) -> Dict[str, Any]:
        key = self._route_key(route)
        cached = self._sim_cache.get(key)
        if cached is not None:
            return cached

        n = len(route)
        distance = 0.0

        arrival_times = [0.0] * n
        service_starts = [0.0] * n
        departure_times = [0.0] * n

        arrival_battery = [0.0] * n
        departure_battery = [0.0] * n

        time = max(0.0, float(self.depot["ready"]))
        battery = self.Q
        depot_due = float(self.depot["due"])

        arrival_times[0] = time
        service_starts[0] = time
        departure_times[0] = time
        arrival_battery[0] = battery
        departure_battery[0] = battery

        first_negative_customer_pos = None
        first_energy_violation_pos = None
        first_time_violation_pos = None
        first_infeasible_pos = None

        for pos in range(1, n):
            i = route[pos - 1]
            j = route[pos]

            dist_ij = float(self.dist_matrix[i, j])
            travel_time = self._travel_time(i, j)
            energy_ij = self._energy(i, j)

            distance += dist_ij
            time += travel_time
            battery -= energy_ij

            arrival_times[pos] = time
            arrival_battery[pos] = battery

            node = self.nodes[j]
            ready = float(node["ready"])
            due = float(node["due"])
            service = float(node["service"])

            start = max(time, ready)
            service_starts[pos] = start

            if battery < -1e-9:
                if first_energy_violation_pos is None:
                    first_energy_violation_pos = pos
                    first_infeasible_pos = pos
                if j in self.customer_to_mask and first_negative_customer_pos is None:
                    first_negative_customer_pos = pos

            if start > due + 1e-9:
                if first_time_violation_pos is None:
                    first_time_violation_pos = pos
                if first_infeasible_pos is None:
                    first_infeasible_pos = pos

            if battery < -1e-9 or start > due + 1e-9:
                result = {
                    "feasible": False,
                    "distance": distance,
                    "arrival_times": arrival_times,
                    "service_starts": service_starts,
                    "departure_times": departure_times,
                    "arrival_battery": arrival_battery,
                    "departure_battery": departure_battery,
                    "first_negative_customer_pos": first_negative_customer_pos,
                    "first_energy_violation_pos": first_energy_violation_pos,
                    "first_time_violation_pos": first_time_violation_pos,
                    "first_infeasible_pos": first_infeasible_pos,
                }
                self._cache_store(
                    self._sim_cache, key, result, self.simulation_cache_limit
                )
                self._cache_store(
                    self._feasibility_cache, key, False, self.scalar_cache_limit
                )
                return result

            if j in self.station_index_set:
                recharge_amount = self.Q - battery
                minutes_per_kwh = self.station_charge_minutes_per_kwh.get(j, self.g)
                time = start + recharge_amount * minutes_per_kwh
                battery = self.Q
            elif j == 0:
                time = start
            else:
                time = start + service

            departure_times[pos] = time
            departure_battery[pos] = battery

        feasible = time <= depot_due + 1e-9
        if not feasible and first_infeasible_pos is None:
            first_infeasible_pos = n - 1
            first_time_violation_pos = n - 1

        result = {
            "feasible": feasible,
            "distance": distance,
            "arrival_times": arrival_times,
            "service_starts": service_starts,
            "departure_times": departure_times,
            "arrival_battery": arrival_battery,
            "departure_battery": departure_battery,
            "first_negative_customer_pos": first_negative_customer_pos,
            "first_energy_violation_pos": first_energy_violation_pos,
            "first_time_violation_pos": first_time_violation_pos,
            "first_infeasible_pos": first_infeasible_pos,
        }
        self._cache_store(
            self._sim_cache, key, result, self.simulation_cache_limit
        )
        self._cache_store(
            self._feasibility_cache, key, feasible, self.scalar_cache_limit
        )
        return result

    # ======================================================================
    # Initial solution
    # ======================================================================
    def _construct_singleton_solution(self) -> List[List[int]]:
        """Build a complete feasible fallback before attempting consolidation."""

        certified = self._normalized_certificate_singletons()
        if certified is not None:
            self.singleton_source = "stage2_certificate_replayed"
            self._singleton_route_cache = {
                customer: list(route)
                for customer, route in zip(self.customer_indices, certified)
            }
            return certified

        self.singleton_source = "solver_repair"
        routes: List[List[int]] = []
        for customer in self.customer_indices:
            route = self._make_single_customer_route(customer)
            if route is None:
                # Keep failure explicit.  The caller's full feasibility check
                # prevents a partial solution from ever becoming an incumbent.
                return []
            routes.append(route)
        return routes

    def _normalized_certificate_singletons(self) -> Optional[List[List[int]]]:
        raw = self.certificate_singleton_routes
        if raw is None:
            return None
        by_customer: Dict[int, List[int]] = {}
        try:
            for raw_route in raw:
                route = [int(node) for node in raw_route]
                customers = [node for node in route if node in self.customer_to_mask]
                if len(customers) != 1 or customers[0] in by_customer:
                    return None
                by_customer[customers[0]] = route
        except (TypeError, ValueError):
            return None
        if set(by_customer) != set(self.customer_indices):
            return None
        ordered = [by_customer[customer] for customer in self.customer_indices]
        # The shared loader has already canonical-replayed these routes, but the
        # solver validates again after adaptation before trusting the fallback.
        return ordered if self.is_solution_feasible(ordered) else None

    def _construct_initial_solution(
        self,
        deadline: Optional[float] = None,
        singleton_routes: Optional[List[List[int]]] = None,
    ) -> List[List[int]]:
        construction_start = time.perf_counter()
        construction_deadline = construction_start + self.initial_construction_budget_s
        if deadline is not None:
            construction_deadline = min(construction_deadline, deadline)

        routes = (
            self._construct_singleton_solution()
            if singleton_routes is None
            else [list(route) for route in singleton_routes]
        )
        if not routes:
            self.initial_construction_stats = {
                "strategy": self.initial_construction_strategy,
                "singleton_fallback_feasible": False,
                "singleton_source": self.singleton_source,
                "elapsed_s": time.perf_counter() - construction_start,
            }
            return []

        # Best-fit decreasing is deterministic and starts from a complete
        # feasible incumbent.  Customers with less time-window slack are placed
        # first; ties retain canonical customer order.
        customer_order = sorted(
            self.customer_indices,
            key=lambda customer: (
                float(self.nodes[customer]["due"])
                - float(self.nodes[customer]["ready"]),
                customer,
            ),
        )
        merged_routes: List[List[int]] = []
        inserted = 0
        budget_exhausted = False
        for order_position, customer in enumerate(customer_order):
            if time.perf_counter() >= construction_deadline:
                budget_exhausted = True
                remaining = customer_order[order_position:]
                merged_routes.extend(routes[customer - 1] for customer in remaining)
                break

            route_ids = list(range(len(merged_routes)))
            candidate_limit = self.initial_merge_candidate_limit
            if candidate_limit is not None and len(route_ids) > candidate_limit:
                # Rank only which routes receive exact insertion evaluation.
                # Every selected candidate is still fully repaired and checked.
                route_ids.sort(
                    key=lambda ridx: min(
                        float(self.dist_matrix[node, customer])
                        for node in merged_routes[ridx]
                        if node in self.customer_to_mask or node == 0
                    )
                )
                route_ids = route_ids[:candidate_limit]

            best = self._best_bounded_initial_insertion(
                merged_routes,
                customer,
                route_ids,
                construction_deadline,
            )
            if time.perf_counter() >= construction_deadline:
                budget_exhausted = True

            if best is None:
                merged_routes.append(routes[customer - 1])
            else:
                ridx, new_route, _ = best
                merged_routes[ridx] = new_route
                inserted += 1

            if budget_exhausted:
                remaining = customer_order[order_position + 1 :]
                merged_routes.extend(routes[item - 1] for item in remaining)
                break

        else:
            budget_exhausted = False

        result = self._cleanup_solution(merged_routes)
        self.initial_construction_stats = {
            "strategy": self.initial_construction_strategy,
            "singleton_fallback_feasible": True,
            "singleton_source": self.singleton_source,
            "singleton_route_count": len(routes),
            "result_route_count": len(result),
            "merged_customer_count": inserted,
            "budget_s": self.initial_construction_budget_s,
            "budget_exhausted": budget_exhausted,
            "elapsed_s": time.perf_counter() - construction_start,
        }
        return result

    def _best_bounded_initial_insertion(
        self,
        routes: List[List[int]],
        customer: int,
        route_ids: List[int],
        deadline: float,
    ) -> Optional[Tuple[int, List[int], float]]:
        """Evaluate a bounded set of promising positions with exact checks.

        The local distance delta is only a ranking proxy.  Returned routes have
        gone through the normal station repair and full route-feasibility test.
        """

        customer_demand = float(self.nodes[customer]["demand"])
        cheap: List[Tuple[float, int, int, float]] = []
        for ridx in route_ids:
            route = routes[ridx]
            route_demand = self._route_demand(route)
            if route_demand + customer_demand > self.C + 1e-9:
                continue
            route_sim = self._simulate_route(route)
            base_distance = self._route_distance(route)
            for pos in range(1, len(route)):
                if not self._quick_customer_insert_filter(
                    route,
                    customer,
                    pos,
                    route_demand=route_demand,
                    route_sim=route_sim,
                ):
                    continue
                previous = route[pos - 1]
                following = route[pos]
                proxy = float(
                    self.dist_matrix[previous, customer]
                    + self.dist_matrix[customer, following]
                    - self.dist_matrix[previous, following]
                )
                cheap.append((proxy, ridx, pos, base_distance))

        # Python's stable sort preserves route/position enumeration for ties.
        cheap.sort(key=lambda item: item[0])
        best: Optional[Tuple[int, List[int], float]] = None
        for _, ridx, pos, base_distance in cheap[: self.initial_exact_insertion_limit]:
            if time.perf_counter() >= deadline:
                break
            route = routes[ridx]
            trial = route[:pos] + [customer] + route[pos:]
            # Construction must obey a hard latency envelope.  Reusing the
            # route's existing charging visits and accepting only a directly
            # feasible insertion is cheap and exact; potentially expensive
            # station repair remains available to the main ALNS search.
            if not self.is_route_feasible(trial):
                continue
            delta = self._route_distance(trial) - base_distance
            if best is None or delta < best[2]:
                best = (ridx, trial, delta)
        return best

    # ======================================================================
    # Customer Removal (CR)
    # ======================================================================
    def _cr_random_customer(self, routes: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
        customers = self._list_customers(routes)
        if not customers:
            return routes, []
        q = min(len(customers), self._num_customers_to_remove())
        removed = self.rng.sample(customers, q)
        mode = self._random_customer_removal_mode()
        return self._remove_customers_with_mode(routes, removed, mode), removed

    def _cr_worst_distance(self, routes: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
        scored = []
        for route in routes:
            for pos in range(1, len(route) - 1):
                u = route[pos]
                if u not in self.customer_to_mask:
                    continue
                cost = self.dist_matrix[route[pos - 1], u] + self.dist_matrix[u, route[pos + 1]]
                scored.append((cost, u))

        if not scored:
            return routes, []

        scored.sort(reverse=True)
        q = min(len(scored), self._num_customers_to_remove())
        removed = self._select_ranked_with_noise(scored, q, self.worst_determinism)
        mode = self._random_customer_removal_mode()
        return self._remove_customers_with_mode(routes, removed, mode), removed

    def _cr_worst_time(self, routes: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
        scored = []
        for route in routes:
            sim = self._simulate_route(route)
            starts = sim["service_starts"]
            for pos in range(1, len(route) - 1):
                u = route[pos]
                if u not in self.customer_to_mask:
                    continue
                e_u = float(self.nodes[u]["ready"])
                cost = abs(starts[pos] - e_u)
                scored.append((cost, u))

        if not scored:
            return routes, []

        scored.sort(reverse=True)
        q = min(len(scored), self._num_customers_to_remove())
        removed = self._select_ranked_with_noise(scored, q, self.worst_determinism)
        mode = self._random_customer_removal_mode()
        return self._remove_customers_with_mode(routes, removed, mode), removed

    def _cr_shaw(self, routes: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
        return self._cr_shaw_family(routes, family="shaw")

    def _cr_proximity(self, routes: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
        return self._cr_shaw_family(routes, family="proximity")

    def _cr_demand_based(self, routes: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
        return self._cr_shaw_family(routes, family="demand")

    def _cr_time_based(self, routes: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
        return self._cr_shaw_family(routes, family="time")

    def _cr_shaw_family(self, routes: List[List[int]], family: str) -> Tuple[List[List[int]], List[int]]:
        customers = self._list_customers(routes)
        if not customers:
            return routes, []

        # _route_of_customer returns the first matching route.  setdefault
        # deliberately preserves that behaviour even for malformed solutions
        # containing a duplicate customer, while replacing repeated scans of
        # the complete solution with one linear pass.
        first_route_of_customer: Dict[int, int] = {}
        for ridx, route in enumerate(routes):
            for node in route:
                if node in self.customer_to_mask:
                    first_route_of_customer.setdefault(node, ridx)

        q = min(len(customers), self._num_customers_to_remove())
        removed = []

        seed = self.rng.choice(customers)
        removed.append(seed)

        while len(removed) < q:
            anchor = self.rng.choice(removed)
            anchor_route = first_route_of_customer.get(anchor)
            related = []

            for u in customers:
                if u in removed:
                    continue

                node_a = self.nodes[anchor]
                node_u = self.nodes[u]

                same_route_term = (
                    -1.0 if first_route_of_customer.get(u) == anchor_route else 1.0
                )
                shaw_score = (
                    self.phi1 * self.dist_matrix[anchor, u]
                    + self.phi2 * abs(float(node_a["ready"]) - float(node_u["ready"]))
                    + self.phi3 * same_route_term
                    + self.phi4 * abs(float(node_a["demand"]) - float(node_u["demand"]))
                )

                if family == "proximity":
                    score = self.dist_matrix[anchor, u]
                elif family == "demand":
                    score = abs(float(node_a["demand"]) - float(node_u["demand"]))
                elif family == "time":
                    score = abs(float(node_a["ready"]) - float(node_u["ready"]))
                else:
                    score = shaw_score

                related.append((score, u))

            if not related:
                break

            related.sort(key=lambda x: x[0])
            chosen = self._select_one_ranked_with_noise(related, self.shaw_deteminism)
            removed.append(chosen)

        mode = self._random_customer_removal_mode()
        return self._remove_customers_with_mode(routes, removed, mode), removed

    def _cr_zone_removal(self, routes: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
        customers = self._list_customers(routes)
        if not customers:
            return routes, []

        zone_map = self.customer_zone_map
        all_zones = sorted(set(zone_map.values()))
        if not all_zones:
            return self._cr_random_customer(routes)

        chosen_zone = self.rng.choice(all_zones)
        zone_customers = [c for c in customers if zone_map.get(c) == chosen_zone]
        if not zone_customers:
            return self._cr_random_customer(routes)

        q = min(len(zone_customers), self._num_customers_to_remove())
        removed = zone_customers[:q]
        mode = self._random_customer_removal_mode()
        return self._remove_customers_with_mode(routes, removed, mode), removed

    def _cr_random_route(self, routes: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
        if not routes:
            return routes, []

        n_routes = len(routes)
        low = 1
        high = max(1, int(math.ceil(self.route_removal_upper_ratio * n_routes)))
        x = self.rng.randint(low, high)

        chosen = set(self.rng.sample(range(n_routes), min(x, n_routes)))
        removed = []
        kept = []

        for ridx, route in enumerate(routes):
            if ridx in chosen:
                removed.extend([u for u in route if u in self.customer_to_mask])
            else:
                kept.append(route)

        return kept, removed

    def _cr_greedy_route(self, routes: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
        if not routes:
            return routes, []

        n_routes = len(routes)
        low = 1
        high = max(1, int(math.ceil(self.route_removal_upper_ratio * n_routes)))
        x = self.rng.randint(low, high)

        order = sorted(
            range(n_routes),
            key=lambda ridx: sum(1 for u in routes[ridx] if u in self.customer_to_mask),
        )
        chosen = set(order[: min(x, n_routes)])

        removed = []
        kept = []

        for ridx, route in enumerate(routes):
            if ridx in chosen:
                removed.extend([u for u in route if u in self.customer_to_mask])
            else:
                kept.append(route)

        return kept, removed

    # ======================================================================
    # Station Removal (SR)
    # ======================================================================
    def _sr_random_station(self, routes: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
        positions = self._all_station_positions(routes)
        if not positions:
            return routes, []

        q = max(1, min(len(positions), len(positions) // 3 if len(positions) >= 3 else 1))
        chosen = sorted(self.rng.sample(positions, q), reverse=True)

        for ridx, pos in chosen:
            del routes[ridx][pos]

        return self._cleanup_solution(routes), []

    def _sr_worst_distance_station(self, routes: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
        scored = []
        for ridx, route in enumerate(routes):
            for pos in range(1, len(route) - 1):
                s = route[pos]
                if s not in self.station_index_set:
                    continue
                detour = (
                    self.dist_matrix[route[pos - 1], s]
                    + self.dist_matrix[s, route[pos + 1]]
                    - self.dist_matrix[route[pos - 1], route[pos + 1]]
                )
                scored.append((detour, ridx, pos))

        if not scored:
            return routes, []

        scored.sort(reverse=True)
        q = max(1, min(len(scored), len(scored) // 3 if len(scored) >= 3 else 1))
        chosen = [(ridx, pos) for _, ridx, pos in scored[:q]]
        chosen = sorted(chosen, reverse=True)

        for ridx, pos in chosen:
            del routes[ridx][pos]

        return self._cleanup_solution(routes), []

    def _sr_worst_charge_usage_station(self, routes: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
        scored = []
        for ridx, route in enumerate(routes):
            sim = self._simulate_route(route)
            arrival_battery = sim["arrival_battery"]
            for pos in range(1, len(route) - 1):
                s = route[pos]
                if s not in self.station_index_set:
                    continue
                scored.append((arrival_battery[pos], ridx, pos))

        if not scored:
            return routes, []

        scored.sort(reverse=True)
        q = max(1, min(len(scored), len(scored) // 3 if len(scored) >= 3 else 1))
        chosen = [(ridx, pos) for _, ridx, pos in scored[:q]]
        chosen = sorted(chosen, reverse=True)

        for ridx, pos in chosen:
            del routes[ridx][pos]

        return self._cleanup_solution(routes), []

    def _sr_full_charge_station(self, routes: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
        return self._sr_random_station(routes)

    # ======================================================================
    # Customer Insertion (CI)
    # ======================================================================
    def _ci_greedy(self, routes: List[List[int]], removed_customers: List[int]) -> List[List[int]]:
        unrouted = list(dict.fromkeys(removed_customers))

        while unrouted:
            best = None
            best_delta = float("inf")

            for c in unrouted:
                candidate = self._best_customer_insertion(routes, c, mode="distance")
                if candidate is None:
                    continue
                ridx, new_route, delta = candidate
                if delta < best_delta:
                    best_delta = delta
                    best = (c, ridx, new_route)

            if best is None:
                break

            c, ridx, new_route = best
            if ridx == len(routes):
                routes.append(new_route)
            else:
                routes[ridx] = new_route
            unrouted.remove(c)

        return self._cleanup_solution(routes)

    def _ci_regret2(self, routes: List[List[int]], removed_customers: List[int]) -> List[List[int]]:
        return self._ci_regretk(routes, removed_customers, k=2)

    def _ci_regret3(self, routes: List[List[int]], removed_customers: List[int]) -> List[List[int]]:
        return self._ci_regretk(routes, removed_customers, k=3)

    def _ci_regretk(self, routes: List[List[int]], removed_customers: List[int], k: int) -> List[List[int]]:
        unrouted = list(dict.fromkeys(removed_customers))

        while unrouted:
            best_choice = None
            best_regret = -float("inf")
            best_primary = float("inf")

            for c in unrouted:
                options = self._all_customer_insertions(routes, c, mode="distance")
                if not options:
                    continue
                options.sort(key=lambda x: x[2])
                primary = options[0][2]

                regret = 0.0
                for i in range(1, min(k, len(options))):
                    regret += options[i][2] - primary
                if len(options) < k:
                    regret += (k - len(options)) * max(1.0, self.max_distance)

                if regret > best_regret or (math.isclose(regret, best_regret) and primary < best_primary):
                    best_regret = regret
                    best_primary = primary
                    best_choice = (c, options[0][0], options[0][1])

            if best_choice is None:
                break

            c, ridx, new_route = best_choice
            if ridx == len(routes):
                routes.append(new_route)
            else:
                routes[ridx] = new_route
            unrouted.remove(c)

        return self._cleanup_solution(routes)

    def _ci_time_based(self, routes: List[List[int]], removed_customers: List[int]) -> List[List[int]]:
        unrouted = list(dict.fromkeys(removed_customers))

        while unrouted:
            best = None
            best_delta = float("inf")

            for c in unrouted:
                candidate = self._best_customer_insertion(routes, c, mode="time")
                if candidate is None:
                    continue
                ridx, new_route, delta = candidate
                if delta < best_delta:
                    best_delta = delta
                    best = (c, ridx, new_route)

            if best is None:
                break

            c, ridx, new_route = best
            if ridx == len(routes):
                routes.append(new_route)
            else:
                routes[ridx] = new_route
            unrouted.remove(c)

        return self._cleanup_solution(routes)

    def _ci_zone_insertion(self, routes: List[List[int]], removed_customers: List[int]) -> List[List[int]]:
        zone_map = self.customer_zone_map
        all_zones = sorted(set(zone_map.values()))
        if not all_zones:
            return self._ci_time_based(routes, removed_customers)

        chosen_zone = self.rng.choice(all_zones)
        route_candidates = [
            ridx
            for ridx, route in enumerate(routes)
            if any((u in self.customer_to_mask and zone_map.get(u) == chosen_zone) for u in route)
        ]
        if not route_candidates:
            return self._ci_time_based(routes, removed_customers)

        unrouted = list(dict.fromkeys(removed_customers))

        while unrouted:
            best = None
            best_delta = float("inf")

            for c in unrouted:
                options = self._all_customer_insertions(
                    routes,
                    c,
                    mode="time",
                    allowed_routes=set(route_candidates),
                )
                if not options:
                    options = self._all_customer_insertions(routes, c, mode="time")
                if not options:
                    continue
                ridx, new_route, delta = min(options, key=lambda x: x[2])
                if delta < best_delta:
                    best_delta = delta
                    best = (c, ridx, new_route)

            if best is None:
                break

            c, ridx, new_route = best
            if ridx == len(routes):
                routes.append(new_route)
            else:
                routes[ridx] = new_route
            unrouted.remove(c)

        return self._cleanup_solution(routes)

    # ======================================================================
    # Station Insertion (SI)
    # ======================================================================
    def _si_greedy_station_insertion(self, route: List[int]) -> Optional[List[int]]:
        return self._station_insertion_core(route, strategy="gsi")

    def _si_greedy_station_insertion_with_comparison(self, route: List[int]) -> Optional[List[int]]:
        return self._station_insertion_core(route, strategy="gsi_comparison")

    def _si_best_station_insertion(self, route: List[int]) -> Optional[List[int]]:
        return self._station_insertion_core(route, strategy="best")

    # ======================================================================
    # Core route repair
    # ======================================================================
    def _try_insert_customer_with_si(self, trial_route: List[int], si_name: str) -> Optional[List[int]]:
        if self._route_demand(trial_route) > self.C + 1e-9:
            return None

        repaired = self._repair_route_with_si(trial_route, si_name)
        return repaired

    def _repair_route_with_si(self, route: List[int], si_name: str) -> Optional[List[int]]:
        route = self._cleanup_route(route)

        for _ in range(20):
            sim = self._simulate_route(route)
            if sim["feasible"]:
                route = self._prune_redundant_stations(route)
                return route if self.is_route_feasible(route) else None

            fail_pos = sim["first_energy_violation_pos"]
            if fail_pos is None:
                return None

            repaired = self.si_ops[si_name](route)
            if repaired is None or repaired == route:
                return None

            route = self._cleanup_route(repaired)

        return None

    def _repair_all_routes_with_si(self, routes: List[List[int]], si_name: str) -> List[List[int]]:
        out = []
        for route in routes:
            route = self._cleanup_route(route)
            if not self._has_customer(route):
                continue

            repaired = self._repair_route_with_si(route, si_name)
            if repaired is None:
                out.append(route)
            else:
                out.append(repaired)
        return self._cleanup_solution(out)

    def _station_insertion_core(self, route: List[int], strategy: str) -> Optional[List[int]]:
        route = self._cleanup_route(route)
        sim = self._simulate_route(route)

        if sim["feasible"]:
            return route

        fail_pos = sim["first_energy_violation_pos"]
        if fail_pos is None:
            return None

        # Include the depot's outgoing arc.  Excluding arc zero prevented repair
        # of customers that require charging before they are first reached.
        candidate_positions = list(range(fail_pos - 1, -1, -1))

        if strategy == "gsi":
            for arc_pos in candidate_positions:
                repaired = self._best_station_on_arc(route, arc_pos)
                if repaired is not None:
                    return repaired
            return None

        if strategy == "gsi_comparison":
            if fail_pos - 1 >= 0:
                cand1 = self._best_station_on_arc(route, fail_pos - 1)
                cand2 = self._best_station_on_arc(route, fail_pos - 2) if fail_pos - 2 >= 0 else None

                candidates = []
                if cand1 is not None:
                    candidates.append(cand1)
                if cand2 is not None:
                    candidates.append(cand2)

                if candidates:
                    candidates.sort(key=lambda r: self._route_distance(r))
                    return candidates[0]

            for arc_pos in candidate_positions:
                repaired = self._best_station_on_arc(route, arc_pos)
                if repaired is not None:
                    return repaired
            return None

        feasible_candidates = []
        for arc_pos in candidate_positions:
            repaired = self._best_station_on_arc(route, arc_pos)
            if repaired is not None:
                feasible_candidates.append(repaired)

        if not feasible_candidates:
            return None

        feasible_candidates.sort(key=lambda r: self._route_distance(r))
        return feasible_candidates[0]

    def _best_station_on_arc(self, route: List[int], arc_pos: int) -> Optional[List[int]]:
        route_key = self._route_key(route)
        cache_key = (route_key, arc_pos)

        if cache_key in self._best_station_arc_cache:
            cached = self._best_station_arc_cache[cache_key]
            return None if cached is None else list(cached)

        i = route[arc_pos]
        j = route[arc_pos + 1]

        best_route = None
        best_delta = float("inf")
        base_dist = self._route_distance(route)
        base_arc = float(self.dist_matrix[i, j])
        base_sim = self._simulate_route(route)
        base_fail_pos = base_sim["first_energy_violation_pos"]
        if base_fail_pos is None:
            self._cache_store(
                self._best_station_arc_cache,
                cache_key,
                None,
                self.station_cache_limit,
            )
            return None

        for s in self.station_indices:
            if s == i or s == j:
                continue

            if self._energy(i, s) > self.Q + 1e-9:
                continue

            detour = float(self.dist_matrix[i, s] + self.dist_matrix[s, j] - base_arc)
            if detour >= best_delta:
                continue

            trial = route[: arc_pos + 1] + [s] + route[arc_pos + 1 :]
            trial_sim = self._simulate_route(trial)
            if not trial_sim["feasible"]:
                trial_fail_pos = trial_sim["first_energy_violation_pos"]
                inserted_pos = arc_pos + 1
                # A progressive station must itself be reachable and must not
                # move the first failing original node earlier.  Equality after
                # accounting for the inserted node is useful: the next repair
                # iteration may add another CS on the remaining long arc.
                if (
                    trial_fail_pos is None
                    or trial_fail_pos <= inserted_pos
                    or trial_fail_pos < base_fail_pos + 1
                    or trial_sim["first_time_violation_pos"] is not None
                ):
                    continue

            delta = self._route_distance(trial) - base_dist
            if delta < best_delta:
                best_delta = delta
                best_route = trial

        self._cache_store(
            self._best_station_arc_cache,
            cache_key,
            None if best_route is None else list(best_route),
            self.station_cache_limit,
        )
        return None if best_route is None else list(best_route)

    # ======================================================================
    # Customer insertion helpers
    # ======================================================================
    def _quick_customer_insert_filter(
        self,
        route: List[int],
        customer: int,
        pos: int,
        *,
        route_demand: Optional[float] = None,
        route_sim: Optional[Dict[str, Any]] = None,
    ) -> bool:
        demand = self._route_demand(route) if route_demand is None else route_demand
        if demand + float(self.nodes[customer]["demand"]) > self.C + 1e-9:
            return False

        sim = self._simulate_route(route) if route_sim is None else route_sim
        prev_node = route[pos - 1]

        depart_prev = sim["departure_times"][pos - 1]
        earliest_arr_c = depart_prev + self._travel_time(prev_node, customer)

        if earliest_arr_c > float(self.nodes[customer]["due"]) + 1e-9:
            return False

        if prev_node in self.station_index_set:
            prev_battery = self.Q
        else:
            prev_battery = sim["departure_battery"][pos - 1]

        if prev_battery + 1e-9 < self._energy(prev_node, customer):
            return False

        return True

    def _all_customer_insertions_part(
        self,
        routes: List[List[int]],
        customer: int,
        mode: str = "distance",
        allowed_routes: Optional[set] = None,
        include_new_route: bool = True,
    ) -> List[Tuple[int, List[int], float]]:
        options = []
        cheap_candidates = []

        route_ids = range(len(routes))
        if allowed_routes is not None:
            route_ids = [ridx for ridx in range(len(routes)) if ridx in allowed_routes]

        customer_demand = float(self.nodes[customer]["demand"])

        for ridx in route_ids:
            route = routes[ridx]

            # 容量先过滤
            route_demand = self._route_demand(route)
            if route_demand + customer_demand > self.C + 1e-9:
                continue

            route_sim = self._simulate_route(route)
            if mode == "time":
                base_dist = None
                base_time = self._route_total_time(route)
            else:
                base_dist = self._route_distance(route)
                base_time = None

            for pos in range(1, len(route)):
                if not self._quick_customer_insert_filter(
                    route,
                    customer,
                    pos,
                    route_demand=route_demand,
                    route_sim=route_sim,
                ):
                    continue

                prev_node = route[pos - 1]
                next_node = route[pos]

                # cheap proxy: 先不用repair，只看局部增量
                dist_delta = (
                    self.dist_matrix[prev_node, customer]
                    + self.dist_matrix[customer, next_node]
                    - self.dist_matrix[prev_node, next_node]
                )

                if mode == "time":
                    proxy = (
                        self.time_matrix[prev_node, customer]
                        + self.time_matrix[customer, next_node]
                        - self.time_matrix[prev_node, next_node]
                    )
                else:
                    proxy = dist_delta

                cheap_candidates.append((proxy, ridx, pos, base_dist, base_time))

        # 只保留 top-k 候选
        cheap_candidates.sort(key=lambda x: x[0])

        if self.customer_top_k is not None and self.customer_top_k > 0:
            cheap_candidates = cheap_candidates[: self.customer_top_k]

        for _, ridx, pos, base_dist, base_time in cheap_candidates:
            route = routes[ridx]
            trial = route[:pos] + [customer] + route[pos:]

            repaired = self._try_insert_customer_with_si(trial, "gsi")
            if repaired is None:
                continue

            delta = (
                self._route_total_time(repaired) - base_time
                if mode == "time"
                else self._route_distance(repaired) - base_dist
            )
            options.append((ridx, repaired, delta))

        # 单独保留 new single-customer route
        if include_new_route:
            new_route = self._make_single_customer_route(customer)
            if new_route is not None:
                delta = (
                    self._route_total_time(new_route)
                    if mode == "time"
                    else self._route_distance(new_route)
                )
                options.append((len(routes), new_route, delta))

        return options

    def _all_customer_insertions(
        self,
        routes: List[List[int]],
        customer: int,
        mode: str = "distance",
        allowed_routes: Optional[set] = None,
        include_new_route: bool = True,
    ) -> List[Tuple[int, List[int], float]]:
        if self.customer_top_k is not None:
            return self._all_customer_insertions_part(
                routes,
                customer,
                mode=mode,
                allowed_routes=allowed_routes,
                include_new_route=include_new_route,
            )
        options = []

        route_ids = range(len(routes))
        if allowed_routes is not None:
            route_ids = [ridx for ridx in range(len(routes)) if ridx in allowed_routes]

        customer_demand = float(self.nodes[customer]["demand"])

        for ridx in route_ids:
            route = routes[ridx]
            route_demand = self._route_demand(route)
            if route_demand + customer_demand > self.C + 1e-9:
                continue

            route_sim = self._simulate_route(route)
            if mode == "time":
                base_dist = None
                base_time = self._route_total_time(route)
            else:
                base_dist = self._route_distance(route)
                base_time = None

            for pos in range(1, len(route)):
                if not self._quick_customer_insert_filter(
                    route,
                    customer,
                    pos,
                    route_demand=route_demand,
                    route_sim=route_sim,
                ):
                    continue

                trial = route[:pos] + [customer] + route[pos:]
                repaired = self._try_insert_customer_with_si(trial, "gsi")
                if repaired is None:
                    continue

                delta = (
                    self._route_total_time(repaired) - base_time
                    if mode == "time"
                    else self._route_distance(repaired) - base_dist
                )
                options.append((ridx, repaired, delta))

        if include_new_route:
            new_route = self._make_single_customer_route(customer)
            if new_route is not None:
                delta = self._route_total_time(new_route) if mode == "time" else self._route_distance(new_route)
                options.append((len(routes), new_route, delta))

        return options

    def _best_customer_insertion(
        self,
        routes: List[List[int]],
        customer: int,
        mode: str = "distance",
    ) -> Optional[Tuple[int, List[int], float]]:
        options = self._all_customer_insertions(routes, customer, mode=mode)
        if not options:
            return None
        return min(options, key=lambda x: x[2])

    # ======================================================================
    # Customer removal modes
    # ======================================================================
    def _random_customer_removal_mode(self) -> str:
        return self.rng.choice(["customer_only", "with_preceding_station", "with_succeeding_station"])

    def _remove_customers_with_mode(
        self,
        routes: List[List[int]],
        removed_customers: List[int],
        mode: str,
    ) -> List[List[int]]:
        removed_set = set(removed_customers)
        out = []

        for route in routes:
            route = list(route)
            pos = 1
            while pos < len(route) - 1:
                u = route[pos]
                if u in removed_set:
                    if mode == "with_preceding_station" and pos - 1 >= 1 and route[pos - 1] in self.station_index_set:
                        del route[pos - 1]
                        pos -= 1

                    del route[pos]

                    if mode == "with_succeeding_station" and pos < len(route) - 1 and route[pos] in self.station_index_set:
                        del route[pos]
                    continue
                pos += 1

            route = self._cleanup_route(route)
            route = self._prune_redundant_stations(route)
            if self._has_customer(route):
                out.append(route)

        return self._cleanup_solution(out)

    # ======================================================================
    # Lightweight / heavyweight postprocess
    # ======================================================================
    def _light_postprocess_solution(self, routes: List[List[int]]) -> List[List[int]]:
        routes = self._cleanup_solution(routes)

        seen = set()
        out = []

        for route in routes:
            filtered = [0]
            for u in route[1:-1]:
                if u in self.customer_to_mask:
                    if u in seen:
                        continue
                    seen.add(u)
                filtered.append(u)
            filtered.append(0)

            filtered = self._cleanup_route(filtered)
            if self._has_customer(filtered):
                out.append(filtered)

        return self._cleanup_solution(out)

    def _postprocess_solution(self, routes: List[List[int]]) -> List[List[int]]:
        routes = [list(r) for r in routes]
        routes = self._cleanup_solution(routes)

        seen = set()
        dedup = []

        for route in routes:
            filtered = [0]
            for u in route[1:-1]:
                if u in self.customer_to_mask:
                    if u in seen:
                        continue
                    seen.add(u)
                filtered.append(u)
            filtered.append(0)

            filtered = self._cleanup_route(filtered)
            filtered = self._prune_redundant_stations(filtered)
            if self._has_customer(filtered):
                dedup.append(filtered)

        routes = dedup

        missing = [c for c in self.customer_indices if c not in seen]
        if missing:
            routes = self._ci_greedy(routes, missing)

        return self._cleanup_solution(routes)

    # ======================================================================
    # Basic helpers
    # ======================================================================
    def _route_key(self, route: List[int]) -> Tuple[int, ...]:
        return tuple(route)

    @staticmethod
    def _cache_store(cache: Dict, key: Any, value: Any, limit: int) -> None:
        """Bound deterministic memoization without changing computed values."""

        if key not in cache and len(cache) >= limit:
            eviction_count = max(1, limit // 8)
            for old_key in list(islice(cache, eviction_count)):
                del cache[old_key]
        cache[key] = value

    def _clear_caches(self) -> None:
        self._sim_cache.clear()
        self._feasibility_cache.clear()
        self._dist_cache.clear()
        self._demand_cache.clear()
        self._time_cache.clear()
        self._best_station_arc_cache.clear()

    def _maybe_clear_caches(self, it: int) -> None:
        if it % 200 == 0:
            self._clear_caches()

    def _route_distance(self, route: List[int]) -> float:
        key = self._route_key(route)
        val = self._dist_cache.get(key)
        if val is not None:
            return val
        val = float(sum(self.dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)))
        self._cache_store(self._dist_cache, key, val, self.scalar_cache_limit)
        return val

    def _route_total_time(self, route: List[int]) -> float:
        key = self._route_key(route)
        val = self._time_cache.get(key)
        if val is not None:
            return val

        sim = self._simulate_route(route)
        val = float("inf") if not sim["feasible"] else (sim["departure_times"][-1] if len(route) > 1 else 0.0)
        self._cache_store(self._time_cache, key, val, self.scalar_cache_limit)
        return val

    def _route_demand(self, route: List[int]) -> float:
        key = self._route_key(route)
        val = self._demand_cache.get(key)
        if val is not None:
            return val
        val = sum(float(self.nodes[u]["demand"]) for u in route if u in self.customer_to_mask)
        self._cache_store(self._demand_cache, key, val, self.scalar_cache_limit)
        return val

    def _energy(self, i: int, j: int) -> float:
        return float(self.energy_matrix[i, j])

    def _travel_time(self, i: int, j: int) -> float:
        return float(self.time_matrix[i, j])

    def _cleanup_route(self, route: List[int]) -> List[int]:
        if not route:
            return [0, 0]
        if route[0] != 0:
            route = [0] + route
        if route[-1] != 0:
            route = route + [0]

        cleaned = [route[0]]
        for u in route[1:]:
            if u == 0 and cleaned[-1] == 0:
                continue
            if u in self.station_index_set and cleaned[-1] == u:
                continue
            cleaned.append(u)

        if cleaned[-1] != 0:
            cleaned.append(0)
        return cleaned

    def _prune_redundant_stations(self, route: List[int]) -> List[int]:
        route = self._cleanup_route(route)
        if not any(u in self.station_index_set for u in route):
            return route

        changed = True
        while changed:
            changed = False
            for pos in range(1, len(route) - 1):
                if route[pos] not in self.station_index_set:
                    continue
                trial = route[:pos] + route[pos + 1 :]
                if self.is_route_feasible(trial):
                    route = trial
                    changed = True
                    break

        return self._cleanup_route(route)

    def _cleanup_solution(self, routes: List[List[int]]) -> List[List[int]]:
        out = []
        for r in routes:
            r = self._cleanup_route(r)
            if self._has_customer(r):
                out.append(r)
        return out

    def _make_single_customer_route(self, customer: int) -> Optional[List[int]]:
        cached = self._singleton_route_cache.get(customer)
        if cached is not None:
            return list(cached)
        route = [0, customer, 0]
        route = self._repair_route_with_si(route, "gsi")
        if route is not None:
            self._singleton_route_cache[customer] = list(route)
            return list(route)
        return None

    def _list_customers(self, routes: List[List[int]]) -> List[int]:
        return [u for r in routes for u in r if u in self.customer_to_mask]

    def _has_customer(self, route: List[int]) -> bool:
        return any(u in self.customer_to_mask for u in route)

    def _served_mask(self, routes: List[List[int]]) -> List[bool]:
        mask = [False] * self.n_customers
        for route in routes:
            for u in route:
                idx = self.customer_to_mask.get(u)
                if idx is not None:
                    mask[idx] = True
        return mask

    def _export_routes(self, routes: List[List[int]]) -> List[List[str]]:
        return [[self.nodes[u]["id"] for u in route] for route in routes]

    def _all_station_positions(self, routes: List[List[int]]) -> List[Tuple[int, int]]:
        positions = []
        for ridx, route in enumerate(routes):
            for pos in range(1, len(route) - 1):
                if route[pos] in self.station_index_set:
                    positions.append((ridx, pos))
        return positions

    def _route_of_customer(self, routes: List[List[int]], customer: int) -> Optional[int]:
        for ridx, route in enumerate(routes):
            if customer in route:
                return ridx
        return None

    def _num_customers_to_remove(self) -> int:
        return self.rng.randint(self.min_remove_customers, self.max_remove_customers)

    def _roulette(self, weights: Dict[str, float]) -> str:
        total = sum(max(0.0, w) for w in weights.values())
        if total <= 0:
            return self.rng.choice(list(weights.keys()))

        x = self.rng.random() * total
        acc = 0.0
        for name, w in weights.items():
            acc += max(0.0, w)
            if acc >= x:
                return name
        return next(iter(weights.keys()))

    def _update_weights(self, weights, scores, uses):
        for name in weights:
            if uses[name] > 0:
                avg = scores[name] / uses[name]
                weights[name] = (1.0 - self.reaction_factor) * weights[name] + self.reaction_factor * avg
            scores[name] = 0.0
            uses[name] = 0

    def update_diversification_history(self, routes: List[List[int]]):
        if self.lambda_div <= 0.0:
            return
        for k, route in enumerate(routes):
            for pos in range(1, len(route) - 1):
                u = route[pos]
                if u not in self.customer_to_mask:
                    continue
                key = (u, k, route[pos - 1], route[pos + 1])
                self.attribute_frequency[key] += 1
                self.attribute_total += 1

    # ======================================================================
    # Ranked randomized selection helpers
    # ======================================================================
    def _select_ranked_with_noise(self, scored_desc: List[Tuple[float, int]], q: int, determinism: int) -> List[int]:
        chosen = []
        items = list(scored_desc)
        while items and len(chosen) < q:
            idx = self._biased_rank_index(len(items), determinism)
            _, u = items.pop(idx)
            chosen.append(u)
        return chosen

    def _select_one_ranked_with_noise(self, scored_asc: List[Tuple[float, int]], determinism: int) -> int:
        idx = self._biased_rank_index(len(scored_asc), determinism)
        return scored_asc[idx][1]

    def _biased_rank_index(self, n: int, determinism: int) -> int:
        if n <= 1:
            return 0
        k = self.rng.random()
        idx = int((k ** determinism) * n)
        return min(max(idx, 0), n - 1)

    # ======================================================================
    # Zone helpers
    # ======================================================================
    def _build_zone_map(self, customers: List[int]) -> Dict[int, int]:
        if not customers:
            return {}

        xs = [float(self.nodes[c]["x"]) for c in customers]
        ys = [float(self.nodes[c]["y"]) for c in customers]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        side = max(1, int(round(math.sqrt(self.n_zones))))
        dx = max(1e-9, (x_max - x_min) / side)
        dy = max(1e-9, (y_max - y_min) / side)

        zone_map = {}
        for c in customers:
            x = float(self.nodes[c]["x"])
            y = float(self.nodes[c]["y"])
            ix = min(side - 1, int((x - x_min) / dx))
            iy = min(side - 1, int((y - y_min) / dy))
            zone_map[c] = iy * side + ix

        return zone_map
