import random
import math
import copy
from collections import deque, defaultdict
import numpy as np
from tqdm import tqdm
import time
from typing import Callable, Optional

from vnst_adapter import Route


NUMERICAL_EPS = 1e-9


class VNSTSolver:
    """
    VNS + Tabu Search implementation with the following key fixes / optimizations:

    1) Tabu Search still enumerates the full neighborhood (no sampling), BUT:
       - It does NOT materialize a huge list of candidate solutions.
       - It evaluates candidates online via apply -> evaluate -> rollback.
       - This avoids O(#candidates) deepcopy + huge memory/GC pressure.

    2) No "candidate-generation side effects":
       - StationReIn local tabu list is updated ONLY when a move is accepted (not while enumerating).

    3) Fix several correctness/performance bugs in the original snippet:
       - create_new_route already includes depot; removed double-depot appends.
       - cyclic_exchange / extra_exchange no longer mutate the input solution in-place.
       - _tabu_search global_value / global_solution update fixed.
       - extend() misuse fixed (extend returns None).
       - battery_to_nearest_rs now returns dict (instead of None).
       - instance_dist_matrix_calculatrion handles empty stations/customers safely.
       - time_violation/time_penalty indexing corrected (avoid i-1 when i=0).
       - load_violation signature clarified: node parameter means "node not yet in route".

    NOTE:
    - This code preserves the "full enumeration" semantics inside TS.
    - Performance will still be dominated by neighborhood size for large instances,
      but it will be dramatically faster than deep-copying every candidate.
    """

    def __init__(
        self,
        instance,
        predefine_route_number=3,
        show_progress=False,
        search_mode="fast",
        move_candidate_limit=80,
        route_neighbor_limit=4,
        position_neighbor_limit=4,
        exchange_neighbor_limit=6,
        station_candidate_limit=5,
    ):
        self.instance = instance
        self.show_progress = bool(show_progress)
        self.search_mode = str(search_mode)
        self.move_candidate_limit = int(move_candidate_limit)
        self.route_neighbor_limit = int(route_neighbor_limit)
        self.position_neighbor_limit = int(position_neighbor_limit)
        self.exchange_neighbor_limit = int(exchange_neighbor_limit)
        self.station_candidate_limit = int(station_candidate_limit)
        self.fast_policy_version = "adaptive_nearest_best_fit_v3"

        # Tabu (global) / SA
        self.tabu_list = deque(maxlen=30)
        self.temp = -1
        self.delta_sa = 0.08

        # TS parameters
        self.tabu_tenure = 30
        self.tabu_iter = 100

        # Penalty params
        self.alpha, self.beta, self.gamma = 10.0, 10.0, 10.0
        self.alpha_min, self.beta_min, self.gamma_min = 0.5, 0.75, 1.0
        self.alpha_max, self.beta_max, self.gamma_max = 5000, 5000, 5000

        # VNS parameters
        self.k_max = 15
        self.η_feas = 700
        self.η_dist = 100
        self.predefine_route_number = predefine_route_number

        # Diversification bookkeeping
        self.attribute_frequency = defaultdict(int)
        self.attribute_total = 0
        self.lambda_div = 1.0
        self._history_max_entries = max(10_000, 80 * len(instance.customers))

        # Global best
        self.global_value = 1e10
        self.global_solution = None

        # Precompute geometry for nearest station
        self.recharging_stations = np.array(
            [[s.x, s.y] for s in instance.stations], dtype=float
        ) if len(instance.stations) > 0 else np.zeros((0, 2), dtype=float)

        self.instance_dist_matrix_calculatrion()

        if getattr(instance, "time_matrix", None) is not None:
            self.time_matrix = np.asarray(instance.time_matrix, dtype=float)
        else:
            self.time_matrix = self.dist_matrix / float(instance.vehicle_params["velocity"])
        if getattr(instance, "energy_matrix", None) is not None:
            self.energy_matrix = np.asarray(instance.energy_matrix, dtype=float)
        else:
            self.energy_matrix = (
                self.dist_matrix * float(instance.vehicle_params["consump_rate"])
            )
        expected_shape = self.dist_matrix.shape
        if self.time_matrix.shape != expected_shape or self.energy_matrix.shape != expected_shape:
            raise ValueError("distance, travel-time, and energy matrices must have identical shapes")
        self._canonicalize_terminal_matrix_order()
        self.all_customer_ids = frozenset(customer.id for customer in instance.customers)
        self.nearest_station = self.battery_to_nearest_rs(instance.depot)

        self.terminated_by_time_limit = False
        self._solve_start = None
        self._deadline = None
        self._incumbent_callback = None
        self._last_reported_value = None

        # Local tabu for StationReIn (arc -> remaining tenure)
        self.StationReIn_tabu_list = {}

        # Canonical nearest-neighbour ranks are reused by every fast TS round.
        # They turn the Cus500 exchange/relocate proposal phase from repeatedly
        # sorting all customer pairs into bounded nearest-neighbour lookups.
        self._customer_by_id = {customer.id: customer for customer in instance.customers}
        self._customer_neighbor_ids = self._precompute_customer_neighbors()

        # A certified singleton solution is published first, then this bounded
        # deterministic pass consolidates customers without ever making a
        # partial or unchecked route set visible to the benchmark runner.
        if len(instance.customers) >= 500:
            self.initial_construction_budget_s = 0.65
            self.initial_merge_route_limit = 12
            self.initial_exact_candidate_limit = 20
        elif len(instance.customers) >= 100:
            self.initial_construction_budget_s = 0.65
            self.initial_merge_route_limit = 16
            self.initial_exact_candidate_limit = 28
        else:
            self.initial_construction_budget_s = 0.50
            self.initial_merge_route_limit = 24
            self.initial_exact_candidate_limit = 40
        self.initial_construction_strategy = "certificate_singleton_best_fit_v1"
        self.initial_construction_stats = {}
        self.singleton_source = "none"

    # -------------------------
    # Basic utilities
    # -------------------------
    def instance_dist_matrix_calculatrion(self):
        """Map node.id -> index for dist/time matrix lookup."""
        terminal_order = list(getattr(self.instance, "terminal_order", ()) or ())
        if terminal_order:
            self.node_id = {
                terminal_id: index for index, terminal_id in enumerate(terminal_order)
            }
        else:
            legacy_order = [
                self.instance.depot.id,
                *(station.id for station in self.instance.stations),
                *(customer.id for customer in self.instance.customers),
            ]
            self.node_id = {
                terminal_id: index for index, terminal_id in enumerate(legacy_order)
            }

        self.dist_matrix = self.instance.dist_matrix

    def _canonicalize_terminal_matrix_order(self):
        """Use canonical terminal ids as direct matrix indices when possible.

        The Stage-2 adapter already supplies contiguous canonical ids and needs
        no copy. Legacy external instances may still use
        ``depot, stations, customers`` order; those matrices are permuted once
        to remove id-dictionary lookups from the search hot path. The fallback
        retains the legacy mapping for non-contiguous external ids. Matrix
        values are only permuted, never recomputed.
        """
        terminal_count = int(self.dist_matrix.shape[0])
        raw_terminal_ids = [
            self.instance.depot.id,
            *(station.id for station in self.instance.stations),
            *(customer.id for customer in self.instance.customers),
        ]
        integer_terminal_ids = all(
            isinstance(terminal_id, (int, np.integer))
            for terminal_id in raw_terminal_ids
        )
        terminal_ids = (
            [int(terminal_id) for terminal_id in raw_terminal_ids]
            if integer_terminal_ids
            else raw_terminal_ids
        )
        self._direct_terminal_index = (
            integer_terminal_ids
            and len(terminal_ids) == terminal_count
            and len(set(terminal_ids)) == terminal_count
            and set(terminal_ids) == set(range(terminal_count))
        )
        if not self._direct_terminal_index:
            return

        permutation = np.fromiter(
            (self.node_id[terminal_id] for terminal_id in range(terminal_count)),
            dtype=np.intp,
            count=terminal_count,
        )
        if not np.array_equal(permutation, np.arange(terminal_count, dtype=np.intp)):
            matrix_index = np.ix_(permutation, permutation)
            self.dist_matrix = np.asarray(self.dist_matrix)[matrix_index]
            self.time_matrix = np.asarray(self.time_matrix)[matrix_index]
            self.energy_matrix = np.asarray(self.energy_matrix)[matrix_index]
        self.node_id = {terminal_id: terminal_id for terminal_id in range(terminal_count)}

    def time_cost(self, node1, node2):
        if self._direct_terminal_index:
            return self.time_matrix[node1.id, node2.id]
        return self.time_matrix[self.node_id[node1.id]][self.node_id[node2.id]]

    def fuel_consumption(self, node1, node2):
        if self._direct_terminal_index:
            return float(self.energy_matrix[node1.id, node2.id])
        return float(self.energy_matrix[self.node_id[node1.id], self.node_id[node2.id]])

    def charging_time(self, station, missing_energy_kwh):
        power = float(self.instance.station_charging_power_kw[station.id])
        power_factor = float(
            self.instance.vehicle_params.get(
                "charging_power_derating_factor",
                self.instance.vehicle_params.get("charging_efficiency", 1.0),
            )
        )
        return max(0.0, float(missing_energy_kwh)) / (power_factor * power) * 3600.0

    def _time_limit_reached(self):
        if self._deadline is None:
            return False
        reached = time.perf_counter() >= self._deadline
        if reached:
            self.terminated_by_time_limit = True
        return reached

    def _report_incumbent(self):
        if (
            self._incumbent_callback is None
            or self._solve_start is None
            or self.global_solution is None
            or not math.isfinite(float(self.global_value))
            or float(self.global_value) >= 1e10
        ):
            return
        # Keep the solver's objective and route as one atomic incumbent.  Some
        # tabu paths retain a historical local best while evaluating a worse
        # current candidate; publishing the current value with those historical
        # routes would corrupt aspiration thresholds even though the runner
        # independently replays the route.  Recompute only on improvements,
        # where the extra full-solution pass is negligible relative to search.
        value = float(
            self.generalized_cost(
                self.global_solution,
                penalty_value=False,
                p_div_value=False,
                allow_infeasible=False,
            )
        )
        if not math.isfinite(value) or value >= 1e10:
            return
        self.global_value = value
        if self._last_reported_value is not None and value >= self._last_reported_value:
            return
        routes = [
            [int(node.id) for node in route.nodes]
            for route in self.global_solution
        ]
        self._incumbent_callback(
            time.perf_counter() - self._solve_start,
            value,
            routes,
        )
        self._last_reported_value = value

    def _precompute_customer_neighbors(self):
        customer_ids = list(self._customer_by_id)
        if not customer_ids:
            return {}
        distance = self.dist_matrix
        node_id = self.node_id
        direct = self._direct_terminal_index
        # A small immutable prefix is enough for every bounded neighbourhood.
        # Stable NumPy sorting preserves customer-order ties while avoiding
        # hundreds of thousands of Python tuples at Cus500.
        retained_count = min(
            max(0, len(customer_ids) - 1),
            max(32, 16 * self.route_neighbor_limit, 4 * self.exchange_neighbor_limit),
        )
        if retained_count == 0:
            return {customer_id: () for customer_id in customer_ids}
        matrix_indices = np.asarray(
            [customer_id if direct else node_id[customer_id] for customer_id in customer_ids],
            dtype=np.intp,
        )
        customer_distance = distance[np.ix_(matrix_indices, matrix_indices)]
        # Stable full ranking is intentional: Stage-2 road matrices can contain
        # exact distance ties, and argpartition would make the cutoff subset
        # platform-dependent.  This runs once at construction (n <= 500 in the
        # cross-algorithm track), not inside search iterations.
        ranks = np.argsort(customer_distance, axis=1, kind="stable")
        neighbors = {}
        for row, customer_id in enumerate(customer_ids):
            selected = []
            for column in ranks[row]:
                other_id = customer_ids[int(column)]
                if other_id == customer_id:
                    continue
                selected.append(other_id)
                if len(selected) >= retained_count:
                    break
            neighbors[customer_id] = tuple(selected)
        return neighbors

    def battery_to_nearest_rs(self, node):
        """Precompute fuel-to-nearest-station lower bound for each customer (used optionally elsewhere)."""
        nearest_station = {self.instance.depot.id: 0.0}
        self.nearest_station_idx = {}

        for st in self.instance.stations:
            nearest_station[st.id] = 0.0

        # If no stations exist, set large value for customers (or 0; depends on your modeling).
        if len(self.instance.stations) == 0:
            for cu in self.instance.customers:
                nearest_station[cu.id] = float("inf")
                self.nearest_station_idx[cu.id] = None
            return nearest_station

        for cu in self.instance.customers:
            energies = [self.fuel_consumption(cu, station) for station in self.instance.stations]
            arg = int(np.argmin(energies))
            st = self.instance.stations[arg]
            self.nearest_station_idx[cu.id] = st.id
            nearest_station[cu.id] = float(energies[arg])

        return nearest_station

    def create_new_route(self):
        """Route starts at depot by convention."""
        return Route([self.instance.depot])

    def effective_fast_policy(self, customer_count=None):
        """Expose the effective bounded-neighbourhood policy for experiment logs."""
        count = len(self.instance.customers) if customer_count is None else customer_count
        route_limit, position_limit, exchange_limit, candidate_limit = (
            self._effective_fast_limits(count)
        )
        return {
            "version": self.fast_policy_version,
            "customer_count": int(count),
            "route_neighbor_limit": route_limit,
            "position_neighbor_limit": position_limit,
            "exchange_neighbor_limit": exchange_limit,
            "move_candidate_limit": candidate_limit,
            "station_candidate_limit": self.station_candidate_limit,
        }

    def clone_route_shallow(self, route):
        """Clone route object, shallow-copy nodes list (Node objects reused)."""
        new_r = Route()
        new_r.nodes = list(route.nodes)
        # keep auxiliary fields if present
        if hasattr(route, "load"):
            new_r.load = route.load
        if hasattr(route, "time"):
            new_r.time = route.time
        if hasattr(route, "fuel"):
            new_r.fuel = route.fuel
        return new_r

    def clone_solution_shallow(self, solution):
        """Clone list of routes, shallow-copy each route's nodes list."""
        return [self.clone_route_shallow(r) for r in solution]

    # -------------------------
    # Feasibility & penalties
    # -------------------------
    def load_violation(self, route, node=None):
        """
        If node is provided, it is assumed node is NOT already in route.
        """
        cap = float(self.instance.vehicle_params["load_cap"])
        load_sum = sum(float(n.demand) for n in route.nodes)
        if node is not None:
            load_sum += float(node.demand)
        return load_sum > cap + NUMERICAL_EPS

    def load_penalty(self, route):
        cap = float(self.instance.vehicle_params["load_cap"])
        load_sum = sum(float(n.demand) for n in route.nodes)
        return max(0.0, load_sum - cap)

    def battery_violation(self, route, node=None):
        fuel_cap = float(self.instance.vehicle_params["fuel_cap"])
        current_fuel = fuel_cap

        for i in range(len(route.nodes) - 1):
            a = route.nodes[i]
            b = route.nodes[i + 1]
            current_fuel -= self.fuel_consumption(a, b)
            if current_fuel < -NUMERICAL_EPS:
                return True
            if b.type == "f":
                current_fuel = fuel_cap

        if node is not None and len(route.nodes) > 0:
            current_fuel -= self.fuel_consumption(route.nodes[-1], node)
            if current_fuel < -NUMERICAL_EPS:
                return True
        return False

    def battery_penalty(self, route):
        fuel_cap = float(self.instance.vehicle_params["fuel_cap"])
        gamma_in = 0.0
        pen = 0.0

        for i in range(len(route.nodes) - 1):
            a = route.nodes[i]
            b = route.nodes[i + 1]
            gamma_in += self.fuel_consumption(a, b)
            pen += max(0.0, gamma_in - fuel_cap)
            if b.type == "f":
                gamma_in = 0.0
        return pen

    def time_violation(self, route, node=None):
        """
        Time-window feasibility with charging time.
        At a station, recharge exactly the energy spent since the previous
        depot/station, using that station's own power and the configured
        charging efficiency.
        """
        current_time = 0.0
        energy_since_last_charge = 0.0

        # walk along nodes
        for i, cur in enumerate(route.nodes):
            # arrive time (current_time already includes travel from previous)
            arrival = max(current_time, float(cur.ready))
            if arrival > float(cur.due) + NUMERICAL_EPS:
                return True

            if cur.type == "c":
                # service
                current_time = arrival + float(cur.service)

            elif cur.type == "f":
                # charge
                if energy_since_last_charge > 0.0:
                    current_time = arrival + self.charging_time(cur, energy_since_last_charge)
                else:
                    current_time = arrival
                energy_since_last_charge = 0.0
            else:
                # depot or others
                current_time = arrival

            # travel to next
            if i < len(route.nodes) - 1:
                nxt = route.nodes[i + 1]
                current_time += self.time_cost(cur, nxt)
                # energy accrued on traveling cur->nxt
                energy_since_last_charge += self.fuel_consumption(cur, nxt)

        # optional append check
        if node is not None and len(route.nodes) > 0:
            last = route.nodes[-1]
            projected = current_time + self.time_cost(last, node)
            if projected > float(node.due) + NUMERICAL_EPS:
                return True
        return False

    def time_penalty(self, route):
        """Lateness penalty with the same charging-time model as time_violation."""
        current_time = 0.0
        energy_since_last_charge = 0.0
        pen = 0.0

        for i, cur in enumerate(route.nodes):
            arrival = max(current_time, float(cur.ready))
            if arrival > float(cur.due):
                pen += (arrival - float(cur.due))
                # Lateness does not move the physical clock backwards. Keeping
                # the real arrival propagates delay to downstream customers so
                # infeasible-neighbour ranking matches the route simulator.

            if cur.type == "c":
                current_time = arrival + float(cur.service)
            elif cur.type == "f":
                if energy_since_last_charge > 0.0:
                    current_time = arrival + self.charging_time(cur, energy_since_last_charge)
                else:
                    current_time = arrival
                energy_since_last_charge = 0.0
            else:
                current_time = arrival

            if i < len(route.nodes) - 1:
                nxt = route.nodes[i + 1]
                current_time += self.time_cost(cur, nxt)
                energy_since_last_charge += self.fuel_consumption(cur, nxt)

        return pen

    def is_route_feasible(self, route):
        nodes = getattr(route, "nodes", None)
        if nodes is None or len(nodes) < 3:
            return False
        depot_id = self.instance.depot.id
        if (
            nodes[0].id != depot_id
            or nodes[-1].id != depot_id
            or nodes[0].type != "d"
            or nodes[-1].type != "d"
        ):
            return False
        if any(node.type == "d" for node in nodes[1:-1]):
            return False
        if not any(node.type == "c" for node in nodes[1:-1]):
            return False
        if any(node.id not in self.node_id for node in nodes):
            return False
        return not (self.load_violation(route) or self.time_violation(route) or self.battery_violation(route))

    @staticmethod
    def _route_cache_key(route):
        return tuple(node.id for node in route.nodes)

    def is_solution_feasible(self, solution, route_feasibility_cache=None):
        """Full feasibility (including 'served exactly once')."""
        if not solution:
            return False
        served = set()

        for route in solution:
            if route_feasibility_cache is None:
                route_feasible = self.is_route_feasible(route)
            else:
                cache_key = self._route_cache_key(route)
                route_feasible = route_feasibility_cache.get(cache_key)
                if route_feasible is None:
                    route_feasible = self.is_route_feasible(route)
                    route_feasibility_cache[cache_key] = route_feasible
            if not route_feasible:
                return False
            for n in route.nodes:
                if n.type == "c":
                    if n.id in served:
                        return False
                    served.add(n.id)

        return served == self.all_customer_ids

    # -------------------------
    # Cost function
    # -------------------------
    def generalized_cost(
        self,
        S,
        penalty_value=True,
        p_div_value=True,
        allow_infeasible=True,
        route_feasibility_cache=None,
    ):
        if not allow_infeasible and not self.is_solution_feasible(
            S, route_feasibility_cache=route_feasibility_cache
        ):
            return 1e10

        total_distance = 0.0
        distance = self.dist_matrix
        if self._direct_terminal_index:
            for route in S:
                nodes = route.nodes
                for i in range(len(nodes) - 1):
                    total_distance += distance[nodes[i].id, nodes[i + 1].id]
        else:
            node_id = self.node_id
            for route in S:
                nodes = route.nodes
                for i in range(len(nodes) - 1):
                    total_distance += distance[node_id[nodes[i].id], node_id[nodes[i + 1].id]]

        total_penalty = 0.0
        if penalty_value:
            for route in S:
                total_penalty += (
                    self.alpha * self.load_penalty(route) +
                    self.beta * self.time_penalty(route) +
                    self.gamma * self.battery_penalty(route)
                )

        p_div_penalty = 0.0
        if p_div_value:
            num_customers = sum(max(0, len(r.nodes) - 2) for r in S)
            num_vehicles = len(S)
            penalty_sum = 0.0
            for k, route in enumerate(S):
                nodes = route.nodes
                for i in range(1, len(nodes) - 1):
                    key = (nodes[i].id, k, nodes[i - 1].id, nodes[i + 1].id)
                    penalty_sum += self.attribute_frequency.get(key, 0)

            denom = (1e-10 + float(self.attribute_total))
            p_div_penalty = (self.lambda_div * total_distance * penalty_sum *
                             math.sqrt(float(num_customers * num_vehicles)) / denom)

        return total_distance + total_penalty + p_div_penalty

    # -------------------------
    # SA acceptance
    # -------------------------
    def accept_sa(self, S_new, S_old):
        cost_new = self.generalized_cost(S_new, penalty_value=False, p_div_value=False, allow_infeasible=False)
        cost_old = self.generalized_cost(S_old, penalty_value=False, p_div_value=False, allow_infeasible=False)
        diff = cost_new - cost_old

        if diff <= 0:
            return True

        if self.temp == -1:
            self.temp = -diff / math.log(0.5)
            self.cooling = (1 - self.delta_sa)
        else:
            self.temp *= self.cooling

        return random.random() < math.exp(-diff / max(1e-12, self.temp))

    # -------------------------
    # Penalty weight update
    # -------------------------
    def update_reset(self):
        self.load_update = False
        self.batt_update = False
        self.tw_update = False

    def update_penalty_weights(self, solution, step):
        delta = 1.2
        penalty_update_interval = 2

        load_v = sum(self.load_penalty(r) for r in solution)
        tw_v = sum(self.time_penalty(r) for r in solution)
        batt_v = sum(self.battery_penalty(r) for r in solution)

        self.load_update = load_v > 0
        self.tw_update = tw_v > 0
        self.batt_update = batt_v > 0

        if step % penalty_update_interval == 0:
            if self.load_update:
                self.alpha = min(self.alpha * delta, self.alpha_max)
            else:
                self.alpha = max(self.alpha / delta, self.alpha_min)

            if self.tw_update:
                self.beta = min(self.beta * delta, self.beta_max)
            else:
                self.beta = max(self.beta / delta, self.beta_min)

            if self.batt_update:
                self.gamma = min(self.gamma * delta, self.gamma_max)
            else:
                self.gamma = max(self.gamma / delta, self.gamma_min)

            self.update_reset()

    # -------------------------
    # Diversification history
    # -------------------------
    def update_diversification_history(self, S):
        for k, route in enumerate(S):
            nodes = route.nodes
            for i in range(1, len(nodes) - 1):
                u = nodes[i].id
                mu = nodes[i - 1].id
                zeta = nodes[i + 1].id
                self.attribute_frequency[(u, k, mu, zeta)] += 1
                self.attribute_total += 1
        if len(self.attribute_frequency) > self._history_max_entries:
            # Keep memory bounded on multi-hour runs.  The deterministic decay
            # preserves relative long-term frequencies without retaining every
            # route attribute ever seen.
            decayed = defaultdict(int)
            total = 0
            for attribute, frequency in self.attribute_frequency.items():
                retained = frequency // 2
                if retained:
                    decayed[attribute] = retained
                    total += retained
            self.attribute_frequency = decayed
            self.attribute_total = total

    # -------------------------
    # Initial solution
    # -------------------------
    def polar_angle(self, customer, depot, random_point):
        dx1, dy1 = random_point.x - depot.x, random_point.y - depot.y
        dx2, dy2 = customer.x - depot.x, customer.y - depot.y
        angle1 = math.atan2(dy1, dx1)
        angle2 = math.atan2(dy2, dx2)
        return (angle2 - angle1) % (2 * math.pi)

    def initial_solution(self):
        depot = self.instance.depot
        customers = list(self.instance.customers)

        if len(customers) == 0:
            return [self.create_new_route() for _ in range(self.predefine_route_number)]

        rp = random.choice(customers)
        customers_sorted = sorted(customers, key=lambda c: self.polar_angle(c, depot, rp))

        predefined_routes = self.predefine_route_number
        routes = []

        current_route = self.create_new_route()  # already has depot
        # ensure route ends with depot when evaluating
        if current_route.nodes[-1].type != "d":
            current_route.nodes.append(depot)

        last_route = self.create_new_route()
        if last_route.nodes[-1].type != "d":
            last_route.nodes.append(depot)

        unassigned = []

        for customer in customers_sorted:
            best_pos = None
            best_cost = float("inf")

            # try insert into current_route before the ending depot
            for pos in range(1, len(current_route.nodes)):  # positions including before last depot
                # do a shallow trial (no deepcopy of nodes)
                trial = self.clone_route_shallow(current_route)
                trial.nodes.insert(pos, customer)

                if (not self.load_violation(trial) and not self.time_violation(trial) and not self.battery_violation(trial)):
                    cst = self.generalized_cost([trial], penalty_value=False, p_div_value=False, allow_infeasible=True)
                    if cst < best_cost:
                        best_cost = cst
                        best_pos = pos

            if best_pos is not None:
                current_route.nodes.insert(best_pos, customer)
            else:
                if len(routes) < predefined_routes:
                    routes.append(current_route)
                    current_route = self.create_new_route()
                    current_route.nodes.append(customer)
                    current_route.nodes.append(depot)
                else:
                    unassigned.append(customer)

        if len(current_route.nodes) > 2:
            routes.append(current_route)

        if len(unassigned) > 0:
            unassigned.sort(key=lambda c: float(c.ready))
            # last_route currently [depot, depot]; keep single start depot
            last_route.nodes = [depot] + unassigned + [depot]
            routes.append(last_route)

        while len(routes) < predefined_routes:
            r = self.create_new_route()
            r.nodes.append(depot)
            routes.append(r)

        return routes

    def singleton_warm_start(self):
        """Build a replayed infinite-fleet incumbent before costly clustering.

        Prefer the Stage-2 certificate routes, which can contain multiple
        charger hops and were already replayed by the canonical adapter.  The
        complete adapted route set is checked again here before it can become an
        incumbent.  Direct singleton trips are a conservative legacy fallback.
        """
        depot = self.instance.depot
        node_by_id = {
            node.id: node
            for node in [depot, *self.instance.customers, *self.instance.stations]
        }
        certificate_routes = getattr(
            self.instance, "certificate_singleton_routes", None
        )
        if certificate_routes is not None:
            try:
                routes = [
                    Route([node_by_id[int(node_id)] for node_id in route])
                    for route in certificate_routes
                ]
            except (KeyError, TypeError, ValueError):
                routes = []
            if routes and self.is_solution_feasible(routes):
                self.singleton_source = "stage2_certificate_replayed"
                return routes

        routes = []
        for customer in self.instance.customers:
            route = Route([depot, customer, depot])
            if not self.is_route_feasible(route):
                self.singleton_source = "none"
                return None
            routes.append(route)
        self.singleton_source = "direct_singleton_replayed"
        return routes

    def _route_distance_value(self, route):
        nodes = route.nodes
        return sum(self._dist(nodes[index], nodes[index + 1]) for index in range(len(nodes) - 1))

    @staticmethod
    def _singleton_insertion_segments(route, customer):
        """Return stable, de-duplicated customer/charger segments to try."""

        customer_position = next(
            index for index, node in enumerate(route.nodes) if node.id == customer.id
        )
        inbound = route.nodes[1:customer_position]
        outbound = route.nodes[customer_position + 1 : -1]
        raw_segments = (
            [customer],
            [*inbound, customer],
            [customer, *outbound],
            [*inbound, customer, *outbound],
        )
        seen = set()
        segments = []
        for segment in raw_segments:
            key = tuple(node.id for node in segment)
            if key not in seen:
                seen.add(key)
                segments.append(segment)
        return segments

    def _consolidate_singleton_solution(self, singleton_routes, deadline=None):
        """Build a complete best-fit solution under a short wall-clock budget.

        Distance deltas only rank work.  Every accepted insertion is checked by
        ``is_route_feasible`` and the complete result is checked again before it
        can replace the singleton fallback.
        """

        construction_start = time.perf_counter()
        construction_deadline = construction_start + self.initial_construction_budget_s
        if deadline is not None:
            construction_deadline = min(construction_deadline, deadline)

        singleton_by_customer = {}
        for route in singleton_routes:
            customers = [node for node in route.nodes if node.type == "c"]
            if len(customers) != 1 or customers[0].id in singleton_by_customer:
                self.initial_construction_stats = {
                    "strategy": self.initial_construction_strategy,
                    "singleton_source": self.singleton_source,
                    "singleton_route_count": len(singleton_routes),
                    "result_route_count": len(singleton_routes),
                    "merged_customer_count": 0,
                    "budget_s": self.initial_construction_budget_s,
                    "budget_exhausted": False,
                    "fallback_reason": "malformed_singleton_routes",
                    "elapsed_s": time.perf_counter() - construction_start,
                }
                return self.clone_solution_shallow(singleton_routes)
            singleton_by_customer[customers[0].id] = route
        if set(singleton_by_customer) != self.all_customer_ids:
            self.initial_construction_stats = {
                "strategy": self.initial_construction_strategy,
                "singleton_source": self.singleton_source,
                "singleton_route_count": len(singleton_routes),
                "result_route_count": len(singleton_routes),
                "merged_customer_count": 0,
                "budget_s": self.initial_construction_budget_s,
                "budget_exhausted": False,
                "fallback_reason": "incomplete_singleton_routes",
                "elapsed_s": time.perf_counter() - construction_start,
            }
            return self.clone_solution_shallow(singleton_routes)

        customer_order = sorted(
            self.instance.customers,
            key=lambda customer: (
                float(customer.due) - float(customer.ready),
                -float(customer.demand),
                int(customer.id),
            ),
        )
        merged_routes = []
        route_loads = []
        customer_route = {}
        merged_customer_count = 0
        budget_exhausted = False

        for order_position, customer in enumerate(customer_order):
            if time.perf_counter() >= construction_deadline:
                budget_exhausted = True
                for remaining in customer_order[order_position:]:
                    route = self.clone_route_shallow(singleton_by_customer[remaining.id])
                    customer_route[remaining.id] = len(merged_routes)
                    merged_routes.append(route)
                    route_loads.append(float(remaining.demand))
                break

            singleton = singleton_by_customer[customer.id]
            route_best_relatedness = {}
            if self._direct_terminal_index:
                distance_row = self.dist_matrix[customer.id]
                for other_id, route_index in customer_route.items():
                    value = float(distance_row[other_id])
                    previous = route_best_relatedness.get(route_index)
                    if previous is None or value < previous:
                        route_best_relatedness[route_index] = value
            else:
                distance_row = self.dist_matrix[self.node_id[customer.id]]
                for other_id, route_index in customer_route.items():
                    value = float(distance_row[self.node_id[other_id]])
                    previous = route_best_relatedness.get(route_index)
                    if previous is None or value < previous:
                        route_best_relatedness[route_index] = value
            route_ids = [
                route_index
                for route_index, _ in sorted(
                    route_best_relatedness.items(), key=lambda item: (item[1], item[0])
                )[: self.initial_merge_route_limit]
            ]

            cheap_candidates = []
            segments = self._singleton_insertion_segments(singleton, customer)
            for route_index in route_ids:
                if (
                    route_loads[route_index] + float(customer.demand)
                    > float(self.instance.vehicle_params["load_cap"]) + NUMERICAL_EPS
                ):
                    continue
                route = merged_routes[route_index]
                base_distance = self._route_distance_value(route)
                for segment_index, segment in enumerate(segments):
                    internal_distance = sum(
                        self._dist(segment[index], segment[index + 1])
                        for index in range(len(segment) - 1)
                    )
                    for insert_position in range(1, len(route.nodes)):
                        previous = route.nodes[insert_position - 1]
                        following = route.nodes[insert_position]
                        proxy = (
                            self._dist(previous, segment[0])
                            + internal_distance
                            + self._dist(segment[-1], following)
                            - self._dist(previous, following)
                        )
                        cheap_candidates.append(
                            (
                                proxy,
                                route_index,
                                insert_position,
                                segment_index,
                                base_distance,
                            )
                        )

            cheap_candidates.sort(key=lambda item: item[0])
            best = None
            for _, route_index, insert_position, segment_index, base_distance in (
                cheap_candidates[: self.initial_exact_candidate_limit]
            ):
                if time.perf_counter() >= construction_deadline:
                    budget_exhausted = True
                    break
                route = merged_routes[route_index]
                segment = segments[segment_index]
                trial = self.clone_route_shallow(route)
                trial.nodes[insert_position:insert_position] = segment
                if not self.is_route_feasible(trial):
                    continue
                delta = self._route_distance_value(trial) - base_distance
                if best is None or delta < best[0]:
                    best = (delta, route_index, trial)

            if best is None:
                route = self.clone_route_shallow(singleton)
                customer_route[customer.id] = len(merged_routes)
                merged_routes.append(route)
                route_loads.append(float(customer.demand))
            else:
                _, route_index, trial = best
                merged_routes[route_index] = trial
                route_loads[route_index] += float(customer.demand)
                customer_route[customer.id] = route_index
                merged_customer_count += 1

        if not self.is_solution_feasible(merged_routes):
            result = self.clone_solution_shallow(singleton_routes)
            fallback_reason = "complete_solution_replay_failed"
            merged_customer_count = 0
        else:
            result = merged_routes
            fallback_reason = ""
        self.initial_construction_stats = {
            "strategy": self.initial_construction_strategy,
            "singleton_source": self.singleton_source,
            "singleton_route_count": len(singleton_routes),
            "result_route_count": len(result),
            "merged_customer_count": merged_customer_count,
            "budget_s": self.initial_construction_budget_s,
            "budget_exhausted": budget_exhausted,
            "fallback_reason": fallback_reason,
            "elapsed_s": time.perf_counter() - construction_start,
        }
        return result

    # -------------------------
    # VNS perturbation
    # -------------------------
    def vns_perturb(self, solution, k):
        neighborhood_structure = {
            1: (2, 1),  2: (2, 2),  3: (2, 3),  4: (2, 4),  5: (2, 5),
            6: (3, 1),  7: (3, 2),  8: (3, 3),  9: (3, 4), 10: (3, 5),
            11: (4, 1), 12: (4, 2), 13: (4, 3), 14: (4, 4), 15: (4, 5)
        }

        if k not in neighborhood_structure:
            return self.clone_solution_shallow(solution)

        if len(solution) == 1:
            if random.random() < 0.3:
                return self.extra_exchange(solution)
            return self.clone_solution_shallow(solution)

        num_routes, max_nodes = neighborhood_structure[k]
        if len(solution) < num_routes:
            return self.clone_solution_shallow(solution)

        return self.cyclic_exchange(solution, num_routes, max_nodes)

    def cyclic_exchange(self, solution, num_routes, max_nodes):
        """Return a perturbed COPY (no in-place mutation of input solution)."""
        base = self.clone_solution_shallow(solution)
        if len(base) < num_routes:
            return base

        idxs = random.sample(range(len(base)), num_routes)
        segments, starts, ends = [], [], []

        for ridx in idxs:
            nodes = base[ridx].nodes
            if len(nodes) < 3:
                return base
            start = random.randint(1, len(nodes) - 2)
            max_len = min(max_nodes, len(nodes) - 2)
            chain_len = random.randint(0, max_len)
            end = min(start + chain_len, len(nodes) - 1)

            segments.append(nodes[start:end])
            starts.append(start)
            ends.append(end)

        for t in range(num_routes):
            nxt = (t + 1) % num_routes
            r = base[idxs[nxt]]
            r.nodes[starts[nxt]:ends[nxt]] = segments[t]

        return base

    def extra_exchange(self, solution):
        """Return a COPY with one customer extracted from the first route and put into a new route."""
        base = self.clone_solution_shallow(solution)
        if len(base) == 0 or len(base[0].nodes) <= 2:
            return base

        # find a customer in route 0
        tries = 0
        while tries < 20:
            node_idx = random.randint(1, len(base[0].nodes) - 2)
            if base[0].nodes[node_idx].type == "c":
                break
            tries += 1
        else:
            return base

        node = base[0].nodes.pop(node_idx)
        r = self.create_new_route()
        r.nodes.append(node)
        r.nodes.append(self.instance.depot)
        base.append(r)
        return base

    # -------------------------
    # Solution cleanup
    # -------------------------
    def solution_fix(self, solution):
        """Remove immediate duplicates + remove routes with no customers."""
        fixed = []
        for r in solution:
            nodes = r.nodes
            if not nodes:
                continue
            new_nodes = [nodes[0]]
            for i in range(1, len(nodes)):
                if nodes[i].id != nodes[i - 1].id:
                    new_nodes.append(nodes[i])
            r.nodes = new_nodes

            if any(n.type == "c" for n in r.nodes):
                fixed.append(r)
        return fixed

    # -------------------------
    # Tabu Search (full enumeration, apply+rollback, no candidate list)
    # -------------------------
    def _decay_station_tabu(self):
        for arc in list(self.StationReIn_tabu_list.keys()):
            self.StationReIn_tabu_list[arc] -= 1
            if self.StationReIn_tabu_list[arc] <= 0:
                del self.StationReIn_tabu_list[arc]

    def _tabu_search(self, S):
        current_solution = self.clone_solution_shallow(S)
        best_solution = self.clone_solution_shallow(S)
        tabu_list = deque(maxlen=self.tabu_tenure)

        # helper to build route signature once per iteration
        def route_sig(route):
            return "->".join(str(n.id) for n in route.nodes)

        for _iter in range(self.tabu_iter):
            if self._time_limit_reached():
                break
            self._decay_station_tabu()

            route_info = [route_sig(r) for r in current_solution]

            # Track best candidate under "full" cost (allow infeasible)
            best_move = None
            best_move_info = None
            best_move_cost = float("inf")

            # Track best feasible (distance-only objective) candidate
            best_feas_cost = float("inf")

            # -------------------------
            # Enumerate 2-opt* (between two routes)
            # -------------------------
            for i in range(len(current_solution) - 1):
                for j in range(i + 1, len(current_solution)):
                    ri = current_solution[i]
                    rj = current_solution[j]
                    ni = len(ri.nodes)
                    nj = len(rj.nodes)
                    if ni <= 2 or nj <= 2:
                        continue

                    for split1 in range(1, ni - 1):
                        for split2 in range(1, nj - 1):
                            old_i_nodes = ri.nodes
                            old_j_nodes = rj.nodes

                            # apply
                            ri.nodes = old_i_nodes[:split1] + old_j_nodes[split2:]
                            rj.nodes = old_j_nodes[:split2] + old_i_nodes[split1:]

                            # evaluate full cost
                            c_full = self.generalized_cost(current_solution, penalty_value=True, p_div_value=True, allow_infeasible=True)
                            info = ("Two_opt", f"{route_info[i]}@{split1}", f"{route_info[j]}@{split2}")

                            if c_full < best_move_cost and info not in tabu_list:
                                best_move_cost = c_full
                                best_move = ("two_opt", i, j, split1, split2, old_i_nodes, old_j_nodes)
                                best_move_info = info

                            # evaluate feasible distance-only
                            c_feas = self.generalized_cost(current_solution, penalty_value=False, p_div_value=False, allow_infeasible=False)
                            if c_feas < best_feas_cost:
                                best_feas_cost = c_feas

                            # rollback
                            ri.nodes = old_i_nodes
                            rj.nodes = old_j_nodes

            # -------------------------
            # Enumerate Relocate (including open new route)
            # -------------------------
            for i in range(len(current_solution)):
                ri = current_solution[i]
                if len(ri.nodes) <= 2:
                    continue

                for split_pos in range(1, len(ri.nodes) - 1):
                    node = ri.nodes[split_pos]
                    if node.type != "c":
                        continue

                    # relocate into existing route j
                    for j in range(len(current_solution)):
                        rj = current_solution[j]
                        for insert_pos in range(1, len(rj.nodes)):  # allow before last depot
                            if i == j and insert_pos == split_pos:
                                continue

                            # apply (in-place pop/insert with undo)
                            removed = ri.nodes.pop(split_pos)
                            adj_insert = insert_pos
                            if i == j and insert_pos > split_pos:
                                adj_insert -= 1
                            rj.nodes.insert(adj_insert, removed)

                            info = ("Relocate", f"{route_info[i]}@{split_pos}", f"{route_info[j]}@{insert_pos}")
                            c_full = self.generalized_cost(current_solution, True, True, True)
                            if c_full < best_move_cost and info not in tabu_list:
                                best_move_cost = c_full
                                best_move = ("relocate", i, j, split_pos, insert_pos, removed)
                                best_move_info = info

                            c_feas = self.generalized_cost(current_solution, False, False, False)
                            if c_feas < best_feas_cost:
                                best_feas_cost = c_feas

                            # rollback
                            rj.nodes.pop(adj_insert)
                            ri.nodes.insert(split_pos, removed)

                    # relocate to a new route (open one)
                    removed = ri.nodes.pop(split_pos)
                    new_route = self.create_new_route()
                    new_route.nodes.append(removed)
                    new_route.nodes.append(self.instance.depot)
                    current_solution.append(new_route)

                    info = ("RelocateNew", f"{route_info[i]}@{split_pos}")
                    c_full = self.generalized_cost(current_solution, True, True, True)
                    if c_full < best_move_cost and info not in tabu_list:
                        best_move_cost = c_full
                        best_move = ("relocate_new", i, split_pos, removed)
                        best_move_info = info

                    c_feas = self.generalized_cost(current_solution, False, False, False)
                    if c_feas < best_feas_cost:
                        best_feas_cost = c_feas

                    # rollback
                    current_solution.pop()
                    ri.nodes.insert(split_pos, removed)

            # -------------------------
            # Enumerate Exchange
            # -------------------------
            for i in range(len(current_solution)):
                ri = current_solution[i]
                for j in range(len(current_solution)):
                    rj = current_solution[j]
                    for p1 in range(1, len(ri.nodes) - 1):
                        if ri.nodes[p1].type != "c":
                            continue
                        for p2 in range(1, len(rj.nodes) - 1):
                            if rj.nodes[p2].type != "c":
                                continue
                            if i == j and p1 == p2:
                                continue

                            # apply swap
                            ri.nodes[p1], rj.nodes[p2] = rj.nodes[p2], ri.nodes[p1]

                            info = ("Exchange", f"{route_info[i]}@{p1}", f"{route_info[j]}@{p2}")
                            c_full = self.generalized_cost(current_solution, True, True, True)
                            if c_full < best_move_cost and info not in tabu_list:
                                best_move_cost = c_full
                                best_move = ("exchange", i, j, p1, p2)
                                best_move_info = info

                            c_feas = self.generalized_cost(current_solution, False, False, False)
                            if c_feas < best_feas_cost:
                                best_feas_cost = c_feas

                            # rollback
                            ri.nodes[p1], rj.nodes[p2] = rj.nodes[p2], ri.nodes[p1]

            # -------------------------
            # Enumerate StationReIn (remove/insert), local tabu checked but updated only if accepted
            # -------------------------
            for i in range(len(current_solution)):
                r = current_solution[i]
                if len(r.nodes) <= 2:
                    continue
                for pos in range(1, len(r.nodes) - 1):
                    cur = r.nodes[pos]

                    # remove station
                    if cur.type == "f":
                        mu = r.nodes[pos - 1]
                        zeta = r.nodes[pos + 1]
                        arc = (mu.id, zeta.id)

                        removed_station = r.nodes.pop(pos)

                        info = ("StationRemove", f"{route_info[i]}@{pos}")
                        c_full = self.generalized_cost(current_solution, True, True, True)
                        if c_full < best_move_cost and info not in tabu_list:
                            best_move_cost = c_full
                            best_move = ("station_remove", i, pos, removed_station, arc)
                            best_move_info = info

                        c_feas = self.generalized_cost(current_solution, False, False, False)
                        if c_feas < best_feas_cost:
                            best_feas_cost = c_feas

                        # rollback
                        r.nodes.insert(pos, removed_station)

                    # insert station
                    else:
                        prev = r.nodes[pos - 1]
                        # try all stations (full enumeration)
                        for st in self.instance.stations:
                            if prev.id == st.id:
                                continue
                            arc = (prev.id, st.id)
                            if self.StationReIn_tabu_list.get(arc, 0) > 0:
                                continue

                            r.nodes.insert(pos, st)

                            info = ("StationInsert", f"{route_info[i]}@{pos}", f"st={st.id}")
                            c_full = self.generalized_cost(current_solution, True, True, True)
                            if c_full < best_move_cost and info not in tabu_list:
                                best_move_cost = c_full
                                best_move = ("station_insert", i, pos, st, arc)
                                best_move_info = info

                            c_feas = self.generalized_cost(current_solution, False, False, False)
                            if c_feas < best_feas_cost:
                                best_feas_cost = c_feas

                            # rollback
                            r.nodes.pop(pos)

            # If no admissible move found (can happen if tabu blocks everything), break
            if best_move is None:
                break

            # Apply chosen move permanently (best non-tabu under full cost)
            m = best_move
            mtype = m[0]

            if mtype == "two_opt":
                _, i, j, split1, split2, old_i_nodes, old_j_nodes = m
                current_solution[i].nodes = old_i_nodes[:split1] + old_j_nodes[split2:]
                current_solution[j].nodes = old_j_nodes[:split2] + old_i_nodes[split1:]

            elif mtype == "relocate":
                _, i, j, split_pos, insert_pos, removed = m
                ri = current_solution[i]
                rj = current_solution[j]
                # remove at split_pos
                node = ri.nodes.pop(split_pos)
                adj_insert = insert_pos
                if i == j and insert_pos > split_pos:
                    adj_insert -= 1
                rj.nodes.insert(adj_insert, node)

            elif mtype == "relocate_new":
                _, i, split_pos, removed = m
                ri = current_solution[i]
                node = ri.nodes.pop(split_pos)
                nr = self.create_new_route()
                nr.nodes.append(node)
                nr.nodes.append(self.instance.depot)
                current_solution.append(nr)

            elif mtype == "exchange":
                _, i, j, p1, p2 = m
                ri = current_solution[i]
                rj = current_solution[j]
                ri.nodes[p1], rj.nodes[p2] = rj.nodes[p2], ri.nodes[p1]

            elif mtype == "station_remove":
                _, i, pos, station_node, arc = m
                r = current_solution[i]
                # remove at pos (should be same station)
                r.nodes.pop(pos)
                # local tabu update on accept
                self.StationReIn_tabu_list[arc] = random.randint(15, 30)

            elif mtype == "station_insert":
                _, i, pos, st, arc = m
                r = current_solution[i]
                r.nodes.insert(pos, st)
                # local tabu update on accept (arc now becomes tabu)
                self.StationReIn_tabu_list[arc] = random.randint(15, 30)

            # Update tabu list & diversification stats
            if best_move_info is not None:
                tabu_list.append(best_move_info)

            current_solution = self.solution_fix(current_solution)
            self.update_diversification_history(current_solution)

            # Track best_solution (you can choose either:
            #   - best feasible distance-only move's outcome, or
            #   - best feasible encountered current_solution
            # Here: update using current_solution feasibility.
            cur_val = self.generalized_cost(current_solution, penalty_value=False, p_div_value=False, allow_infeasible=False)
            best_val = self.generalized_cost(best_solution, penalty_value=False, p_div_value=False, allow_infeasible=False)
            if cur_val < best_val:
                best_solution = self.clone_solution_shallow(current_solution)
                best_val = cur_val

            # Update global best
            if best_val < self.global_value:
                self.global_value = best_val
                self.global_solution = self.clone_solution_shallow(best_solution)
                self._report_incumbent()

        return best_solution

    # -------------------------
    # Fast candidate-budgeted tabu search
    # -------------------------
    def _dist(self, a, b):
        if self._direct_terminal_index:
            return float(self.dist_matrix[a.id, b.id])
        return float(self.dist_matrix[self.node_id[a.id], self.node_id[b.id]])

    def _route_insert_positions(self, route, node, limit=None):
        candidates = []
        distance = self.dist_matrix
        if self._direct_terminal_index:
            node_index = node.id
            for pos in range(1, len(route.nodes)):
                prev_index = route.nodes[pos - 1].id
                next_index = route.nodes[pos].id
                delta = (
                    float(distance[prev_index, node_index])
                    + float(distance[node_index, next_index])
                    - float(distance[prev_index, next_index])
                )
                candidates.append((delta, pos))
        else:
            node_id = self.node_id
            node_index = node_id[node.id]
            for pos in range(1, len(route.nodes)):
                prev_index = node_id[route.nodes[pos - 1].id]
                next_index = node_id[route.nodes[pos].id]
                delta = (
                    float(distance[prev_index, node_index])
                    + float(distance[node_index, next_index])
                    - float(distance[prev_index, next_index])
                )
                candidates.append((delta, pos))
        candidates.sort(key=lambda x: x[0])
        if limit is not None and limit > 0:
            candidates = candidates[:limit]
        return candidates

    def _route_relatedness(self, route, node):
        best = float("inf")
        distance = self.dist_matrix
        if self._direct_terminal_index:
            distance_row = distance[node.id]
            for other in route.nodes:
                if other.id == node.id:
                    continue
                best = min(best, float(distance_row[other.id]))
        else:
            node_id = self.node_id
            distance_row = distance[node_id[node.id]]
            for other in route.nodes:
                if other.id == node.id:
                    continue
                best = min(best, float(distance_row[node_id[other.id]]))
        return best

    def _customer_positions(self, solution):
        positions = []
        for ridx, route in enumerate(solution):
            for pos in range(1, len(route.nodes) - 1):
                node = route.nodes[pos]
                if node.type == "c":
                    positions.append((ridx, pos, node))
        return positions

    def _effective_fast_limits(self, customer_count):
        """Scale proposal/evaluation work without changing public parameters."""
        scale = max(1, int(customer_count))
        if scale >= 500:
            multiplier = 1.5
            route_limit = min(self.route_neighbor_limit, 3)
            position_limit = min(self.position_neighbor_limit, 3)
            exchange_limit = min(self.exchange_neighbor_limit, 4)
        elif scale >= 100:
            multiplier = 1.75
            route_limit = min(self.route_neighbor_limit, 4)
            position_limit = min(self.position_neighbor_limit, 4)
            exchange_limit = min(self.exchange_neighbor_limit, 6)
        else:
            multiplier = 2.0
            route_limit = self.route_neighbor_limit
            position_limit = self.position_neighbor_limit
            exchange_limit = self.exchange_neighbor_limit
        candidate_limit = self.move_candidate_limit
        if candidate_limit > 0:
            candidate_limit = max(12, int(round(candidate_limit * multiplier)))
        return (
            max(1, int(route_limit)),
            max(1, int(position_limit)),
            max(0, int(exchange_limit)),
            candidate_limit,
        )

    @staticmethod
    def _bounded_insert(moves, item, limit):
        """Maintain a stable, bounded proxy list in O(limit), not O(N log N)."""
        if limit <= 0:
            moves.append(item)
            return
        if len(moves) < limit:
            moves.append(item)
            return
        worst_index = max(range(len(moves)), key=lambda index: (moves[index][0], index))
        if item[0] < moves[worst_index][0]:
            moves[worst_index] = item

    def _ranked_candidate_moves_fast(self, solution):
        """Build a bounded, relatedness-driven fast neighborhood."""
        moves = []
        customer_positions = self._customer_positions(solution)
        if not customer_positions:
            return moves

        (
            route_neighbor_limit,
            position_neighbor_limit,
            exchange_neighbor_limit,
            candidate_limit,
        ) = self._effective_fast_limits(len(customer_positions))
        position_by_customer_id = {
            node.id: (route_index, position, node)
            for route_index, position, node in customer_positions
        }
        route_customer_ids = []
        customer_route = {}
        for route_index, route in enumerate(solution):
            route_ids = []
            for node in route.nodes[1:-1]:
                if node.type == "c":
                    route_ids.append(node.id)
                    customer_route[node.id] = route_index
            route_customer_ids.append(route_ids)

        # Relocate moves: derive candidate routes from precomputed nearest
        # customers, then rank insertion positions only on those few routes.
        for ridx, pos, node in customer_positions:
            route_scores = [(0.0, ridx)]
            seen_routes = {ridx}
            for other_id in self._customer_neighbor_ids.get(node.id, ()):
                other_route = customer_route.get(other_id)
                if other_route is None or other_route in seen_routes:
                    continue
                route_scores.append((self._dist(node, self._customer_by_id[other_id]), other_route))
                seen_routes.add(other_route)
                if len(route_scores) >= route_neighbor_limit:
                    break

            for _, j in route_scores:
                route = solution[j]
                for proxy, insert_pos in self._route_insert_positions(
                    route, node, position_neighbor_limit
                ):
                    if ridx == j and (insert_pos == pos or insert_pos == pos + 1):
                        continue
                    self._bounded_insert(
                        moves,
                        (proxy, ("relocate", ridx, pos, j, insert_pos)),
                        candidate_limit,
                    )

            # Opening a route is useful during feasibility recovery, but keep it behind ordinary insertions.
            if len(solution[ridx].nodes) > 3:
                depot = self.instance.depot
                if self._direct_terminal_index:
                    depot_index, node_index = depot.id, node.id
                else:
                    depot_index = self.node_id[depot.id]
                    node_index = self.node_id[node.id]
                proxy = (
                    float(self.dist_matrix[depot_index, node_index])
                    + float(self.dist_matrix[node_index, depot_index])
                )
                self._bounded_insert(
                    moves,
                    (proxy + 1e-6, ("relocate_new", ridx, pos)),
                    candidate_limit,
                )

        distance = self.dist_matrix
        direct_terminal_index = self._direct_terminal_index
        node_id = self.node_id

        # Exchange moves: O(n*k) lookup into the immutable nearest-customer
        # ranks rather than rebuilding and sorting n complete neighbour lists.
        for i, p1, node1 in customer_positions:
            source_index = node1.id if direct_terminal_index else node_id[node1.id]
            distance_row = distance[source_index]
            nearest_ids = self._customer_neighbor_ids.get(node1.id, ())
            for other_id in nearest_ids[:exchange_neighbor_limit]:
                other_position = position_by_customer_id.get(other_id)
                if other_position is None:
                    continue
                j, p2, node2 = other_position
                if (j, p2, i, p1) < (i, p1, j, p2):
                    continue
                other_index = node2.id if direct_terminal_index else node_id[node2.id]
                proxy = float(distance_row[other_index])
                self._bounded_insert(
                    moves,
                    (proxy, ("exchange", i, p1, j, p2)),
                    candidate_limit,
                )

        # 2-opt*: restrict route pairs to those connected by at least one of the
        # precomputed nearest-customer relations.  Within a selected pair scan
        # all split combinations, retaining only its best few moves.
        route_pairs = set()
        for customer_id, i in customer_route.items():
            for other_id in self._customer_neighbor_ids.get(customer_id, ())[:route_neighbor_limit]:
                j = customer_route.get(other_id)
                if j is not None and i != j:
                    route_pairs.add((min(i, j), max(i, j)))
        for i, j in sorted(route_pairs):
            ri = solution[i]
            if len(ri.nodes) <= 3:
                continue
            rj = solution[j]
            if len(rj.nodes) <= 3:
                continue
            split_candidates = []
            for s1 in range(1, len(ri.nodes) - 1):
                for s2 in range(1, len(rj.nodes) - 1):
                    if direct_terminal_index:
                        ri_prev = ri.nodes[s1 - 1].id
                        ri_cur = ri.nodes[s1].id
                        rj_prev = rj.nodes[s2 - 1].id
                        rj_cur = rj.nodes[s2].id
                    else:
                        ri_prev = node_id[ri.nodes[s1 - 1].id]
                        ri_cur = node_id[ri.nodes[s1].id]
                        rj_prev = node_id[rj.nodes[s2 - 1].id]
                        rj_cur = node_id[rj.nodes[s2].id]
                    old = float(distance[ri_prev, ri_cur]) + float(distance[rj_prev, rj_cur])
                    new = float(distance[ri_prev, rj_cur]) + float(distance[rj_prev, ri_cur])
                    split_candidates.append((new - old, s1, s2))
            split_candidates.sort(key=lambda x: x[0])
            for proxy, s1, s2 in split_candidates[:position_neighbor_limit]:
                self._bounded_insert(
                    moves,
                    (proxy, ("two_opt", i, j, s1, s2)),
                    candidate_limit,
                )

        # Station remove/insert moves. Insertions focus on the first battery violation arc.
        for ridx, route in enumerate(solution):
            for pos in range(1, len(route.nodes) - 1):
                if route.nodes[pos].type == "f":
                    prev_node = route.nodes[pos - 1]
                    next_node = route.nodes[pos + 1]
                    if direct_terminal_index:
                        prev_index = prev_node.id
                        station_index = route.nodes[pos].id
                        next_index = next_node.id
                    else:
                        prev_index = node_id[prev_node.id]
                        station_index = node_id[route.nodes[pos].id]
                        next_index = node_id[next_node.id]
                    proxy = (
                        float(distance[prev_index, next_index])
                        - float(distance[prev_index, station_index])
                        - float(distance[station_index, next_index])
                    )
                    self._bounded_insert(
                        moves,
                        (proxy, ("station_remove", ridx, pos)),
                        candidate_limit,
                    )

            sim_fail = None
            fuel_cap = float(self.instance.vehicle_params["fuel_cap"])
            current_fuel = fuel_cap
            for pos in range(len(route.nodes) - 1):
                a, b = route.nodes[pos], route.nodes[pos + 1]
                current_fuel -= self.fuel_consumption(a, b)
                if current_fuel < -NUMERICAL_EPS:
                    sim_fail = pos
                    break
                if b.type == "f":
                    current_fuel = fuel_cap
            if sim_fail is not None:
                a, b = route.nodes[sim_fail], route.nodes[sim_fail + 1]
                station_candidates = []
                for st in self.instance.stations:
                    if st.id in {a.id, b.id}:
                        continue
                    if self.fuel_consumption(a, st) > fuel_cap + 1e-9:
                        continue
                    if self.fuel_consumption(st, b) > fuel_cap + 1e-9:
                        continue
                    if direct_terminal_index:
                        a_index, station_index, b_index = a.id, st.id, b.id
                    else:
                        a_index = node_id[a.id]
                        station_index = node_id[st.id]
                        b_index = node_id[b.id]
                    detour = (
                        float(distance[a_index, station_index])
                        + float(distance[station_index, b_index])
                        - float(distance[a_index, b_index])
                    )
                    station_candidates.append((detour, st))
                station_candidates.sort(key=lambda x: x[0])
                for proxy, st in station_candidates[: max(0, self.station_candidate_limit)]:
                    self._bounded_insert(
                        moves,
                        (proxy, ("station_insert", ridx, sim_fail + 1, st)),
                        candidate_limit,
                    )

        moves.sort(key=lambda x: x[0])
        if candidate_limit > 0:
            moves = moves[:candidate_limit]
        return moves

    def _candidate_moves_fast(self, solution):
        return [move for _, move in self._ranked_candidate_moves_fast(solution)]

    def _apply_fast_move(self, solution, move):
        candidate = self.clone_solution_shallow(solution)
        mtype = move[0]

        try:
            if mtype == "relocate":
                _, i, pos, j, insert_pos = move
                if i >= len(candidate) or j >= len(candidate):
                    return None
                node = candidate[i].nodes.pop(pos)
                adj_insert = insert_pos
                if i == j and insert_pos > pos:
                    adj_insert -= 1
                adj_insert = max(1, min(adj_insert, len(candidate[j].nodes)))
                candidate[j].nodes.insert(adj_insert, node)

            elif mtype == "relocate_new":
                _, i, pos = move
                if i >= len(candidate):
                    return None
                node = candidate[i].nodes.pop(pos)
                nr = self.create_new_route()
                nr.nodes.append(node)
                nr.nodes.append(self.instance.depot)
                candidate.append(nr)

            elif mtype == "exchange":
                _, i, p1, j, p2 = move
                if i >= len(candidate) or j >= len(candidate):
                    return None
                candidate[i].nodes[p1], candidate[j].nodes[p2] = candidate[j].nodes[p2], candidate[i].nodes[p1]

            elif mtype == "two_opt":
                _, i, j, s1, s2 = move
                if i >= len(candidate) or j >= len(candidate):
                    return None
                old_i = candidate[i].nodes
                old_j = candidate[j].nodes
                candidate[i].nodes = old_i[:s1] + old_j[s2:]
                candidate[j].nodes = old_j[:s2] + old_i[s1:]

            elif mtype == "station_remove":
                _, i, pos = move
                if i >= len(candidate):
                    return None
                if candidate[i].nodes[pos].type != "f":
                    return None
                candidate[i].nodes.pop(pos)

            elif mtype == "station_insert":
                _, i, pos, station = move
                if i >= len(candidate):
                    return None
                pos = max(1, min(pos, len(candidate[i].nodes)))
                candidate[i].nodes.insert(pos, station)

            else:
                return None
        except (IndexError, ValueError):
            return None

        return self.solution_fix(candidate)

    def _move_tabu_key(self, move):
        mtype = move[0]
        if mtype in {"relocate", "relocate_new"}:
            return (mtype, move[1], move[2])
        if mtype == "exchange":
            return (mtype, min((move[1], move[2]), (move[3], move[4])), max((move[1], move[2]), (move[3], move[4])))
        if mtype == "two_opt":
            return (mtype, move[1], move[2], move[3], move[4])
        if mtype == "station_insert":
            return (mtype, move[1], move[2], getattr(move[3], "id", None))
        return tuple(move[:3])

    def _tabu_search_fast(self, S):
        current_solution = self.clone_solution_shallow(S)
        best_solution = self.clone_solution_shallow(S)
        best_solution_value = None
        tabu_list = deque(maxlen=self.tabu_tenure)

        for _iter in range(self.tabu_iter):
            if self._time_limit_reached():
                break
            self._decay_station_tabu()
            moves = self._candidate_moves_fast(current_solution)
            if not moves:
                break

            best_candidate = None
            best_key = None
            best_cost = float("inf")
            best_feasible_value = float("inf")
            route_feasibility_cache = {}

            for move in moves:
                key = self._move_tabu_key(move)
                candidate = self._apply_fast_move(current_solution, move)
                if candidate is None or not candidate:
                    continue

                feasible_value = self.generalized_cost(
                    candidate,
                    penalty_value=False,
                    p_div_value=False,
                    allow_infeasible=False,
                    route_feasibility_cache=route_feasibility_cache,
                )
                candidate_feasible = math.isfinite(feasible_value) and feasible_value < 1e9
                is_aspiration = candidate_feasible and feasible_value < self.global_value
                if key in tabu_list and not is_aspiration:
                    continue

                if candidate_feasible:
                    cost = feasible_value
                else:
                    cost = self.generalized_cost(candidate, penalty_value=True, p_div_value=False, allow_infeasible=True)

                if cost < best_cost or (math.isclose(cost, best_cost) and feasible_value < best_feasible_value):
                    best_cost = cost
                    best_candidate = candidate
                    best_key = key
                    best_feasible_value = feasible_value

            if best_candidate is None:
                break

            current_solution = self.solution_fix(best_candidate)
            if best_key is not None:
                tabu_list.append(best_key)
            self.update_diversification_history(current_solution)

            cur_val = best_feasible_value
            if best_solution_value is None:
                best_solution_value = self.generalized_cost(
                    best_solution,
                    penalty_value=False,
                    p_div_value=False,
                    allow_infeasible=False,
                )
            if cur_val < best_solution_value:
                best_solution = self.clone_solution_shallow(current_solution)
                best_solution_value = cur_val

            if best_solution_value < self.global_value:
                self.global_value = best_solution_value
                self.global_solution = self.clone_solution_shallow(best_solution)
                self._report_incumbent()

        return best_solution

    # -------------------------
    # Public solve()
    # -------------------------
    def apply_tabu_search(self, S_prime):
        if self.search_mode == "full":
            return self._tabu_search(S_prime)
        return self._tabu_search_fast(S_prime)

    def solve(
        self,
        time_limit_s: Optional[float] = None,
        incumbent_callback: Optional[Callable[[float, float, list[list[int]]], None]] = None,
    ):
        self._solve_start = time.perf_counter()
        self._deadline = (
            None
            if time_limit_s is None
            else self._solve_start + max(0.0, float(time_limit_s))
        )
        self._incumbent_callback = incumbent_callback
        self._last_reported_value = None
        self.terminated_by_time_limit = False
        warm_start = self.singleton_warm_start()
        if warm_start is not None:
            self.global_solution = self.clone_solution_shallow(warm_start)
            self.global_value = self.generalized_cost(
                warm_start,
                penalty_value=False,
                p_div_value=False,
                allow_infeasible=False,
            )
            self._report_incumbent()

            S = self._consolidate_singleton_solution(
                warm_start,
                deadline=self._deadline,
            )
        else:
            S = self.initial_solution()
        initial_value = self.generalized_cost(
            S,
            penalty_value=False,
            p_div_value=False,
            allow_infeasible=False,
        )
        if initial_value < self.global_value:
            self.global_solution = self.clone_solution_shallow(S)
            self.global_value = initial_value
            self._report_incumbent()

        κ = 1
        i = 0
        feasibilityPhase = not self.is_solution_feasible(S)

        pbar = tqdm(total=self.η_dist + self.η_feas, disable=not self.show_progress)

        while (feasibilityPhase or (not feasibilityPhase and i < self.η_dist)) and not self._time_limit_reached():
            S_prime = self.vns_perturb(S, κ)
            S_double = self.apply_tabu_search(S_prime)

            if self.accept_sa(S_double, S):
                S = self.clone_solution_shallow(S_double)
                κ = 1
            else:
                κ = (κ % self.k_max) + 1

            # update best / global
            solution_feasible = self.is_solution_feasible(S)
            if solution_feasible:
                val = self.generalized_cost(
                    S,
                    penalty_value=False,
                    p_div_value=False,
                    allow_infeasible=True,
                )
                if val < self.global_value:
                    self.global_value = val
                    self.global_solution = self.clone_solution_shallow(S)
                    self._report_incumbent()

            if feasibilityPhase:
                if not solution_feasible:
                    if i == self.η_feas:
                        S = self.add_vehicle(S)
                        i -= 1
                else:
                    feasibilityPhase = False
                    i = 0
                    pbar.reset(total=self.η_dist)

            self.update_penalty_weights(S, i)
            i += 1
            pbar.update(1)

            # Tabu search can improve global_solution internally. Reporting at
            # the outer-iteration boundary gives the improvement a conservative
            # timestamp and therefore cannot leak it into an earlier checkpoint.
            self._report_incumbent()

        pbar.close()
        return self.global_solution

    # -------------------------
    # add_vehicle / violates_constraints (kept close to your original)
    # -------------------------
    def copy_route_deep_nodes(self, route):
        """If you still want a deep node copy in add_vehicle, keep it here."""
        new_route = Route()
        new_route.nodes = copy.deepcopy(route.nodes)
        if hasattr(route, "load"):
            new_route.load = route.load
        if hasattr(route, "time"):
            new_route.time = route.time
        if hasattr(route, "fuel"):
            new_route.fuel = route.fuel
        return new_route

    def violates_constraints(self, route, search_idx):
        new_route = self.clone_route_shallow(route)
        new_route.nodes = new_route.nodes[: search_idx + 1]
        if new_route.nodes[-1].type != "d":
            new_route.nodes.append(self.instance.depot)
        return self.battery_violation(new_route) or self.time_violation(new_route) or self.load_violation(new_route)

    def add_vehicle(self, S):
        new_routes = []
        candidate_customers = []

        for route in S:
            if self.is_route_feasible(route):
                new_routes.append(route)
                continue

            route_add = self.clone_route_shallow(route)
            route_len = len(route_add.nodes)
            idx = 1

            while idx < route_len - 1 and not self.is_route_feasible(route_add):
                cur = route_add.nodes[idx]
                if self.violates_constraints(route_add, idx):
                    if cur.type == "c":
                        candidate_customers.append(cur)
                    route_add.nodes.pop(idx)
                    route_len -= 1
                else:
                    idx += 1

            if len(route_add.nodes) > 2:
                new_routes.append(route_add)

        candidate_routes = []
        while candidate_customers:
            node = candidate_customers.pop()
            inserted = False

            for r in candidate_routes:
                for pos in reversed(range(1, len(r.nodes))):
                    r.nodes.insert(pos, node)
                    if self.is_route_feasible(r):
                        inserted = True
                        break
                    r.nodes.pop(pos)
                if inserted:
                    break

            if not inserted:
                rnew = self.create_new_route()
                rnew.nodes.append(node)
                rnew.nodes.append(self.instance.depot)
                candidate_routes.append(rnew)

        new_routes.extend(candidate_routes)
        return new_routes

    # -------------------------
    # Debug printing
    # -------------------------
    def print_solution(self, solution):
        res = []
        for r in solution:
            res.append(" -> ".join(str(n.id) for n in r.nodes))
        res.sort()
        print(" | ".join(res))
