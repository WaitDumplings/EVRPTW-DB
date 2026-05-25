from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evrptw_hierarchy.core.models import ActiveInstance, RegionBoard, VehicleConfig
from evrptw_hierarchy.graph.distance_oracle import DistanceOracle
from evrptw_hierarchy.graph.shortest_path import floyd_warshall_time
from evrptw_hierarchy.sampling.cs_activation import activate_charging_stations
from evrptw_hierarchy.sampling.daily_models import (
    sample_day_profile,
    sample_demands_cm3,
    sample_service_time_min,
    sample_time_windows,
)


@dataclass
class ActiveDaySampler:
    config: dict[str, Any]
    vehicle: VehicleConfig
    rng: np.random.Generator
    last_active_sampling_metadata: dict[str, Any] = field(default_factory=dict, init=False)
    _board_sampling_cache: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def _sampling_cache_for(self, board: RegionBoard) -> dict[str, Any]:
        key = f"{board.region_id}:{board.mother_board_id}:{len(board.customers)}:{len(board.charging_stations)}"
        cache = self._board_sampling_cache.get(key)
        if cache is not None:
            return cache
        cluster_labels = np.asarray(board.cluster_labels, dtype=np.int32)
        micro_zone_labels = np.asarray(board.micro_zone_labels, dtype=np.int32)
        cluster_count = int(cluster_labels.max()) + 1
        micro_count = int(micro_zone_labels.max()) + 1
        cluster_indices = [np.flatnonzero(cluster_labels == idx).astype(np.int32) for idx in range(cluster_count)]
        cache = {
            "cluster_labels": cluster_labels,
            "micro_zone_labels": micro_zone_labels,
            "cluster_count": cluster_count,
            "micro_count": micro_count,
            "cluster_sizes": np.bincount(cluster_labels, minlength=cluster_count).astype(float),
            "cluster_indices": cluster_indices,
        }
        self._board_sampling_cache[key] = cache
        return cache

    def _target_customers_per_active_cluster(self, cfg: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        target = float(cfg.get("target_customers_per_active_cluster", 130.0))
        meta: dict[str, Any] = {
            "target_source": "fixed",
            "target_customers_per_active_cluster": float(target),
            "calibration_role": "manual_active_community_size",
            "uses_route_count_as_vehicle_target": False,
            "uses_sequence_or_realized_route_outcome": False,
        }
        route_cfg = cfg.get("community_size_calibration", {}) or cfg.get("route_size_calibration", {})
        if bool(route_cfg.get("enabled", False)):
            median = float(route_cfg.get("median_customers_per_route", target))
            sigma = float(route_cfg.get("lognormal_sigma", 0.0))
            low = float(route_cfg.get("min_customers_per_route", max(1.0, median * 0.4)))
            high = float(route_cfg.get("max_customers_per_route", max(low, median * 2.0)))
            sampled = float(self.rng.lognormal(math.log(max(median, 1e-9)), max(sigma, 0.0)))
            target = float(np.clip(sampled, low, high))
            meta = {
                "target_source": "amazon_community_size_lognormal",
                "target_customers_per_active_cluster": float(target),
                "calibration_role": "input_side_active_community_size_proxy",
                "uses_route_count_as_vehicle_target": False,
                "uses_sequence_or_realized_route_outcome": False,
                "sampled_customers_per_route_before_clip": float(sampled),
                "median_customers_per_route": float(median),
                "lognormal_sigma": float(sigma),
                "min_customers_per_route": float(low),
                "max_customers_per_route": float(high),
                "source": str(route_cfg.get("source", "")),
                "statistic": str(route_cfg.get("statistic", "")),
            }
        return target, meta

    def sample_active_customers(self, board: RegionBoard, num_customers: int) -> np.ndarray:
        n = len(board.customers)
        if num_customers > n:
            raise ValueError(f"num_customers={num_customers} exceeds mother board customer pool {n}.")
        cache = self._sampling_cache_for(board)
        cluster_labels = cache["cluster_labels"]
        micro_zone_labels = cache["micro_zone_labels"]
        cluster_count = int(cache["cluster_count"])
        cluster_sizes = cache["cluster_sizes"]
        cluster_indices = cache["cluster_indices"]
        cfg = self.config.get("active_customer_sampling", {})
        cluster_sigma = float(cfg.get("macro_activity_lognormal_sigma", 0.8))
        micro_sigma = float(cfg.get("micro_activity_lognormal_sigma", 0.5))
        cluster_activity = self.rng.lognormal(0.0, cluster_sigma, size=cluster_count)
        micro_activity = self.rng.lognormal(0.0, micro_sigma, size=int(cache["micro_count"]))

        selected_clusters: list[int] = []
        target, sampling_meta = self._target_customers_per_active_cluster(cfg)
        if bool(cfg.get("sample_active_cluster_subset", True)):
            sqrt_divisor = float(cfg.get("small_scale_cluster_sqrt_divisor", 4.0))
            min_by_sqrt = int(math.ceil(math.sqrt(max(num_customers, 1)) / max(sqrt_divisor, 1e-9)))
            if num_customers <= 1:
                min_by_sparse_day = 1
            else:
                log_base = float(cfg.get("sparse_day_log_cluster_base", 2.0))
                log_scale = float(cfg.get("sparse_day_log_cluster_scale", 1.0))
                min_by_sparse_day = int(math.ceil(log_scale * math.log(num_customers + 1, max(log_base, 1.0001))))
                min_by_sparse_day = max(int(cfg.get("small_scale_min_active_clusters", 2)), min_by_sparse_day)
                min_by_sparse_day = min(min_by_sparse_day, int(num_customers))
            min_by_scale = max(min_by_sqrt, min_by_sparse_day)
            active_cluster_target = max(1, int(math.ceil(num_customers / max(target, 1.0))), min_by_scale)
            active_cluster_target = min(active_cluster_target, cluster_count, int(num_customers))
            cluster_weights = cluster_activity * np.maximum(cluster_sizes, 1.0)
            cluster_weights = cluster_weights / max(float(cluster_weights.sum()), 1e-12)
            available = list(range(cluster_count))
            selected_capacity = 0.0
            capacity_buffer = float(cfg.get("active_cluster_capacity_buffer", 1.08))
            while available and (len(selected_clusters) < active_cluster_target or selected_capacity < num_customers * capacity_buffer):
                probs = cluster_weights[np.asarray(available, dtype=int)]
                probs = probs / max(float(probs.sum()), 1e-12)
                pick_pos = int(self.rng.choice(len(available), p=probs))
                pick = available.pop(pick_pos)
                selected_clusters.append(pick)
                selected_capacity += cluster_sizes[pick]
            cluster_mask = np.zeros(n, dtype=bool)
            for cluster_id in selected_clusters:
                cluster_mask[cluster_indices[int(cluster_id)]] = True
            sampling_meta.update({
                "active_cluster_target": int(active_cluster_target),
                "selected_cluster_count": int(len(selected_clusters)),
                "selected_cluster_capacity": float(selected_capacity),
                "capacity_buffer": float(capacity_buffer),
                "cluster_count": int(cluster_count),
            })
        else:
            cluster_mask = np.ones(n, dtype=bool)
            sampling_meta.update({
                "active_cluster_target": int(cluster_count),
                "selected_cluster_count": int(cluster_count),
                "selected_cluster_capacity": int(n),
                "capacity_buffer": 1.0,
                "cluster_count": int(cluster_count),
            })

        base_weights = cluster_activity[cluster_labels] * micro_activity[micro_zone_labels]
        weights = base_weights * cluster_mask.astype(float)
        if np.count_nonzero(weights) < num_customers:
            weights = base_weights
            selected_clusters = []
            sampling_meta["fallback_to_full_region"] = True
        else:
            sampling_meta["fallback_to_full_region"] = False

        weights = weights / max(float(weights.sum()), 1e-12)
        self.last_active_sampling_metadata = sampling_meta
        if not bool(cfg.get("ensure_selected_cluster_coverage", True)) or not selected_clusters:
            return self.rng.choice(n, size=int(num_customers), replace=False, p=weights).astype(np.int32)

        chosen: list[int] = []
        selected_for_coverage = selected_clusters[: min(len(selected_clusters), int(num_customers))]
        for cluster_id in selected_for_coverage:
            candidates = cluster_indices[int(cluster_id)]
            candidates = candidates[weights[candidates] > 0]
            if candidates.size == 0:
                continue
            probs = weights[candidates]
            probs = probs / max(float(probs.sum()), 1e-12)
            chosen.append(int(self.rng.choice(candidates, p=probs)))

        chosen = list(dict.fromkeys(chosen))
        remaining = int(num_customers) - len(chosen)
        if remaining <= 0:
            return np.asarray(chosen[: int(num_customers)], dtype=np.int32)
        remaining_weights = weights.copy()
        if chosen:
            remaining_weights[np.asarray(chosen, dtype=int)] = 0.0
        if np.count_nonzero(remaining_weights) < remaining:
            available_mask = np.ones(n, dtype=bool)
            if chosen:
                available_mask[np.asarray(chosen, dtype=int)] = False
            pool = np.flatnonzero(available_mask)
            extra = self.rng.choice(pool, size=remaining, replace=False)
        else:
            remaining_weights = remaining_weights / max(float(remaining_weights.sum()), 1e-12)
            extra = self.rng.choice(n, size=remaining, replace=False, p=remaining_weights)
        out = np.concatenate([np.asarray(chosen, dtype=np.int32), np.asarray(extra, dtype=np.int32)])
        self.rng.shuffle(out)
        return out.astype(np.int32)

    def _active_distance_matrix(
        self,
        board: RegionBoard,
        active_customer_ids: np.ndarray,
        active_cs_ids: np.ndarray,
        oracle: DistanceOracle,
    ) -> tuple[np.ndarray, np.ndarray]:
        terminal_node_ids = np.concatenate([
            np.asarray([board.depot_node_id], dtype=np.int32),
            board.customer_node_ids[active_customer_ids],
            board.cs_node_ids[active_cs_ids],
        ])
        dist = oracle.matrix(terminal_node_ids)
        return dist, terminal_node_ids

    def _shortest_time_matrix(
        self,
        distance_matrix_km: np.ndarray,
        num_customers: int,
        effective_speed_kmh: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_time = np.asarray(distance_matrix_km, dtype=np.float32) / max(float(effective_speed_kmh), 1e-12) * 60.0
        n = raw_time.shape[0]
        battery_range = self.vehicle.battery_range_km
        feasible = np.asarray(distance_matrix_km) <= battery_range + 1e-6
        transition = np.where(feasible, raw_time, np.inf).astype(np.float32)
        np.fill_diagonal(transition, 0.0)
        if num_customers + 1 < n:
            transition[num_customers + 1 :, :] += self.vehicle.full_charge_time_min
            np.fill_diagonal(transition, 0.0)

        if bool(np.all(np.isfinite(distance_matrix_km)) and np.all(feasible)):
            # When all active terminals are mutually battery-reachable, detouring
            # through charging stations cannot improve travel time because full
            # charging time is positive. This is the common EDV-700/local-region
            # case and keeps Cus1800 generation O(n^2).
            return raw_time.astype(np.float32), transition, transition.copy()

        max_floyd_nodes = int(self.config.get("time_matrix", {}).get("max_floyd_nodes", 700))
        if n <= max_floyd_nodes:
            shortest = floyd_warshall_time(transition)
        else:
            shortest = self._dense_terminal_dijkstra_apsp(transition)
        return raw_time.astype(np.float32), transition, shortest

    def _dense_terminal_dijkstra_apsp(self, transition: np.ndarray) -> np.ndarray:
        cost = np.asarray(transition, dtype=np.float64)
        n = cost.shape[0]
        out = np.empty((n, n), dtype=np.float32)
        for source in range(n):
            dist = cost[source].copy()
            dist[source] = 0.0
            visited = np.zeros(n, dtype=bool)
            for _ in range(n):
                masked = np.where(visited, np.inf, dist)
                u = int(np.argmin(masked))
                if not np.isfinite(masked[u]):
                    break
                visited[u] = True
                cand = dist[u] + cost[u]
                better = cand < dist
                dist[better] = cand[better]
            out[source] = dist.astype(np.float32)
        return out

    def _cs_time_to_depot(self, transition: np.ndarray, num_customers: int) -> np.ndarray:
        # Legacy-compatible semantics: each active CS is assumed full at
        # departure, so the first CS -> next-node leg does not pay a charging
        # time. Later departures from intermediate CS nodes still pay the full
        # charging time encoded in ``transition``.
        cost = np.asarray(transition, dtype=np.float64)
        cs_start = int(num_customers) + 1
        out = np.empty(max(cost.shape[0] - cs_start, 0), dtype=np.float32)
        full_charge = float(self.vehicle.full_charge_time_min)
        if out.size == 0:
            return out
        if np.all(np.isfinite(cost)):
            # In the common local-delivery case all active terminals are mutually
            # battery-reachable. Road shortest-path distances satisfy triangle
            # inequality and every intermediate CS detour adds nonnegative full
            # charge time, so direct CS -> depot is optimal.
            return np.maximum(0.0, cost[cs_start:, 0] - full_charge).astype(np.float32)
        for local, source in enumerate(range(cs_start, cost.shape[0])):
            dist = cost[source].copy()
            finite = np.isfinite(dist)
            dist[finite] = np.maximum(0.0, dist[finite] - full_charge)
            dist[source] = 0.0
            visited = np.zeros(cost.shape[0], dtype=bool)
            for _ in range(cost.shape[0]):
                masked = np.where(visited, np.inf, dist)
                u = int(np.argmin(masked))
                if not np.isfinite(masked[u]):
                    break
                visited[u] = True
                cand = dist[u] + cost[u]
                better = cand < dist
                dist[better] = cand[better]
            out[local] = float(dist[0])
        return out

    def build_instance(
        self,
        board: RegionBoard,
        usage_index: int,
        instance_index: int,
        num_customers: int,
        num_charging_stations: int,
        max_attempts: int,
        oracle: DistanceOracle,
    ) -> ActiveInstance:
        last_error = "not_started"
        for attempt in range(int(max_attempts)):
            active_customer_ids = self.sample_active_customers(board, num_customers)
            day = sample_day_profile(self.config, self.rng)
            effective_speed = self.vehicle.design_speed_kmh * float(day["congestion_factor"])
            active_cs_ids, cs_meta = activate_charging_stations(
                board, active_customer_ids, num_charging_stations, oracle, self.rng, self.config
            )
            distance_matrix, terminal_node_ids = self._active_distance_matrix(board, active_customer_ids, active_cs_ids, oracle)
            raw_time, transition, shortest_time = self._shortest_time_matrix(distance_matrix, num_customers, effective_speed)
            depot_to_customer = shortest_time[0, 1:1 + num_customers]
            customer_to_depot = shortest_time[1:1 + num_customers, 0]
            if not np.all(np.isfinite(depot_to_customer)) or not np.all(np.isfinite(customer_to_depot)):
                last_error = "active_customers_not_reachable_with_selected_cs"
                continue

            demands_cm3, package_counts = sample_demands_cm3(self.config, num_customers, self.rng)
            service_time_min = sample_service_time_min(self.config, demands_cm3, package_counts, self.rng)
            tw, tw_meta = sample_time_windows(
                self.config,
                str(day["day_type"]),
                int(day["working_start_min"]),
                int(day["working_end_min"]),
                depot_to_customer,
                customer_to_depot,
                service_time_min,
                self.rng,
            )
            feasible_span = (day["working_end_min"] - day["working_start_min"]) - depot_to_customer - customer_to_depot - service_time_min
            if np.any(feasible_span < -1e-6):
                last_error = "insufficient_working_horizon"
                continue

            audit = self._greedy_audit(shortest_time, demands_cm3, service_time_min, tw, float(day["working_start_min"]), float(day["working_end_min"]))
            if not bool(audit["success"]):
                last_error = str(audit.get("error", "greedy_audit_failed"))
                continue
            cs_time_to_depot = self._cs_time_to_depot(transition, num_customers)
            storage_cfg = self.config.get("storage", {})
            save_raw_time = bool(storage_cfg.get("save_raw_travel_time_matrix", False))
            save_transition = bool(storage_cfg.get("save_ev_transition_time_matrix", False))
            save_shortest = bool(storage_cfg.get("save_shortest_time_matrix", False))
            return ActiveInstance(
                instance_id=f"instance_{instance_index:06d}",
                region_id=board.region_id,
                mother_board_id=board.mother_board_id,
                operating_day_id=f"{board.region_id}_day_{usage_index:05d}",
                day_type=str(day["day_type"]),
                working_start_s=int(day["working_start_min"]) * 60,
                working_end_s=int(day["working_end_min"]) * 60,
                active_customer_ids=active_customer_ids,
                active_cs_ids=active_cs_ids,
                depot=board.road_nodes[board.depot_node_id].astype(np.float32),
                customers=board.customers[active_customer_ids].astype(np.float32),
                charging_stations=board.charging_stations[active_cs_ids].astype(np.float32),
                distance_matrix_km=distance_matrix.astype(np.float32),
                raw_travel_time_matrix_s=(raw_time * 60.0).astype(np.float32) if save_raw_time else None,
                ev_transition_time_matrix_s=(transition * 60.0).astype(np.float32) if save_transition else None,
                shortest_time_matrix_s=(shortest_time * 60.0).astype(np.float32) if save_shortest else None,
                cs_time_to_depot_s=(cs_time_to_depot * 60.0).astype(np.float32),
                demands_cm3=demands_cm3.astype(np.float32),
                package_counts=package_counts.astype(np.int32),
                service_time_s=(service_time_min * 60.0).astype(np.float32),
                tw_s=(tw * 60.0).astype(np.float32),
                vehicle={
                    "name": self.vehicle.name,
                    "battery_capacity_kwh": self.vehicle.battery_capacity_kwh,
                    "consumption_kwh_per_km": self.vehicle.consumption_kwh_per_km,
                    "cargo_capacity_cm3": self.vehicle.cargo_capacity_cm3,
                    "charging_power_kw": self.vehicle.charging_power_kw,
                    "full_charge_time_s": self.vehicle.full_charge_time_min * 60.0,
                },
                speed_profile={
                    "design_speed_kmh": self.vehicle.design_speed_kmh,
                    "congestion_factor": float(day["congestion_factor"]),
                    "effective_speed_kmh": float(effective_speed),
                },
                cs_activation=cs_meta,
                greedy_audit=audit,
                metadata={
                    "generation_attempt": attempt + 1,
                    "terminal_node_ids": terminal_node_ids.astype(np.int32),
                    "time_window_metadata": tw_meta,
                    "active_customer_sampling": self.last_active_sampling_metadata,
                    "last_error_before_success": last_error if attempt else "",
                    "charging_policy": "full_charge",
                    "saved_time_unit": "seconds",
                    "internal_generation_time_unit": "minutes",
                    "time_matrix_storage": {
                        "distance_matrix_km": True,
                        "raw_travel_time_matrix_s": save_raw_time,
                        "ev_transition_time_matrix_s": save_transition,
                        "shortest_time_matrix_s": save_shortest,
                    },
                },
            )
        raise RuntimeError(f"Could not generate feasible active-day instance after {max_attempts} attempts: {last_error}")

    def _greedy_audit(self, shortest_time: np.ndarray, demands_cm3: np.ndarray, service_time_min: np.ndarray, tw: np.ndarray, working_start_min: float, working_end_min: float) -> dict[str, Any]:
        n = len(demands_cm3)
        if n == 0:
            return {
                "success": True,
                "served_customers": 0,
                "unserved_customers": 0,
                "vehicle_upper_bound": 0,
                "vehicle_count": 0,
                "total_route_time_proxy_s": 0.0,
                "total_load_cm3": 0.0,
                "error": "",
            }

        shortest = np.asarray(shortest_time, dtype=np.float64)
        demands = np.asarray(demands_cm3, dtype=np.float64)
        service = np.asarray(service_time_min, dtype=np.float64)
        tw_start = np.asarray(tw[:, 0], dtype=np.float64)
        tw_end = np.asarray(tw[:, 1], dtype=np.float64)
        unvisited = np.ones(n, dtype=bool)
        routes = 0
        total_time = 0.0
        total_load = 0.0
        capacity = float(self.vehicle.cargo_capacity_cm3)
        eps = 1e-6

        while np.any(unvisited):
            routes += 1
            cur = 0
            clock = float(working_start_min)
            load = 0.0
            progressed = False
            while True:
                remaining = np.flatnonzero(unvisited)
                if remaining.size == 0:
                    if cur != 0 and np.isfinite(shortest[cur, 0]):
                        total_time += float(shortest[cur, 0])
                    break

                nodes = remaining + 1
                travel = shortest[cur, nodes]
                back = shortest[nodes, 0]
                service_start = np.maximum(clock + travel, tw_start[remaining])
                finish = service_start + service[remaining]
                feasible = (
                    (load + demands[remaining] <= capacity)
                    & np.isfinite(travel)
                    & np.isfinite(back)
                    & (service_start <= tw_end[remaining] + eps)
                    & (finish + back <= float(working_end_min) + eps)
                )
                if not np.any(feasible):
                    if cur != 0 and np.isfinite(shortest[cur, 0]):
                        total_time += float(shortest[cur, 0])
                    break

                feasible_remaining = remaining[feasible]
                feasible_arrival = service_start[feasible]
                best = int(feasible_remaining[int(np.argmin(feasible_arrival))])
                node = 1 + best
                travel_to_best = float(shortest[cur, node])
                clock = max(clock + travel_to_best, float(tw_start[best])) + float(service[best])
                load += float(demands[best])
                total_load += float(demands[best])
                unvisited[best] = False
                cur = node
                progressed = True

            if not progressed:
                return {
                    "success": False,
                    "served_customers": int(n - int(np.count_nonzero(unvisited))),
                    "unserved_customers": int(np.count_nonzero(unvisited)),
                    "vehicle_upper_bound": None,
                    "vehicle_count": int(routes),
                    "error": "no_feasible_next_customer",
                }
        return {
            "success": True,
            "served_customers": int(n),
            "unserved_customers": 0,
            "vehicle_upper_bound": int(routes),
            "vehicle_count": int(routes),
            "total_route_time_proxy_s": float(total_time * 60.0),
            "total_load_cm3": float(total_load),
            "error": "",
        }
