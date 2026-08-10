"""Edge-aware terminal matrices over a directed CLE road graph."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .road_state import auxiliary_power_kw, connector_costs

NO_PREDECESSOR = -9999
RoutePolicy = Literal["distance", "running_time"]


def _bearing_degrees(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(
        math.radians, (lon1, lat1, lon2, lat2)
    )
    delta_lon = lon2_rad - lon1_rad
    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(
        lat2_rad
    ) * math.cos(delta_lon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


@dataclass(frozen=True)
class AccessOption:
    edge_index: int
    edge_key: tuple[str, str, str]
    outbound_node: int
    inbound_node: int
    offset_from_u_m: float
    offset_to_v_m: float
    reference_length_m: float


@dataclass(frozen=True)
class TerminalAccess:
    terminal_index: int
    connector_distance_m: float
    connector_time_s: float
    connector_energy_kwh: float
    options: tuple[AccessOption, ...]


@dataclass(frozen=True)
class RoutingMatrices:
    distance_matrix_km: np.ndarray
    distance_path_travel_time_s: np.ndarray
    distance_path_energy_kwh: np.ndarray
    running_time_shortest_matrix_s: np.ndarray
    running_time_path_distance_km: np.ndarray
    running_time_path_energy_kwh: np.ndarray
    distance_source_option: np.ndarray
    distance_destination_option: np.ndarray
    running_time_source_option: np.ndarray
    running_time_destination_option: np.ndarray
    report: dict[str, Any]


class PhysicalRoadNetwork:
    """Static topology plus one family-level speed/energy realization."""

    def __init__(self, graph: nx.MultiDiGraph, road_state: pd.DataFrame, profile: dict[str, Any]):
        if not graph.is_directed():
            raise ValueError("CLE routing graph must be directed")
        self.graph = graph
        self.profile = profile
        self.node_ids = tuple(sorted(map(str, graph.nodes)))
        self.node_to_index = {node: index for index, node in enumerate(self.node_ids)}
        frame = road_state.copy().reset_index(drop=True)
        required = {
            "edge_u",
            "edge_v",
            "edge_key",
            "length_m",
            "edge_travel_time_s",
            "edge_energy_kwh",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Road state is missing routing columns: {sorted(missing)}")
        frame["edge_u"] = frame["edge_u"].astype(str)
        frame["edge_v"] = frame["edge_v"].astype(str)
        frame["edge_key"] = frame["edge_key"].astype(str)
        missing_nodes = sorted(
            (set(frame["edge_u"]) | set(frame["edge_v"])) - set(self.node_to_index)
        )
        if missing_nodes:
            raise ValueError(f"Road-state edges reference {len(missing_nodes)} absent graph nodes")
        frame["u_index"] = frame["edge_u"].map(self.node_to_index).astype(np.int32)
        frame["v_index"] = frame["edge_v"].map(self.node_to_index).astype(np.int32)
        node_x = {str(node): float(data["x"]) for node, data in graph.nodes(data=True)}
        node_y = {str(node): float(data["y"]) for node, data in graph.nodes(data=True)}
        frame["bearing_degrees"] = [
            _bearing_degrees(node_x[u], node_y[u], node_x[v], node_y[v])
            for u, v in zip(frame["edge_u"], frame["edge_v"])
        ]
        self.edges = frame
        self.edge_length_m = frame["length_m"].to_numpy(dtype=float)
        self.edge_time_s = frame["edge_travel_time_s"].to_numpy(dtype=float)
        self.edge_energy_kwh = frame["edge_energy_kwh"].to_numpy(dtype=float)
        self.edge_bearing_degrees = frame["bearing_degrees"].to_numpy(dtype=float)
        self.edge_by_key = {
            (str(row.edge_u), str(row.edge_v), str(row.edge_key)): int(index)
            for index, row in frame.iterrows()
        }
        if len(self.edge_by_key) != len(frame):
            raise ValueError("Road-state edge keys are not unique")
        self._adjacency: dict[RoutePolicy, csr_matrix] = {}
        self._chosen_edge: dict[RoutePolicy, dict[tuple[int, int], int]] = {}
        self._build_policy_graph("distance", "length_m")
        self._build_policy_graph("running_time", "edge_travel_time_s")

    @classmethod
    def from_files(
        cls,
        graph_path: str | Path,
        road_state: pd.DataFrame,
        profile: dict[str, Any],
    ) -> PhysicalRoadNetwork:
        graph = nx.read_graphml(graph_path)
        if not isinstance(graph, nx.MultiDiGraph):
            graph = nx.MultiDiGraph(graph)
        return cls(graph, road_state, profile)

    def _build_policy_graph(self, policy: RoutePolicy, primary_column: str) -> None:
        ordered = self.edges.sort_values(
            ["u_index", "v_index", primary_column, "length_m", "edge_key"], kind="stable"
        )
        chosen = ordered.drop_duplicates(["u_index", "v_index"], keep="first")
        rows = chosen["u_index"].to_numpy(dtype=np.int32)
        columns = chosen["v_index"].to_numpy(dtype=np.int32)
        data = chosen[primary_column].to_numpy(dtype=float)
        self._adjacency[policy] = csr_matrix(
            (data, (rows, columns)),
            shape=(len(self.node_ids), len(self.node_ids)),
            dtype=np.float64,
        )
        self._chosen_edge[policy] = {
            (int(row.u_index), int(row.v_index)): int(index)
            for index, row in chosen.iterrows()
        }

    def _turn_penalty_s(self, incoming_edge: int, outgoing_edge: int) -> float:
        incoming = float(self.edge_bearing_degrees[incoming_edge])
        outgoing = float(self.edge_bearing_degrees[outgoing_edge])
        delta = (outgoing - incoming + 540.0) % 360.0 - 180.0
        angle = abs(delta)
        cfg = self.profile["turn_penalty"]
        if angle <= float(cfg["straight_max_degrees"]):
            return 0.0
        if angle >= float(cfg["u_turn_min_degrees"]):
            return float(cfg["u_turn_s"])
        return float(cfg["right_turn_s"] if delta > 0.0 else cfg["left_turn_s"])

    def _parse_access(self, terminal_row: pd.Series) -> TerminalAccess:
        payload = terminal_row["directed_projection_offsets"]
        refs = json.loads(payload) if isinstance(payload, str) else payload
        if not isinstance(refs, list) or not refs:
            raise ValueError(
                f"Terminal {terminal_row['terminal_index']} has no directed road-access refs"
            )
        options: list[AccessOption] = []
        seen: set[tuple[str, str, str, float, float]] = set()
        for ref in refs:
            edge_key = (str(ref["u"]), str(ref["v"]), str(ref["key"]))
            if edge_key not in self.edge_by_key:
                raise ValueError(
                    f"Terminal {terminal_row['terminal_index']} references absent edge {edge_key}"
                )
            offset_from = float(ref["offset_from_u_m"])
            offset_to = float(ref["offset_to_v_m"])
            marker = (*edge_key, round(offset_from, 6), round(offset_to, 6))
            if marker in seen:
                continue
            seen.add(marker)
            options.append(
                AccessOption(
                    edge_index=self.edge_by_key[edge_key],
                    edge_key=edge_key,
                    outbound_node=self.node_to_index[edge_key[1]],
                    inbound_node=self.node_to_index[edge_key[0]],
                    offset_from_u_m=offset_from,
                    offset_to_v_m=offset_to,
                    reference_length_m=max(float(ref["length_m"]), 1e-9),
                )
            )
        connector_distance_km, connector_time_s, connector_energy = connector_costs(
            float(terminal_row["connector_length_m"]), profile=self.profile
        )
        return TerminalAccess(
            terminal_index=int(terminal_row["terminal_index"]),
            connector_distance_m=connector_distance_km * 1000.0,
            connector_time_s=connector_time_s,
            connector_energy_kwh=connector_energy,
            options=tuple(options),
        )

    def _partial_costs(self, option: AccessOption, *, outbound: bool) -> tuple[float, float, float]:
        offset = option.offset_to_v_m if outbound else option.offset_from_u_m
        fraction = float(np.clip(offset / option.reference_length_m, 0.0, 1.0))
        return (
            float(self.edge_length_m[option.edge_index]) * fraction,
            float(self.edge_time_s[option.edge_index]) * fraction,
            float(self.edge_energy_kwh[option.edge_index]) * fraction,
        )

    def _tree_costs(
        self,
        predecessors: np.ndarray,
        *,
        root: int,
        initial_edge: int,
        target_nodes: set[int],
        policy: RoutePolicy,
    ) -> dict[int, tuple[float, float, float, int]]:
        cache: dict[int, tuple[float, float, float, int]] = {
            root: (0.0, 0.0, 0.0, initial_edge)
        }
        auxiliary_kw = auxiliary_power_kw(self.profile["energy"])
        chosen_edge = self._chosen_edge[policy]
        for target in target_nodes:
            if target in cache:
                continue
            path: list[int] = []
            cursor = target
            visited: set[int] = set()
            while cursor not in cache:
                if cursor in visited:
                    raise RuntimeError("Shortest-path predecessor cycle detected")
                visited.add(cursor)
                predecessor = int(predecessors[cursor])
                if predecessor == NO_PREDECESSOR:
                    break
                path.append(cursor)
                cursor = predecessor
            if cursor not in cache:
                continue
            for node in reversed(path):
                predecessor = int(predecessors[node])
                previous_distance, previous_time, previous_energy, incoming_edge = cache[
                    predecessor
                ]
                edge_index = chosen_edge[(predecessor, node)]
                penalty = self._turn_penalty_s(incoming_edge, edge_index)
                cache[node] = (
                    previous_distance + float(self.edge_length_m[edge_index]),
                    previous_time + float(self.edge_time_s[edge_index]) + penalty,
                    previous_energy
                    + float(self.edge_energy_kwh[edge_index])
                    + auxiliary_kw * penalty / 3600.0,
                    edge_index,
                )
        return cache

    def _direct_candidate(
        self,
        source: TerminalAccess,
        source_option: AccessOption,
        destination: TerminalAccess,
        destination_option: AccessOption,
    ) -> tuple[float, float, float] | None:
        if source_option.edge_key != destination_option.edge_key:
            return None
        delta = destination_option.offset_from_u_m - source_option.offset_from_u_m
        if delta < -1e-8:
            return None
        fraction = float(np.clip(delta / source_option.reference_length_m, 0.0, 1.0))
        return (
            source.connector_distance_m
            + float(self.edge_length_m[source_option.edge_index]) * fraction
            + destination.connector_distance_m,
            source.connector_time_s
            + float(self.edge_time_s[source_option.edge_index]) * fraction
            + destination.connector_time_s,
            source.connector_energy_kwh
            + float(self.edge_energy_kwh[source_option.edge_index]) * fraction
            + destination.connector_energy_kwh,
        )

    def _route_policy(
        self,
        terminals: tuple[TerminalAccess, ...],
        *,
        policy: RoutePolicy,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        count = len(terminals)
        distance_m = np.full((count, count), np.inf, dtype=np.float64)
        travel_time_s = np.full((count, count), np.inf, dtype=np.float64)
        energy_kwh = np.full((count, count), np.inf, dtype=np.float64)
        source_witness = np.full((count, count), -1, dtype=np.int8)
        destination_witness = np.full((count, count), -1, dtype=np.int8)
        target_nodes = {
            option.inbound_node for terminal in terminals for option in terminal.options
        }
        primary = self._adjacency[policy]
        auxiliary_kw = auxiliary_power_kw(self.profile["energy"])
        for source_index, _source in enumerate(terminals):
            distance_m[source_index, source_index] = 0.0
            travel_time_s[source_index, source_index] = 0.0
            energy_kwh[source_index, source_index] = 0.0
            source_witness[source_index, source_index] = 0
            destination_witness[source_index, source_index] = 0
        tasks = [
            (source_index, source_option_index, source_option)
            for source_index, source in enumerate(terminals)
            for source_option_index, source_option in enumerate(source.options)
        ]
        tasks_by_root: dict[int, list[tuple[int, int, AccessOption]]] = {}
        for task in tasks:
            tasks_by_root.setdefault(task[2].outbound_node, []).append(task)
        roots = sorted(tasks_by_root)
        batch_size = 24
        for start in range(0, len(roots), batch_size):
            root_batch = roots[start : start + batch_size]
            _, predecessor_batch = dijkstra(
                primary,
                directed=True,
                indices=np.asarray(root_batch, dtype=np.int32),
                return_predecessors=True,
            )
            if predecessor_batch.ndim == 1:
                predecessor_batch = predecessor_batch.reshape(1, -1)
            for root_position, root in enumerate(root_batch):
                predecessors = predecessor_batch[root_position]
                for source_index, source_option_index, source_option in tasks_by_root[root]:
                    source = terminals[source_index]
                    tree = self._tree_costs(
                        predecessors,
                        root=source_option.outbound_node,
                        initial_edge=source_option.edge_index,
                        target_nodes=target_nodes,
                        policy=policy,
                    )
                    source_partial = self._partial_costs(source_option, outbound=True)
                    for destination_index, destination in enumerate(terminals):
                        if source_index == destination_index:
                            continue
                        for destination_option_index, destination_option in enumerate(
                            destination.options
                        ):
                            candidates: list[tuple[float, float, float]] = []
                            direct = self._direct_candidate(
                                source, source_option, destination, destination_option
                            )
                            if direct is not None:
                                candidates.append(direct)
                            path = tree.get(destination_option.inbound_node)
                            if path is not None:
                                graph_distance, graph_time, graph_energy, incoming_edge = path
                                destination_partial = self._partial_costs(
                                    destination_option, outbound=False
                                )
                                final_turn = self._turn_penalty_s(
                                    incoming_edge, destination_option.edge_index
                                )
                                candidates.append(
                                    (
                                        source.connector_distance_m
                                        + source_partial[0]
                                        + graph_distance
                                        + destination_partial[0]
                                        + destination.connector_distance_m,
                                        source.connector_time_s
                                        + source_partial[1]
                                        + graph_time
                                        + final_turn
                                        + destination_partial[1]
                                        + destination.connector_time_s,
                                        source.connector_energy_kwh
                                        + source_partial[2]
                                        + graph_energy
                                        + auxiliary_kw * final_turn / 3600.0
                                        + destination_partial[2]
                                        + destination.connector_energy_kwh,
                                    )
                                )
                            if not candidates:
                                continue
                            candidate = min(
                                candidates,
                                key=(lambda value: value[0])
                                if policy == "distance"
                                else (lambda value: value[1]),
                            )
                            candidate_primary = (
                                candidate[0] if policy == "distance" else candidate[1]
                            )
                            current_primary = (
                                distance_m[source_index, destination_index]
                                if policy == "distance"
                                else travel_time_s[source_index, destination_index]
                            )
                            if candidate_primary < current_primary - 1e-9:
                                distance_m[source_index, destination_index] = candidate[0]
                                travel_time_s[source_index, destination_index] = candidate[1]
                                energy_kwh[source_index, destination_index] = candidate[2]
                                source_witness[
                                    source_index, destination_index
                                ] = source_option_index
                                destination_witness[
                                    source_index, destination_index
                                ] = destination_option_index
        return distance_m, travel_time_s, energy_kwh, source_witness, destination_witness

    def route_terminals(self, terminal_index: pd.DataFrame) -> RoutingMatrices:
        expected = np.arange(len(terminal_index))
        if not np.array_equal(terminal_index["terminal_index"].to_numpy(dtype=int), expected):
            raise ValueError("terminal_index rows must be ordered contiguously from zero")
        terminals = tuple(
            self._parse_access(row) for _, row in terminal_index.iterrows()
        )
        distance = self._route_policy(terminals, policy="distance")
        running_time = self._route_policy(terminals, policy="running_time")
        arrays = (*distance[:3], *running_time[:3])
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("At least one selected terminal pair is unreachable")
        distance_km = distance[0] / 1000.0
        running_distance_km = running_time[0] / 1000.0
        off_diagonal = ~np.eye(len(terminals), dtype=bool)
        asymmetry = {
            "distance": float(
                np.mean(np.abs(distance_km - distance_km.T)[off_diagonal] > 1e-6)
            ),
            "distance_path_time": float(
                np.mean(np.abs(distance[1] - distance[1].T)[off_diagonal] > 1e-3)
            ),
            "running_time": float(
                np.mean(np.abs(running_time[1] - running_time[1].T)[off_diagonal] > 1e-3)
            ),
            "distance_path_energy": float(
                np.mean(np.abs(distance[2] - distance[2].T)[off_diagonal] > 1e-8)
            ),
        }
        report = {
            "schema": "cle_evrptw_family_routing_report_v1",
            "terminal_count": len(terminals),
            "physical_graph_node_count": len(self.node_ids),
            "directed_edge_count": len(self.edges),
            "distance_path_policy": "directed_shortest_physical_distance_with_exact_edge_projection_v1",
            "running_time_path_policy": "directed_shortest_edge_running_time_then_turn_evaluated_v1",
            "turn_penalty_in_running_time_path_optimization": False,
            "turn_penalty_model_id": str(self.profile["turn_penalty"]["model_id"]),
            "signal_delay_included": bool(
                self.profile["turn_penalty"]["signal_delay_included"]
            ),
            "route_storage": "reconstruct_on_demand_from_cle_road_state_and_terminal_access",
            "asymmetric_pair_fraction": asymmetry,
        }
        return RoutingMatrices(
            distance_matrix_km=distance_km.astype(np.float32),
            distance_path_travel_time_s=distance[1].astype(np.float32),
            distance_path_energy_kwh=distance[2].astype(np.float32),
            running_time_shortest_matrix_s=running_time[1].astype(np.float32),
            running_time_path_distance_km=running_distance_km.astype(np.float32),
            running_time_path_energy_kwh=running_time[2].astype(np.float32),
            distance_source_option=distance[3],
            distance_destination_option=distance[4],
            running_time_source_option=running_time[3],
            running_time_destination_option=running_time[4],
            report=report,
        )
