"""Edge-aware terminal matrices over a directed CLE road graph."""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, dijkstra

from .road_state import connector_costs

NO_PREDECESSOR = -9999


class TerminalConnectivityError(ValueError):
    """A terminal roster violates the exact directed routing contract."""

    retryable = False

    def __init__(
        self,
        message: str,
        *,
        roster_fingerprint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.roster_fingerprint = roster_fingerprint


def terminal_index_fingerprint(terminal_index: pd.DataFrame) -> str:
    required = {"terminal_kind", "source_id"}
    if missing := required - set(terminal_index.columns):
        raise ValueError(f"Terminal roster lacks fingerprint columns: {sorted(missing)}")
    payload = "\n".join(
        sorted(
            f"{kind}:{source_id}"
            for kind, source_id in zip(
                terminal_index["terminal_kind"].astype(str),
                terminal_index["source_id"].astype(str),
            )
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _connector_reference_speed_kph(frame: pd.DataFrame) -> float:
    """Use the length-weighted median U-edge speed, or all edges if U is absent."""

    if "instance_speed_kph" not in frame.columns:
        raise ValueError("Road state lacks instance_speed_kph")
    speeds = pd.to_numeric(frame["instance_speed_kph"], errors="coerce")
    candidates = frame.assign(_connector_speed_kph=speeds)
    if "operating_mode" in candidates.columns:
        u_candidates = candidates.loc[candidates["operating_mode"].astype(str).eq("U")]
        if not u_candidates.empty:
            candidates = u_candidates
    ordered = candidates.sort_values("_connector_speed_kph")
    weights = ordered["length_m"].to_numpy(dtype=float)
    values = ordered["_connector_speed_kph"].to_numpy(dtype=float)
    if (
        not np.isfinite(weights).all()
        or not np.isfinite(values).all()
        or np.any(weights <= 0.0)
        or np.any(values <= 0.0)
    ):
        raise ValueError("Cannot derive a positive finite connector reference speed")
    midpoint = float(weights.sum()) / 2.0
    index = min(int(np.searchsorted(np.cumsum(weights), midpoint)), len(values) - 1)
    return float(values[index])


def _bearing_degrees(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(math.radians, (lon1, lat1, lon2, lat2))
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
    outbound_distance_m: float
    outbound_time_s: float
    inbound_distance_m: float
    inbound_time_s: float
    inbound_arrival_edges: tuple[int, ...]
    inbound_turn_penalties_s: tuple[float, ...]
    inbound_turn_penalty_by_edge: dict[int, float]


@dataclass(frozen=True)
class TerminalAccess:
    terminal_index: int
    connector_distance_m: float
    connector_time_s: float
    options: tuple[AccessOption, ...]


@dataclass(frozen=True)
class RoutingMatrices:
    """The four persisted matrices in a Stage-2 matrix family."""

    distance_matrix_km: np.ndarray
    distance_path_travel_time_s: np.ndarray
    running_time_shortest_matrix_s: np.ndarray
    running_time_path_distance_km: np.ndarray
    distance_source_option: np.ndarray
    distance_destination_option: np.ndarray
    running_time_source_option: np.ndarray
    running_time_destination_option: np.ndarray
    report: dict[str, Any]


@dataclass(frozen=True)
class DepotTerminalStar:
    """Directed depot-to-terminal and terminal-to-depot preselection costs.

    These vectors use the same edge projections and connector costs as the
    final closure, but omit turn penalties.  They are used only to define the
    Amazon-calibrated territory before a small final terminal set is routed
    with the exact turn-aware policy.
    """

    outbound_time_s: np.ndarray
    inbound_time_s: np.ndarray
    outbound_distance_km: np.ndarray
    inbound_distance_km: np.ndarray
    node_outbound_reachable: np.ndarray
    node_return_reachable: np.ndarray
    turn_outbound_reachable: np.ndarray
    turn_return_reachable: np.ndarray
    report: dict[str, Any]

    @property
    def connectivity_eligible(self) -> np.ndarray:
        return (
            self.node_outbound_reachable
            & self.node_return_reachable
            & self.turn_outbound_reachable
            & self.turn_return_reachable
        )


class PhysicalRoadNetwork:
    """Static topology plus one family-level speed realization."""

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        road_state: pd.DataFrame,
        profile: dict[str, Any],
    ):
        if not graph.is_directed():
            raise ValueError("CLE routing graph must be directed")
        self.graph = graph
        self.profile = profile
        turn_cfg = profile["turn_penalty"]
        self._straight_max_degrees = float(turn_cfg["straight_max_degrees"])
        self._u_turn_min_degrees = float(turn_cfg["u_turn_min_degrees"])
        self._right_turn_s = float(turn_cfg["right_turn_s"])
        self._left_turn_s = float(turn_cfg["left_turn_s"])
        self._u_turn_s = float(turn_cfg["u_turn_s"])
        self._turn_penalty_signature = (
            self._straight_max_degrees,
            self._u_turn_min_degrees,
            self._right_turn_s,
            self._left_turn_s,
            self._u_turn_s,
        )
        self.node_ids = tuple(sorted(map(str, graph.nodes)))
        self.node_to_index = {node: index for index, node in enumerate(self.node_ids)}
        frame = road_state.copy().reset_index(drop=True)
        required = {
            "edge_u",
            "edge_v",
            "edge_key",
            "length_m",
            "edge_travel_time_s",
            "instance_speed_kph",
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
        self.edge_count = len(frame)
        self._edge_keys_in_order = tuple(zip(frame["edge_u"], frame["edge_v"], frame["edge_key"]))
        self.edge_length_m = frame["length_m"].to_numpy(dtype=float)
        self.edge_time_s = frame["edge_travel_time_s"].to_numpy(dtype=float)
        self.edge_bearing_degrees = frame["bearing_degrees"].to_numpy(dtype=float)
        self.edge_u_index = frame["u_index"].to_numpy(dtype=np.int32)
        self.edge_v_index = frame["v_index"].to_numpy(dtype=np.int32)
        self.connector_speed_kph = _connector_reference_speed_kph(frame)
        self.edge_by_key = {
            (str(row.edge_u), str(row.edge_v), str(row.edge_key)): int(index)
            for index, row in frame.iterrows()
        }
        if len(self.edge_by_key) != len(frame):
            raise ValueError("Road-state edge keys are not unique")
        self._distance_adjacency: csr_matrix
        self._time_adjacency: csr_matrix
        self._distance_chosen_edge: dict[tuple[int, int], int]
        self._distance_tie_candidates: dict[tuple[int, int], np.ndarray]
        self._turn_aware_adjacency: csr_matrix
        self._incoming_edges_by_node: tuple[np.ndarray, ...]
        self._turn_transition_rows: np.ndarray
        self._turn_transition_columns: np.ndarray
        self._turn_transition_penalties_s: np.ndarray
        self._turn_penalty_lookup: dict[int, float]
        self._build_distance_graph()
        self._build_time_graph()
        self._build_turn_aware_graph()

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

    def with_road_state(
        self,
        road_state: pd.DataFrame,
        profile: dict[str, Any],
    ) -> PhysicalRoadNetwork:
        """Reuse immutable city topology with a new family-level speed state."""

        frame = road_state.reset_index(drop=True)
        required = {
            "edge_u",
            "edge_v",
            "edge_key",
            "length_m",
            "edge_travel_time_s",
            "instance_speed_kph",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Road state is missing routing columns: {sorted(missing)}")
        if len(frame) != len(self.edges):
            raise ValueError("Road-state edge count differs from cached city topology")
        edge_keys = tuple(
            zip(
                frame["edge_u"].astype(str),
                frame["edge_v"].astype(str),
                frame["edge_key"].astype(str),
            )
        )
        if edge_keys != self._edge_keys_in_order:
            raise ValueError("Road-state edge order differs from cached city topology")
        frame["u_index"] = self.edge_u_index
        frame["v_index"] = self.edge_v_index

        network = object.__new__(PhysicalRoadNetwork)
        for name in (
            "graph",
            "node_ids",
            "node_to_index",
            "edges",
            "edge_count",
            "_edge_keys_in_order",
            "edge_length_m",
            "edge_bearing_degrees",
            "edge_u_index",
            "edge_v_index",
            "edge_by_key",
            "connector_speed_kph",
            "_distance_adjacency",
            "_time_adjacency",
            "_distance_chosen_edge",
            "_distance_tie_candidates",
            "_incoming_edges_by_node",
            "_turn_transition_rows",
            "_turn_transition_columns",
            "_turn_transition_penalties_s",
            "_turn_penalty_lookup",
            "_turn_penalty_signature",
            "_topological_reversal_forbidden_count",
        ):
            setattr(network, name, getattr(self, name))
        network.profile = profile
        turn_cfg = profile["turn_penalty"]
        network._straight_max_degrees = float(turn_cfg["straight_max_degrees"])
        network._u_turn_min_degrees = float(turn_cfg["u_turn_min_degrees"])
        network._right_turn_s = float(turn_cfg["right_turn_s"])
        network._left_turn_s = float(turn_cfg["left_turn_s"])
        network._u_turn_s = float(turn_cfg["u_turn_s"])
        if (
            network._straight_max_degrees,
            network._u_turn_min_degrees,
            network._right_turn_s,
            network._left_turn_s,
            network._u_turn_s,
        ) != network._turn_penalty_signature:
            raise ValueError("Cached city topology requires the same turn-penalty profile")
        network.edge_time_s = frame["edge_travel_time_s"].to_numpy(dtype=float)
        network.connector_speed_kph = _connector_reference_speed_kph(frame)
        if network._distance_tie_candidates:
            network._distance_chosen_edge = network._distance_chosen_edge.copy()
            for pair, candidates in network._distance_tie_candidates.items():
                winner = min(
                    map(int, candidates),
                    key=lambda edge_index: (
                        float(network.edge_time_s[edge_index]),
                        network._edge_keys_in_order[edge_index][2],
                    ),
                )
                network._distance_chosen_edge[pair] = winner
        weights = (
            network.edge_time_s[network._turn_transition_columns]
            + network._turn_transition_penalties_s
        )
        network._turn_aware_adjacency = csr_matrix(
            (weights, (network._turn_transition_rows, network._turn_transition_columns)),
            shape=(network.edge_count, network.edge_count),
            dtype=np.float64,
        )
        ordered_time = frame.assign(_edge_position=np.arange(len(frame))).sort_values(
            ["u_index", "v_index", "edge_travel_time_s", "length_m", "edge_key"],
            kind="stable",
        )
        chosen_time = ordered_time.drop_duplicates(["u_index", "v_index"], keep="first")
        network._time_adjacency = csr_matrix(
            (
                chosen_time["edge_travel_time_s"].to_numpy(dtype=float),
                (
                    chosen_time["u_index"].to_numpy(dtype=np.int32),
                    chosen_time["v_index"].to_numpy(dtype=np.int32),
                ),
            ),
            shape=(len(network.node_ids), len(network.node_ids)),
            dtype=np.float64,
        )
        return network

    def _build_distance_graph(self) -> None:
        ordered = self.edges.sort_values(
            ["u_index", "v_index", "length_m", "edge_travel_time_s", "edge_key"],
            kind="stable",
        )
        chosen = ordered.drop_duplicates(["u_index", "v_index"], keep="first")
        rows = chosen["u_index"].to_numpy(dtype=np.int32)
        columns = chosen["v_index"].to_numpy(dtype=np.int32)
        data = chosen["length_m"].to_numpy(dtype=float)
        self._distance_adjacency = csr_matrix(
            (data, (rows, columns)),
            shape=(len(self.node_ids), len(self.node_ids)),
            dtype=np.float64,
        )
        self._distance_chosen_edge = {
            (int(row.u_index), int(row.v_index)): int(index) for index, row in chosen.iterrows()
        }
        minimum_lengths = self.edges.groupby(["u_index", "v_index"], sort=False)[
            "length_m"
        ].transform("min")
        minimum_edges = self.edges.loc[
            self.edges["length_m"].eq(minimum_lengths),
            ["u_index", "v_index"],
        ].copy()
        minimum_edges["candidate_count"] = minimum_edges.groupby(
            ["u_index", "v_index"], sort=False
        )["u_index"].transform("size")
        tied = minimum_edges.loc[minimum_edges["candidate_count"].gt(1)]
        self._distance_tie_candidates = {
            (int(u_index), int(v_index)): group.index.to_numpy(dtype=np.int32)
            for (u_index, v_index), group in tied.groupby(["u_index", "v_index"], sort=False)
        }

    def _build_time_graph(self) -> None:
        ordered = self.edges.sort_values(
            ["u_index", "v_index", "edge_travel_time_s", "length_m", "edge_key"],
            kind="stable",
        )
        chosen = ordered.drop_duplicates(["u_index", "v_index"], keep="first")
        self._time_adjacency = csr_matrix(
            (
                chosen["edge_travel_time_s"].to_numpy(dtype=float),
                (
                    chosen["u_index"].to_numpy(dtype=np.int32),
                    chosen["v_index"].to_numpy(dtype=np.int32),
                ),
            ),
            shape=(len(self.node_ids), len(self.node_ids)),
            dtype=np.float64,
        )

    def _build_turn_aware_graph(self) -> None:
        """Build a directed line graph whose transition weights include turns."""

        edge_count = self.edge_count
        outgoing: list[list[int]] = [[] for _ in self.node_ids]
        incoming: list[list[int]] = [[] for _ in self.node_ids]
        for edge_index, (u_index, v_index) in enumerate(zip(self.edge_u_index, self.edge_v_index)):
            outgoing[int(u_index)].append(edge_index)
            incoming[int(v_index)].append(edge_index)
        self._incoming_edges_by_node = tuple(
            np.asarray(values, dtype=np.int32) for values in incoming
        )
        rows_list: list[int] = []
        columns_list: list[int] = []
        weights_list: list[float] = []
        penalties_list: list[float] = []
        forbid_reversal = (
            self.profile["turn_penalty"].get(
                "virtual_access_connector_immediate_reversal"
            )
            == "topologically_forbidden"
        )
        forbidden_count = 0
        for incoming_edge, v_index in enumerate(self.edge_v_index):
            next_edges = outgoing[int(v_index)]
            for outgoing_edge in next_edges:
                if forbid_reversal and (
                    int(self.edge_u_index[incoming_edge])
                    == int(self.edge_v_index[outgoing_edge])
                    and int(self.edge_v_index[incoming_edge])
                    == int(self.edge_u_index[outgoing_edge])
                ):
                    forbidden_count += 1
                    continue
                penalty = self._compute_turn_penalty_s(incoming_edge, outgoing_edge)
                rows_list.append(incoming_edge)
                columns_list.append(outgoing_edge)
                penalties_list.append(penalty)
                weights_list.append(float(self.edge_time_s[outgoing_edge]) + penalty)
        rows = np.asarray(rows_list, dtype=np.int32)
        columns = np.asarray(columns_list, dtype=np.int32)
        weights = np.asarray(weights_list, dtype=np.float64)
        turn_penalties = np.asarray(penalties_list, dtype=np.float64)
        self._topological_reversal_forbidden_count = forbidden_count
        self._turn_transition_rows = rows
        self._turn_transition_columns = columns
        self._turn_transition_penalties_s = turn_penalties
        self._turn_penalty_lookup = {
            int(incoming_edge) * edge_count + int(outgoing_edge): float(penalty)
            for incoming_edge, outgoing_edge, penalty in zip(rows, columns, turn_penalties)
        }
        self._turn_aware_adjacency = csr_matrix(
            (weights, (rows, columns)),
            shape=(edge_count, edge_count),
            dtype=np.float64,
        )

    def _compute_turn_penalty_s(self, incoming_edge: int, outgoing_edge: int) -> float:
        incoming = float(self.edge_bearing_degrees[incoming_edge])
        outgoing = float(self.edge_bearing_degrees[outgoing_edge])
        delta = (outgoing - incoming + 540.0) % 360.0 - 180.0
        angle = abs(delta)
        if angle <= self._straight_max_degrees:
            return 0.0
        if angle >= self._u_turn_min_degrees:
            return self._u_turn_s
        return self._right_turn_s if delta > 0.0 else self._left_turn_s

    def _turn_penalty_s(self, incoming_edge: int, outgoing_edge: int) -> float:
        cached = self._turn_penalty_lookup.get(
            int(incoming_edge) * self.edge_count + int(outgoing_edge)
        )
        if cached is not None:
            return cached
        return self._compute_turn_penalty_s(incoming_edge, outgoing_edge)

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
            edge_index = self.edge_by_key[edge_key]
            reference_length_m = max(float(ref["length_m"]), 1e-9)
            outbound_fraction = float(np.clip(offset_to / reference_length_m, 0.0, 1.0))
            inbound_fraction = float(np.clip(offset_from / reference_length_m, 0.0, 1.0))
            inbound_arrival_edges = tuple(
                map(int, self._incoming_edges_by_node[self.node_to_index[edge_key[0]]])
            )
            inbound_turn_penalties_s = tuple(
                self._turn_penalty_s(incoming_edge, edge_index)
                for incoming_edge in inbound_arrival_edges
            )
            options.append(
                AccessOption(
                    edge_index=edge_index,
                    edge_key=edge_key,
                    outbound_node=self.node_to_index[edge_key[1]],
                    inbound_node=self.node_to_index[edge_key[0]],
                    offset_from_u_m=offset_from,
                    offset_to_v_m=offset_to,
                    reference_length_m=reference_length_m,
                    outbound_distance_m=(float(self.edge_length_m[edge_index]) * outbound_fraction),
                    outbound_time_s=(float(self.edge_time_s[edge_index]) * outbound_fraction),
                    inbound_distance_m=(float(self.edge_length_m[edge_index]) * inbound_fraction),
                    inbound_time_s=(float(self.edge_time_s[edge_index]) * inbound_fraction),
                    inbound_arrival_edges=inbound_arrival_edges,
                    inbound_turn_penalties_s=inbound_turn_penalties_s,
                    inbound_turn_penalty_by_edge=dict(
                        zip(inbound_arrival_edges, inbound_turn_penalties_s)
                    ),
                )
            )
        connector_distance_km, connector_time_s = connector_costs(
            float(terminal_row["connector_length_m"]),
            speed_kph=self.connector_speed_kph,
        )
        return TerminalAccess(
            terminal_index=int(terminal_row["terminal_index"]),
            connector_distance_m=connector_distance_km * 1000.0,
            connector_time_s=connector_time_s,
            options=tuple(options),
        )

    def route_depot_star(self, terminal_index: pd.DataFrame) -> DepotTerminalStar:
        """Route one depot against a large candidate roster in linear storage.

        Row zero must be the depot.  The approximation is deliberately limited
        to territory eligibility; the materialized matrices still use
        :meth:`route_terminals` and therefore include exact edge projections
        and turn penalties.
        """

        expected = np.arange(len(terminal_index))
        if not np.array_equal(terminal_index["terminal_index"].to_numpy(dtype=int), expected):
            raise ValueError("terminal_index rows must be ordered contiguously from zero")
        terminals = tuple(self._parse_access(row) for _, row in terminal_index.iterrows())
        if not terminals:
            raise ValueError("Depot star requires at least one terminal")
        depot = terminals[0]
        count = len(terminals)
        outbound_time = np.full(count, np.inf, dtype=np.float64)
        inbound_time = np.full(count, np.inf, dtype=np.float64)
        outbound_distance = np.full(count, np.inf, dtype=np.float64)
        inbound_distance = np.full(count, np.inf, dtype=np.float64)
        outbound_time[0] = inbound_time[0] = 0.0
        outbound_distance[0] = inbound_distance[0] = 0.0

        destination_nodes = sorted(
            {option.inbound_node for terminal in terminals for option in terminal.options}
        )
        for depot_option in depot.options:
            time_values = dijkstra(
                self._time_adjacency,
                directed=True,
                indices=depot_option.outbound_node,
            )
            distance_values = dijkstra(
                self._distance_adjacency,
                directed=True,
                indices=depot_option.outbound_node,
            )
            for terminal_position, destination in enumerate(terminals[1:], start=1):
                for destination_option in destination.options:
                    direct = self._direct_candidate(
                        depot, depot_option, destination, destination_option
                    )
                    if direct is not None:
                        outbound_distance[terminal_position] = min(
                            outbound_distance[terminal_position], direct[0]
                        )
                        outbound_time[terminal_position] = min(
                            outbound_time[terminal_position], direct[1]
                        )
                    node = destination_option.inbound_node
                    if not np.isfinite(time_values[node]) or not np.isfinite(
                        distance_values[node]
                    ):
                        continue
                    candidate_time = (
                        depot.connector_time_s
                        + depot_option.outbound_time_s
                        + float(time_values[node])
                        + destination_option.inbound_time_s
                        + destination.connector_time_s
                    )
                    candidate_distance = (
                        depot.connector_distance_m
                        + depot_option.outbound_distance_m
                        + float(distance_values[node])
                        + destination_option.inbound_distance_m
                        + destination.connector_distance_m
                    )
                    if (candidate_time, candidate_distance) < (
                        outbound_time[terminal_position],
                        outbound_distance[terminal_position],
                    ):
                        outbound_time[terminal_position] = candidate_time
                        outbound_distance[terminal_position] = candidate_distance

        # Reversed node-level shortest paths give every candidate's return to
        # each depot inbound node without one Dijkstra call per customer.
        for depot_option in depot.options:
            reverse_time = dijkstra(
                self._time_adjacency.T,
                directed=True,
                indices=depot_option.inbound_node,
            )
            reverse_distance = dijkstra(
                self._distance_adjacency.T,
                directed=True,
                indices=depot_option.inbound_node,
            )
            for terminal_position, source in enumerate(terminals[1:], start=1):
                for source_option in source.options:
                    direct = self._direct_candidate(
                        source, source_option, depot, depot_option
                    )
                    if direct is not None:
                        inbound_distance[terminal_position] = min(
                            inbound_distance[terminal_position], direct[0]
                        )
                        inbound_time[terminal_position] = min(
                            inbound_time[terminal_position], direct[1]
                        )
                    node = source_option.outbound_node
                    if not np.isfinite(reverse_time[node]) or not np.isfinite(
                        reverse_distance[node]
                    ):
                        continue
                    candidate_time = (
                        source.connector_time_s
                        + source_option.outbound_time_s
                        + float(reverse_time[node])
                        + depot_option.inbound_time_s
                        + depot.connector_time_s
                    )
                    candidate_distance = (
                        source.connector_distance_m
                        + source_option.outbound_distance_m
                        + float(reverse_distance[node])
                        + depot_option.inbound_distance_m
                        + depot.connector_distance_m
                    )
                    if (candidate_time, candidate_distance) < (
                        inbound_time[terminal_position],
                        inbound_distance[terminal_position],
                    ):
                        inbound_time[terminal_position] = candidate_time
                        inbound_distance[terminal_position] = candidate_distance
        node_outbound_reachable = np.isfinite(outbound_time) & np.isfinite(
            outbound_distance
        )
        node_return_reachable = np.isfinite(inbound_time) & np.isfinite(
            inbound_distance
        )
        turn_outbound_reachable, turn_return_reachable = (
            self._turn_aware_depot_reachability(terminals)
        )
        connectivity_eligible = (
            node_outbound_reachable
            & node_return_reachable
            & turn_outbound_reachable
            & turn_return_reachable
        )
        return DepotTerminalStar(
            outbound_time_s=outbound_time,
            inbound_time_s=inbound_time,
            outbound_distance_km=outbound_distance / 1000.0,
            inbound_distance_km=inbound_distance / 1000.0,
            node_outbound_reachable=node_outbound_reachable,
            node_return_reachable=node_return_reachable,
            turn_outbound_reachable=turn_outbound_reachable,
            turn_return_reachable=turn_return_reachable,
            report={
                "schema": "cle_evrptw_depot_terminal_star_v2",
                "terminal_count": count,
                "path_policy": "directed_edge_time_without_turn_penalty_for_territory_only_v1",
                "turn_aware_final_matrix_required": True,
                "destination_node_count": len(destination_nodes),
                "node_outbound_unreachable_count": int(
                    (~node_outbound_reachable).sum()
                ),
                "node_return_unreachable_count": int((~node_return_reachable).sum()),
                "turn_outbound_unreachable_count": int(
                    (~turn_outbound_reachable).sum()
                ),
                "turn_return_unreachable_count": int((~turn_return_reachable).sum()),
                "connectivity_quarantined_count": int((~connectivity_eligible).sum()),
                "connectivity_preflight": (
                    "depot_bidirectional_node_and_canonical_turn_topology_v1"
                ),
            },
        )

    @staticmethod
    def _reachable_union(adjacency: csr_matrix, sources: list[int]) -> np.ndarray:
        reachable = np.zeros(adjacency.shape[0], dtype=bool)
        for source in sorted(set(map(int, sources))):
            order = breadth_first_order(
                adjacency,
                i_start=source,
                directed=True,
                return_predecessors=False,
            )
            reachable[np.asarray(order, dtype=np.int32)] = True
        return reachable

    def _turn_aware_depot_reachability(
        self,
        terminals: tuple[TerminalAccess, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Audit terminal communication using the canonical directed line graph."""

        depot = terminals[0]
        forward_edges = self._reachable_union(
            self._turn_aware_adjacency,
            [option.edge_index for option in depot.options],
        )
        depot_arrival_edges = [
            edge
            for option in depot.options
            for edge in option.inbound_arrival_edges
        ]
        return_edges = self._reachable_union(
            self._turn_aware_adjacency.T.tocsr(),
            depot_arrival_edges,
        )
        outbound = np.zeros(len(terminals), dtype=bool)
        inbound = np.zeros(len(terminals), dtype=bool)
        outbound[0] = inbound[0] = True
        for position, terminal in enumerate(terminals[1:], start=1):
            outbound[position] = any(
                forward_edges[edge]
                for option in terminal.options
                for edge in option.inbound_arrival_edges
            )
            inbound[position] = any(
                return_edges[option.edge_index] for option in terminal.options
            )
            if not outbound[position]:
                outbound[position] = any(
                    self._direct_candidate(depot, left, terminal, right) is not None
                    for left in depot.options
                    for right in terminal.options
                )
            if not inbound[position]:
                inbound[position] = any(
                    self._direct_candidate(terminal, left, depot, right) is not None
                    for left in terminal.options
                    for right in depot.options
                )
        return outbound, inbound

    def _partial_costs(self, option: AccessOption, *, outbound: bool) -> tuple[float, float]:
        if outbound:
            return option.outbound_distance_m, option.outbound_time_s
        return option.inbound_distance_m, option.inbound_time_s

    def _direct_candidate(
        self,
        source: TerminalAccess,
        source_option: AccessOption,
        destination: TerminalAccess,
        destination_option: AccessOption,
    ) -> tuple[float, float] | None:
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
        )

    def _distance_tree_costs(
        self,
        predecessors: np.ndarray,
        *,
        root: int,
        initial_edge: int,
        target_nodes: set[int],
    ) -> dict[int, tuple[float, float, int]]:
        cache: dict[int, tuple[float, float, int]] = {root: (0.0, 0.0, initial_edge)}
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
                previous_distance, previous_time, incoming_edge = cache[predecessor]
                edge_index = self._distance_chosen_edge[(predecessor, node)]
                cache[node] = (
                    previous_distance + float(self.edge_length_m[edge_index]),
                    previous_time
                    + float(self.edge_time_s[edge_index])
                    + self._turn_penalty_s(incoming_edge, edge_index),
                    edge_index,
                )
        return cache

    def _route_distance(
        self, terminals: tuple[TerminalAccess, ...]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        count = len(terminals)
        distance_m = np.full((count, count), np.inf, dtype=np.float64)
        travel_time_s = np.full((count, count), np.inf, dtype=np.float64)
        source_witness = np.full((count, count), -1, dtype=np.int8)
        destination_witness = np.full((count, count), -1, dtype=np.int8)
        target_nodes = {
            option.inbound_node for terminal in terminals for option in terminal.options
        }
        for index in range(count):
            distance_m[index, index] = 0.0
            travel_time_s[index, index] = 0.0
            source_witness[index, index] = 0
            destination_witness[index, index] = 0
        tasks = [
            (source_index, option_index, option)
            for source_index, source in enumerate(terminals)
            for option_index, option in enumerate(source.options)
        ]
        tasks_by_root: dict[int, list[tuple[int, int, AccessOption]]] = {}
        for task in tasks:
            tasks_by_root.setdefault(task[2].outbound_node, []).append(task)
        roots = sorted(tasks_by_root)
        for start in range(0, len(roots), 24):
            root_batch = roots[start : start + 24]
            _, predecessor_batch = dijkstra(
                self._distance_adjacency,
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
                    tree = self._distance_tree_costs(
                        predecessors,
                        root=source_option.outbound_node,
                        initial_edge=source_option.edge_index,
                        target_nodes=target_nodes,
                    )
                    source_partial = self._partial_costs(source_option, outbound=True)
                    for destination_index, destination in enumerate(terminals):
                        if source_index == destination_index:
                            continue
                        for destination_option_index, destination_option in enumerate(
                            destination.options
                        ):
                            candidates: list[tuple[float, float]] = []
                            direct = self._direct_candidate(
                                source, source_option, destination, destination_option
                            )
                            if direct is not None:
                                candidates.append(direct)
                            path = tree.get(destination_option.inbound_node)
                            if path is not None:
                                graph_distance, graph_time, incoming_edge = path
                                destination_partial = self._partial_costs(
                                    destination_option, outbound=False
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
                                        + destination_option.inbound_turn_penalty_by_edge[
                                            incoming_edge
                                        ]
                                        + destination_partial[1]
                                        + destination.connector_time_s,
                                    )
                                )
                            if not candidates:
                                continue
                            candidate = min(candidates, key=lambda value: (value[0], value[1]))
                            current = (
                                distance_m[source_index, destination_index],
                                travel_time_s[source_index, destination_index],
                            )
                            if candidate < current:
                                distance_m[source_index, destination_index] = candidate[0]
                                travel_time_s[source_index, destination_index] = candidate[1]
                                source_witness[source_index, destination_index] = (
                                    source_option_index
                                )
                                destination_witness[source_index, destination_index] = (
                                    destination_option_index
                                )
        return distance_m, travel_time_s, source_witness, destination_witness

    def _line_tree_distances(
        self,
        predecessors: np.ndarray,
        *,
        root_edge: int,
        target_edges: set[int],
    ) -> dict[int, float]:
        cache: dict[int, float] = {root_edge: 0.0}
        for target in target_edges:
            if target in cache:
                continue
            path: list[int] = []
            cursor = target
            visited: set[int] = set()
            while cursor not in cache:
                if cursor in visited:
                    raise RuntimeError("Turn-aware predecessor cycle detected")
                visited.add(cursor)
                predecessor = int(predecessors[cursor])
                if predecessor == NO_PREDECESSOR:
                    break
                path.append(cursor)
                cursor = predecessor
            if cursor not in cache:
                continue
            distance = cache[cursor]
            for edge_index in reversed(path):
                distance += float(self.edge_length_m[edge_index])
                cache[edge_index] = distance
        return cache

    def _route_running_time(
        self, terminals: tuple[TerminalAccess, ...]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        count = len(terminals)
        distance_m = np.full((count, count), np.inf, dtype=np.float64)
        travel_time_s = np.full((count, count), np.inf, dtype=np.float64)
        source_witness = np.full((count, count), -1, dtype=np.int8)
        destination_witness = np.full((count, count), -1, dtype=np.int8)
        for index in range(count):
            distance_m[index, index] = 0.0
            travel_time_s[index, index] = 0.0
            source_witness[index, index] = 0
            destination_witness[index, index] = 0
        tasks = [
            (source_index, option_index, option)
            for source_index, source in enumerate(terminals)
            for option_index, option in enumerate(source.options)
        ]
        tasks_by_edge: dict[int, list[tuple[int, int, AccessOption]]] = {}
        for task in tasks:
            tasks_by_edge.setdefault(task[2].edge_index, []).append(task)
        roots = sorted(tasks_by_edge)

        flat_options = tuple(option for terminal in terminals for option in terminal.options)
        terminal_option_counts = np.asarray(
            [len(terminal.options) for terminal in terminals], dtype=np.int32
        )
        terminal_option_starts = np.concatenate(
            (
                np.asarray([0], dtype=np.int32),
                np.cumsum(terminal_option_counts[:-1], dtype=np.int32),
            )
        )
        option_local_indices = np.concatenate(
            [np.arange(len(terminal.options), dtype=np.int16) for terminal in terminals]
        )
        option_terminal_indices = np.repeat(
            np.arange(count, dtype=np.int32), terminal_option_counts
        )
        option_edge_indices = np.asarray(
            [option.edge_index for option in flat_options], dtype=np.int32
        )
        option_offsets_from_u_m = np.asarray(
            [option.offset_from_u_m for option in flat_options], dtype=np.float64
        )
        option_inbound_distance_m = np.asarray(
            [option.inbound_distance_m for option in flat_options], dtype=np.float64
        )
        option_inbound_time_s = np.asarray(
            [option.inbound_time_s for option in flat_options], dtype=np.float64
        )
        option_destination_connector_distance_m = np.asarray(
            [
                terminals[int(terminal_index)].connector_distance_m
                for terminal_index in option_terminal_indices
            ],
            dtype=np.float64,
        )
        option_destination_connector_time_s = np.asarray(
            [
                terminals[int(terminal_index)].connector_time_s
                for terminal_index in option_terminal_indices
            ],
            dtype=np.float64,
        )
        destination_distance_tail = (
            option_inbound_distance_m + option_destination_connector_distance_m
        )
        destination_time_tail = option_inbound_time_s + option_destination_connector_time_s

        arrival_counts = np.asarray(
            [len(option.inbound_arrival_edges) for option in flat_options],
            dtype=np.int32,
        )
        if np.any(arrival_counts <= 0):
            raise ValueError("A terminal access option has no incoming physical-road edge")
        arrival_starts = np.concatenate(
            (
                np.asarray([0], dtype=np.int32),
                np.cumsum(arrival_counts[:-1], dtype=np.int32),
            )
        )
        arrival_edges = np.asarray(
            [
                incoming_edge
                for option in flat_options
                for incoming_edge in option.inbound_arrival_edges
            ],
            dtype=np.int32,
        )
        arrival_turn_penalties_s = np.asarray(
            [penalty for option in flat_options for penalty in option.inbound_turn_penalties_s],
            dtype=np.float64,
        )
        target_edges = set(map(int, arrival_edges))
        options_by_edge: dict[int, np.ndarray] = {}
        for edge_index in np.unique(option_edge_indices):
            options_by_edge[int(edge_index)] = np.flatnonzero(option_edge_indices == edge_index)

        for start in range(0, len(roots), 12):
            root_batch = roots[start : start + 12]
            distance_batch, predecessor_batch = dijkstra(
                self._turn_aware_adjacency,
                directed=True,
                indices=np.asarray(root_batch, dtype=np.int32),
                return_predecessors=True,
            )
            if distance_batch.ndim == 1:
                distance_batch = distance_batch.reshape(1, -1)
                predecessor_batch = predecessor_batch.reshape(1, -1)
            for root_position, root_edge in enumerate(root_batch):
                turn_times = distance_batch[root_position]
                predecessors = predecessor_batch[root_position]
                line_distances = self._line_tree_distances(
                    predecessors,
                    root_edge=root_edge,
                    target_edges=target_edges,
                )
                arrival_distances = np.fromiter(
                    (
                        line_distances.get(int(incoming_edge), np.inf)
                        for incoming_edge in arrival_edges
                    ),
                    dtype=np.float64,
                    count=len(arrival_edges),
                )
                arrival_times = turn_times[arrival_edges] + arrival_turn_penalties_s
                arrival_valid = np.isfinite(arrival_distances) & np.isfinite(arrival_times)
                valid_arrival_times = np.where(arrival_valid, arrival_times, np.inf)
                best_option_times = np.minimum.reduceat(valid_arrival_times, arrival_starts)
                best_option_times_expanded = np.repeat(best_option_times, arrival_counts)
                best_option_distances = np.minimum.reduceat(
                    np.where(
                        arrival_valid & (valid_arrival_times == best_option_times_expanded),
                        arrival_distances,
                        np.inf,
                    ),
                    arrival_starts,
                )
                graph_option_times = best_option_times + destination_time_tail
                graph_option_distances = best_option_distances + destination_distance_tail

                for source_index, source_option_index, source_option in tasks_by_edge[root_edge]:
                    source = terminals[source_index]
                    source_partial = self._partial_costs(source_option, outbound=True)
                    source_distance_head = source.connector_distance_m + source_partial[0]
                    source_time_head = source.connector_time_s + source_partial[1]
                    candidate_option_distances = graph_option_distances + source_distance_head
                    candidate_option_times = graph_option_times + source_time_head

                    direct_indices = options_by_edge.get(source_option.edge_index)
                    if direct_indices is not None:
                        delta_m = (
                            option_offsets_from_u_m[direct_indices] - source_option.offset_from_u_m
                        )
                        reachable = delta_m >= -1e-8
                        if np.any(reachable):
                            reachable_indices = direct_indices[reachable]
                            fractions = np.clip(
                                delta_m[reachable] / source_option.reference_length_m,
                                0.0,
                                1.0,
                            )
                            direct_distances = (
                                source.connector_distance_m
                                + float(self.edge_length_m[source_option.edge_index]) * fractions
                                + option_destination_connector_distance_m[reachable_indices]
                            )
                            direct_times = (
                                source.connector_time_s
                                + float(self.edge_time_s[source_option.edge_index]) * fractions
                                + option_destination_connector_time_s[reachable_indices]
                            )
                            current_direct_times = candidate_option_times[reachable_indices]
                            current_direct_distances = candidate_option_distances[reachable_indices]
                            direct_is_better = (direct_times < current_direct_times) | (
                                (direct_times == current_direct_times)
                                & (direct_distances <= current_direct_distances)
                            )
                            selected = reachable_indices[direct_is_better]
                            candidate_option_times[selected] = direct_times[direct_is_better]
                            candidate_option_distances[selected] = direct_distances[
                                direct_is_better
                            ]

                    best_terminal_times = np.minimum.reduceat(
                        candidate_option_times, terminal_option_starts
                    )
                    best_terminal_times_expanded = np.repeat(
                        best_terminal_times, terminal_option_counts
                    )
                    best_terminal_distances = np.minimum.reduceat(
                        np.where(
                            candidate_option_times == best_terminal_times_expanded,
                            candidate_option_distances,
                            np.inf,
                        ),
                        terminal_option_starts,
                    )
                    best_terminal_distances_expanded = np.repeat(
                        best_terminal_distances, terminal_option_counts
                    )
                    destination_matches = (
                        candidate_option_times == best_terminal_times_expanded
                    ) & (candidate_option_distances == best_terminal_distances_expanded)
                    best_destination_options = np.minimum.reduceat(
                        np.where(destination_matches, option_local_indices, 127),
                        terminal_option_starts,
                    ).astype(np.int8)

                    current_times = travel_time_s[source_index]
                    current_distances = distance_m[source_index]
                    update = np.isfinite(best_terminal_times) & (
                        (best_terminal_times < current_times)
                        | (
                            (best_terminal_times == current_times)
                            & (best_terminal_distances < current_distances)
                        )
                    )
                    update[source_index] = False
                    distance_m[source_index, update] = best_terminal_distances[update]
                    travel_time_s[source_index, update] = best_terminal_times[update]
                    source_witness[source_index, update] = source_option_index
                    destination_witness[source_index, update] = best_destination_options[update]
        return distance_m, travel_time_s, source_witness, destination_witness

    def route_terminals(self, terminal_index: pd.DataFrame) -> RoutingMatrices:
        expected = np.arange(len(terminal_index))
        if not np.array_equal(terminal_index["terminal_index"].to_numpy(dtype=int), expected):
            raise ValueError("terminal_index rows must be ordered contiguously from zero")
        terminals = tuple(self._parse_access(row) for _, row in terminal_index.iterrows())
        distance = self._route_distance(terminals)
        running_time = self._route_running_time(terminals)
        arrays = (*distance[:2], *running_time[:2])
        if any(not np.isfinite(array).all() for array in arrays):
            raise TerminalConnectivityError(
                "NONRETRYABLE_TERMINAL_CONNECTIVITY: exact selected-terminal "
                "closure contains at least one unreachable directed pair after preflight",
                roster_fingerprint=terminal_index_fingerprint(terminal_index),
            )
        distance_km = distance[0] / 1000.0
        running_distance_km = running_time[0] / 1000.0
        off_diagonal = ~np.eye(len(terminals), dtype=bool)
        asymmetry = {
            "distance": float(np.mean(np.abs(distance_km - distance_km.T)[off_diagonal] > 1e-6)),
            "distance_path_time": float(
                np.mean(np.abs(distance[1] - distance[1].T)[off_diagonal] > 1e-3)
            ),
            "running_time": float(
                np.mean(np.abs(running_time[1] - running_time[1].T)[off_diagonal] > 1e-3)
            ),
            "running_time_path_distance": float(
                np.mean(np.abs(running_distance_km - running_distance_km.T)[off_diagonal] > 1e-6)
            ),
        }
        canonical_zero_turn = all(
            value == 0.0
            for value in (self._right_turn_s, self._left_turn_s, self._u_turn_s)
        )
        report = {
            "schema": "cle_evrptw_family_routing_report_v2",
            "terminal_count": len(terminals),
            "physical_graph_node_count": len(self.node_ids),
            "directed_edge_count": self.edge_count,
            "turn_aware_transition_count": int(self._turn_aware_adjacency.nnz),
            "distance_path_policy": (
                "directed_shortest_physical_distance_with_exact_edge_projection_v1"
            ),
            "running_time_path_policy": (
                "directed_zero_turn_shortest_running_time_v3"
                if canonical_zero_turn
                else "geometry_turn_penalty_v1_optional_adapter"
            ),
            "turn_penalty_in_running_time_path_optimization": not canonical_zero_turn,
            "turn_penalty_model_id": str(self.profile["turn_penalty"]["model_id"]),
            "canonical_zero_turn": canonical_zero_turn,
            "topological_immediate_edge_reversal_rule": (
                "forbidden" if self._topological_reversal_forbidden_count else "not_triggered"
            ),
            "topological_immediate_edge_reversal_transition_count": int(
                self._topological_reversal_forbidden_count
            ),
            "signal_delay_included": bool(self.profile["turn_penalty"]["signal_delay_included"]),
            "route_storage": (
                "reconstruct_on_demand_from_cle_road_state_terminal_access_and_policy"
            ),
            "stored_matrix_count": 4,
            "energy_matrix_storage": "derived_from_path_distance_times_scalar_h",
            "asymmetric_pair_fraction": asymmetry,
        }
        return RoutingMatrices(
            distance_matrix_km=distance_km.astype(np.float32),
            distance_path_travel_time_s=distance[1].astype(np.float32),
            running_time_shortest_matrix_s=running_time[1].astype(np.float32),
            running_time_path_distance_km=running_distance_km.astype(np.float32),
            distance_source_option=distance[2],
            distance_destination_option=distance[3],
            running_time_source_option=running_time[2],
            running_time_destination_option=running_time[3],
            report=report,
        )
