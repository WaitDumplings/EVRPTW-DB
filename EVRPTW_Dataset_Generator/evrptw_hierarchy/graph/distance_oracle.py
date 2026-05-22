from __future__ import annotations

import numpy as np

from evrptw_hierarchy.graph.shortest_path import build_adjacency, dijkstra_one


class DistanceOracle:
    """Road shortest-path cache for one region board.

    Two modes are supported:

    - source_cache: cache source-to-all Dijkstra vectors on demand.
    - terminal_matrix: precompute shortest distances among depot/customers/CS
      terminals once, then answer active-day matrices by slicing.

    The terminal matrix is not serialized into the mother board pickle. It is an
    in-memory generation accelerator for repeatedly sampling days from a region.
    """

    def __init__(
        self,
        num_nodes: int,
        edges: np.ndarray,
        lengths_km: np.ndarray,
        terminal_node_ids: np.ndarray | None = None,
        use_terminal_matrix: bool = False,
    ):
        self.adjacency = build_adjacency(num_nodes, edges, lengths_km)
        self._cache: dict[int, np.ndarray] = {}
        self.terminal_node_ids = None if terminal_node_ids is None else np.asarray(terminal_node_ids, dtype=np.int32)
        self._terminal_lookup: dict[int, int] = {}
        if self.terminal_node_ids is not None:
            self._terminal_lookup = {int(node): idx for idx, node in enumerate(self.terminal_node_ids.tolist())}
        self._terminal_matrix: np.ndarray | None = None
        self.use_terminal_matrix = bool(use_terminal_matrix and self.terminal_node_ids is not None)
        if self.use_terminal_matrix:
            self.precompute_terminal_matrix()

    @property
    def terminal_matrix_ready(self) -> bool:
        return self._terminal_matrix is not None

    @property
    def terminal_count(self) -> int:
        return 0 if self.terminal_node_ids is None else int(self.terminal_node_ids.size)

    @property
    def terminal_matrix_size_mb(self) -> float:
        if self._terminal_matrix is None:
            return 0.0
        return float(self._terminal_matrix.nbytes / (1024.0 * 1024.0))

    def precompute_terminal_matrix(self) -> None:
        if self.terminal_node_ids is None:
            raise ValueError("terminal_node_ids are required for terminal matrix precomputation.")
        terminals = self.terminal_node_ids
        out = np.empty((terminals.size, terminals.size), dtype=np.float32)
        for row, node_id in enumerate(terminals):
            dist = dijkstra_one(self.adjacency, int(node_id))
            out[row] = dist[terminals].astype(np.float32)
        self._terminal_matrix = out

    def _terminal_indices(self, node_ids: np.ndarray) -> np.ndarray | None:
        if self._terminal_matrix is None:
            return None
        idx = []
        for node_id in np.asarray(node_ids, dtype=int).tolist():
            value = self._terminal_lookup.get(int(node_id))
            if value is None:
                return None
            idx.append(value)
        return np.asarray(idx, dtype=np.int32)

    def distances_from(self, source_node_id: int) -> np.ndarray:
        key = int(source_node_id)
        if key not in self._cache:
            self._cache[key] = dijkstra_one(self.adjacency, key)
        return self._cache[key]

    def matrix_between(self, source_node_ids: np.ndarray, target_node_ids: np.ndarray) -> np.ndarray:
        sources = np.asarray(source_node_ids, dtype=np.int32)
        targets = np.asarray(target_node_ids, dtype=np.int32)
        src_idx = self._terminal_indices(sources)
        tgt_idx = self._terminal_indices(targets)
        if src_idx is not None and tgt_idx is not None:
            return self._terminal_matrix[np.ix_(src_idx, tgt_idx)].astype(np.float32, copy=False)
        out = np.empty((sources.size, targets.size), dtype=np.float32)
        for row, node_id in enumerate(sources):
            out[row] = self.distances_from(int(node_id))[targets].astype(np.float32)
        return out

    def matrix(self, terminal_node_ids: np.ndarray) -> np.ndarray:
        terminals = np.asarray(terminal_node_ids, dtype=np.int32)
        return self.matrix_between(terminals, terminals)
