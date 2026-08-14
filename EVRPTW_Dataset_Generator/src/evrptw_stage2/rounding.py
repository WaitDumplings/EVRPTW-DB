"""Deterministic controlled rounding used by Stage-2 spatial activation.

Both the route-by-decile quota matrix and every view-tree partition are
transportation problems.  The implementation below preserves the requested
integer row and column margins exactly and minimizes absolute deviation from
the fractional target.  It deliberately exposes the invariants instead of
silently repairing an invalid result.
"""

from __future__ import annotations

import hashlib

import networkx as nx
import numpy as np


def stable_u64(seed: int, namespace: str, *parts: object) -> int:
    payload = "|".join([str(seed), namespace, *(str(part) for part in parts)])
    return int.from_bytes(hashlib.blake2b(payload.encode(), digest_size=8).digest(), "big")


def largest_remainder(
    values: np.ndarray,
    *,
    total: int,
    seed: int,
    namespace: str,
    labels: list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    """Round non-negative fractional values to an exact integer total."""

    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).all() or np.any(vector < 0.0):
        raise ValueError("largest_remainder requires a finite non-negative vector")
    if total < 0:
        raise ValueError("largest_remainder total must be non-negative")
    if len(vector) == 0:
        if total:
            raise ValueError("Cannot allocate a positive total to an empty vector")
        return np.zeros(0, dtype=np.int64)
    if not np.isclose(vector.sum(), float(total), rtol=0.0, atol=1e-7):
        raise ValueError(
            f"Fractional vector sums to {vector.sum():.12g}, expected {total}"
        )
    result = np.floor(vector + 1e-12).astype(np.int64)
    remaining = int(total - result.sum())
    if remaining < 0 or remaining > len(vector):
        raise ValueError("Largest-remainder deficit is outside its feasible range")
    labels = labels or tuple(map(str, range(len(vector))))
    order = sorted(
        range(len(vector)),
        key=lambda index: (
            -(vector[index] - result[index]),
            stable_u64(seed, namespace, labels[index]),
            str(labels[index]),
        ),
    )
    if remaining:
        result[np.asarray(order[:remaining], dtype=int)] += 1
    if int(result.sum()) != total:
        raise AssertionError("Largest-remainder total was not preserved")
    return result


def controlled_matrix_round(
    fractional: np.ndarray,
    *,
    row_targets: np.ndarray,
    column_targets: np.ndarray,
    seed: int,
    namespace: str,
    row_labels: list[str] | tuple[str, ...] | None = None,
    column_labels: list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    """Round a fractional matrix while preserving both margins exactly.

    A floor solution is augmented by a deterministic min-cost flow over cells
    whose fractional part is positive.  Transportation-polytope integrality
    gives an integer solution whenever the supplied margins are compatible
    with the fractional matrix.
    """

    values = np.asarray(fractional, dtype=float)
    rows = np.asarray(row_targets, dtype=np.int64)
    columns = np.asarray(column_targets, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("controlled_matrix_round requires a matrix")
    if values.shape != (len(rows), len(columns)):
        raise ValueError("Controlled-rounding margin dimensions do not match")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Controlled-rounding matrix must be finite and non-negative")
    if np.any(rows < 0) or np.any(columns < 0) or int(rows.sum()) != int(columns.sum()):
        raise ValueError("Controlled-rounding margins must be non-negative and balanced")
    if not np.allclose(values.sum(axis=1), rows, rtol=0.0, atol=1.0):
        raise ValueError("Row targets are incompatible with fractional row sums")
    if not np.allclose(values.sum(axis=0), columns, rtol=0.0, atol=1.0):
        raise ValueError("Column targets are incompatible with fractional column sums")

    base = np.floor(values + 1e-12).astype(np.int64)
    row_deficit = rows - base.sum(axis=1)
    column_deficit = columns - base.sum(axis=0)
    if np.any(row_deficit < 0) or np.any(column_deficit < 0):
        raise ValueError("Floor solution exceeds an integer margin")
    if int(row_deficit.sum()) != int(column_deficit.sum()):
        raise ValueError("Controlled-rounding deficits are unbalanced")
    if not int(row_deficit.sum()):
        return base

    row_labels = row_labels or tuple(f"r{index}" for index in range(len(rows)))
    column_labels = column_labels or tuple(f"c{index}" for index in range(len(columns)))
    source = "source"
    sink = "sink"
    graph = nx.DiGraph()
    graph.add_node(source, demand=-int(row_deficit.sum()))
    graph.add_node(sink, demand=int(column_deficit.sum()))
    for row_index, deficit in enumerate(row_deficit):
        node = f"row:{row_index}"
        graph.add_node(node, demand=0)
        graph.add_edge(source, node, capacity=int(deficit), weight=0)
    for column_index, deficit in enumerate(column_deficit):
        node = f"column:{column_index}"
        graph.add_node(node, demand=0)
        graph.add_edge(node, sink, capacity=int(deficit), weight=0)

    fractions = values - base
    # Primary cost is the exact incremental L1 cost, with a tiny stable
    # integer tie-break that cannot reverse two distinct primary costs.
    primary_scale = 10**9
    tie_modulus = 10**4
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            fraction = float(fractions[row_index, column_index])
            if fraction <= 1e-10:
                continue
            incremental_l1 = 1.0 - 2.0 * fraction
            primary = round(incremental_l1 * primary_scale)
            tie = stable_u64(
                seed,
                namespace,
                row_labels[row_index],
                column_labels[column_index],
            ) % tie_modulus
            graph.add_edge(
                f"row:{row_index}",
                f"column:{column_index}",
                capacity=1,
                weight=primary * tie_modulus + int(tie),
            )
    try:
        flow = nx.min_cost_flow(graph)
    except nx.NetworkXUnfeasible as error:
        raise ValueError("Controlled matrix rounding is infeasible") from error
    result = base.copy()
    for row_index in range(values.shape[0]):
        outgoing = flow[f"row:{row_index}"]
        for column_index in range(values.shape[1]):
            result[row_index, column_index] += int(
                outgoing.get(f"column:{column_index}", 0)
            )
    if not np.array_equal(result.sum(axis=1), rows):
        raise AssertionError("Controlled rounding violated a row margin")
    if not np.array_equal(result.sum(axis=0), columns):
        raise AssertionError("Controlled rounding violated a column margin")
    if np.any(result < 0):
        raise AssertionError("Controlled rounding produced a negative cell")
    return result


def balanced_cell_partition(
    cell_sizes: np.ndarray,
    child_sizes: np.ndarray,
    *,
    seed: int,
    namespace: str,
    cell_labels: list[str] | tuple[str, ...],
    child_labels: list[str] | tuple[str, ...],
) -> np.ndarray:
    """Return exact cell-by-child allocation counts for a view partition."""

    cells = np.asarray(cell_sizes, dtype=np.int64)
    children = np.asarray(child_sizes, dtype=np.int64)
    if np.any(cells < 0) or np.any(children < 0) or cells.sum() != children.sum():
        raise ValueError("Cell and child totals must be balanced and non-negative")
    if not cells.sum():
        return np.zeros((len(cells), len(children)), dtype=np.int64)
    fractional = np.outer(cells, children) / float(cells.sum())
    return controlled_matrix_round(
        fractional,
        row_targets=cells,
        column_targets=children,
        seed=seed,
        namespace=namespace,
        row_labels=cell_labels,
        column_labels=child_labels,
    )
