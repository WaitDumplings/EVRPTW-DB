from __future__ import annotations

from typing import Any

import numpy as np

from evrptw_hierarchy.core.models import RegionBoard
from evrptw_hierarchy.graph.distance_oracle import DistanceOracle


def build_candidate_pool(board: RegionBoard, active_customer_ids: np.ndarray, max_candidates: int | None = None) -> np.ndarray:
    candidate: set[int] = set()
    for cid in np.asarray(active_customer_ids, dtype=int):
        candidate.update(int(x) for x in board.customer_candidate_cs_ids[int(cid)].tolist())
    active_clusters = np.unique(board.cluster_labels[np.asarray(active_customer_ids, dtype=int)])
    for cluster_id in active_clusters:
        candidate.update(int(x) for x in board.cluster_candidate_cs_ids[int(cluster_id)].tolist())
    if not candidate:
        candidate.update(range(len(board.charging_stations)))
    return np.asarray(sorted(candidate), dtype=np.int32)


def activate_charging_stations(
    board: RegionBoard,
    active_customer_ids: np.ndarray,
    num_charging_stations: int,
    oracle: DistanceOracle,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = config.get("cs_activation", {})
    k = int(num_charging_stations)
    max_candidates = int(cfg.get("max_candidate_pool", max(4 * k, 64)))
    candidate_ids = build_candidate_pool(board, active_customer_ids, max_candidates=None)
    if candidate_ids.size < k:
        missing = [idx for idx in range(len(board.charging_stations)) if idx not in set(candidate_ids.tolist())]
        if missing:
            extra = np.asarray(missing[: k - candidate_ids.size], dtype=np.int32)
            candidate_ids = np.concatenate([candidate_ids, extra])
    active_customer_ids = np.asarray(active_customer_ids, dtype=int)
    customer_nodes = board.customer_node_ids[active_customer_ids]
    cs_nodes = board.cs_node_ids[candidate_ids]

    dist_customer_cs = oracle.matrix_between(customer_nodes, cs_nodes).astype(np.float32, copy=False)

    cluster_labels = board.cluster_labels[active_customer_ids]
    cluster_weights = np.ones(len(active_customer_ids), dtype=np.float32)
    weights = cluster_weights / max(float(cluster_weights.sum()), 1e-12)
    original_candidate_pool_size = int(candidate_ids.size)
    if candidate_ids.size > max_candidates:
        finite = np.isfinite(dist_customer_cs)
        safe = np.where(finite, dist_customer_cs, np.inf)
        mean_score = np.full(candidate_ids.size, np.inf, dtype=np.float64)
        p90_score = np.full(candidate_ids.size, np.inf, dtype=np.float64)
        for col in range(candidate_ids.size):
            col_dist = safe[:, col]
            finite_col = col_dist[np.isfinite(col_dist)]
            if finite_col.size:
                mean_score[col] = float(np.mean(finite_col))
                p90_score[col] = float(np.quantile(finite_col, 0.90))
        candidate_score = mean_score + 0.25 * p90_score
        keep = np.argsort(candidate_score, kind="mergesort")[:max_candidates]
        candidate_ids = candidate_ids[keep]
        cs_nodes = cs_nodes[keep]
        dist_customer_cs = dist_customer_cs[:, keep]

    selected_local: list[int] = []
    available = list(range(len(candidate_ids)))
    repulsion_km = float(cfg.get("repulsion_km", 1.5))
    alpha = float(cfg.get("alpha_mean", 1.0))
    beta = float(cfg.get("beta_p90", 0.6))
    gamma = float(cfg.get("gamma_max", 0.4))
    eta = float(cfg.get("eta_redundancy", 0.3))

    def objective(local_ids: list[int]) -> float:
        if not local_ids:
            return float("inf")
        d = np.min(dist_customer_cs[:, local_ids], axis=1)
        finite = np.isfinite(d)
        if not np.any(finite):
            return float("inf")
        penalty = float(cfg.get("unreachable_customer_penalty", 10000.0)) * float(np.mean(~finite))
        safe_d = d.copy()
        safe_d[~finite] = float(np.max(d[finite]) + penalty + 1.0)
        score = alpha * float(np.sum(weights * safe_d))
        score += beta * float(np.quantile(safe_d, 0.90))
        score += gamma * float(np.max(safe_d))
        score += penalty
        if len(local_ids) >= 2:
            cs_sel_nodes = cs_nodes[local_ids]
            pair = oracle.matrix_between(cs_sel_nodes, cs_sel_nodes)
            pair = pair[pair > 1e-9]
            min_pair = float(pair.min()) if pair.size else np.inf
            if np.isfinite(min_pair) and min_pair < repulsion_km:
                score += eta * (repulsion_km - min_pair) * 10.0
        return score

    for _ in range(k):
        best_idx = None
        best_score = float("inf")
        for local in available:
            score = objective(selected_local + [local])
            if score < best_score:
                best_score = score
                best_idx = local
        if best_idx is None:
            break
        selected_local.append(best_idx)
        available.remove(best_idx)

    if len(selected_local) < k:
        remaining = [x for x in range(len(candidate_ids)) if x not in selected_local]
        rng.shuffle(remaining)
        selected_local.extend(remaining[: k - len(selected_local)])

    selected = candidate_ids[np.asarray(selected_local[:k], dtype=int)]
    final_d = np.min(dist_customer_cs[:, selected_local[:k]], axis=1) if len(selected_local) else np.full(len(active_customer_ids), np.inf)
    finite_final = np.isfinite(final_d)
    safe_final = final_d[finite_final] if np.any(finite_final) else np.asarray([float("nan")])
    covered_clusters = len(np.unique(cluster_labels[finite_final]))
    metadata = {
        "policy": "graph_facility_location_greedy",
        "candidate_pool_size": int(candidate_ids.size),
        "original_candidate_pool_size": int(original_candidate_pool_size),
        "candidate_truncation_policy": "mean_plus_p90_customer_coverage" if original_candidate_pool_size > int(candidate_ids.size) else "none",
        "selected_count": int(selected.size),
        "mean_customer_to_nearest_cs_km": float(np.mean(safe_final)),
        "p90_customer_to_nearest_cs_km": float(np.quantile(safe_final, 0.90)),
        "max_customer_to_nearest_cs_km": float(np.max(safe_final)),
        "active_cluster_count": int(len(np.unique(cluster_labels))),
        "covered_cluster_count": int(covered_clusters),
        "unreachable_customer_rate_to_selected_cs": float(np.mean(~finite_final)),
        "objective": float(objective(selected_local[:k])) if selected_local else float("inf"),
    }
    return selected.astype(np.int32), metadata
