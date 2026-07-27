from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = REPO_ROOT / "EVRPTW_Dataset" / "Geo_AC_v1" / "release" / "dataset_v1" / "raw_data"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "EVRPTW_Dataset"
    / "Geo_AC_v1"
    / "analysis_outputs"
    / "road_vs_euclidean_demo"
    / "road_vs_euclidean_demo.png"
)
DEFAULT_TERRITORIES = (
    "austin_tx_travis_county",
    "boston_ma_middlesex_county",
    "cook_il_chicago_core",
    "new_york_ny_queens_county",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a small road shortest-path vs Euclidean-distance demo.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--territory-id", action="append", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--node-output", type=Path, default=None)
    parser.add_argument("--node-euclidean-output", type=Path, default=None)
    parser.add_argument("--edge-output", type=Path, default=None)
    parser.add_argument("--max-customer-sample", type=int, default=50_000)
    parser.add_argument("--min-euclidean-km", type=float, default=1.2)
    parser.add_argument("--max-euclidean-km", type=float, default=8.0)
    parser.add_argument("--dijkstra-cutoff-km", type=float, default=35.0)
    parser.add_argument("--max-depot-snap-km", type=float, default=0.45)
    parser.add_argument("--max-customer-snap-km", type=float, default=0.25)
    parser.add_argument("--pad-km", type=float, default=0.9)
    parser.add_argument("--charger-route-radius-km", type=float, default=1.2)
    parser.add_argument("--max-context-customers", type=int, default=90)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for node_id in nodes["node_id"].astype(str):
        graph.add_node(node_id)
    for row in edges[["u", "v", "length_km"]].itertuples(index=False):
        u = str(row.u)
        v = str(row.v)
        w = float(row.length_km)
        if not np.isfinite(w) or w <= 0:
            continue
        if graph.has_edge(u, v):
            if w < float(graph[u][v]["weight"]):
                graph[u][v]["weight"] = w
        else:
            graph.add_edge(u, v, weight=w)
    return graph


def nearest_node_id(nodes: pd.DataFrame, tree: cKDTree, xy: np.ndarray) -> tuple[str, float]:
    dist, idx = tree.query(np.asarray(xy, dtype=float))
    return str(nodes.iloc[int(idx)]["node_id"]), float(dist)


def choose_customer_snap(row: pd.Series, dist: dict[str, float]) -> tuple[str | None, float]:
    candidates = [str(row.get("snap_edge_u", "")), str(row.get("snap_edge_v", ""))]
    best_node: str | None = None
    best_dist = float("inf")
    for node in candidates:
        value = float(dist.get(node, float("inf")))
        if value < best_dist:
            best_node = node
            best_dist = value
    return best_node, best_dist


def find_demo(raw_root: Path, territories: list[str], args: argparse.Namespace) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for territory_id in territories:
        root = raw_root / territory_id
        nodes = pd.read_csv(root / "road_nodes.csv")
        edges = pd.read_csv(root / "road_edges.csv")
        customers = pd.read_csv(root / "latent_customer.csv")
        chargers = pd.read_csv(root / "charging_station.csv")
        depots = pd.read_csv(root / "depot_candidate.csv")
        if len(customers) > int(args.max_customer_sample):
            customers = customers.sample(int(args.max_customer_sample), random_state=int(args.seed))

        graph = load_graph(nodes, edges)
        tree = cKDTree(nodes[["x_km", "y_km"]].to_numpy(dtype=float))
        node_xy = {
            str(row.node_id): (float(row.x_km), float(row.y_km))
            for row in nodes[["node_id", "x_km", "y_km"]].itertuples(index=False)
        }
        customer_xy = customers[["x_km", "y_km"]].to_numpy(dtype=float)
        for depot in depots.itertuples(index=False):
            depot_xy = np.asarray([float(depot.x_km), float(depot.y_km)], dtype=float)
            depot_node, depot_snap_km = nearest_node_id(nodes, tree, depot_xy)
            if depot_snap_km > float(args.max_depot_snap_km):
                continue
            try:
                dist = nx.single_source_dijkstra_path_length(
                    graph,
                    depot_node,
                    cutoff=float(args.dijkstra_cutoff_km),
                    weight="weight",
                )
            except nx.NetworkXNoPath:
                continue
            euclidean = np.linalg.norm(customer_xy - depot_xy.reshape(1, 2), axis=1)
            mask = (euclidean >= float(args.min_euclidean_km)) & (euclidean <= float(args.max_euclidean_km))
            if not np.any(mask):
                continue
            subset = customers.loc[mask].copy()
            subset_euclidean = euclidean[mask]
            for (_, customer), euclid in zip(subset.iterrows(), subset_euclidean):
                customer_node, road_km = choose_customer_snap(customer, dist)
                if customer_node is None or not np.isfinite(road_km) or road_km <= 0:
                    continue
                snap_xy = np.asarray([float(customer.snap_x_km), float(customer.snap_y_km)], dtype=float)
                node_xy_arr = np.asarray(node_xy.get(customer_node, (np.nan, np.nan)), dtype=float)
                customer_snap_km = float(np.linalg.norm(snap_xy - node_xy_arr))
                if not np.isfinite(customer_snap_km) or customer_snap_km > float(args.max_customer_snap_km):
                    continue
                ratio = float(road_km) / max(float(euclid), 1e-9)
                if ratio < 1.35:
                    continue
                route_bbox = np.vstack([
                    depot_xy,
                    np.asarray([float(customer.x_km), float(customer.y_km)]),
                ])
                span = float(np.max(route_bbox.max(axis=0) - route_bbox.min(axis=0)))
                score = ratio - 0.015 * max(span - 4.0, 0.0)
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "territory_id": territory_id,
                        "nodes": nodes,
                        "edges": edges,
                        "customers": customers,
                        "chargers": chargers,
                        "depots": depots,
                        "graph": graph,
                        "node_xy": node_xy,
                        "depot": depot,
                        "customer": customer,
                        "depot_node": depot_node,
                        "depot_snap_km": float(depot_snap_km),
                        "customer_node": customer_node,
                        "customer_snap_km": float(customer_snap_km),
                        "road_km": float(road_km),
                        "euclidean_km": float(euclid),
                        "ratio": ratio,
                    }
    if best is None:
        raise RuntimeError("Could not find a suitable road-vs-Euclidean demo pair.")
    return best


def nearest_charger(chargers: pd.DataFrame, midpoint: np.ndarray) -> pd.Series:
    coords = chargers[["x_km", "y_km"]].to_numpy(dtype=float)
    idx = int(np.argmin(np.linalg.norm(coords - midpoint.reshape(1, 2), axis=1)))
    return chargers.iloc[idx]


def _distance_to_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    polyline = np.asarray(polyline, dtype=float)
    out = np.full(points.shape[0], np.inf, dtype=float)
    for start, end in zip(polyline[:-1], polyline[1:]):
        seg = end - start
        denom = float(np.dot(seg, seg))
        if denom <= 1e-12:
            cand = np.linalg.norm(points - start.reshape(1, 2), axis=1)
        else:
            t = np.clip(((points - start.reshape(1, 2)) @ seg) / denom, 0.0, 1.0)
            proj = start.reshape(1, 2) + t.reshape(-1, 1) * seg.reshape(1, 2)
            cand = np.linalg.norm(points - proj, axis=1)
        out = np.minimum(out, cand)
    return out


def _polyline_point_at_fraction(polyline: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    seg = np.diff(polyline, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    total = float(lengths.sum())
    if total <= 1e-12:
        return polyline[0], np.asarray([1.0, 0.0])
    target = float(np.clip(fraction, 0.0, 1.0)) * total
    accum = 0.0
    for idx, length in enumerate(lengths):
        if accum + float(length) >= target:
            local = (target - accum) / max(float(length), 1e-12)
            point = polyline[idx] + local * (polyline[idx + 1] - polyline[idx])
            direction = polyline[idx + 1] - polyline[idx]
            direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
            return point, direction
        accum += float(length)
    direction = polyline[-1] - polyline[-2]
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    return polyline[-1], direction


def _best_route_label_point(route_xy: np.ndarray) -> np.ndarray:
    x = route_xy[:, 0]
    y = route_xy[:, 1]
    x_norm = (x - float(x.min())) / max(float(x.max() - x.min()), 1e-12)
    y_norm = (y - float(y.min())) / max(float(y.max() - y.min()), 1e-12)
    left_bridge_mask = (x_norm < 0.42) & (y_norm > 0.32) & (y_norm < 0.64)
    if np.any(left_bridge_mask):
        target = np.asarray([0.23, 0.47], dtype=float)
        candidate_idx = np.where(left_bridge_mask)[0]
        local = np.column_stack([x_norm[candidate_idx], y_norm[candidate_idx]])
        score = np.linalg.norm(local - target.reshape(1, 2), axis=1)
        return route_xy[int(candidate_idx[int(np.argmin(score))])]
    score = 0.58 * (1.0 - x_norm) + 0.42 * (1.0 - np.abs(y_norm - 0.48))
    idx = int(np.argmax(score))
    return route_xy[idx]


def _bridge_symbol_candidates(route_xy: np.ndarray, road_segments: list[tuple[tuple[float, float], tuple[float, float]]]) -> list[tuple[np.ndarray, np.ndarray]]:
    road_midpoints = np.asarray(
        [
            [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
            for a, b in road_segments
        ],
        dtype=float,
    )
    scored: list[tuple[float, int]] = []
    for idx, (start, end) in enumerate(zip(route_xy[:-1], route_xy[1:])):
        vec = end - start
        length = float(np.linalg.norm(vec))
        if length < 0.12:
            continue
        mid = 0.5 * (start + end)
        if road_midpoints.size:
            local_density = int(np.count_nonzero(np.linalg.norm(road_midpoints - mid.reshape(1, 2), axis=1) < 0.45))
        else:
            local_density = 0
        score = length / float(local_density + 2)
        scored.append((score, idx))
    scored.sort(reverse=True)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for _, idx in scored:
        start = route_xy[idx]
        end = route_xy[idx + 1]
        mid = 0.5 * (start + end)
        if any(np.linalg.norm(mid - old_mid) < 1.8 for old_mid, _ in out):
            continue
        direction = end - start
        direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
        out.append((mid, direction))
        if len(out) >= 2:
            break
    return out


def _draw_bridge_symbol(ax: plt.Axes, center: np.ndarray, direction: np.ndarray, size: float = 0.66) -> None:
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    normal = np.asarray([-direction[1], direction[0]], dtype=float)
    color = "#4b5563"
    halo = "white"
    rail_width = size * 0.13
    for side in (-1.0, 1.0):
        rail_center = center + normal * rail_width * side
        deck_a = rail_center - direction * size
        deck_b = rail_center + direction * size
        ax.plot(
            [deck_a[0], deck_b[0]],
            [deck_a[1], deck_b[1]],
            color=halo,
            linewidth=4.2,
            zorder=9,
            solid_capstyle="round",
        )
        ax.plot(
            [deck_a[0], deck_b[0]],
            [deck_a[1], deck_b[1]],
            color=color,
            linewidth=2.0,
            zorder=10,
            solid_capstyle="round",
        )
    for offset in (-0.72, -0.36, 0.0, 0.36, 0.72):
        p = center + direction * size * offset
        a = p - normal * size * 0.28
        b = p + normal * size * 0.28
        ax.plot([a[0], b[0]], [a[1], b[1]], color=halo, linewidth=4.8, zorder=9, solid_capstyle="round")
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=2.2, zorder=10, solid_capstyle="round")


def _collect_demo_plot_state(demo: dict[str, Any], pad_km: float, args: argparse.Namespace) -> dict[str, Any]:
    graph: nx.Graph = demo["graph"]
    node_xy: dict[str, tuple[float, float]] = demo["node_xy"]
    route_nodes = nx.shortest_path(graph, demo["depot_node"], demo["customer_node"], weight="weight")
    route_xy = np.asarray([node_xy[node] for node in route_nodes], dtype=float)
    depot_xy = np.asarray([float(demo["depot"].x_km), float(demo["depot"].y_km)])
    customer_xy = np.asarray([float(demo["customer"].x_km), float(demo["customer"].y_km)])
    depot_snap_xy = np.asarray(node_xy[demo["depot_node"]], dtype=float)
    customer_snap_xy = np.asarray(node_xy[demo["customer_node"]], dtype=float)

    all_xy = np.vstack([route_xy, depot_xy, customer_xy, depot_snap_xy, customer_snap_xy])
    xmin, ymin = all_xy.min(axis=0) - float(pad_km)
    xmax, ymax = all_xy.max(axis=0) + float(pad_km)

    nodes = demo["nodes"].copy()
    node_pos = nodes.set_index(nodes["node_id"].astype(str))[["x_km", "y_km"]]
    road_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for row in demo["edges"][["u", "v"]].itertuples(index=False):
        u = str(row.u)
        v = str(row.v)
        if u not in node_pos.index or v not in node_pos.index:
            continue
        ux, uy = node_pos.loc[u]
        vx, vy = node_pos.loc[v]
        if (
            max(float(ux), float(vx)) < xmin
            or min(float(ux), float(vx)) > xmax
            or max(float(uy), float(vy)) < ymin
            or min(float(uy), float(vy)) > ymax
        ):
            continue
        road_segments.append(((float(ux), float(uy)), (float(vx), float(vy))))

    customers = demo["customers"]
    customer_coords = customers[["x_km", "y_km"]].to_numpy(dtype=float)
    in_view_customers = (
        (customer_coords[:, 0] >= xmin)
        & (customer_coords[:, 0] <= xmax)
        & (customer_coords[:, 1] >= ymin)
        & (customer_coords[:, 1] <= ymax)
    )
    context_customers = customers.loc[in_view_customers].copy()
    if len(context_customers) > int(args.max_context_customers):
        context_customers = context_customers.sample(int(args.max_context_customers), random_state=int(args.seed))

    chargers = demo["chargers"]
    charger_coords = chargers[["x_km", "y_km"]].to_numpy(dtype=float)
    in_view_chargers = (
        (charger_coords[:, 0] >= xmin)
        & (charger_coords[:, 0] <= xmax)
        & (charger_coords[:, 1] >= ymin)
        & (charger_coords[:, 1] <= ymax)
    )
    near_route = _distance_to_polyline(charger_coords, route_xy) <= float(args.charger_route_radius_km)
    route_chargers = chargers.loc[in_view_chargers & near_route].copy()
    if route_chargers.empty:
        visible = chargers.loc[in_view_chargers].copy()
        if not visible.empty:
            visible_coords = visible[["x_km", "y_km"]].to_numpy(dtype=float)
            idx = int(np.argmin(_distance_to_polyline(visible_coords, route_xy)))
            route_chargers = visible.iloc[[idx]].copy()

    return {
        "route_xy": route_xy,
        "depot_xy": depot_xy,
        "customer_xy": customer_xy,
        "depot_snap_xy": depot_snap_xy,
        "customer_snap_xy": customer_snap_xy,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "road_segments": road_segments,
        "context_customers": context_customers,
        "route_chargers": route_chargers,
    }


def _finish_demo_axes(ax: plt.Axes, state: dict[str, Any], title: str) -> None:
    ax.set_xlim(state["xmin"], state["xmax"])
    ax.set_ylim(state["ymin"], state["ymax"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#9aa3af")
        spine.set_linewidth(1.0)
    ax.grid(False)


def plot_demo_layers(
    demo: dict[str, Any],
    node_output: Path | None,
    node_euclidean_output: Path | None,
    edge_output: Path | None,
    pad_km: float,
    args: argparse.Namespace,
) -> None:
    if node_output is None and node_euclidean_output is None and edge_output is None:
        return
    state = _collect_demo_plot_state(demo, pad_km, args)
    depot_xy = state["depot_xy"]
    customer_xy = state["customer_xy"]

    def draw_nodes(ax: plt.Axes) -> None:
        context_customers = state["context_customers"]
        route_chargers = state["route_chargers"]
        if not context_customers.empty:
            ax.scatter(
                context_customers["x_km"],
                context_customers["y_km"],
                marker="o",
                s=24,
                color="#f3a1c0",
                alpha=0.46,
                edgecolors="none",
                label="Other customers",
                zorder=3,
            )
        if not route_chargers.empty:
            ax.scatter(
                route_chargers["x_km"],
                route_chargers["y_km"],
                marker="^",
                s=74,
                color="#059669",
                alpha=0.94,
                edgecolors="white",
                linewidths=0.6,
                label="Charging stations",
                zorder=6,
            )
        ax.scatter([depot_xy[0]], [depot_xy[1]], marker="s", s=122, color="#111827", label="Depot", zorder=7)
        ax.scatter([customer_xy[0]], [customer_xy[1]], marker="o", s=140, color="#e11d74", label="Target customer", zorder=7)

    if node_output is not None:
        node_output.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9.5, 8.2), dpi=220)
        draw_nodes(ax)
        _finish_demo_axes(ax, state, "NY Queens County: Node Layer")
        ax.legend(loc="lower right", framealpha=0.94, fontsize=9.5)
        fig.tight_layout(pad=0.25)
        fig.savefig(node_output)
        plt.close(fig)

    if node_euclidean_output is not None:
        node_euclidean_output.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9.5, 8.2), dpi=220)
        ax.plot(
            [depot_xy[0], customer_xy[0]],
            [depot_xy[1], customer_xy[1]],
            color="#d4442e",
            linestyle="--",
            linewidth=2.35,
            label="Euclidean straight line",
            zorder=4,
        )
        draw_nodes(ax)
        straight_mid = 0.5 * (depot_xy + customer_xy)
        ax.text(
            straight_mid[0] + 0.15,
            straight_mid[1],
            f"{demo['euclidean_km']:.2f} km",
            color="#b42318",
            fontsize=10.5,
            weight="bold",
            ha="left",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.8},
            zorder=8,
        )
        _finish_demo_axes(ax, state, "NY Queens County: Node + Euclidean Prior")
        ax.legend(loc="lower right", framealpha=0.94, fontsize=9.5)
        fig.tight_layout(pad=0.25)
        fig.savefig(node_euclidean_output)
        plt.close(fig)

    if edge_output is not None:
        edge_output.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9.5, 8.2), dpi=220)
        for (ux, uy), (vx, vy) in state["road_segments"]:
            ax.plot([ux, vx], [uy, vy], color="#aeb7c3", linewidth=0.55, alpha=0.88, zorder=1)
        _finish_demo_axes(ax, state, "NY Queens County: Road Network Layer")
        fig.tight_layout(pad=0.25)
        fig.savefig(edge_output)
        plt.close(fig)


def plot_demo(demo: dict[str, Any], output: Path, pad_km: float, args: argparse.Namespace) -> None:
    graph: nx.Graph = demo["graph"]
    node_xy: dict[str, tuple[float, float]] = demo["node_xy"]
    route_nodes = nx.shortest_path(graph, demo["depot_node"], demo["customer_node"], weight="weight")
    route_xy = np.asarray([node_xy[node] for node in route_nodes], dtype=float)
    depot_xy = np.asarray([float(demo["depot"].x_km), float(demo["depot"].y_km)])
    customer_xy = np.asarray([float(demo["customer"].x_km), float(demo["customer"].y_km)])
    depot_snap_xy = np.asarray(node_xy[demo["depot_node"]], dtype=float)
    customer_snap_xy = np.asarray(node_xy[demo["customer_node"]], dtype=float)

    all_xy = np.vstack([route_xy, depot_xy, customer_xy, depot_snap_xy, customer_snap_xy])
    xmin, ymin = all_xy.min(axis=0) - float(pad_km)
    xmax, ymax = all_xy.max(axis=0) + float(pad_km)

    nodes = demo["nodes"].copy()
    node_pos = nodes.set_index(nodes["node_id"].astype(str))[["x_km", "y_km"]]
    edges = demo["edges"]
    road_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for row in edges[["u", "v"]].itertuples(index=False):
        u = str(row.u)
        v = str(row.v)
        if u not in node_pos.index or v not in node_pos.index:
            continue
        ux, uy = node_pos.loc[u]
        vx, vy = node_pos.loc[v]
        if (
            max(float(ux), float(vx)) < xmin
            or min(float(ux), float(vx)) > xmax
            or max(float(uy), float(vy)) < ymin
            or min(float(uy), float(vy)) > ymax
        ):
            continue
        road_segments.append(((float(ux), float(uy)), (float(vx), float(vy))))

    customers = demo["customers"]
    customer_coords = customers[["x_km", "y_km"]].to_numpy(dtype=float)
    in_view_customers = (
        (customer_coords[:, 0] >= xmin)
        & (customer_coords[:, 0] <= xmax)
        & (customer_coords[:, 1] >= ymin)
        & (customer_coords[:, 1] <= ymax)
    )
    context_customers = customers.loc[in_view_customers].copy()
    if len(context_customers) > int(args.max_context_customers):
        context_customers = context_customers.sample(int(args.max_context_customers), random_state=int(args.seed))

    chargers = demo["chargers"]
    charger_coords = chargers[["x_km", "y_km"]].to_numpy(dtype=float)
    in_view_chargers = (
        (charger_coords[:, 0] >= xmin)
        & (charger_coords[:, 0] <= xmax)
        & (charger_coords[:, 1] >= ymin)
        & (charger_coords[:, 1] <= ymax)
    )
    near_route = _distance_to_polyline(charger_coords, route_xy) <= float(args.charger_route_radius_km)
    route_chargers = chargers.loc[in_view_chargers & near_route].copy()
    if route_chargers.empty:
        # Keep at least the nearest visible CS so the legend and marker remain meaningful.
        visible = chargers.loc[in_view_chargers].copy()
        if not visible.empty:
            visible_coords = visible[["x_km", "y_km"]].to_numpy(dtype=float)
            idx = int(np.argmin(_distance_to_polyline(visible_coords, route_xy)))
            route_chargers = visible.iloc[[idx]].copy()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 8.2), dpi=220)
    for (ux, uy), (vx, vy) in road_segments:
        ax.plot([ux, vx], [uy, vy], color="#d3d8df", linewidth=0.42, alpha=0.72, zorder=1)

    if not context_customers.empty:
        ax.scatter(
            context_customers["x_km"],
            context_customers["y_km"],
            marker="o",
            s=20,
            color="#f3a1c0",
            alpha=0.42,
            edgecolors="none",
            label="Other customers",
            zorder=3,
        )
    if not route_chargers.empty:
        ax.scatter(
            route_chargers["x_km"],
            route_chargers["y_km"],
            marker="^",
            s=66,
            color="#059669",
            alpha=0.92,
            edgecolors="white",
            linewidths=0.6,
            label="Charging stations",
            zorder=6,
        )

    ax.plot(route_xy[:, 0], route_xy[:, 1], color="#1f5fbf", linewidth=3.2, label="Road shortest path", zorder=5)
    arrow_start, _ = _polyline_point_at_fraction(route_xy, 0.57)
    arrow_end, _ = _polyline_point_at_fraction(route_xy, 0.62)
    ax.add_patch(
        FancyArrowPatch(
            arrow_start,
            arrow_end,
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=0,
            color="#1f5fbf",
            zorder=8,
        )
    )
    ax.plot(
        [depot_xy[0], depot_snap_xy[0]],
        [depot_xy[1], depot_snap_xy[1]],
        color="#1f5fbf",
        linewidth=1.6,
        alpha=0.75,
        zorder=5,
    )
    ax.plot(
        [customer_xy[0], customer_snap_xy[0]],
        [customer_xy[1], customer_snap_xy[1]],
        color="#1f5fbf",
        linewidth=1.6,
        alpha=0.75,
        zorder=5,
    )
    ax.plot(
        [depot_xy[0], customer_xy[0]],
        [depot_xy[1], customer_xy[1]],
        color="#d4442e",
        linestyle="--",
        linewidth=2.35,
        label="Euclidean straight line",
        zorder=4,
    )
    ax.scatter([depot_xy[0]], [depot_xy[1]], marker="s", s=110, color="#111827", label="Depot", zorder=7)
    ax.scatter([customer_xy[0]], [customer_xy[1]], marker="o", s=125, color="#e11d74", label="Target customer", zorder=7)

    ax.text(
        xmin + 0.48 * (xmax - xmin),
        ymin + 0.56 * (ymax - ymin),
        "Jamaica Bay",
        color="#6b7280",
        fontsize=34,
        fontstyle="italic",
        alpha=0.31,
        rotation=-16,
        ha="center",
        va="center",
        zorder=2,
    )

    route_label_pos = _best_route_label_point(route_xy)
    ax.text(
        route_label_pos[0] - 0.52,
        route_label_pos[1] - 0.44,
        f"{demo['road_km']:.2f} km",
        color="#174ea6",
        fontsize=10.5,
        weight="bold",
        ha="right",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.8},
        zorder=8,
    )
    straight_mid = 0.5 * (depot_xy + customer_xy)
    ax.text(
        straight_mid[0] + 0.15,
        straight_mid[1],
        f"{demo['euclidean_km']:.2f} km",
        color="#b42318",
        fontsize=10.5,
        weight="bold",
        ha="left",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.8},
        zorder=8,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#9aa3af")
        spine.set_linewidth(1.0)
    ax.legend(loc="lower right", framealpha=0.94, fontsize=9.5)
    ax.grid(False)
    fig.tight_layout(pad=0.25)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    territories = args.territory_id or list(DEFAULT_TERRITORIES)
    demo = find_demo(args.raw_root, territories, args)
    plot_demo(demo, args.output, float(args.pad_km), args)
    plot_demo_layers(demo, args.node_output, args.node_euclidean_output, args.edge_output, float(args.pad_km), args)
    print(
        {
            "output": str(args.output),
            "node_output": str(args.node_output) if args.node_output is not None else None,
            "node_euclidean_output": str(args.node_euclidean_output) if args.node_euclidean_output is not None else None,
            "edge_output": str(args.edge_output) if args.edge_output is not None else None,
            "territory_id": demo["territory_id"],
            "road_km": round(float(demo["road_km"]), 4),
            "euclidean_km": round(float(demo["euclidean_km"]), 4),
            "ratio": round(float(demo["ratio"]), 4),
        }
    )


if __name__ == "__main__":
    main()
