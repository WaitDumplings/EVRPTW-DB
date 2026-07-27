from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_STEM = (
    REPO_ROOT
    / "EVRPTW_Dataset"
    / "Geo_AC_v1"
    / "analysis_outputs"
    / "road_vs_euclidean_demo"
    / "road_metric_schematic_overview"
)
DEFAULT_NODE_LAYER_STEM = DEFAULT_OUTPUT_STEM.with_name("road_metric_schematic_node_layer")
DEFAULT_EDGE_LAYER_STEM = DEFAULT_OUTPUT_STEM.with_name("road_metric_schematic_edge_layer")
DEFAULT_NODE_EDGE_LAYER_STEM = DEFAULT_OUTPUT_STEM.with_name("road_metric_schematic_node_edge_layer")
DEFAULT_PPT_BASE_STEM = DEFAULT_OUTPUT_STEM.with_name("road_metric_schematic_ppt_base")
DEFAULT_DPI = 600
DEFAULT_FIG_WIDTH = 6.2
DEFAULT_FIG_HEIGHT = 3.65


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a vector road-metric schematic for paper overview panels.")
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--node-layer-stem", type=Path, default=DEFAULT_NODE_LAYER_STEM)
    parser.add_argument("--edge-layer-stem", type=Path, default=DEFAULT_EDGE_LAYER_STEM)
    parser.add_argument("--node-edge-layer-stem", type=Path, default=DEFAULT_NODE_EDGE_LAYER_STEM)
    parser.add_argument("--ppt-base-stem", type=Path, default=DEFAULT_PPT_BASE_STEM)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf", "svg"], choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--fig-width", type=float, default=DEFAULT_FIG_WIDTH)
    parser.add_argument("--fig-height", type=float, default=DEFAULT_FIG_HEIGHT)
    return parser.parse_args()


def _draw_obstacle(ax: plt.Axes, *, with_label: bool = True) -> None:
    x = np.linspace(0.45, 8.95, 180)
    upper = 3.18 + 0.16 * np.sin(1.45 * x + 0.7) + 0.06 * np.sin(3.2 * x)
    lower = 2.18 + 0.13 * np.sin(1.15 * x + 1.9) + 0.05 * np.sin(2.8 * x + 0.6)
    verts = [(float(xi), float(yi)) for xi, yi in zip(x, upper)]
    verts.extend((float(xi), float(yi)) for xi, yi in zip(x[::-1], lower[::-1]))
    verts.append(verts[0])
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    patch = PathPatch(
        MplPath(verts, codes),
        facecolor="#d8ecf6",
        edgecolor="#a9cadc",
        linewidth=1.0,
        alpha=0.78,
        zorder=1,
    )
    ax.add_patch(patch)
    for y in (2.33, 2.55, 2.77, 2.99):
        ax.plot([0.65, 8.7], [y, y + 0.12 * np.sin(y * 2.1)], color="white", linewidth=0.9, alpha=0.42, zorder=2)
    if with_label:
        ax.text(2.05, 2.72, "river", color="#5f7f91", fontsize=8.8, ha="center", va="center", zorder=3)


def _draw_minor_roads(
    ax: plt.Axes,
    road_path: np.ndarray,
    *,
    color: str = "#d5dbe2",
    line_width: float = 0.75,
    line_alpha: float = 0.82,
    connector_color: str = "#cbd3dd",
    connector_width: float = 1.05,
    connector_alpha: float = 0.9,
) -> None:
    roads = [
        [(0.85, 1.25), (2.1, 1.25), (3.05, 1.1), (4.25, 0.78), (5.55, 0.98), (6.75, 1.28), (8.25, 1.35)],
        [(1.15, 0.78), (1.15, 1.92)],
        [(2.0, 0.72), (2.0, 1.86)],
        [(3.25, 0.9), (3.25, 1.78)],
        [(7.15, 0.72), (7.15, 1.8)],
        [(8.0, 0.85), (8.0, 1.95)],
        [(1.0, 3.85), (2.65, 3.85), (3.6, 4.1), (4.75, 4.28), (5.5, 4.18), (6.75, 4.45), (8.55, 4.35)],
        [(7.05, 3.65), (7.05, 4.72)],
        [(8.05, 3.7), (8.05, 4.75)],
    ]
    for road in roads:
        xs, ys = zip(*road)
        ax.plot(xs, ys, color=color, linewidth=line_width, alpha=line_alpha, zorder=0)
    ax.plot(
        road_path[:, 0],
        road_path[:, 1],
        color=connector_color,
        linewidth=connector_width,
        alpha=connector_alpha,
        zorder=0,
        solid_capstyle="round",
    )


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


def _draw_arrow_on_polyline(ax: plt.Axes, polyline: np.ndarray, fraction: float, color: str, scale: float = 16.0) -> None:
    point, direction = _polyline_point_at_fraction(polyline, fraction)
    start = point - direction * 0.16
    end = point + direction * 0.16
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=0,
            color=color,
            zorder=9,
        )
    )


def _schematic_state() -> dict[str, np.ndarray]:
    depot = np.asarray([1.05, 1.03])
    customer = np.asarray([8.36, 4.35])
    cs = np.asarray([6.55, 1.18])
    other_customers = np.asarray(
        [
            [1.62, 1.25],
            [3.08, 1.08],
            [5.88, 1.05],
            [7.38, 1.22],
            [6.38, 4.34],
        ],
        dtype=float,
    )
    road_path = np.asarray(
        [
            depot,
            [1.72, 0.78],
            [2.85, 0.66],
            [4.25, 0.78],
            [5.55, 0.98],
            [6.75, 1.28],
            [7.9, 1.78],
            [8.58, 2.65],
            [8.72, 3.42],
            customer,
        ],
        dtype=float,
    )
    euclidean_path = np.vstack([depot, customer])
    return {
        "depot": depot,
        "customer": customer,
        "cs": cs,
        "other_customers": other_customers,
        "road_path": road_path,
        "euclidean_path": euclidean_path,
    }


def _finish_panel_axes(ax: plt.Axes, *, show_border: bool = True) -> None:
    ax.set_xlim(0.35, 9.25)
    ax.set_ylim(0.12, 5.35)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if show_border:
        ax.add_patch(Rectangle((0.38, 0.16), 8.84, 5.16, fill=False, edgecolor="#97a6b7", linewidth=1.15, zorder=20))


def _draw_node_layer(ax: plt.Axes) -> None:
    state = _schematic_state()
    depot = state["depot"]
    customer = state["customer"]
    cs = state["cs"]
    other_customers = state["other_customers"]

    ax.scatter([depot[0]], [depot[1]], marker="s", s=132, color="#111827", zorder=8)
    ax.scatter(
        other_customers[:, 0],
        other_customers[:, 1],
        marker="o",
        s=54,
        color="#e11d74",
        alpha=0.2,
        edgecolors="none",
        zorder=4,
    )
    ax.scatter([customer[0]], [customer[1]], marker="o", s=168, color="#e11d74", zorder=8)
    ax.scatter([cs[0]], [cs[1]], marker="D", s=98, color="#059669", edgecolors="white", linewidths=0.9, zorder=8)
    ax.text(depot[0] - 0.18, depot[1] - 0.30, "depot i", color="#111827", fontsize=9.8, ha="center", va="top")
    ax.text(customer[0], customer[1] + 0.34, "customer j", color="#e11d74", fontsize=9.4, ha="center", va="bottom")
    ax.text(cs[0] + 0.26, cs[1] - 0.30, "charging\nstation", color="#047857", fontsize=8.2, ha="left", va="top")
    _finish_panel_axes(ax)


def _draw_edge_layer(ax: plt.Axes) -> None:
    state = _schematic_state()
    road_path = state["road_path"]

    _draw_minor_roads(ax, road_path)
    _draw_obstacle(ax)
    _finish_panel_axes(ax)


def _draw_node_edge_layer(ax: plt.Axes) -> None:
    state = _schematic_state()
    depot = state["depot"]
    customer = state["customer"]
    cs = state["cs"]
    other_customers = state["other_customers"]
    road_path = state["road_path"]

    _draw_minor_roads(ax, road_path)
    _draw_obstacle(ax)
    ax.scatter([depot[0]], [depot[1]], marker="s", s=132, color="#111827", zorder=8)
    ax.scatter(
        other_customers[:, 0],
        other_customers[:, 1],
        marker="o",
        s=54,
        color="#e11d74",
        alpha=0.2,
        edgecolors="none",
        zorder=4,
    )
    ax.scatter([customer[0]], [customer[1]], marker="o", s=168, color="#e11d74", zorder=8)
    ax.scatter([cs[0]], [cs[1]], marker="D", s=98, color="#059669", edgecolors="white", linewidths=0.9, zorder=8)
    ax.text(depot[0] - 0.18, depot[1] - 0.30, "depot i", color="#111827", fontsize=9.8, ha="center", va="top")
    ax.text(customer[0], customer[1] + 0.34, "customer j", color="#e11d74", fontsize=9.4, ha="center", va="bottom")
    ax.text(cs[0] + 0.26, cs[1] - 0.30, "charging\nstation", color="#047857", fontsize=8.2, ha="left", va="top")
    _finish_panel_axes(ax, show_border=False)


def _draw_schematic(ax: plt.Axes) -> None:
    state = _schematic_state()
    depot = state["depot"]
    customer = state["customer"]
    cs = state["cs"]
    other_customers = state["other_customers"]
    road_path = state["road_path"]
    euclidean_path = state["euclidean_path"]
    euclidean_color = (0.831, 0.267, 0.180, 0.30)

    _draw_minor_roads(ax, road_path)
    _draw_obstacle(ax)
    ax.plot(road_path[:, 0], road_path[:, 1], color="#1f5fbf", linewidth=3.8, zorder=6, solid_capstyle="round")
    for frac in (0.32, 0.61, 0.86):
        _draw_arrow_on_polyline(ax, road_path, frac, "#1f5fbf", scale=20.0)

    ax.plot(
        [depot[0], customer[0]],
        [depot[1], customer[1]],
        color=euclidean_color,
        linestyle=(0, (5, 3)),
        linewidth=2.85,
        zorder=5,
    )
    _draw_arrow_on_polyline(ax, euclidean_path, 0.61, euclidean_color, scale=19.5)

    ax.scatter([depot[0]], [depot[1]], marker="s", s=132, color="#111827", zorder=8)
    ax.scatter(
        other_customers[:, 0],
        other_customers[:, 1],
        marker="o",
        s=54,
        color="#e11d74",
        alpha=0.2,
        edgecolors="none",
        zorder=4,
    )
    ax.scatter([customer[0]], [customer[1]], marker="o", s=168, color="#e11d74", zorder=8)
    ax.scatter([cs[0]], [cs[1]], marker="D", s=98, color="#059669", edgecolors="white", linewidths=0.9, zorder=8)

    ax.text(depot[0] - 0.18, depot[1] - 0.30, "depot i", color="#111827", fontsize=9.8, ha="center", va="top")
    ax.text(customer[0], customer[1] + 0.34, "customer j", color="#e11d74", fontsize=9.4, ha="center", va="bottom")
    ax.text(cs[0] + 0.26, cs[1] - 0.30, "charging\nstation", color="#047857", fontsize=8.2, ha="left", va="top")

    ax.text(4.18, 3.39, "Euclidean proxy", color=euclidean_color, fontsize=9.1, rotation=23, ha="center", va="center")
    ax.text(4.7, 0.58, r"road-network shortest path $d_{ij}$", color="#174ea6", fontsize=9.3, weight="bold", ha="center", va="center")
    ax.text(1.0, 5.08, "Real-road instance", color="#1f2937", fontsize=10.8, weight="bold", ha="left", va="center")
    ax.text(1.0, 4.83, r"$D^{R}=[d_{ij}],\;D^{R}\ne (D^{R})^\top$", color="#1f2937", fontsize=9.4, ha="left", va="center")
    _finish_panel_axes(ax)


def _draw_ppt_base(ax: plt.Axes) -> None:
    state = _schematic_state()
    depot = state["depot"]
    customer = state["customer"]
    cs = state["cs"]
    other_customers = state["other_customers"]
    road_path = state["road_path"]
    euclidean_path = state["euclidean_path"]
    euclidean_color = (0.831, 0.267, 0.180, 0.30)

    _draw_minor_roads(
        ax,
        road_path,
        color="#b8c2cf",
        line_width=1.15,
        line_alpha=0.92,
        connector_color="#b1bcc9",
        connector_width=1.45,
        connector_alpha=0.95,
    )
    _draw_obstacle(ax, with_label=False)
    ax.plot(road_path[:, 0], road_path[:, 1], color="#1f5fbf", linewidth=3.8, zorder=6, solid_capstyle="round")
    for frac in (0.32, 0.61, 0.86):
        _draw_arrow_on_polyline(ax, road_path, frac, "#1f5fbf", scale=20.0)

    ax.plot(
        [depot[0], customer[0]],
        [depot[1], customer[1]],
        color=euclidean_color,
        linestyle=(0, (5, 3)),
        linewidth=2.85,
        zorder=5,
    )
    _draw_arrow_on_polyline(ax, euclidean_path, 0.61, euclidean_color, scale=19.5)

    ax.scatter([depot[0]], [depot[1]], marker="s", s=132, color="#111827", zorder=8)
    ax.scatter(
        other_customers[:, 0],
        other_customers[:, 1],
        marker="o",
        s=66,
        color="#d61f69",
        alpha=0.34,
        edgecolors="none",
        zorder=4,
    )
    ax.scatter([customer[0]], [customer[1]], marker="o", s=168, color="#e11d74", zorder=8)
    ax.scatter([cs[0]], [cs[1]], marker="D", s=98, color="#059669", edgecolors="white", linewidths=0.9, zorder=8)
    _finish_panel_axes(ax, show_border=False)


def main() -> None:
    args = parse_args()
    figures = [
        (args.output_stem, _draw_schematic),
        (args.node_layer_stem, _draw_node_layer),
        (args.edge_layer_stem, _draw_edge_layer),
        (args.node_edge_layer_stem, _draw_node_edge_layer),
        (args.ppt_base_stem, _draw_ppt_base),
    ]
    for output_stem, drawer in figures:
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(float(args.fig_width), float(args.fig_height)), dpi=int(args.dpi))
        drawer(ax)
        fig.tight_layout(pad=0.05)
        for fmt in args.formats:
            out = output_stem.with_suffix(f".{fmt}")
            fig.savefig(out, dpi=int(args.dpi), transparent=False)
            print(out)
        plt.close(fig)


if __name__ == "__main__":
    main()
