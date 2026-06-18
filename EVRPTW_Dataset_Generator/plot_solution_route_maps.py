"""Plot solved EVRPTW instances selected from a solver summary CSV.

The release bundles are pickle streams: a header record followed by one record
per instance. This script scans the stream, selects instances by `instance_id`,
and plots depot, active customers, active charging stations, and route chords.
"""

from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import math
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd


def _parse_routes(value: Any) -> list[list[int]]:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
    return [[int(node) for node in route] for route in parsed]


def _selected_summary_rows(summary_csv: Path, min_routes: int) -> list[dict[str, Any]]:
    df = pd.read_csv(summary_csv)
    rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        routes = _parse_routes(row.get("routes_json"))
        if len(routes) >= min_routes:
            rows.append(
                {
                    "instance_id": str(row["instance_id"]),
                    "routes": routes,
                    "route_count": len(routes),
                    "objective_distance_km": row.get("objective_distance_km"),
                    "vehicle_count": row.get("vehicle_count"),
                    "status_name": row.get("status_name"),
                    "runtime_s": row.get("runtime_s"),
                }
            )
    rows.sort(key=lambda r: (str(r["instance_id"])))
    return rows


def _read_selected_instances(bundle_path: Path, selected_ids: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    with bundle_path.open("rb") as f:
        header = pickle.load(f)
        n = int(header["num_instances"])
        for _ in range(n):
            instance = pickle.load(f)
            instance_id = str(instance.get("instance_id"))
            if instance_id in selected_ids:
                found[instance_id] = instance
                if len(found) == len(selected_ids):
                    break
    missing = sorted(selected_ids.difference(found))
    if missing:
        raise ValueError(f"Selected instance IDs were not found in bundle: {missing[:10]}")
    return found


def _terminal_coordinates(instance: dict[str, Any]) -> np.ndarray:
    depot = np.asarray(instance["depot"], dtype=float).reshape(1, 2)
    customers = np.asarray(instance["customers"], dtype=float)
    charging_stations = np.asarray(instance["charging_stations"], dtype=float)
    return np.vstack([depot, customers, charging_stations])


def _route_colors(num_routes: int) -> list[Any]:
    tab20 = list(plt.get_cmap("tab20").colors)
    high_contrast_order = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    colors = [tab20[i] for i in high_contrast_order]
    if num_routes <= len(colors):
        return colors[:num_routes]
    hsv = plt.get_cmap("hsv")
    return colors + [hsv(i / max(num_routes - len(colors), 1)) for i in range(num_routes - len(colors))]


def _add_route_connector(
    axis: Any,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    route_idx: int,
    segment_idx: int,
    route_color: Any,
    *,
    small: bool = False,
) -> None:
    if segment_idx % 2:
        rad = ((route_idx % 11) - 5) * -0.045
    else:
        rad = ((route_idx % 11) - 5) * 0.045
    patch = FancyArrowPatch(
        (float(start_xy[0]), float(start_xy[1])),
        (float(end_xy[0]), float(end_xy[1])),
        arrowstyle="-",
        connectionstyle=f"arc3,rad={rad:.3f}",
        color=route_color,
        linewidth=1.45 if small else 2.15,
        alpha=0.92 if small else 0.88,
        zorder=1,
    )
    axis.add_patch(patch)


def _plot_instance(
    instance: dict[str, Any],
    summary: dict[str, Any],
    output_path: Path,
    dpi: int,
) -> dict[str, Any]:
    terminal_xy = _terminal_coordinates(instance)
    num_customers = int(np.asarray(instance["customers"]).shape[0])
    num_cs = int(np.asarray(instance["charging_stations"]).shape[0])
    routes = summary["routes"]

    depot_xy = terminal_xy[0]
    customer_xy = terminal_xy[1 : 1 + num_customers]
    cs_xy = terminal_xy[1 + num_customers :]

    fig, ax = plt.subplots(figsize=(11.5, 8.0), dpi=dpi)
    route_colors = _route_colors(len(routes))
    route_handles: list[Line2D] = []
    route_customer_offsets: dict[int, list[int]] = {}

    for route_idx, route in enumerate(routes):
        route_nodes = [node for node in route if 0 <= int(node) < len(terminal_xy)]
        if len(route_nodes) < 2:
            continue
        xy = terminal_xy[route_nodes]
        route_color = route_colors[route_idx]
        route_customer_offsets[route_idx] = sorted(
            {int(node) - 1 for node in route_nodes if 1 <= int(node) <= num_customers}
        )
        for segment_idx, (start_node, end_node) in enumerate(zip(route_nodes[:-1], route_nodes[1:])):
            if int(start_node) == int(end_node):
                continue
            start_xy = terminal_xy[int(start_node)]
            end_xy = terminal_xy[int(end_node)]
            _add_route_connector(ax, start_xy, end_xy, route_idx, segment_idx, route_color)
        route_handles.append(Line2D([0], [0], color=route_color, lw=2.5, label=f"R{route_idx + 1}"))

    assigned_customers: set[int] = set()
    for route_idx, offsets in route_customer_offsets.items():
        if not offsets:
            continue
        assigned_customers.update(offsets)
        xy = customer_xy[offsets]
        route_color = route_colors[route_idx]
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=42,
            c=[route_color],
            alpha=0.96,
            edgecolor="white",
            linewidth=0.6,
            label=f"Active customers (n={num_customers}, colored by route)" if route_idx == 0 else None,
            zorder=5,
        )
        label_xy = xy[0]
        ax.text(
            label_xy[0],
            label_xy[1],
            f" R{route_idx + 1}",
            fontsize=7,
            color=route_color,
            weight="bold",
            va="center",
            zorder=8,
        )

    unassigned = sorted(set(range(num_customers)).difference(assigned_customers))
    if unassigned:
        xy = customer_xy[unassigned]
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=34,
            c="#999999",
            alpha=0.82,
            edgecolor="white",
            linewidth=0.45,
            label="Active customers without parsed route",
            zorder=4,
        )

    used_cs_terminal_ids = {
        int(node)
        for route in routes
        for node in route
        if 1 + num_customers <= int(node) < 1 + num_customers + num_cs
    }
    used_cs_offsets = [node - 1 - num_customers for node in used_cs_terminal_ids]

    ax.scatter(
        cs_xy[:, 0],
        cs_xy[:, 1],
        s=86,
        marker="^",
        c="#2ca02c",
        alpha=0.95,
        edgecolor="black",
        linewidth=0.6,
        label=f"Active charging stations (n={num_cs})",
        zorder=5,
    )
    if used_cs_offsets:
        used_xy = cs_xy[used_cs_offsets]
        ax.scatter(
            used_xy[:, 0],
            used_xy[:, 1],
            s=160,
            marker="^",
            facecolors="none",
            edgecolors="#ffbf00",
            linewidth=2.0,
            label=f"Charging stations used in routes (n={len(used_cs_offsets)})",
            zorder=6,
        )

    ax.scatter(
        [depot_xy[0]],
        [depot_xy[1]],
        s=210,
        marker="*",
        c="#d62728",
        edgecolor="black",
        linewidth=0.8,
        label="Depot",
        zorder=7,
    )
    ax.text(depot_xy[0], depot_xy[1], " depot", fontsize=9, weight="bold", va="center")

    short_route_indices = [
        route_idx
        for route_idx, offsets in route_customer_offsets.items()
        if 0 < len(offsets) <= 2
    ]
    if short_route_indices:
        axins = ax.inset_axes([0.56, 0.54, 0.40, 0.36])
        short_customer_offsets = sorted(
            {offset for route_idx in short_route_indices for offset in route_customer_offsets[route_idx]}
        )
        short_xy = np.vstack([depot_xy.reshape(1, 2), customer_xy[short_customer_offsets]])
        for route_idx in short_route_indices:
            route = [node for node in routes[route_idx] if 0 <= int(node) < len(terminal_xy)]
            route_color = route_colors[route_idx]
            for segment_idx, (start_node, end_node) in enumerate(zip(route[:-1], route[1:])):
                if int(start_node) == int(end_node):
                    continue
                _add_route_connector(
                    axins,
                    terminal_xy[int(start_node)],
                    terminal_xy[int(end_node)],
                    route_idx,
                    segment_idx,
                    route_color,
                    small=True,
                )
            offsets = route_customer_offsets[route_idx]
            if offsets:
                xy = customer_xy[offsets]
                axins.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    s=38,
                    c=[route_color],
                    alpha=0.96,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=4,
                )
                axins.annotate(
                    f"R{route_idx + 1}",
                    xy=(float(xy[0, 0]), float(xy[0, 1])),
                    xytext=(4, 4 + (route_idx % 4) * 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=route_color,
                    weight="bold",
                    zorder=6,
                )
        axins.scatter(
            [depot_xy[0]],
            [depot_xy[1]],
            s=120,
            marker="*",
            c="#d62728",
            edgecolor="black",
            linewidth=0.6,
            zorder=5,
        )
        pad = max(float(np.ptp(short_xy[:, 0])), float(np.ptp(short_xy[:, 1])), 0.35) * 0.35
        axins.set_xlim(float(short_xy[:, 0].min() - pad), float(short_xy[:, 0].max() + pad))
        axins.set_ylim(float(short_xy[:, 1].min() - pad), float(short_xy[:, 1].max() + pad))
        axins.set_aspect("equal", adjustable="box")
        axins.grid(True, color="#e5e5e5", linewidth=0.55)
        axins.set_title("Depot / short-route zoom", fontsize=8)
        axins.tick_params(labelsize=6)

    for cs_idx, xy in enumerate(cs_xy, start=1 + num_customers):
        label = f"CS{cs_idx}"
        ax.text(xy[0], xy[1], f" {label}", fontsize=7, color="#1b5e20", va="center")

    all_x = terminal_xy[:, 0]
    all_y = terminal_xy[:, 1]
    pad_x = max((all_x.max() - all_x.min()) * 0.08, 0.3)
    pad_y = max((all_y.max() - all_y.min()) * 0.08, 0.3)
    ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e0e0e0", linewidth=0.7)
    ax.set_xlabel("Projected x (km)")
    ax.set_ylabel("Projected y (km)")

    objective = summary.get("objective_distance_km")
    objective_text = "NA" if pd.isna(objective) else f"{float(objective):.2f} km"
    ax.set_title(
        f"{instance['instance_id']} | {instance['region_id']} | "
        f"routes={summary['route_count']} | objective={objective_text}\n"
        "Curved connectors separate overlapping routes; feasibility distances come from the instance distance matrix.",
        fontsize=11,
    )
    marker_legend = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=True)
    ax.add_artist(marker_legend)
    if route_handles:
        route_columns = 1 if len(route_handles) <= 8 else 2
        ax.legend(
            handles=route_handles,
            loc="lower left",
            bbox_to_anchor=(1.01, 0.0),
            fontsize=7,
            frameon=True,
            ncol=route_columns,
            title="Routes",
            title_fontsize=8,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "instance_id": instance["instance_id"],
        "region_id": instance["region_id"],
        "route_count": summary["route_count"],
        "objective_distance_km": objective,
        "vehicle_count": summary.get("vehicle_count"),
        "status_name": summary.get("status_name"),
        "used_charging_station_count": len(used_cs_offsets),
        "png": output_path.name,
    }


def _write_contact_sheet(image_paths: list[Path], output_path: Path, columns: int = 3) -> None:
    if not image_paths:
        return
    rows = int(math.ceil(len(image_paths) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 5.2, rows * 4.4), dpi=140)
    axes_arr = np.asarray(axes).reshape(rows, columns)
    for ax in axes_arr.ravel():
        ax.axis("off")
    for ax, image_path in zip(axes_arr.ravel(), image_paths):
        ax.imshow(mpimg.imread(image_path))
        ax.set_title(image_path.stem, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_html(output_dir: Path, rows: list[dict[str, Any]], contact_sheet: Path) -> None:
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Cus50 train route >= 5 maps</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;}"
        "img{max-width:100%;border:1px solid #ddd;margin:12px 0;} table{border-collapse:collapse;}"
        "td,th{border-bottom:1px solid #ddd;padding:6px 10px;text-align:left;}</style>",
        "</head><body>",
        "<h1>Cus50/train instances with route count >= 5</h1>",
        f"<p>Total: {len(rows)}. Markers: red star depot, route-colored customers, green active charging stations.</p>",
        f"<h2>Contact sheet</h2><img src='{html.escape(contact_sheet.name)}'>",
        "<h2>Instances</h2><table><tr><th>instance</th><th>region</th><th>routes</th><th>used CS</th><th>objective km</th><th>map</th></tr>",
    ]
    for row in rows:
        obj = row.get("objective_distance_km")
        obj_text = "" if pd.isna(obj) else f"{float(obj):.3f}"
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(row['instance_id']))}</td>"
            f"<td>{html.escape(str(row['region_id']))}</td>"
            f"<td>{int(row['route_count'])}</td>"
            f"<td>{int(row['used_charging_station_count'])}</td>"
            f"<td>{html.escape(obj_text)}</td>"
            f"<td><a href='{html.escape(row['png'])}'>{html.escape(row['png'])}</a></td>"
            "</tr>"
        )
    lines.append("</table>")
    for row in rows:
        lines.append(f"<h2>{html.escape(str(row['instance_id']))}</h2><img src='{html.escape(row['png'])}'>")
    lines.append("</body></html>")
    (output_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances-pkl", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-routes", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    selected_rows = _selected_summary_rows(args.summary_csv, args.min_routes)
    selected_ids = {row["instance_id"] for row in selected_rows}
    instances = _read_selected_instances(args.instances_pkl, selected_ids)

    plot_rows: list[dict[str, Any]] = []
    image_paths: list[Path] = []
    for row in selected_rows:
        instance_id = row["instance_id"]
        image_path = args.output_dir / f"{instance_id}_routes_ge{args.min_routes}.png"
        plot_row = _plot_instance(instances[instance_id], row, image_path, args.dpi)
        plot_rows.append(plot_row)
        image_paths.append(image_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_csv = args.output_dir / f"route_ge{args.min_routes}_instances.csv"
    with index_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance_id",
                "region_id",
                "route_count",
                "objective_distance_km",
                "vehicle_count",
                "status_name",
                "used_charging_station_count",
                "png",
            ],
        )
        writer.writeheader()
        writer.writerows(plot_rows)

    contact_sheet = args.output_dir / f"contact_sheet_route_ge{args.min_routes}.png"
    _write_contact_sheet(image_paths, contact_sheet)
    _write_html(args.output_dir, plot_rows, contact_sheet)

    print(json.dumps({"output_dir": str(args.output_dir), "count": len(plot_rows)}, indent=2))


if __name__ == "__main__":
    main()
