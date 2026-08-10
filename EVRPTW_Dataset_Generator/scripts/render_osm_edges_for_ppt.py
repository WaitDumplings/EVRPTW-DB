#!/usr/bin/env python3
"""Render a slide-ready, edge-only OSM road graph image."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd
from matplotlib.lines import Line2D


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--edge-color", default="#124E78")
    parser.add_argument("--background", default="#F7F9FC")
    parser.add_argument("--edge-linewidth", type=float, default=0.22)
    parser.add_argument(
        "--transit-only-color",
        help="Color outside-city transit-only edges separately when present.",
    )
    parser.add_argument("--boundary", type=Path)
    parser.add_argument("--boundary-color", default="#2A9D8F")
    parser.add_argument("--boundary-linewidth", type=float, default=1.1)
    parser.add_argument(
        "--chargers",
        type=Path,
        help="Optional cle chargers.parquet layer to overlay.",
    )
    parser.add_argument("--charger-l2-color", default="#7B2CBF")
    parser.add_argument("--charger-dc-color", default="#D62828")
    parser.add_argument("--charger-l2-size", type=float, default=9.0)
    parser.add_argument("--charger-dc-size", type=float, default=18.0)
    parser.add_argument(
        "--customers",
        type=Path,
        help="Optional latent_locations.parquet layer to overlay as service-site points.",
    )
    parser.add_argument(
        "--depots",
        type=Path,
        help="Optional depot candidate CSV, GeoJSON, or Parquet layer to overlay.",
    )
    parser.add_argument("--depot-strict-color", default="#111827")
    parser.add_argument("--depot-optional-color", default="#F2B705")
    parser.add_argument("--depot-strict-size", type=float, default=72.0)
    parser.add_argument("--depot-optional-size", type=float, default=30.0)
    parser.add_argument(
        "--show-legend",
        action="store_true",
        help="Draw a compact legend for the visible road, boundary, and charger layers.",
    )
    args = parser.parse_args()

    graph = ox.load_graphml(args.graph)
    if args.transit_only_color:
        plot_graph = graph
        transit_flags = [
            _as_bool(attributes.get("transit_only", False))
            for _, _, _, attributes in graph.edges(keys=True, data=True)
        ]
        edge_colors = [
            args.transit_only_color if transit else args.edge_color
            for transit in transit_flags
        ]
        edge_widths = [
            args.edge_linewidth * (2.4 if transit else 1.0)
            for transit in transit_flags
        ]
    else:
        plot_graph = ox.convert.to_undirected(graph)
        transit_flags = []
        edge_colors = args.edge_color
        edge_widths = args.edge_linewidth
    figure, axis = ox.plot_graph(
        plot_graph,
        figsize=(12.8, 7.2),
        bgcolor=args.background,
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        edge_alpha=0.72,
        show=False,
        close=False,
    )
    if args.boundary:
        boundary = gpd.read_file(args.boundary).to_crs(plot_graph.graph["crs"])
        boundary.plot(
            ax=axis,
            facecolor="none",
            edgecolor=args.boundary_color,
            linewidth=args.boundary_linewidth,
            zorder=5,
        )
    customer_counts: dict[str, int] = {}
    customer_order = [
        "house",
        "manufactured_home",
        "small_apt",
        "medium_apt",
        "large_apt",
    ]
    customer_styles = {
        "house": {"color": "#8A96A3", "size": 0.18, "alpha": 0.12, "marker": "o"},
        "manufactured_home": {
            "color": "#43AA8B",
            "size": 1.2,
            "alpha": 0.52,
            "marker": "s",
        },
        "small_apt": {"color": "#00B4D8", "size": 0.5, "alpha": 0.30, "marker": "o"},
        "medium_apt": {"color": "#E76FBD", "size": 1.2, "alpha": 0.46, "marker": "o"},
        "large_apt": {"color": "#FF006E", "size": 3.0, "alpha": 0.70, "marker": "o"},
    }
    if args.customers:
        customer_table = pd.read_parquet(
            args.customers,
            columns=[
                "location_lon",
                "location_lat",
                "service_location_type",
                "cle_candidate_eligible",
            ],
        )
        customer_table = customer_table.loc[
            customer_table["cle_candidate_eligible"].map(_as_bool)
        ].copy()
        customers = gpd.GeoDataFrame(
            customer_table,
            geometry=gpd.points_from_xy(
                customer_table["location_lon"], customer_table["location_lat"]
            ),
            crs="EPSG:4326",
        ).to_crs(plot_graph.graph["crs"])
        for location_type in customer_order:
            subset = customers.loc[
                customers["service_location_type"] == location_type
            ]
            customer_counts[location_type] = len(subset)
            if subset.empty:
                continue
            style = customer_styles[location_type]
            axis.scatter(
                subset.geometry.x,
                subset.geometry.y,
                s=style["size"],
                c=style["color"],
                marker=style["marker"],
                alpha=style["alpha"],
                linewidths=0,
                rasterized=True,
                zorder=0.9,
            )
        customer_counts["total"] = len(customers)
    charger_counts: dict[str, int] = {}
    if args.chargers:
        chargers = gpd.read_parquet(args.chargers).to_crs(plot_graph.graph["crs"])
        if "charger_candidate_eligible" in chargers.columns:
            chargers = chargers.loc[
                chargers["charger_candidate_eligible"].map(_as_bool)
            ].copy()
        l2 = chargers.loc[chargers["reference_charge_mode"] == "ac_level2_j1772"]
        dc = chargers.loc[chargers["reference_charge_mode"] == "dc_fast_ccs1"]
        charger_counts = {"l2": len(l2), "dc": len(dc), "total": len(chargers)}
        if not l2.empty:
            l2.plot(
                ax=axis,
                color=args.charger_l2_color,
                marker="o",
                markersize=args.charger_l2_size,
                edgecolor=args.background,
                linewidth=0.25,
                alpha=0.88,
                zorder=8,
            )
        if not dc.empty:
            dc.plot(
                ax=axis,
                color=args.charger_dc_color,
                marker="D",
                markersize=args.charger_dc_size,
                edgecolor=args.background,
                linewidth=0.4,
                alpha=0.96,
                zorder=9,
            )
    depot_counts: dict[str, int] = {}
    if args.depots:
        if args.depots.suffix.lower() == ".parquet":
            depots = gpd.read_parquet(args.depots)
        elif args.depots.suffix.lower() in {".geojson", ".json"}:
            depots = gpd.read_file(args.depots)
        else:
            depot_table = pd.read_csv(args.depots)
            depots = gpd.GeoDataFrame(
                depot_table,
                geometry=gpd.points_from_xy(
                    depot_table["longitude"], depot_table["latitude"]
                ),
                crs="EPSG:4326",
            )
        depots = depots.to_crs(plot_graph.graph["crs"])
        eligibility_column = next(
            (
                column
                for column in (
                    "cle_candidate_eligible",
                    "depot_candidate_eligible",
                )
                if column in depots.columns
            ),
            None,
        )
        if eligibility_column:
            depots = depots.loc[depots[eligibility_column].map(_as_bool)].copy()
        strict = depots.loc[depots["evidence_tier"] == "A_osm_explicit"]
        optional = depots.loc[depots["evidence_tier"] == "B_warehouse_proxy"]
        depot_counts = {
            "strict": len(strict),
            "optional": len(optional),
            "total": len(depots),
        }
        if not optional.empty:
            optional.plot(
                ax=axis,
                color=args.depot_optional_color,
                marker="^",
                markersize=args.depot_optional_size,
                edgecolor="#6B5200",
                linewidth=0.45,
                alpha=0.94,
                zorder=10,
            )
        if not strict.empty:
            strict.plot(
                ax=axis,
                color=args.depot_strict_color,
                marker="*",
                markersize=args.depot_strict_size,
                edgecolor=args.background,
                linewidth=0.7,
                alpha=1.0,
                zorder=11,
            )
    if args.show_legend:
        handles: list[Line2D] = [
            Line2D(
                [0],
                [0],
                color=args.edge_color,
                linewidth=2.0,
                label="OSM roads inside service city",
            )
        ]
        if args.transit_only_color:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=args.transit_only_color,
                    linewidth=2.0,
                    label="OSM transit-only extend roads",
                )
            )
        if args.boundary:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=args.boundary_color,
                    linewidth=2.0,
                    label="Land-only service boundary",
                )
            )
        if args.customers:
            customer_labels = {
                "house": "House",
                "manufactured_home": "Manufactured home",
                "small_apt": "Small apartment · 2–4 units",
                "medium_apt": "Medium apartment · 5–19 units",
                "large_apt": "Large apartment · 20+ units",
            }
            for location_type in customer_order:
                style = customer_styles[location_type]
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=style["marker"],
                        linestyle="None",
                        markerfacecolor=style["color"],
                        markeredgecolor="none",
                        markersize=(
                            4.0
                            if location_type == "house"
                            else 5.0 + customer_order.index(location_type) * 0.7
                        ),
                        label=(
                            f"{customer_labels[location_type]} "
                            f"(n={customer_counts[location_type]:,})"
                        ),
                    )
                )
        if args.chargers:
            handles.extend(
                [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="None",
                        markerfacecolor=args.charger_l2_color,
                        markeredgecolor=args.background,
                        markersize=6.5,
                        label=f"AC Level 2 · J1772 (n={charger_counts['l2']:,})",
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker="D",
                        linestyle="None",
                        markerfacecolor=args.charger_dc_color,
                        markeredgecolor=args.background,
                        markersize=6.5,
                        label=f"DC fast · CCS1 (n={charger_counts['dc']:,})",
                    ),
                ]
            )
        if args.depots:
            handles.extend(
                [
                    Line2D(
                        [0],
                        [0],
                        marker="*",
                        linestyle="None",
                        markerfacecolor=args.depot_strict_color,
                        markeredgecolor=args.background,
                        markersize=10,
                        label=f"Strict depot candidate (n={depot_counts['strict']:,})",
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker="^",
                        linestyle="None",
                        markerfacecolor=args.depot_optional_color,
                        markeredgecolor="#6B5200",
                        markersize=7,
                        label=f"Optional warehouse proxy (n={depot_counts['optional']:,})",
                    ),
                ]
            )
        legend = axis.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.018, 0.982),
            frameon=True,
            facecolor=args.background,
            edgecolor="#B7C2CF",
            framealpha=0.94,
            fontsize=6.4 if len(handles) > 8 else 7.5,
            handlelength=2.4,
            labelspacing=0.65,
            borderpad=0.75,
            ncol=2 if len(handles) > 8 else 1,
            columnspacing=1.25,
        )
        legend.set_zorder(20)
    axis.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output,
        dpi=args.dpi,
        facecolor=args.background,
        edgecolor="none",
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(figure)
    print(
        f"wrote {args.output.resolve()} | "
        f"nodes={plot_graph.number_of_nodes():,} "
        f"edges={plot_graph.number_of_edges():,} "
        f"transit_only_edges={sum(transit_flags):,} "
        f"chargers={charger_counts.get('total', 0):,} "
        f"depots={depot_counts.get('total', 0):,} "
        f"customers={customer_counts.get('total', 0):,}"
    )


if __name__ == "__main__":
    main()
