from __future__ import annotations

import base64
import colorsys
import html
from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from shapely.geometry import GeometryCollection, LineString, MultiLineString


def component_color(rank: int) -> str:
    """Return a deterministic, high-contrast color for a 1-based component rank."""
    if rank == 1:
        return "#006d8f"
    hue = ((rank - 2) * 0.618033988749895 + 0.04) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.66, 0.76)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def _line_segments(geometries: Iterable) -> list[list[tuple[float, float]]]:
    segments: list[list[tuple[float, float]]] = []
    for geometry in geometries:
        if geometry is None or geometry.is_empty:
            continue
        if isinstance(geometry, LineString):
            segments.append(list(geometry.coords))
        elif isinstance(geometry, (MultiLineString, GeometryCollection)):
            segments.extend(_line_segments(geometry.geoms))
    return segments


def _focus_on_road_extent(axis, edges, padding_fraction: float = 0.025) -> None:
    """Keep remote roadless municipal islands from shrinking the useful road map."""
    min_x, min_y, max_x, max_y = edges.total_bounds
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    axis.set_xlim(min_x - width * padding_fraction, max_x + width * padding_fraction)
    axis.set_ylim(min_y - height * padding_fraction, max_y + height * padding_fraction)


def _boolean_mask(series):
    if series.dtype == bool:
        return series
    return series.map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})


def render_connectivity_map(
    graph,
    boundary: gpd.GeoDataFrame,
    components: pd.DataFrame,
    output_dir: Path,
    city_label: str,
    legend_limit: int = 12,
) -> dict[str, str]:
    """Render the full graph by weak component and write a complete HTML legend."""
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes, edges = ox.convert.graph_to_gdfs(graph)
    local_crs = boundary.estimate_utm_crs()
    boundary_local = boundary.to_crs(local_crs)
    nodes = nodes.to_crs(local_crs)
    edges = edges.to_crs(local_crs)

    figure, axis = plt.subplots(figsize=(12.8, 8.0), dpi=160)
    figure.patch.set_facecolor("#f7f8f9")
    axis.set_facecolor("#f7f8f9")
    boundary_local.boundary.plot(ax=axis, color="#8b969e", linewidth=0.7, zorder=1)

    for row in components.itertuples(index=False):
        component_edges = edges[edges["weak_component_id"] == row.component_id]
        segments = _line_segments(component_edges.geometry)
        if segments:
            width = 0.42 if row.rank == 1 else 0.7
            alpha = 0.90 if row.rank == 1 else 0.98
            collection = LineCollection(
                segments,
                colors=[component_color(row.rank)],
                linewidths=width,
                alpha=alpha,
                zorder=2 if row.rank == 1 else 3,
            )
            axis.add_collection(collection)
        if row.rank > 1:
            component_nodes = nodes[nodes["weak_component_id"] == row.component_id]
            axis.scatter(
                component_nodes.geometry.x,
                component_nodes.geometry.y,
                s=13,
                c=component_color(row.rank),
                edgecolors="#f7f8f9",
                linewidths=0.35,
                zorder=5,
            )

    axis.autoscale()
    _focus_on_road_extent(axis, edges)
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.set_title(
        f"{city_label} — weakly connected drive components",
        fontsize=13,
        fontweight="normal",
        loc="left",
        pad=10,
    )

    shown = components.head(legend_limit)
    handles = [
        Line2D(
            [0],
            [0],
            color=component_color(int(row.rank)),
            lw=2.4,
            label=f"{row.component_id}: {int(row.node_count):,} nodes",
        )
        for row in shown.itertuples(index=False)
    ]
    if len(components) > legend_limit:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#737f87",
                lw=2.4,
                label=f"{len(components) - legend_limit} more: see HTML/CSV legend",
            )
        )
    axis.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
        ncol=1,
    )
    figure.subplots_adjust(left=0.01, right=0.77, bottom=0.01, top=0.94)

    png_path = output_dir / "connectivity_components.png"
    svg_path = output_dir / "connectivity_components.svg"
    figure.savefig(png_path, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(svg_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)

    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    legend_items = []
    for row in components.itertuples(index=False):
        legend_items.append(
            "<li>"
            f'<span class="swatch" style="--component-color:{component_color(int(row.rank))}"></span>'
            f"<code>{html.escape(str(row.component_id))}</code>"
            f"<span>{int(row.node_count):,} nodes · {int(row.directed_edge_count):,} directed edges</span>"
            "</li>"
        )
    html_path = output_dir / "connectivity_components.html"
    html_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(city_label)} connectivity components</title>
<style>
:root {{ color-scheme: light dark; --bg:#f7f8f9; --fg:#17232c; --muted:#5e6d77; --rule:#d7dee3; }}
@media (prefers-color-scheme:dark) {{ :root {{ --bg:#11171c; --fg:#edf3f6; --muted:#9caab3; --rule:#34414a; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:18px; background:var(--bg); color:var(--fg); font-family:system-ui,sans-serif; }}
main {{ max-width:1180px; margin:auto; }}
h1 {{ margin:0 0 12px; font-size:clamp(20px,3vw,34px); font-weight:500; }}
img {{ display:block; width:100%; height:auto; }}
details {{ margin-top:14px; border-top:1px solid var(--rule); padding-top:10px; }}
summary {{ cursor:pointer; font-weight:500; }}
ul {{ list-style:none; padding:0; display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:7px 18px; }}
li {{ display:grid; grid-template-columns:14px 55px 1fr; align-items:center; gap:7px; color:var(--muted); font-size:13px; }}
.swatch {{ width:14px; height:4px; background:var(--component-color); }}
code {{ color:var(--fg); }}
</style>
</head>
<body><main>
<h1>{html.escape(city_label)} — weakly connected drive components</h1>
<img src="data:image/png;base64,{encoded}" alt="Road graph colored by weakly connected component">
<details open><summary>Complete component legend ({len(components)})</summary><ul>{"".join(legend_items)}</ul></details>
</main></body>
</html>
""",
        encoding="utf-8",
    )
    return {
        "png": png_path.name,
        "svg": svg_path.name,
        "html": html_path.name,
        "legend_semantics": (
            f"PNG labels the {min(legend_limit, len(components))} largest weak components; "
            "HTML and components.csv enumerate every component"
        ),
    }


def render_operational_map(
    graph,
    boundary: gpd.GeoDataFrame,
    output_dir: Path,
    city_label: str,
    operational_summary: dict,
) -> dict[str, str]:
    """Render the connected routing graph with outside-city transit roads separated."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _, edges = ox.convert.graph_to_gdfs(graph)
    local_crs = boundary.estimate_utm_crs()
    boundary_local = boundary.to_crs(local_crs)
    edges = edges.to_crs(local_crs)
    transit_mask = _boolean_mask(edges["transit_only"])
    transit_edges = edges[transit_mask]
    city_edges = edges[~transit_mask]

    figure, axis = plt.subplots(figsize=(12.8, 8.0), dpi=160)
    figure.patch.set_facecolor("#f7f8f9")
    axis.set_facecolor("#f7f8f9")
    boundary_local.boundary.plot(ax=axis, color="#7c8991", linewidth=0.8, zorder=3)

    transit_segments = _line_segments(transit_edges.geometry)
    if transit_segments:
        axis.add_collection(
            LineCollection(
                transit_segments,
                colors=["#d07a3c"],
                linewidths=0.45,
                alpha=0.58,
                zorder=1,
            )
        )
    city_segments = _line_segments(city_edges.geometry)
    if city_segments:
        axis.add_collection(
            LineCollection(
                city_segments,
                colors=["#006d8f"],
                linewidths=0.48,
                alpha=0.90,
                zorder=2,
            )
        )

    axis.autoscale()
    _focus_on_road_extent(axis, edges)
    axis.set_aspect("equal")
    axis.set_axis_off()
    buffer_km = float(operational_summary["selected_buffer_km"])
    node_coverage = float(operational_summary["city_node_coverage"]) * 100.0
    length_coverage = float(operational_summary["city_physical_road_length_coverage"]) * 100.0
    axis.set_title(
        f"{city_label} — operational OSM routing graph",
        fontsize=13,
        fontweight="normal",
        loc="left",
        pad=10,
    )
    handles = [
        Line2D([0], [0], color="#006d8f", lw=2.4, label="Inside-city service roads"),
        Line2D([0], [0], color="#d07a3c", lw=2.4, label="Outside-city transit-only roads"),
        Line2D([0], [0], color="#7c8991", lw=1.2, label="Exact city boundary"),
        Line2D([0], [0], color="none", label=f"Selected buffer: {buffer_km:g} km"),
        Line2D([0], [0], color="none", label=f"City nodes covered: {node_coverage:.3f}%"),
        Line2D([0], [0], color="none", label=f"Road length covered: {length_coverage:.3f}%"),
    ]
    axis.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
    )
    figure.subplots_adjust(left=0.01, right=0.77, bottom=0.01, top=0.94)

    png_path = output_dir / "operational_connectivity.png"
    svg_path = output_dir / "operational_connectivity.svg"
    figure.savefig(png_path, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(svg_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)

    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    html_path = output_dir / "operational_connectivity.html"
    html_path.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(city_label)} operational routing graph</title>
<style>body{{margin:0;padding:18px;background:#f7f8f9;color:#17232c;font-family:system-ui,sans-serif}}main{{max-width:1180px;margin:auto}}h1{{font-size:clamp(20px,3vw,34px);font-weight:500;margin:0 0 8px}}p{{color:#53636d;margin:0 0 12px}}img{{display:block;width:100%;height:auto}}</style>
</head><body><main>
<h1>{html.escape(city_label)} — operational OSM routing graph</h1>
<p>One weak component · {buffer_km:g} km routing buffer · {node_coverage:.3f}% city-node coverage · {length_coverage:.3f}% city-road-length coverage. Orange roads are outside-city transit-only connectors; no synthetic edges are used.</p>
<img src="data:image/png;base64,{encoded}" alt="Operational routing graph with city and transit-only roads">
</main></body></html>""",
        encoding="utf-8",
    )
    return {
        "png": png_path.name,
        "svg": svg_path.name,
        "html": html_path.name,
        "legend_semantics": "teal=inside-city service roads; orange=outside-city transit-only OSM roads",
    }
