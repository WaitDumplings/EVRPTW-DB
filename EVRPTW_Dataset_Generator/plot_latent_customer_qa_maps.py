from __future__ import annotations

import argparse
import html
import os
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "EVRPTW_Dataset" / "Geo_AC_v1" / "source_data_na_us20"
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot full latent-customer QA maps for Geo-AC-v1 service territories."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--territory-id", action="append", default=None)
    parser.add_argument("--dpi", type=int, default=190)
    parser.add_argument("--max-road-edges", type=int, default=200_000)
    parser.add_argument("--latent-point-size", type=float, default=0.38)
    parser.add_argument("--latent-alpha", type=float, default=0.15)
    return parser.parse_args()


def _territory_dirs(source_root: Path, requested: list[str] | None) -> list[Path]:
    dirs = sorted(path for path in source_root.iterdir() if (path / "normalized").exists())
    if requested:
        keep = set(str(item) for item in requested)
        dirs = [path for path in dirs if path.name in keep]
    if not dirs:
        raise ValueError("No territory normalized directories found.")
    return dirs


def _boundary(qa_dir: Path, gpd: Any) -> Any:
    preview = qa_dir / "preview_layers.geojson"
    if not preview.exists():
        return None
    layers = gpd.read_file(preview).to_crs("EPSG:4326")
    if "layer" not in layers.columns:
        return layers[layers.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].head(1)
    return layers[layers["layer"] == "service_territory"]


def _road_segments(nodes: pd.DataFrame, edges: pd.DataFrame, max_edges: int) -> list[list[tuple[float, float]]]:
    if len(edges) > int(max_edges):
        edges = edges.sample(n=int(max_edges), random_state=20260527).sort_index()
    node_xy = nodes[["node_id", "lon", "lat"]].copy()
    node_xy["node_id"] = node_xy["node_id"].astype(str)
    node_xy = node_xy.set_index("node_id")
    merged = (
        edges.astype({"u": str, "v": str})
        .merge(node_xy.rename(columns={"lon": "u_lon", "lat": "u_lat"}), left_on="u", right_index=True, how="inner")
        .merge(node_xy.rename(columns={"lon": "v_lon", "lat": "v_lat"}), left_on="v", right_index=True, how="inner")
    )
    return [
        [(float(row.u_lon), float(row.u_lat)), (float(row.v_lon), float(row.v_lat))]
        for row in merged.itertuples(index=False)
    ]


def _plot_one(
    territory_dir: Path,
    output_dir: Path,
    dpi: int,
    max_road_edges: int,
    latent_point_size: float,
    latent_alpha: float,
) -> dict[str, Any]:
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    territory_id = territory_dir.name
    normalized = territory_dir / "normalized"
    qa_dir = territory_dir / "qa"
    nodes = pd.read_csv(normalized / "road_nodes.csv", dtype={"node_id": str})
    edges = pd.read_csv(normalized / "road_edges.csv", dtype={"u": str, "v": str})
    latent = pd.read_csv(normalized / "latent_customer.csv", usecols=["lon", "lat"])
    chargers = pd.read_csv(normalized / "charging_station.csv")
    depots = pd.read_csv(normalized / "depot_candidate.csv")
    boundary = _boundary(qa_dir, gpd)
    segments = _road_segments(nodes, edges, int(max_road_edges))

    fig, ax = plt.subplots(figsize=(12, 10))
    if boundary is not None and not boundary.empty:
        boundary.plot(ax=ax, color="#eef6ff", edgecolor="#1f4e79", linewidth=1.4, alpha=0.72, zorder=0)
    ax.add_collection(
        LineCollection(
            segments,
            colors="#6b7280",
            linewidths=0.20,
            alpha=0.24,
            label=f"Road edges (n={len(edges):,})" + (" sampled" if len(edges) > max_road_edges else ""),
            zorder=1,
        )
    )
    ax.scatter(
        latent["lon"],
        latent["lat"],
        s=float(latent_point_size),
        color="#db2777",
        alpha=float(latent_alpha),
        linewidths=0,
        rasterized=True,
        label=f"Latent customers (n={len(latent):,})",
        zorder=2,
    )
    ax.scatter(
        chargers["lon"],
        chargers["lat"],
        s=18,
        color="#16a34a",
        edgecolors="white",
        linewidths=0.35,
        alpha=0.90,
        label=f"Public charging stations (n={len(chargers):,})",
        zorder=4,
    )
    fallback = depots["source"].astype(str).str.contains("fallback", case=False, na=False)
    open_dep = depots[~fallback]
    fb_dep = depots[fallback]
    if len(open_dep):
        ax.scatter(
            open_dep["lon"],
            open_dep["lat"],
            marker="^",
            s=120,
            color="#dc2626",
            edgecolors="white",
            linewidths=0.8,
            label=f"Open depot candidates (n={len(open_dep)})",
            zorder=6,
        )
    if len(fb_dep):
        ax.scatter(
            fb_dep["lon"],
            fb_dep["lat"],
            marker="^",
            s=120,
            color="#f59e0b",
            edgecolors="#111827",
            linewidths=0.7,
            label=f"Fallback depot candidates (n={len(fb_dep)})",
            zorder=6,
        )

    if boundary is not None and not boundary.empty:
        minx, miny, maxx, maxy = boundary.total_bounds
    else:
        minx = min(nodes["lon"].min(), latent["lon"].min())
        miny = min(nodes["lat"].min(), latent["lat"].min())
        maxx = max(nodes["lon"].max(), latent["lon"].max())
        maxy = max(nodes["lat"].max(), latent["lat"].max())
    padx = max((maxx - minx) * 0.035, 1e-4)
    pady = max((maxy - miny) * 0.035, 1e-4)
    ax.set_xlim(minx - padx, maxx + padx)
    ax.set_ylim(miny - pady, maxy + pady)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{territory_id}: road-frontage latent customers, chargers, depots, road network")
    ax.legend(loc="upper right", frameon=True, fontsize=8, markerscale=1.6)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{territory_id}_latent_customers_chargers_depots_roads.png"
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    return {
        "territory_id": territory_id,
        "path": path,
        "latent": int(len(latent)),
        "chargers": int(len(chargers)),
        "depots": int(len(depots)),
        "road_edges": int(len(edges)),
    }


def _write_gallery(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    def card(row: dict[str, Any]) -> str:
        path = Path(row["path"])
        caption = (
            f"latent={row['latent']:,} | chargers={row['chargers']:,} | "
            f"depots={row['depots']:,} | road_edges={row['road_edges']:,}"
        )
        return (
            '<figure class="card">'
            f'<a href="{html.escape(path.name)}"><img src="{html.escape(path.name)}" alt="{html.escape(row["territory_id"])}"></a>'
            f'<figcaption><strong>{html.escape(row["territory_id"])}</strong><br>{html.escape(caption)}</figcaption>'
            "</figure>"
        )

    body = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>Geo-AC-v1 Latent Customer QA Maps</title>",
            "<style>",
            "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;color:#1f2933}",
            ".legend{display:flex;gap:18px;align-items:center;margin:12px 0 24px;flex-wrap:wrap}",
            ".swatch{display:inline-block;width:14px;height:14px;clip-path:polygon(50% 0,0 100%,100% 100%);margin-right:6px;vertical-align:-2px}",
            ".open{background:#dc2626}.fallback{background:#f59e0b;border:1px solid #111}",
            ".dot{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:6px;vertical-align:-1px}.latent{background:#db2777}.charger{background:#16a34a}",
            ".line{display:inline-block;width:28px;height:0;border-top:2px solid #6b7280;margin-right:6px;vertical-align:4px}",
            ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}",
            ".card{border:1px solid #d8dee9;border-radius:8px;padding:10px;margin:0;background:#fff}",
            ".card img{width:100%;height:auto;display:block;border:1px solid #edf2f7}",
            "figcaption{font-size:13px;line-height:1.4;margin-top:8px}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Geo-AC-v1 Latent Customer QA Maps</h1>",
            '<div class="legend"><span><span class="line"></span>Road edges</span>'
            '<span><span class="dot latent"></span>Latent customers</span>'
            '<span><span class="dot charger"></span>Public charging stations</span>'
            '<span><span class="swatch open"></span>Open depot candidate</span>'
            '<span><span class="swatch fallback"></span>Fallback depot candidate</span></div>',
            '<div class="grid">',
            "\n".join(card(row) for row in rows),
            "</div>",
            "</body>",
            "</html>",
        ]
    )
    path = output_dir / "index.html"
    path.write_text(body, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_dir = (args.output_dir or (source_root / "qa_maps_latent")).resolve()
    rows = []
    for territory_dir in _territory_dirs(source_root, args.territory_id):
        row = _plot_one(
            territory_dir,
            output_dir,
            int(args.dpi),
            int(args.max_road_edges),
            float(args.latent_point_size),
            float(args.latent_alpha),
        )
        rows.append(row)
        print({k: (str(v) if isinstance(v, Path) else v) for k, v in row.items()})
    gallery = _write_gallery(output_dir, rows)
    print({"gallery": str(gallery), "map_count": len(rows)})


if __name__ == "__main__":
    main()
