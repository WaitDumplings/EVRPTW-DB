from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "EVRPTW_Dataset" / "Geo_AC_v1" / "source_data_na_us20"
DEFAULT_TERRITORY_ID = "la_ca_san_gabriel_pomona_industry"
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview latent-customer generation and road-edge snapping for one community."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--territory-id", default=DEFAULT_TERRITORY_ID)
    parser.add_argument("--community-id", default=None)
    parser.add_argument("--customer-count", type=int, default=80)
    parser.add_argument("--placement-mode", choices=["road_frontage", "near_road"], default="road_frontage")
    parser.add_argument("--road-radius-km", type=float, default=0.85)
    parser.add_argument("--weight-sigma-km", type=float, default=0.38)
    parser.add_argument("--min-spacing-km", type=float, default=0.035)
    parser.add_argument("--min-access-km", type=float, default=0.004)
    parser.add_argument("--max-access-km", type=float, default=0.020)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _normalized_dir(source_root: Path, territory_id: str) -> Path:
    path = source_root / territory_id / "normalized"
    if not path.exists():
        raise FileNotFoundError(f"Missing normalized directory: {path}")
    return path


def _load_tables(normalized_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(normalized_dir / "road_nodes.csv", dtype={"node_id": str})
    edges = pd.read_csv(normalized_dir / "road_edges.csv", dtype={"u": str, "v": str})
    seeds = pd.read_csv(normalized_dir / "customer_seed.csv", dtype={"community_id": str})
    return nodes, edges, seeds


def _edge_table(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    node_cols = nodes[["node_id", "lon", "lat", "x_km", "y_km"]].copy()
    merged = (
        edges.merge(
            node_cols.rename(
                columns={"node_id": "u", "lon": "u_lon", "lat": "u_lat", "x_km": "u_x", "y_km": "u_y"}
            ),
            on="u",
            how="inner",
        )
        .merge(
            node_cols.rename(
                columns={"node_id": "v", "lon": "v_lon", "lat": "v_lat", "x_km": "v_x", "y_km": "v_y"}
            ),
            on="v",
            how="inner",
        )
        .copy()
    )
    merged["length_km"] = pd.to_numeric(merged["length_km"], errors="coerce")
    merged = merged[np.isfinite(merged["length_km"]) & (merged["length_km"] > 0)].copy()
    return merged.reset_index(drop=True)


def _segment_projection(point_xy: np.ndarray, edge_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = edge_df[["u_x", "u_y"]].to_numpy(dtype=float)
    v = edge_df[["v_x", "v_y"]].to_numpy(dtype=float)
    uv = v - u
    denom = np.sum(uv * uv, axis=1)
    denom = np.maximum(denom, 1e-12)
    t = np.sum((point_xy[None, :] - u) * uv, axis=1) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = u + t[:, None] * uv
    dist = np.linalg.norm(point_xy[None, :] - proj, axis=1)
    return proj, dist, t


def _choose_seed(seeds: pd.DataFrame, edges: pd.DataFrame, community_id: str | None, road_radius_km: float) -> pd.Series:
    if community_id:
        matches = seeds[seeds["community_id"].astype(str) == str(community_id)]
        if matches.empty:
            raise ValueError(f"Community {community_id} not found.")
        return matches.iloc[0]

    candidates = seeds.copy()
    candidates["occupancy"] = pd.to_numeric(candidates["occupancy"], errors="coerce").fillna(0.0)
    candidates = candidates[candidates["occupancy"] > 0].nlargest(min(200, len(candidates)), "occupancy")
    best: tuple[float, int] | None = None
    for row_idx, row in candidates.iterrows():
        point = np.asarray([float(row["x_km"]), float(row["y_km"])], dtype=float)
        _, dist, _ = _segment_projection(point, edges)
        nearby = dist <= float(road_radius_km)
        local_length = float(edges.loc[nearby, "length_km"].sum())
        local_count = int(nearby.sum())
        if local_count < 20:
            continue
        score = float(np.log1p(row["occupancy"]) * np.log1p(local_length) * np.log1p(local_count))
        if best is None or score > best[0]:
            best = (score, int(row_idx))
    if best is None:
        return candidates.iloc[0]
    return seeds.loc[best[1]]


def _hard_core_accept(point: np.ndarray, accepted: list[np.ndarray], min_spacing_km: float) -> bool:
    if not accepted:
        return True
    arr = np.vstack(accepted)
    return bool(np.min(np.linalg.norm(arr - point[None, :], axis=1)) >= float(min_spacing_km))


def _generate_customers(
    seed_row: pd.Series,
    edges: pd.DataFrame,
    count: int,
    rng: np.random.Generator,
    placement_mode: str,
    road_radius_km: float,
    weight_sigma_km: float,
    min_spacing_km: float,
    min_access_km: float,
    max_access_km: float,
) -> pd.DataFrame:
    seed_xy = np.asarray([float(seed_row["x_km"]), float(seed_row["y_km"])], dtype=float)
    edge_mid = 0.5 * (
        edges[["u_x", "u_y"]].to_numpy(dtype=float) + edges[["v_x", "v_y"]].to_numpy(dtype=float)
    )
    _, seed_to_edge, _ = _segment_projection(seed_xy, edges)
    center_dist = np.linalg.norm(edge_mid - seed_xy[None, :], axis=1)
    local = edges[(seed_to_edge <= road_radius_km) | (center_dist <= road_radius_km)].copy()
    if local.empty:
        raise ValueError("No local road edges near selected community.")

    center_dist = np.linalg.norm(
        0.5 * (local[["u_x", "u_y"]].to_numpy(dtype=float) + local[["v_x", "v_y"]].to_numpy(dtype=float))
        - seed_xy[None, :],
        axis=1,
    )
    if placement_mode == "road_frontage":
        local = local[(local["length_km"] >= 0.015) & (local["length_km"] <= 0.45)].copy()
        if local.empty:
            raise ValueError("No frontage-like local road edges near selected community.")
        center_dist = np.linalg.norm(
            0.5 * (local[["u_x", "u_y"]].to_numpy(dtype=float) + local[["v_x", "v_y"]].to_numpy(dtype=float))
            - seed_xy[None, :],
            axis=1,
        )
    weights = local["length_km"].to_numpy(dtype=float) * np.exp(-0.5 * (center_dist / weight_sigma_km) ** 2)
    weights = np.maximum(weights, 1e-12)
    weights = weights / weights.sum()

    u = local[["u_x", "u_y"]].to_numpy(dtype=float)
    v = local[["v_x", "v_y"]].to_numpy(dtype=float)
    uv = v - u
    seg_len = np.linalg.norm(uv, axis=1)
    unit = uv / np.maximum(seg_len[:, None], 1e-12)
    normal = np.column_stack([-unit[:, 1], unit[:, 0]])

    accepted: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(5000, int(count) * 250)
    while len(rows) < int(count) and attempts < max_attempts:
        attempts += 1
        edge_pos = int(rng.choice(len(local), p=weights))
        t = float(rng.uniform(0.02, 0.98))
        snap_xy = u[edge_pos] + t * uv[edge_pos]
        side = -1.0 if rng.random() < 0.5 else 1.0
        access = float(rng.uniform(min_access_km, max_access_km))
        along_noise = float(rng.normal(0.0, 0.004 if placement_mode == "road_frontage" else 0.008))
        customer_xy = snap_xy + side * access * normal[edge_pos] + along_noise * unit[edge_pos]
        if np.linalg.norm(customer_xy - seed_xy) > road_radius_km * 1.12:
            continue
        if not _hard_core_accept(customer_xy, accepted, min_spacing_km):
            continue
        accepted.append(customer_xy)
        edge_row = local.iloc[edge_pos]
        edge_length = float(edge_row["length_km"])
        u_dist = t * edge_length
        v_dist = (1.0 - t) * edge_length
        snap_node_id = str(edge_row["u"] if u_dist <= v_dist else edge_row["v"])
        rows.append(
            {
                "customer_id": f"preview_customer_{len(rows):03d}",
                "community_id": str(seed_row["community_id"]),
                "x_km": float(customer_xy[0]),
                "y_km": float(customer_xy[1]),
                "snap_x_km": float(snap_xy[0]),
                "snap_y_km": float(snap_xy[1]),
                "snap_edge_u": edge_row["u"],
                "snap_edge_v": edge_row["v"],
                "snap_node_id": snap_node_id,
                "snap_distance_km": float(np.linalg.norm(customer_xy - snap_xy)),
                "connector_distance_km": float(access + min(u_dist, v_dist)),
                "source": f"preview_{placement_mode}_sampled_from_community_occupancy",
            }
        )

    if len(rows) < int(count):
        raise RuntimeError(f"Only generated {len(rows)} customers after {attempts} attempts.")
    return pd.DataFrame(rows)


def _add_lonlat(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from pyproj import Transformer
    except ImportError:
        return df
    transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(df["x_km"].to_numpy(dtype=float) * 1000.0, df["y_km"].to_numpy(dtype=float) * 1000.0)
    snap_lon, snap_lat = transformer.transform(
        df["snap_x_km"].to_numpy(dtype=float) * 1000.0,
        df["snap_y_km"].to_numpy(dtype=float) * 1000.0,
    )
    out = df.copy()
    out.insert(3, "lon", lon)
    out.insert(4, "lat", lat)
    out.insert(9, "snap_lon", snap_lon)
    out.insert(10, "snap_lat", snap_lat)
    return out


def _plot_preview(
    territory_id: str,
    seed_row: pd.Series,
    edges: pd.DataFrame,
    nearby_seeds: pd.DataFrame,
    customers: pd.DataFrame,
    output_path: Path,
    road_radius_km: float,
    placement_mode: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    seed_xy = np.asarray([float(seed_row["x_km"]), float(seed_row["y_km"])], dtype=float)
    center_dist = np.linalg.norm(
        0.5 * (edges[["u_x", "u_y"]].to_numpy(dtype=float) + edges[["v_x", "v_y"]].to_numpy(dtype=float))
        - seed_xy[None, :],
        axis=1,
    )
    local_edges = edges[center_dist <= road_radius_km * 1.35].copy()
    segments = [
        [(float(row.u_x), float(row.u_y)), (float(row.v_x), float(row.v_y))]
        for row in local_edges.itertuples(index=False)
    ]

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    ax.add_collection(LineCollection(segments, colors="#6b7280", linewidths=0.6, alpha=0.45, label="Road edges"))
    if not nearby_seeds.empty:
        ax.scatter(
            nearby_seeds["x_km"],
            nearby_seeds["y_km"],
            s=20,
            color="#93c5fd",
            edgecolors="white",
            linewidths=0.3,
            label="Nearby community seeds",
            zorder=3,
        )
    ax.scatter(
        [float(seed_row["x_km"])],
        [float(seed_row["y_km"])],
        s=180,
        marker="*",
        color="#1d4ed8",
        edgecolors="white",
        linewidths=0.8,
        label="Selected community seed",
        zorder=6,
    )
    connector_segments = [
        [(float(row.x_km), float(row.y_km)), (float(row.snap_x_km), float(row.snap_y_km))]
        for row in customers.itertuples(index=False)
    ]
    ax.add_collection(
        LineCollection(
            connector_segments,
            colors="#be185d",
            linewidths=0.45,
            alpha=0.52,
            label="Customer access connector",
            zorder=4,
        )
    )
    ax.scatter(
        customers["snap_x_km"],
        customers["snap_y_km"],
        s=10,
        color="#111827",
        alpha=0.68,
        label="Snap point on road edge",
        zorder=5,
    )
    ax.scatter(
        customers["x_km"],
        customers["y_km"],
        s=34,
        color="#db2777",
        edgecolors="white",
        linewidths=0.45,
        label=f"Preview latent customers (n={len(customers)})",
        zorder=7,
    )
    margin = road_radius_km * 1.12
    ax.set_xlim(seed_xy[0] - margin, seed_xy[0] + margin)
    ax.set_ylim(seed_xy[1] - margin, seed_xy[1] + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Projected x (km, EPSG:5070)")
    ax.set_ylabel("Projected y (km, EPSG:5070)")
    snap = customers["snap_distance_km"].to_numpy(dtype=float)
    title = (
        f"{territory_id}: {placement_mode} latent customer preview around community {seed_row['community_id']}\n"
        f"occupancy={float(seed_row['occupancy']):.0f}, "
        f"snap p50={np.quantile(snap, 0.50) * 1000:.0f}m, "
        f"p90={np.quantile(snap, 0.90) * 1000:.0f}m"
    )
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    normalized_dir = _normalized_dir(args.source_root.resolve(), str(args.territory_id))
    output_dir = args.output_dir or args.source_root.resolve() / str(args.territory_id) / "qa"
    qa_map_dir = args.source_root.resolve() / "qa_maps"
    nodes, raw_edges, seeds = _load_tables(normalized_dir)
    edges = _edge_table(nodes, raw_edges)
    selected_seed = _choose_seed(seeds, edges, args.community_id, args.road_radius_km)
    rng = np.random.default_rng(int(args.seed))
    customers = _generate_customers(
        selected_seed,
        edges,
        int(args.customer_count),
        rng,
        str(args.placement_mode),
        float(args.road_radius_km),
        float(args.weight_sigma_km),
        float(args.min_spacing_km),
        float(args.min_access_km),
        float(args.max_access_km),
    )
    customers = _add_lonlat(customers)
    community_id = str(selected_seed["community_id"])
    csv_path = output_dir / f"latent_customer_preview_{args.placement_mode}_n{args.customer_count}_{community_id}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    customers.to_csv(csv_path, index=False)

    seed_xy = np.asarray([float(selected_seed["x_km"]), float(selected_seed["y_km"])], dtype=float)
    nearby = seeds[
        np.linalg.norm(seeds[["x_km", "y_km"]].to_numpy(dtype=float) - seed_xy[None, :], axis=1)
        <= float(args.road_radius_km)
    ].copy()
    image_path = (
        qa_map_dir
        / f"latent_customer_zoom_{args.placement_mode}_n{args.customer_count}_{args.territory_id}_{community_id}.png"
    )
    _plot_preview(
        str(args.territory_id),
        selected_seed,
        edges,
        nearby,
        customers,
        image_path,
        float(args.road_radius_km),
        str(args.placement_mode),
    )
    summary = {
        "territory_id": str(args.territory_id),
        "community_id": community_id,
        "occupancy": float(selected_seed["occupancy"]),
        "placement_mode": str(args.placement_mode),
        "customer_count": int(len(customers)),
        "snap_p50_m": float(np.quantile(customers["snap_distance_km"], 0.50) * 1000.0),
        "snap_p90_m": float(np.quantile(customers["snap_distance_km"], 0.90) * 1000.0),
        "csv": str(csv_path),
        "image": str(image_path),
    }
    print(summary)


if __name__ == "__main__":
    main()
