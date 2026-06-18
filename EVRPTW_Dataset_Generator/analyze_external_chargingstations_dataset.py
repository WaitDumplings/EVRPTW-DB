from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = Path("/Users/maojie/Downloads/EVRPTW_ChargingStations_Dataset")
OUTPUT_DIR = REPO_ROOT / "EVRPTW_Dataset_Generator" / "analysis_outputs" / "external_chargingstations_dataset"
OURS_SUMMARY = REPO_ROOT / "EVRPTW_Dataset" / "Geo_AC_v1" / "source_data_na_us20" / "metadata" / "dataset_summary.json"
OURS_TERRITORY_TABLE = REPO_ROOT / "EVRPTW_Dataset" / "Geo_AC_v1" / "source_data_na_us20" / "metadata" / "territory_table.csv"
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib_cache"))


def parse_instance(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 7:
                continue
            try:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                demand = float(parts[3])
                ready = float(parts[4])
                due = float(parts[5])
                service = float(parts[6])
            except ValueError:
                continue
            kind = "customer"
            if node_id == 0:
                kind = "depot"
            elif node_id > 1000:
                kind = "station"
            rows.append(
                {
                    "instance": path.stem,
                    "node_id": node_id,
                    "x": x,
                    "y": y,
                    "demand": demand,
                    "ready_time": ready,
                    "due_date": due,
                    "service_time": service,
                    "kind": kind,
                }
            )
    if not rows:
        raise ValueError(f"No node rows parsed from {path}")
    return pd.DataFrame(rows)


def read_station_attrs(path: Path) -> pd.DataFrame:
    attrs = pd.read_csv(path)
    attrs["price_mean"] = (attrs["price_mu"] + 0.5 * attrs["price_sigma"] ** 2).map(math.exp)
    attrs["wait_mean"] = (attrs["wait_mu"] + 0.5 * attrs["wait_sigma"] ** 2).map(math.exp)
    return attrs


def class_name(instance: str) -> str:
    if instance.startswith("RC"):
        return instance[:3]
    return instance[:2]


def collect_variant(variant_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_frames = []
    attr_frames = []
    for txt_path in sorted(variant_dir.glob("*.txt")):
        nodes = parse_instance(txt_path)
        node_frames.append(nodes)
        attr_path = txt_path.with_name(f"{txt_path.stem}_stations.csv")
        if attr_path.exists():
            attrs = read_station_attrs(attr_path)
            attrs.insert(0, "instance", txt_path.stem)
            attr_frames.append(attrs)
    return pd.concat(node_frames, ignore_index=True), pd.concat(attr_frames, ignore_index=True)


def plot_samples(nodes: pd.DataFrame, attrs: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    samples = ["C1_10_1", "C2_10_1", "R1_10_1", "R2_10_1", "RC1_10_1", "RC2_10_1"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5))
    for ax, name in zip(axes.ravel(), samples):
        sub = nodes[nodes["instance"] == name]
        customers = sub[sub["kind"] == "customer"]
        depot = sub[sub["kind"] == "depot"]
        stations = sub[sub["kind"] == "station"].merge(
            attrs[attrs["instance"] == name], left_on="node_id", right_on="station_id", how="left"
        )
        ax.scatter(customers["x"], customers["y"], s=4, c="#2563eb", alpha=0.28, linewidths=0, label="Customers")
        station_plot = ax.scatter(
            stations["x"],
            stations["y"],
            s=18,
            c=stations["density_score"],
            cmap="viridis",
            alpha=0.88,
            edgecolors="white",
            linewidths=0.2,
            label="Synthetic charging stations",
        )
        ax.scatter(depot["x"], depot["y"], s=150, c="#dc2626", marker="*", edgecolors="white", linewidths=0.8, label="Depot")
        ax.set_title(f"{name}: {len(customers)} customers, {len(stations)} stations")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-10, 510)
        ax.set_ylim(-10, 510)
        ax.grid(alpha=0.18, linewidth=0.5)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    cax = fig.add_axes([0.925, 0.25, 0.018, 0.50])
    cbar = fig.colorbar(station_plot, cax=cax)
    cbar.set_label("Station density_score")
    fig.suptitle("EVRPTW_ChargingStations_Dataset sample layouts", fontsize=16, fontweight="bold")
    fig.subplots_adjust(left=0.05, right=0.885, top=0.92, bottom=0.12, wspace=0.12, hspace=0.20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def plot_attributes(attrs: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fields = [
        ("density_score", "Density score"),
        ("charging_rate", "Charging rate"),
        ("price_mean", "Expected price"),
        ("wait_mean", "Expected wait time"),
    ]
    for ax, (field, title) in zip(axes.ravel(), fields):
        ax.hist(attrs[field], bins=40, color="#0f766e", alpha=0.82, edgecolor="white", linewidth=0.3)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("Synthetic charging station attribute distributions", fontsize=16, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def plot_class_summary(nodes: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    rows = []
    for instance, sub in nodes.groupby("instance"):
        customers = sub[sub["kind"] == "customer"]
        stations = sub[sub["kind"] == "station"]
        rows.append(
            {
                "instance": instance,
                "class": class_name(instance),
                "customers": len(customers),
                "stations": len(stations),
                "x_min": customers["x"].min(),
                "x_max": customers["x"].max(),
                "y_min": customers["y"].min(),
                "y_max": customers["y"].max(),
                "customer_bbox_area": (customers["x"].max() - customers["x"].min()) * (customers["y"].max() - customers["y"].min()),
            }
        )
    summary = pd.DataFrame(rows)
    class_summary = summary.groupby("class").agg(
        instances=("instance", "count"),
        customers=("customers", "mean"),
        stations=("stations", "mean"),
        bbox_area=("customer_bbox_area", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].bar(class_summary["class"], class_summary["stations"], color="#059669")
    axes[0].set_title("Stations per instance by class")
    axes[0].set_ylabel("Mean synthetic stations")
    axes[0].grid(axis="y", alpha=0.18)
    axes[1].bar(class_summary["class"], class_summary["bbox_area"], color="#2563eb")
    axes[1].set_title("Customer bounding-box area by class")
    axes[1].set_ylabel("Coordinate area")
    axes[1].grid(axis="y", alpha=0.18)
    fig.suptitle("Homberger-based dataset scale summary", fontsize=15, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)
    return summary


def write_comparison(
    external_nodes: pd.DataFrame,
    external_attrs: pd.DataFrame,
    external_summary: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    ours = json.loads(OURS_SUMMARY.read_text(encoding="utf-8"))
    our_territories = list(csv.DictReader(OURS_TERRITORY_TABLE.open("r", encoding="utf-8")))
    station_xy = external_nodes[external_nodes["kind"] == "station"][["instance", "x", "y"]].drop_duplicates()
    unique_station_xy = station_xy[["x", "y"]].drop_duplicates()
    station_signature_counts = []
    for _, group in external_nodes[external_nodes["kind"] == "station"].groupby("instance"):
        coords = tuple(sorted(zip(group["x"].astype(int), group["y"].astype(int))))
        station_signature_counts.append(hash(coords))
    signature_reuse = pd.Series(station_signature_counts).value_counts()
    external = {
        "source": "EVRPTW_ChargingStations_Dataset",
        "base_instances": int(external_summary["instance"].nunique()),
        "variants": 2,
        "customers_per_instance": int(external_summary["customers"].median()),
        "stations_per_instance": int(external_summary["stations"].median()),
        "total_variant_a_station_rows": int((external_nodes["kind"] == "station").sum()),
        "unique_station_coordinates_variant_a": int(len(unique_station_xy)),
        "station_coordinate_layouts_variant_a": int(len(signature_reuse)),
        "largest_reused_station_layout_count": int(signature_reuse.max()),
        "coordinate_system": "synthetic Homberger/Solomon 2D coordinates",
        "road_network": "none",
        "charger_source": "synthetic random stations in customer bounding box",
        "customer_source": "Homberger synthetic VRPTW customers",
        "time_window_regimes": ["original", "distance-relaxed"],
        "station_attributes": {
            "charging_rate_min": float(external_attrs["charging_rate"].min()),
            "charging_rate_max": float(external_attrs["charging_rate"].max()),
            "expected_price_min": float(external_attrs["price_mean"].min()),
            "expected_price_max": float(external_attrs["price_mean"].max()),
            "expected_wait_min": float(external_attrs["wait_mean"].min()),
            "expected_wait_max": float(external_attrs["wait_mean"].max()),
        },
    }
    geo_ac = {
        "source": "Geo-AC-v1 / NA-US-20",
        "territories": int(ours["territory_count"]),
        "fixed_eval_instances": int(ours["standard_eval"]["total_instances"]),
        "latent_customers": int(ours["totals"]["latent_customers"]),
        "customer_seeds": int(ours["totals"]["customer_seeds"]),
        "charging_stations": int(ours["totals"]["charging_stations"]),
        "depot_candidates": int(ours["totals"]["depot_candidates"]),
        "road_nodes": int(ours["totals"]["road_nodes"]),
        "road_edges": int(ours["totals"]["road_edges"]),
        "coordinate_system": "real lon/lat and projected road graph coordinates",
        "road_network": "OSM/TIGER routable graph",
        "charger_source": "public AFDC/NREL charging stations",
        "customer_source": "ACS occupancy-weighted road-frontage latent customers",
        "territory_min_latent_customers": int(min(int(row["latent_customers"]) for row in our_territories)),
        "territory_max_latent_customers": int(max(int(row["latent_customers"]) for row in our_territories)),
    }
    payload = {"external_dataset": external, "geo_ac_v1_na_us20": geo_ac}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def write_index(output_dir: Path, payload: dict[str, Any]) -> None:
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>External EVRPTW ChargingStations Dataset QA</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #172033; }}
    img {{ max-width: 100%; border: 1px solid #d7dde7; margin: 12px 0 28px; }}
    code, pre {{ background: #f5f7fb; padding: 2px 4px; border-radius: 4px; }}
    table {{ border-collapse: collapse; margin: 16px 0 28px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 12px; text-align: left; }}
  </style>
</head>
<body>
  <h1>EVRPTW_ChargingStations_Dataset QA</h1>
  <p>Downloaded dataset path: <code>{EXTERNAL_ROOT}</code></p>
  <h2>Sample layouts</h2>
  <img src="external_samples_by_class.png" alt="External sample layouts">
  <h2>Station attributes</h2>
  <img src="external_station_attributes.png" alt="External station attributes">
  <h2>Class summary</h2>
  <img src="external_class_summary.png" alt="External class summary">
  <h2>Comparison JSON</h2>
  <pre>{json.dumps(payload, indent=2)}</pre>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes_a, attrs_a = collect_variant(EXTERNAL_ROOT / "variant_A_original_TW")
    summary = plot_class_summary(nodes_a, OUTPUT_DIR / "external_class_summary.png")
    plot_samples(nodes_a, attrs_a, OUTPUT_DIR / "external_samples_by_class.png")
    plot_attributes(attrs_a, OUTPUT_DIR / "external_station_attributes.png")
    summary.to_csv(OUTPUT_DIR / "external_instance_summary.csv", index=False)
    attrs_a.describe(include="all").to_csv(OUTPUT_DIR / "external_station_attribute_describe.csv")
    payload = write_comparison(nodes_a, attrs_a, summary, OUTPUT_DIR / "external_vs_geo_ac_summary.json")
    write_index(OUTPUT_DIR, payload)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "external_instances": int(summary["instance"].nunique())}, indent=2))


if __name__ == "__main__":
    main()
