from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))

from evrptw_core.io import INSTANCE_BUNDLE_FORMAT, iter_instance_dicts  # noqa: E402


DEFAULT_TERRITORIES = (
    "atlanta_ga_fulton_county",
    "la_ca_san_gabriel_pomona_industry",
    "maricopa_az_east_valley",
    "new_york_ny_queens_county",
    "king_wa_seattle_eastside",
)


def terminal_coordinates(payload: dict[str, Any]) -> np.ndarray:
    parts = [
        np.asarray(payload["depot"], dtype=np.float64).reshape(1, 2),
        np.asarray(payload["customers"], dtype=np.float64),
        np.asarray(payload["charging_stations"], dtype=np.float64),
    ]
    return np.vstack(parts)


def offdiag_values(matrix: np.ndarray) -> np.ndarray:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    values = np.asarray(matrix, dtype=np.float64)[mask]
    return values[np.isfinite(values)]


def offdiag_pair_values(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = ~np.eye(a.shape[0], dtype=bool)
    av = np.asarray(a, dtype=np.float64)[mask]
    bv = np.asarray(b, dtype=np.float64)[mask]
    keep = np.isfinite(av) & np.isfinite(bv)
    return av[keep], bv[keep]


def correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    av, bv = offdiag_pair_values(a, b)
    if av.size < 2 or float(np.std(av)) <= 1e-12 or float(np.std(bv)) <= 1e-12:
        return None
    return float(np.corrcoef(av, bv)[0, 1])


def quantiles(values: np.ndarray, qs: tuple[float, ...] = (0.5, 0.9, 0.99)) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {f"p{int(q * 100):02d}": float("nan") for q in qs}
    return {f"p{int(q * 100):02d}": float(np.quantile(finite, q)) for q in qs}


def pseudo_elevation_m(coords: np.ndarray) -> np.ndarray:
    center = np.median(coords, axis=0)
    rel = coords - center
    elev = (
        160.0
        + 22.0 * rel[:, 1]
        + 9.0 * rel[:, 0]
        + 95.0 * np.sin(rel[:, 0] / 7.5)
        + 70.0 * np.cos(rel[:, 1] / 8.5)
        + 36.0 * np.sin((rel[:, 0] + rel[:, 1]) / 5.5)
    )
    return np.clip(elev, 0.0, 1200.0)


def build_raw_time_matrix_s(payload: dict[str, Any], distance: np.ndarray, coords: np.ndarray) -> np.ndarray:
    vehicle = payload.get("vehicle", {})
    speed_profile = payload.get("speed_profile", {})
    base_speed = float(
        speed_profile.get("effective_speed_kmh")
        or vehicle.get("design_speed_kmh")
        or 18.0
    )
    base_speed = max(base_speed, 1.0)

    center = np.median(coords, axis=0)
    span = max(float(np.quantile(np.linalg.norm(coords - center, axis=1), 0.9)), 1.0)
    src = coords[:, None, :]
    dst = coords[None, :, :]
    mid = 0.5 * (src + dst)
    delta = dst - src
    mid_radius = np.linalg.norm(mid - center, axis=2)
    centrality = np.exp(-((mid_radius / max(0.7 * span, 1e-6)) ** 2))
    pattern = (
        0.34 * np.sin((mid[:, :, 0] - center[0]) / 4.8)
        + 0.28 * np.cos((mid[:, :, 1] - center[1]) / 5.6)
        + 0.24 * np.sin(0.20 * delta[:, :, 0] - 0.15 * delta[:, :, 1])
    )
    pair_multiplier = np.clip(1.0 - 0.58 * centrality + pattern, 0.30, 1.85)
    speed = np.maximum(base_speed * pair_multiplier, 1.0)
    out = np.asarray(distance, dtype=np.float64) / speed * 3600.0
    np.fill_diagonal(out, 0.0)
    return out.astype(np.float32)


def build_energy_matrix_kwh(payload: dict[str, Any], distance: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vehicle = payload.get("vehicle", {})
    base_rate = float(vehicle.get("consumption_kwh_per_km", 0.404))
    base_rate = max(base_rate, 1e-6)
    mass_kg = float(vehicle.get("gross_vehicle_mass_kg", 4300.0))
    drivetrain_efficiency = 0.82
    regen_efficiency = 0.38

    elev = pseudo_elevation_m(coords)
    center = np.median(coords, axis=0)
    span = max(float(np.quantile(np.linalg.norm(coords - center, axis=1), 0.9)), 1.0)
    src = coords[:, None, :]
    dst = coords[None, :, :]
    mid = 0.5 * (src + dst)
    mid_radius = np.linalg.norm(mid - center, axis=2)
    centrality = np.exp(-((mid_radius / max(0.75 * span, 1e-6)) ** 2))
    surface_factor = np.clip(
        0.86
        + 0.34 * centrality
        + 0.24 * np.sin((mid[:, :, 0] - center[0]) / 6.5)
        - 0.18 * np.cos((mid[:, :, 1] - center[1]) / 7.5)
        + 0.16 * np.sin(0.12 * (dst[:, :, 0] - src[:, :, 0]) + 0.09 * (dst[:, :, 1] - src[:, :, 1])),
        0.48,
        1.62,
    )
    base = np.asarray(distance, dtype=np.float64) * base_rate * surface_factor
    delta_m = elev[None, :] - elev[:, None]
    potential = mass_kg * 9.80665 * np.abs(delta_m) / 3.6e6
    uphill = np.where(delta_m > 0.0, potential / drivetrain_efficiency, 0.0)
    downhill_credit = np.where(delta_m < 0.0, potential * regen_efficiency, 0.0)
    energy = base + uphill - downhill_credit
    energy = np.maximum(energy, np.asarray(distance, dtype=np.float64) * base_rate * 0.45)
    np.fill_diagonal(energy, 0.0)
    return energy.astype(np.float32), elev.astype(np.float32)


def select_payloads(source: Path, territories: tuple[str, ...], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for payload in iter_instance_dicts(source):
        region = str(payload.get("region_id", ""))
        if territories and region not in territories:
            continue
        if region in seen:
            continue
        selected.append(deepcopy(payload))
        seen.add(region)
        if len(selected) >= int(limit):
            break
    if len(selected) < int(limit):
        for payload in iter_instance_dicts(source):
            key = str(payload.get("instance_id", ""))
            if any(str(x.get("instance_id", "")) == key for x in selected):
                continue
            selected.append(deepcopy(payload))
            if len(selected) >= int(limit):
                break
    if len(selected) < int(limit):
        raise RuntimeError(f"Only selected {len(selected)} instances from {source}; requested {limit}.")
    return selected


def write_bundle(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(
            {
                "format": INSTANCE_BUNDLE_FORMAT,
                "num_instances": len(payloads),
                "description": "EVRPTW multi-metric Cus50 smoke bundle",
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        for payload in payloads:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def enrich_payload(payload: dict[str, Any], index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    out = deepcopy(payload)
    source_id = str(out.get("instance_id", f"source_{index:06d}"))
    out["instance_id"] = f"smoke_Cus50_{index:06d}"
    distance = np.asarray(out["distance_matrix_km"], dtype=np.float32)
    coords = terminal_coordinates(out)
    raw_time = build_raw_time_matrix_s(out, distance, coords)
    energy, elevation = build_energy_matrix_kwh(out, distance, coords)
    out["raw_travel_time_matrix_s"] = raw_time
    out["energy_matrix_kwh"] = energy
    out["ev_transition_time_matrix_s"] = None
    out["shortest_time_matrix_s"] = None
    metadata = dict(out.get("metadata", {}))
    metadata.update(
        {
            "source_instance_id": source_id,
            "source_region_id": payload.get("region_id", ""),
            "multimetric_smoke": True,
            "multimetric_time_model": "static_pair_speed_proxy_v1",
            "multimetric_energy_model": "static_pseudo_elevation_proxy_v1",
            "terminal_elevation_proxy_m": elevation,
            "time_matrix_storage": {
                "distance_matrix_km": True,
                "raw_travel_time_matrix_s": True,
                "ev_transition_time_matrix_s": False,
                "shortest_time_matrix_s": False,
                "energy_matrix_kwh": True,
            },
        }
    )
    out["metadata"] = metadata
    diagnostics = diagnostics_for(out, payload)
    return out, diagnostics


def diagnostics_for(multimetric: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    del baseline
    distance = np.asarray(multimetric["distance_matrix_km"], dtype=np.float64)
    raw_time = np.asarray(multimetric["raw_travel_time_matrix_s"], dtype=np.float64)
    energy = np.asarray(multimetric["energy_matrix_kwh"], dtype=np.float64)
    n_customers = int(np.asarray(multimetric["customers"]).shape[0])
    n_cs = int(np.asarray(multimetric["charging_stations"]).shape[0])
    battery = float(multimetric.get("vehicle", {}).get("battery_capacity_kwh", 100.0))
    off_energy = offdiag_values(energy)
    off_distance = offdiag_values(distance)
    energy_ratio = off_energy / max(battery, 1e-12)
    customer_nodes = np.arange(1, n_customers + 1, dtype=int)
    cs_nodes = np.arange(n_customers + 1, n_customers + 1 + n_cs, dtype=int)
    if cs_nodes.size:
        customer_to_cs = energy[np.ix_(customer_nodes, cs_nodes)]
        cs_to_customer = energy[np.ix_(cs_nodes, customer_nodes)]
        customer_has_outbound_cs = np.min(customer_to_cs, axis=1) <= battery + 1e-7
        customer_has_inbound_cs = np.min(cs_to_customer, axis=0) <= battery + 1e-7
        customer_cs_roundtrip = customer_has_outbound_cs & customer_has_inbound_cs
    else:
        customer_cs_roundtrip = np.zeros(n_customers, dtype=bool)
    depot_to_customer = energy[0, customer_nodes]
    customer_to_depot = energy[customer_nodes, 0]
    asym = np.abs(energy - energy.T)
    asym_vals = offdiag_values(asym)
    speed = np.divide(
        off_distance * 3600.0,
        offdiag_values(raw_time),
        out=np.full_like(off_distance, np.nan),
        where=offdiag_values(raw_time) > 0.0,
    )
    row = {
        "instance_id": multimetric["instance_id"],
        "source_instance_id": multimetric.get("metadata", {}).get("source_instance_id", ""),
        "region_id": multimetric.get("region_id", ""),
        "day_type": multimetric.get("day_type", ""),
        "num_customers": n_customers,
        "num_charging_stations": n_cs,
        "battery_capacity_kwh": battery,
        "distance_time_corr": correlation(distance, raw_time),
        "distance_energy_corr": correlation(distance, energy),
        "direct_arc_feasible_share": float(np.mean(off_energy <= battery + 1e-7)) if off_energy.size else float("nan"),
        "customer_has_reachable_cs_share": float(np.mean(customer_cs_roundtrip)) if n_customers else float("nan"),
        "depot_customer_direct_share": float(np.mean((depot_to_customer <= battery + 1e-7) & (customer_to_depot <= battery + 1e-7))) if n_customers else float("nan"),
        "energy_asymmetry_max_kwh": float(np.nanmax(asym_vals)) if asym_vals.size else 0.0,
        "energy_asymmetry_p90_kwh": float(np.nanquantile(asym_vals, 0.9)) if asym_vals.size else 0.0,
    }
    for prefix, values in (
        ("distance_km", off_distance),
        ("speed_kmh", speed),
        ("energy_kwh", off_energy),
        ("energy_ratio_to_battery", energy_ratio),
    ):
        for key, value in quantiles(values).items():
            row[f"{prefix}_{key}"] = value
        row[f"{prefix}_max"] = float(np.nanmax(values)) if np.asarray(values).size else float("nan")
    return row


def write_diagnostics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path = path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")


def write_readme(output_root: Path, source: Path) -> None:
    text = f"""# Cus50 Multi-Metric Smoke Bundle

Source bundle:

```text
{source}
```

Contents:

- `baseline_singlemetric/instances.pkl`: same five source instances without new time/energy matrices. The updated Gurobi solver falls back to the old single-metric semantics.
- `multimetric/instances.pkl`: same five instances with `raw_travel_time_matrix_s` and `energy_matrix_kwh`.
- `diagnostics/multimetric_summary.csv`: correlation, energy/battery scale, CS reachability, and asymmetry diagnostics.

Run on a Gurobi machine from the repository root:

```bash
conda run -n maojie python EVRPTW_Benchmark/Exact/Gurobi_Solver/run_gurobi.py \\
  --dataset_path {output_root}/baseline_singlemetric/instances.pkl \\
  --save_path {output_root}/gurobi_baseline_singlemetric \\
  --time_limit_s 7200 \\
  --checkpoints_s 7200 \\
  --threads 1 \\
  --verbose

conda run -n maojie python EVRPTW_Benchmark/Exact/Gurobi_Solver/run_gurobi.py \\
  --dataset_path {output_root}/multimetric/instances.pkl \\
  --save_path {output_root}/gurobi_multimetric \\
  --time_limit_s 7200 \\
  --checkpoints_s 7200 \\
  --threads 1 \\
  --verbose
```

Compare:

- `gurobi_baseline_singlemetric/gurobi_summary.csv`
- `gurobi_multimetric/gurobi_summary.csv`

The multi-metric bundle is for solver smoke testing. Its energy matrix uses a deterministic pseudo-elevation proxy, not a public DEM. It is intentionally marked in instance metadata as `static_pseudo_elevation_proxy_v1`.
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 5-instance Cus50 multi-metric Gurobi smoke bundle.")
    parser.add_argument(
        "--source",
        default="EVRPTW_Dataset/Geo_AC_v1/release/dataset_v1/dataset/val/Cus50/instances.pkl",
        help="Source Cus50 instances.pkl bundle.",
    )
    parser.add_argument(
        "--output-root",
        default="EVRPTW_Dataset/Geo_AC_v1/multimetric_smoke_cus50_5",
        help="Output smoke bundle root.",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--territories", default=",".join(DEFAULT_TERRITORIES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output_root = Path(args.output_root)
    territories = tuple(item.strip() for item in args.territories.split(",") if item.strip())
    payloads = select_payloads(source, territories, int(args.limit))
    baseline_payloads: list[dict[str, Any]] = []
    multimetric_payloads: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for idx, payload in enumerate(payloads):
        base = deepcopy(payload)
        base["instance_id"] = f"smoke_Cus50_{idx:06d}"
        base_meta = dict(base.get("metadata", {}))
        base_meta["source_instance_id"] = payload.get("instance_id", "")
        base_meta["source_region_id"] = payload.get("region_id", "")
        base_meta["multimetric_smoke_baseline"] = True
        base["metadata"] = base_meta
        baseline_payloads.append(base)
        enriched, row = enrich_payload(payload, idx)
        multimetric_payloads.append(enriched)
        diagnostics.append(row)

    write_bundle(output_root / "baseline_singlemetric" / "instances.pkl", baseline_payloads)
    write_bundle(output_root / "multimetric" / "instances.pkl", multimetric_payloads)
    write_diagnostics(output_root / "diagnostics" / "multimetric_summary.csv", diagnostics)
    write_readme(output_root, source)
    print(f"Wrote {len(multimetric_payloads)} smoke instances to {output_root}")


if __name__ == "__main__":
    main()
