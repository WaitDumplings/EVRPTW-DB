#!/usr/bin/env python3
"""Build the complete V2.1 M2/M3 pairing ledger and immutable Q90 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from evrptw_stage2.acceptance import (
    build_metric_pairing_ledger,
    evaluate_q90_gate,
    station_block_bootstrap_q90,
    write_metric_pairing_ledger,
)
from evrptw_stage2.provenance import resolve_git_provenance
from evrptw_stage2.amazon import load_amazon_stage2_artifacts


SOURCE_MODE = "SINGLE_STRUCTURE_DAY|SINGLE_ORDER_DAY"
PRIMARY_SCALES = ("cus100", "cus500", "cus1000")
ALL_SCALES = {"cus50": 50, "cus100": 100, "cus500": 500, "cus1000": 1000}


def _stable_u64(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _view_distributions(
    family_dir: Path,
    family: dict[str, Any],
    view_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    view = json.loads((view_dir / "view_manifest.json").read_text(encoding="utf-8"))
    indices = np.load(view_dir / view["terminal_parent_indices"], allow_pickle=False)
    customer_count = int(view["customer_count"])
    customer_parent = indices[1 : 1 + customer_count].astype(int)
    terminal_index = pd.read_parquet(family_dir / family["terminal_index"])
    customers = terminal_index.iloc[customer_parent]
    matrix_path = family_dir / family["matrix_files"]["running_time_shortest_matrix_s"]
    parent_time = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    customer_time = np.asarray(parent_time[np.ix_(customer_parent, customer_parent)], dtype=float)
    nearest = customer_time.copy()
    np.fill_diagonal(nearest, np.inf)
    m2 = nearest.min(axis=1)
    p50: list[float] = []
    p90: list[float] = []
    regions = customers["sampling_cluster_id"].astype(str).to_numpy()
    for region in sorted(set(regions)):
        local = np.flatnonzero(regions == region)
        if len(local) < 2:
            continue
        sub = customer_time[np.ix_(local, local)]
        directed = sub[~np.eye(len(local), dtype=bool)]
        p50.append(float(np.quantile(directed, 0.50)))
        p90.append(float(np.quantile(directed, 0.90)))
    structure = family["selection_report"]["amazon_structure_source"]
    t_env = float(structure["source_t_env_s"])
    source_mode = "|".join(map(str, family["order_source_report"]["source_mode"]))
    row = {
        "day_type": str(view["day_type"]),
        "scale_id": str(view["scale_id"]),
        "source_mode": source_mode,
        "generated_view_id": str(view["view_id"]),
        "structure_source_id": "+".join(map(str, structure["structure_source_ids"])),
    }
    distributions = {
        "M2": np.asarray(m2, dtype=float) / max(t_env, 1.0),
        "M3_P50": np.asarray(p50, dtype=float) / max(t_env, 1.0),
        "M3_P90": np.asarray(p90, dtype=float) / max(t_env, 1.0),
    }
    return row, distributions


def _generated_inventory(
    output_root: Path,
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    rows: list[dict[str, Any]] = []
    distributions: dict[tuple[str, str], np.ndarray] = {}
    for family_manifest_path in sorted(
        (output_root / "materialized" / "families").glob("*/family_manifest.json")
    ):
        family_dir = family_manifest_path.parent
        family = json.loads(family_manifest_path.read_text(encoding="utf-8"))
        for view_dir in sorted((family_dir / "views").iterdir()):
            row, values = _view_distributions(family_dir, family, view_dir)
            rows.append(row)
            for component, distribution in values.items():
                distributions[(row["generated_view_id"], component)] = distribution
    if not rows:
        raise FileNotFoundError("No materialized views were found for realism evaluation")
    return pd.DataFrame.from_records(rows), distributions


def _holdout_inventory(
    amazon: Any,
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], np.ndarray]]:
    holdout_days = {
        station_day_id
        for station_day_id, pool in amazon.station_day_pool.items()
        if pool == "METRIC-HOLDOUT"
    }
    rows: list[dict[str, Any]] = []
    distributions: dict[tuple[str, str, str], np.ndarray] = {}
    for station_day_id in sorted(holdout_days):
        templates = amazon.templates.loc[
            amazon.templates["station_day_id"].astype(str).eq(station_day_id)
        ].copy()
        if templates.empty:
            continue
        values = templates["station_to_stop_time_s"].to_numpy(dtype=float)
        t_env = float(np.quantile(values, 0.99))
        templates = templates.loc[templates["station_to_stop_time_s"].le(t_env)].copy()
        day_type = str(templates["day_type"].iloc[0])
        station_code = str(templates["station_code"].iloc[0])
        for scale_id, customer_count in ALL_SCALES.items():
            if len(templates) < customer_count:
                continue
            ranked = templates.copy()
            ranked["_rank"] = [
                _stable_u64(seed, "holdout_scale", scale_id, template_id)
                for template_id in ranked["template_id"].astype(str)
            ]
            selected = ranked.sort_values(["_rank", "template_id"], kind="stable").head(
                customer_count
            )
            route_ids = set(selected["route_id"].astype(str))
            route_reference = amazon.route_spatial_reference.loc[
                amazon.route_spatial_reference["route_id"].astype(str).isin(route_ids)
            ]
            rows.append(
                {
                    "day_type": day_type,
                    "scale_id": scale_id,
                    "source_mode": SOURCE_MODE,
                    "holdout_station_day_id": station_day_id,
                    "station_code": station_code,
                }
            )
            distributions[(station_day_id, scale_id, "M2")] = pd.to_numeric(
                selected["amazon_route_nearest_neighbor_time_s"], errors="coerce"
            ).dropna().to_numpy(dtype=float) / max(t_env, 1.0)
            for component, column in (
                ("M3_P50", "within_route_pairwise_time_p50_s"),
                ("M3_P90", "within_route_pairwise_time_p90_s"),
            ):
                distributions[(station_day_id, scale_id, component)] = pd.to_numeric(
                    route_reference[column], errors="coerce"
                ).dropna().to_numpy(dtype=float) / max(t_env, 1.0)
    return pd.DataFrame.from_records(rows), distributions


def _attach_distances(
    ledger: pd.DataFrame,
    generated: dict[tuple[str, str], np.ndarray],
    holdout: dict[tuple[str, str, str], np.ndarray],
) -> pd.DataFrame:
    result = ledger.copy()
    values: list[float] = []
    for row in result.itertuples(index=False):
        component = str(row.metric_component)
        scale_id = str(row.scale_id)
        left = holdout[(str(row.holdout_station_day_left), scale_id, component)]
        if row.pair_kind == "generated_to_holdout":
            right = generated[(str(row.generated_view_id), component)]
        else:
            right = holdout[(str(row.holdout_station_day_right), scale_id, component)]
        values.append(
            float(wasserstein_distance(left, right))
            if len(left) and len(right)
            else float("nan")
        )
    result["distance"] = values
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-root", type=Path, required=True)
    parser.add_argument("--amazon-artifact-root", type=Path, required=True)
    parser.add_argument("--cohort-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    code_provenance = resolve_git_provenance(
        Path(__file__).resolve().parents[2],
        require_clean=True,
        require_branch="stage2-repair-candidate",
    )

    amazon = load_amazon_stage2_artifacts(
        args.amazon_artifact_root,
        cohort_split_path=args.cohort_split,
    )
    generated_rows, generated_distributions = _generated_inventory(args.instance_root)
    holdout_rows, holdout_distributions = _holdout_inventory(amazon, seed=args.seed)
    ledger = build_metric_pairing_ledger(generated_rows, holdout_rows)
    distances = _attach_distances(ledger, generated_distributions, holdout_distributions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metric_pairing_ledger(ledger, args.output_dir / "metric_pairing_ledger.parquet")
    distances.to_parquet(args.output_dir / "metric_distances.parquet", index=False)
    required = [
        (day_type, scale_id, SOURCE_MODE)
        for day_type in ("weekday", "weekend")
        for scale_id in PRIMARY_SCALES
    ]
    gate = evaluate_q90_gate(distances, required_primary_strata=required)
    gate["code_provenance"] = code_provenance
    gate["generated_view_count"] = int(generated_rows["generated_view_id"].nunique())
    gate["qualified_holdout_station_day_count"] = int(
        holdout_rows["holdout_station_day_id"].nunique()
    )
    gate["holdout_station_count"] = int(holdout_rows["station_code"].nunique())
    bootstrap = station_block_bootstrap_q90(
        distances,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    bootstrap["code_provenance"] = code_provenance
    _write_json(args.output_dir / "station_block_bootstrap.json", bootstrap)
    _write_json(args.output_dir / "q90_gate.json", gate)
    print(json.dumps(gate, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
