#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from EVRPTW_Benchmark.Reinforcement_Learning.common.euclidean import (
    EUCLIDEAN_MANIFEST_SCHEMA,
    haversine_matrix_km,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.stage2_data import Stage2TaskPool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate train-only E-track speeds.")
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--scale", default="Cus100")
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--views-per-day-type", type=int, default=100)
    parser.add_argument("--pairs-per-view", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.train_index)
    required = {"view_id", "split_id", "track_id", "scale_id", "day_type"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"train index is missing columns: {missing}")
    if set(frame["split_id"].astype(str)) != {"train"} or set(frame["track_id"].astype(str)) != {"train"}:
        raise ValueError("Euclidean calibration index must contain train rows only")
    scale = str(args.scale).lower()
    frame = frame.loc[frame["scale_id"].astype(str).str.lower() == scale].copy()
    if frame.empty:
        raise ValueError(f"no {args.scale} rows in train index")
    pool = Stage2TaskPool(
        dataset_path=args.train_index,
        family_root=args.family_root,
        scale=args.scale,
        split_ids="train",
        track_ids="train",
        seed=args.seed,
    )
    by_id = {task.view_id: task for task in pool.tasks}
    speeds: dict[str, float] = {}
    evidence: dict[str, dict] = {}
    for day_index, day_type in enumerate(("weekday", "weekend")):
        rows = frame.loc[frame["day_type"].astype(str) == day_type]
        if len(rows) < args.views_per_day_type:
            raise ValueError(f"insufficient {day_type} calibration views")
        rng = np.random.default_rng(
            np.random.SeedSequence([args.seed, day_index, 0x4555434C])
        )
        chosen = rows.iloc[
            rng.choice(len(rows), size=args.views_per_day_type, replace=False)
        ]["view_id"].astype(str).tolist()
        distance_sum = 0.0
        time_sum = 0.0
        pair_count = 0
        for view_offset, view_id in enumerate(chosen):
            instance = pool.instance(by_id[view_id])
            coordinates = np.vstack(
                [instance.depot.reshape(1, 2), instance.customers, instance.charging_stations]
            )
            euclidean = haversine_matrix_km(coordinates)
            graph_time = np.asarray(instance.shortest_time_matrix_s, dtype=np.float64)
            n = len(coordinates)
            pair_rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, day_index, view_offset, 0x4F445052])
            )
            origins = pair_rng.integers(0, n, size=args.pairs_per_view)
            destinations = pair_rng.integers(0, n - 1, size=args.pairs_per_view)
            destinations += destinations >= origins
            valid = np.isfinite(graph_time[origins, destinations]) & (
                graph_time[origins, destinations] > 0.0
            )
            distance_sum += float(euclidean[origins[valid], destinations[valid]].sum())
            time_sum += float(graph_time[origins[valid], destinations[valid]].sum())
            pair_count += int(valid.sum())
        if distance_sum <= 0.0 or time_sum <= 0.0:
            raise RuntimeError(f"invalid Euclidean calibration totals for {day_type}")
        speeds[day_type] = 3600.0 * distance_sum / time_sum
        evidence[day_type] = {
            "selected_view_ids": chosen,
            "requested_pairs_per_view": args.pairs_per_view,
            "valid_pair_count": pair_count,
            "euclidean_distance_sum_km": distance_sum,
            "graph_travel_time_sum_s": time_sum,
        }
    manifest = {
        "schema": EUCLIDEAN_MANIFEST_SCHEMA,
        "calibration_split": "train",
        "calibration_track": "train",
        "calibration_scale": args.scale,
        "seed": args.seed,
        "views_per_day_type": args.views_per_day_type,
        "pairs_per_view": args.pairs_per_view,
        "speed_kmh_by_day_type": speeds,
        "day_type_evidence": evidence,
        "shared_across_training_cities": True,
        "validation_or_test_used": False,
        "file_hash_validation_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
