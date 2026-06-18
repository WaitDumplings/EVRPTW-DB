"""Update Gurobi summary/time-trace CSVs with route-merge postprocessed routes.

The script is intentionally narrow: it reads the merge summary produced by
`postprocess_solution_route_merges.py`, backs up the target CSVs, and updates
only matching instance rows. It does not alter Gurobi bounds or status fields.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def parse_routes(text: str) -> list[list[int]]:
    routes = json.loads(text)
    return [[int(node) for node in route] for route in routes]


def flatten_routes(routes: list[list[int]]) -> list[int]:
    sequence: list[int] = []
    for route in routes:
        if not sequence:
            sequence.extend(route)
        else:
            sequence.extend(route[1:])
    return sequence


def backup(path: Path, tag: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}.before_route_merge_{tag}_{timestamp}.bak")
    shutil.copy2(path, backup_path)
    return backup_path


def update_summary(
    summary_csv: Path,
    updates: dict[str, dict[str, Any]],
    *,
    abs_tolerance_km: float,
    rel_tolerance: float,
    tag: str,
) -> tuple[Path, int]:
    backup_path = backup(summary_csv, tag)
    df = pd.read_csv(summary_csv)
    updated = 0

    for instance_id, update in updates.items():
        mask = df["instance_id"].astype(str) == instance_id
        if int(mask.sum()) != 1:
            raise ValueError(f"Expected exactly one summary row for {instance_id}, found {int(mask.sum())}")

        routes = update["routes"]
        route_sequence = flatten_routes(routes)
        old_distance = float(update["distance_before_km"])
        tolerance = max(float(abs_tolerance_km), float(rel_tolerance) * old_distance)

        df.loc[mask, "objective_distance_km"] = float(update["distance_after_km"])
        df.loc[mask, "vehicle_count"] = int(update["route_count_after"])
        df.loc[mask, "routes_json"] = json.dumps(routes, separators=(",", ":"))
        df.loc[mask, "route_sequence_json"] = json.dumps(route_sequence, separators=(",", ":"))
        if "tie_break_applied" in df.columns:
            df.loc[mask, "tie_break_applied"] = True
        if "distance_tolerance" in df.columns:
            df.loc[mask, "distance_tolerance"] = tolerance
        if "stage1_best_distance_km" in df.columns:
            df.loc[mask, "stage1_best_distance_km"] = old_distance
        updated += 1

    df.to_csv(summary_csv, index=False)
    return backup_path, updated


def update_time_trace(
    time_trace_csv: Path,
    updates: dict[str, dict[str, Any]],
    *,
    checkpoint_s: float,
    tag: str,
) -> tuple[Path, int]:
    backup_path = backup(time_trace_csv, tag)
    df = pd.read_csv(time_trace_csv)
    updated = 0

    for instance_id, update in updates.items():
        mask = (df["instance_id"].astype(str) == instance_id) & (
            pd.to_numeric(df["checkpoint_s"], errors="coerce").round(6) == round(float(checkpoint_s), 6)
        )
        if int(mask.sum()) != 1:
            raise ValueError(
                f"Expected exactly one time-trace row for {instance_id} at checkpoint {checkpoint_s}, "
                f"found {int(mask.sum())}"
            )

        routes = update["routes"]
        route_sequence = flatten_routes(routes)
        df.loc[mask, "objective_distance_km"] = float(update["distance_after_km"])
        df.loc[mask, "vehicle_count"] = int(update["route_count_after"])
        df.loc[mask, "routes_json"] = json.dumps(routes, separators=(",", ":"))
        df.loc[mask, "route_sequence_json"] = json.dumps(route_sequence, separators=(",", ":"))
        if "source" in df.columns:
            df.loc[mask, "source"] = "checkpoint_incumbent_route_merge_postprocessed"
        updated += 1

    df.to_csv(time_trace_csv, index=False)
    return backup_path, updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-summary-csv", type=Path, required=True)
    parser.add_argument("--gurobi-summary-csv", type=Path, required=True)
    parser.add_argument("--gurobi-time-trace-csv", type=Path, required=True)
    parser.add_argument("--checkpoint-s", type=float, default=7200.0)
    parser.add_argument("--abs-tolerance-km", type=float, default=0.05)
    parser.add_argument("--rel-tolerance", type=float, default=0.001)
    parser.add_argument("--tag", default="cus50_train_route_ge5")
    args = parser.parse_args()

    merge_df = pd.read_csv(args.merge_summary_csv)
    updates: dict[str, dict[str, Any]] = {}
    for row in merge_df.to_dict(orient="records"):
        routes = parse_routes(str(row["merged_routes_json"]))
        updates[str(row["instance_id"])] = {
            "routes": routes,
            "route_count_before": int(row["route_count_before"]),
            "route_count_after": int(row["route_count_after"]),
            "distance_before_km": float(row["distance_before_km"]),
            "distance_after_km": float(row["distance_after_km"]),
            "distance_delta_km": float(row["distance_delta_km"]),
        }

    summary_backup, summary_updated = update_summary(
        args.gurobi_summary_csv,
        updates,
        abs_tolerance_km=args.abs_tolerance_km,
        rel_tolerance=args.rel_tolerance,
        tag=args.tag,
    )
    trace_backup, trace_updated = update_time_trace(
        args.gurobi_time_trace_csv,
        updates,
        checkpoint_s=args.checkpoint_s,
        tag=args.tag,
    )

    print(
        json.dumps(
            {
                "updated_instances": len(updates),
                "summary_updated_rows": summary_updated,
                "time_trace_updated_rows": trace_updated,
                "summary_backup": str(summary_backup),
                "time_trace_backup": str(trace_backup),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
