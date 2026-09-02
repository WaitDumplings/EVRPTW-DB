#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def collect(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    instance_frames = []
    training_rows: list[dict[str, Any]] = []
    for provenance_path in output_root.rglob("provenance.json"):
        directory = provenance_path.parent
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        job = provenance["job"]
        result_path = directory / "job_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        if job["kind"] in {"train", "pilot"}:
            training = directory / "training_result.json"
            payload = json.loads(training.read_text(encoding="utf-8")) if training.exists() else {}
            training_rows.append({**{key: job.get(key) for key in ("job_id", "method", "scale", "seed", "kind")}, **payload, "job_status": result.get("status")})
            continue
        summary = directory / "summary.csv"
        if not summary.exists():
            continue
        frame = pd.read_csv(summary)
        for key in ("job_id", "method", "scale", "seed", "kind", "test_id", "decode_budget"):
            frame[key] = job.get(key)
        instance_frames.append(frame)
    instances = pd.concat(instance_frames, ignore_index=True) if instance_frames else pd.DataFrame()
    return instances, pd.DataFrame(training_rows)


def aggregate(instances: pd.DataFrame) -> pd.DataFrame:
    if instances.empty:
        return pd.DataFrame()
    instances = instances.copy()
    instances["verifier_passed"] = instances["verifier_passed"].astype(bool)
    keys = ["method", "scale", "seed", "test_id", "decode_budget"]
    rows = []
    for values, frame in instances.groupby(keys, dropna=False):
        feasible = frame[frame["verifier_passed"]]
        rows.append(
            {
                **dict(zip(keys, values)),
                "instances": len(frame),
                "complete_and_feasible": len(feasible),
                "complete_and_feasible_rate": len(feasible) / max(len(frame), 1),
                "mean_verified_distance_km_conditional": feasible["objective_distance_km"].mean() if len(feasible) else None,
                "mean_vehicle_count_conditional": feasible["vehicle_count"].mean() if len(feasible) else None,
                "mean_inference_wall_time_s": frame["runtime_s"].mean(),
            }
        )
    return pd.DataFrame(rows)


def degradation(aggregates: pd.DataFrame) -> pd.DataFrame:
    if aggregates.empty:
        return pd.DataFrame()
    index = ["method", "scale", "seed", "decode_budget"]
    pivot = aggregates.pivot_table(index=index, columns="test_id", values=["complete_and_feasible_rate", "mean_verified_distance_km_conditional"], aggfunc="first")
    rows = []
    for keys, row in pivot.iterrows():
        out = dict(zip(index, keys))
        for target in ("T2", "T3"):
            for metric in ("complete_and_feasible_rate", "mean_verified_distance_km_conditional"):
                try:
                    out[f"{target}_minus_T1__{metric}"] = row[(metric, target)] - row[(metric, "T1")]
                except KeyError:
                    out[f"{target}_minus_T1__{metric}"] = None
        rows.append(out)
    return pd.DataFrame(rows)


def transfer_change(aggregates: pd.DataFrame) -> pd.DataFrame:
    if aggregates.empty:
        return pd.DataFrame()
    selected = aggregates[
        aggregates["test_id"].isin(["paired_Cus1000", "zero_shot_Cus2000"])
    ]
    if selected.empty:
        return pd.DataFrame()
    metrics = [
        "complete_and_feasible_rate",
        "mean_verified_distance_km_conditional",
        "mean_vehicle_count_conditional",
        "mean_inference_wall_time_s",
    ]
    pivot = selected.pivot_table(
        index=["method", "seed", "decode_budget"],
        columns="test_id",
        values=metrics,
        aggfunc="first",
    )
    rows = []
    for keys, row in pivot.iterrows():
        out = dict(zip(["method", "seed", "decode_budget"], keys))
        for metric in metrics:
            try:
                out[f"Cus2000_minus_paired_Cus1000__{metric}"] = (
                    row[(metric, "zero_shot_Cus2000")]
                    - row[(metric, "paired_Cus1000")]
                )
            except KeyError:
                out[f"Cus2000_minus_paired_Cus1000__{metric}"] = None
        rows.append(out)
    return pd.DataFrame(rows)




def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize frozen DRL experiments.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    instances, training = collect(args.output_root)
    aggregates = aggregate(instances)
    shifts = degradation(aggregates)
    transfer = transfer_change(aggregates)
    instances.to_csv(args.destination / "drl_per_instance.csv", index=False)
    training.to_csv(args.destination / "drl_training_runs.csv", index=False)
    aggregates.to_csv(args.destination / "drl_aggregate.csv", index=False)
    shifts.to_csv(args.destination / "drl_distribution_shift_degradation.csv", index=False)
    transfer.to_csv(args.destination / "drl_scale_transfer_change.csv", index=False)
    machine = {
        "schema": "drl_experiment_summary_v1",
        "per_instance_rows": len(instances),
        "training_runs": len(training),
        "aggregate_cells": len(aggregates),
        "feasibility_denominators_retained": True,
    }
    (args.destination / "drl_summary.json").write_text(json.dumps(machine, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    table = aggregates.to_markdown(index=False) if not aggregates.empty else "No completed evaluation cells."
    (args.destination / "drl_paper_table.md").write_text("# DRL benchmark results\n\n" + table + "\n", encoding="utf-8")
    print(json.dumps(machine, sort_keys=True))


if __name__ == "__main__":
    main()
