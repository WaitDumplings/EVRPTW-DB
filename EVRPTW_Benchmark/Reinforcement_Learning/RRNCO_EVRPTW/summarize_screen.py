from __future__ import annotations

import argparse
import json
from pathlib import Path


def validation_key(row: dict) -> tuple[float, float]:
    value = row.get("mean_verified_objective")
    return (
        float(row["complete_and_feasible_rate"]),
        float("-inf") if value is None else -float(value),
    )


def summarize(output: Path) -> dict:
    training = json.loads((output / "training_result.json").read_text())
    validations = [
        json.loads(line)
        for line in (output / "validation_history.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not validations:
        raise ValueError(f"no validation records: {output}")
    best = max(validations, key=validation_key)
    return {
        "output": str(output.resolve()),
        "method": training["method"],
        "status": training["status"],
        "completed_training_epochs": training["completed_training_epochs"],
        "wall_time_s": training["wall_time_s"],
        "peak_gpu_memory_gib": training["peak_gpu_memory_bytes"] / 2**30,
        "best_epoch": best["logical_epoch"],
        "validation_instances": best["instances"],
        "validation_candidates": best["candidate_count"],
        "feasible_rate": best["complete_and_feasible_rate"],
        "mean_verified_objective": best["mean_verified_objective"],
        "mean_verified_distance_km": best["mean_verified_distance_km"],
        "mean_verified_vehicles": best["mean_verified_vehicle_count"],
        "curve": [
            {
                key: row.get(key)
                for key in (
                    "logical_epoch",
                    "complete_and_feasible_rate",
                    "mean_verified_objective",
                    "mean_verified_distance_km",
                    "mean_verified_vehicle_count",
                    "validation_wall_time_s",
                )
            }
            for row in validations
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outputs", type=Path, nargs="+")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    payload = {"schema": "rrnco_ev_cus100_screen_summary_v1", "runs": [
        summarize(path) for path in args.outputs
    ]}
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json is not None:
        args.json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
