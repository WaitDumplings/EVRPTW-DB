"""Pair canonical validation rows without changing training/evaluation artifacts.

Example: python -m EVRPTW_Benchmark.Reinforcement_Learning.scripts.audit_rrnco_comparison \
    --run rrnco=/path/to/validation_summary.json --run am=/path/to/validation_summary.json

The first run is the candidate. Reports explicitly separate a common-feasible
subset from a full-cohort result and expose missing evaluation/training metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


EVALUATION_FIELDS = ("objective_config", "candidate_count", "validation_rollout_steps",
                     "validation_seed", "decode_type", "scale", "split")
TRAINING_FIELDS = ("seed", "training_rollout_steps", "effective_batch_size",
                   "training_trajectory_count", "customer_exposures", "optimizer_steps",
                   "reward_contract_sha256", "training_stream_contract_sha256")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _pick(sources: list[dict], *keys: str) -> Any:
    return next((source[key] for source in sources for key in keys
                 if source.get(key) is not None), None)


def load_run(name: str, path: Path) -> dict:
    path = path / "validation_summary.json" if path.is_dir() else path
    summary = _read(path)
    if not summary:
        raise ValueError(f"missing or empty validation summary: {path}")
    training = _read(path.parent / "training_result.json")
    provenance = _read(path.parent / "provenance.json")
    signature = summary.get("resolved_training_signature") or training.get("resolved_training_signature") or {}
    sources = [summary, signature, training, provenance, provenance.get("job") or {}]
    metadata = {field: _pick(sources, field) for field in EVALUATION_FIELDS + TRAINING_FIELDS}
    metadata["candidate_count"] = _pick(sources, "candidate_count", "validation_candidates", "validation_candidate_count")
    metadata["decode_type"] = _pick(sources, "decode_type", "validation_decode_type")
    metadata["validation_rollout_steps"] = _pick(sources, "validation_rollout_steps", "max_steps")
    metadata["split"] = _pick(sources, "split", "split_ids")
    method_specific = signature.get("method_specific") or {}
    metadata["graph_mode"] = _pick([summary, method_specific, signature, training], "graph_mode", "relation_mode")
    metadata["architecture"] = method_specific.get("architecture")
    metadata["method_specific"] = method_specific or None
    metadata["selected_epoch"] = summary.get("logical_epoch")
    metadata["training_status"] = training.get("status", "unfinished_or_unrecorded")
    metadata["completed_training_epochs"] = training.get("completed_training_epochs")
    metadata["wall_time_s"] = training.get("wall_time_s")
    rows = {}
    for row in summary.get("rows", []):
        identifier = row.get("view_id") or row.get("instance_id")
        if not identifier:
            raise ValueError(f"row missing view_id/instance_id: {path}")
        if identifier in rows:
            raise ValueError(f"duplicate view identifier {identifier}: {path}")
        if row.get("view_id") and row.get("instance_id") and row["view_id"] != row["instance_id"]:
            raise ValueError(f"conflicting view_id and instance_id: {path}")
        feasible = row.get("verifier_passed") is True and row.get("environment_success", True) is True
        cost = row.get("objective_cost_usd", row.get("objective_value"))
        distance = row.get("objective_distance_km")
        vehicles = row.get("vehicle_count", row.get("vehicles_started"))
        if feasible:
            if cost is None or not math.isfinite(float(cost)):
                raise ValueError(f"feasible row has invalid cost {identifier}: {path}")
            config = metadata["objective_config"] or {}
            if config.get("mode") == "energy_vehicle_cost" and distance is not None and vehicles is not None:
                expected = (float(config["vehicle_fixed_cost_usd"]) * float(vehicles)
                            + float(config["electricity_price_usd_per_kwh"])
                            * float(config["consumption_kwh_per_km"]) * float(distance))
                if not math.isclose(float(cost), expected, rel_tol=1e-7, abs_tol=1e-5):
                    raise ValueError(f"row cost disagrees with objective config: {identifier}: {path}")
        rows[str(identifier)] = {"feasible": feasible, "cost": cost, "distance": distance, "vehicles": vehicles}
    ids = sorted(rows)
    return {"name": name, "path": str(path.resolve()), "metadata": metadata,
            "instances": summary.get("instances"), "complete_and_feasible": summary.get("complete_and_feasible"),
            "mean_verified_cost_usd": summary.get("mean_verified_cost_usd", summary.get("mean_verified_objective")),
            "row_count": len(rows), "view_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest() if ids else None,
            "rows": rows}


def _field_audit(left: dict, right: dict, fields: tuple[str, ...]) -> dict:
    differences, missing = {}, []
    for key in fields:
        a, b = left["metadata"].get(key), right["metadata"].get(key)
        if a is None or b is None:
            missing.append(key)
        elif a != b:
            differences[key] = {left["name"]: a, right["name"]: b}
    return {"differences": differences, "missing": missing}


def compare(candidate: dict, baseline: dict) -> dict:
    left, right = candidate["rows"], baseline["rows"]
    shared = sorted(left.keys() & right.keys())
    evaluation = _field_audit(candidate, baseline, EVALUATION_FIELDS)
    training = _field_audit(candidate, baseline, TRAINING_FIELDS)
    full_rows = (bool(left) and bool(right)
                 and len(left) == candidate["instances"] and len(right) == baseline["instances"])
    same_cohort = bool(full_rows and left.keys() == right.keys())
    objectives_known_equal = (candidate["metadata"]["objective_config"] is not None
                              and candidate["metadata"]["objective_config"] == baseline["metadata"]["objective_config"])
    paired = [identifier for identifier in shared if left[identifier]["feasible"] and right[identifier]["feasible"]]
    output = {"candidate": candidate["name"], "baseline": baseline["name"],
              "evaluation_audit": evaluation, "training_audit": training,
              "same_complete_cohort": same_cohort, "shared_view_count": len(shared),
              "candidate_only_views": len(left.keys() - right.keys()),
              "baseline_only_views": len(right.keys() - left.keys()),
              "jointly_feasible_count": len(paired),
              "full_cohort_comparison_verified": bool(same_cohort and len(paired) == len(shared)
                                                       and not evaluation["differences"] and not evaluation["missing"]),
              "paired_cost": None, "same_vehicle_distance": None,
              "causal_graph_claim_supported": False}
    if paired and objectives_known_equal:
        a = fmean(float(left[i]["cost"]) for i in paired)
        b = fmean(float(right[i]["cost"]) for i in paired)
        deltas = [float(left[i]["cost"]) - float(right[i]["cost"]) for i in paired]
        output["paired_cost"] = {"scope": "full_cohort" if same_cohort and len(paired) == len(shared) else "jointly_feasible_subset",
                                 "candidate_mean": a, "baseline_mean": b, "candidate_minus_baseline": a - b,
                                 "candidate_reduction_percent": 100 * (b - a) / b if b else None,
                                 "wins": sum(d < -1e-7 for d in deltas), "ties": sum(abs(d) <= 1e-7 for d in deltas),
                                 "losses": sum(d > 1e-7 for d in deltas)}
        same_vehicle = [i for i in paired if left[i]["vehicles"] is not None and left[i]["vehicles"] == right[i]["vehicles"]
                        and left[i]["distance"] is not None and right[i]["distance"] is not None]
        if same_vehicle:
            output["same_vehicle_distance"] = {"count": len(same_vehicle),
                "candidate_mean_km": fmean(float(left[i]["distance"]) for i in same_vehicle),
                "baseline_mean_km": fmean(float(right[i]["distance"]) for i in same_vehicle)}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=SUMMARY_OR_DIRECTORY")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = []
    for specification in args.run:
        name, separator, path = specification.partition("=")
        if not separator or not name or not path:
            parser.error("--run requires NAME=SUMMARY_OR_DIRECTORY")
        runs.append(load_run(name, Path(path)))
    if len(runs) < 2 or len({run["name"] for run in runs}) != len(runs):
        parser.error("provide at least two runs with unique names")
    report = {"schema": "rrnco_paired_validation_audit_v1",
              "interpretation": "Validation comparisons are exploratory. Graph causality requires matched trained ablations and held-out evaluation.",
              "runs": [{key: value for key, value in run.items() if key != "rows"} for run in runs],
              "comparisons": [compare(runs[0], baseline) for baseline in runs[1:]]}
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
