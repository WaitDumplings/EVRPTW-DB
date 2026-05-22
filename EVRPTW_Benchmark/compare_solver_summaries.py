from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_summary(path: Path, solver_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"instance_id", "feasible", "objective_distance_km", "vehicle_count", "runtime_s"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    out = df[["instance_id", "feasible", "objective_distance_km", "vehicle_count", "runtime_s"]].copy()
    out = out.rename(columns={
        "feasible": f"{solver_name}_feasible",
        "objective_distance_km": f"{solver_name}_objective_distance_km",
        "vehicle_count": f"{solver_name}_vehicle_count",
        "runtime_s": f"{solver_name}_runtime_s",
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two EVRPTW benchmark summary CSV files by instance_id.")
    parser.add_argument("--reference_summary", required=True, help="Reference solver summary, e.g. Gurobi.")
    parser.add_argument("--candidate_summary", required=True, help="Candidate solver summary, e.g. ALNS.")
    parser.add_argument("--reference_name", default="reference")
    parser.add_argument("--candidate_name", default="candidate")
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()

    ref = load_summary(Path(args.reference_summary), args.reference_name)
    cand = load_summary(Path(args.candidate_summary), args.candidate_name)
    merged = ref.merge(cand, on="instance_id", how="outer")

    ref_obj = pd.to_numeric(merged[f"{args.reference_name}_objective_distance_km"], errors="coerce")
    cand_obj = pd.to_numeric(merged[f"{args.candidate_name}_objective_distance_km"], errors="coerce")
    merged["objective_abs_gap_candidate_minus_reference"] = cand_obj - ref_obj
    merged["objective_rel_gap_candidate_vs_reference"] = np.where(
        ref_obj.abs() > 1e-12,
        (cand_obj - ref_obj) / ref_obj.abs(),
        np.nan,
    )

    ref_vehicle = pd.to_numeric(merged[f"{args.reference_name}_vehicle_count"], errors="coerce")
    cand_vehicle = pd.to_numeric(merged[f"{args.candidate_name}_vehicle_count"], errors="coerce")
    merged["vehicle_gap_candidate_minus_reference"] = cand_vehicle - ref_vehicle

    out = Path(args.save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"Saved comparison: {out}")
    feasible_col = f"{args.candidate_name}_feasible"
    if feasible_col in merged:
        print(f"candidate_feasible_count={int(merged[feasible_col].fillna(False).astype(bool).sum())}/{len(merged)}")
    gap = merged["objective_rel_gap_candidate_vs_reference"].dropna()
    if not gap.empty:
        print(f"relative_gap_mean={float(gap.mean()):.6f}")
        print(f"relative_gap_max={float(gap.max()):.6f}")


if __name__ == "__main__":
    main()
