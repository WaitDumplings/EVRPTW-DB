from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def paired_parent_family_bootstrap(
    results: pd.DataFrame,
    *,
    method_a: str,
    method_b: str,
    replicates: int = 2000,
    seed: int = 49081,
) -> dict[str, Any]:
    required = {"family_id", "instance_id", "method", "seed", "verifier_passed", "objective_distance_km"}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(f"result table is missing columns: {missing}")
    if int(replicates) <= 0:
        raise ValueError("bootstrap replicates must be positive")
    frame = results.loc[results["method"].isin([method_a, method_b])].copy()
    if frame.empty:
        raise ValueError("no rows for requested methods")
    frame["verifier_passed"] = frame["verifier_passed"].astype(bool)
    # Seeds are repeated model fits, not independent test instances. Aggregate
    # within instance/method before clustering at the parent-family level.
    per_instance = (
        frame.groupby(["family_id", "instance_id", "method"], sort=True)
        .agg(
            feasible_rate=("verifier_passed", "mean"),
            objective_distance_km=(
                "objective_distance_km",
                lambda values: float(np.nanmean(pd.to_numeric(values, errors="coerce"))),
            ),
            training_seed_count=("seed", "nunique"),
        )
        .reset_index()
    )
    pivot_feasible = per_instance.pivot(index=["family_id", "instance_id"], columns="method", values="feasible_rate")
    if method_a not in pivot_feasible or method_b not in pivot_feasible:
        raise ValueError("both methods must occur in the paired cohort")
    pivot_objective = per_instance.pivot(index=["family_id", "instance_id"], columns="method", values="objective_distance_km")
    paired_index = pivot_feasible[[method_a, method_b]].dropna().index
    if len(paired_index) == 0:
        raise ValueError("methods have no paired instances")
    feasible_diff = pivot_feasible.loc[paired_index, method_a] - pivot_feasible.loc[paired_index, method_b]
    common = paired_index[
        (pivot_feasible.loc[paired_index, method_a] == 1.0)
        & (pivot_feasible.loc[paired_index, method_b] == 1.0)
        & pivot_objective.loc[paired_index, method_a].notna()
        & pivot_objective.loc[paired_index, method_b].notna()
    ]
    objective_diff = (
        pivot_objective.loc[common, method_a] - pivot_objective.loc[common, method_b]
    )
    families = np.asarray(sorted(set(index[0] for index in paired_index)), dtype=object)
    rng = np.random.default_rng(int(seed))
    feasibility_samples = np.empty(int(replicates), dtype=float)
    objective_samples = np.full(int(replicates), np.nan, dtype=float)
    for replicate in range(int(replicates)):
        sampled = rng.choice(families, size=len(families), replace=True)
        feasibility_values = []
        objective_values = []
        for family in sampled:
            feasibility_values.extend(feasible_diff.xs(family, level="family_id").tolist())
            if family in set(index[0] for index in common):
                objective_values.extend(objective_diff.xs(family, level="family_id").tolist())
        feasibility_samples[replicate] = float(np.mean(feasibility_values))
        if objective_values:
            objective_samples[replicate] = float(np.mean(objective_values))
    finite_objective = objective_samples[np.isfinite(objective_samples)]
    return {
        "schema": "drl_parent_family_paired_bootstrap_v1",
        "method_a": method_a,
        "method_b": method_b,
        "cluster_unit": "parent_family",
        "training_seeds_treated_as_independent_instances": False,
        "paired_family_count": len(families),
        "paired_instance_count": len(paired_index),
        "common_feasible_instance_count": len(common),
        "replicates": int(replicates),
        "seed": int(seed),
        "feasible_rate_difference_a_minus_b": {
            "estimate": float(feasible_diff.mean()),
            "ci95": [float(np.quantile(feasibility_samples, 0.025)), float(np.quantile(feasibility_samples, 0.975))],
        },
        "common_feasible_distance_difference_km_a_minus_b": {
            "estimate": float(objective_diff.mean()) if len(objective_diff) else None,
            "ci95": (
                [float(np.quantile(finite_objective, 0.025)), float(np.quantile(finite_objective, 0.975))]
                if len(finite_objective)
                else None
            ),
        },
    }


__all__ = ["paired_parent_family_bootstrap"]
