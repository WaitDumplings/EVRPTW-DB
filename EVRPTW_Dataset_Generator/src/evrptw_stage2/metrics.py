"""Phase-1 spatial-activation metrics and corpus aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def _normalized_w1(left: np.ndarray, right: np.ndarray, normalizer: float) -> float:
    if not len(left) or not len(right):
        return float("nan")
    return float(wasserstein_distance(left, right) / max(float(normalizer), 1.0))


def _distribution_summary(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0, "mean": None, "p50": None, "p90": None}
    return {
        "count": len(finite),
        "mean": float(finite.mean()),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
    }


def _frame_numeric_mean(frame: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _nearest_neighbor_times(customer_time: np.ndarray) -> np.ndarray:
    values = np.asarray(customer_time, dtype=float).copy()
    np.fill_diagonal(values, np.inf)
    # Symmetrization makes the generated diagnostic comparable to Amazon's
    # bidirectional stop-pair average used in preprocessing.
    symmetric = (values + values.T) / 2.0
    return np.min(symmetric, axis=1)


def _within_region_pairwise(
    customer_time: np.ndarray,
    regions: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for region in sorted(set(map(str, regions))):
        indices = np.flatnonzero(np.asarray(regions, dtype=str) == region)
        if len(indices) < 2:
            records.append(
                {
                    "sampling_cluster_id": region,
                    "customer_count": len(indices),
                    "pair_count": 0,
                    "pairwise_time_p50_s": np.nan,
                    "pairwise_time_p90_s": np.nan,
                }
            )
            continue
        sub = customer_time[np.ix_(indices, indices)]
        symmetric = (sub + sub.T) / 2.0
        values = symmetric[np.triu_indices(len(indices), k=1)]
        records.append(
            {
                "sampling_cluster_id": region,
                "customer_count": len(indices),
                "pair_count": len(values),
                "pairwise_time_p50_s": float(np.quantile(values, 0.50)),
                "pairwise_time_p90_s": float(np.quantile(values, 0.90)),
            }
        )
    return pd.DataFrame.from_records(records)


def _community_concentration(community_ids: np.ndarray) -> dict[str, Any]:
    counts = pd.Series(community_ids, dtype=str).value_counts()
    shares = counts.to_numpy(dtype=float) / max(float(counts.sum()), 1.0)
    return {
        "active_community_count": len(counts),
        "active_community_share_of_customers": float(len(counts) / max(len(community_ids), 1)),
        "community_hhi": float(np.square(shares).sum()),
        "largest_community_customer_share": float(shares.max()) if len(shares) else 0.0,
    }


def build_phase1_family_metrics(
    *,
    family_manifest_fields: dict[str, Any],
    terminal_index: pd.DataFrame,
    running_time_matrix_s: np.ndarray,
    radial_baseline: pd.DataFrame,
    selection_report: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Compute metrics for one attempted-and-successful parent family."""

    customer_count = int(selection_report["parent_customer_count"])
    customer_rows = terminal_index.iloc[1 : 1 + customer_count].copy()
    customer_time = np.asarray(
        running_time_matrix_s[1 : 1 + customer_count, 1 : 1 + customer_count],
        dtype=float,
    )
    proposal_radial = pd.to_numeric(
        customer_rows["depot_running_time_s"], errors="coerce"
    ).to_numpy(dtype=float)
    target_radial = np.asarray(
        selection_report["spatial_activation"]["target_radial_time_s"], dtype=float
    )
    baseline_radial = pd.to_numeric(
        radial_baseline["depot_running_time_s"], errors="coerce"
    ).to_numpy(dtype=float)
    t_env = float(selection_report["territory"]["amazon_t_env_s"])
    generated_nn = _nearest_neighbor_times(customer_time)
    amazon_reference = selection_report.get("amazon_spatial_reference", {})
    amazon_nn = np.asarray(amazon_reference.get("nearest_neighbor_time_s", []), dtype=float)
    region_pair = _within_region_pairwise(
        customer_time,
        customer_rows["sampling_cluster_id"].astype(str).to_numpy(),
    )
    proposal_concentration = _community_concentration(
        customer_rows["community_id"].astype(str).to_numpy()
    )
    baseline_concentration = _community_concentration(
        radial_baseline["community_id"].astype(str).to_numpy()
    )
    spatial = selection_report["spatial_activation"]
    quota = spatial["quota"]
    hard_gates = {
        "exact_parent_customer_count": len(customer_rows) == customer_count,
        "unique_parent_customer_ids": not customer_rows["source_id"].astype(str).duplicated().any(),
        "split_pool_recorded": selection_report["customer_pool"]
        in {"train", "heldout", "all_release_eligible"},
        "route_decile_row_margins_exact": bool(quota["row_margins_exact"]),
        "route_decile_column_margins_exact": bool(quota["column_margins_exact"]),
        "global_customer_uniqueness": bool(spatial["global_customer_uniqueness"]),
        "view_union_exact": bool(spatial["view_tree"]["union_exact"]),
        "view_pairwise_disjoint": bool(spatial["view_tree"]["pairwise_disjoint"]),
        "view_child_sizes_exact": bool(spatial["view_tree"]["child_sizes_exact"]),
    }
    hard_gates["passed"] = all(hard_gates.values())
    metrics = {
        "schema": "evrptw_phase1_family_metrics_v1",
        "statistical_unit": "attempted_parent_family",
        **family_manifest_fields,
        "hard_gates": hard_gates,
        "m1_radial": {
            "normalizer": "amazon_t_env_s",
            "proposal_family_normalized_w1": _normalized_w1(
                proposal_radial, target_radial, t_env
            ),
            "radial_baseline_family_normalized_w1": _normalized_w1(
                baseline_radial, target_radial, t_env
            ),
            "proposal_not_worse_than_radial_baseline": _normalized_w1(
                proposal_radial, target_radial, t_env
            )
            <= _normalized_w1(baseline_radial, target_radial, t_env) + 1e-12,
            "proposal_distribution": _distribution_summary(proposal_radial),
            "target_distribution": _distribution_summary(target_radial),
            "radial_baseline_distribution": _distribution_summary(baseline_radial),
        },
        "m2_network_nearest_neighbor": {
            "generated": _distribution_summary(generated_nn),
            "amazon_reference": _distribution_summary(amazon_nn),
            "normalized_w1_to_amazon": _normalized_w1(generated_nn, amazon_nn, t_env),
            "gate": False,
        },
        "m3_within_region_pairwise": {
            "generated_region_p50_distribution": _distribution_summary(
                region_pair["pairwise_time_p50_s"].to_numpy(dtype=float)
            ),
            "generated_region_p90_distribution": _distribution_summary(
                region_pair["pairwise_time_p90_s"].to_numpy(dtype=float)
            ),
            "amazon_route_p50_distribution": _distribution_summary(
                np.asarray(
                    amazon_reference.get("within_route_pairwise_time_p50_s", []),
                    dtype=float,
                )
            ),
            "amazon_route_p90_distribution": _distribution_summary(
                np.asarray(
                    amazon_reference.get("within_route_pairwise_time_p90_s", []),
                    dtype=float,
                )
            ),
            "validation_only": True,
            "gate": False,
        },
        "m4_region_structure": {
            "region_count": int(spatial["region_count"]),
            "region_size_distribution": _distribution_summary(
                np.asarray(list(spatial["region_sizes"].values()), dtype=float)
            ),
            "routes_dropped_by_rounding_count": len(
                spatial["quota"]["routes_dropped_by_rounding"]
            ),
            "by_construction_audit": True,
        },
        "m5_community_concentration": {
            "proposal": proposal_concentration,
            "radial_baseline": baseline_concentration,
            "proposal_minus_baseline_community_count": (
                proposal_concentration["active_community_count"]
                - baseline_concentration["active_community_count"]
            ),
            "proposal_minus_baseline_hhi": (
                proposal_concentration["community_hhi"]
                - baseline_concentration["community_hhi"]
            ),
            "diagnostic_only": True,
            "amazon_comparator_available": False,
        },
        "reliability": {
            "territory_reserve_ratio": float(
                selection_report["territory"]["territory_reserve_ratio"]
            ),
            "energy_screen_removed_share": float(
                selection_report["territory"]["energy_screen_removed_share"]
            ),
            "seed_fallback_count": int(spatial["seed_fallback_count"]),
            "seed_fallback_rate_per_region": float(
                spatial["seed_fallback_count"] / max(spatial["region_count"], 1)
            ),
            "region_attempts_used": int(spatial["region_attempts_used"]),
            "region_redraw_count": int(spatial["region_redraw_count"]),
            "community_growth_steps": int(spatial["community_growth_steps"]),
            "assignment_competition_expansions": int(
                spatial["assignment_competition_expansions"]
            ),
        },
        "selection_bias": {
            "proposal_vs_radial_baseline_reported": True,
            "region_redraw_conditioning_visible": True,
        },
    }
    observations = pd.DataFrame(
        {
            "family_id": family_manifest_fields["family_id"],
            "city_slug": family_manifest_fields["city_slug"],
            "day_type": family_manifest_fields["day_type"],
            "parent_scale_id": family_manifest_fields["parent_scale_id"],
            "amazon_t_env_s": t_env,
            "proposal_depot_time_s": proposal_radial,
            "structure_target_time_s": target_radial,
            "radial_baseline_depot_time_s": baseline_radial,
            "sampling_cluster_id": customer_rows["sampling_cluster_id"].astype(str).to_numpy(),
            "community_id": customer_rows["community_id"].astype(str).to_numpy(),
            "activation_decile": customer_rows["activation_decile"].astype(int).to_numpy(),
        }
    )
    return metrics, observations, region_pair


def aggregate_phase1_metrics(output_root: str | Path) -> dict[str, Any]:
    """Aggregate all completed family metrics and all attempted-family failures."""

    root = Path(output_root)
    report_root = root / "reports" / "phase1"
    report_root.mkdir(parents=True, exist_ok=True)
    metric_paths = sorted((root / "materialized" / "families").glob("*/phase1_metrics.json"))
    if not metric_paths:
        raise FileNotFoundError("No completed family Phase-1 metrics were found")
    metrics = [json.loads(path.read_text(encoding="utf-8")) for path in metric_paths]
    flat = pd.json_normalize(metrics, sep=".")
    flat.to_parquet(report_root / "family_metrics.parquet", index=False)
    observations = pd.concat(
        [pd.read_parquet(path.parent / "phase1_observations.parquet") for path in metric_paths],
        ignore_index=True,
    )
    rejection_paths = sorted((root / "rejections").glob("*.json"))
    rejection_records = [
        attempt
        for path in rejection_paths
        for attempt in json.loads(path.read_text(encoding="utf-8")).get("attempts", [])
    ]
    rejections = pd.DataFrame.from_records(rejection_records)
    if not rejections.empty:
        rejections.to_parquet(report_root / "rejected_attempts.parquet", index=False)
    strata: list[dict[str, Any]] = []
    for keys, group in observations.groupby(
        ["city_slug", "day_type", "parent_scale_id"], sort=True
    ):
        family_group = flat.loc[
            flat["city_slug"].astype(str).eq(str(keys[0]))
            & flat["day_type"].astype(str).eq(str(keys[1]))
            & flat["parent_scale_id"].astype(str).eq(str(keys[2]))
        ]

        proposal = group["proposal_depot_time_s"].to_numpy(dtype=float)
        target = group["structure_target_time_s"].to_numpy(dtype=float)
        baseline = group["radial_baseline_depot_time_s"].to_numpy(dtype=float)
        normalizer_values = group["amazon_t_env_s"].to_numpy(dtype=float)
        if not np.allclose(normalizer_values, normalizer_values[0]):
            raise ValueError(f"Amazon T_env is inconsistent within stratum {keys}")
        normalizer = float(normalizer_values[0])
        strata.append(
            {
                "city_slug": keys[0],
                "day_type": keys[1],
                "parent_scale_id": keys[2],
                "family_count": int(group["family_id"].nunique()),
                "customer_observation_count": len(group),
                "m1_corpus_proposal_normalized_w1": _normalized_w1(
                    proposal, target, normalizer
                ),
                "m1_corpus_radial_baseline_normalized_w1": _normalized_w1(
                    baseline, target, normalizer
                ),
                "m1_family_proposal_normalized_w1_mean": _frame_numeric_mean(
                    family_group,
                    "m1_radial.proposal_family_normalized_w1"
                ),
                "m1_family_radial_baseline_normalized_w1_mean": _frame_numeric_mean(
                    family_group,
                    "m1_radial.radial_baseline_family_normalized_w1"
                ),
                "m1_proposal_not_worse_than_baseline_share": _frame_numeric_mean(
                    family_group,
                    "m1_radial.proposal_not_worse_than_radial_baseline"
                ),
                "m2_family_normalized_w1_to_amazon_mean": _frame_numeric_mean(
                    family_group,
                    "m2_network_nearest_neighbor.normalized_w1_to_amazon"
                ),
                "m2_generated_nearest_neighbor_p50_s_mean": _frame_numeric_mean(
                    family_group,
                    "m2_network_nearest_neighbor.generated.p50"
                ),
                "m2_amazon_nearest_neighbor_p50_s_mean": _frame_numeric_mean(
                    family_group,
                    "m2_network_nearest_neighbor.amazon_reference.p50"
                ),
                "m3_generated_region_pair_p50_s_mean": _frame_numeric_mean(
                    family_group,
                    "m3_within_region_pairwise.generated_region_p50_distribution.mean"
                ),
                "m3_amazon_route_pair_p50_s_mean": _frame_numeric_mean(
                    family_group,
                    "m3_within_region_pairwise.amazon_route_p50_distribution.mean"
                ),
                "m4_region_count_mean": _frame_numeric_mean(
                    family_group, "m4_region_structure.region_count"
                ),
                "m5_proposal_minus_baseline_community_count_mean": _frame_numeric_mean(
                    family_group,
                    "m5_community_concentration.proposal_minus_baseline_community_count"
                ),
                "m5_proposal_minus_baseline_hhi_mean": _frame_numeric_mean(
                    family_group,
                    "m5_community_concentration.proposal_minus_baseline_hhi"
                ),
                "reliability_territory_reserve_ratio_mean": _frame_numeric_mean(
                    family_group,
                    "reliability.territory_reserve_ratio"
                ),
                "reliability_seed_fallback_rate_mean": _frame_numeric_mean(
                    family_group,
                    "reliability.seed_fallback_rate_per_region"
                ),
                "reliability_region_redraw_count_mean": _frame_numeric_mean(
                    family_group,
                    "reliability.region_redraw_count"
                ),
                "reliability_assignment_competition_expansions_mean": _frame_numeric_mean(
                    family_group,
                    "reliability.assignment_competition_expansions"
                ),
            }
        )
    strata_frame = pd.DataFrame.from_records(strata)
    strata_frame.to_csv(report_root / "stratified_metrics.csv", index=False)
    successful_attempts = len(metrics)
    rejected_attempts = len(rejection_records)
    attempted_attempts = successful_attempts + rejected_attempts
    rejected_family_ids = {
        str(record.get("family_id")) for record in rejection_records
    }
    successful_family_ids = set(flat["family_id"].astype(str))
    attempted_family_slots = len(rejected_family_ids | successful_family_ids)
    first_attempt_successes = int(
        flat["materialization_attempt_number"].astype(int).eq(0).sum()
    )
    rejection_reason_counts = (
        rejections["error_type"].astype(str).value_counts().sort_index().to_dict()
        if not rejections.empty
        else {}
    )
    summary = {
        "schema": "evrptw_phase1_corpus_metrics_v1",
        "statistical_unit": "attempted_parent_family",
        "successful_parent_family_count": successful_attempts,
        "rejected_parent_family_attempt_count": rejected_attempts,
        "attempted_parent_family_slot_count": attempted_family_slots,
        "attempted_parent_family_attempt_count": attempted_attempts,
        "first_attempt_success_count": first_attempt_successes,
        "raw_first_attempt_success_rate": (
            first_attempt_successes / attempted_family_slots
            if attempted_family_slots
            else None
        ),
        "conditional_attempt_success_rate": (
            successful_attempts / attempted_attempts if attempted_attempts else None
        ),
        "rejection_error_type_counts": {
            str(key): int(value) for key, value in rejection_reason_counts.items()
        },
        "region_attempts_used_distribution": _distribution_summary(
            flat["reliability.region_attempts_used"].to_numpy(dtype=float)
        ),
        "seed_fallback_rate_per_region_distribution": _distribution_summary(
            flat["reliability.seed_fallback_rate_per_region"].to_numpy(dtype=float)
        ),
        "all_hard_gates_passed": bool(flat["hard_gates.passed"].all()),
        "family_m1_proposal_not_worse_share": float(
            flat["m1_radial.proposal_not_worse_than_radial_baseline"].mean()
        ),
        "outputs": {
            "family_metrics": "family_metrics.parquet",
            "stratified_metrics": "stratified_metrics.csv",
            "rejected_attempts": (
                "rejected_attempts.parquet" if not rejections.empty else None
            ),
        },
        "gating_policy": {
            "hard_correctness_gates": True,
            "m1_comparative_gate_candidate": True,
            "m2_m3_m5_report_only": True,
            "numeric_thresholds_frozen": False,
        },
    }
    (report_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
