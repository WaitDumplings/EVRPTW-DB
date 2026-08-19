"""Phase-1 spatial-activation metrics and corpus aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


PHASE1_FAMILY_METRICS_SCHEMA = "evrptw_phase1_family_metrics_v2"


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
    return np.min(values, axis=1)


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
        values = sub[~np.eye(len(indices), dtype=bool)]
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
    t_env = float(selection_report["territory"]["source_t_env_s"])
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
        "schema": PHASE1_FAMILY_METRICS_SCHEMA,
        "statistical_unit": "attempted_parent_family",
        **family_manifest_fields,
        "hard_gates": hard_gates,
        "m1_radial": {
            "normalizer": "source_t_env_s",
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
            "source_t_env_s": t_env,
            "proposal_depot_time_s": proposal_radial,
            "structure_target_time_s": target_radial,
            "radial_baseline_depot_time_s": baseline_radial,
            "proposal_depot_time_normalized": proposal_radial / max(t_env, 1.0),
            "structure_target_time_normalized": target_radial / max(t_env, 1.0),
            "radial_baseline_time_normalized": baseline_radial / max(t_env, 1.0),
            "sampling_cluster_id": customer_rows["sampling_cluster_id"].astype(str).to_numpy(),
            "community_id": customer_rows["community_id"].astype(str).to_numpy(),
            "activation_decile": customer_rows["activation_decile"].astype(int).to_numpy(),
        }
    )
    return metrics, observations, region_pair


def _write_source_audit_frames(
    report_root: Path,
    *,
    source_rows: list[dict[str, Any]],
    template_rows: list[dict[str, Any]],
    bias_rows: list[dict[str, Any]],
    fragmentation_rows: list[dict[str, Any]],
    charger_rows: list[dict[str, Any]],
) -> dict[str, str]:
    source_frame = pd.DataFrame.from_records(source_rows)
    template_frame = pd.DataFrame.from_records(template_rows)
    bias_frame = pd.DataFrame.from_records(bias_rows)
    fragmentation_frame = pd.json_normalize(fragmentation_rows, sep=".")
    charger_frame = pd.DataFrame.from_records(charger_rows)
    template_family_counts = template_frame.groupby("order_template_id")[
        "family_id"
    ].nunique()
    template_frame["corpus_family_reuse_count"] = template_frame[
        "order_template_id"
    ].map(template_family_counts)
    source_frame.to_parquet(report_root / "amazon_source_family_ledger.parquet", index=False)
    template_frame.to_parquet(report_root / "amazon_template_usage.parquet", index=False)
    bias_frame.to_parquet(report_root / "matching_bias_audit.parquet", index=False)
    fragmentation_frame.to_parquet(report_root / "fragmentation_audit.parquet", index=False)
    charger_frame.to_parquet(report_root / "charger_selection_audit.parquet", index=False)
    within_family_duplicate_count = int(
        template_frame.duplicated(["family_id", "order_template_id"]).sum()
    )
    source_usage_summary = {
        "schema": "evrptw_amazon_source_usage_summary_v1",
        "family_count": len(source_frame),
        "source_pool_counts": source_frame["source_pool"].value_counts().sort_index().to_dict(),
        "generation_track_counts": source_frame["generation_track"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "structure_source_mode_counts": source_frame["structure_source_mode"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "order_source_mode_counts": source_frame["order_source_mode"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "structure_order_relationship_counts": source_frame[
            "structure_order_source_relationship"
        ]
        .value_counts()
        .sort_index()
        .to_dict(),
        "order_template_assignment_count": len(template_frame),
        "unique_order_template_count": int(template_frame["order_template_id"].nunique()),
        "templates_reused_across_families_count": int((template_family_counts > 1).sum()),
        "within_family_template_duplicate_count": within_family_duplicate_count,
        "within_family_template_reuse_forbidden_and_absent": within_family_duplicate_count == 0,
    }
    (report_root / "source_usage_summary.json").write_text(
        json.dumps(source_usage_summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "amazon_source_family_ledger": "amazon_source_family_ledger.parquet",
        "amazon_template_usage": "amazon_template_usage.parquet",
        "source_usage_summary": "source_usage_summary.json",
        "matching_bias_audit": "matching_bias_audit.parquet",
        "fragmentation_audit": "fragmentation_audit.parquet",
        "charger_selection_audit": "charger_selection_audit.parquet",
    }


def aggregate_phase1_metrics(output_root: str | Path) -> dict[str, Any]:
    """Aggregate all completed family metrics and all attempted-family failures."""

    root = Path(output_root)
    report_root = root / "reports" / "phase1"
    report_root.mkdir(parents=True, exist_ok=True)
    metric_paths = sorted((root / "materialized" / "families").glob("*/phase1_metrics.json"))
    if not metric_paths:
        raise FileNotFoundError("No completed family Phase-1 metrics were found")
    metrics = [json.loads(path.read_text(encoding="utf-8")) for path in metric_paths]
    stale_metric_paths = [
        str(path)
        for path, payload in zip(metric_paths, metrics, strict=True)
        if payload.get("schema") != PHASE1_FAMILY_METRICS_SCHEMA
    ]
    if stale_metric_paths:
        raise ValueError(
            "Phase-1 family metric schema mismatch; expected "
            f"{PHASE1_FAMILY_METRICS_SCHEMA}: {stale_metric_paths}"
        )
    family_manifest_paths = [path.parent / "family_manifest.json" for path in metric_paths]
    family_manifests = (
        [json.loads(path.read_text(encoding="utf-8")) for path in family_manifest_paths]
        if all(path.is_file() for path in family_manifest_paths)
        else []
    )
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

        proposal = group["proposal_depot_time_normalized"].to_numpy(dtype=float)
        target = group["structure_target_time_normalized"].to_numpy(dtype=float)
        baseline = group["radial_baseline_time_normalized"].to_numpy(dtype=float)
        normalizer = 1.0
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
    corpus_rows: list[dict[str, Any]] = []
    for keys, group in observations.groupby(
        ["day_type", "parent_scale_id"], sort=True
    ):
        family_group = flat.loc[
            flat["day_type"].astype(str).eq(str(keys[0]))
            & flat["parent_scale_id"].astype(str).eq(str(keys[1]))
        ]
        corpus_rows.append(
            {
                "day_type": str(keys[0]),
                "scale_id": str(keys[1]),
                "family_count": int(group["family_id"].nunique()),
                "city_count": int(group["city_slug"].nunique()),
                "customer_observation_count": len(group),
                "m1_corpus_generated_to_assigned_structure_normalized_w1": (
                    _normalized_w1(
                        group["proposal_depot_time_normalized"].to_numpy(dtype=float),
                        group["structure_target_time_normalized"].to_numpy(dtype=float),
                        1.0,
                    )
                ),
                "m1_corpus_radial_baseline_to_assigned_structure_normalized_w1": (
                    _normalized_w1(
                        group["radial_baseline_time_normalized"].to_numpy(dtype=float),
                        group["structure_target_time_normalized"].to_numpy(dtype=float),
                        1.0,
                    )
                ),
                "m1_family_generated_normalized_w1_mean": _frame_numeric_mean(
                    family_group, "m1_radial.proposal_family_normalized_w1"
                ),
            }
        )
    pd.DataFrame.from_records(corpus_rows).to_csv(
        report_root / "corpus_metrics.csv", index=False
    )

    source_rows: list[dict[str, Any]] = []
    template_rows: list[dict[str, Any]] = []
    bias_rows: list[dict[str, Any]] = []
    fragmentation_rows: list[dict[str, Any]] = []
    charger_rows: list[dict[str, Any]] = []
    for manifest_path, manifest in zip(family_manifest_paths, family_manifests):
        family_dir = manifest_path.parent
        family_id = str(manifest["family_id"])
        structure = manifest["selection_report"]["amazon_structure_source"]
        order = manifest["order_source_report"]
        source_rows.append(
            {
                "family_id": family_id,
                "city_slug": str(manifest["city_slug"]),
                "day_type": str(manifest["day_type"]),
                "parent_scale_id": str(manifest["parent_scale_id"]),
                "source_pool": str(order["source_pool"]),
                "generation_track": str(order["generation_track"]),
                "structure_source_mode": str(structure["structure_source_mode"]),
                "structure_source_ids": "|".join(
                    sorted(map(str, structure["structure_source_ids"]))
                ),
                "order_source_mode": str(order["selected_order_source_mode"]),
                "order_source_ids": "|".join(
                    sorted(map(str, order["selected_station_day_ids"]))
                ),
                "structure_order_source_relationship": str(
                    order["structure_order_source_relationship"]
                ),
                "release_role": str(order["release_role"]),
            }
        )
        bias_rows.append(
            {
                "family_id": family_id,
                "city_slug": str(manifest["city_slug"]),
                "day_type": str(manifest["day_type"]),
                "parent_scale_id": str(manifest["parent_scale_id"]),
                **{
                    f"eligible_pool_{key}": value
                    for key, value in order["matching_bias_audit"]["eligible_pool"].items()
                },
                **{
                    f"matched_templates_{key}": value
                    for key, value in order["matching_bias_audit"]["matched_templates"].items()
                },
            }
        )
        terminal_index = pd.read_parquet(
            family_dir / manifest["terminal_index"],
            columns=[
                "terminal_kind",
                "order_template_id",
                "reference_charge_mode",
                "effective_charging_power_source",
            ],
        )
        assigned = terminal_index.loc[
            terminal_index["terminal_kind"].eq("customer"), "order_template_id"
        ].dropna()
        template_rows.extend(
            {
                "family_id": family_id,
                "city_slug": str(manifest["city_slug"]),
                "day_type": str(manifest["day_type"]),
                "parent_scale_id": str(manifest["parent_scale_id"]),
                "source_pool": str(order["source_pool"]),
                "order_template_id": str(template_id),
            }
            for template_id in assigned.astype(str)
        )
        parent_chargers = manifest["selection_report"]["charger_selection"]
        legacy_audit = parent_chargers["road_time_vs_haversine_legacy_audit"]
        selected_positions = list(map(int, parent_chargers["selected_roster_positions"]))
        parent_charger_terminals = terminal_index.loc[
            terminal_index["terminal_kind"].eq("charging_station")
        ]
        charger_rows.append(
            {
                "family_id": family_id,
                "view_id": None,
                "scale_id": str(manifest["parent_scale_id"]),
                "selection_level": "parent",
                "candidate_roster_count": int(parent_chargers["candidate_roster_count"]),
                "bidirectional_energy_eligible_count": int(
                    parent_chargers["bidirectional_energy_eligible_count"]
                ),
                "selected_count": len(selected_positions),
                "selected_roster_positions": "|".join(map(str, selected_positions)),
                "prefix_semantics": False,
                "is_literal_roster_prefix": selected_positions
                == list(range(len(selected_positions))),
                "road_time_delta_mean_s": float(
                    legacy_audit["road_time_delta"]["mean_s"]
                ),
                "road_time_delta_p95_s": float(
                    legacy_audit["road_time_delta"]["p95_s"]
                ),
                "haversine_legacy_delta_mean_s": float(
                    legacy_audit["haversine_legacy_road_time_delta"]["mean_s"]
                ),
                "haversine_legacy_delta_p95_s": float(
                    legacy_audit["haversine_legacy_road_time_delta"]["p95_s"]
                ),
                "road_time_haversine_selected_overlap_count": int(
                    legacy_audit["selected_roster_overlap_count"]
                ),
                "selected_dc_fast_count": int(
                    parent_charger_terminals["reference_charge_mode"].eq("dc_fast").sum()
                ),
                "selected_ac_level2_count": int(
                    parent_charger_terminals["reference_charge_mode"].eq("ac_level2").sum()
                ),
                "reported_power_count": int(
                    parent_charger_terminals["effective_charging_power_source"]
                    .astype(str)
                    .str.startswith("reported")
                    .sum()
                ),
                "national_mode_median_power_count": int(
                    parent_charger_terminals["effective_charging_power_source"]
                    .astype(str)
                    .str.startswith("national_mode")
                    .sum()
                ),
            }
        )
        for view_manifest_path in sorted((family_dir / "views").glob("*/view_manifest.json")):
            view = json.loads(view_manifest_path.read_text(encoding="utf-8"))
            m4 = view["spatial_metrics"]["m4_region_first_partition"]
            fragmentation_rows.append(
                {
                    "family_id": family_id,
                    "view_id": str(view["view_id"]),
                    "city_slug": str(view["city_slug"]),
                    "day_type": str(view["day_type"]),
                    "scale_id": str(view["scale_id"]),
                    **m4,
                }
            )
            terminal_indices = list(
                map(int, view["charger_selection"]["parent_terminal_indices"])
            )
            child_charger_terminals = terminal_index.iloc[terminal_indices]
            charger_rows.append(
                {
                    "family_id": family_id,
                    "view_id": str(view["view_id"]),
                    "scale_id": str(view["scale_id"]),
                    "selection_level": "child_view",
                    "candidate_roster_count": int(
                        manifest["parent_charging_station_count"]
                    ),
                    "bidirectional_energy_eligible_count": int(
                        manifest["parent_charging_station_count"]
                    ),
                    "selected_count": len(terminal_indices),
                    "selected_roster_positions": "|".join(map(str, terminal_indices)),
                    "prefix_semantics": bool(
                        view["charger_selection"]["prefix_semantics"]
                    ),
                    "is_literal_roster_prefix": terminal_indices
                    == list(
                        range(
                            1 + int(manifest["parent_customer_count"]),
                            1
                            + int(manifest["parent_customer_count"])
                            + len(terminal_indices),
                        )
                    ),
                    "road_time_delta_mean_s": None,
                    "road_time_delta_p95_s": None,
                    "haversine_legacy_delta_mean_s": None,
                    "haversine_legacy_delta_p95_s": None,
                    "road_time_haversine_selected_overlap_count": None,
                    "selected_dc_fast_count": int(
                        child_charger_terminals["reference_charge_mode"].eq("dc_fast").sum()
                    ),
                    "selected_ac_level2_count": int(
                        child_charger_terminals["reference_charge_mode"].eq("ac_level2").sum()
                    ),
                    "reported_power_count": int(
                        child_charger_terminals["effective_charging_power_source"]
                        .astype(str)
                        .str.startswith("reported")
                        .sum()
                    ),
                    "national_mode_median_power_count": int(
                        child_charger_terminals["effective_charging_power_source"]
                        .astype(str)
                        .str.startswith("national_mode")
                        .sum()
                    ),
                }
            )

    source_outputs = (
        _write_source_audit_frames(
            report_root,
            source_rows=source_rows,
            template_rows=template_rows,
            bias_rows=bias_rows,
            fragmentation_rows=fragmentation_rows,
            charger_rows=charger_rows,
        )
        if family_manifests
        else {}
    )
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
            "corpus_metrics": "corpus_metrics.csv",
            "rejected_attempts": (
                "rejected_attempts.parquet" if not rejections.empty else None
            ),
            **source_outputs,
        },
        "gating_policy": {
            "hard_correctness_gates": True,
            "m1_operational_transfer_gate": (
                "reports/stage2_repair/amazon_operational_transfer_acceptance_v2.json"
            ),
            "m2_m3_report_only": (
                "reports/stage2_repair/cross_city_spatial_diagnostic_v1.json"
            ),
            "m4_operational_structure_gate": True,
            "m5_report_only": True,
            "historical_q90_v1_fail_retained": True,
            "numeric_thresholds_frozen": True,
        },
    }
    (report_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
