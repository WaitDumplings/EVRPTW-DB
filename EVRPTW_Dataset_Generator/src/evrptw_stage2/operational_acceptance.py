"""D-5 v2 operational transfer acceptance and spatial diagnostics.

Amazon evidence controls operational templates.  CLE evidence controls target-
city geography.  M2/M3/M5 are deliberately excluded from every hard-gate
decision in this module.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .progress import atomic_write_json


OPERATIONAL_CONFIG_SCHEMA = "amazon_operational_transfer_acceptance_config_v2"
OPERATIONAL_REPORT_SCHEMA = "amazon_operational_transfer_acceptance_v2"
SPATIAL_DIAGNOSTIC_SCHEMA = "cross_city_spatial_diagnostic_v1"
Q90_V1_SCHEMA = "evrptw_station_block_q90_gate_v1"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _artifact_snapshot(root: Path) -> tuple[tuple[str, int, int], ...]:
    """Capture a no-hash, same-process mutation guard for family artifacts."""

    return tuple(
        (str(path.relative_to(root)), int(stat.st_size), int(stat.st_mtime_ns))
        for path in sorted(path for path in root.rglob("*") if path.is_file())
        for stat in [path.stat()]
    )


def preserve_q90_v1_failure(source: Path, destination: Path) -> None:
    """Preserve the original Q90 v1 bytes without computing a digest."""

    payload = source.read_bytes()
    parsed = json.loads(payload)
    if parsed.get("schema") != Q90_V1_SCHEMA:
        raise ValueError("The supplied historical report is not Q90 v1")
    if parsed.get("release_calibrated") is not False:
        raise ValueError("The historical Q90 v1 report is not the frozen FAIL evidence")
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError("Frozen Q90 v1 preservation target differs from source")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def build_spatial_diagnostic_v1(
    q90_v1: Mapping[str, Any],
    family_metrics: pd.DataFrame,
    *,
    original_report_path: Path,
    preserved_report_path: Path,
    code_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reclassify, but never delete or change, the historical M2/M3 result."""

    if q90_v1.get("schema") != Q90_V1_SCHEMA:
        raise ValueError("Spatial diagnostic requires the historical Q90 v1 report")
    rows = [dict(row) for row in q90_v1.get("rows", [])]
    ratios = [
        float(row["generated_to_holdout_q90"]) / float(row["real_to_real_q90"])
        for row in rows
        if row.get("generated_to_holdout_q90") is not None
        and row.get("real_to_real_q90") not in {None, 0}
    ]
    m5_columns = {
        "proposal_community_count": (
            "m5_community_concentration.proposal.active_community_count"
        ),
        "proposal_community_hhi": "m5_community_concentration.proposal.community_hhi",
        "uniform_community_count": (
            "m5_community_concentration.radial_baseline.active_community_count"
        ),
        "uniform_community_hhi": (
            "m5_community_concentration.radial_baseline.community_hhi"
        ),
    }
    m5_rows: list[dict[str, Any]] = []
    if {"city_slug", "day_type", *m5_columns.values()} <= set(family_metrics.columns):
        for keys, group in family_metrics.groupby(["city_slug", "day_type"], sort=True):
            m5_rows.append(
                {
                    "city_slug": str(keys[0]),
                    "day_type": str(keys[1]),
                    "family_count": int(len(group)),
                    **{
                        label: float(pd.to_numeric(group[column]).mean())
                        for label, column in m5_columns.items()
                    },
                }
            )
    passed_count = sum(bool(row.get("passed")) for row in rows)
    return {
        "schema": SPATIAL_DIAGNOSTIC_SCHEMA,
        "status": "complete_report_only",
        "hard_gate": False,
        "contributes_to_operational_acceptance": False,
        "construct_validity_review": {
            "triggered_by_q90_v1_failure": True,
            "decision": "M2_M3_reclassified_as_cross_domain_spatial_diagnostics",
            "threshold_changed": False,
            "historical_result_deleted": False,
            "rationale": (
                "M2 and M3 measure city-specific road-network morphology; anonymized "
                "Amazon stations do not identify a transferable target-city geometry."
            ),
        },
        "historical_q90_v1": {
            "original_path": str(original_report_path.resolve()),
            "preserved_path": str(preserved_report_path.resolve()),
            "release_calibrated": bool(q90_v1.get("release_calibrated")),
            "row_count": len(rows),
            "passed_row_count": passed_count,
            "failed_row_count": len(rows) - passed_count,
            "generated_to_real_ratio_min": min(ratios) if ratios else None,
            "generated_to_real_ratio_max": max(ratios) if ratios else None,
            "rows": rows,
        },
        "diagnostic_components": {
            "M2": "nearest_customer_directed_road_time",
            "M3": "within_region_directed_pairwise_road_time",
            "M5": "community_concentration_vs_same_count_baseline",
        },
        "m5_city_day_diagnostics": m5_rows,
        "paper_statement": (
            "M2/M3 characterize local road-network morphology and spatial "
            "concentration. Because these quantities are city-specific and the "
            "Amazon locations are anonymized, they are reported as cross-domain "
            "diagnostics rather than enforced as transfer constraints."
        ),
        "code_provenance": dict(code_provenance or {}),
        "hash_validation_performed": False,
    }


def evaluate_matching_bias(
    frame: pd.DataFrame,
    policy: Mapping[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    """Apply the frozen, interpretable matching-bias guardrails."""

    relative_limit = float(policy["maximum_relative_difference"])
    absolute_limit = float(policy["maximum_tw_presence_absolute_difference"])
    rows: list[dict[str, Any]] = []
    all_passed = True
    for record in frame.to_dict(orient="records"):
        details: dict[str, Any] = {
            "family_id": str(record["family_id"]),
            "relative_differences": {},
            "absolute_differences": {},
        }
        row_passed = True
        for field in policy["relative_fields"]:
            eligible = float(record[f"eligible_pool_{field}"])
            matched = float(record[f"matched_templates_{field}"])
            difference = abs(matched - eligible) / max(abs(eligible), 1e-12)
            details["relative_differences"][str(field)] = difference
            row_passed &= bool(np.isfinite(difference) and difference <= relative_limit)
        for field in policy["absolute_fields"]:
            eligible = float(record[f"eligible_pool_{field}"])
            matched = float(record[f"matched_templates_{field}"])
            difference = abs(matched - eligible)
            details["absolute_differences"][str(field)] = difference
            row_passed &= bool(np.isfinite(difference) and difference <= absolute_limit)
        counts_valid = (
            int(record["matched_templates_count"]) > 0
            and int(record["eligible_pool_count"]) >= int(record["matched_templates_count"])
        )
        details["counts_valid"] = counts_valid
        details["passed"] = bool(row_passed and counts_valid)
        all_passed &= details["passed"]
        rows.append(details)
    return bool(all_passed and rows), rows


def _arrays_equal_after_storage_cast(actual: np.ndarray, expected: np.ndarray) -> bool:
    return np.array_equal(actual, np.asarray(expected).astype(actual.dtype, copy=False))


def _audit_families_and_views(
    family_root: Path,
    templates: pd.DataFrame,
    cohort_split: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    template_index = templates.set_index("template_id", drop=False)
    assignment = {
        str(row["station_day_id"]): {
            "pool": str(row["pool"]),
            "generation_track": row.get("generation_track"),
            "day_type": str(row["day_type"]),
        }
        for row in cohort_split["station_day_assignments"]
    }
    family_paths = sorted(family_root.glob("*/family_manifest.json"))
    route_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    view_failures: list[dict[str, Any]] = []
    template_failure_count = 0
    audited_view_count = 0
    inherited_customer_count = 0
    primary_modes = config["order_template_transfer"]
    for manifest_path in family_paths:
        family_dir = manifest_path.parent
        manifest = _read_json(manifest_path)
        family_id = str(manifest["family_id"])
        customer_count = int(manifest["parent_customer_count"])
        spatial = manifest["selection_report"]["spatial_activation"]
        quota = spatial["quota"]
        source = manifest["selection_report"]["amazon_structure_source"]
        order = manifest["order_source_report"]
        region_sizes = list(map(int, spatial["region_sizes"].values()))
        route_passed = bool(
            int(source["structure_source_route_count"]) == int(quota["source_route_count"])
            == int(quota["retained_region_count"])
            == int(spatial["region_count"])
            and not quota["routes_dropped_by_rounding"]
            and bool(quota["row_margins_exact"])
            and bool(quota["column_margins_exact"])
            and sum(region_sizes) == customer_count
        )
        route_rows.append({"family_id": family_id, "passed": route_passed})

        source_days = {
            *map(str, source["structure_source_ids"]),
            *map(str, order["selected_station_day_ids"]),
        }
        source_contract = all(
            day in assignment
            and assignment[day]["pool"] == str(order["source_pool"])
            and assignment[day]["day_type"] == str(manifest["day_type"])
            and (
                assignment[day]["pool"] != "GEN-EVAL"
                or str(assignment[day]["generation_track"]) == str(order["generation_track"])
            )
            for day in source_days
        )
        mode_contract = bool(
            source["structure_source_mode"] == primary_modes["primary_structure_source_mode"]
            and order["selected_order_source_mode"] == primary_modes["primary_order_source_mode"]
            and order["release_role"] == "primary_candidate"
            and order["template_reuse"] is False
            and order["hall_coverage_complete"] is True
        )
        terminal_index = pd.read_parquet(family_dir / manifest["terminal_index"])
        customers = terminal_index.loc[terminal_index["terminal_kind"].eq("customer")]
        template_ids = customers["order_template_id"].astype(str).tolist()
        template_contract = bool(
            len(template_ids) == customer_count
            and len(set(template_ids)) == customer_count
            and set(template_ids) <= set(template_index.index.astype(str))
        )
        if template_contract:
            expected_days = template_index.loc[template_ids, "station_day_id"].astype(str)
            template_contract = bool(
                set(expected_days) <= set(map(str, order["selected_station_day_ids"]))
                and customers["order_station_day_id"].astype(str).tolist()
                == expected_days.tolist()
            )
        if not template_contract:
            template_failure_count += 1
        source_rows.append(
            {
                "family_id": family_id,
                "source_contract_passed": source_contract,
                "mode_contract_passed": mode_contract,
                "template_contract_passed": template_contract,
            }
        )

        for view_path in sorted((family_dir / "views").glob("*/view_manifest.json")):
            view = _read_json(view_path)
            view_dir = view_path.parent
            n = int(view["customer_count"])
            parent_indices = np.load(view_dir / view["terminal_parent_indices"])
            customer_parent_indices = parent_indices[1 : 1 + n]
            view_template_ids = terminal_index.iloc[customer_parent_indices][
                "order_template_id"
            ].astype(str).tolist()
            view_templates = template_index.loc[view_template_ids]
            with np.load(view_dir / view["customer_attributes"]) as attributes:
                checks = {
                    "package_count": _arrays_equal_after_storage_cast(
                        attributes["package_counts"],
                        view_templates["package_count"].to_numpy(),
                    ),
                    "demand_cm3": _arrays_equal_after_storage_cast(
                        attributes["demands_cm3"],
                        view_templates["demand_cm3"].to_numpy(),
                    ),
                    "service_time_s": _arrays_equal_after_storage_cast(
                        attributes["service_time_s"],
                        view_templates["service_time_s"].to_numpy(),
                    ),
                    "time_windows_s": _arrays_equal_after_storage_cast(
                        attributes["time_windows_s"],
                        view_templates[["tw_start_s", "tw_end_s"]].to_numpy(),
                    ),
                }
            view_passed = bool(
                all(checks.values())
                and view["attribute_report"]["order_template_inheritance"] is True
                and view["attribute_report"]["time_windows"][
                    "feasibility_clipping_applied"
                ]
                is False
            )
            if not view_passed:
                view_failures.append(
                    {
                        "family_id": family_id,
                        "view_id": str(view["view_id"]),
                        "checks": checks,
                    }
                )
            audited_view_count += 1
            inherited_customer_count += n
    return {
        "family_count": len(family_paths),
        "audited_view_count": audited_view_count,
        "inherited_customer_count": inherited_customer_count,
        "route_region_all_passed": bool(route_rows and all(row["passed"] for row in route_rows)),
        "source_provenance_all_passed": bool(
            source_rows
            and all(
                row["source_contract_passed"]
                and row["mode_contract_passed"]
                and row["template_contract_passed"]
                for row in source_rows
            )
        ),
        "template_failure_count": template_failure_count,
        "view_attribute_inheritance_all_passed": not view_failures,
        "view_failure_count": len(view_failures),
        "view_failures": view_failures,
        "route_rows": route_rows,
        "source_rows": source_rows,
    }


def build_operational_transfer_acceptance_v2(
    *,
    instance_root: Path,
    amazon_artifact_root: Path,
    cohort_split_path: Path,
    config_path: Path,
    code_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read-only acceptance over an already materialized 140-family pilot."""

    config = _read_json(config_path)
    if config.get("schema") != OPERATIONAL_CONFIG_SCHEMA:
        raise ValueError("Unsupported operational acceptance config")
    family_root = instance_root / "materialized" / "families"
    before = _artifact_snapshot(family_root)
    reports = instance_root / "reports"
    phase1 = _read_json(reports / "phase1" / "summary.json")
    run = _read_json(instance_root / "stage2_run_report.json")
    family_metrics = pd.read_parquet(reports / "phase1" / "family_metrics.parquet")
    corpus_metrics = pd.read_csv(reports / "phase1" / "corpus_metrics.csv")
    source_ledger = pd.read_parquet(
        reports / "phase1" / "amazon_source_family_ledger.parquet"
    )
    bias = pd.read_parquet(reports / "phase1" / "matching_bias_audit.parquet")
    templates = pd.read_parquet(amazon_artifact_root / "templates.parquet")
    cohort_split = _read_json(cohort_split_path)

    family_audit = _audit_families_and_views(
        family_root, templates, cohort_split, config
    )
    bias_passed, bias_rows = evaluate_matching_bias(bias, config["matching_bias"])
    family_m1 = pd.to_numeric(
        family_metrics["m1_radial.proposal_family_normalized_w1"], errors="coerce"
    )
    corpus_m1 = pd.to_numeric(
        corpus_metrics[
            "m1_corpus_generated_to_assigned_structure_normalized_w1"
        ],
        errors="coerce",
    )
    day_audit = (
        source_ledger.groupby(["city_slug", "generation_track", "day_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    weekday_required = int(config["required_counts"]["weekday_per_city_track"])
    weekend_required = int(config["required_counts"]["weekend_per_city_track"])
    day_ratio_passed = bool(
        len(day_audit) == int(config["required_counts"]["city_track_strata"])
        and day_audit.get("weekday", pd.Series(dtype=int)).eq(weekday_required).all()
        and day_audit.get("weekend", pd.Series(dtype=int)).eq(weekend_required).all()
    )
    operational_day_summary = []
    for day_type, group in bias.groupby("day_type", sort=True):
        operational_day_summary.append(
            {
                "day_type": str(day_type),
                "family_count": int(len(group)),
                **{
                    column: float(pd.to_numeric(group[column]).mean())
                    for column in [
                        "matched_templates_demand_cm3_mean",
                        "matched_templates_package_count_mean",
                        "matched_templates_service_time_s_mean",
                        "matched_templates_tw_presence_rate",
                        "matched_templates_tw_width_s_mean",
                    ]
                },
            }
        )

    successful = [
        item
        for item in run.get("materialized", [])
        if item.get("status") in {"materialized", "reused_verified"}
    ]
    verified = [item for item in run.get("verified", []) if item.get("passed") is True]
    required = config["required_counts"]
    checks = {
        "A_runner_passed": run.get("passed") is True,
        "A_planned_materialized_verified_140": (
            int(run.get("execution", {}).get("selected_family_count", -1))
            == int(required["parent_families"])
            == len(successful)
            == len(verified)
        ),
        "A_no_timeout_rejection_unresolved": bool(
            not run.get("unresolved_family_ids", [])
            and not run.get("timed_out_family_ids", [])
            and not run.get("aborted_family_ids", [])
            and not run.get("rejected_attempts", [])
        ),
        "A_phase1_correctness_passed": phase1.get("all_hard_gates_passed") is True,
        "A_exact_5_to_2_day_type": day_ratio_passed,
        "B_m1_family_radial_fidelity": bool(
            len(family_m1) == int(required["parent_families"])
            and family_m1.notna().all()
            and family_m1.le(
                float(config["m1_radial_fidelity"]["maximum_family_normalized_w1"])
                + 1e-12
            ).all()
        ),
        "B_m1_corpus_radial_fidelity": bool(
            len(corpus_m1) == 2
            and corpus_m1.notna().all()
            and corpus_m1.le(
                float(config["m1_radial_fidelity"]["maximum_corpus_normalized_w1"])
                + 1e-12
            ).all()
        ),
        "B_route_region_structure_exact": family_audit["route_region_all_passed"],
        "B_amazon_source_provenance_exact": family_audit[
            "source_provenance_all_passed"
        ],
        "B_order_attributes_inherited_exactly": (
            family_audit["view_attribute_inheritance_all_passed"]
            and family_audit["audited_view_count"] == int(required["materialized_views"])
        ),
        "B_order_template_matching_bias_within_frozen_limits": bias_passed,
        "B_weekday_weekend_operational_summaries_complete": (
            {row["day_type"] for row in operational_day_summary}
            == {"weekday", "weekend"}
        ),
    }
    after = _artifact_snapshot(family_root)
    family_artifacts_modified = before != after
    checks["A_family_artifacts_unchanged_during_audit"] = not family_artifacts_modified
    report = {
        "schema": OPERATIONAL_REPORT_SCHEMA,
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "scope": "existing_140_family_non_release_pilot_read_only",
        "checks": checks,
        "m1_radial": {
            "family_count": int(len(family_m1)),
            "family_maximum_normalized_w1": float(family_m1.max()),
            "family_p90_normalized_w1": float(family_m1.quantile(0.90)),
            "corpus_rows": corpus_metrics.to_dict(orient="records"),
            "frozen_policy": dict(config["m1_radial_fidelity"]),
        },
        "route_region_and_template_audit": family_audit,
        "matching_bias": {
            "passed": bias_passed,
            "frozen_policy": dict(config["matching_bias"]),
            "failed_family_count": sum(not row["passed"] for row in bias_rows),
            "rows": bias_rows,
        },
        "weekday_weekend": {
            "five_to_two_rows": day_audit.to_dict(orient="records"),
            "operational_distribution_summaries": operational_day_summary,
        },
        "spatial_metrics_excluded_from_hard_gates": ["M2", "M3", "M5"],
        "spatial_diagnostic_required_separately": SPATIAL_DIAGNOSTIC_SCHEMA,
        "family_artifacts_modified": family_artifacts_modified,
        "family_artifact_snapshot_method": "path_size_mtime_ns_same_process_no_hash",
        "family_artifact_file_count": len(before),
        "config_path": str(config_path.resolve()),
        "code_provenance": dict(code_provenance or {}),
        "hash_validation_performed": False,
    }
    return report


def write_d5_v2_reports(
    *,
    instance_root: Path,
    amazon_artifact_root: Path,
    cohort_split_path: Path,
    config_path: Path,
    q90_v1_path: Path,
    operational_output: Path,
    spatial_output: Path,
    preserved_q90_output: Path,
    code_provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write the two versioned reports while retaining Q90 v1 verbatim."""

    preserve_q90_v1_failure(q90_v1_path, preserved_q90_output)
    family_metrics = pd.read_parquet(
        instance_root / "reports" / "phase1" / "family_metrics.parquet"
    )
    spatial = build_spatial_diagnostic_v1(
        _read_json(q90_v1_path),
        family_metrics,
        original_report_path=q90_v1_path,
        preserved_report_path=preserved_q90_output,
        code_provenance=code_provenance,
    )
    operational = build_operational_transfer_acceptance_v2(
        instance_root=instance_root,
        amazon_artifact_root=amazon_artifact_root,
        cohort_split_path=cohort_split_path,
        config_path=config_path,
        code_provenance=code_provenance,
    )
    atomic_write_json(spatial_output, spatial)
    atomic_write_json(operational_output, operational)
    return operational, spatial
