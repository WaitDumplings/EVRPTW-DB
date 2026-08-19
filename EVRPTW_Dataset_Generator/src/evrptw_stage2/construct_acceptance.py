"""Construct-valid Stage-2 acceptance v3.

Correctness decides release eligibility.  Amazon marginal similarity and
target-city spatial morphology are retained as diagnostics and never alter
``passed`` in this module.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .progress import atomic_write_json


CONFIG_SCHEMA = "stage2_acceptance_v3_construct_valid_config"
REPORT_SCHEMA = "stage2_acceptance_v3_construct_valid"
DIAGNOSTIC_SCHEMA = "amazon_operational_diagnostics_v3"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_snapshot(root: Path) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (str(path.relative_to(root)), int(stat.st_size), int(stat.st_mtime_ns))
        for path in sorted(path for path in root.rglob("*") if path.is_file())
        for stat in [path.stat()]
    )


def _stored_equal(actual: np.ndarray, expected: np.ndarray) -> bool:
    return np.array_equal(actual, np.asarray(expected).astype(actual.dtype, copy=False))


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"p25": None, "p50": None, "p75": None, "p90": None}
    q = np.quantile(values, [0.25, 0.50, 0.75, 0.90])
    return {"p25": float(q[0]), "p50": float(q[1]), "p75": float(q[2]), "p90": float(q[3])}


def _w1(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if not len(left) or not len(right):
        return None
    return float(wasserstein_distance(left, right))


def _pmf(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=int)
    if not len(values):
        return {}
    keys, counts = np.unique(values, return_counts=True)
    return {str(int(key)): float(count / len(values)) for key, count in zip(keys, counts)}


def _ecdf_quantiles(values: np.ndarray) -> list[dict[str, float]]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return []
    probability = np.linspace(0.0, 1.0, 21)
    quantile = np.quantile(values, probability)
    return [
        {"probability": float(p), "value_s": float(q)}
        for p, q in zip(probability, quantile)
    ]


def _diagnostic_row(
    key: tuple[str, str, str],
    generated: Mapping[str, list[np.ndarray]],
    eligible: Mapping[str, list[np.ndarray]],
) -> dict[str, Any]:
    def combined(source: Mapping[str, list[np.ndarray]], field: str) -> np.ndarray:
        chunks = source.get(field, [])
        return np.concatenate(chunks) if chunks else np.asarray([], dtype=float)

    gp = combined(generated, "package_count")
    ep = combined(eligible, "package_count")
    gv = combined(generated, "demand_cm3")
    ev = combined(eligible, "demand_cm3")
    gs = combined(generated, "service_time_s")
    es = combined(eligible, "service_time_s")
    gt = combined(generated, "tw_present").astype(bool)
    et = combined(eligible, "tw_present").astype(bool)
    gw = combined(generated, "tw_width_s")
    ew = combined(eligible, "tw_width_s")
    gr = combined(generated, "route_size")
    er = combined(eligible, "route_size")
    service_normalizer = max(float(np.subtract(*np.quantile(es, [0.75, 0.25]))), 1.0) if len(es) else 1.0
    gpq, epq = _quantiles(gp), _quantiles(ep)
    return {
        "city_slug": key[0],
        "scale_id": key[1],
        "day_type": key[2],
        "generated_customer_observations": int(len(gp)),
        "eligible_customer_observations_with_view_weighting": int(len(ep)),
        "package_count": {
            "generated_pmf": _pmf(gp),
            "eligible_pmf": _pmf(ep),
            "wasserstein1_packages": _w1(gp, ep),
            "p50_absolute_difference_packages": None if gpq["p50"] is None else abs(float(gpq["p50"]) - float(epq["p50"])),
            "p90_absolute_difference_packages": None if gpq["p90"] is None else abs(float(gpq["p90"]) - float(epq["p90"])),
        },
        "demand_volume": {
            "log1p_wasserstein1": _w1(np.log1p(gv), np.log1p(ev)),
            "generated_quantiles_cm3": _quantiles(gv),
            "eligible_quantiles_cm3": _quantiles(ev),
        },
        "service_time": {
            "generated_quantiles_s": _quantiles(gs),
            "eligible_quantiles_s": _quantiles(es),
            "normalized_wasserstein1": None if _w1(gs, es) is None else float(_w1(gs, es) / service_normalizer),
            "normalizer_eligible_iqr_s": service_normalizer,
        },
        "time_window_presence": {
            "generated_rate": float(gt.mean()) if len(gt) else None,
            "eligible_rate": float(et.mean()) if len(et) else None,
            "percentage_point_difference": float(100.0 * (gt.mean() - et.mean())) if len(gt) and len(et) else None,
        },
        "time_window_width": {
            "generated_specified_quantiles_s": _quantiles(gw),
            "eligible_specified_quantiles_s": _quantiles(ew),
            "generated_ecdf_quantile_grid": _ecdf_quantiles(gw),
            "eligible_ecdf_quantile_grid": _ecdf_quantiles(ew),
        },
        "route_size": {
            "generated_positive_quota_histogram": {str(k): int(v) for k, v in zip(*np.unique(gr.astype(int), return_counts=True))} if len(gr) else {},
            "eligible_source_histogram": {str(k): int(v) for k, v in zip(*np.unique(er.astype(int), return_counts=True))} if len(er) else {},
            "generated_quantiles": _quantiles(gr),
            "eligible_quantiles": _quantiles(er),
        },
    }


def evaluate_construct_valid_acceptance(
    *,
    instance_root: Path,
    amazon_artifact_root: Path,
    cohort_split_path: Path,
    config_path: Path,
    code_provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _read_json(config_path)
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("Unsupported Stage-2 acceptance v3 config")
    family_root = instance_root / "materialized" / "families"
    before = _artifact_snapshot(family_root)
    run = _read_json(instance_root / "stage2_run_report.json")
    phase1 = _read_json(instance_root / "reports" / "phase1" / "summary.json")
    c0 = _read_json(instance_root / "reports" / "stage2_repair" / "c0_exact_comparison.json")
    templates = pd.read_parquet(amazon_artifact_root / "templates.parquet")
    template_index = templates.set_index("template_id", drop=False)
    template_id_set = set(template_index.index.astype(str))
    templates_by_day = {str(day): frame for day, frame in templates.groupby("station_day_id", sort=False)}
    split = _read_json(cohort_split_path)
    assignments = {str(row["station_day_id"]): row for row in split["station_day_assignments"]}
    required = config["required_counts"]
    route_rows: list[dict[str, Any]] = []
    source_failures: list[dict[str, Any]] = []
    inheritance_failures: list[dict[str, Any]] = []
    view_count = 0
    inherited_count = 0
    generated_groups: dict[tuple[str, str, str], dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    eligible_groups: dict[tuple[str, str, str], dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    family_paths = sorted(family_root.glob("*/family_manifest.json"))
    for manifest_path in family_paths:
        family_dir = manifest_path.parent
        manifest = _read_json(manifest_path)
        family_id = str(manifest["family_id"])
        n_parent = int(manifest["parent_customer_count"])
        selection = manifest["selection_report"]
        source = selection["amazon_structure_source"]
        quota_meta = selection["spatial_activation"]["quota"]
        order = manifest["order_source_report"]
        terminal = pd.read_parquet(family_dir / manifest["terminal_index"])
        customers = terminal.loc[terminal["terminal_kind"].eq("customer")].copy()
        route_quota = customers["structure_route_id"].astype(str).value_counts().sort_index()
        route_count_used = int((route_quota > 0).sum())
        structure_days = set(map(str, source["structure_source_ids"]))
        structure_route_ids = set().union(
            *(set(templates_by_day[day]["route_id"].astype(str)) for day in structure_days)
        )
        region_sizes = list(map(int, selection["spatial_activation"]["region_sizes"].values()))
        route_checks = {
            "quotas_are_nonnegative_integers": bool((route_quota >= 0).all() and all(float(v).is_integer() for v in route_quota)),
            "quota_sum_equals_N": int(route_quota.sum()) == n_parent,
            "route_count_used_equals_positive_quota_count": route_count_used == len(route_quota),
            "route_count_used_equals_retained_region_count": route_count_used == int(quota_meta["retained_region_count"]),
            "route_count_used_equals_quota_source_route_count": route_count_used == int(quota_meta["source_route_count"]),
            "all_used_routes_trace_to_structure_source": set(route_quota.index) <= structure_route_ids,
            "region_sizes_sum_equals_N": sum(region_sizes) == n_parent,
            "row_margins_exact": quota_meta["row_margins_exact"] is True,
            "column_margins_exact": quota_meta["column_margins_exact"] is True,
        }
        route_rows.append({
            "family_id": family_id,
            "source_route_count_available_legacy": int(source.get("source_route_count_available", source.get("structure_source_route_count", -1))),
            "route_count_used": route_count_used,
            "route_quota_used": {str(k): int(v) for k, v in route_quota.items()},
            "checks": route_checks,
            "passed": all(route_checks.values()),
        })

        order_days = set(map(str, order["selected_station_day_ids"]))
        source_days = structure_days | order_days
        source_ok = all(
            day in assignments
            and str(assignments[day]["pool"]) == str(order["source_pool"])
            and str(assignments[day]["day_type"]) == str(manifest["day_type"])
            and (str(assignments[day]["pool"]) != "GEN-EVAL" or str(assignments[day].get("generation_track")) == str(order["generation_track"]))
            for day in source_days
        )
        if not source_ok:
            source_failures.append({"family_id": family_id, "source_days": sorted(source_days)})

        parent_template_ids = customers["order_template_id"].astype(str).tolist()
        parent_templates_ok = bool(
            len(parent_template_ids) == n_parent
            and len(set(parent_template_ids)) == n_parent
            and set(parent_template_ids) <= template_id_set
            and set(customers["order_station_day_id"].astype(str)) <= order_days
        )
        if not parent_templates_ok:
            inheritance_failures.append({"family_id": family_id, "view_id": None, "reason": "parent_template_trace"})

        eligible = pd.concat(
            [templates_by_day[day] for day in sorted(order_days)], ignore_index=True
        )
        for view_path in sorted((family_dir / "views").glob("*/view_manifest.json")):
            view = _read_json(view_path)
            view_dir = view_path.parent
            n = int(view["customer_count"])
            indices = np.load(view_dir / view["terminal_parent_indices"], allow_pickle=False)
            selected = terminal.iloc[indices[1 : 1 + n]]
            template_ids = selected["order_template_id"].astype(str).tolist()
            exact = len(template_ids) == n and len(set(template_ids)) == n and set(template_ids) <= template_id_set
            expected = template_index.loc[template_ids] if exact else None
            with np.load(view_dir / view["customer_attributes"], allow_pickle=False) as attributes:
                if exact:
                    exact = all([
                        _stored_equal(attributes["package_counts"], expected["package_count"].to_numpy()),
                        _stored_equal(attributes["demands_cm3"], expected["demand_cm3"].to_numpy()),
                        _stored_equal(attributes["service_time_s"], expected["service_time_s"].to_numpy()),
                        _stored_equal(attributes["time_windows_s"], expected[["tw_start_s", "tw_end_s"]].to_numpy()),
                    ])
                if exact:
                    key = (str(manifest["city_slug"]), str(view["scale_id"]), str(manifest["day_type"]))
                    generated_groups[key]["package_count"].append(attributes["package_counts"].astype(float))
                    generated_groups[key]["demand_cm3"].append(attributes["demands_cm3"].astype(float))
                    generated_groups[key]["service_time_s"].append(attributes["service_time_s"].astype(float))
                    generated_groups[key]["tw_present"].append(expected["tw_was_specified"].to_numpy(dtype=bool))
                    generated_groups[key]["tw_width_s"].append((expected.loc[expected["tw_was_specified"], "tw_end_s"] - expected.loc[expected["tw_was_specified"], "tw_start_s"]).to_numpy(dtype=float))
                    generated_groups[key]["route_size"].append(selected["structure_route_id"].astype(str).value_counts().to_numpy(dtype=float))
                    eligible_groups[key]["package_count"].append(eligible["package_count"].to_numpy(dtype=float))
                    eligible_groups[key]["demand_cm3"].append(eligible["demand_cm3"].to_numpy(dtype=float))
                    eligible_groups[key]["service_time_s"].append(eligible["service_time_s"].to_numpy(dtype=float))
                    eligible_groups[key]["tw_present"].append(eligible["tw_was_specified"].to_numpy(dtype=bool))
                    eligible_groups[key]["tw_width_s"].append((eligible.loc[eligible["tw_was_specified"], "tw_end_s"] - eligible.loc[eligible["tw_was_specified"], "tw_start_s"]).to_numpy(dtype=float))
                    eligible_groups[key]["route_size"].append(eligible.groupby("route_id").size().to_numpy(dtype=float))
            if not exact or view["attribute_report"]["order_template_inheritance"] is not True or view["attribute_report"]["time_windows"]["feasibility_clipping_applied"] is not False:
                inheritance_failures.append({"family_id": family_id, "view_id": str(view["view_id"]), "reason": "customer_attribute_inheritance"})
            view_count += 1
            inherited_count += n

    successful = [item for item in run.get("materialized", []) if item.get("status") in {"materialized", "reused_verified"}]
    verified = [item for item in run.get("verified", []) if item.get("passed") is True]
    discipline = run.get("run_discipline", {})
    checks = {
        "schema_version_and_provenance_complete": bool(run.get("schema") and run.get("code_provenance")),
        "c0_split_membership_leakage_counts_and_5to2_passed": c0.get("passed") is True,
        "phase1_family_correctness_passed": phase1.get("all_hard_gates_passed") is True,
        "planned_materialized_verified_counts_exact": int(run.get("execution", {}).get("selected_family_count", -1)) == int(required["parent_families"]) == len(family_paths) == len(successful) == len(verified),
        "materialized_view_count_exact": view_count == int(required["materialized_views"]),
        "quota_route_count_and_source_provenance_self_consistent": bool(route_rows and all(row["passed"] for row in route_rows) and not source_failures),
        "amazon_customer_attributes_inherited_exactly": not inheritance_failures,
        "matrix_connectivity_feasibility_nested_views_and_verifier_passed": bool(verified and all(item.get("passed") is True for item in verified)),
        "no_timeout_rejection_unresolved_or_abort": not any([run.get("unresolved_family_ids", []), run.get("timed_out_family_ids", []), run.get("aborted_family_ids", []), run.get("rejected_attempts", [])]),
        "no_orphan_process_groups": int(run.get("remaining_process_group_count", 0)) == 0,
        "runner_terminal_report_passed": run.get("passed") is True and discipline.get("stopped") is False,
    }
    after = _artifact_snapshot(family_root)
    checks["family_artifacts_unchanged_during_read_only_audit"] = before == after
    diagnostics_rows = [
        _diagnostic_row(key, generated_groups[key], eligible_groups[key])
        for key in sorted(generated_groups)
    ]
    diagnostics = {
        "schema": DIAGNOSTIC_SCHEMA,
        "status": "complete_report_only",
        "hard_gate": False,
        "contributes_to_acceptance": False,
        "aggregation_unit": "city_x_scale_x_day_type",
        "row_count": len(diagnostics_rows),
        "rows": diagnostics_rows,
        "additional_report_only_metrics": config["report_only_diagnostics"],
        "historical_matching_bias_v2_preserved": True,
        "historical_q90_v1_preserved": True,
        "code_provenance": dict(code_provenance or {}),
        "hash_validation_performed": False,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "scope": "existing_140_family_non_release_pilot_read_only",
        "benchmark_definition": config["benchmark_definition"],
        "checks": checks,
        "counts": {
            "planned_families": int(run.get("execution", {}).get("selected_family_count", -1)),
            "materialized_families": len(successful),
            "verified_families": len(verified),
            "audited_family_manifests": len(family_paths),
            "audited_views": view_count,
            "inherited_customer_observations": inherited_count,
        },
        "canonical_route_count": config["canonical_route_count"],
        "route_quota_audit": {"failed_family_count": sum(not row["passed"] for row in route_rows), "rows": route_rows},
        "source_provenance_failures": source_failures,
        "inheritance_failures": inheritance_failures,
        "diagnostics_report": DIAGNOSTIC_SCHEMA,
        "diagnostics_do_not_contribute_to_passed": True,
        "family_artifacts_modified": before != after,
        "family_artifact_snapshot_method": "path_size_mtime_ns_same_process_no_hash",
        "config_path": str(config_path.resolve()),
        "code_provenance": dict(code_provenance or {}),
        "hash_validation_performed": False,
    }
    return report, diagnostics


def write_construct_valid_acceptance(
    *,
    acceptance_output: Path,
    diagnostics_output: Path,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    acceptance, diagnostics = evaluate_construct_valid_acceptance(**kwargs)
    atomic_write_json(diagnostics_output, diagnostics)
    atomic_write_json(acceptance_output, acceptance)
    return acceptance, diagnostics
