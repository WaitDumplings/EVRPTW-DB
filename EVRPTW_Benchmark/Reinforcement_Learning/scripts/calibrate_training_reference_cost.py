#!/usr/bin/env python3
"""Build the frozen DRL reward contract from deterministic training references.

The calibration is deliberately independent of every learned policy.  It uses
only canonical graph (``G``) Stage-2 training views and the bounded ALNS initial
constructor.  The constructor runs to its deterministic candidate-count limits;
wall-clock time never participates in a route decision.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
META_ROOT = REPO_ROOT / "EVRPTW_Benchmark" / "MetaHeuristics"
ALNS_ROOT = META_ROOT / "ALNS_Solver"
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Dataset_Generator" / "src"))
sys.path.insert(0, str(META_ROOT))
sys.path.insert(0, str(ALNS_ROOT))

from benchmark_common import (  # noqa: E402
    CANONICAL_REPLAY_PROFILE_ID,
    Stage2ViewTask,
    load_stage2_instance,
    normalize_scale,
    read_stage2_tasks,
    validate_routes as validate_meta_routes,
)
from instance_adapter import to_alns_tensor_instance  # noqa: E402
from solver import ALNS_Solver  # noqa: E402

from EVRPTW_Benchmark.Exact.Gurobi_Solver.route_validator import (  # noqa: E402
    validate_routes as validate_objective_routes,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import (  # noqa: E402
    ObjectiveConfig,
    load_objective,
    resolve_objective,
    route_dispatch_count,
)
from EVRPTW_Benchmark.Reinforcement_Learning.common.reward_contract import (  # noqa: E402
    REWARD_CONTRACT_SCHEMA,
    reward_contract_digest,
)


CONTRACT_ID = "drl_energy_vehicle_reference_scale_v2"
CALIBRATION_SCHEMA = "drl_training_reference_calibration_v1"
COHORT_SCHEMA = "drl_training_reference_cohort_v1"
PER_VIEW_SCHEMA = "drl_training_reference_view_v1"
COHORT_HASH_SCHEME = "blake2b_training_reference_view_rank_v1"
ROUTE_HASH_SCHEME = "sha256_canonical_routes_v1"
CONSTRUCTOR_PROFILE_ID = "alns_singleton_best_fit_deterministic_reference_v1"
OBJECTIVE_VERIFIER_PROFILE_ID = "gurobi_stage2_route_validator_v1"
DEFAULT_SCALES = ("Cus500", "Cus1000")
DEFAULT_CITY_COUNT = 10
DEFAULT_DAY_QUOTAS = {"weekday": 36, "weekend": 14}
DEFAULT_VIEWS_PER_CITY = sum(DEFAULT_DAY_QUOTAS.values())
DEFAULT_COHORT_SEED = 20_260_904
DEFAULT_OBJECTIVE = (
    REPO_ROOT
    / "EVRPTW_Benchmark"
    / "Reinforcement_Learning"
    / "configs"
    / "rivian_energy_vehicle_cost_v1.json"
)
DEFAULT_CONTRACT_OUTPUT = (
    REPO_ROOT
    / "EVRPTW_Benchmark"
    / "Reinforcement_Learning"
    / "configs"
    / "drl_reward_contract_energy_vehicle_v2.json"
)
DEFAULT_ARTIFACT_DIR = (
    REPO_ROOT
    / "EVRPTW_Benchmark"
    / "results"
    / "DRL_rq_v1"
    / "artifacts"
    / "reward_contract_energy_vehicle_v2"
)
REQUIRED_INDEX_COLUMNS = {
    "view_id",
    "family_id",
    "family_cohort_id",
    "consumer_cohort_id",
    "split_id",
    "track_id",
    "city_slug",
    "day_type",
    "scale_id",
    "customer_count",
    "view_seed",
}


class CalibrationError(RuntimeError):
    """Raised when a partial or unverifiable calibration would be produced."""


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "sort_keys": True,
        "ensure_ascii": False,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def route_sha256(routes: Sequence[Sequence[int]]) -> str:
    canonical = [[int(node) for node in route] for route in routes]
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_cohort_rank(
    *, seed: int, scale: str, city_slug: str, day_type: str, view_id: str
) -> str:
    payload = (
        f"{COHORT_HASH_SCHEME}\0{int(seed)}\0{normalize_scale(scale)}\0"
        f"{str(city_slug)}\0{str(day_type)}\0{str(view_id)}"
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def select_fixed_training_cohort(
    index: pd.DataFrame,
    *,
    scales: Sequence[str] = DEFAULT_SCALES,
    day_quotas: Mapping[str, int] = DEFAULT_DAY_QUOTAS,
    seed: int = DEFAULT_COHORT_SEED,
    expected_city_count: int = DEFAULT_CITY_COUNT,
) -> pd.DataFrame:
    """Select an order-independent, approximately proportional cohort."""

    missing = sorted(REQUIRED_INDEX_COLUMNS.difference(index.columns))
    if missing:
        raise CalibrationError(f"training index is missing columns: {missing}")
    if index["view_id"].astype(str).duplicated().any():
        raise CalibrationError("training index contains duplicate view_id values")
    quotas = {str(day): int(quota) for day, quota in day_quotas.items()}
    if set(quotas) != {"weekday", "weekend"} or any(
        quota <= 0 for quota in quotas.values()
    ):
        raise CalibrationError(
            "day_quotas must define positive weekday and weekend quotas"
        )
    city_count = int(expected_city_count)
    if city_count <= 0:
        raise CalibrationError("expected_city_count must be positive")

    frame = index.copy()
    for field in ("view_id", "family_id", "city_slug", "day_type"):
        invalid = frame[field].isna() | (frame[field].astype(str).str.strip() == "")
        if invalid.any():
            raise CalibrationError(f"training index contains null/empty {field}")
    frame["_scale_label"] = frame["scale_id"].map(normalize_scale)
    requested = tuple(normalize_scale(scale) for scale in scales)
    if len(set(requested)) != len(requested):
        raise CalibrationError("calibration scales must be unique")

    selected_parts: list[pd.DataFrame] = []
    cohort_position = 0
    expected_days = {"weekday", "weekend"}
    common_cities: tuple[str, ...] | None = None
    for scale in requested:
        candidates = frame.loc[frame["_scale_label"] == scale].copy()
        if candidates.empty:
            raise CalibrationError(f"training index has no rows for {scale}")
        non_training = candidates.loc[
            (candidates["split_id"].astype(str) != "train")
            | (candidates["track_id"].astype(str) != "train")
        ]
        if not non_training.empty:
            raise CalibrationError(
                f"{scale} source contains validation/test rows; calibration is training-only"
            )
        for cohort_field in ("consumer_cohort_id", "family_cohort_id"):
            if set(candidates[cohort_field].astype(str)) != {"core/train"}:
                raise CalibrationError(
                    f"{scale} requires {cohort_field}=core/train"
                )
        expected_customers = int(scale.removeprefix("Cus"))
        if set(candidates["customer_count"].astype(int)) != {expected_customers}:
            raise CalibrationError(f"{scale} has inconsistent customer_count values")
        cities = tuple(sorted(candidates["city_slug"].astype(str).unique()))
        days = set(candidates["day_type"].astype(str).unique())
        if len(cities) != city_count:
            raise CalibrationError(
                f"{scale} requires exactly {city_count} training cities, found {len(cities)}"
            )
        if days != expected_days:
            raise CalibrationError(
                f"{scale} requires weekday/weekend strata, found {sorted(days)}"
            )
        if common_cities is None:
            common_cities = cities
        elif cities != common_cities:
            raise CalibrationError("calibration scales do not share the same city cohort")

        for city_slug in cities:
            for day_type in sorted(expected_days):
                quota = quotas[day_type]
                stratum = candidates.loc[
                    (candidates["city_slug"].astype(str) == city_slug)
                    & (candidates["day_type"].astype(str) == day_type)
                ].copy()
                if len(stratum) < quota:
                    raise CalibrationError(
                        f"{scale}/{city_slug}/{day_type} has {len(stratum)} views; "
                        f"requires {quota}"
                    )
                stratum["cohort_rank"] = [
                    stable_cohort_rank(
                        seed=seed,
                        scale=scale,
                        city_slug=city_slug,
                        day_type=day_type,
                        view_id=view_id,
                    )
                    for view_id in stratum["view_id"].astype(str)
                ]
                stratum = stratum.sort_values(
                    ["cohort_rank", "view_id"], kind="stable"
                ).iloc[:quota].copy()
                stratum["stratum_position"] = np.arange(quota, dtype=np.int64)
                stratum["cohort_position"] = np.arange(
                    cohort_position, cohort_position + quota, dtype=np.int64
                )
                cohort_position += quota
                selected_parts.append(stratum)

    selected = pd.concat(selected_parts, ignore_index=True)
    expected_total = len(requested) * city_count * sum(quotas.values())
    if len(selected) != expected_total or not selected["view_id"].astype(str).is_unique:
        raise CalibrationError("fixed cohort cardinality/identity invariant failed")
    selected["scale_label"] = selected["_scale_label"]
    columns = [
        "cohort_position",
        "stratum_position",
        "cohort_rank",
        "view_id",
        "family_id",
        "city_slug",
        "day_type",
        "scale_label",
        "customer_count",
        "view_seed",
    ]
    return selected[columns].copy()


def build_work_items(
    train_index: str | Path,
    family_root: str | Path,
    cohort: pd.DataFrame,
    objective: ObjectiveConfig | Mapping[str, Any],
) -> list[dict[str, Any]]:
    objective_config = resolve_objective(objective)
    tasks = read_stage2_tasks(train_index, family_root=family_root)
    by_view: dict[str, Stage2ViewTask] = {}
    for task in tasks:
        if task.view_id in by_view:
            raise CalibrationError(f"duplicate Stage-2 task for {task.view_id}")
        by_view[task.view_id] = task

    work: list[dict[str, Any]] = []
    for record in cohort.to_dict(orient="records"):
        view_id = str(record["view_id"])
        task = by_view.get(view_id)
        if task is None:
            raise CalibrationError(f"selected view is absent from Stage-2 loader: {view_id}")
        if (
            task.split_id != "train"
            or task.track_id != "train"
            or task.scale_label != str(record["scale_label"])
            or task.city_slug != str(record["city_slug"])
            or task.family_id != str(record["family_id"])
            or task.customer_count != int(record["customer_count"])
            or task.view_seed != int(record["view_seed"])
        ):
            raise CalibrationError(f"index/Stage-2 task metadata mismatch for {view_id}")
        work.append(
            {
                "task": task.to_dict(),
                "cohort": {
                    key: (
                        int(value)
                        if key in {
                            "cohort_position",
                            "stratum_position",
                            "customer_count",
                            "view_seed",
                        }
                        else str(value)
                    )
                    for key, value in record.items()
                },
                "objective": objective_config.to_dict(),
            }
        )
    return work


def calibrate_view(work_item: Mapping[str, Any]) -> dict[str, Any]:
    """Construct and independently verify one reference; raise on any failure."""

    task = Stage2ViewTask.from_dict(dict(work_item["task"]))
    cohort = dict(work_item["cohort"])
    objective = resolve_objective(work_item["objective"])
    instance = load_stage2_instance(task)
    if str(instance.day_type) != str(cohort["day_type"]):
        raise CalibrationError(f"loaded day_type mismatch for {task.view_id}")

    solver = ALNS_Solver(to_alns_tensor_instance(instance), seed=0, format="tensor")
    routes = solver.construct_deterministic_reference_solution()
    if solver.singleton_source != "stage2_certificate_replayed":
        raise CalibrationError(
            f"{task.view_id} did not use a replayed Stage-2 feasibility certificate"
        )

    meta_audit = validate_meta_routes(instance, routes)
    objective_audit = validate_objective_routes(instance, routes)
    if not meta_audit["passed"] or not objective_audit["passed"]:
        violations = {
            "meta": list(meta_audit.get("violations", [])),
            "objective": list(objective_audit.get("violations", [])),
        }
        raise CalibrationError(
            f"canonical route replay failed for {task.view_id}: {violations}"
        )
    meta_distance = float(meta_audit["objective_distance_km"])
    distance = float(objective_audit["objective_distance_km"])
    if not math.isclose(meta_distance, distance, rel_tol=1e-12, abs_tol=1e-9):
        raise CalibrationError(
            f"independent verifier distance mismatch for {task.view_id}: "
            f"{meta_distance} != {distance}"
        )
    vehicles = int(route_dispatch_count(routes))
    if vehicles <= 0 or vehicles != len(routes):
        raise CalibrationError(
            f"canonical reference dispatch count mismatch for {task.view_id}"
        )
    objective_fields = objective.fields(distance, vehicles)
    cost = float(objective_fields["objective_value"])
    if not math.isfinite(cost) or cost <= 0.0:
        raise CalibrationError(f"invalid reference objective for {task.view_id}: {cost}")

    construction = solver.initial_construction_stats
    deterministic_audit = {
        "profile_id": CONSTRUCTOR_PROFILE_ID,
        "strategy": str(construction.get("strategy", "")),
        "singleton_source": str(construction.get("singleton_source", "")),
        "singleton_route_count": int(construction.get("singleton_route_count", 0)),
        "result_route_count": int(construction.get("result_route_count", 0)),
        "merged_customer_count": int(construction.get("merged_customer_count", 0)),
        "deterministic": bool(construction.get("deterministic", False)),
        "wall_clock_cutoff_enabled": bool(
            construction.get("wall_clock_cutoff_enabled", True)
        ),
        "termination_basis": str(construction.get("termination_basis", "")),
        "initial_merge_candidate_limit": solver.initial_merge_candidate_limit,
        "initial_exact_insertion_limit": solver.initial_exact_insertion_limit,
    }
    if (
        not deterministic_audit["deterministic"]
        or deterministic_audit["wall_clock_cutoff_enabled"]
        or deterministic_audit["termination_basis"] != "candidate_limits_only"
        or construction.get("budget_exhausted")
    ):
        raise CalibrationError(
            f"non-deterministic construction metadata for {task.view_id}"
        )

    clean_routes = [[int(node) for node in route] for route in routes]
    return {
        "schema": PER_VIEW_SCHEMA,
        **cohort,
        "representation": "G",
        "routes": clean_routes,
        "route_hash_scheme": ROUTE_HASH_SCHEME,
        "route_sha256": route_sha256(clean_routes),
        "route_validation_passed": True,
        "objective_verifier_profile_id": OBJECTIVE_VERIFIER_PROFILE_ID,
        "canonical_replay_profile_id": CANONICAL_REPLAY_PROFILE_ID,
        "objective_distance_km": distance,
        "vehicles_started": vehicles,
        **objective_fields,
        "charging_visit_count": int(objective_audit["charging_visit_count"]),
        "total_charging_time_s": float(objective_audit["total_charging_time_s"]),
        "charging_power_source": str(objective_audit["charging_power_source"]),
        "charging_power_derating_factor": float(
            objective_audit["charging_power_derating_factor"]
        ),
        "constructor": deterministic_audit,
    }


def run_work_items(
    work_items: Sequence[Mapping[str, Any]], *, num_workers: int
) -> list[dict[str, Any]]:
    workers = int(num_workers)
    if workers <= 0:
        raise CalibrationError("num_workers must be positive")
    if workers == 1:
        rows = [calibrate_view(item) for item in work_items]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(calibrate_view, work_items, chunksize=1))
    return sorted(rows, key=lambda row: int(row["cohort_position"]))


def validate_complete_results(
    cohort: pd.DataFrame,
    rows: Sequence[Mapping[str, Any]],
    objective: ObjectiveConfig | Mapping[str, Any],
) -> None:
    objective_config = resolve_objective(objective)
    expected = cohort.sort_values("cohort_position")["view_id"].astype(str).tolist()
    observed = [str(row.get("view_id", "")) for row in rows]
    if observed != expected:
        raise CalibrationError("calibration results do not exactly match the fixed cohort")
    if len(set(observed)) != len(observed):
        raise CalibrationError("calibration results contain duplicate view IDs")
    for row in rows:
        if row.get("route_validation_passed") is not True:
            raise CalibrationError(f"unverified result for {row.get('view_id')}")
        if route_sha256(row.get("routes", [])) != row.get("route_sha256"):
            raise CalibrationError(f"route hash mismatch for {row.get('view_id')}")
        distance = float(row.get("objective_distance_km", math.nan))
        vehicles = int(row.get("vehicles_started", 0))
        cost = float(row.get("objective_value", math.nan))
        if not math.isfinite(distance) or distance < 0.0 or vehicles <= 0:
            raise CalibrationError(f"invalid D/K result for {row.get('view_id')}")
        if not math.isfinite(cost) or cost <= 0.0:
            raise CalibrationError(f"invalid C result for {row.get('view_id')}")
        routes = row.get("routes", [])
        if route_dispatch_count(routes) != vehicles:
            raise CalibrationError(f"K/routes mismatch for {row.get('view_id')}")
        expected_fields = objective_config.fields(distance, vehicles)
        for field in (
            "objective_value",
            "objective_cost_usd",
            "electricity_cost_usd",
            "vehicle_cost_usd",
        ):
            actual = row.get(field)
            expected_value = expected_fields[field]
            if actual is None or expected_value is None or not math.isclose(
                float(actual), float(expected_value), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise CalibrationError(
                    f"{field}/D/K mismatch for {row.get('view_id')}"
                )
        if (
            str(row.get("objective_mode")) != objective_config.mode
            or str(row.get("objective_profile_id")) != objective_config.profile_id
            or str(row.get("objective_unit")) != objective_config.unit
        ):
            raise CalibrationError(
                f"objective metadata mismatch for {row.get('view_id')}"
            )


def linear_quantile(values: Iterable[float], probability: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or np.any(~np.isfinite(array)):
        raise CalibrationError("quantile input must contain finite values")
    return float(np.quantile(array, float(probability), method="linear"))


def summarize_scale_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict, dict]:
    costs = np.asarray([float(row["objective_value"]) for row in rows], dtype=np.float64)
    distances = np.asarray(
        [float(row["objective_distance_km"]) for row in rows], dtype=np.float64
    )
    vehicles = np.asarray([int(row["vehicles_started"]) for row in rows], dtype=np.int64)
    if not len(costs) or np.any(~np.isfinite(costs)) or np.any(costs <= 0.0):
        raise CalibrationError("reference costs must be finite and positive")
    objective_scale = float(np.median(costs))
    normalized_q99 = linear_quantile(costs / objective_scale, 0.99)
    failure_base = normalized_q99 + 1.0
    terms = {
        "objective_scale": objective_scale,
        "failure_base": failure_base,
        "unserved_coefficient": 1.0,
    }
    statistics = {
        "sample_count": int(len(costs)),
        "objective_cost_usd": {
            "min": float(costs.min()),
            "median": objective_scale,
            "q99_linear": linear_quantile(costs, 0.99),
            "max": float(costs.max()),
        },
        "objective_distance_km": {
            "min": float(distances.min()),
            "median": float(np.median(distances)),
            "max": float(distances.max()),
        },
        "vehicles_started": {
            "min": int(vehicles.min()),
            "median": float(np.median(vehicles)),
            "max": int(vehicles.max()),
        },
        "normalized_objective_q99_linear": normalized_q99,
        "failure_base": failure_base,
    }
    return terms, statistics


def build_contract_payload(
    *,
    objective: ObjectiveConfig | Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    source_index_sha256: str,
    cohort_file_sha256: str,
    per_view_file_sha256: str,
    cohort_seed: int = DEFAULT_COHORT_SEED,
    day_quotas: Mapping[str, int] = DEFAULT_DAY_QUOTAS,
    expected_city_count: int = DEFAULT_CITY_COUNT,
) -> dict[str, Any]:
    objective_config = resolve_objective(objective)
    if not objective_config.is_cost:
        raise CalibrationError("reference-cost calibration requires a cost objective")
    by_scale: dict[str, list[Mapping[str, Any]]] = {scale: [] for scale in DEFAULT_SCALES}
    for row in rows:
        scale = normalize_scale(row["scale_label"])
        if scale not in by_scale:
            raise CalibrationError(f"unexpected calibration scale: {scale}")
        by_scale[scale].append(row)
    quotas = {str(day): int(quota) for day, quota in day_quotas.items()}
    if set(quotas) != {"weekday", "weekend"} or any(
        quota <= 0 for quota in quotas.values()
    ):
        raise CalibrationError(
            "day_quotas must define positive weekday and weekend quotas"
        )
    expected_per_scale = int(expected_city_count) * sum(quotas.values())
    if any(len(scale_rows) != expected_per_scale for scale_rows in by_scale.values()):
        raise CalibrationError("calibration scale does not contain the fixed cohort size")

    scales: dict[str, dict[str, float]] = {}
    scale_statistics: dict[str, dict[str, Any]] = {}
    for scale in DEFAULT_SCALES:
        terms, statistics = summarize_scale_rows(by_scale[scale])
        scales[scale] = terms
        scale_statistics[scale] = statistics

    objective_payload = objective_config.to_dict()
    objective_digest = hashlib.sha256(
        json.dumps(
            objective_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    payload: dict[str, Any] = {
        "schema": REWARD_CONTRACT_SCHEMA,
        "contract_id": CONTRACT_ID,
        "objective": objective_payload,
        "scales": scales,
        "calibration": {
            "schema": CALIBRATION_SCHEMA,
            "representation": "G",
            "source_split": "train",
            "source_track": "train",
            "source_view_index_sha256": str(source_index_sha256),
            "objective_config_sha256": objective_digest,
            "cohort": {
                "schema": COHORT_SCHEMA,
                "selection_unit": "view",
                "hash_scheme": COHORT_HASH_SCHEME,
                "seed": int(cohort_seed),
                "city_count": int(expected_city_count),
                "day_types": ["weekday", "weekend"],
                "day_quota_per_city": {
                    "weekday": quotas["weekday"],
                    "weekend": quotas["weekend"],
                },
                "views_per_city": sum(quotas.values()),
                "views_per_scale": expected_per_scale,
                "scales": list(DEFAULT_SCALES),
                "weighting": "approximately_proportional_city_day_stratified_72_28",
                "source_day_mix": {
                    "weekday": 0.714,
                    "weekend": 0.286,
                },
                "selected_day_mix": {
                    "weekday": quotas["weekday"] / sum(quotas.values()),
                    "weekend": quotas["weekend"] / sum(quotas.values()),
                },
            },
            "constructor": {
                "profile_id": CONSTRUCTOR_PROFILE_ID,
                "base_algorithm_profile_id": "alns_stage2_scalable_v2",
                "initial_construction_strategy": "singleton_best_fit_v1",
                "singleton_source_required": "stage2_certificate_replayed",
                "termination_basis": "candidate_limits_only",
                "wall_clock_cutoff_enabled": False,
                "stochastic_search_iterations": 0,
            },
            "verification": {
                "objective_verifier_profile_id": OBJECTIVE_VERIFIER_PROFILE_ID,
                "canonical_replay_profile_id": CANONICAL_REPLAY_PROFILE_ID,
                "require_both_pass": True,
                "D_and_K_source": "same_verified_routes",
                "K_definition": "depot_to_nondepot_dispatch_count",
            },
            "aggregation": {
                "objective_scale": "median(C)",
                "failure_base": "linear_quantile_0.99(C/objective_scale)+1",
                "unserved_coefficient": 1.0,
                "quantile_method": "linear",
            },
            "scale_statistics": scale_statistics,
            "artifacts": {
                "cohort_file": "cohort.json",
                "cohort_file_sha256": str(cohort_file_sha256),
                "per_view_file": "per_view.jsonl",
                "per_view_file_sha256": str(per_view_file_sha256),
            },
        },
    }
    payload["sha256"] = reward_contract_digest(payload)
    return payload


def cohort_payload(
    cohort: pd.DataFrame, *, source_index_sha256: str
) -> dict[str, Any]:
    records = []
    for raw in cohort.sort_values("cohort_position").to_dict(orient="records"):
        records.append(
            {
                key: (
                    int(value)
                    if key in {
                        "cohort_position",
                        "stratum_position",
                        "customer_count",
                        "view_seed",
                    }
                    else str(value)
                )
                for key, value in raw.items()
            }
        )
    return {
        "schema": COHORT_SCHEMA,
        "representation": "G",
        "source_split": "train",
        "source_track": "train",
        "source_view_index_sha256": str(source_index_sha256),
        "hash_scheme": COHORT_HASH_SCHEME,
        "seed": DEFAULT_COHORT_SEED,
        "day_quota_per_city": dict(DEFAULT_DAY_QUOTAS),
        "views_per_city": DEFAULT_VIEWS_PER_CITY,
        "weighting": "approximately_proportional_city_day_stratified_72_28",
        "views": records,
    }


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def write_calibration_outputs(
    *,
    artifact_dir: str | Path,
    contract_path: str | Path,
    cohort: pd.DataFrame,
    rows: Sequence[Mapping[str, Any]],
    objective: ObjectiveConfig | Mapping[str, Any],
    source_index_sha256: str,
) -> dict[str, Any]:
    """Atomically write complete artifacts, publishing the contract last."""

    validate_complete_results(cohort, rows, objective)
    cohort_document = cohort_payload(
        cohort, source_index_sha256=source_index_sha256
    )
    cohort_bytes = canonical_json_bytes(cohort_document, pretty=True)
    per_view_bytes = b"".join(
        canonical_json_bytes(dict(row), pretty=False) for row in rows
    )
    cohort_digest = hashlib.sha256(cohort_bytes).hexdigest()
    per_view_digest = hashlib.sha256(per_view_bytes).hexdigest()
    contract = build_contract_payload(
        objective=objective,
        rows=rows,
        source_index_sha256=source_index_sha256,
        cohort_file_sha256=cohort_digest,
        per_view_file_sha256=per_view_digest,
    )

    artifact_root = Path(artifact_dir)
    atomic_write_bytes(artifact_root / "cohort.json", cohort_bytes)
    atomic_write_bytes(artifact_root / "per_view.jsonl", per_view_bytes)
    # Consumers only see a new contract after both referenced audit artifacts
    # are durable.  A failed calibration therefore cannot publish partial terms.
    atomic_write_bytes(contract_path, canonical_json_bytes(contract, pretty=True))
    return contract


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate Cus500/Cus1000 DRL reward terms from deterministic, "
            "training-only ALNS constructive references."
        )
    )
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--objective-config", type=Path, default=DEFAULT_OBJECTIVE)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-contract", type=Path, default=DEFAULT_CONTRACT_OUTPUT)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.train_index.is_file():
        raise FileNotFoundError(args.train_index)
    if not args.family_root.is_dir():
        raise FileNotFoundError(args.family_root)
    objective = load_objective(args.objective_config)
    if not objective.is_cost:
        raise CalibrationError("objective config must use energy_vehicle_cost")
    index = pd.read_parquet(args.train_index)
    cohort = select_fixed_training_cohort(index)
    work_items = build_work_items(
        args.train_index, args.family_root, cohort, objective
    )
    rows = run_work_items(work_items, num_workers=args.num_workers)
    validate_complete_results(cohort, rows, objective)
    contract = write_calibration_outputs(
        artifact_dir=args.artifact_dir,
        contract_path=args.output_contract,
        cohort=cohort,
        rows=rows,
        objective=objective,
        source_index_sha256=file_sha256(args.train_index),
    )
    print(
        json.dumps(
            {
                "contract": str(args.output_contract.resolve()),
                "sha256": contract["sha256"],
                "artifact_dir": str(args.artifact_dir.resolve()),
                "views": len(rows),
                "scales": contract["scales"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
