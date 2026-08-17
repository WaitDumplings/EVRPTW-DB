"""Deterministic matrix-family and lower-scale view planning."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Stage2Config
from .rounding import largest_remainder

CORE_SCALES = ("cus100", "cus500", "cus1000")


def derive_seed(master_seed: int, namespace: str, *parts: object) -> int:
    text = "|".join([str(master_seed), namespace, *(str(part) for part in parts)])
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big") % (
        2**63 - 1
    )


def materialization_attempt_inputs(
    family: Mapping[str, Any],
    views: pd.DataFrame,
    *,
    attempt_number: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Derive a reproducible replacement seed namespace after a rejected attempt."""
    result_family = dict(family)
    base_seed = int(family["family_seed"])
    attempt_seed = (
        base_seed
        if attempt_number == 0
        else derive_seed(base_seed, "materialization_attempt", attempt_number)
    )
    result_family.update(
        {
            "base_family_seed": base_seed,
            "materialization_attempt_number": int(attempt_number),
            "materialization_attempt_seed": attempt_seed,
            "family_seed": attempt_seed,
            "depot_seed": derive_seed(attempt_seed, "depot"),
            "customer_superset_seed": derive_seed(attempt_seed, "customers"),
            "charger_seed": derive_seed(attempt_seed, "chargers"),
            "road_state_seed": derive_seed(attempt_seed, "road_state"),
            "vehicle_seed": derive_seed(attempt_seed, "vehicle"),
        }
    )
    result_views = views.copy()
    for index, row in result_views.iterrows():
        view_seed = derive_seed(
            attempt_seed,
            "view",
            str(row["scale_id"]),
            int(row["branch_index"]),
        )
        result_views.at[index, "view_seed"] = view_seed
    return result_family, result_views


def _family_id(dataset_id: str, cohort_id: str, city_slug: str, ordinal: int) -> str:
    token = f"{dataset_id}|{cohort_id}|{city_slug}|{ordinal}".encode()
    return "mf_" + hashlib.blake2b(token, digest_size=12).hexdigest()


def _view_id(family_id: str, scale_id: str, branch_index: int) -> str:
    token = f"{family_id}|{scale_id}|{branch_index}".encode()
    return "iv_" + hashlib.blake2b(token, digest_size=12).hexdigest()


def _balanced_counts(total: int, cities: tuple[str, ...]) -> dict[str, int]:
    if not cities:
        return {}
    if total % len(cities):
        raise ValueError(f"Count {total} is not divisible by {len(cities)} cities")
    return {city: total // len(cities) for city in cities}


def _cohort_specs(config: Stage2Config) -> list[dict[str, Any]]:
    return [
        {
            "cohort_id": "core/train",
            "split_id": "train",
            "track_id": "train",
            "cities": config.train_cities,
            "count": config.train_parent_family_count,
            "parent_scale_id": "cus1000",
            "customer_pool": "train",
        },
        {
            "cohort_id": "core/val",
            "split_id": "val",
            "track_id": "validation",
            "cities": config.train_cities,
            "count": config.validation_parent_family_count,
            "parent_scale_id": "cus1000",
            "customer_pool": "train",
        },
        {
            "cohort_id": "core/test/test1_new_seed",
            "split_id": "test",
            "track_id": "test1_new_seed",
            "cities": config.train_cities,
            "count": config.core_test_parent_family_count,
            "parent_scale_id": "cus1000",
            "customer_pool": "train",
        },
        {
            "cohort_id": "core/test/test2_heldout_locations",
            "split_id": "test",
            "track_id": "test2_heldout_locations",
            "cities": config.train_cities,
            "count": config.core_test_parent_family_count,
            "parent_scale_id": "cus1000",
            "customer_pool": "heldout",
        },
        {
            "cohort_id": "core/test/test3_heldout_city",
            "split_id": "test",
            "track_id": "test3_heldout_city",
            "cities": (config.heldout_city,),
            "count": config.core_test_parent_family_count,
            "parent_scale_id": "cus1000",
            "customer_pool": "all_release_eligible",
        },
        {
            "cohort_id": "scalability_cus2000/test/unseen_scale_same_cities",
            "split_id": "test",
            "track_id": "unseen_scale_same_cities",
            "cities": config.train_cities,
            "count": config.scalability_parent_family_count,
            "parent_scale_id": "cus2000",
            "customer_pool": "train",
        },
    ]


def _branch_count(scale_id: str) -> int:
    return {"cus50": 20, "cus100": 10, "cus500": 2, "cus1000": 1, "cus2000": 1}[
        scale_id
    ]


def _consumer_cohort(family_cohort: str, scale_id: str) -> str:
    if scale_id != "cus50":
        return family_cohort
    mapping = {
        "core/train": "compatibility_cus50/train",
        "core/val": "compatibility_cus50/val",
        "core/test/test1_new_seed": (
            "compatibility_cus50/test/test1_new_seed_same_cities"
        ),
    }
    return mapping[family_cohort]


def _view_scales_for_cohort(cohort_id: str) -> tuple[str, ...]:
    if cohort_id == "core/train":
        return ("cus50", "cus100", "cus500", "cus1000")
    if cohort_id in {"core/val", "core/test/test1_new_seed"}:
        return ("cus50", "cus100", "cus500", "cus1000")
    if cohort_id in {
        "core/test/test2_heldout_locations",
        "core/test/test3_heldout_city",
    }:
        return CORE_SCALES
    if cohort_id == "scalability_cus2000/test/unseen_scale_same_cities":
        return ("cus1000", "cus2000")
    raise KeyError(cohort_id)


def build_generation_plan(
    config: Stage2Config,
    *,
    available_cities: Iterable[str] | None = None,
    pilot_families_per_city: int | None = None,
    include_tracks: Iterable[str] | None = None,
    non_release_pilot: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return family index, view index, and a compact plan registry.

    ``pilot_families_per_city`` replaces each cohort's official count and is
    legal only for a non-release pilot.
    """

    if pilot_families_per_city is not None:
        if not non_release_pilot:
            raise ValueError("pilot_families_per_city requires non_release_pilot=True")
        if int(pilot_families_per_city) <= 0:
            raise ValueError("pilot_families_per_city must be positive")
    available = set(available_cities or (*config.train_cities, config.heldout_city))
    if pilot_families_per_city is None:
        required_cities = {*config.train_cities, config.heldout_city}
        missing_cities = sorted(required_cities - available)
        if missing_cities:
            raise ValueError(
                f"Official planning requires all ten training cities and the held-out city; "
                f"missing: {missing_cities}"
            )
    track_filter = set(include_tracks or [])
    families: list[dict[str, Any]] = []
    views: list[dict[str, Any]] = []

    for cohort in _cohort_specs(config):
        if track_filter and str(cohort["track_id"]) not in track_filter:
            continue
        cities = tuple(city for city in cohort["cities"] if city in available)
        if not cities:
            continue
        if pilot_families_per_city is None:
            if len(cities) != len(cohort["cities"]):
                missing = sorted(set(cohort["cities"]) - available)
                raise ValueError(
                    f"Official plan is missing cities for {cohort['cohort_id']}: {missing}"
                )
            counts = _balanced_counts(int(cohort["count"]), cities)
        else:
            counts = {city: int(pilot_families_per_city) for city in cities}

        for city_slug in cities:
            city_count = counts[city_slug]
            fractional_day_counts = np.asarray(
                [
                    city_count * config.weekday_weight
                    / (config.weekday_weight + config.weekend_weight),
                    city_count * config.weekend_weight
                    / (config.weekday_weight + config.weekend_weight),
                ],
                dtype=float,
            )
            day_counts = largest_remainder(
                fractional_day_counts,
                total=city_count,
                seed=config.master_seed,
                namespace=f"day_type_quota:{cohort['cohort_id']}:{city_slug}",
                labels=["weekday", "weekend"],
            )
            day_labels = np.asarray(
                ["weekday"] * int(day_counts[0])
                + ["weekend"] * int(day_counts[1]),
                dtype=object,
            )
            day_rng = np.random.default_rng(
                derive_seed(
                    config.master_seed,
                    "day_type_slot_shuffle",
                    cohort["cohort_id"],
                    city_slug,
                )
            )
            day_rng.shuffle(day_labels)
            for city_ordinal in range(counts[city_slug]):
                family_id = _family_id(
                    config.dataset_id, str(cohort["cohort_id"]), city_slug, city_ordinal
                )
                family_seed = derive_seed(
                    config.master_seed,
                    "matrix_family",
                    cohort["cohort_id"],
                    city_slug,
                    city_ordinal,
                )
                parent_scale = config.scale(str(cohort["parent_scale_id"]))
                family = {
                    "family_id": family_id,
                    "family_cohort_id": str(cohort["cohort_id"]),
                    "split_id": str(cohort["split_id"]),
                    "track_id": str(cohort["track_id"]),
                    "city_slug": city_slug,
                    "city_ordinal": int(city_ordinal),
                    "parent_scale_id": parent_scale.scale_id,
                    "parent_customer_count": parent_scale.customers,
                    "parent_charging_station_count": parent_scale.charging_stations,
                    "parent_terminal_count": parent_scale.terminal_count,
                    "customer_pool": str(cohort["customer_pool"]),
                    "day_type": str(day_labels[city_ordinal]),
                    "day_type_allocation_policy": "fixed_city_cohort_largest_remainder_5_to_2_v1",
                    "family_seed": family_seed,
                    "depot_seed": derive_seed(family_seed, "depot"),
                    "customer_superset_seed": derive_seed(family_seed, "customers"),
                    "charger_seed": derive_seed(family_seed, "chargers"),
                    "road_state_seed": derive_seed(family_seed, "road_state"),
                    "vehicle_seed": derive_seed(family_seed, "vehicle"),
                    "matrix_dtype": config.matrix_dtype,
                    "path_policy_id": config.matrix_path_policy,
                    "non_release_pilot": bool(non_release_pilot),
                    "materialization_status": "planned",
                }
                families.append(family)

                for scale_id in _view_scales_for_cohort(str(cohort["cohort_id"])):
                    scale = config.scale(scale_id)
                    count = parent_scale.customers // scale.customers
                    if cohort["cohort_id"] == "scalability_cus2000/test/unseen_scale_same_cities":
                        branch_indices = (0,)
                    elif cohort["split_id"] == "train":
                        branch_indices = range(count)
                    else:
                        leaf = derive_seed(family_seed, "evaluation_chain_leaf") % 20
                        branch_indices = (
                            {
                                "cus50": leaf,
                                "cus100": leaf // 2,
                                "cus500": leaf // 10,
                                "cus1000": 0,
                            }[scale_id],
                        )
                    for branch_index in branch_indices:
                        view_id = _view_id(family_id, scale_id, int(branch_index))
                        consumer = _consumer_cohort(str(cohort["cohort_id"]), scale_id)
                        view_seed = derive_seed(family_seed, "view", scale_id, branch_index)
                        views.append(
                            {
                                "view_id": view_id,
                                "family_id": family_id,
                                "family_cohort_id": str(cohort["cohort_id"]),
                                "consumer_cohort_id": consumer,
                                "split_id": str(cohort["split_id"]),
                                "track_id": str(cohort["track_id"]),
                                "city_slug": city_slug,
                                "scale_id": scale_id,
                                "customer_count": scale.customers,
                                "charging_station_count": scale.charging_stations,
                                "terminal_count": scale.terminal_count,
                                "branch_index": int(branch_index),
                                "branch_count": int(count),
                                "customer_pool": str(cohort["customer_pool"]),
                                "day_type": str(day_labels[city_ordinal]),
                                "nested_evaluation_chain": bool(
                                    cohort["split_id"] != "train"
                                ),
                                "view_seed": view_seed,
                                "matrix_storage": (
                                    "parent" if scale_id == parent_scale.scale_id else "index_view"
                                ),
                                "non_release_pilot": bool(non_release_pilot),
                                "materialization_status": "planned",
                            }
                        )

    family_frame = pd.DataFrame.from_records(families)
    view_frame = pd.DataFrame.from_records(views)
    if family_frame.empty or view_frame.empty:
        raise ValueError("Generation plan contains no families or views")
    if family_frame["family_id"].duplicated().any():
        raise AssertionError("family_id collision detected")
    if view_frame["view_id"].duplicated().any():
        raise AssertionError("view_id collision detected")
    family_ids = set(family_frame["family_id"])
    if not set(view_frame["family_id"]).issubset(family_ids):
        raise AssertionError("A view references a family outside this plan")
    owner_count = family_frame.groupby("family_id")["family_cohort_id"].nunique()
    if int(owner_count.max()) != 1:
        raise AssertionError("A matrix family belongs to more than one split/cohort")

    counts_by_consumer = {
        str(key): int(value)
        for key, value in view_frame["consumer_cohort_id"].value_counts().sort_index().items()
    }
    counts_by_scale = {
        str(key): int(value)
        for key, value in view_frame["scale_id"].value_counts().sort_index().items()
    }
    bytes_per_value = np.dtype(config.matrix_dtype).itemsize
    matrix_bytes_by_parent_scale = {
        str(scale_id): int(
            len(group)
            * int(group["parent_terminal_count"].iloc[0]) ** 2
            * bytes_per_value
            * config.stored_parent_matrix_count
        )
        for scale_id, group in family_frame.groupby("parent_scale_id", sort=True)
    }
    registry = {
        "schema": "cle_evrptw_generation_plan_v3",
        "dataset_id": config.dataset_id,
        "benchmark_version": config.benchmark_version,
        "master_seed": config.master_seed,
        "non_release_pilot": bool(non_release_pilot),
        "official_counts": pilot_families_per_city is None,
        "pilot_families_per_city": pilot_families_per_city,
        "family_count": len(family_frame),
        "view_count": len(view_frame),
        "family_counts_by_cohort": {
            str(key): int(value)
            for key, value in family_frame["family_cohort_id"].value_counts().sort_index().items()
        },
        "view_counts_by_consumer_cohort": counts_by_consumer,
        "view_counts_by_scale": counts_by_scale,
        "available_cities": sorted(available),
        "matrix_bundle_id": config.matrix_bundle_id,
        "stored_parent_matrix_count": config.stored_parent_matrix_count,
        "estimated_parent_matrix_bytes_by_scale": matrix_bytes_by_parent_scale,
        "estimated_parent_matrix_bytes_total": int(sum(matrix_bytes_by_parent_scale.values())),
    }
    return family_frame, view_frame, registry


def write_generation_plan(
    output_root: str | Path,
    family_frame: pd.DataFrame,
    view_frame: pd.DataFrame,
    registry: dict[str, Any],
) -> dict[str, Any]:
    root = Path(output_root)
    registry_path = root / "split_registry.json"
    if registry_path.exists():
        raise FileExistsError(f"Refusing to overwrite generation plan: {registry_path}")
    written: list[str] = []
    for cohort_id, families in family_frame.groupby("family_cohort_id", sort=True):
        cohort_dir = root / str(cohort_id)
        cohort_dir.mkdir(parents=True, exist_ok=True)
        path = cohort_dir / "family_index.parquet"
        families.reset_index(drop=True).to_parquet(path, index=False)
        written.append(str(path.relative_to(root)))
    for cohort_id, views in view_frame.groupby("consumer_cohort_id", sort=True):
        cohort_dir = root / str(cohort_id)
        cohort_dir.mkdir(parents=True, exist_ok=True)
        path = cohort_dir / "view_index.parquet"
        views.reset_index(drop=True).to_parquet(path, index=False)
        written.append(str(path.relative_to(root)))
    registry = dict(registry)
    registry["artifacts"] = sorted(written)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return registry
