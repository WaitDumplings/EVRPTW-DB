#!/usr/bin/env python3
"""Freeze the V2.1 blocked Amazon station/station-day cohort split."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DAY_TYPES = ("weekday", "weekend")
POOLS = ("GEN-TRAIN", "GEN-EVAL", "METRIC-HOLDOUT")
PRIMARY_SCALES = (100, 500, 1_000)
SUPPORT_SCALES = (100, 500, 1_000, 2_000)
H3_STATIONS = ("DCH2", "DLA9", "DSE2")
EVALUATION_TRACKS = (
    "validation",
    "test1_new_seed",
    "test2_heldout_locations",
    "test3_heldout_city",
    "unseen_scale_same_cities",
)


def _with_per_source_structure_counts(
    templates: pd.DataFrame, station_days: pd.DataFrame
) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for station_day_id, rows in templates.groupby("station_day_id", sort=True):
        values = rows["station_to_stop_time_s"].to_numpy(dtype=float)
        t_env = float(np.quantile(values, 0.99))
        counts[str(station_day_id)] = int(np.count_nonzero(values <= t_env))
    result = station_days.copy()
    result["structure_usable_stop_count"] = (
        result["station_day_id"].astype(str).map(counts).fillna(0).astype(int)
    )
    return result


def _stable_u64(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _qualified_h3(
    templates: pd.DataFrame,
    station_days: pd.DataFrame,
) -> tuple[str, ...]:
    station_mass = templates.groupby("station_code", sort=True).size().to_dict()
    total_mass = len(templates)
    candidates: list[tuple[int, float, tuple[str, ...]]] = []
    for stations in itertools.combinations(sorted(station_mass), 3):
        mass_fraction = sum(int(station_mass[item]) for item in stations) / total_mass
        if not 0.145 <= mass_fraction <= 0.155:
            continue
        rows = station_days.loc[station_days["station_code"].isin(stations)]
        primary_supported = all(
            (
                rows.loc[rows["day_type"].eq(day_type), source_column]
                .ge(1_000)
                .any()
            )
            for day_type in DAY_TYPES
            for source_column in (
                "order_usable_stop_count",
                "structure_usable_stop_count",
            )
        )
        if not primary_supported:
            continue
        minimum_cus2000_support = min(
            int(
                rows.loc[rows["day_type"].eq(day_type), source_column]
                .ge(2_000)
                .sum()
            )
            for day_type in DAY_TYPES
            for source_column in (
                "order_usable_stop_count",
                "structure_usable_stop_count",
            )
        )
        candidates.append(
            (
                -minimum_cus2000_support,
                abs(mass_fraction - 0.15),
                stations,
            )
        )
    if not candidates:
        raise ValueError("No qualified three-station METRIC-HOLDOUT candidate exists")
    selected = min(candidates)[2]
    if selected != H3_STATIONS:
        raise AssertionError(
            f"Frozen H3 {H3_STATIONS} differs from deterministic search result {selected}"
        )
    return selected


def _closest_subset(
    rows: pd.DataFrame,
    *,
    target_mass: int,
    seed: int,
    day_type: str,
) -> set[str]:
    """Choose a deterministic near-target subset with a strong single-day anchor."""

    if rows.empty:
        raise ValueError(f"No generation station-days remain for {day_type}")
    work = rows.copy()
    work["_support"] = work[
        ["order_usable_stop_count", "structure_usable_stop_count"]
    ].min(axis=1)
    work["_rank"] = [
        _stable_u64(seed, "station_day", day_type, value)
        for value in work["station_day_id"].astype(str)
    ]
    anchor = work.sort_values(
        ["_support", "_rank", "station_day_id"],
        ascending=[False, True, True],
        kind="stable",
    ).iloc[0]
    anchor_id = str(anchor["station_day_id"])
    anchor_mass = int(anchor["order_usable_stop_count"])
    remaining_target = max(0, target_mass - anchor_mass)
    candidates = work.loc[~work["station_day_id"].astype(str).eq(anchor_id)].sort_values(
        ["_rank", "station_day_id"], kind="stable"
    )
    weights = candidates["order_usable_stop_count"].astype(int).to_numpy()
    identifiers = candidates["station_day_id"].astype(str).tolist()
    if not len(weights) or remaining_target == 0:
        return {anchor_id}

    cap = min(int(weights.sum()), remaining_target + int(weights.max()))
    reachable = np.zeros(cap + 1, dtype=bool)
    reachable[0] = True
    previous_sum = np.full(cap + 1, -1, dtype=np.int32)
    previous_item = np.full(cap + 1, -1, dtype=np.int32)
    for item_index, weight in enumerate(weights):
        weight = int(weight)
        if weight > cap:
            continue
        source_sums = np.flatnonzero(reachable[: cap - weight + 1])
        destination_sums = source_sums + weight
        fresh = ~reachable[destination_sums]
        if not fresh.any():
            continue
        sources = source_sums[fresh]
        destinations = destination_sums[fresh]
        reachable[destinations] = True
        previous_sum[destinations] = sources
        previous_item[destinations] = item_index

    reachable_sums = np.flatnonzero(reachable)
    best_sum = int(
        min(
            map(int, reachable_sums),
            key=lambda value: (abs(value - remaining_target), value),
        )
    )
    selected = {anchor_id}
    cursor = best_sum
    while cursor:
        item_index = int(previous_item[cursor])
        if item_index < 0:
            raise AssertionError("Broken deterministic subset reconstruction")
        selected.add(identifiers[item_index])
        cursor = int(previous_sum[cursor])
    return selected


def _pool_support(rows: pd.DataFrame, total_mass: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "station_day_count": int(len(rows)),
        "station_count": int(rows["station_code"].nunique()),
        "usable_template_mass": int(rows["order_usable_stop_count"].sum()),
        "support_mass_fraction": float(rows["order_usable_stop_count"].sum() / total_mass),
        "by_day_type": {},
    }
    for day_type in DAY_TYPES:
        day_rows = rows.loc[rows["day_type"].eq(day_type)]
        day_result: dict[str, Any] = {
            "station_day_count": int(len(day_rows)),
            "usable_template_mass": int(day_rows["order_usable_stop_count"].sum()),
        }
        for scale in SUPPORT_SCALES:
            day_result[f"single_order_days_ge_{scale}"] = int(
                day_rows["order_usable_stop_count"].ge(scale).sum()
            )
            day_result[f"single_structure_days_ge_{scale}"] = int(
                day_rows["structure_usable_stop_count"].ge(scale).sum()
            )
            station_totals = day_rows.groupby("station_code", sort=True)[
                ["order_usable_stop_count", "structure_usable_stop_count"]
            ].sum()
            day_result[f"same_station_order_composite_stations_ge_{scale}"] = int(
                station_totals["order_usable_stop_count"].ge(scale).sum()
            )
            day_result[f"same_station_structure_composite_stations_ge_{scale}"] = int(
                station_totals["structure_usable_stop_count"].ge(scale).sum()
            )
        result["by_day_type"][day_type] = day_result
    return result


def _disjoint(left: set[str], right: set[str]) -> bool:
    return not bool(left & right)


def _allocate_evaluation_tracks(
    rows: pd.DataFrame,
    *,
    seed: int,
    total_mass: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Partition GEN-EVAL station-days into deterministic track ledgers."""

    assignments: dict[str, str] = {}
    for day_type in DAY_TYPES:
        day_rows = rows.loc[rows["day_type"].eq(day_type)].copy()
        day_rows["_joint_support"] = day_rows[
            ["order_usable_stop_count", "structure_usable_stop_count"]
        ].min(axis=1)
        # Reserve the hardest scalability support first; remaining primary
        # tracks require a true single day at N=1000.
        for track in ("unseen_scale_same_cities", *EVALUATION_TRACKS[:-1]):
            minimum = 2_000 if track == "unseen_scale_same_cities" else 1_000
            candidates = day_rows.loc[
                ~day_rows["station_day_id"].astype(str).isin(assignments)
                & day_rows["_joint_support"].ge(minimum)
            ].copy()
            if candidates.empty:
                raise ValueError(
                    f"GEN-EVAL cannot reserve {day_type}/{track} support at N={minimum}"
                )
            candidates["_rank"] = [
                _stable_u64(seed, "evaluation_track_anchor", day_type, track, value)
                for value in candidates["station_day_id"].astype(str)
            ]
            chosen = candidates.sort_values(
                ["_joint_support", "_rank", "station_day_id"],
                ascending=[False, True, True],
                kind="stable",
            ).iloc[0]
            assignments[str(chosen["station_day_id"])] = track

        remaining = day_rows.loc[
            ~day_rows["station_day_id"].astype(str).isin(assignments)
        ].copy()
        remaining["_rank"] = [
            _stable_u64(seed, "evaluation_track_fill", day_type, value)
            for value in remaining["station_day_id"].astype(str)
        ]
        track_mass = {
            track: int(
                day_rows.loc[
                    day_rows["station_day_id"].astype(str).map(assignments).eq(track),
                    "order_usable_stop_count",
                ].sum()
            )
            for track in EVALUATION_TRACKS
        }
        for row in remaining.sort_values(["_rank", "station_day_id"], kind="stable").itertuples():
            track = min(
                EVALUATION_TRACKS,
                key=lambda value: (
                    track_mass[value],
                    _stable_u64(seed, "evaluation_track_tie", day_type, row.station_day_id, value),
                    value,
                ),
            )
            assignments[str(row.station_day_id)] = track
            track_mass[track] += int(row.order_usable_stop_count)

    track_sets = {
        track: {station_day_id for station_day_id, value in assignments.items() if value == track}
        for track in EVALUATION_TRACKS
    }
    all_ids = set(rows["station_day_id"].astype(str))
    disjoint_exhaustive = (
        all(
            _disjoint(track_sets[left], track_sets[right])
            for left, right in itertools.combinations(EVALUATION_TRACKS, 2)
        )
        and set.union(*track_sets.values()) == all_ids
    )
    if not disjoint_exhaustive:
        raise AssertionError("GEN-EVAL track station-day ledgers are not a partition")
    support = {
        track: _pool_support(
            rows.loc[rows["station_day_id"].astype(str).isin(track_sets[track])],
            total_mass,
        )
        for track in EVALUATION_TRACKS
    }
    for track in EVALUATION_TRACKS:
        for day_type in DAY_TYPES:
            day_support = support[track]["by_day_type"][day_type]
            minimum = 2_000 if track == "unseen_scale_same_cities" else 1_000
            if not day_support[f"single_order_days_ge_{minimum}"]:
                raise ValueError(f"{track}/{day_type} lacks order support at N={minimum}")
            if not day_support[f"single_structure_days_ge_{minimum}"]:
                raise ValueError(f"{track}/{day_type} lacks structure support at N={minimum}")
    return assignments, {
        "policy": "deterministic_disjoint_station_day_mass_balance_with_support_anchors_v1",
        "tracks": list(EVALUATION_TRACKS),
        "station_day_ledgers_pairwise_disjoint_and_exhaustive": disjoint_exhaustive,
        "exact_template_reuse_between_evaluation_tracks": False,
        "support": support,
    }


def build_split(artifact_root: Path, *, seed: int) -> dict[str, Any]:
    manifest = json.loads((artifact_root / "manifest.json").read_text(encoding="utf-8"))
    templates = pd.read_parquet(artifact_root / manifest["outputs"]["templates"])
    station_days = pd.read_parquet(artifact_root / manifest["outputs"]["station_days"])
    station_days = _with_per_source_structure_counts(templates, station_days)
    _qualified_h3(templates, station_days)

    holdout_mask = station_days["station_code"].isin(H3_STATIONS)
    holdout_ids = set(station_days.loc[holdout_mask, "station_day_id"].astype(str))
    generation_days = station_days.loc[~holdout_mask].copy()
    eval_ids: set[str] = set()
    target_eval_mass_by_day_type: dict[str, int] = {}
    for day_type in DAY_TYPES:
        total_day_mass = int(
            station_days.loc[station_days["day_type"].eq(day_type), "order_usable_stop_count"].sum()
        )
        target = int(round(0.15 * total_day_mass))
        target_eval_mass_by_day_type[day_type] = target
        eval_ids.update(
            _closest_subset(
                generation_days.loc[generation_days["day_type"].eq(day_type)],
                target_mass=target,
                seed=seed,
                day_type=day_type,
            )
        )
    all_ids = set(station_days["station_day_id"].astype(str))
    train_ids = all_ids - holdout_ids - eval_ids

    assignments = station_days.copy()
    assignments["pool"] = "GEN-TRAIN"
    assignments.loc[assignments["station_day_id"].isin(eval_ids), "pool"] = "GEN-EVAL"
    assignments.loc[assignments["station_day_id"].isin(holdout_ids), "pool"] = "METRIC-HOLDOUT"
    assignments = assignments.sort_values("station_day_id", kind="stable")

    day_sets = {
        "GEN-TRAIN": train_ids,
        "GEN-EVAL": eval_ids,
        "METRIC-HOLDOUT": holdout_ids,
    }
    template_sets = {
        pool: set(
            templates.loc[templates["station_day_id"].astype(str).isin(ids), "template_id"].astype(str)
        )
        for pool, ids in day_sets.items()
    }
    route_sets = {
        pool: set(
            templates.loc[templates["station_day_id"].astype(str).isin(ids), "route_id"].astype(str)
        )
        for pool, ids in day_sets.items()
    }
    generation_stations = set(
        assignments.loc[~assignments["pool"].eq("METRIC-HOLDOUT"), "station_code"].astype(str)
    )
    assertions = {
        "metric_stations_disjoint_from_generation": _disjoint(
            set(H3_STATIONS), generation_stations
        ),
        "station_day_pools_pairwise_disjoint_and_exhaustive": (
            all(_disjoint(day_sets[left], day_sets[right]) for left, right in itertools.combinations(POOLS, 2))
            and set.union(*day_sets.values()) == all_ids
        ),
        "template_id_pools_pairwise_disjoint": all(
            _disjoint(template_sets[left], template_sets[right])
            for left, right in itertools.combinations(POOLS, 2)
        ),
        "route_id_pools_pairwise_disjoint": all(
            _disjoint(route_sets[left], route_sets[right])
            for left, right in itertools.combinations(POOLS, 2)
        ),
    }
    if not all(assertions.values()):
        raise AssertionError(f"Amazon cohort leakage assertion failed: {assertions}")

    support = {
        pool: _pool_support(assignments.loc[assignments["pool"].eq(pool)], len(templates))
        for pool in POOLS
    }
    for pool in POOLS:
        for day_type in DAY_TYPES:
            day_support = support[pool]["by_day_type"][day_type]
            for scale in PRIMARY_SCALES:
                if not day_support[f"single_order_days_ge_{scale}"]:
                    raise ValueError(f"{pool}/{day_type} lacks single-order support at N={scale}")
                if not day_support[f"single_structure_days_ge_{scale}"]:
                    raise ValueError(f"{pool}/{day_type} lacks single-structure support at N={scale}")

    eval_track_by_day, eval_track_report = _allocate_evaluation_tracks(
        assignments.loc[assignments["pool"].eq("GEN-EVAL")],
        seed=seed,
        total_mass=len(templates),
    )

    records = []
    for row in assignments.itertuples(index=False):
        records.append(
            {
                "station_day_id": str(row.station_day_id),
                "station_code": str(row.station_code),
                "date": str(row.date),
                "day_type": str(row.day_type),
                "pool": str(row.pool),
                "generation_track": (
                    "train"
                    if str(row.pool) == "GEN-TRAIN"
                    else (
                        "metric_only"
                        if str(row.pool) == "METRIC-HOLDOUT"
                        else eval_track_by_day[str(row.station_day_id)]
                    )
                ),
                "order_usable_template_mass": int(row.order_usable_stop_count),
                "structure_usable_template_mass": int(row.structure_usable_stop_count),
            }
        )
    return {
        "schema": "evrptw_amazon_cohort_split_v1",
        "directive": "STAGE2_REPAIR_DIRECTIVE_V2.1_FINAL",
        "source_artifact_schema": manifest["schema"],
        "source_artifact_id": manifest["artifact_id"],
        "frozen_seed": int(seed),
        "target_support_mass_fraction": {
            "GEN-TRAIN": 0.70,
            "GEN-EVAL": 0.15,
            "METRIC-HOLDOUT": 0.15,
        },
        "target_gen_eval_mass_by_day_type": target_eval_mass_by_day_type,
        "metric_holdout": {
            "selection_id": "H3",
            "station_codes": list(H3_STATIONS),
            "selection_policy": (
                "three_station_maximin_cus2000_support_then_mass_deviation_v1"
            ),
            "whole_station_isolation": True,
        },
        "track_to_pool": {
            "train": "GEN-TRAIN",
            "validation": "GEN-EVAL",
            "test1_new_seed": "GEN-EVAL",
            "test2_heldout_locations": "GEN-EVAL",
            "test3_heldout_city": "GEN-EVAL",
            "unseen_scale_same_cities": "GEN-EVAL",
        },
        "evaluation_track_allocation": eval_track_report,
        "source_mode_contract": {
            "primary_scales": ["cus100", "cus500", "cus1000"],
            "primary_structure_source_mode": "SINGLE_STRUCTURE_DAY",
            "primary_order_source_mode": "SINGLE_ORDER_DAY",
            "composite_allowed_scales": ["cus2000"],
            "composite_release_role": "report_only",
            "retry_may_change_source_id": True,
            "retry_may_change_source_mode": False,
        },
        "leakage_assertions": assertions,
        "support": support,
        "station_day_assignments": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    payload = build_split(args.artifact_root, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), **payload["support"]}, indent=2))


if __name__ == "__main__":
    main()
