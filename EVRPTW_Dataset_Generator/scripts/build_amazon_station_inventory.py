#!/usr/bin/env python3
"""Build the signed Stage-2 T0 Amazon station inventory without raw-data reparsing."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DAY_TYPES = ("weekday", "weekend")
SUPPORT_THRESHOLDS = (1_000, 2_000)


def _with_per_source_structure_counts(
    templates: pd.DataFrame, station_days: pd.DataFrame
) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for station_day_id, rows in templates.groupby("station_day_id", sort=True):
        values = rows["station_to_stop_time_s"].to_numpy(dtype=float)
        t_env = float(np.quantile(values, 0.99))
        counts[str(station_day_id)] = int((values <= t_env).sum())
    result = station_days.copy()
    result["structure_usable_stop_count"] = (
        result["station_day_id"].astype(str).map(counts).fillna(0).astype(int)
    )
    return result


def _support_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for day_type in DAY_TYPES:
        rows = frame.loc[frame["day_type"].eq(day_type)]
        result[day_type] = {
            "station_day_count": int(len(rows)),
            **{
                f"order_days_ge_{threshold}": int(
                    rows["order_usable_stop_count"].ge(threshold).sum()
                )
                for threshold in SUPPORT_THRESHOLDS
            },
            **{
                f"structure_days_ge_{threshold}": int(
                    rows["structure_usable_stop_count"].ge(threshold).sum()
                )
                for threshold in SUPPORT_THRESHOLDS
            },
        }
    return result


def _candidate_record(
    stations: tuple[str, ...],
    *,
    station_mass: dict[str, int],
    total_mass: int,
    station_days: pd.DataFrame,
) -> dict[str, Any]:
    rows = station_days.loc[station_days["station_code"].isin(stations)]
    mass = sum(station_mass[station] for station in stations)
    support = _support_counts(rows)
    minimum_cus2000_support = min(
        support[day_type][f"{source}_days_ge_2000"]
        for day_type in DAY_TYPES
        for source in ("order", "structure")
    )
    return {
        "station_codes": list(stations),
        "metro_codes": sorted({station[:3] for station in stations}),
        "usable_template_mass": int(mass),
        "support_mass_fraction": float(mass / total_mass),
        "absolute_deviation_from_15pct": float(abs(mass / total_mass - 0.15)),
        "minimum_day_type_source_cus2000_support": int(minimum_cus2000_support),
        "support": support,
    }


def _holdout_candidates(
    templates: pd.DataFrame,
    station_days: pd.DataFrame,
) -> dict[str, Any]:
    station_mass = {
        str(key): int(value)
        for key, value in templates.groupby("station_code", sort=True).size().items()
    }
    stations = sorted(station_mass)
    total_mass = int(len(templates))
    candidates: list[dict[str, Any]] = []
    for count in range(1, len(stations) + 1):
        for combination in itertools.combinations(stations, count):
            mass_fraction = sum(station_mass[item] for item in combination) / total_mass
            if not 0.145 <= mass_fraction <= 0.155:
                continue
            record = _candidate_record(
                combination,
                station_mass=station_mass,
                total_mass=total_mass,
                station_days=station_days,
            )
            if all(
                record["support"][day_type]["order_days_ge_1000"] > 0
                and record["support"][day_type]["structure_days_ge_1000"] > 0
                for day_type in DAY_TYPES
            ):
                candidates.append(record)

    robust = sorted(
        candidates,
        key=lambda item: (
            -item["minimum_day_type_source_cus2000_support"],
            item["absolute_deviation_from_15pct"],
            len(item["station_codes"]),
            item["station_codes"],
        ),
    )
    mass_exact = sorted(
        candidates,
        key=lambda item: (
            item["absolute_deviation_from_15pct"],
            -item["minimum_day_type_source_cus2000_support"],
            len(item["station_codes"]),
            item["station_codes"],
        ),
    )
    metro_diverse = sorted(
        [item for item in candidates if item["minimum_day_type_source_cus2000_support"] > 0],
        key=lambda item: (
            -len(item["metro_codes"]),
            -item["minimum_day_type_source_cus2000_support"],
            item["absolute_deviation_from_15pct"],
            len(item["station_codes"]),
            item["station_codes"],
        ),
    )
    return {
        "target_support_mass_fraction": 0.15,
        "candidate_search_window": [0.145, 0.155],
        "eligibility": (
            "whole stations; both day types must have order and structure support at N=1000"
        ),
        "ranking_notes": {
            "robust_cus2000": (
                "maximize the minimum order/structure, weekday/weekend N=2000 day count"
            ),
            "mass_exact": "minimize absolute support-mass deviation from 0.15",
            "metro_diverse": (
                "require N=2000 support, then maximize represented station-prefix metros "
                "before N=2000 robustness"
            ),
        },
        "robust_cus2000": robust[:10],
        "mass_exact": mass_exact[:10],
        "metro_diverse": metro_diverse[:10],
    }


def build_inventory(artifact_root: Path) -> dict[str, Any]:
    manifest = json.loads((artifact_root / "manifest.json").read_text(encoding="utf-8"))
    templates = pd.read_parquet(artifact_root / manifest["outputs"]["templates"])
    station_days = pd.read_parquet(artifact_root / manifest["outputs"]["station_days"])
    station_days = _with_per_source_structure_counts(templates, station_days)
    required_template_columns = {"station_code", "day_type"}
    required_day_columns = {
        "station_code",
        "day_type",
        "order_usable_stop_count",
        "structure_usable_stop_count",
    }
    if missing := required_template_columns - set(templates.columns):
        raise ValueError(f"Amazon templates lack T0 columns: {sorted(missing)}")
    if missing := required_day_columns - set(station_days.columns):
        raise ValueError(f"Amazon station days lack T0 columns: {sorted(missing)}")

    station_records: list[dict[str, Any]] = []
    for station_code, days in station_days.groupby("station_code", sort=True):
        station_templates = templates.loc[templates["station_code"].eq(station_code)]
        station_records.append(
            {
                "station_code": str(station_code),
                "metro_code": str(station_code)[:3],
                "station_day_counts": {
                    day_type: int(days["day_type"].eq(day_type).sum())
                    for day_type in DAY_TYPES
                },
                "usable_template_mass": int(len(station_templates)),
                "usable_template_mass_by_day_type": {
                    day_type: int(station_templates["day_type"].eq(day_type).sum())
                    for day_type in DAY_TYPES
                },
                "support": _support_counts(days),
            }
        )

    return {
        "schema": "evrptw_amazon_station_inventory_v1",
        "artifact_schema": manifest["schema"],
        "artifact_id": manifest["artifact_id"],
        "source_scope": manifest["source_scope"],
        "metro_definition": (
            "first three characters of station_code; the five frozen Amazon station markets"
        ),
        "unique_metro_count": len({record["metro_code"] for record in station_records}),
        "unique_station_code_count": len(station_records),
        "station_day_count": int(len(station_days)),
        "usable_template_mass": int(len(templates)),
        "stations": station_records,
        "metric_holdout_candidates": _holdout_candidates(templates, station_days),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_inventory(args.artifact_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
