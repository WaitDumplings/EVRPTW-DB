#!/usr/bin/env python3
"""Compare a fresh Stage-2 C0 plan against the frozen approved baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_parts(root: Path, name: str) -> pd.DataFrame:
    paths = sorted(root.rglob(name))
    if not paths:
        raise FileNotFoundError(f"No {name} under {root}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _frame_contract(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    key_columns: list[str],
) -> dict[str, Any]:
    missing_keys = (set(key_columns) - set(baseline.columns)) | (
        set(key_columns) - set(candidate.columns)
    )
    if missing_keys:
        raise ValueError(f"Comparison keys are missing: {sorted(missing_keys)}")
    baseline_sorted = baseline.sort_values(key_columns).reset_index(drop=True)
    candidate_sorted = candidate.sort_values(key_columns).reset_index(drop=True)
    columns_equal = list(baseline_sorted.columns) == list(candidate_sorted.columns)
    duplicate_baseline = bool(baseline_sorted.duplicated(key_columns).any())
    duplicate_candidate = bool(candidate_sorted.duplicated(key_columns).any())
    exact = False
    mismatch = None
    if columns_equal and not duplicate_baseline and not duplicate_candidate:
        try:
            pd.testing.assert_frame_equal(
                baseline_sorted,
                candidate_sorted,
                check_dtype=False,
                check_exact=True,
            )
            exact = True
        except AssertionError as error:
            mismatch = str(error)[:2_000]
    return {
        "passed": exact,
        "baseline_row_count": len(baseline_sorted),
        "candidate_row_count": len(candidate_sorted),
        "columns_equal": columns_equal,
        "baseline_duplicate_key": duplicate_baseline,
        "candidate_duplicate_key": duplicate_candidate,
        "mismatch": mismatch,
    }


def compare_c0(baseline_root: Path, candidate_root: Path) -> dict[str, Any]:
    baseline_splits = _read_parts(baseline_root / "customer_splits", "customer_split_manifest.parquet")
    candidate_splits = _read_parts(candidate_root / "customer_splits", "customer_split_manifest.parquet")
    split = _frame_contract(
        baseline_splits,
        candidate_splits,
        key_columns=["city_slug", "latent_service_location_id"],
    )

    baseline_families = _read_parts(baseline_root / "generation_plan", "family_index.parquet")
    candidate_families = _read_parts(candidate_root / "generation_plan", "family_index.parquet")
    families = _frame_contract(
        baseline_families,
        candidate_families,
        key_columns=["family_id"],
    )

    baseline_views = _read_parts(baseline_root / "generation_plan", "view_index.parquet")
    candidate_views = _read_parts(candidate_root / "generation_plan", "view_index.parquet")
    views = _frame_contract(
        baseline_views,
        candidate_views,
        key_columns=["view_id"],
    )

    slot_rows = []
    for (city, track), rows in candidate_families.groupby(
        ["city_slug", "track_id"], sort=True
    ):
        counts = rows["day_type"].astype(str).value_counts()
        slot_rows.append(
            {
                "city_slug": str(city),
                "track_id": str(track),
                "family_count": len(rows),
                "weekday_count": int(counts.get("weekday", 0)),
                "weekend_count": int(counts.get("weekend", 0)),
                "passed": (
                    len(rows) == 7
                    and int(counts.get("weekday", 0)) == 5
                    and int(counts.get("weekend", 0)) == 2
                ),
            }
        )
    fixed_counts = {
        "ten_city_split_membership": (
            baseline_splits["city_slug"].nunique()
            == candidate_splits["city_slug"].nunique()
            == 10
            and split["passed"]
        ),
        "family_count_140": len(baseline_families) == len(candidate_families) == 140,
        "view_count_2590": len(baseline_views) == len(candidate_views) == 2_590,
        "twenty_city_track_slots_are_5_to_2": (
            len(slot_rows) == 20 and all(row["passed"] for row in slot_rows)
        ),
    }
    passed = (
        split["passed"]
        and families["passed"]
        and views["passed"]
        and all(fixed_counts.values())
    )
    return {
        "schema": "cle_evrptw_stage2_c0_exact_comparison_v1",
        "passed": passed,
        "baseline_root": str(baseline_root),
        "candidate_root": str(candidate_root),
        "split_membership_and_fields": split,
        "family_registry": families,
        "view_registry": views,
        "fixed_counts": fixed_counts,
        "slot_ledger_5_to_2": slot_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_c0(args.baseline_root, args.candidate_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
