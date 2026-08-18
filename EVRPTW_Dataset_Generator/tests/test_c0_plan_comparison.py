from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "compare_stage2_c0_plans.py"
SPEC = importlib.util.spec_from_file_location("compare_c0", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def _write_plan(root: Path, *, mutate_view: bool = False) -> None:
    split_rows = []
    family_rows = []
    view_rows = []
    for city_index in range(10):
        city = f"city-{city_index}"
        split_rows.append(
            {
                "city_slug": city,
                "latent_service_location_id": f"customer-{city_index}",
                "customer_pool": "train",
            }
        )
        for track in ("train", "validation"):
            for slot in range(7):
                family_id = f"{city}-{track}-{slot}"
                day_type = "weekday" if slot < 5 else "weekend"
                family_rows.append(
                    {
                        "family_id": family_id,
                        "city_slug": city,
                        "track_id": track,
                        "day_type": day_type,
                    }
                )
    for index in range(2_590):
        view_rows.append(
            {
                "view_id": f"view-{index}",
                "family_id": family_rows[index % len(family_rows)]["family_id"],
                "value": index + (1 if mutate_view and index == 0 else 0),
            }
        )
    split_path = root / "customer_splits" / "all" / "customer_split_manifest.parquet"
    family_path = root / "generation_plan" / "core" / "family_index.parquet"
    view_path = root / "generation_plan" / "core" / "view_index.parquet"
    split_path.parent.mkdir(parents=True)
    family_path.parent.mkdir(parents=True)
    pd.DataFrame(split_rows).to_parquet(split_path, index=False)
    pd.DataFrame(family_rows).to_parquet(family_path, index=False)
    pd.DataFrame(view_rows).to_parquet(view_path, index=False)


def test_c0_exact_comparison_accepts_exact_and_rejects_changed_view(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    exact = tmp_path / "exact"
    changed = tmp_path / "changed"
    _write_plan(baseline)
    _write_plan(exact)
    _write_plan(changed, mutate_view=True)
    accepted = COMPARE.compare_c0(baseline, exact)
    rejected = COMPARE.compare_c0(baseline, changed)
    assert accepted["passed"]
    assert all(accepted["fixed_counts"].values())
    assert not rejected["passed"]
    assert not rejected["view_registry"]["passed"]
