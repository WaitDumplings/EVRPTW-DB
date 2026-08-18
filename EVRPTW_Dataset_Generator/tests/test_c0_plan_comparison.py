from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "compare_stage2_c0_plans.py"
SPEC = importlib.util.spec_from_file_location("compare_c0", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "stage2_runner", ROOT / "scripts" / "build_stage2_instances.py"
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


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


def test_reuse_frozen_customer_split_copies_approved_artifacts(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    source = baseline / "customer_splits" / "city"
    source.mkdir(parents=True)
    report = {"schema": "cle_evrptw_customer_split_report_v1", "city_slug": "city"}
    (source / "customer_split_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    payloads = {
        "customer_split_manifest.parquet": b"split",
        "community_manifest.parquet": b"communities",
        "community_adjacency.parquet": b"adjacency",
    }
    for name, payload in payloads.items():
        (source / name).write_bytes(payload)

    output = tmp_path / "candidate"
    reused = RUNNER._reuse_frozen_customer_split(baseline, output, "city")
    assert reused["frozen_split_reused"] is True
    assert reused["frozen_split_source"] == str(source.resolve())
    for name, payload in payloads.items():
        assert (output / "customer_splits" / "city" / name).read_bytes() == payload
