from __future__ import annotations

import numpy as np
import pandas as pd

from EVRPTW_Benchmark.Reinforcement_Learning.common.support_selection import (
    DEFAULT_DESCRIPTOR_COLUMNS,
    select_parent_family_supports,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    metrics = []
    for index in range(40):
        family_id = f"f{index:03d}"
        rows.append(
            {
                "family_id": family_id,
                "view_id": f"v{index:03d}",
                "city_slug": f"city{index % 2}",
                "day_type": "weekday" if index % 4 < 2 else "weekend",
                "split_id": "train",
                "track_id": "train",
            }
        )
        record = {"family_id": family_id}
        for offset, column in enumerate(DEFAULT_DESCRIPTOR_COLUMNS[:6]):
            record[column] = float((index + 1) * (offset + 1) + np.sin(index + offset))
        metrics.append(record)
    return pd.DataFrame(rows), pd.DataFrame(metrics)


def test_support_selection_is_parent_family_deterministic_and_exact() -> None:
    index, metrics = _frames()
    first = select_parent_family_supports(index, metrics, fraction=0.10, seed=17)
    second = select_parent_family_supports(index, metrics, fraction=0.10, seed=17)
    assert first == second
    assert len(first.full_family_ids) == 40
    assert len(first.random_family_ids) == 4
    assert len(first.coverage_family_ids) == 4
    assert len(set(first.random_family_ids)) == 4
    assert len(set(first.coverage_family_ids)) == 4
    assert set(first.random_family_ids) <= set(first.full_family_ids)
    assert set(first.coverage_family_ids) <= set(first.full_family_ids)


def test_coverage_diagnostics_are_reported_for_registered_geometry() -> None:
    index, metrics = _frames()
    result = select_parent_family_supports(index, metrics, fraction=0.20, seed=3)
    assert result.manifest["coverage_mean_nearest_support_distance"] >= 0.0
    assert result.manifest["coverage_p95_nearest_support_distance"] >= 0.0
    assert result.manifest["random_mean_nearest_support_distance"] >= 0.0
    assert result.manifest["algorithm"] == "city_day_stratified_random_and_farthest_first_v1"
    assert result.manifest["selection_uses_validation_or_test"] is False
