from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from evrptw_stage2.operational_acceptance import (
    build_spatial_diagnostic_v1,
    evaluate_matching_bias,
    preserve_q90_v1_failure,
)


def _q90_fail() -> dict[str, object]:
    return {
        "schema": "evrptw_station_block_q90_gate_v1",
        "release_calibrated": False,
        "rows": [
            {
                "metric_component": "M2",
                "generated_to_holdout_q90": 2.0,
                "real_to_real_q90": 1.0,
                "passed": False,
            },
            {
                "metric_component": "M3_P90",
                "generated_to_holdout_q90": 6.0,
                "real_to_real_q90": 1.0,
                "passed": False,
            },
        ],
    }


def test_q90_v1_is_preserved_byte_for_byte_and_not_reinterpreted(tmp_path: Path) -> None:
    source = tmp_path / "q90.json"
    destination = tmp_path / "q90_v1_original_fail.json"
    source.write_text(json.dumps(_q90_fail(), indent=1) + "\n", encoding="utf-8")
    preserve_q90_v1_failure(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    preserve_q90_v1_failure(source, destination)
    destination.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        preserve_q90_v1_failure(source, destination)


def test_spatial_diagnostic_retains_fail_but_has_no_acceptance_effect(tmp_path: Path) -> None:
    family_metrics = pd.DataFrame(
        {
            "city_slug": ["a"],
            "day_type": ["weekday"],
            "m5_community_concentration.proposal.active_community_count": [2],
            "m5_community_concentration.proposal.community_hhi": [0.6],
            "m5_community_concentration.radial_baseline.active_community_count": [4],
            "m5_community_concentration.radial_baseline.community_hhi": [0.3],
        }
    )
    report = build_spatial_diagnostic_v1(
        _q90_fail(),
        family_metrics,
        original_report_path=tmp_path / "old.json",
        preserved_report_path=tmp_path / "preserved.json",
    )
    assert report["hard_gate"] is False
    assert report["contributes_to_operational_acceptance"] is False
    assert report["historical_q90_v1"]["passed_row_count"] == 0
    assert report["historical_q90_v1"]["failed_row_count"] == 2
    assert report["historical_q90_v1"]["generated_to_real_ratio_max"] == 6.0
    assert report["construct_validity_review"]["threshold_changed"] is False


def test_matching_bias_uses_frozen_relative_and_tw_limits() -> None:
    policy = {
        "maximum_relative_difference": 0.10,
        "maximum_tw_presence_absolute_difference": 0.02,
        "relative_fields": ["demand_cm3_mean"],
        "absolute_fields": ["tw_presence_rate"],
    }
    frame = pd.DataFrame(
        {
            "family_id": ["f1"],
            "eligible_pool_count": [100],
            "matched_templates_count": [90],
            "eligible_pool_demand_cm3_mean": [100.0],
            "matched_templates_demand_cm3_mean": [109.0],
            "eligible_pool_tw_presence_rate": [0.10],
            "matched_templates_tw_presence_rate": [0.12],
        }
    )
    passed, rows = evaluate_matching_bias(frame, policy)
    assert passed
    assert rows[0]["passed"]
    frame.loc[0, "matched_templates_demand_cm3_mean"] = 111.0
    failed, rows = evaluate_matching_bias(frame, policy)
    assert not failed
    assert not rows[0]["passed"]
