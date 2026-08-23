from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evrptw_stage2.acceptance import (
    build_metric_pairing_ledger,
    evaluate_q90_gate,
    station_block_bootstrap_q90,
    write_metric_pairing_ledger,
)
from evrptw_stage2.amazon import AmazonStage2Artifacts
from evrptw_stage2.profile import load_reference_profile
from evrptw_stage2.selection import (
    _resolve_charging_power,
    battery_feasible_communicating_mask,
    haversine_legacy_comparison_audit,
    road_time_replacement_deltas,
    select_road_time_charger_indices,
)
from evrptw_stage2.spatial_activation import _region_first_partition


ROOT = Path(__file__).parents[1]
PROFILE = ROOT / "configs" / "us_reference_instance_profile_v2.json"


def _synthetic_amazon() -> AmazonStage2Artifacts:
    templates = pd.DataFrame(
        {
            "template_id": [f"t{index}" for index in range(12)],
            "station_day_id": ["A:2020-01-01"] * 4
            + ["A:2020-01-02"] * 8,
            "station_code": ["A"] * 12,
            "day_type": ["weekday"] * 12,
            "route_id": ["r1"] * 4 + ["r2"] * 8,
            "station_to_stop_time_s": np.arange(1, 13, dtype=float),
            "within_spatial_envelope": [True] * 12,
            "radial_decile": np.arange(12) % 10,
        }
    )
    station_days = pd.DataFrame(
        {
            "station_day_id": ["A:2020-01-01", "A:2020-01-02"],
            "station_code": ["A", "A"],
            "date": ["2020-01-01", "2020-01-02"],
            "day_type": ["weekday", "weekday"],
            "structure_usable_stop_count": [4, 8],
            "order_usable_stop_count": [4, 8],
        }
    )
    return AmazonStage2Artifacts(
        root=Path("."),
        manifest={"t_env_s": 12.0, "radial_decile_edges_s": list(range(11))},
        templates=templates,
        structure_routes=pd.DataFrame(
            {
                "station_day_id": ["A:2020-01-01", "A:2020-01-02"],
                "route_id": ["r1", "r2"],
            }
        ),
        station_days=station_days,
        route_spatial_reference=pd.DataFrame(),
        cohort_split={"track_to_pool": {"train": "GEN-TRAIN"}},
        station_day_pool={
            "A:2020-01-01": "GEN-TRAIN",
            "A:2020-01-02": "GEN-TRAIN",
        },
    )


def test_v2_profile_rejects_v1_schema(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy_profile.json"
    legacy.write_text(json.dumps({"schema": "cle_evrptw_us_reference_instance_profile_v1"}))
    with pytest.raises(ValueError, match="Unsupported"):
        load_reference_profile(legacy)


def test_v2_profile_rejects_deleted_sampler_keys(tmp_path: Path) -> None:
    payload = json.loads(PROFILE.read_text(encoding="utf-8"))
    payload["packages"] = {}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="deleted V1 keys"):
        load_reference_profile(path)


def test_frozen_cohort_has_h3_and_all_leakage_assertions() -> None:
    payload = json.loads(
        (ROOT / "configs" / "amazon_cohort_split_v1.json").read_text(encoding="utf-8")
    )
    assert payload["metric_holdout"]["station_codes"] == ["DCH2", "DLA9", "DSE2"]
    assert all(payload["leakage_assertions"].values())
    allocation = payload["evaluation_track_allocation"]
    assert allocation["station_day_ledgers_pairwise_disjoint_and_exhaustive"]
    assert not allocation["exact_template_reuse_between_evaluation_tracks"]
    eval_rows = [row for row in payload["station_day_assignments"] if row["pool"] == "GEN-EVAL"]
    track_sets = {
        track: {
            row["station_day_id"]
            for row in eval_rows
            if row["generation_track"] == track
        }
        for track in allocation["tracks"]
    }
    assert all(track_sets.values())
    assert sum(map(len, track_sets.values())) == len(
        set().union(*track_sets.values())
    )


def test_structure_source_uses_its_own_envelope_and_edges() -> None:
    targets, report = _synthetic_amazon().structure_source(
        day_type="weekday",
        customer_count=4,
        seed=3,
        pool="GEN-TRAIN",
    )
    assert report["structure_source_mode"] == "SINGLE_STRUCTURE_DAY"
    assert len(report["source_radial_decile_edges_s"]) == 11
    assert report["source_radial_decile_edges_s"][-1] == pytest.approx(
        report["source_t_env_s"]
    )
    assert targets["station_to_stop_time_s"].le(report["source_t_env_s"]).all()


def test_primary_structure_source_cannot_downgrade_to_composite() -> None:
    with pytest.raises(ValueError, match="PRIMARY_SINGLE_STRUCTURE_DAY_UNSUPPORTED"):
        _synthetic_amazon().structure_source(
            day_type="weekday",
            customer_count=10,
            seed=3,
            pool="GEN-TRAIN",
            allow_composite=False,
        )


def test_one_way_energy_reachable_charger_is_not_eligible() -> None:
    energy = np.asarray(
        [[0.0, 5.0, 5.0], [50.0, 0.0, 5.0], [50.0, 50.0, 0.0]]
    )
    mask = battery_feasible_communicating_mask(energy, battery_capacity_kwh=10.0)
    assert mask.tolist() == [True, False, False]


def test_replacement_delta_is_clamped_nonnegative() -> None:
    times = np.asarray(
        [[0.0, 10.0, 2.0], [10.0, 0.0, 2.0], [2.0, 2.0, 0.0]]
    )
    delta = road_time_replacement_deltas(
        times,
        customer_indices=np.asarray([1]),
        charger_indices=np.asarray([2]),
    )
    assert delta.tolist() == [[0.0]]


def test_road_time_charger_selection_is_deterministic_and_not_prefix_bound() -> None:
    deltas = np.asarray(
        [[50.0, 40.0, 0.0, 100.0], [50.0, 40.0, 100.0, 0.0]]
    )
    left, _ = select_road_time_charger_indices(deltas, count=2, seed=9)
    right, _ = select_road_time_charger_indices(deltas, count=2, seed=9)
    np.testing.assert_array_equal(left, right)
    assert left.tolist() != [0, 1]


def test_haversine_legacy_comparison_is_report_only() -> None:
    customers = pd.DataFrame(
        {"location_lon": [0.0, 1.0], "location_lat": [0.0, 0.0]}
    )
    chargers = pd.DataFrame(
        {
            "resolved_longitude": [0.0, 1.0, 0.5],
            "resolved_latitude": [0.0, 0.0, 0.0],
        }
    )
    deltas = np.asarray([[10.0, 90.0, 20.0], [90.0, 10.0, 20.0]])
    selected, _ = select_road_time_charger_indices(deltas, count=1, seed=3)
    report = haversine_legacy_comparison_audit(
        customers,
        chargers,
        deltas,
        road_time_selected_positions=selected,
        count=1,
        seed=3,
    )
    assert report["role"].startswith("pilot_report_only")
    assert report["road_time_delta"][
        "objective_mean_plus_0.25_p90_plus_0.10_max_s"
    ] <= report["haversine_legacy_road_time_delta"][
        "objective_mean_plus_0.25_p90_plus_0.10_max_s"
    ]


def test_region_first_partition_is_exact_disjoint_and_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "latent_service_location_id": [f"x{index}" for index in range(12)],
            "sampling_cluster_id": ["large"] * 7 + ["small"] * 5,
            "activation_decile": np.arange(12) % 3,
            "community_id": np.arange(12).astype(str),
        }
    )
    first, report = _region_first_partition(
        frame,
        child_sizes=[6, 6],
        seed=11,
        namespace="test",
        community_graph=None,
    )
    second, _ = _region_first_partition(
        frame,
        child_sizes=[6, 6],
        seed=11,
        namespace="test",
        community_graph=None,
    )
    assert [len(child) for child in first] == [6, 6]
    assert set(first[0]["latent_service_location_id"]).isdisjoint(
        set(first[1]["latent_service_location_id"])
    )
    assert [child["latent_service_location_id"].tolist() for child in first] == [
        child["latent_service_location_id"].tolist() for child in second
    ]
    assert report["split_region_count"] >= 1


def test_missing_national_mode_median_is_hard_error() -> None:
    profile = load_reference_profile(PROFILE)
    profile["charging"]["national_mode_medians_kw"] = {}
    chargers = pd.DataFrame(
        {
            "reference_charge_mode": ["dc_fast"],
            "station_power_kw": [np.nan],
        }
    )
    with pytest.raises(ValueError, match="NATIONAL_MODE_MEDIAN_UNAVAILABLE"):
        _resolve_charging_power(
            chargers,
            profile=profile,
            generation_mode="non_release_pilot",
        )


def test_profile_national_medians_match_frozen_registry() -> None:
    profile = load_reference_profile(PROFILE)
    assert profile["charging"]["national_mode_medians_kw"] == {
        "ac_level2": 6.5,
        "dc_fast": 200.0,
    }


def _pairing_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    source_mode = "SINGLE_STRUCTURE_DAY|SINGLE_ORDER_DAY"
    generated = pd.DataFrame(
        {
            "day_type": ["weekday"],
            "scale_id": ["cus100"],
            "source_mode": [source_mode],
            "generated_view_id": ["v1"],
            "structure_source_id": ["g1"],
        }
    )
    holdout = pd.DataFrame(
        {
            "day_type": ["weekday", "weekday", "weekday"],
            "scale_id": ["cus100"] * 3,
            "source_mode": [source_mode] * 3,
            "holdout_station_day_id": ["h1", "h2", "h3"],
            "station_code": ["s1", "s2", "s3"],
        }
    )
    return generated, holdout


def test_metric_pairing_ledger_enumerates_all_pairs_and_is_byte_stable(
    tmp_path: Path,
) -> None:
    generated, holdout = _pairing_inputs()
    ledger = build_metric_pairing_ledger(generated, holdout, metric_components=["M2"])
    assert (ledger["pair_kind"] == "generated_to_holdout").sum() == 3
    assert (ledger["pair_kind"] == "real_to_real").sum() == 3
    first = tmp_path / "a.parquet"
    second = tmp_path / "b.parquet"
    write_metric_pairing_ledger(ledger, first)
    write_metric_pairing_ledger(ledger, second)
    assert first.read_bytes() == second.read_bytes()


def test_missing_primary_holdout_keeps_release_uncalibrated() -> None:
    source_mode = "SINGLE_STRUCTURE_DAY|SINGLE_ORDER_DAY"
    distances = pd.DataFrame(
        columns=[
            "day_type",
            "scale_id",
            "source_mode",
            "metric_component",
            "pair_kind",
            "distance",
        ]
    )
    result = evaluate_q90_gate(
        distances,
        required_primary_strata=[("weekday", "cus100", source_mode)],
    )
    assert not result["release_calibrated"]
    assert result["missing_primary_strata"]


def test_q90_gate_uses_generated_and_real_pair_distributions() -> None:
    source_mode = "SINGLE_STRUCTURE_DAY|SINGLE_ORDER_DAY"
    rows = []
    for kind, values in (
        ("generated_to_holdout", [0.1, 0.2, 0.3]),
        ("real_to_real", [0.2, 0.3, 0.4]),
    ):
        for value in values:
            rows.append(
                {
                    "day_type": "weekday",
                    "scale_id": "cus100",
                    "source_mode": source_mode,
                    "metric_component": "M2",
                    "pair_kind": kind,
                    "distance": value,
                }
            )
    result = evaluate_q90_gate(
        pd.DataFrame(rows),
        required_primary_strata=[("weekday", "cus100", source_mode)],
    )
    assert result["release_calibrated"]


def test_station_block_bootstrap_is_deterministic_and_report_only() -> None:
    rows = []
    for kind, station_block, values in (
        ("generated_to_holdout", "s1", [0.1, 0.2]),
        ("generated_to_holdout", "s2", [0.2, 0.3]),
        ("real_to_real", "s1|s2", [0.3, 0.4]),
    ):
        for value in values:
            rows.append(
                {
                    "day_type": "weekday",
                    "scale_id": "cus100",
                    "source_mode": "SINGLE_STRUCTURE_DAY|SINGLE_ORDER_DAY",
                    "metric_component": "M2",
                    "pair_kind": kind,
                    "station_block": station_block,
                    "distance": value,
                }
            )
    distances = pd.DataFrame(rows)
    first = station_block_bootstrap_q90(distances, replicates=25, seed=7)
    second = station_block_bootstrap_q90(distances, replicates=25, seed=7)
    assert first == second
    assert first["role"] == "report_only_not_used_by_release_gate"
