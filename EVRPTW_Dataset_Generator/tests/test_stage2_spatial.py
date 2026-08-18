from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evrptw_stage2.amazon import (
    build_amazon_stage2_artifacts,
    load_amazon_stage2_artifacts,
)
from evrptw_stage2.metrics import aggregate_phase1_metrics
from evrptw_stage2.orders import (
    build_view_attributes_from_amazon,
    match_amazon_order_templates,
)
from evrptw_stage2.profile import load_reference_profile
from evrptw_stage2.rounding import controlled_matrix_round
from evrptw_stage2.spatial_activation import (
    SpatialActivationError,
    _grow_regions,
    _grow_regions_reference,
    activate_spatial_customers,
    nested_customer_order,
)

PROFILE = Path(__file__).parents[1] / "configs" / "us_reference_instance_profile_v2.json"


def test_amazon_preprocessor_builds_station_day_artifacts(tmp_path: Path) -> None:
    route_id = "RouteID_1"
    route_data = {
        route_id: {
            "station_code": "DLA3",
            "date_YYYY_MM_DD": "2018-01-02",
            "stops": {
                "station": {"type": "Station"},
                "a": {"type": "Dropoff"},
                "b": {"type": "Dropoff"},
            },
        }
    }
    package = {
        "scan_status": "DELIVERED",
        "dimensions": {"depth_cm": 10, "height_cm": 10, "width_cm": 10},
        "planned_service_time_seconds": 30,
        "time_window": {"start_time_utc": None, "end_time_utc": None},
    }
    package_data = {route_id: {"a": {"p1": package}, "b": {"p2": package}}}
    travel = {
        route_id: {
            "station": {"station": 0, "a": 100, "b": 100},
            "a": {"station": 100, "a": 0, "b": 50},
            "b": {"station": 100, "a": 60, "b": 0},
        }
    }
    for name, payload in (
        ("route_data.json", route_data),
        ("package_data.json", package_data),
        ("travel_times.json", travel),
    ):
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "artifacts"
    manifest = build_amazon_stage2_artifacts(
        route_data_path=tmp_path / "route_data.json",
        package_data_path=tmp_path / "package_data.json",
        travel_times_path=tmp_path / "travel_times.json",
        output_dir=output,
    )
    cohort = tmp_path / "cohort.json"
    cohort.write_text(
        json.dumps(
            {
                "schema": "evrptw_amazon_cohort_split_v1",
                "leakage_assertions": {
                    "metric_stations_disjoint_from_generation": True,
                    "station_day_pools_pairwise_disjoint_and_exhaustive": True,
                    "template_id_pools_pairwise_disjoint": True,
                    "route_id_pools_pairwise_disjoint": True,
                },
                    "track_to_pool": {"train": "GEN-TRAIN"},
                    "evaluation_track_allocation": {
                        "tracks": [],
                        "station_day_ledgers_pairwise_disjoint_and_exhaustive": True,
                        "exact_template_reuse_between_evaluation_tracks": False,
                    },
                    "station_day_assignments": [
                        {"station_day_id": "DLA3:2018-01-02", "pool": "GEN-TRAIN"}
                    ],
                }
            ),
        encoding="utf-8",
    )
    artifacts = load_amazon_stage2_artifacts(output, cohort_split_path=cohort)
    assert manifest["template_count"] == 2
    assert manifest["template_id_contract"] == "station_code:date:route_id:stop_id"
    assert artifacts.templates["template_id"].str.startswith(
        "DLA3:2018-01-02:RouteID_1:"
    ).all()
    assert manifest["source_license"] == "CC-BY-NC-4.0"
    assert manifest["source_files"]["route_data"] == (
        "almrrc2021-data-training/model_build_inputs/route_data.json"
    )
    assert str(tmp_path) not in json.dumps(manifest["source_files"])
    assert artifacts.t_env_s > 0
    targets, source = artifacts.structure_source(
        day_type="weekday", customer_count=1, seed=3, pool="GEN-TRAIN"
    )
    assert len(targets) >= 1
    assert source["structure_source_mode"] == "SINGLE_STRUCTURE_DAY"


def test_controlled_rounding_preserves_both_margins() -> None:
    fractional = np.asarray([[1.2, 1.8], [0.8, 2.2]])
    result = controlled_matrix_round(
        fractional,
        row_targets=np.asarray([3, 3]),
        column_targets=np.asarray([2, 4]),
        seed=7,
        namespace="test",
        row_labels=["a", "b"],
        column_labels=["x", "y"],
    )
    np.testing.assert_array_equal(result.sum(axis=1), [3, 3])
    np.testing.assert_array_equal(result.sum(axis=0), [2, 4])


def test_spatial_activation_is_exact_unique_and_quota_safe() -> None:
    records = []
    for community_index in range(4):
        for decile in range(2):
            for local in range(6):
                records.append(
                    {
                        "latent_service_location_id": (
                            f"c{community_index}-d{decile}-n{local}"
                        ),
                        "community_id": f"c{community_index}",
                        "residential_units": local + 1,
                        "radial_decile": decile,
                        "depot_running_time_s": 300.0 + 600.0 * decile + local,
                    }
                )
    customers = pd.DataFrame.from_records(records)
    adjacency = pd.DataFrame(
        [
            {
                "source_community_id": f"c{left}",
                "target_community_id": f"c{right}",
                "crossing_time_s": 10.0,
            }
            for left, right in [
                (0, 1),
                (1, 0),
                (1, "transit"),
                ("transit", 1),
                ("transit", 2),
                (2, "transit"),
                (2, 3),
                (3, 2),
            ]
        ]
    )
    structure = pd.DataFrame(
        [
            {
                "template_id": f"t{index}",
                "route_id": "r0" if index < 6 else "r1",
                "radial_decile": index % 2,
                "station_to_stop_time_s": 300.0 + 600.0 * (index % 2),
            }
            for index in range(12)
        ]
    )
    progress_events: list[tuple[str, dict[str, object]]] = []
    result = activate_spatial_customers(
        customers,
        adjacency,
        structure,
        customer_count=12,
        seed=123,
        region_redraw_cap=1,
        progress_callback=lambda stage, details: progress_events.append(
            (stage, dict(details))
        ),
    )
    assert len(result.customers) == 12
    assert not result.customers["latent_service_location_id"].duplicated().any()
    assert result.metadata["quota"]["row_margins_exact"]
    assert result.metadata["quota"]["column_margins_exact"]
    assert result.metadata["global_customer_uniqueness"]
    completed = {
        stage
        for stage, details in progress_events
        if details.get("status") == "completed"
    }
    assert {
        "quota_matrix",
        "community_graph",
        "region_seed_selection",
        "region_growth",
        "global_customer_assignment",
        "nested_customer_order",
        "selected_customer_join",
        "radial_baseline",
    } <= completed
    assert {
        "global_assignment.graph_build",
        "global_assignment.min_cost_flow",
        "global_assignment.result_extract",
    } <= completed
    assert result.metadata["performance_profile"]
    assert all(
        event["wall_seconds"] >= 0.0
        and event["cpu_seconds"] >= 0.0
        and event["peak_rss_bytes"] > 0
        for event in result.metadata["performance_profile"]
    )
    without_callback = activate_spatial_customers(
        customers,
        adjacency,
        structure,
        customer_count=12,
        seed=123,
        region_redraw_cap=1,
    )
    pd.testing.assert_frame_equal(result.customers, without_callback.customers)
    pd.testing.assert_frame_equal(result.assignment, without_callback.assignment)
    pd.testing.assert_frame_equal(result.radial_baseline, without_callback.radial_baseline)
    profiled_metadata = dict(result.metadata)
    unprofiled_metadata = dict(without_callback.metadata)
    profiled_metadata.pop("performance_profile")
    unprofiled_metadata.pop("performance_profile")
    assert profiled_metadata == unprofiled_metadata


def test_nested_order_encodes_exact_cus1000_tree() -> None:
    assignment = pd.DataFrame(
        {
            "latent_service_location_id": [f"loc-{index}" for index in range(1000)],
            "sampling_cluster_id": [f"r{index % 8}" for index in range(1000)],
            "activation_decile": [index % 10 for index in range(1000)],
            "community_id": [f"c{index % 16}" for index in range(1000)],
        }
    )
    ordered, report = nested_customer_order(
        assignment,
        customer_count=1000,
        seed=99,
    )
    assert len(ordered) == 1000
    assert not ordered["latent_service_location_id"].duplicated().any()
    assert report["cus500_nodes"] == 2
    assert report["cus100_nodes"] == 10
    assert report["cus50_nodes"] == 20


def _growth_outcome(
    implementation: object,
    customers: pd.DataFrame,
    quotas: pd.DataFrame,
    graph: object,
    seeds: dict[str, str],
    *,
    seed: int,
) -> tuple[object, ...]:
    trace: list[dict[str, object]] = []
    progress: list[tuple[str, dict[str, object]]] = []
    try:
        regions, steps = implementation(  # type: ignore[operator]
            customers,
            quotas,
            graph,
            seeds,
            seed=seed,
            decision_trace=trace,
            progress_callback=lambda stage, details: progress.append(
                (stage, dict(details))
            ),
        )
    except SpatialActivationError as error:
        return "error", error.code, error.diagnostics, trace, progress
    return "ok", regions, steps, trace, progress


@pytest.mark.parametrize("scenario_seed", range(30))
def test_exact_growth_cache_matches_reference_on_random_directed_graphs(
    scenario_seed: int,
) -> None:
    import networkx as nx

    rng = np.random.default_rng(scenario_seed)
    communities = [f"c{index}" for index in range(9)]
    graph = nx.DiGraph()
    graph.add_nodes_from(communities)
    for index, source in enumerate(communities):
        target = communities[(index + 1) % len(communities)]
        graph.add_edge(source, target, weight=float(rng.integers(1, 5)))
        if bool(rng.integers(2)):
            graph.add_edge(target, source, weight=float(rng.integers(1, 5)))
    for _ in range(12):
        source, target = rng.choice(communities, size=2, replace=False)
        graph.add_edge(str(source), str(target), weight=float(rng.integers(1, 5)))

    records: list[dict[str, object]] = []
    decile_totals = {decile: 0 for decile in range(4)}
    for community in communities[:-1]:
        for decile in range(4):
            count = int(rng.integers(4))
            decile_totals[decile] += count
            records.extend(
                {"community_id": community, "radial_decile": decile}
                for _ in range(count)
            )
    for decile, total in decile_totals.items():
        if total == 0:
            records.append({"community_id": communities[0], "radial_decile": decile})
            decile_totals[decile] = 1
    customers = pd.DataFrame.from_records(records)
    quotas = pd.DataFrame.from_records(
        [
            {
                "region_id": region,
                "radial_decile": decile,
                "quota": max(1, decile_totals[decile] // divisor),
            }
            for region, divisor in (("r0", 2), ("r1", 3))
            for decile in range(4)
        ]
    )
    seeds = {"r0": communities[0], "r1": communities[4]}
    reference = _growth_outcome(
        _grow_regions_reference,
        customers,
        quotas,
        graph,
        seeds,
        seed=scenario_seed,
    )
    optimized = _growth_outcome(
        _grow_regions,
        customers,
        quotas,
        graph,
        seeds,
        seed=scenario_seed,
    )
    assert optimized == reference


def test_exact_growth_cache_matches_reference_for_transit_ties_and_failure() -> None:
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_weighted_edges_from(
        [
            ("seed", "transit-a", 2.0),
            ("seed", "transit-b", 2.0),
            ("transit-a", "customer", 1.0),
            ("customer", "transit-b", 1.0),
        ],
        weight="weight",
    )
    customers = pd.DataFrame(
        [
            {"community_id": "seed", "radial_decile": 0},
            {"community_id": "customer", "radial_decile": 1},
        ]
    )
    quotas = pd.DataFrame(
        [{"region_id": "r0", "radial_decile": 1, "quota": 1}]
    )
    seeds = {"r0": "seed"}
    assert _growth_outcome(
        _grow_regions,
        customers,
        quotas,
        graph,
        seeds,
        seed=77,
    ) == _growth_outcome(
        _grow_regions_reference,
        customers,
        quotas,
        graph,
        seeds,
        seed=77,
    )

    disconnected = nx.DiGraph()
    disconnected.add_nodes_from(["seed", "customer"])
    optimized_error = _growth_outcome(
        _grow_regions,
        customers,
        quotas,
        disconnected,
        seeds,
        seed=77,
    )
    reference_error = _growth_outcome(
        _grow_regions_reference,
        customers,
        quotas,
        disconnected,
        seeds,
        seed=77,
    )
    assert optimized_error == reference_error
    assert optimized_error[:3] == (
        "error",
        "REGION_GROWTH_EXHAUSTED",
        {"unmet_region_deciles": {"r0": [1]}, "growth_steps": 0},
    )


def test_amazon_covering_matching_and_inherited_attributes() -> None:
    profile = load_reference_profile(PROFILE)
    customer_count = 3
    terminal_count = 5  # depot + 3 customers + 1 CS
    running_time = np.full((terminal_count, terminal_count), 60.0, dtype=np.float32)
    np.fill_diagonal(running_time, 0.0)
    running_distance = np.full((terminal_count, terminal_count), 1.0, dtype=np.float32)
    np.fill_diagonal(running_distance, 0.0)
    templates = pd.DataFrame(
        {
            "template_id": ["t0", "t1", "t2"],
            "station_day_id": ["DLA3:2018-01-01"] * 3,
            "package_count": [1, 2, 3],
            "demand_cm3": [1000.0, 2000.0, 3000.0],
            "service_time_s": [30.0, 40.0, 50.0],
            "tw_start_s": [28800.0] * 3,
            "tw_end_s": [86400.0] * 3,
            "tw_was_specified": [False] * 3,
        }
    )
    source = {
        "order_source_mode": "SINGLE_ORDER_DAY",
        "station_code": "DLA3",
        "station_day_ids": ["DLA3:2018-01-01"],
    }
    matched, match_report = match_amazon_order_templates(
        customer_count=customer_count,
        order_sources=[(source, templates)],
        matching_seed=11,
        operating_start_s=28800,
        operating_end_s=86400,
        running_time_matrix_s=running_time,
        running_time_path_distance_matrix_km=running_distance,
        charging_power_kw=np.asarray([100.0], dtype=np.float32),
        profile=profile,
    )
    attributes = build_view_attributes_from_amazon(
        pd.DataFrame({"source_id": ["a", "b", "c"]}),
        matched,
        day_type="weekday",
        operating_start_s=28800,
        operating_end_s=86400,
        running_time_matrix_s=running_time,
        running_time_path_distance_matrix_km=running_distance,
        charging_power_kw=np.asarray([100.0], dtype=np.float32),
        profile=profile,
        order_source_report=match_report,
    )
    assert match_report["hall_coverage_complete"]
    np.testing.assert_array_equal(np.sort(attributes.package_counts), [1, 2, 3])
    assert attributes.report["order_template_inheritance"]


def test_phase1_aggregation_persists_family_stratum_and_summary(tmp_path: Path) -> None:
    family = tmp_path / "materialized" / "families" / "family-1"
    family.mkdir(parents=True)
    metrics = {
        "schema": "evrptw_phase1_family_metrics_v2",
        "family_id": "family-1",
        "city_slug": "san-diego",
        "day_type": "weekday",
        "parent_scale_id": "cus1000",
        "materialization_attempt_number": 0,
        "hard_gates": {"passed": True},
        "m1_radial": {
            "proposal_family_normalized_w1": 0.1,
            "radial_baseline_family_normalized_w1": 0.2,
            "proposal_not_worse_than_radial_baseline": True,
        },
        "m2_network_nearest_neighbor": {
            "normalized_w1_to_amazon": 0.3,
            "generated": {"p50": 40.0},
            "amazon_reference": {"p50": 45.0},
        },
        "m3_within_region_pairwise": {
            "generated_region_p50_distribution": {"mean": 100.0},
            "amazon_route_p50_distribution": {"mean": 110.0},
        },
        "m4_region_structure": {"region_count": 8},
        "m5_community_concentration": {
            "proposal_minus_baseline_community_count": 2,
            "proposal_minus_baseline_hhi": -0.04,
        },
        "reliability": {
            "territory_reserve_ratio": 1.4,
            "seed_fallback_rate_per_region": 0.125,
            "region_attempts_used": 1,
            "region_redraw_count": 0,
            "assignment_competition_expansions": 2,
        },
    }
    (family / "phase1_metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "family_id": ["family-1", "family-1"],
            "city_slug": ["san-diego", "san-diego"],
            "day_type": ["weekday", "weekday"],
            "parent_scale_id": ["cus1000", "cus1000"],
            "source_t_env_s": [3600.0, 3600.0],
            "proposal_depot_time_s": [300.0, 600.0],
            "structure_target_time_s": [350.0, 550.0],
            "radial_baseline_depot_time_s": [200.0, 800.0],
            "proposal_depot_time_normalized": [300.0 / 3600.0, 600.0 / 3600.0],
            "structure_target_time_normalized": [350.0 / 3600.0, 550.0 / 3600.0],
            "radial_baseline_time_normalized": [200.0 / 3600.0, 800.0 / 3600.0],
        }
    ).to_parquet(family / "phase1_observations.parquet", index=False)

    summary = aggregate_phase1_metrics(tmp_path)
    report = tmp_path / "reports" / "phase1"
    assert summary["all_hard_gates_passed"]
    assert summary["raw_first_attempt_success_rate"] == 1.0
    assert (report / "family_metrics.parquet").is_file()
    assert (report / "stratified_metrics.csv").is_file()
    assert (report / "summary.json").is_file()


def test_phase1_aggregation_rejects_stale_family_metric_schema(tmp_path: Path) -> None:
    family = tmp_path / "materialized" / "families" / "family-1"
    family.mkdir(parents=True)
    (family / "phase1_metrics.json").write_text(
        json.dumps({"schema": "evrptw_phase1_family_metrics_v1"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Phase-1 family metric schema mismatch"):
        aggregate_phase1_metrics(tmp_path)
