from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from evrptw_stage2.artifacts import load_materialized_view, verify_materialized_family
from evrptw_stage2.community import build_customer_split
from evrptw_stage2.config import load_stage2_config
from evrptw_stage2.materialize import view_parent_terminal_indices
from evrptw_stage2.orders import FULL_CS_TO_DEPOT_CACHE_CONTRACT, build_view_attributes
from evrptw_stage2.planning import build_generation_plan, materialization_attempt_inputs
from evrptw_stage2.profile import load_reference_profile
from evrptw_stage2.reader import CLEEligibilityError, load_portable_cle
from evrptw_stage2.road_state import build_family_road_state
from evrptw_stage2.routing import PhysicalRoadNetwork
from evrptw_stage2.selection import _community_reference_points, _select_charger_rows

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "cle_evrptw_stage2_v1.json"


def _write_fake_cle(root: Path, *, release_eligible: bool = False) -> Path:
    city = root / "cities" / "test-city"
    for directory in ("graph", "service_locations", "infrastructure", "profiles"):
        (city / directory).mkdir(parents=True, exist_ok=True)
    (city / "graph" / "graph_operational.graphml").write_text("<graphml/>", encoding="utf-8")

    locations = gpd.GeoDataFrame(
        {
            "latent_service_location_id": [f"loc-{index}" for index in range(100)],
            "service_location_type": [
                "house" if index % 2 else "small_apt" for index in range(100)
            ],
            "residential_unit_band": ["1" if index % 2 else "2-4" for index in range(100)],
            "residential_units": [1 if index % 2 else 3 for index in range(100)],
            "location_lon": [(index // 10) * 0.02 + 0.005 for index in range(100)],
            "location_lat": [(index % 10) * 0.0005 + 0.005 for index in range(100)],
            "anchor_scc_id": [0] * 100,
            "cle_default_instance_eligible": [index != 99 for index in range(100)],
            "customer_release_eligible": [release_eligible and index != 99 for index in range(100)],
        },
        geometry=[
            Point((index // 10) * 0.02 + 0.005, (index % 10) * 0.0005 + 0.005)
            for index in range(100)
        ],
        crs="EPSG:4326",
    )
    locations.to_parquet(city / "service_locations" / "latent_locations.parquet", index=False)
    pd.DataFrame(
        {
            "candidate_id": ["depot-eligible", "depot-ineligible"],
            "depot_candidate_eligible": [True, False],
            "depot_release_eligible": [release_eligible, False],
        }
    ).to_parquet(city / "infrastructure" / "depots.parquet", index=False)
    pd.DataFrame(
        {
            "charger_id": ["charger-eligible", "charger-ineligible"],
            "charger_candidate_eligible": [True, False],
            "charger_release_eligible": [release_eligible, False],
        }
    ).to_parquet(city / "infrastructure" / "chargers.parquet", index=False)
    pd.DataFrame(
        {
            "edge_u": [1],
            "edge_v": [2],
            "edge_key": [0],
            "length_m": [100.0],
            "legal_speed_kph": [40.0],
            "reference_speed_kph": [30.0],
            "operating_mode": ["U"],
        }
    ).to_parquet(city / "profiles" / "directed_legal_speeds.parquet", index=False)
    manifest = {
        "schema": "evrptw_city_logistics_environment_v1",
        "city_slug": "test-city",
        "portable_package_verified": True,
        "release_eligible": release_eligible,
        "release_blockers": [] if release_eligible else ["test_blocker"],
        "outputs": {
            "operational_graph": "graph/graph_operational.graphml",
            "latent_locations": "service_locations/latent_locations.parquet",
            "depots": "infrastructure/depots.parquet",
            "chargers": "infrastructure/chargers.parquet",
            "directed_legal_speeds": "profiles/directed_legal_speeds.parquet",
        },
    }
    (city / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return city


def test_stage2_config_freezes_exposure_and_scale_contract() -> None:
    config = load_stage2_config(CONFIG_PATH)
    assert config.train_parent_family_count == 5_000
    for scale_id in ("cus50", "cus100", "cus500", "cus1000"):
        scale = config.scale(scale_id)
        assert scale.customers * scale.train_views == 5_000_000
    assert config.scale("cus2000").test_instances == {"unseen_scale_same_cities": 500}


def test_stage2_road_state_uses_moves_road_type_factor_without_edge_noise() -> None:
    profile = load_reference_profile(
        Path(__file__).parents[1] / "configs" / "us_reference_instance_profile_v1.json"
    )
    directed = pd.DataFrame(
        {
            "edge_u": ["a", "b", "c"],
            "edge_v": ["b", "c", "d"],
            "edge_key": ["0", "0", "0"],
            "edge_id": ["a:b:0", "b:c:0", "c:d:0"],
            "physical_segment_id": ["ab", "bc", "cd"],
            "length_m": [100.0, 100.0, 100.0],
            "operating_mode": ["H", "M", "U"],
            "legal_speed_kph": [100.0, 70.0, 45.0],
            "reference_speed_kph": [80.0, 50.0, 30.0],
        }
    )
    state, report = build_family_road_state(
        directed,
        day_type="weekday",
        road_state_seed=123,
        profile=profile,
    )
    assert state.loc[1, "road_state_factor"] == pytest.approx(state.loc[2, "road_state_factor"])
    assert set(state["moves_road_type"]) == {
        "urban_restricted_access",
        "urban_unrestricted_access",
    }
    assert "edge_energy_kwh" not in state
    assert report["additional_random_edge_factors"] is False


def test_stage2_road_state_replays_stored_baselines_without_rng() -> None:
    profile = load_reference_profile(
        Path(__file__).parents[1] / "configs" / "us_reference_instance_profile_v1.json"
    )
    directed = pd.DataFrame(
        {
            "edge_u": ["a", "b", "c"],
            "edge_v": ["b", "c", "d"],
            "edge_key": ["0", "0", "0"],
            "edge_id": ["a:b:0", "b:c:0", "c:d:0"],
            "physical_segment_id": ["ab", "bc", "cd"],
            "length_m": [100.0, 100.0, 100.0],
            "operating_mode": ["H", "M", "U"],
            "legal_speed_kph": [100.0, 70.0, 45.0],
            "reference_speed_kph": [80.0, 50.0, 30.0],
        }
    )
    sampled, sampled_report = build_family_road_state(
        directed,
        day_type="weekday",
        road_state_seed=123,
        profile=profile,
    )
    replayed, replayed_report = build_family_road_state(
        directed,
        day_type="weekday",
        road_state_seed=999999,
        profile=profile,
        moves_road_type_baseline_factors=sampled_report[
            "moves_road_type_baseline_factors"
        ],
    )
    np.testing.assert_array_equal(
        sampled["instance_speed_kph"].to_numpy(),
        replayed["instance_speed_kph"].to_numpy(),
    )
    np.testing.assert_array_equal(
        sampled["edge_travel_time_s"].to_numpy(),
        replayed["edge_travel_time_s"].to_numpy(),
    )
    assert replayed_report["baseline_factor_source"] == "stored_family_manifest"


def test_official_reader_rejects_unreleased_cle(tmp_path: Path) -> None:
    _write_fake_cle(tmp_path, release_eligible=False)
    with pytest.raises(CLEEligibilityError, match="not release eligible"):
        load_portable_cle(
            tmp_path,
            "test-city",
            mode="official",
            minimum_customers=1,
            minimum_depots=1,
            minimum_chargers=1,
        )


def test_research_reader_filters_default_candidate_rows(tmp_path: Path) -> None:
    _write_fake_cle(tmp_path, release_eligible=False)
    cle = load_portable_cle(
        tmp_path,
        "test-city",
        mode="research",
        minimum_customers=1,
        minimum_depots=1,
        minimum_chargers=1,
    )
    assert len(cle.read_service_locations()) == 99
    assert cle.read_depots(columns=["candidate_id"])["candidate_id"].tolist() == [
        "depot-eligible"
    ]
    assert cle.read_chargers(columns=["charger_id"])["charger_id"].tolist() == [
        "charger-eligible"
    ]
    assert cle.research_generation


def test_pilot_reader_always_filters_candidate_rows(tmp_path: Path) -> None:
    _write_fake_cle(tmp_path, release_eligible=False)
    cle = load_portable_cle(
        tmp_path,
        "test-city",
        mode="non_release_pilot",
        minimum_customers=1,
        minimum_depots=1,
        minimum_chargers=1,
    )
    assert (
        cle.read_service_locations(columns=["latent_service_location_id"])[
            "latent_service_location_id"
        ].tolist()[-1]
        == "loc-98"
    )
    assert cle.read_depots(columns=["candidate_id"])["candidate_id"].tolist() == ["depot-eligible"]
    assert cle.read_chargers(columns=["charger_id"])["charger_id"].tolist() == ["charger-eligible"]


def test_complete_community_split_is_deterministic_and_group_safe(tmp_path: Path) -> None:
    _write_fake_cle(tmp_path, release_eligible=False)
    cle = load_portable_cle(
        tmp_path,
        "test-city",
        mode="non_release_pilot",
        minimum_customers=1,
        minimum_depots=1,
        minimum_chargers=1,
    )
    blocks = gpd.GeoDataFrame(
        {"GEOID": [f"bg-{index}" for index in range(10)]},
        geometry=[
            Polygon(
                [
                    (index * 0.02, 0.0),
                    (index * 0.02 + 0.01, 0.0),
                    (index * 0.02 + 0.01, 0.01),
                    (index * 0.02, 0.01),
                ]
            )
            for index in range(10)
        ],
        crs="EPSG:4326",
    )
    blocks_path = tmp_path / "blocks.geojson"
    blocks.to_file(blocks_path, driver="GeoJSON")
    report = build_customer_split(
        cle,
        block_groups_path=blocks_path,
        output_dir=tmp_path / "split",
        split_seed=123,
    )
    ledger = pd.read_parquet(tmp_path / "split" / "customer_split_manifest.parquet")
    assert report["eligible_location_count"] == 99
    assert ledger.groupby("community_id")["customer_pool"].nunique().max() == 1
    assert ledger.loc[ledger["customer_pool"].eq("heldout"), "training_ineligible"].all()
    assert 0.1 <= report["actual_heldout_fraction"] <= 0.3


def test_official_generation_plan_has_frozen_family_and_view_counts() -> None:
    config = load_stage2_config(CONFIG_PATH)
    families, views, registry = build_generation_plan(config)
    assert len(families) == 7_500
    assert len(views) == 172_500
    assert registry["view_counts_by_scale"] == {
        "cus50": 101_000,
        "cus100": 52_000,
        "cus500": 12_000,
        "cus1000": 7_000,
        "cus2000": 500,
    }
    assert families.groupby("family_id")["family_cohort_id"].nunique().max() == 1
    train_views = views.loc[views["family_cohort_id"].eq("core/train")]
    assert train_views.groupby("scale_id").size().to_dict() == {
        "cus50": 100_000,
        "cus100": 50_000,
        "cus500": 10_000,
        "cus1000": 5_000,
    }


def test_materialization_attempt_seeds_are_deterministic_and_isolated() -> None:
    family = {
        "family_seed": 123,
        "depot_seed": 1,
        "customer_superset_seed": 2,
        "charger_seed": 3,
        "road_state_seed": 4,
        "vehicle_seed": 5,
    }
    views = pd.DataFrame(
        {
            "scale_id": ["cus100"],
            "branch_index": [2],
            "view_seed": [6],
            "package_seed": [7],
            "service_time_seed": [8],
            "time_window_seed": [9],
        }
    )
    attempt_zero, zero_views = materialization_attempt_inputs(family, views, attempt_number=0)
    attempt_one_a, one_views_a = materialization_attempt_inputs(family, views, attempt_number=1)
    attempt_one_b, one_views_b = materialization_attempt_inputs(family, views, attempt_number=1)
    assert attempt_zero["family_seed"] == 123
    assert attempt_one_a == attempt_one_b
    assert attempt_one_a["family_seed"] != attempt_zero["family_seed"]
    assert one_views_a.to_dict("records") == one_views_b.to_dict("records")
    assert one_views_a.iloc[0]["view_seed"] != zero_views.iloc[0]["view_seed"]


def test_pilot_generation_plan_can_target_one_city() -> None:
    config = load_stage2_config(CONFIG_PATH)
    families, views, registry = build_generation_plan(
        config,
        available_cities=["san-diego"],
        pilot_families_per_city=2,
        include_tracks=["train", "validation", "test1_new_seed", "test2_heldout_locations"],
        non_release_pilot=True,
    )
    assert len(families) == 8
    assert len(views) == 88
    assert registry["official_counts"] is False
    assert families["non_release_pilot"].all()
    assert views["non_release_pilot"].all()


def test_edge_projection_routing_uses_directional_partial_edge_costs() -> None:
    profile = load_reference_profile(
        Path(__file__).parents[1] / "configs" / "us_reference_instance_profile_v1.json"
    )
    graph = nx.MultiDiGraph()
    graph.add_node("a", x=-117.0, y=32.0)
    graph.add_node("b", x=-116.999, y=32.0)
    graph.add_edge("a", "b", key=0)
    graph.add_edge("b", "a", key=0)
    road_state = pd.DataFrame(
        {
            "edge_u": ["a", "b"],
            "edge_v": ["b", "a"],
            "edge_key": ["0", "0"],
            "length_m": [100.0, 100.0],
            "edge_travel_time_s": [10.0, 20.0],
        }
    )
    network = PhysicalRoadNetwork(graph, road_state, profile)

    def refs(offset_ab: float) -> str:
        return json.dumps(
            [
                {
                    "u": "a",
                    "v": "b",
                    "key": "0",
                    "length_m": 100.0,
                    "offset_from_u_m": offset_ab,
                    "offset_to_v_m": 100.0 - offset_ab,
                },
                {
                    "u": "b",
                    "v": "a",
                    "key": "0",
                    "length_m": 100.0,
                    "offset_from_u_m": 100.0 - offset_ab,
                    "offset_to_v_m": offset_ab,
                },
            ]
        )

    terminals = pd.DataFrame(
        {
            "terminal_index": [0, 1],
            "directed_projection_offsets": [refs(25.0), refs(75.0)],
            "connector_length_m": [0.0, 0.0],
        }
    )
    matrices = network.route_terminals(terminals)
    assert matrices.distance_matrix_km[0, 1] == pytest.approx(0.05)
    assert matrices.distance_matrix_km[1, 0] == pytest.approx(0.05)
    assert matrices.distance_path_travel_time_s[0, 1] == pytest.approx(5.0)
    assert matrices.distance_path_travel_time_s[1, 0] == pytest.approx(10.0)
    assert matrices.running_time_shortest_matrix_s[0, 1] == pytest.approx(5.0)
    assert matrices.running_time_shortest_matrix_s[1, 0] == pytest.approx(10.0)


def test_cached_topology_accepts_new_family_road_state() -> None:
    profile = load_reference_profile(
        Path(__file__).parents[1] / "configs" / "us_reference_instance_profile_v1.json"
    )
    graph = nx.MultiDiGraph()
    graph.add_node("a", x=-117.0, y=32.0)
    graph.add_node("b", x=-116.999, y=32.0)
    graph.add_edge("a", "b", key=0)
    graph.add_edge("b", "a", key=0)
    road_state = pd.DataFrame(
        {
            "edge_u": ["a", "b"],
            "edge_v": ["b", "a"],
            "edge_key": ["0", "0"],
            "length_m": [100.0, 100.0],
            "edge_travel_time_s": [10.0, 20.0],
        }
    )
    cached = PhysicalRoadNetwork(graph, road_state, profile)
    next_state = road_state.copy()
    next_state["edge_travel_time_s"] = [5.0, 40.0]
    network = cached.with_road_state(next_state, profile)
    terminals = pd.DataFrame(
        {
            "terminal_index": [0, 1],
            "directed_projection_offsets": [
                json.dumps(
                    [
                        {
                            "u": "a",
                            "v": "b",
                            "key": "0",
                            "length_m": 100.0,
                            "offset_from_u_m": 25.0,
                            "offset_to_v_m": 75.0,
                        },
                        {
                            "u": "b",
                            "v": "a",
                            "key": "0",
                            "length_m": 100.0,
                            "offset_from_u_m": 75.0,
                            "offset_to_v_m": 25.0,
                        },
                    ]
                ),
                json.dumps(
                    [
                        {
                            "u": "a",
                            "v": "b",
                            "key": "0",
                            "length_m": 100.0,
                            "offset_from_u_m": 75.0,
                            "offset_to_v_m": 25.0,
                        },
                        {
                            "u": "b",
                            "v": "a",
                            "key": "0",
                            "length_m": 100.0,
                            "offset_from_u_m": 25.0,
                            "offset_to_v_m": 75.0,
                        },
                    ]
                ),
            ],
            "connector_length_m": [0.0, 0.0],
        }
    )
    matrices = network.route_terminals(terminals)
    assert network._distance_adjacency is cached._distance_adjacency
    assert matrices.distance_matrix_km[0, 1] == pytest.approx(0.05)
    assert matrices.distance_path_travel_time_s[0, 1] == pytest.approx(2.5)
    assert matrices.distance_path_travel_time_s[1, 0] == pytest.approx(20.0)


def test_cached_topology_reselects_equal_length_parallel_edge() -> None:
    profile = load_reference_profile(
        Path(__file__).parents[1] / "configs" / "us_reference_instance_profile_v1.json"
    )
    graph = nx.MultiDiGraph()
    graph.add_node("a", x=-117.0, y=32.0)
    graph.add_node("b", x=-116.999, y=32.0)
    graph.add_edge("a", "b", key=0)
    graph.add_edge("a", "b", key=1)
    road_state = pd.DataFrame(
        {
            "edge_u": ["a", "a"],
            "edge_v": ["b", "b"],
            "edge_key": ["0", "1"],
            "length_m": [100.0, 100.0],
            "edge_travel_time_s": [5.0, 10.0],
        }
    )
    cached = PhysicalRoadNetwork(graph, road_state, profile)
    pair = (cached.node_to_index["a"], cached.node_to_index["b"])
    assert cached._distance_chosen_edge[pair] == 0

    next_state = road_state.copy()
    next_state["edge_travel_time_s"] = [20.0, 2.0]
    network = cached.with_road_state(next_state, profile)

    assert network._distance_adjacency is cached._distance_adjacency
    assert network._distance_chosen_edge[pair] == 1


def test_running_time_path_optimizes_turn_penalties_not_only_edge_times() -> None:
    profile = load_reference_profile(
        Path(__file__).parents[1] / "configs" / "us_reference_instance_profile_v1.json"
    )
    profile["turn_penalty"]["left_turn_s"] = 100.0
    profile["turn_penalty"]["right_turn_s"] = 100.0
    graph = nx.MultiDiGraph()
    coordinates = {
        "a": (-0.001, 0.0),
        "b": (0.0, 0.0),
        "c": (0.0, 0.001),
        "f": (0.001, 0.0),
        "d": (0.002, 0.0),
        "e": (0.003, 0.0),
    }
    for node, (x, y) in coordinates.items():
        graph.add_node(node, x=x, y=y)
    edges = [
        ("a", "b", 100.0, 10.0),
        ("b", "c", 10.0, 1.0),
        ("c", "d", 10.0, 1.0),
        ("b", "f", 100.0, 10.0),
        ("f", "d", 100.0, 10.0),
        ("d", "e", 100.0, 10.0),
        ("e", "a", 100.0, 10.0),
    ]
    for u, v, _, _ in edges:
        graph.add_edge(u, v, key=0)
    road_state = pd.DataFrame(
        {
            "edge_u": [edge[0] for edge in edges],
            "edge_v": [edge[1] for edge in edges],
            "edge_key": ["0"] * len(edges),
            "length_m": [edge[2] for edge in edges],
            "edge_travel_time_s": [edge[3] for edge in edges],
        }
    )
    network = PhysicalRoadNetwork(graph, road_state, profile)

    def one_ref(u: str, v: str, offset: float) -> str:
        return json.dumps(
            [
                {
                    "u": u,
                    "v": v,
                    "key": "0",
                    "length_m": 100.0,
                    "offset_from_u_m": offset,
                    "offset_to_v_m": 100.0 - offset,
                }
            ]
        )

    matrices = network.route_terminals(
        pd.DataFrame(
            {
                "terminal_index": [0, 1],
                "directed_projection_offsets": [
                    one_ref("a", "b", 50.0),
                    one_ref("d", "e", 50.0),
                ],
                "connector_length_m": [0.0, 0.0],
            }
        )
    )
    assert matrices.distance_matrix_km[0, 1] == pytest.approx(0.12)
    assert matrices.running_time_path_distance_km[0, 1] == pytest.approx(0.30)
    assert matrices.running_time_shortest_matrix_s[0, 1] == pytest.approx(30.0)
    assert matrices.report["turn_penalty_in_running_time_path_optimization"] is True


def test_nested_view_indices_use_customer_blocks_and_charger_prefixes() -> None:
    indices = view_parent_terminal_indices(
        {
            "scale_id": "cus100",
            "customer_count": 100,
            "charging_station_count": 20,
            "branch_index": 3,
        },
        parent_customer_count=1000,
        parent_charging_station_count=50,
    )
    assert indices[:4].tolist() == [0, 301, 302, 303]
    assert indices[100] == 400
    assert indices[101:].tolist() == list(range(1001, 1021))


def test_charger_selection_uses_community_reference_points_not_daily_customers() -> None:
    latent_pool = pd.DataFrame(
        {
            "latent_service_location_id": ["a1", "a2", "b1", "b2", "b3"],
            "community_id": ["a", "a", "b", "b", "b"],
            "location_lon": [0.0, 0.02, 1.0, 1.01, 1.02],
            "location_lat": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    chargers = pd.DataFrame(
        {
            "charger_id": ["left", "right", "far"],
            "resolved_longitude": [0.01, 1.01, 3.0],
            "resolved_latitude": [0.0, 0.0, 0.0],
        }
    )
    reference_points = _community_reference_points(latent_pool)
    selected, report = _select_charger_rows(
        chargers,
        reference_points,
        count=2,
        seed=7,
    )
    assert set(selected["charger_id"]) == {"left", "right"}
    assert report["reference_point_count"] == 2
    assert "independent of daily active customer IDs" in report["reference_point_semantics"]


def test_view_attributes_use_volume_and_pass_sufficient_feasibility() -> None:
    profile = load_reference_profile(
        Path(__file__).parents[1] / "configs" / "us_reference_instance_profile_v1.json"
    )
    customers = pd.DataFrame(
        {
            "residential_units": [1, 40],
            "service_location_type": ["house", "medium_apt"],
        }
    )
    travel_time = np.asarray(
        [
            [0.0, 300.0, 450.0, 100.0],
            [300.0, 0.0, 200.0, 120.0],
            [450.0, 200.0, 0.0, 150.0],
            [100.0, 120.0, 150.0, 0.0],
        ],
        dtype=np.float32,
    )
    energy = travel_time / 100.0
    specific_energy = profile["energy"]["specific_energy_consumption_kwh_per_km"]
    attributes = build_view_attributes(
        customers,
        day_type="weekday",
        package_seed=11,
        service_time_seed=12,
        time_window_seed=13,
        operating_start_s=8 * 3600,
        operating_end_s=24 * 3600,
        running_time_matrix_s=travel_time,
        running_time_path_distance_matrix_km=energy / specific_energy,
        charging_power_kw=np.asarray([100.0], dtype=np.float32),
        profile=profile,
    )
    assert attributes.package_counts.shape == (2,)
    assert (attributes.package_counts >= 1).all()
    assert (attributes.demands_cm3 > 0).all()
    assert attributes.time_windows_s.shape == (2, 2)
    assert attributes.full_cs_to_depot_time_s.shape == (1,)
    assert attributes.report["feasibility_gate"]["passed"] is True


def test_feasibility_certificate_can_use_full_charge_station() -> None:
    profile = load_reference_profile(
        Path(__file__).parents[1] / "configs" / "us_reference_instance_profile_v1.json"
    )
    customers = pd.DataFrame({"residential_units": [1], "service_location_type": ["house"]})
    # Terminal order: depot, customer, charger. Direct customer round trip uses
    # 120 kWh and is impossible with a 100 kWh pack. The constructed route
    # depot -> customer -> charger -> depot uses 80 kWh before charging and is feasible.
    travel_time = np.asarray(
        [[0.0, 600.0, 500.0], [600.0, 0.0, 300.0], [500.0, 300.0, 0.0]],
        dtype=np.float32,
    )
    energy = np.asarray(
        [[0.0, 60.0, 40.0], [60.0, 0.0, 20.0], [40.0, 20.0, 0.0]],
        dtype=np.float32,
    )
    specific_energy = profile["energy"]["specific_energy_consumption_kwh_per_km"]
    attributes = build_view_attributes(
        customers,
        day_type="weekday",
        package_seed=21,
        service_time_seed=22,
        time_window_seed=23,
        operating_start_s=8 * 3600,
        operating_end_s=24 * 3600,
        running_time_matrix_s=travel_time,
        running_time_path_distance_matrix_km=energy / specific_energy,
        charging_power_kw=np.asarray([100.0], dtype=np.float32),
        profile=profile,
    )
    assert bool(attributes.feasibility_requires_charging[0]) is True
    assert attributes.feasibility_charging_visit_count[0] == 1
    certificate_uses_charger = (
        attributes.feasibility_inbound_full_state_terminal_index[0] == 2
        or attributes.feasibility_first_post_customer_charger_terminal_index[0] == 2
    )
    assert certificate_uses_charger
    assert attributes.feasibility_energy_margin_kwh[0] == pytest.approx(20.0)
    assert attributes.report["feasibility_gate"]["requires_charging_count"] == 1


def test_feasibility_certificate_uses_multihop_full_cs_return_cache() -> None:
    profile = load_reference_profile(
        Path(__file__).parents[1] / "configs" / "us_reference_instance_profile_v1.json"
    )
    customers = pd.DataFrame({"residential_units": [1], "service_location_type": ["house"]})
    # Terminal order: depot, customer, CS-A, CS-B.  After depot -> customer,
    # neither customer -> depot nor customer -> CS-B is energy-feasible.  The
    # only feasible return is customer -> CS-A -> CS-B -> depot.  In
    # particular, CS-A cannot reach the depot on one battery and therefore
    # exercises the cached multi-hop full-CS-to-depot path.
    travel_time = np.asarray(
        [
            [0.0, 500.0, 900.0, 600.0],
            [900.0, 0.0, 400.0, 900.0],
            [900.0, 900.0, 0.0, 600.0],
            [600.0, 900.0, 900.0, 0.0],
        ],
        dtype=np.float32,
    )
    energy = np.asarray(
        [
            [0.0, 50.0, 150.0, 60.0],
            [150.0, 0.0, 40.0, 150.0],
            [150.0, 150.0, 0.0, 60.0],
            [60.0, 150.0, 150.0, 0.0],
        ],
        dtype=np.float32,
    )
    specific_energy = profile["energy"]["specific_energy_consumption_kwh_per_km"]
    attributes = build_view_attributes(
        customers,
        day_type="weekday",
        package_seed=31,
        service_time_seed=32,
        time_window_seed=33,
        operating_start_s=8 * 3600,
        operating_end_s=24 * 3600,
        running_time_matrix_s=travel_time,
        running_time_path_distance_matrix_km=energy / specific_energy,
        charging_power_kw=np.asarray([100.0, 100.0], dtype=np.float32),
        profile=profile,
    )
    assert bool(attributes.feasibility_requires_charging[0]) is True
    assert attributes.feasibility_first_post_customer_charger_terminal_index[0] == 2
    assert attributes.feasibility_charging_visit_count[0] == 2
    assert attributes.feasibility_energy_margin_kwh[0] == pytest.approx(10.0)
    assert attributes.full_cs_to_depot_time_s.tolist() == pytest.approx([3360.0, 600.0])
    assert attributes.report["full_cs_to_depot_cache"]["finite_return_count"] == 2
    assert attributes.report["full_cs_to_depot_cache"]["maximum_time_s"] == pytest.approx(3360.0)
    assert attributes.report["feasibility_gate"]["passed"] is True


def test_v2_view_stores_and_loads_full_cs_to_depot_cache(tmp_path: Path) -> None:
    profile = load_reference_profile(
        Path(__file__).parents[1] / "configs" / "us_reference_instance_profile_v1.json"
    )
    family = tmp_path / "family"
    view = family / "views" / "view-1"
    matrices_dir = family / "matrices"
    view.mkdir(parents=True)
    matrices_dir.mkdir()
    pd.DataFrame(
        {
            "source_id": ["depot", "customer", "charger"],
            "longitude": [-117.0, -117.1, -117.2],
            "latitude": [32.7, 32.8, 32.9],
        }
    ).to_parquet(family / "terminal_index.parquet", index=False)
    travel_time = np.asarray(
        [[0.0, 600.0, 500.0], [600.0, 0.0, 300.0], [500.0, 300.0, 0.0]],
        dtype=np.float32,
    )
    energy = np.asarray(
        [[0.0, 60.0, 40.0], [60.0, 0.0, 20.0], [40.0, 20.0, 0.0]],
        dtype=np.float32,
    )
    specific_energy = profile["energy"]["specific_energy_consumption_kwh_per_km"]
    distance = travel_time / 100.0
    matrix_values = {
        "distance_matrix_km": distance,
        "distance_path_travel_time_s": travel_time,
        "running_time_shortest_matrix_s": travel_time,
        "running_time_path_distance_km": energy / specific_energy,
    }
    matrix_files = {}
    for name, values in matrix_values.items():
        relative = f"matrices/{name}.npy"
        np.save(family / relative, values, allow_pickle=False)
        matrix_files[name] = relative

    attributes = build_view_attributes(
        pd.DataFrame({"residential_units": [1], "service_location_type": ["house"]}),
        day_type="weekday",
        package_seed=41,
        service_time_seed=42,
        time_window_seed=43,
        operating_start_s=8 * 3600,
        operating_end_s=24 * 3600,
        running_time_matrix_s=travel_time,
        running_time_path_distance_matrix_km=energy / specific_energy,
        charging_power_kw=np.asarray([100.0], dtype=np.float32),
        profile=profile,
    )
    np.save(view / "terminal_parent_indices.npy", np.arange(3, dtype=np.int32))
    np.savez_compressed(
        view / "customer_attributes.npz",
        package_counts=attributes.package_counts,
        demands_cm3=attributes.demands_cm3,
        service_time_s=attributes.service_time_s,
        time_windows_s=attributes.time_windows_s,
        feasible_arrival_time_s=attributes.feasible_arrival_time_s,
        feasible_return_duration_s=attributes.feasible_return_duration_s,
        feasibility_requires_charging=attributes.feasibility_requires_charging,
        feasibility_charging_visit_count=attributes.feasibility_charging_visit_count,
        feasibility_inbound_full_state_terminal_index=(
            attributes.feasibility_inbound_full_state_terminal_index
        ),
        feasibility_first_post_customer_charger_terminal_index=(
            attributes.feasibility_first_post_customer_charger_terminal_index
        ),
        feasibility_energy_margin_kwh=attributes.feasibility_energy_margin_kwh,
        order_sampling_attempts=attributes.order_sampling_attempts,
    )
    np.savez_compressed(
        view / "charging_attributes.npz",
        charging_power_kw=np.asarray([100.0], dtype=np.float32),
        full_cs_to_depot_time_s=attributes.full_cs_to_depot_time_s,
    )
    view_manifest = {
        "schema": "cle_evrptw_materialized_view_v3",
        "view_id": "view-1",
        "family_id": "family-1",
        "consumer_cohort_id": "core/train",
        "split_id": "train",
        "track_id": "train",
        "day_type": "weekday",
        "scale_id": "cus1",
        "customer_count": 1,
        "charging_station_count": 1,
        "operating_horizon_s": [8 * 3600, 24 * 3600],
        "terminal_parent_indices": "terminal_parent_indices.npy",
        "customer_attributes": "customer_attributes.npz",
        "charging_attributes": "charging_attributes.npz",
        "full_cs_to_depot_cache": dict(FULL_CS_TO_DEPOT_CACHE_CONTRACT),
        "vehicle": {
            "battery_capacity_kwh": 100.0,
            "cargo_capacity_cm3": float(profile["vehicle"]["cargo_capacity_cm3"]),
            "unlimited_fleet": True,
            "specific_energy_consumption_kwh_per_km": specific_energy,
        },
        "energy_model": dict(profile["energy"]),
        "charging_policy": dict(profile["charging"]),
        "non_release_pilot": True,
    }
    (view / "view_manifest.json").write_text(json.dumps(view_manifest), encoding="utf-8")
    family_manifest = {
        "schema": "cle_evrptw_materialized_matrix_family_v2",
        "family_id": "family-1",
        "city_slug": "test-city",
        "terminal_index": "terminal_index.parquet",
        "terminal_count": 3,
        "matrix_files": matrix_files,
        "view_ids": ["view-1"],
        "view_count": 1,
        "matrix_total_bytes": sum(
            int((family / relative).stat().st_size) for relative in matrix_files.values()
        ),
        "non_release_pilot": True,
        "reference_profile_id": profile["profile_id"],
        "reference_profile_status": profile["profile_status"],
        "energy_model": dict(profile["energy"]),
    }
    (family / "family_manifest.json").write_text(json.dumps(family_manifest), encoding="utf-8")

    loaded = load_materialized_view(family, "view-1")
    assert loaded["metadata"]["full_cs_to_depot_cache_source"] == "stored"
    assert loaded["full_cs_to_depot_time_s"].tolist() == pytest.approx([500.0])
    report = verify_materialized_family(family)
    assert report["passed"] is True
    assert report["stored_full_cs_cache_view_count"] == 1
    assert report["unreachable_full_cs_return_count"] == 0
