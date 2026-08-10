from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from evrptw_stage2.community import build_customer_split
from evrptw_stage2.config import load_stage2_config
from evrptw_stage2.materialize import view_parent_terminal_indices
from evrptw_stage2.orders import build_view_attributes
from evrptw_stage2.planning import build_generation_plan, materialization_attempt_inputs
from evrptw_stage2.profile import load_reference_profile
from evrptw_stage2.reader import CLEEligibilityError, load_portable_cle
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
            "service_location_type": ["house" if index % 2 else "small_apt" for index in range(100)],
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
    assert config.scale("cus2000").test_instances == {"unseen_scale_same_cities": 100}


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
    assert cle.read_service_locations(columns=["latent_service_location_id"])[
        "latent_service_location_id"
    ].tolist()[-1] == "loc-98"
    assert cle.read_depots(columns=["candidate_id"])["candidate_id"].tolist() == [
        "depot-eligible"
    ]
    assert cle.read_chargers(columns=["charger_id"])["charger_id"].tolist() == [
        "charger-eligible"
    ]


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
    assert len(families) == 7_100
    assert len(views) == 172_100
    assert registry["view_counts_by_scale"] == {
        "cus50": 101_000,
        "cus100": 52_000,
        "cus500": 12_000,
        "cus1000": 7_000,
        "cus2000": 100,
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
    attempt_zero, zero_views = materialization_attempt_inputs(
        family, views, attempt_number=0
    )
    attempt_one_a, one_views_a = materialization_attempt_inputs(
        family, views, attempt_number=1
    )
    attempt_one_b, one_views_b = materialization_attempt_inputs(
        family, views, attempt_number=1
    )
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
            "edge_energy_kwh": [0.04, 0.08],
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
    assert matrices.distance_path_energy_kwh[0, 1] == pytest.approx(0.02)
    assert matrices.distance_path_energy_kwh[1, 0] == pytest.approx(0.04)


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
    attributes = build_view_attributes(
        customers,
        day_type="weekday",
        package_seed=11,
        service_time_seed=12,
        time_window_seed=13,
        operating_start_s=8 * 3600,
        operating_end_s=24 * 3600,
        running_time_matrix_s=travel_time,
        running_time_energy_matrix_kwh=energy,
        charging_power_kw=np.asarray([100.0], dtype=np.float32),
        profile=profile,
    )
    assert attributes.package_counts.shape == (2,)
    assert (attributes.package_counts >= 1).all()
    assert (attributes.demands_cm3 > 0).all()
    assert attributes.time_windows_s.shape == (2, 2)
    assert attributes.report["feasibility_gate"]["passed"] is True


def test_feasibility_certificate_can_use_full_charge_station() -> None:
    profile = load_reference_profile(
        Path(__file__).parents[1] / "configs" / "us_reference_instance_profile_v1.json"
    )
    customers = pd.DataFrame(
        {"residential_units": [1], "service_location_type": ["house"]}
    )
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
    attributes = build_view_attributes(
        customers,
        day_type="weekday",
        package_seed=21,
        service_time_seed=22,
        time_window_seed=23,
        operating_start_s=8 * 3600,
        operating_end_s=24 * 3600,
        running_time_matrix_s=travel_time,
        running_time_energy_matrix_kwh=energy,
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
