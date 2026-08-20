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
from evrptw_stage2.materialize import (
    _make_progress_emitter,
    view_parent_terminal_indices,
)
from evrptw_stage2.parallel import (
    rejection_is_retryable,
    remaining_attempt_numbers,
    worker_error_is_fatal,
)
from evrptw_stage2.planning import build_generation_plan, materialization_attempt_inputs
from evrptw_stage2.profile import load_reference_profile
from evrptw_stage2.reader import CLEEligibilityError, load_portable_cle
from evrptw_stage2.road_state import build_family_road_state
from evrptw_stage2.routing import (
    DepotTerminalStar,
    PhysicalRoadNetwork,
    TerminalConnectivityError,
)
from evrptw_stage2.selection import (
    JointSupportConsistencyError,
    _prepare_customer_territory,
    _select_depot_group,
    depot_candidate_order,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "cle_evrptw_stage2_v2.json"


def test_c3_depot_star_cache_is_scoped_to_one_family_and_depot() -> None:
    customers = pd.DataFrame(
        {
            "latent_service_location_id": ["c1", "c2"],
            "customer_pool": ["train", "train"],
            "community_id": ["g1", "g1"],
            "road_connectivity_subgroup": ["s1", "s1"],
            "location_lon": [0.1, 0.2],
            "location_lat": [0.1, 0.2],
            "physical_edge_id": ["e1", "e1"],
            "directed_projection_offsets": ["[]", "[]"],
            "connector_length_m": [0.0, 0.0],
            "road_projection_node_id": ["n1", "n1"],
            "service_access_node_id": ["n1", "n1"],
            "anchor_scc_id": ["scc", "scc"],
        }
    )
    depot = pd.Series(
        {
            "candidate_id": "d1",
            "longitude": 0.0,
            "latitude": 0.0,
            "physical_edge_id": "e1",
            "directed_projection_offsets": "[]",
            "connector_length_m": 0.0,
            "road_projection_node_id": "n1",
            "facility_access_node_id": "n1",
            "anchor_scc_id": "scc",
        }
    )

    class Network:
        calls = 0

        def route_depot_star(self, terminal_index: pd.DataFrame) -> DepotTerminalStar:
            self.calls += 1
            count = len(terminal_index)
            reachable = np.ones(count, dtype=bool)
            return DepotTerminalStar(
                outbound_time_s=np.asarray([0.0, 10.0, 20.0]),
                inbound_time_s=np.asarray([0.0, 11.0, 21.0]),
                outbound_distance_km=np.asarray([0.0, 1.0, 2.0]),
                inbound_distance_km=np.asarray([0.0, 1.1, 2.1]),
                node_outbound_reachable=reachable,
                node_return_reachable=reachable,
                turn_outbound_reachable=reachable,
                turn_return_reachable=reachable,
                report={"terminal_count": count},
            )

    network = Network()
    cache: dict = {}
    family = {"parent_customer_count": 1, "customer_pool": "train"}
    profile = {
        "energy": {
            "specific_energy_consumption_kwh_per_km": 1.0,
            "battery_capacity_kwh": 100.0,
        }
    }
    metadata = {
        "source_t_env_s": 100.0,
        "source_radial_decile_edges_s": list(range(11)),
        "structure_source_ids": ["source"],
    }
    first, _ = _prepare_customer_territory(
        object(),
        family=family,
        depot=depot,
        structure_metadata=metadata,
        customer_split_path="unused",
        profile=profile,
        network=network,
        customer_split_roster=customers,
        depot_star_cache=cache,
    )
    second, _ = _prepare_customer_territory(
        object(),
        family=family,
        depot=depot,
        structure_metadata=metadata,
        customer_split_path="unused",
        profile=profile,
        network=network,
        customer_split_roster=customers,
        depot_star_cache=cache,
    )
    assert network.calls == 1
    pd.testing.assert_frame_equal(first, second)


def _write_fake_cle(root: Path, *, release_eligible: bool = False) -> Path:
    city = root / "cities" / "test-city"
    for directory in ("graph", "service_locations", "infrastructure", "profiles"):
        (city / directory).mkdir(parents=True, exist_ok=True)
    graph = nx.MultiDiGraph()
    graph.add_node("1", x=0.005, y=0.005)
    graph.add_node("2", x=0.025, y=0.005)
    graph.add_edge("1", "2", key=0, length=2_000.0)
    graph.add_edge("2", "1", key=0, length=2_000.0)
    nx.write_graphml(graph, city / "graph" / "graph_operational.graphml")

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
            "protected_inbound_access_eligible": [True] * 100,
            "protected_outbound_access_eligible": [True] * 100,
            "protected_roundtrip_eligible": [True] * 100,
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
            "protected_inbound_access_eligible": [True, True],
            "protected_outbound_access_eligible": [True, True],
            "protected_roundtrip_eligible": [True, True],
        }
    ).to_parquet(city / "infrastructure" / "depots.parquet", index=False)
    pd.DataFrame(
        {
            "charger_id": ["charger-eligible", "charger-ineligible"],
            "charger_candidate_eligible": [True, False],
            "charger_release_eligible": [release_eligible, False],
            "protected_inbound_access_eligible": [True, True],
            "protected_outbound_access_eligible": [True, True],
            "protected_roundtrip_eligible": [True, True],
        }
    ).to_parquet(city / "infrastructure" / "chargers.parquet", index=False)
    pd.DataFrame(
        {
            "edge_u": [1],
            "edge_v": [2],
            "edge_key": [0],
            "length_m": [100.0],
            "legal_speed_kph": [40.0],
            "moves_road_type": ["urban_unrestricted_access"],
            "reference_speed_weekday_kph": [30.0],
            "reference_speed_weekend_kph": [31.0],
            "operating_mode": ["U"],
        }
    ).to_parquet(city / "profiles" / "directed_legal_speeds.parquet", index=False)
    (city / "profiles" / "speed_manifest.json").write_text(
        json.dumps(
            {
                "schema": "evrptw_directed_speed_profiles_v6",
                "reference_speed_contract": {"profile_id": "test-moves-profile"},
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema": "evrptw_city_logistics_environment_v1",
        "connectivity_contract": {"id": "directed_projection_roundtrip_v2"},
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
            "speed_manifest": "profiles/speed_manifest.json",
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
        Path(__file__).parents[1] / "configs" / "us_reference_instance_profile_v2.json"
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
            "moves_road_type": [
                "urban_restricted_access",
                "urban_unrestricted_access",
                "urban_unrestricted_access",
            ],
            "reference_speed_weekday_kph": [80.0, 50.0, 30.0],
            "reference_speed_weekend_kph": [85.0, 55.0, 35.0],
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
    assert report["reference_speed_column"] == "reference_speed_weekday_kph"
    np.testing.assert_array_equal(
        state["instance_speed_kph"].to_numpy(), np.asarray([80.0, 50.0, 30.0])
    )


def test_stage2_road_state_replays_stored_baselines_without_rng() -> None:
    profile = load_reference_profile(
        Path(__file__).parents[1] / "configs" / "us_reference_instance_profile_v2.json"
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
            "moves_road_type": [
                "urban_restricted_access",
                "urban_unrestricted_access",
                "urban_unrestricted_access",
            ],
            "reference_speed_weekday_kph": [80.0, 50.0, 30.0],
            "reference_speed_weekend_kph": [85.0, 55.0, 35.0],
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


def test_official_reader_explicitly_accepts_frozen_technical_candidate(
    tmp_path: Path,
) -> None:
    _write_fake_cle(tmp_path, release_eligible=False)
    manifest_path = tmp_path / "cities" / "test-city" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["technical_verification_passed"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cle = load_portable_cle(
        tmp_path,
        "test-city",
        mode="official",
        minimum_customers=1,
        minimum_depots=1,
        minimum_chargers=1,
        official_cle_contract="frozen_technical_candidate_v1",
    )
    assert cle.customer_eligibility_field == "cle_default_instance_eligible"
    assert cle.depot_eligibility_field == "depot_candidate_eligible"
    assert cle.charger_eligibility_field == "charger_candidate_eligible"
    assert cle.eligibility_contract == "frozen_technical_candidate_v1"
    assert cle.eligibility_summary()["manual_cle_release_claimed"] is False
    assert cle.warnings


def test_official_toy_reader_uses_candidate_contract_and_is_non_release(
    tmp_path: Path,
) -> None:
    _write_fake_cle(tmp_path, release_eligible=False)
    manifest_path = tmp_path / "cities" / "test-city" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["technical_verification_passed"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cle = load_portable_cle(
        tmp_path,
        "test-city",
        mode="official_toy",
        minimum_customers=1,
        minimum_depots=1,
        minimum_chargers=1,
        official_cle_contract="frozen_technical_candidate_v1",
    )
    assert cle.non_release_pilot is True
    assert cle.eligibility_contract == "frozen_technical_candidate_v1"
    assert cle.eligibility_summary()["manual_cle_release_claimed"] is False
    assert any("non-release test corpus" in warning for warning in cle.warnings)


def test_official_toy_reader_rejects_default_strict_contract(
    tmp_path: Path,
) -> None:
    _write_fake_cle(tmp_path, release_eligible=False)
    manifest_path = tmp_path / "cities" / "test-city" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["technical_verification_passed"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CLEEligibilityError, match="not release eligible"):
        load_portable_cle(
            tmp_path,
            "test-city",
            mode="official_toy",
            minimum_customers=1,
            minimum_depots=1,
            minimum_chargers=1,
        )


def test_reader_rejects_legacy_cle_v1_root(tmp_path: Path) -> None:
    cle_root = tmp_path / "CLE_v1" / "us_11city"
    _write_fake_cle(cle_root, release_eligible=False)
    with pytest.raises(CLEEligibilityError, match="read-only legacy evidence"):
        load_portable_cle(
            cle_root,
            "test-city",
            mode="non_release_pilot",
            minimum_customers=1,
            minimum_depots=1,
            minimum_chargers=1,
        )


def test_reader_rejects_stale_speed_contract(tmp_path: Path) -> None:
    city = _write_fake_cle(tmp_path, release_eligible=False)
    speed_manifest = city / "profiles" / "speed_manifest.json"
    speed_manifest.write_text(
        json.dumps(
            {
                "schema": "evrptw_directed_speed_profiles_v5",
                "reference_speed_contract": {"profile_id": "old-profile"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CLEEligibilityError, match="stale speed schema"):
        load_portable_cle(
            tmp_path,
            "test-city",
            mode="research",
            minimum_customers=1,
            minimum_depots=1,
            minimum_chargers=1,
        )


def test_reader_rejects_stale_directional_connectivity_contract(tmp_path: Path) -> None:
    city = _write_fake_cle(tmp_path, release_eligible=False)
    manifest_path = city / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("connectivity_contract")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CLEEligibilityError, match="directed_projection_roundtrip_v2"):
        load_portable_cle(
            tmp_path,
            "test-city",
            mode="research",
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
    build_customer_split(
        cle,
        block_groups_path=blocks_path,
        output_dir=tmp_path / "split-repeat",
        split_seed=123,
    )
    ledger = pd.read_parquet(tmp_path / "split" / "customer_split_manifest.parquet")
    repeated = pd.read_parquet(
        tmp_path / "split-repeat" / "customer_split_manifest.parquet"
    )
    membership_columns = [
        "latent_service_location_id",
        "community_id",
        "customer_pool",
        "training_ineligible",
    ]
    pd.testing.assert_frame_equal(
        ledger[membership_columns].sort_values("latent_service_location_id").reset_index(drop=True),
        repeated[membership_columns]
        .sort_values("latent_service_location_id")
        .reset_index(drop=True),
    )
    assert report["eligible_location_count"] == 99
    assert ledger.groupby("community_id")["customer_pool"].nunique().max() == 1
    assert ledger.loc[ledger["customer_pool"].eq("heldout"), "training_ineligible"].all()
    assert 0.1 <= report["actual_heldout_fraction"] <= 0.3


def test_official_generation_plan_has_frozen_family_and_view_counts() -> None:
    config = load_stage2_config(CONFIG_PATH)
    families, views, registry = build_generation_plan(config)
    assert len(families) == 7_500
    assert len(views) == 173_000
    assert registry["view_counts_by_scale"] == {
        "cus50": 101_000,
        "cus100": 52_000,
        "cus500": 12_000,
        "cus1000": 7_500,
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
    for (_, _, _), group in families.groupby(
        ["city_slug", "family_cohort_id", "parent_scale_id"], sort=True
    ):
        counts = group["day_type"].value_counts()
        assert int(counts.get("weekday", 0) + counts.get("weekend", 0)) == len(group)
        assert abs(float(counts.get("weekday", 0) / len(group)) - 5 / 7) <= 1 / len(group)
    evaluation = views.loc[
        views["family_cohort_id"].eq("core/test/test1_new_seed")
    ]
    for _, group in evaluation.groupby("family_id"):
        branches = group.set_index("scale_id")["branch_index"].astype(int)
        assert branches["cus100"] == branches["cus50"] // 2
        assert branches["cus500"] == branches["cus50"] // 10
        assert branches["cus1000"] == 0


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


def test_c3_retry_preserves_the_joint_spatial_capacity_tuple() -> None:
    family = {
        "family_seed": 123,
        "depot_seed": 1,
        "customer_superset_seed": 2,
        "charger_seed": 3,
        "road_state_seed": 4,
        "vehicle_seed": 5,
        "joint_support_contract_id": "c3_joint_spatial_support_v1",
        "selected_depot_id": "depot-a",
        "selected_structure_source_id": "station-day-a",
        "capacity_contract_fingerprint": "ccf-test",
    }
    views = pd.DataFrame(
        {
            "scale_id": ["cus1000"],
            "branch_index": [0],
            "view_seed": [6],
        }
    )
    attempt, _ = materialization_attempt_inputs(
        family,
        views,
        attempt_number=2,
    )
    assert attempt["family_seed"] != family["family_seed"]
    assert attempt["charger_seed"] != family["charger_seed"]
    for key in (
        "depot_seed",
        "customer_superset_seed",
        "road_state_seed",
        "selected_depot_id",
        "selected_structure_source_id",
        "capacity_contract_fingerprint",
    ):
        assert attempt[key] == family[key]


def test_c3_depot_candidate_order_preserves_the_legacy_first_choice() -> None:
    depots = pd.DataFrame(
        {
            "candidate_id": ["a1", "a2", "b1", "c1"],
            "facility_group_id": ["a", "a", "b", "c"],
            "strict_depot_candidate_eligible": [True, False, True, True],
            "optional_depot_candidate_eligible": [False, True, False, False],
        }
    )
    legacy, _ = _select_depot_group(depots, seed=871, track="practical")
    ordered, metadata = depot_candidate_order(
        depots,
        seed=871,
        track="practical",
    )
    assert str(ordered[0]["candidate_id"]) == str(legacy["candidate_id"])
    assert len(ordered) == 3
    assert len({str(row["facility_group_id"]) for row in ordered}) == 3
    assert metadata["legacy_first_facility_group_id"] == str(
        legacy["facility_group_id"]
    )


def test_resume_keeps_max_attempts_as_a_lifetime_family_cap() -> None:
    assert list(remaining_attempt_numbers(0, 4)) == [0, 1, 2, 3]
    assert list(remaining_attempt_numbers(2, 4)) == [2, 3]
    assert list(remaining_attempt_numbers(4, 4)) == []
    assert list(remaining_attempt_numbers(7, 4)) == []
    assert list(remaining_attempt_numbers(1, 4, retry_closed=True)) == []
    with pytest.raises(ValueError, match="recorded_attempt_count"):
        remaining_attempt_numbers(-1, 4)


def test_terminal_connectivity_contract_failure_is_not_retryable() -> None:
    error = TerminalConnectivityError(
        "NONRETRYABLE_TERMINAL_CONNECTIVITY: synthetic contract failure"
    )
    assert rejection_is_retryable(error) is False
    assert rejection_is_retryable(ValueError("stochastic rejection")) is True
    assert worker_error_is_fatal(ValueError("stochastic rejection")) is False
    assert worker_error_is_fatal(TypeError("programming fault")) is True
    assert rejection_is_retryable(TypeError("programming fault")) is False
    c3_error = JointSupportConsistencyError(
        "synthetic replay mismatch",
        capacity_contract_fingerprint="ccf-test",
    )
    assert rejection_is_retryable(c3_error) is False
    assert c3_error.roster_fingerprint == "ccf-test"


def test_nested_progress_detail_named_stage_does_not_collide() -> None:
    observed: list[tuple[str, dict[str, object]]] = []
    progress = _make_progress_emitter(
        lambda event_name, details: observed.append((event_name, dict(details)))
    )
    progress(
        "terminal_selection.customer_preflight",
        stage="customer_preflight",
        status="completed",
    )
    assert observed == [
        (
            "terminal_selection.customer_preflight",
            {"stage": "customer_preflight", "status": "completed"},
        )
    ]


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
        Path(__file__).parents[1] / "configs" / "us_reference_instance_profile_v2.json"
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
            "instance_speed_kph": [36.0, 18.0],
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


def test_depot_star_quarantines_one_way_endpoint_that_cannot_return() -> None:
    profile = load_reference_profile(
        Path(__file__).parents[1] / "configs" / "us_reference_instance_profile_v2.json"
    )
    graph = nx.MultiDiGraph()
    for node, x in (("a", 0.0), ("b", 0.001), ("c", 0.002)):
        graph.add_node(node, x=x, y=0.0)
    edges = [("a", "b"), ("b", "a"), ("b", "c")]
    for u, v in edges:
        graph.add_edge(u, v, key=0)
    road_state = pd.DataFrame(
        {
            "edge_u": [u for u, _ in edges],
            "edge_v": [v for _, v in edges],
            "edge_key": ["0"] * len(edges),
            "length_m": [100.0] * len(edges),
            "edge_travel_time_s": [10.0] * len(edges),
            "instance_speed_kph": [36.0] * len(edges),
        }
    )
    network = PhysicalRoadNetwork(graph, road_state, profile)

    def ref(u: str, v: str, offset: float) -> str:
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

    star = network.route_depot_star(
        pd.DataFrame(
            {
                "terminal_index": [0, 1],
                "directed_projection_offsets": [
                    json.dumps(
                        [
                            json.loads(ref("a", "b", 50.0))[0],
                            json.loads(ref("b", "a", 50.0))[0],
                        ]
                    ),
                    ref("b", "c", 0.0),
                ],
                "connector_length_m": [0.0, 0.0],
            }
        )
    )
    assert star.node_outbound_reachable.tolist() == [True, True]
    assert star.node_return_reachable.tolist() == [True, False]
    assert star.connectivity_eligible.tolist() == [True, False]
    assert star.report["connectivity_quarantined_count"] == 1


def test_depot_star_quarantines_turn_only_immediate_reversal_trap() -> None:
    profile = load_reference_profile(
        Path(__file__).parents[1] / "configs" / "us_reference_instance_profile_v2.json"
    )
    graph = nx.MultiDiGraph()
    coordinates = {"a": (0.0, 0.0), "u": (0.001, 0.0), "v": (0.002, 0.0)}
    for node, (x, y) in coordinates.items():
        graph.add_node(node, x=x, y=y)
    edges = [("a", "u"), ("u", "a"), ("u", "v"), ("v", "u")]
    for left, right in edges:
        graph.add_edge(left, right, key=0)
    road_state = pd.DataFrame(
        {
            "edge_u": [left for left, _ in edges],
            "edge_v": [right for _, right in edges],
            "edge_key": ["0"] * len(edges),
            "length_m": [100.0] * len(edges),
            "edge_travel_time_s": [10.0] * len(edges),
            "instance_speed_kph": [36.0] * len(edges),
        }
    )
    network = PhysicalRoadNetwork(graph, road_state, profile)

    def ref(u: str, v: str) -> str:
        return json.dumps(
            [
                {
                    "u": u,
                    "v": v,
                    "key": "0",
                    "length_m": 100.0,
                    "offset_from_u_m": 50.0,
                    "offset_to_v_m": 50.0,
                }
            ]
        )

    star = network.route_depot_star(
        pd.DataFrame(
            {
                "terminal_index": [0, 1],
                "directed_projection_offsets": [ref("a", "u"), ref("u", "v")],
                "connector_length_m": [0.0, 0.0],
            }
        )
    )
    assert star.node_outbound_reachable.tolist() == [True, True]
    assert star.node_return_reachable.tolist() == [True, True]
    assert star.turn_outbound_reachable.tolist() == [True, True]
    assert star.turn_return_reachable.tolist() == [True, False]
    assert star.connectivity_eligible.tolist() == [True, False]


def test_cached_topology_accepts_new_family_road_state() -> None:
    profile = load_reference_profile(
        Path(__file__).parents[1] / "configs" / "us_reference_instance_profile_v2.json"
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
            "instance_speed_kph": [36.0, 18.0],
        }
    )
    cached = PhysicalRoadNetwork(graph, road_state, profile)
    next_state = road_state.copy()
    next_state["edge_travel_time_s"] = [5.0, 40.0]
    next_state["instance_speed_kph"] = [72.0, 9.0]
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
    assert network.edges["edge_travel_time_s"].tolist() == [5.0, 40.0]
    assert network.edges["instance_speed_kph"].tolist() == [72.0, 9.0]
    assert cached.edges["edge_travel_time_s"].tolist() == [10.0, 20.0]
    fresh = PhysicalRoadNetwork(graph, next_state, profile)
    pd.testing.assert_frame_equal(network.edges, fresh.edges)
    assert matrices.distance_matrix_km[0, 1] == pytest.approx(0.05)
    assert matrices.distance_path_travel_time_s[0, 1] == pytest.approx(2.5)
    assert matrices.distance_path_travel_time_s[1, 0] == pytest.approx(20.0)


def test_cached_topology_reselects_equal_length_parallel_edge() -> None:
    profile = load_reference_profile(
        Path(__file__).parents[1] / "configs" / "us_reference_instance_profile_v2.json"
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
            "instance_speed_kph": [72.0, 36.0],
        }
    )
    cached = PhysicalRoadNetwork(graph, road_state, profile)
    pair = (cached.node_to_index["a"], cached.node_to_index["b"])
    assert cached._distance_chosen_edge[pair] == 0

    next_state = road_state.copy()
    next_state["edge_travel_time_s"] = [20.0, 2.0]
    next_state["instance_speed_kph"] = [18.0, 180.0]
    network = cached.with_road_state(next_state, profile)

    assert network._distance_adjacency is cached._distance_adjacency
    assert network._distance_chosen_edge[pair] == 1


def test_running_time_path_optimizes_turn_penalties_not_only_edge_times() -> None:
    profile = load_reference_profile(
        Path(__file__).parents[1] / "configs" / "us_reference_instance_profile_v2.json"
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
            "instance_speed_kph": [edge[2] / edge[3] * 3.6 for edge in edges],
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


def test_nested_view_indices_use_customer_blocks_and_reselect_chargers() -> None:
    times = np.zeros((106, 106), dtype=np.float32)
    times[0, 1:101] = 100.0
    times[1:101, 0] = 100.0
    times[0, 101:106] = 100.0
    times[101:106, 0] = 100.0
    deltas = np.asarray(
        [[50.0, 40.0, 0.0, 100.0, 80.0]] * 5
        + [[50.0, 40.0, 100.0, 0.0, 80.0]] * 5,
        dtype=np.float32,
    )
    times[31:41, 101:106] = deltas
    times[101:106, 31:41] = deltas.T
    indices = view_parent_terminal_indices(
        {
            "scale_id": "test10",
            "customer_count": 10,
            "charging_station_count": 2,
            "branch_index": 3,
            "view_seed": 20260810,
        },
        parent_customer_count=100,
        parent_charging_station_count=5,
        running_time_matrix_s=times,
    )
    assert indices[:4].tolist() == [0, 31, 32, 33]
    assert indices[10] == 40
    assert len(indices[11:]) == 2
    assert indices[11:].tolist() != [101, 102]
