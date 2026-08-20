from __future__ import annotations

import pandas as pd
import pytest

from evrptw_stage2.selection_capsule import (
    SELECTION_CAPSULE_SCHEMA,
    SelectionCapsuleError,
    load_family_selection_capsule,
    write_task_selection_capsule,
)


def _family() -> dict[str, object]:
    return {
        "family_id": "mf_test",
        "city_slug": "test-city",
        "day_type": "weekday",
        "parent_customer_count": 2,
        "customer_superset_seed": 101,
        "road_state_seed": 202,
        "joint_support_contract_id": "c3_joint_spatial_support_v1",
        "capacity_contract_fingerprint": "capacity-test",
        "c3_selection_capsule_schema": SELECTION_CAPSULE_SCHEMA,
        "c3_selection_capsule_relpath": "capsules/task-0000",
    }


def _capsule() -> dict[str, object]:
    selected = pd.DataFrame(
        {
            "latent_service_location_id": ["customer-1", "customer-2"],
            "sampling_cluster_id": ["region-1", "region-2"],
            "structure_route_id": ["route-1", "route-2"],
            "activation_decile": [0, 1],
            "location_lon": [-118.1, -118.2],
            "location_lat": [34.1, 34.2],
            "physical_edge_id": ["edge-1", "edge-2"],
            "directed_projection_offsets": [{"a": 1}, {"a": 2}],
            "connector_length_m": [1.0, 2.0],
            "road_projection_node_id": ["road-1", "road-2"],
            "service_access_node_id": ["access-1", "access-2"],
            "anchor_scc_id": ["scc-1", "scc-1"],
            "community_id": ["community-1", "community-2"],
            "service_location_type": ["house", "apartment"],
            "residential_unit_band": ["single", "multi"],
            "residential_units": [1, 4],
            "depot_running_time_s": [10.0, 20.0],
        }
    )
    baseline = pd.DataFrame(
        {
            "latent_service_location_id": ["customer-2", "customer-1"],
            "community_id": ["community-2", "community-1"],
            "radial_decile": [0, 1],
            "depot_running_time_s": [20.0, 10.0],
        }
    )
    return {
        "family_id": "mf_test",
        "binding": {
            "family_id": "mf_test",
            "city_slug": "test-city",
            "day_type": "weekday",
            "parent_customer_count": 2,
            "customer_superset_seed": 101,
            "road_state_seed": 202,
            "selected_depot_id": "depot-1",
            "selected_structure_source_ids": ["source-1"],
            "joint_support_contract_id": "c3_joint_spatial_support_v1",
            "capacity_contract_fingerprint": "capacity-test",
        },
        "selected_customers": selected,
        "radial_baseline": baseline,
        "territory_report": {
            "split_pool_count": 20,
            "connectivity_eligible_count": 19,
            "territory_count": 18,
        },
        "spatial_activation_metadata": {
            "schema": "evrptw_spatial_activation_report_v3",
            "quota": {
                "required_decile_counts": [1, 1],
                "available_decile_counts": [9, 9],
            },
        },
    }


def test_selection_capsule_round_trip_and_exact_binding(tmp_path) -> None:
    family = _family()
    base = tmp_path / str(family["c3_selection_capsule_relpath"])
    report = write_task_selection_capsule(base, [_capsule()])

    loaded = load_family_selection_capsule(
        tmp_path,
        family,
        selected_depot_id="depot-1",
        selected_structure_source_ids=["source-1"],
    )

    assert report["schema"] == SELECTION_CAPSULE_SCHEMA
    assert report["hash_validation_performed"] is False
    assert loaded is not None
    assert loaded.selected_customers["latent_service_location_id"].tolist() == [
        "customer-1",
        "customer-2",
    ]
    assert loaded.radial_baseline["latent_service_location_id"].tolist() == [
        "customer-2",
        "customer-1",
    ]


def test_selection_capsule_rejects_seed_mismatch(tmp_path) -> None:
    family = _family()
    write_task_selection_capsule(
        tmp_path / str(family["c3_selection_capsule_relpath"]), [_capsule()]
    )
    family["road_state_seed"] = 999

    with pytest.raises(SelectionCapsuleError, match="binding mismatch"):
        load_family_selection_capsule(
            tmp_path,
            family,
            selected_depot_id="depot-1",
            selected_structure_source_ids=["source-1"],
        )


def test_selection_capsule_rejects_path_escape(tmp_path) -> None:
    family = _family()
    family["c3_selection_capsule_relpath"] = "../outside"

    with pytest.raises(SelectionCapsuleError, match="output-root relative"):
        load_family_selection_capsule(
            tmp_path,
            family,
            selected_depot_id="depot-1",
            selected_structure_source_ids=["source-1"],
        )
