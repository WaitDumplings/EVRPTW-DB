from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point, box

from evrptw_cle.customer_gate import verify_customer_gate
from evrptw_cle.util import sha256_file


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (repo_root / "configs/gate02_customer_location_contract_v1.json").read_text(
            encoding="utf-8"
        )
    )
    contract["road_access"]["selection_status"] = "frozen_from_ten_city_audit"
    contract["road_access"]["selected_threshold_m"] = 100
    contract_path = tmp_path / "configs/contract.json"
    _write_json(contract_path, contract)

    slug = "sample-city"
    preset_path = tmp_path / "configs/preset.json"
    _write_json(
        preset_path,
        {
            "preset_id": "top10_us_cities_population_v1",
            "cities": [{"slug": slug, "display_name": "Sample City"}],
        },
    )
    boundary_root = tmp_path / "boundaries"
    boundary_path = boundary_root / slug / "land_boundary.geojson"
    boundary_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(geometry=[box(-1, -1, 1, 1)], crs="EPSG:4326").to_file(
        boundary_path, driver="GeoJSON"
    )

    city_root = tmp_path / "cities"
    graph_path = city_root / slug / "graph_operational.graphml"
    graph_path.parent.mkdir(parents=True)
    graph_path.write_text("frozen graph fixture", encoding="utf-8")
    _write_json(
        city_root / slug / "manifest.json",
        {"operational_graph": graph_path.name},
    )

    customer_root = tmp_path / "customers"
    customer_dir = customer_root / slug
    customer_dir.mkdir(parents=True)
    location_path = customer_dir / "latent_service_locations.parquet"
    row = {
        "latent_service_location_id": "sample_1",
        "microsoft_building_id": "msft_1",
        "geometry_source": "microsoft_usbuildingfootprints_polygon",
        "geometry_match_method": "nsi_point_within_microsoft_footprint",
        "geometry_match_distance_m": 0.0,
        "footprint_area_factor": 1.2,
        "geometry_evidence_tier": "G1_containment",
        "nsi_fd_ids": "[1]",
        "raw_nsi_occtypes": "[RES3B]",
        "raw_nsi_resunits_sum": 3,
        "residential_units": 3,
        "residential_units_lower": 3,
        "residential_units_upper": 3,
        "units_evidence": "nsi_resunits_positive",
        "unit_evidence_tier": "U1_positive_resunits",
        "residential_unit_band": "units_2_4",
        "service_location_type": "small_apt",
        "type_evidence": "nsi_resunits",
        "mixed_use_flag": False,
        "source_confidence_tier": "G1_U1",
        "geometry_core_eligible": True,
        "road_anchor_method": "footprint_boundary_to_operational_edge_projection",
        "access_layer": "operational_public",
        "connector_kind": "through_road",
        "legal_access_tier": "operational_eligible",
        "physical_edge_id": "edge_1",
        "directed_edge_refs": "[[1,2,0],[2,1,0]]",
        "edge_u": "1",
        "edge_v": "2",
        "edge_key": "0",
        "road_access_distance_m": 20.0,
        "road_anchor_lon": 0.0,
        "road_anchor_lat": 0.0,
        "access_threshold_m": 100,
        "road_access_default_eligible": True,
        "customer_release_eligible": True,
        "quarantine_reason": "",
        "active_customer": False,
    }
    gpd.GeoDataFrame([row], geometry=[Point(0, 0)], crs="EPSG:4326").to_parquet(
        location_path, index=False
    )
    manifest = {
        "schema": "evrptw_customer_cle_v2",
        "status": "gate02_candidate",
        "city_slug": slug,
        "boundary": {"sha256": sha256_file(boundary_path)},
        "road_graph": {"sha256": sha256_file(graph_path)},
        "source": {
            "nsi": {"dataset": "USACE National Structure Inventory 2026 Base"},
            "microsoft_footprints": {
                "dataset": "Microsoft USBuildingFootprints",
                "sha256": "a" * 64,
            },
        },
        "physical_location": {
            "core_match_methods": [
                "nsi_point_within_microsoft_footprint",
                "nearest_microsoft_footprint_area_consistent"
            ],
            "near_match_manual_audit_passed": True
        },
        "classification": {"default_house_count": 0},
        "road_access": {
            "frozen_acceptance_threshold_m": 100,
        },
        "output_sha256": {"latent_service_locations": sha256_file(location_path)},
    }
    _write_json(customer_dir / "customer_cle_manifest.json", manifest)
    return preset_path, contract_path, boundary_root, city_root, customer_root


def test_customer_gate_passes_frozen_physical_location_contract(tmp_path: Path) -> None:
    inputs = _build_fixture(tmp_path)
    report = verify_customer_gate(*inputs, expected_city_count=1)
    assert report["passed"] is True
    assert report["summary"]["passed_city_count"] == 1


def test_customer_gate_rejects_default_house(tmp_path: Path) -> None:
    inputs = _build_fixture(tmp_path)
    manifest_path = inputs[-1] / "sample-city/customer_cle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["classification"]["default_house_count"] = 1
    _write_json(manifest_path, manifest)
    report = verify_customer_gate(*inputs, expected_city_count=1)
    assert report["passed"] is False
    assert any("default_house_count" in error for error in report["cities"][0]["errors"])


def test_customer_gate_rejects_unit_type_mismatch(tmp_path: Path) -> None:
    inputs = _build_fixture(tmp_path)
    location_path = inputs[-1] / "sample-city/latent_service_locations.parquet"
    frame = gpd.read_parquet(location_path)
    frame["service_location_type"] = "house"
    frame.to_parquet(location_path, index=False)
    manifest_path = inputs[-1] / "sample-city/customer_cle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_sha256"]["latent_service_locations"] = sha256_file(location_path)
    _write_json(manifest_path, manifest)
    report = verify_customer_gate(*inputs, expected_city_count=1)
    assert report["passed"] is False
    assert any("inconsistent with units" in error for error in report["cities"][0]["errors"])
