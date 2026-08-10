from __future__ import annotations

import json
import shutil
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from evrptw_cle.cle import package_cle, verify_cle
from evrptw_cle.util import sha256_file, write_json


def test_cle_build_verifier_checks_output_hashes(tmp_path: Path) -> None:
    output = tmp_path / "evidence.txt"
    output.write_text("frozen", encoding="utf-8")
    manifest = {
        "status": "technical_cle_build_complete_release_blocked",
        "release_eligible": False,
        "release_blocker_count": 1,
        "outputs": {"evidence": output.name},
        "output_sha256": {"evidence": sha256_file(output)},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_cle(tmp_path)["passed"]

    output.write_text("changed", encoding="utf-8")
    report = verify_cle(tmp_path)
    assert not report["passed"]
    assert "SHA-256 mismatch for evidence" in report["errors"]


def test_strict_portability_rejects_build_only_cle(tmp_path: Path) -> None:
    output = tmp_path / "evidence.txt"
    output.write_text("frozen", encoding="utf-8")
    manifest = {
        "status": "technical_cle_build_complete_release_blocked",
        "release_eligible": False,
        "release_blocker_count": 1,
        "outputs": {"evidence": output.name},
        "output_sha256": {"evidence": sha256_file(output)},
    }
    write_json(tmp_path / "manifest.json", manifest)
    technical = verify_cle(tmp_path)
    portable = verify_cle(tmp_path, require_portable=True)
    assert technical["passed"]
    assert not technical["portable"]
    assert not portable["passed"]
    assert "portable manifest omits operational_graph" in portable["errors"]


def test_verifier_reports_missing_connectivity_column_without_crashing(
    tmp_path: Path,
) -> None:
    location_path = tmp_path / "service_locations/latent_locations.parquet"
    location_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {
            "latent_service_location_id": ["legacy"],
            "active_customer": [False],
            "customer_release_eligible": [False],
            "cle_candidate_eligible": [True],
        },
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    ).to_parquet(location_path, index=False)
    write_json(
        tmp_path / "manifest.json",
        {
            "status": "legacy_work_artifact",
            "release_eligible": False,
            "release_blocker_count": 1,
            "layer_counts": {"latent_service_location_candidates": 1},
            "outputs": {
                "latent_locations": "service_locations/latent_locations.parquet"
            },
            "output_sha256": {
                "latent_locations": sha256_file(location_path),
            },
        },
    )

    report = verify_cle(tmp_path)

    assert not report["passed"]
    assert any(
        "protected_roundtrip_eligible" in error for error in report["errors"]
    )


def test_package_cle_copies_runtime_graph_and_survives_source_removal(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    source_cle = work / "cles/test-city"
    source_cle.mkdir(parents=True)
    graph = work / "cities/test-city/graph_operational.graphml"
    road_manifest = work / "cities/test-city/manifest.json"
    graph.parent.mkdir(parents=True)
    graph.write_text("<graphml />", encoding="utf-8")
    write_json(
        road_manifest,
        {
            "schema": "road_manifest_test",
            "provenance": {"pbf_file": str(tmp_path / "source.osm.pbf")},
        },
    )
    for relative, payload in (
        ("boundary/admin_boundary.geojson", "{}"),
        ("boundary/service_boundary.geojson", "{}"),
        (
            "infrastructure/facility_manifest.json",
            json.dumps(
                {"inputs": {"afdc": {"path": str(tmp_path / "afdc.csv")}}}
            ),
        ),
        (
            "profiles/speed_manifest.json",
            json.dumps({"graph": {"path": str(graph)}}),
        ),
    ):
        path = source_cle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    graph_reference = source_cle / "graph/graph_reference.json"
    graph_reference.parent.mkdir(parents=True)
    write_json(
        graph_reference,
        {
            "schema": "evrptw_graph_reference_v1",
            "operational_graph": {
                "path": str(graph.resolve()),
                "sha256": sha256_file(graph),
            },
            "road_manifest": {
                "path": str(road_manifest.resolve()),
                "sha256": sha256_file(road_manifest),
            },
        },
    )
    manifest = {
        "status": "technical_cle_build_complete_release_blocked",
        "release_eligible": False,
        "release_blocker_count": 1,
        "outputs": {"graph_reference": "graph/graph_reference.json"},
        "output_sha256": {"graph_reference": sha256_file(graph_reference)},
    }
    write_json(source_cle / "manifest.json", manifest)

    destination = tmp_path / "release/CLE_v1/test-city"
    result = package_cle(
        source_cle_dir=source_cle,
        graph_path=graph,
        road_manifest_path=road_manifest,
        destination_cle_dir=destination,
    )
    assert result["passed"]
    assert result["portable"]
    assert (destination / "graph/graph_operational.graphml").is_file()
    reference = json.loads(
        (destination / "graph/graph_reference.json").read_text(encoding="utf-8")
    )
    assert reference["operational_graph"]["path"] == "graph_operational.graphml"
    packaged_facilities = json.loads(
        (destination / "infrastructure/facility_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert packaged_facilities["inputs"]["afdc"]["path"] == "afdc.csv"
    assert "absolute build paths removed" in packaged_facilities[
        "portable_provenance_path_policy"
    ]
    packaged_roads = json.loads(
        (destination / "graph/road_manifest.json").read_text(encoding="utf-8")
    )
    assert packaged_roads["provenance"]["pbf_file"] == "source.osm.pbf"

    reused = package_cle(
        source_cle_dir=source_cle,
        graph_path=graph,
        road_manifest_path=road_manifest,
        destination_cle_dir=destination,
    )
    assert reused["packaging_action"] == "reused_existing_verified_package"

    manifest["status"] = "new_work_candidate"
    write_json(source_cle / "manifest.json", manifest)
    with pytest.raises(FileExistsError, match="invalid or stale"):
        package_cle(
            source_cle_dir=source_cle,
            graph_path=graph,
            road_manifest_path=road_manifest,
            destination_cle_dir=destination,
        )

    shutil.rmtree(work)
    verification = verify_cle(destination, require_portable=True)
    assert verification["passed"]
