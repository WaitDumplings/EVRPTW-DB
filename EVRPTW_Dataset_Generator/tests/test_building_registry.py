from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from evrptw_cle.building_registry import (
    extract_registered_city,
    preflight_registered_city,
)
from evrptw_cle.util import sha256_file


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    config_dir = tmp_path / "configs"
    boundary_dir = tmp_path / "boundaries/sample-city"
    source_root = tmp_path / "sources"
    config_dir.mkdir()
    boundary_dir.mkdir(parents=True)
    source_root.mkdir()

    boundary_path = boundary_dir / "land_boundary.geojson"
    gpd.GeoDataFrame(geometry=[box(-118.3, 33.9, -118.0, 34.2)], crs="EPSG:4326").to_file(
        boundary_path, driver="GeoJSON"
    )
    source_path = source_root / "Sample.geojson"
    source_path.write_text(
        "{\n"
        '  "type":"FeatureCollection",\n'
        '  "features":\n'
        "  [\n"
        '    {"type":"Feature","geometry":{"type":"Polygon","coordinates":'
        "[[[-118.2,34.0],[-118.1,34.0],[-118.1,34.1],[-118.2,34.1],[-118.2,34.0]]]},"
        '"properties":{"release":2,"capture_dates_range":"2020"}},\n'
        '    {"type":"Feature","geometry":{"type":"Polygon","coordinates":'
        "[[[-117.2,34.0],[-117.1,34.0],[-117.1,34.1],[-117.2,34.1],[-117.2,34.0]]]},"
        '"properties":{"release":1,"capture_dates_range":""}}\n'
        "  ]\n"
        "}\n",
        encoding="utf-8",
    )
    official_path = config_dir / "official.json"
    _write_json(
        official_path,
        {"preset_id": "sample", "cities": [{"slug": "sample-city"}]},
    )
    config_path = config_dir / "registry.json"
    _write_json(
        config_path,
        {
            "schema": "evrptw_city_building_registry_v1",
            "registry_id": "sample_registry",
            "official_city_preset": "official.json",
            "source_dataset": "Microsoft USBuildingFootprints",
            "expected_geometry_type": "Polygon",
            "membership_rule": "representative point covered by boundary",
            "polygon_policy": "retain complete polygon",
            "building_id_rule": "city plus source index",
            "batch_size": 1,
            "density_grid_m": 500,
            "cities": {
                "sample-city": {
                    "label": "Sample City",
                    "state": "Sample",
                    "state_fips": "00",
                    "source_file": source_path.name,
                    "source_sha256": sha256_file(source_path),
                    "source_bytes": source_path.stat().st_size,
                    "source_feature_count": 2,
                    "boundary_file": str(boundary_path),
                    "boundary_sha256": sha256_file(boundary_path),
                    "area_crs": "EPSG:32611",
                }
            },
        },
    )
    return config_path, source_root, tmp_path / "outputs"


def test_registered_city_preflight_checks_source_and_boundary(tmp_path: Path) -> None:
    config_path, source_root, output_root = _fixture(tmp_path)
    report = preflight_registered_city(
        config_path=config_path,
        city_slug="sample-city",
        source_root=source_root,
        output_root=output_root,
    )
    assert report["passed"]
    assert report["checks"]["source_sha256"] == json.loads(
        config_path.read_text(encoding="utf-8")
    )["cities"]["sample-city"]["source_sha256"]


def test_registered_city_extracts_in_staging_and_refuses_overwrite(tmp_path: Path) -> None:
    config_path, source_root, output_root = _fixture(tmp_path)
    manifest = extract_registered_city(
        config_path=config_path,
        city_slug="sample-city",
        source_root=source_root,
        output_root=output_root,
    )
    assert manifest["status"] == "complete"
    assert manifest["summary"]["building_count"] == 1
    assert (output_root / "sample-city/building_summary.json").exists()
    assert (output_root / "manifests/sample-city.json").exists()
    assert not list((output_root / ".staging").iterdir())
    with pytest.raises(FileExistsError):
        extract_registered_city(
            config_path=config_path,
            city_slug="sample-city",
            source_root=source_root,
            output_root=output_root,
        )


def test_registered_city_preflight_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    config_path, source_root, output_root = _fixture(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["cities"]["sample-city"]["source_sha256"] = "0" * 64
    _write_json(config_path, payload)
    report = preflight_registered_city(
        config_path=config_path,
        city_slug="sample-city",
        source_root=source_root,
        output_root=output_root,
    )
    assert not report["passed"]
    assert "source SHA-256 differs" in " ".join(report["errors"])


def test_registered_city_refuses_existing_manifest_before_extraction(tmp_path: Path) -> None:
    config_path, source_root, output_root = _fixture(tmp_path)
    manifest_path = output_root / "manifests/sample-city.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="run manifest"):
        extract_registered_city(
            config_path=config_path,
            city_slug="sample-city",
            source_root=source_root,
            output_root=output_root,
        )
    assert not (output_root / ".staging").exists()
