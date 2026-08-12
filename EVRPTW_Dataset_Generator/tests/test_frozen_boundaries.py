from __future__ import annotations

import json
from pathlib import Path

from evrptw_cle.util import sha256_file

ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_ROOT = ROOT / "boundaries/us-11city-2025"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_boundary_metadata_is_portable_and_hash_aligned() -> None:
    manifest = _read_json(BOUNDARY_ROOT / "manifest.json")
    building_registry = _read_json(ROOT / "configs/us_11city_building_extraction_v1.json")

    assert len(manifest["cities"]) == 11
    assert set(building_registry["cities"]) == {
        record["slug"] for record in manifest["cities"]
    }

    for record in manifest["cities"]:
        slug = record["slug"]
        city_root = BOUNDARY_ROOT / slug
        metadata = _read_json(city_root / "metadata.json")

        assert not Path(metadata["place_source_file"]).is_absolute()
        assert not Path(metadata["county_source_file"]).is_absolute()
        assert all(
            not Path(source["file"]).is_absolute()
            for source in metadata["land_mask_qa"]["areawater_sources"]
        )

        admin_boundary = city_root / "admin_boundary.geojson"
        land_boundary = city_root / "land_boundary.geojson"
        assert sha256_file(admin_boundary) == record["admin_boundary_sha256"]
        assert sha256_file(land_boundary) == record["land_boundary_sha256"]
        assert (
            building_registry["cities"][slug]["boundary_sha256"]
            == record["land_boundary_sha256"]
        )
