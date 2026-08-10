#!/usr/bin/env python3
"""Extract city charging POIs from the same frozen OSM PBFs used for roads."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from evrptw_cle.util import sha256_file, write_json


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _source_id(export_id: Any) -> str:
    value = str(export_id or "")
    if len(value) < 2:
        return value
    prefix, numeric_text = value[0], value[1:]
    try:
        numeric = int(numeric_text)
    except ValueError:
        return value
    if prefix == "n":
        return f"n{numeric}"
    if prefix == "w":
        return f"w{numeric}"
    if prefix == "r":
        return f"r{numeric}"
    if prefix == "a" and numeric % 2 == 0:
        return f"w{numeric // 2}"
    if prefix == "a":
        return f"r{(numeric - 1) // 2}"
    return value


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr.strip()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset", type=Path, default=Path("configs/top10_us_cities_population_v1.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sources/osm/osm_charging_pois_top10.csv"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    preset_path = args.preset if args.preset.is_absolute() else root / args.preset
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite OSM charging POI table: {output}")
    osmium = shutil.which("osmium")
    if osmium is None:
        raise RuntimeError("osmium executable is required")
    preset = json.loads(preset_path.read_text(encoding="utf-8"))
    frames: list[pd.DataFrame] = []
    source_records: list[dict[str, Any]] = []
    for item in preset["cities"]:
        slug = str(item["slug"])
        pbf = _resolve(preset_path.parent, item["pbf_file"])
        boundary = _resolve(preset_path.parent, item["query_mask_file"])
        if not pbf.is_file() or not boundary.is_file():
            raise FileNotFoundError(f"{slug}: missing PBF or service boundary")
        with tempfile.TemporaryDirectory(prefix=f"{slug}-charging-poi-") as temp:
            temp_path = Path(temp)
            city_pbf = temp_path / "city.osm.pbf"
            poi_pbf = temp_path / "charging.osm.pbf"
            geojson = temp_path / "charging.geojson"
            _run(
                [
                    osmium,
                    "extract",
                    "--polygon",
                    str(boundary),
                    "--strategy",
                    "complete_ways",
                    "--output",
                    str(city_pbf),
                    str(pbf),
                ]
            )
            _run(
                [
                    osmium,
                    "tags-filter",
                    "--output",
                    str(poi_pbf),
                    str(city_pbf),
                    "nwr/amenity=charging_station",
                    "nwr/man_made=charge_point",
                ]
            )
            _run(
                [
                    osmium,
                    "export",
                    "--add-unique-id",
                    "type_id",
                    "--output-format",
                    "geojson",
                    "--output",
                    str(geojson),
                    str(poi_pbf),
                ]
            )
            frame = gpd.read_file(geojson).to_crs(4326)
            if frame.empty:
                continue
            points = frame.geometry.representative_point()
            normalized = pd.DataFrame(
                {
                    "city_slug": slug,
                    "source_osm_id": frame["id"].map(_source_id),
                    "name": frame.get("name", ""),
                    "operator": frame.get("operator", ""),
                    "brand": frame.get("brand", ""),
                    "addr_housenumber": frame.get("addr:housenumber", ""),
                    "addr_street": frame.get("addr:street", ""),
                    "addr_city": frame.get("addr:city", ""),
                    "addr_state": frame.get("addr:state", ""),
                    "addr_postcode": frame.get("addr:postcode", ""),
                    "longitude": points.x,
                    "latitude": points.y,
                }
            )
            frames.append(normalized)
        source_records.append(
            {"city_slug": slug, "pbf": str(pbf), "pbf_sha256": sha256_file(pbf)}
        )
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined = combined.drop_duplicates(["city_slug", "source_osm_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    manifest = {
        "schema": "evrptw_osm_charging_pois_v1",
        "row_count": len(combined),
        "preset": str(preset_path.resolve()),
        "sources": source_records,
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
