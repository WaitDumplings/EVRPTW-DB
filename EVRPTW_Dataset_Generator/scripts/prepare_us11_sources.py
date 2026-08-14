#!/usr/bin/env python3
"""Prepare every public source required by the frozen U.S. 11-city pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from download_hpms_city import download_hpms_city
from download_microsoft_state_buildings import download_state_buildings

GENERATOR_ROOT = Path(__file__).resolve().parents[1]
PRESET_PATH = GENERATOR_ROOT / "configs/us_11city_population_v1.json"
BUILDING_REGISTRY_PATH = GENERATOR_ROOT / "configs/us_11city_building_extraction_v1.json"
STATE_REGISTRY_PATH = GENERATOR_ROOT / "configs/us_states_v1.json"
BLOCK_GROUP_PRESET_PATH = GENERATOR_ROOT / "configs/us_census_block_groups_v1.json"
SOURCES_ROOT = GENERATOR_ROOT / "data/sources"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _run(command: list[str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=GENERATOR_ROOT, check=True)


def _state_context() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    state_registry = _read_json(STATE_REGISTRY_PATH)
    states_by_fips = {
        str(item["fips"]): item for item in state_registry["states"]
    }
    building_registry = _read_json(BUILDING_REGISTRY_PATH)
    city_state_fips = {
        str(slug): str(item["state_fips"])
        for slug, item in building_registry["cities"].items()
    }
    return states_by_fips, city_state_fips


def source_plan() -> list[dict[str, str]]:
    """Return the complete, deduplicated fixed-cohort source contract."""

    preset = _read_json(PRESET_PATH)
    building_registry = _read_json(BUILDING_REGISTRY_PATH)
    block_groups = _read_json(BLOCK_GROUP_PRESET_PATH)
    plan: list[dict[str, str]] = []

    pbf_paths = {
        _resolve(PRESET_PATH, str(city["pbf_file"])) for city in preset["cities"]
    }
    plan.extend(
        {"layer": "osm_pbf", "id": path.name, "path": str(path)}
        for path in sorted(pbf_paths)
    )

    building_files = {
        str(city["source_file"])
        for city in building_registry["cities"].values()
    }
    building_root = SOURCES_ROOT / "microsoft-us-building-footprints"
    plan.extend(
        {
            "layer": "microsoft_buildings",
            "id": state_file,
            "path": str(building_root / state_file),
        }
        for state_file in sorted(building_files)
    )

    hpms_root = SOURCES_ROOT / "hpms"
    plan.extend(
        {
            "layer": "hpms_city_window",
            "id": str(city["slug"]),
            "path": str(hpms_root / f"{city['slug']}.geojson"),
        }
        for city in preset["cities"]
    )

    afdc_root = SOURCES_ROOT / "afdc"
    plan.extend(
        [
            {
                "layer": "afdc_raw",
                "id": "us_public_available_electric",
                "path": str(afdc_root / "afdc_us_public_available_electric.csv"),
            },
            {
                "layer": "afdc_census_evidence",
                "id": "us_11city_states",
                "path": str(afdc_root / "afdc_census_address_anchors.csv"),
            },
            {
                "layer": "afdc_resolved",
                "id": "us_11city_v1",
                "path": str(
                    afdc_root
                    / "afdc_us_public_available_electric_resolved_us_11city_v1.csv"
                ),
            },
            {
                "layer": "osm_charging_pois",
                "id": "us_11city",
                "path": str(SOURCES_ROOT / "osm/osm_charging_pois_us_11city.csv"),
            },
        ]
    )

    block_group_root = SOURCES_ROOT / "census_block_groups_2025"
    plan.extend(
        {
            "layer": "census_block_groups",
            "id": state,
            "path": str(block_group_root / f"tl_2025_{fips}_bg.zip"),
        }
        for state, fips in sorted(block_groups["states"].items())
    )
    return plan


def inspect_sources() -> dict[str, Any]:
    records = []
    for item in source_plan():
        path = Path(item["path"])
        available = path.is_file() and path.stat().st_size > 0
        records.append(
            {
                **item,
                "available": available,
                "bytes": path.stat().st_size if available else 0,
            }
        )
    missing = [item for item in records if not item["available"]]
    return {
        "schema": "evrptw_us11_source_readiness_v1",
        "status": "complete" if not missing else "incomplete",
        "required_file_count": len(records),
        "available_file_count": len(records) - len(missing),
        "missing_file_count": len(missing),
        "missing": missing,
        "sources": records,
    }


def _prepare_pbf() -> None:
    _run(
        [
            sys.executable,
            "scripts/fetch_pbf_sources.py",
            "--preset",
            str(PRESET_PATH),
            "--manifest",
            str(SOURCES_ROOT / "geofabrik/source_manifest.json"),
            "--skip-sha256",
        ]
    )


def _prepare_buildings() -> None:
    registry = _read_json(BUILDING_REGISTRY_PATH)
    state_files = sorted(
        {str(city["source_file"]) for city in registry["cities"].values()}
    )
    output_root = SOURCES_ROOT / "microsoft-us-building-footprints"
    for state_file in state_files:
        download_state_buildings(state_file=state_file, output_root=output_root)


def _prepare_hpms() -> None:
    preset = _read_json(PRESET_PATH)
    states_by_fips, city_state_fips = _state_context()
    output_root = SOURCES_ROOT / "hpms"
    for city in preset["cities"]:
        slug = str(city["slug"])
        state = states_by_fips[city_state_fips[slug]]
        service_url = (
            "https://geo.dot.gov/server/rest/services/Hosted/"
            f"{state['hpms_service_token']}_2018_PR/FeatureServer"
        )
        download_hpms_city(
            service_url=service_url,
            boundary_path=_resolve(PRESET_PATH, str(city["query_mask_file"])),
            output_path=output_root / f"{slug}.geojson",
        )


def _prepare_afdc_raw() -> Path:
    output = SOURCES_ROOT / "afdc/afdc_us_public_available_electric.csv"
    if output.is_file() and output.stat().st_size > 0:
        print(f"REUSE {output}", flush=True)
        return output
    _run(
        [
            sys.executable,
            "scripts/download_afdc_snapshot.py",
            "--output",
            str(output),
        ]
    )
    return output


def _prepare_osm_charging_pois() -> Path:
    output = SOURCES_ROOT / "osm/osm_charging_pois_us_11city.csv"
    if output.is_file() and output.stat().st_size > 0:
        print(f"REUSE {output}", flush=True)
        return output
    _run(
        [
            sys.executable,
            "scripts/extract_osm_charging_pois.py",
            "--preset",
            str(PRESET_PATH),
            "--output",
            str(output),
        ]
    )
    return output


def _prepare_census_address_evidence(afdc_path: Path) -> Path:
    combined = SOURCES_ROOT / "afdc/afdc_census_address_anchors.csv"
    if combined.is_file() and combined.stat().st_size > 0:
        print(f"REUSE {combined}", flush=True)
        return combined

    states_by_fips, city_state_fips = _state_context()
    state_abbreviations = sorted(
        {str(states_by_fips[fips]["abbr"]) for fips in city_state_fips.values()}
    )
    state_outputs: list[Path] = []
    for abbreviation in state_abbreviations:
        output = (
            SOURCES_ROOT
            / "afdc"
            / f"afdc_census_address_anchors_{abbreviation.casefold()}.csv"
        )
        state_outputs.append(output)
        if output.is_file() and output.stat().st_size > 0:
            print(f"REUSE {output}", flush=True)
            continue
        _run(
            [
                sys.executable,
                "scripts/geocode_afdc_addresses_census.py",
                "--afdc",
                str(afdc_path),
                "--state",
                abbreviation,
                "--output",
                str(output),
            ]
        )

    frames = [pd.read_csv(path, low_memory=False) for path in state_outputs]
    evidence = pd.concat(frames, ignore_index=True)
    if evidence["afdc_id"].duplicated().any():
        duplicates = evidence.loc[evidence["afdc_id"].duplicated(), "afdc_id"].head()
        raise ValueError(f"Census evidence contains duplicate AFDC IDs: {duplicates.tolist()}")
    evidence = evidence.sort_values("afdc_id", kind="stable").reset_index(drop=True)
    combined.parent.mkdir(parents=True, exist_ok=True)
    staged = combined.with_suffix(".csv.part")
    evidence.to_csv(staged, index=False)
    staged.replace(combined)
    manifest = {
        "schema": "evrptw_afdc_census_address_anchors_cohort_v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "state_filters": state_abbreviations,
        "row_count": len(evidence),
        "state_outputs": [str(path.resolve()) for path in state_outputs],
        "output": str(combined.resolve()),
        "semantic_limit": (
            "Census coordinates corroborate street-address anchors and are not exact "
            "charging-space coordinates."
        ),
    }
    combined.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return combined


def _prepare_resolved_afdc(
    afdc_path: Path, census_path: Path, osm_pois_path: Path
) -> None:
    output = (
        SOURCES_ROOT
        / "afdc/afdc_us_public_available_electric_resolved_us_11city_v1.csv"
    )
    if output.is_file() and output.stat().st_size > 0:
        print(f"REUSE {output}", flush=True)
        return
    _run(
        [
            sys.executable,
            "scripts/resolve_afdc_coordinates.py",
            "--afdc",
            str(afdc_path),
            "--census-results",
            str(census_path),
            "--osm-pois",
            str(osm_pois_path),
            "--output",
            str(output),
        ]
    )


def _prepare_block_groups() -> None:
    _run(
        [
            sys.executable,
            "scripts/fetch_census_block_groups.py",
            "--preset",
            str(BLOCK_GROUP_PRESET_PATH),
            "--output-dir",
            str(SOURCES_ROOT / "census_block_groups_2025"),
        ]
    )


def prepare_sources() -> dict[str, Any]:
    _prepare_pbf()
    _prepare_buildings()
    _prepare_hpms()
    afdc_path = _prepare_afdc_raw()
    osm_pois_path = _prepare_osm_charging_pois()
    census_path = _prepare_census_address_evidence(afdc_path)
    _prepare_resolved_afdc(afdc_path, census_path, osm_pois_path)
    _prepare_block_groups()
    readiness = inspect_sources()
    if readiness["status"] != "complete":
        missing_paths = [item["path"] for item in readiness["missing"]]
        raise RuntimeError(f"Source preparation finished with missing files: {missing_paths}")
    manifest = {
        **readiness,
        "schema": "evrptw_us11_source_preparation_manifest_v1",
        "prepared_utc": datetime.now(UTC).isoformat(),
        "network_policy": "download missing public inputs; reuse nonempty existing files",
        "nsi_policy": "cached when available; otherwise downloaded in deterministic CLE tiles",
        "amazon_policy": "Stage-2 only; generate_instances.sh prepares it on demand",
    }
    manifest_path = SOURCES_ROOT / "us11_source_preparation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Report required/missing files without downloading or modifying sources.",
    )
    args = parser.parse_args()
    result = inspect_sources() if args.check_only else prepare_sources()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
