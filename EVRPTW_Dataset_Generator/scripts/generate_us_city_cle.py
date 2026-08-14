#!/usr/bin/env python3
"""One-command source preparation and CLE generation for one U.S. city."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from download_hpms_city import download_hpms_city
from download_microsoft_state_buildings import download_state_buildings
from prepare_us_city import prepare_us_city


def _run(command: list[str], *, cwd: Path) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _remove_generated_pair(path: Path) -> None:
    path.unlink(missing_ok=True)
    path.with_suffix(".manifest.json").unlink(missing_ok=True)


def _download_sources(
    *,
    contract: dict[str, Any],
    generator_root: Path,
    api_key: str | None,
    force: bool,
) -> None:
    sources = contract["sources"]
    buildings = sources["microsoft_buildings"]
    download_state_buildings(
        state_file=buildings["state_file"],
        output_root=Path(buildings["path"]).parent,
        source_url=buildings["url"],
        force=force,
    )

    pbf_path = Path(sources["pbf"]["path"])
    if force:
        pbf_path.unlink(missing_ok=True)
    _run(
        [
            sys.executable,
            "scripts/fetch_pbf_sources.py",
            "--preset",
            contract["configs"]["preset"],
            "--manifest",
            str(Path(contract["custom_root"]) / "pbf_source_manifest.json"),
            "--skip-sha256",
        ],
        cwd=generator_root,
    )

    hpms = sources["hpms"]
    download_hpms_city(
        service_url=hpms["service_url"],
        boundary_path=Path(contract["boundaries"]["land"]),
        output_path=Path(hpms["path"]),
        force=force,
    )

    afdc_path = Path(sources["afdc"]["path"])
    if not afdc_path.is_file() or force:
        if not api_key:
            raise RuntimeError(
                "AFDC source is absent. Set NLR_API_KEY (legacy NREL_API_KEY is "
                "also accepted) or pass --nrel-api-key; a free developer.nlr.gov "
                "key is sufficient."
            )
        command = [
            sys.executable,
            "scripts/download_afdc_snapshot.py",
            "--output",
            str(afdc_path),
            "--api-key",
            api_key,
        ]
        if force and afdc_path.exists():
            command.append("--force")
        _run(command, cwd=generator_root)
    else:
        print(f"REUSE {afdc_path}", flush=True)

    city_slug = str(contract["city"]["slug"])
    state_abbr = str(contract["city"]["state"]["abbr"])
    afdc = sources["afdc"]
    osm_pois = Path(afdc["osm_charging_pois_path"])
    census_anchors = Path(afdc["census_address_anchors_path"])
    resolved_afdc = Path(afdc["resolved_path"])
    if force:
        _remove_generated_pair(osm_pois)
        _remove_generated_pair(census_anchors)
    if not osm_pois.is_file():
        _run(
            [
                sys.executable,
                "scripts/extract_osm_charging_pois.py",
                "--preset",
                contract["configs"]["preset"],
                "--output",
                str(osm_pois),
            ],
            cwd=generator_root,
        )
    else:
        print(f"REUSE {osm_pois}", flush=True)
    if not census_anchors.is_file():
        _run(
            [
                sys.executable,
                "scripts/geocode_afdc_addresses_census.py",
                "--afdc",
                str(afdc_path),
                "--output",
                str(census_anchors),
                "--state",
                state_abbr,
            ],
            cwd=generator_root,
        )
    else:
        print(f"REUSE {census_anchors}", flush=True)

    if force:
        _remove_generated_pair(resolved_afdc)
    if not resolved_afdc.is_file():
        _run(
            [
                sys.executable,
                "scripts/resolve_afdc_coordinates.py",
                "--afdc",
                str(afdc_path),
                "--census-results",
                str(census_anchors),
                "--osm-pois",
                str(osm_pois),
                "--output",
                str(resolved_afdc),
            ],
            cwd=generator_root,
        )
    else:
        print(f"REUSE {resolved_afdc}", flush=True)
    print(
        f"AFDC coordinate evidence ready for {city_slug}: {resolved_afdc}",
        flush=True,
    )


def _build_cle(
    *, contract: dict[str, Any], generator_root: Path, nsi_workers: int
) -> None:
    _run(
        [
            sys.executable,
            "scripts/build_cle_cohort.py",
            "--profile",
            contract["configs"]["profile"],
            "--work-root",
            contract["outputs"]["work_root"],
            "--release-root",
            contract["outputs"]["release_root"],
            "--nsi-workers",
            str(nsi_workers),
            "--replace-release-package",
        ],
        cwd=generator_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", required=True, help='Census Place name, e.g. "San Diego"')
    parser.add_argument("--state", required=True, help="State name, abbreviation, or FIPS")
    parser.add_argument("--city-slug")
    parser.add_argument("--census-place-geoid")
    parser.add_argument(
        "--geofabrik-region",
        help='Optional smaller official extract, e.g. "california/socal".',
    )
    parser.add_argument("--pbf-url", help="Explicit OSM PBF override")
    parser.add_argument("--microsoft-url", help="Explicit state-building archive override")
    parser.add_argument("--hpms-service-url", help="Explicit HPMS FeatureServer override")
    parser.add_argument("--custom-root", type=Path)
    parser.add_argument(
        "--nrel-api-key",
        dest="developer_api_key",
        default=os.environ.get("NLR_API_KEY") or os.environ.get("NREL_API_KEY"),
        help="NLR Developer Network key; legacy option/environment naming remains supported.",
    )
    parser.add_argument("--nsi-workers", type=int, default=4)
    parser.add_argument("--force-downloads", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Resolve the city and write boundaries/configs without other source downloads.",
    )
    parser.add_argument(
        "--sources-only",
        action="store_true",
        help="Prepare configs and download inputs, but do not build the CLE.",
    )
    args = parser.parse_args()
    if args.prepare_only and args.sources_only:
        parser.error("--prepare-only and --sources-only are mutually exclusive")

    generator_root = Path(__file__).resolve().parents[1]
    contract = prepare_us_city(
        city=args.city,
        state_value=args.state,
        generator_root=generator_root,
        city_slug=args.city_slug,
        census_place_geoid=args.census_place_geoid,
        geofabrik_region=args.geofabrik_region,
        pbf_url=args.pbf_url,
        microsoft_url=args.microsoft_url,
        hpms_service_url=args.hpms_service_url,
        custom_root=args.custom_root,
    )
    print(json.dumps(contract, indent=2), flush=True)
    if args.prepare_only:
        return
    _download_sources(
        contract=contract,
        generator_root=generator_root,
        api_key=args.developer_api_key,
        force=args.force_downloads,
    )
    if args.sources_only:
        return
    _build_cle(
        contract=contract,
        generator_root=generator_root,
        nsi_workers=args.nsi_workers,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "city_slug": contract["city"]["slug"],
                "release_root": contract["outputs"]["release_root"],
                "contract": contract["contract_path"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
