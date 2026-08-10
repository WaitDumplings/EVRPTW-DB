#!/usr/bin/env python3
"""Build, resume, verify, and index the frozen top-10 city CLEs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evrptw_cle.building_registry import extract_registered_city
from evrptw_cle.cle import package_cle, verify_cle
from evrptw_cle.preflight import preflight_profile
from evrptw_cle.util import sha256_file, write_json
from evrptw_cle.verification import verify_city_output

STAGES = (
    "preflight",
    "roads",
    "buildings",
    "depots",
    "cles",
    "package",
    "index",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_preset(
    preset_path: Path,
    selected: list[dict[str, Any]],
    output_root: Path,
) -> Path:
    """Materialize a path-stable preset when --cities selects a subset."""

    preset = _read_json(preset_path)
    if len(selected) == len(preset.get("cities", [])):
        return preset_path
    resolved_items: list[dict[str, Any]] = []
    for source in selected:
        item = dict(source)
        for field in ("boundary_file", "query_mask_file", "pbf_file"):
            if item.get(field):
                item[field] = str((preset_path.parent / item[field]).resolve())
        resolved_items.append(item)
    payload = {
        **preset,
        "preset_id": f"{preset['preset_id']}__selected",
        "parent_preset": str(preset_path.resolve()),
        "cities": resolved_items,
    }
    path = output_root / "qa/selected_city_preset.json"
    write_json(path, payload)
    return path


def _run(command: list[str], repo_root: Path) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=repo_root, check=True)


def _road_command(
    repo_root: Path,
    preset_path: Path,
    city_root: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "evrptw_cle.cli",
        "batch",
        "--preset",
        str(preset_path),
        "--output-root",
        str(city_root),
        "--component-policy",
        "all",
        "--network-type",
        "drive",
        "--query-buffer-m",
        "5000",
        "--query-simplify-m",
        "100",
        "--query-component-min-area-km2",
        "0",
        "--build-operational-graph",
        "--routing-buffer-ladder-km",
        "0",
        "1",
        "2",
        "5",
        "10",
        "20",
        "--min-retained-node-coverage",
        "0.99",
        "--min-retained-road-length-coverage",
        "0.995",
        "--skip-existing",
    ]


def _depot_command(
    repo_root: Path,
    preset_path: Path,
    city_root: Path,
    depot_root: Path,
    min_warehouse_area_m2: float,
) -> list[str]:
    return [
        sys.executable,
        "scripts/build_top10_depot_audit.py",
        "--preset",
        str(preset_path),
        "--repo-root",
        str(repo_root),
        "--boundary-root",
        str(repo_root / "boundaries/top10-population-2025"),
        "--city-root",
        str(city_root),
        "--output-dir",
        str(depot_root),
        "--min-warehouse-area-m2",
        str(min_warehouse_area_m2),
    ]


def _cle_command(
    *,
    slug: str,
    repo_root: Path,
    building_config: Path,
    building_source_root: Path,
    afdc_path: Path,
    output_root: Path,
    nsi_workers: int,
    refresh_facilities: bool,
    hpms_edge_evidence_root: Path | None,
    vehicle_speed_cap_kph: float | None,
    include_pilot_speed_scenarios: bool,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/build_city_cle.py",
        "--city-slug",
        slug,
        "--building-config",
        str(building_config),
        "--building-source-root",
        str(building_source_root),
        "--afdc",
        str(afdc_path),
        "--depot-root",
        str(output_root / "depot_candidates"),
        "--city-root",
        str(output_root / "cities"),
        "--building-root",
        str(output_root / "buildings"),
        "--customer-root",
        str(output_root / "customers"),
        "--customer-analysis-root",
        str(output_root / "customer_geometry"),
        "--customer-access-root",
        str(output_root / "customer_access"),
        "--cle-root",
        str(output_root / "cles"),
        "--nsi-workers",
        str(nsi_workers),
    ]
    if hpms_edge_evidence_root is not None:
        command.extend(["--hpms-edge-evidence-root", str(hpms_edge_evidence_root)])
    if vehicle_speed_cap_kph is not None:
        command.extend(["--vehicle-speed-cap-kph", str(vehicle_speed_cap_kph)])
    if include_pilot_speed_scenarios:
        command.append("--include-pilot-speed-scenarios")
    if refresh_facilities:
        command.append("--refresh-facilities")
    return command


def _city_index_row(
    item: dict[str, Any],
    city_root: Path,
    cle_root: Path,
    index_root: Path,
    *,
    require_portable: bool,
) -> dict[str, Any]:
    slug = str(item["slug"])
    road_dir = city_root / slug
    cle_dir = cle_root / slug
    row: dict[str, Any] = {
        "city_slug": slug,
        "city_label": item.get("display_name", item["query"].split(",")[0]),
        "census_place_geoid": item["census_place_geoid"],
        "road_status": "missing",
        "cle_status": "missing",
        "verification_passed": False,
    }
    if (road_dir / "manifest.json").exists():
        road_verification = verify_city_output(road_dir)
        road_manifest = _read_json(road_dir / "manifest.json")
        operational = road_manifest.get("operational_connectivity") or {}
        row.update(
            {
                "road_status": "verified" if road_verification["passed"] else "failed",
                "operational_buffer_km": operational.get("selected_buffer_km"),
                "city_node_coverage": operational.get("city_node_coverage"),
                "city_physical_road_length_coverage": operational.get(
                    "city_physical_road_length_coverage"
                ),
                "operational_nodes": operational.get("operational_node_count"),
                "operational_directed_edges": operational.get(
                    "operational_directed_edge_count"
                ),
            }
        )
    if (cle_dir / "manifest.json").exists():
        verification = verify_cle(cle_dir, require_portable=require_portable)
        manifest = _read_json(cle_dir / "manifest.json")
        counts = manifest["layer_counts"]
        row.update(
            {
                "cle_status": manifest["status"],
                "verification_passed": bool(verification["passed"]),
                "technical_verification_passed": bool(
                    verification["technical_verification_passed"]
                ),
                "portable": bool(verification["portable"]),
                "latent_service_locations": counts["latent_service_location_candidates"],
                "service_access_connectors": counts["service_access_connectors"],
                "depot_candidates": counts["depot_candidates"],
                "strict_depot_candidates": counts.get("strict_depot_candidates", 0),
                "optional_depot_candidates": counts.get("optional_depot_candidates", 0),
                "charger_candidates": counts["charger_candidates"],
                "directed_legal_speed_edges": counts["directed_legal_speed_edges"],
                "static_speed_scenarios": counts["static_operational_scenarios"],
                "release_eligible": manifest["release_eligible"],
                "release_blocker_count": manifest["release_blocker_count"],
                "manifest_path": str(
                    (cle_dir / "manifest.json").resolve().relative_to(
                        index_root.resolve()
                    )
                    if index_root.resolve()
                    in (cle_dir / "manifest.json").resolve().parents
                    else (cle_dir / "manifest.json").resolve()
                ),
                "manifest_sha256": sha256_file(cle_dir / "manifest.json"),
            }
        )
    return row


def build_index(
    preset_path: Path,
    city_root: Path,
    cle_root: Path,
    index_root: Path,
    failures: list[dict[str, str]],
    *,
    require_portable: bool,
    schema: str,
    basename: str,
) -> dict[str, Any]:
    preset = _read_json(preset_path)
    rows = [
        _city_index_row(
            item,
            city_root,
            cle_root,
            index_root,
            require_portable=require_portable,
        )
        for item in preset["cities"]
    ]
    completed = [row for row in rows if row["verification_passed"]]
    payload = {
        "schema": schema,
        "generated_utc": datetime.now(UTC).isoformat(),
        "preset": {
            "id": preset["preset_id"],
            "path": (
                preset_path.name if require_portable else str(preset_path.resolve())
            ),
            "sha256": sha256_file(preset_path),
        },
        "status": "complete" if len(completed) == len(rows) and not failures else "incomplete",
        "city_count": len(rows),
        "verified_cle_count": len(completed),
        "failures": failures,
        "verification_contract": (
            "portable_package" if require_portable else "technical_work_artifact"
        ),
        "cities": rows,
    }
    index_root.mkdir(parents=True, exist_ok=True)
    write_json(index_root / f"{basename}.json", payload)
    columns = sorted({key for row in rows for key in row})
    with (index_root / f"{basename}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", type=Path, default=Path("configs/us_top10_cle_v1.json")
    )
    parser.add_argument(
        "--preset", type=Path
    )
    parser.add_argument(
        "--building-config",
        type=Path,
        default=None,
    )
    parser.add_argument("--building-source-root", type=Path)
    parser.add_argument(
        "--afdc",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=None,
        help="Generator-owned intermediate and debug artifact root.",
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        default=None,
        help="Portable CLE package root under EVRPTW_Dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Deprecated alias for --work-root; retained for old local commands.",
    )
    parser.add_argument("--stages", choices=STAGES, nargs="+", default=list(STAGES))
    parser.add_argument("--cities", nargs="+")
    parser.add_argument("--nsi-workers", type=int, default=4)
    parser.add_argument(
        "--warehouse-area-reference-m2",
        dest="min_warehouse_area_m2",
        type=float,
        default=None,
        help="Sensitivity flag only; it is not a Tier-B hard filter.",
    )
    parser.add_argument("--hpms-edge-evidence-root", type=Path)
    parser.add_argument("--vehicle-speed-cap-kph", type=float)
    parser.add_argument("--include-pilot-speed-scenarios", action="store_true")
    parser.add_argument(
        "--refresh-facilities",
        action="store_true",
        help="Rebuild CLE facility layers and dependent QA from depot inputs.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    profile_path = args.profile if args.profile.is_absolute() else repo_root / args.profile
    profile = _read_json(profile_path)

    def profile_path_value(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (repo_root / path).resolve()

    preset_path = (
        args.preset if args.preset is not None else profile_path_value(profile["city_preset"])
    )
    if not preset_path.is_absolute():
        preset_path = repo_root / preset_path
    building_config = (
        args.building_config
        if args.building_config is not None
        else profile_path_value(profile["building_registry"])
    )
    if not building_config.is_absolute():
        building_config = repo_root / building_config
    building_source_root = (
        args.building_source_root.resolve()
        if args.building_source_root is not None
        else profile_path_value(profile["source_paths"]["microsoft_building_root"])
    )
    raw_afdc = profile_path_value(profile["source_paths"]["afdc_raw_csv"])
    resolved_afdc = profile_path_value(profile["source_paths"]["afdc_resolved_csv"])
    afdc_path = args.afdc or (resolved_afdc if resolved_afdc.exists() else raw_afdc)
    if not afdc_path.is_absolute():
        afdc_path = repo_root / afdc_path
    if args.work_root is not None and args.output_root is not None:
        parser.error("use --work-root or deprecated --output-root, not both")
    configured_work_root = profile.get("work_root", profile.get("output_root"))
    if configured_work_root is None:
        parser.error("profile must define work_root")
    work_root = args.work_root or args.output_root or profile_path_value(
        configured_work_root
    )
    if not work_root.is_absolute():
        work_root = repo_root / work_root
    configured_release_root = profile.get("release_root")
    if args.release_root is not None:
        release_root = args.release_root
    elif configured_release_root is not None:
        release_root = profile_path_value(configured_release_root)
    else:
        release_root = repo_root.parent / "EVRPTW_Dataset/CLE_v1/us_top10"
    if not release_root.is_absolute():
        release_root = repo_root / release_root
    hpms_edge_evidence_root = args.hpms_edge_evidence_root
    if hpms_edge_evidence_root is None:
        configured_hpms = profile_path_value(
            profile["source_paths"]["hpms_edge_match_root"]
        )
        hpms_edge_evidence_root = configured_hpms if configured_hpms.exists() else None
    vehicle_speed_cap_kph = args.vehicle_speed_cap_kph
    if vehicle_speed_cap_kph is None:
        vehicle_speed_cap_kph = profile["speed_profile"].get("vehicle_speed_cap_kph")
    min_warehouse_area_m2 = args.min_warehouse_area_m2
    if min_warehouse_area_m2 is None:
        min_warehouse_area_m2 = float(
            profile["depot_policy"]["reference_area_flag_m2"]
        )
    preset = _read_json(preset_path)
    requested = set(args.cities or [item["slug"] for item in preset["cities"]])
    unknown = requested - {item["slug"] for item in preset["cities"]}
    if unknown:
        parser.error(f"unknown city slugs: {sorted(unknown)}")
    selected = [item for item in preset["cities"] if item["slug"] in requested]
    execution_preset_path = _selected_preset(preset_path, selected, work_root)
    failures: list[dict[str, str]] = []

    if "preflight" in args.stages:
        report = preflight_profile(profile_path, selected_slugs=requested)
        write_json(work_root / "qa/preflight.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["passed"]:
            raise SystemExit(2)
    if "roads" in args.stages:
        _run(
            _road_command(repo_root, execution_preset_path, work_root / "cities"),
            repo_root,
        )
    if "buildings" in args.stages:
        for item in selected:
            slug = str(item["slug"])
            manifest_path = work_root / "buildings/manifests" / f"{slug}.json"
            if manifest_path.exists():
                print(f"SKIP BUILDINGS {slug} complete", flush=True)
                continue
            print(f"BUILDINGS {slug}", flush=True)
            extract_registered_city(
                config_path=building_config,
                city_slug=slug,
                source_root=building_source_root,
                output_root=work_root / "buildings",
            )
    if "depots" in args.stages:
        _run(
            _depot_command(
                repo_root,
                execution_preset_path,
                work_root / "cities",
                work_root / "depot_candidates",
                min_warehouse_area_m2,
            ),
            repo_root,
        )
    if "cles" in args.stages:
        for item in selected:
            slug = str(item["slug"])
            print(f"CLE {slug}", flush=True)
            try:
                _run(
                    _cle_command(
                        slug=slug,
                        repo_root=repo_root,
                        building_config=building_config,
                        building_source_root=building_source_root,
                        afdc_path=afdc_path,
                        output_root=work_root,
                        nsi_workers=args.nsi_workers,
                        refresh_facilities=args.refresh_facilities,
                        hpms_edge_evidence_root=hpms_edge_evidence_root,
                        vehicle_speed_cap_kph=vehicle_speed_cap_kph,
                        include_pilot_speed_scenarios=args.include_pilot_speed_scenarios,
                    ),
                    repo_root,
                )
            except subprocess.CalledProcessError as error:
                failures.append({"city_slug": slug, "error": repr(error)})
                if not args.continue_on_error:
                    build_index(
                        execution_preset_path,
                        work_root / "cities",
                        work_root / "cles",
                        work_root,
                        failures,
                        require_portable=False,
                        schema="evrptw_top10_cle_build_index_v2",
                        basename="top10_cle_index",
                    )
                    raise
    if "package" in args.stages:
        for item in selected:
            slug = str(item["slug"])
            print(f"PACKAGE {slug}", flush=True)
            try:
                package_result = package_cle(
                    source_cle_dir=work_root / "cles" / slug,
                    graph_path=work_root
                    / "cities"
                    / slug
                    / "graph_operational.graphml",
                    road_manifest_path=work_root / "cities" / slug / "manifest.json",
                    destination_cle_dir=release_root / "cities" / slug,
                )
                print(json.dumps(package_result, ensure_ascii=False, indent=2))
            except (FileNotFoundError, FileExistsError, ValueError) as error:
                failures.append({"city_slug": slug, "error": repr(error)})
                if not args.continue_on_error:
                    break

    work_payload = build_index(
        execution_preset_path,
        work_root / "cities",
        work_root / "cles",
        work_root,
        failures,
        require_portable=False,
        schema="evrptw_top10_cle_build_index_v2",
        basename="top10_cle_index",
    )
    release_payload = build_index(
        execution_preset_path,
        work_root / "cities",
        release_root / "cities",
        release_root,
        failures,
        require_portable=True,
        schema="evrptw_portable_cle_cohort_index_v1",
        basename="cle_index",
    )
    print(
        json.dumps(
            {"work": work_payload, "release": release_payload},
            ensure_ascii=False,
            indent=2,
        )
    )
    # Intermediate stages intentionally produce an incomplete final index. Treat
    # that as a nonzero exit only when this invocation was asked to assemble or
    # validate CLEs; otherwise roads/buildings/depots remain composable.
    requires_portable_index = "package" in args.stages or args.stages == ["index"]
    required_payload = release_payload if requires_portable_index else work_payload
    if ("cles" in args.stages or requires_portable_index) and required_payload[
        "status"
    ] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
