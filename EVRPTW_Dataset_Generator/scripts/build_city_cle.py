#!/usr/bin/env python3
"""Build, resume, and verify one City Logistics Environment (CLE)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from evrptw_cle.access_audit import build_service_access_local_route_audit
from evrptw_cle.building_registry import extract_registered_city
from evrptw_cle.cle import (
    assemble_cle,
    verify_cle,
)
from evrptw_cle.customer_access import build_footprint_access_audit
from evrptw_cle.customer_spatial import (
    audit_unmatched_nearest_footprints,
    build_microsoft_nsi_spatial_pilot,
)
from evrptw_cle.facilities import build_facility_layers
from evrptw_cle.nsi import (
    NSICustomerOptions,
    build_nsi_customer_cle,
    verify_nsi_customer_cle,
)
from evrptw_cle.speed import build_legal_speed_layer
from evrptw_cle.speed_audit import build_static_speed_route_audit
from evrptw_cle.util import sha256_file
from evrptw_cle.verification import verify_city_output


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolved(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _step(name: str, action: str) -> None:
    print(f"[{name}] {action}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-slug", required=True)
    parser.add_argument(
        "--building-config",
        type=Path,
        default=Path("configs/top10_building_extraction_v1.json"),
    )
    parser.add_argument("--building-source-root", type=Path, required=True)
    parser.add_argument("--afdc", type=Path, required=True)
    parser.add_argument(
        "--depot-root", type=Path, default=Path("analysis/depot_preview/2026-08-04")
    )
    parser.add_argument("--city-root", type=Path, default=Path("data/cities"))
    parser.add_argument("--building-root", type=Path, default=Path("data/buildings"))
    parser.add_argument("--customer-root", type=Path, default=Path("data/customers"))
    parser.add_argument(
        "--customer-analysis-root", type=Path, default=Path("analysis/customer_gate")
    )
    parser.add_argument(
        "--customer-access-root",
        type=Path,
        help=(
            "Optional graph-versioned road-access root. Geometry matching remains under "
            "--customer-analysis-root and can be reused across road-graph rebuilds."
        ),
    )
    parser.add_argument("--cle-root", type=Path, default=Path("data/cles"))
    parser.add_argument("--nsi-workers", type=int, default=4)
    parser.add_argument(
        "--access-distance-qa-reference-m",
        dest="candidate_access_threshold_m",
        type=float,
        default=200.0,
        help="QA reference only; records are not deleted solely for exceeding it.",
    )
    parser.add_argument(
        "--hpms-edge-evidence-root",
        type=Path,
        help="Optional directory containing <city-slug>.parquet or .csv conflated edge evidence.",
    )
    parser.add_argument("--vehicle-speed-cap-kph", type=float)
    parser.add_argument(
        "--include-pilot-speed-scenarios",
        action="store_true",
        help="Write engineering weekday/weekend scenarios for QA; official scenarios belong to Stage 2.",
    )
    parser.add_argument("--speed-scenario-seed", type=int, default=20270805)
    parser.add_argument("--speed-scenarios-per-day-type", type=int, default=2)
    parser.add_argument("--speed-global-sigma", type=float, default=0.02)
    parser.add_argument("--speed-road-group-sigma", type=float, default=0.04)
    parser.add_argument("--speed-corridor-sigma", type=float, default=0.05)
    parser.add_argument("--speed-direction-sigma", type=float, default=0.03)
    parser.add_argument("--speed-factor-min", type=float, default=0.75)
    parser.add_argument("--speed-factor-max", type=float, default=1.15)
    parser.add_argument(
        "--refresh-facilities",
        action="store_true",
        help=(
            "Rebuild the city facility directory from the current AFDC and depot "
            "candidate inputs, then refresh dependent route QA and final manifests."
        ),
    )
    parser.add_argument(
        "--refresh-speed-route-qa",
        action="store_true",
        help=(
            "Recompute the static speed route QA and rebuild the final manifest hashes. "
            "All source and intermediate layers are still reused after verification."
        ),
    )
    args = parser.parse_args()

    config_path = args.building_config.resolve()
    registry = _read_json(config_path)
    if args.city_slug not in registry.get("cities", {}):
        parser.error(f"unknown registered city: {args.city_slug}")
    entry = registry["cities"][args.city_slug]
    slug = args.city_slug
    label = str(entry["label"])
    area_crs = str(entry["area_crs"])
    service_boundary = _resolved(config_path, str(entry["boundary_file"]))
    admin_boundary = service_boundary.with_name("admin_boundary.geojson")
    city_dir = args.city_root / slug
    road_manifest = city_dir / "manifest.json"
    graph = city_dir / "graph_operational.graphml"
    building_dir = args.building_root / slug
    building_manifest = args.building_root / "manifests" / f"{slug}.json"
    customer_dir = args.customer_root / slug
    nsi_manifest = customer_dir / "customer_cle_manifest.json"
    analysis_dir = args.customer_analysis_root / f"{slug}_spatial_match"
    access_dir = (
        args.customer_access_root / slug
        if args.customer_access_root is not None
        else analysis_dir
    )
    cle_dir = args.cle_root / slug
    hpms_edge_evidence = None
    if args.hpms_edge_evidence_root is not None:
        for suffix in (".parquet", ".csv"):
            candidate = args.hpms_edge_evidence_root / f"{slug}{suffix}"
            if candidate.exists():
                hpms_edge_evidence = candidate
                break

    _step("road", "verify existing operational road graph")
    road_verification = verify_city_output(city_dir)
    if not road_verification["passed"]:
        raise RuntimeError(f"road verification failed: {road_verification['errors']}")

    if not building_manifest.exists():
        _step("buildings", "extract registered Microsoft footprints")
        extract_registered_city(
            config_path=config_path,
            city_slug=slug,
            source_root=args.building_source_root,
            output_root=args.building_root,
        )
    else:
        _step("buildings", "reuse registered extraction manifest")
    if not building_dir.exists():
        raise FileNotFoundError(f"building output is missing: {building_dir}")

    if not nsi_manifest.exists():
        _step("NSI", "download, classify, group, and attach preliminary road access")
        build_nsi_customer_cle(
            NSICustomerOptions(
                city_slug=slug,
                city_label=label,
                boundary_file=service_boundary,
                graph_file=graph,
                output_dir=customer_dir,
                area_crs=area_crs,
                workers=args.nsi_workers,
            )
        )
    else:
        _step("NSI", "reuse frozen raw tiles and customer pilot")
    nsi_verification = verify_nsi_customer_cle(customer_dir)
    if not nsi_verification["passed"]:
        raise RuntimeError(f"NSI verification failed: {nsi_verification['errors']}")

    spatial_manifest = analysis_dir / "spatial_match_manifest.json"
    if not spatial_manifest.exists():
        _step("geometry", "match NSI evidence to Microsoft footprints")
        build_microsoft_nsi_spatial_pilot(
            city_slug=slug,
            building_dir=building_dir,
            nsi_records_path=customer_dir / "nsi_ordinary_residential_records.parquet",
            nsi_locations_path=customer_dir / "latent_service_locations.parquet",
            boundary_path=service_boundary,
            output_dir=analysis_dir,
        )
    else:
        _step("geometry", "reuse Microsoft/NSI containment result")

    nearest_audit = analysis_dir / "unmatched_nearest_footprint_audit.json"
    if not nearest_audit.exists():
        _step("geometry-G2", "apply the frozen 10 m / area-factor-4 candidate rule")
        audit_unmatched_nearest_footprints(
            building_dir=building_dir,
            crosswalk_path=analysis_dir / "microsoft_nsi_record_crosswalk.parquet",
            nsi_locations_path=customer_dir / "latent_service_locations.parquet",
            output_dir=analysis_dir,
            area_crs=area_crs,
        )
    else:
        _step("geometry-G2", "reuse nearest-footprint candidate audit")

    access_audit = access_dir / "footprint_road_access_audit.json"
    access_candidates = access_dir / "footprint_road_access_candidates.parquet"
    if not access_audit.exists():
        _step("customer-road", "anchor footprint boundaries to directed operational edges")
        build_footprint_access_audit(
            city_slug=slug,
            location_path=analysis_dir / "geometry_resolved_footprint_candidates.parquet",
            graph_path=graph,
            output_dir=access_dir,
            area_crs=area_crs,
        )
    else:
        _step("customer-road", "reuse directed footprint access audit")
        recorded_graph_sha = _read_json(access_audit).get("inputs", {}).get(
            "graph_sha256"
        )
        if recorded_graph_sha != sha256_file(graph):
            raise RuntimeError(
                "Existing customer-access audit belongs to a different road graph; "
                "use a new --customer-access-root"
            )

    facility_dir = cle_dir / "infrastructure"
    facility_manifest = facility_dir / "facility_manifest.json"
    if args.refresh_facilities and facility_dir.exists():
        _step("facilities", "remove the targeted stale facility layer before rebuild")
        shutil.rmtree(facility_dir)
    if not facility_manifest.exists():
        _step("facilities", "filter and road-anchor AFDC and OSM candidates")
        build_facility_layers(
            city_slug=slug,
            afdc_path=args.afdc,
            depot_candidates_path=args.depot_root / f"{slug}_depot_candidates.csv",
            depot_summary_path=args.depot_root / f"{slug}_depot_summary.json",
            boundary_path=service_boundary,
            graph_path=graph,
            output_dir=facility_dir,
            area_crs=area_crs,
        )
    else:
        _step("facilities", "reuse road-anchored facility candidates")

    speed_dir = cle_dir / "profiles"
    speed_manifest = speed_dir / "speed_manifest.json"
    if not speed_manifest.exists():
        _step(
            "speed",
            "build the HPMS/OSM legal layer and NREL H/M/U reference-speed profile",
        )
        build_legal_speed_layer(
            city_slug=slug,
            graph_path=graph,
            output_dir=speed_dir,
            hpms_edge_evidence_path=hpms_edge_evidence,
            vehicle_speed_cap_kph=args.vehicle_speed_cap_kph,
            build_pilot_scenarios=args.include_pilot_speed_scenarios,
            scenario_seed=args.speed_scenario_seed,
            scenarios_per_day_type=args.speed_scenarios_per_day_type,
            global_sigma=args.speed_global_sigma,
            road_group_sigma=args.speed_road_group_sigma,
            corridor_sigma=args.speed_corridor_sigma,
            direction_sigma=args.speed_direction_sigma,
            factor_min=args.speed_factor_min,
            factor_max=args.speed_factor_max,
        )
    else:
        _step("speed", "reuse directed legal-speed layer")

    refresh_speed_route_qa = args.refresh_speed_route_qa or args.refresh_facilities
    speed_route_audit = cle_dir / "qa/static_speed_route_audit.json"
    if args.include_pilot_speed_scenarios and (
        not speed_route_audit.exists() or refresh_speed_route_qa
    ):
        _step("speed-route-QA", "audit directed OD asymmetry and scenario-dependent fastest paths")
        build_static_speed_route_audit(
            graph_path=graph,
            locations_path=access_candidates,
            depots_path=facility_dir / "depots.parquet",
            scenarios_path=speed_dir / "static_operational_scenarios.parquet",
            output_dir=cle_dir / "qa",
            seed=args.speed_scenario_seed,
        )
    elif args.include_pilot_speed_scenarios:
        _step("speed-route-QA", "reuse static speed route audit")
    else:
        _step("speed-route-QA", "skip: operational scenarios are a Stage-2 concern")

    local_access_audit = (
        cle_dir / "qa/local_access/service_access_local_route_audit.json"
    )
    if args.include_pilot_speed_scenarios and not local_access_audit.exists():
        _step(
            "customer-local-route-QA",
            "materialize active virtual stops and test exact same-edge directed routes",
        )
        build_service_access_local_route_audit(
            graph_path=graph,
            access_candidates_path=access_candidates,
            service_nodes_path=access_dir / "service_access_nodes.parquet",
            projection_nodes_path=access_dir / "road_projection_nodes.parquet",
            connectors_path=access_dir / "service_access_connectors.parquet",
            scenarios_path=speed_dir / "static_operational_scenarios.parquet",
            output_dir=cle_dir / "qa/local_access",
            seed=args.speed_scenario_seed,
        )
    elif args.include_pilot_speed_scenarios:
        _step("customer-local-route-QA", "reuse exact virtual-access route audit")
    else:
        _step("customer-local-route-QA", "skip weighted route QA until Stage-2 speed realization")

    final_manifest = cle_dir / "manifest.json"
    if refresh_speed_route_qa and final_manifest.exists():
        final_manifest.unlink()
    if not final_manifest.exists():
        _step("assembler", "assemble source registry, layer table, QA gates, and checksums")
        assemble_cle(
            city_slug=slug,
            city_label=label,
            cle_dir=cle_dir,
            admin_boundary_path=admin_boundary,
            service_boundary_path=service_boundary,
            road_manifest_path=road_manifest,
            graph_path=graph,
            building_manifest_path=building_manifest,
            nsi_manifest_path=nsi_manifest,
            spatial_manifest_path=spatial_manifest,
            nearest_audit_path=nearest_audit,
            access_audit_path=access_audit,
            access_candidates_path=access_candidates,
            service_access_nodes_path=access_dir / "service_access_nodes.parquet",
            road_projection_nodes_path=access_dir / "road_projection_nodes.parquet",
            service_access_connectors_path=access_dir
            / "service_access_connectors.parquet",
            facility_manifest_path=facility_manifest,
            speed_manifest_path=speed_manifest,
            candidate_access_threshold_m=args.candidate_access_threshold_m,
        )
    else:
        _step("assembler", "reuse assembled CLE")

    _step("verify", "recompute all published output hashes and semantic invariants")
    result = verify_cle(cle_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
