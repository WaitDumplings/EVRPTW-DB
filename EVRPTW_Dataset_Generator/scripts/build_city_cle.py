#!/usr/bin/env python3
"""Build, resume, and verify one City Logistics Environment (CLE)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from evrptw_cle.building_registry import extract_registered_city
from evrptw_cle.cle import (
    assemble_cle,
    verify_cle,
)
from evrptw_cle.customer_access import (
    build_footprint_access_audit,
    refresh_footprint_access_connectivity,
)
from evrptw_cle.customer_spatial import (
    audit_unmatched_nearest_footprints,
    build_microsoft_nsi_spatial_pilot,
)
from evrptw_cle.facilities import build_facility_layers
from evrptw_cle.hpms_match import (
    HPMSMatchOptions,
    build_hpms_edge_matches,
    validate_hpms_edge_matches,
)
from evrptw_cle.nsi import (
    NSICustomerOptions,
    build_nsi_customer_cle,
    verify_nsi_customer_cle,
)
from evrptw_cle.speed import build_legal_speed_layer
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
        default=Path("configs/us_11city_building_extraction_v1.json"),
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
        help=(
            "Directory containing or receiving <city-slug>.parquet normalized "
            "HPMS-to-OSM evidence."
        ),
    )
    parser.add_argument(
        "--hpms-source",
        type=Path,
        help="Raw state/city HPMS geospatial source used when normalized evidence is absent.",
    )
    parser.add_argument(
        "--require-hpms-match",
        action="store_true",
        help="Fail instead of silently falling back to OSM when HPMS evidence is unavailable.",
    )
    parser.add_argument("--hpms-candidate-radius-m", type=float, default=75.0)
    parser.add_argument("--hpms-overlap-buffer-m", type=float, default=25.0)
    parser.add_argument("--hpms-minimum-overlap-ratio", type=float, default=0.20)
    parser.add_argument(
        "--hpms-maximum-orientation-delta-deg", type=float, default=30.0
    )
    parser.add_argument(
        "--hpms-high-confidence-distance-m", type=float, default=25.0
    )
    parser.add_argument(
        "--hpms-high-confidence-overlap-ratio", type=float, default=0.50
    )
    parser.add_argument(
        "--hpms-high-confidence-orientation-delta-deg",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--hpms-ambiguity-distance-margin-m", type=float, default=10.0
    )
    parser.add_argument("--hpms-ambiguity-overlap-margin", type=float, default=0.20)
    parser.add_argument("--vehicle-speed-cap-kph", type=float)
    parser.add_argument(
        "--moves-speed-profile",
        type=Path,
        default=Path("configs/us_moves5_speed_profile_v1.json"),
        help="Compact speed-retention profile derived from a frozen MOVES5 database.",
    )
    parser.add_argument(
        "--refresh-facilities",
        action="store_true",
        help=(
            "Rebuild the city facility directory from the current AFDC and depot "
            "candidate inputs, then refresh dependent route QA and final manifests."
        ),
    )
    parser.add_argument(
        "--refresh-protected-connectivity",
        action="store_true",
        help=(
            "Relabel the frozen customer-access layer with directed SCC fields, rebuild "
            "facility SCC labels, and refresh dependent CLE manifests."
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

    if hpms_edge_evidence is None and args.hpms_source is not None:
        if args.hpms_edge_evidence_root is None:
            parser.error("--hpms-source requires --hpms-edge-evidence-root")
        _step("HPMS", "clip raw HPMS and conflate it with directed OSM edges")
        hpms_edge_evidence = args.hpms_edge_evidence_root / f"{slug}.parquet"
        build_hpms_edge_matches(
            city_slug=slug,
            hpms_path=args.hpms_source,
            graph_path=graph,
            boundary_path=service_boundary,
            output_path=hpms_edge_evidence,
            options=HPMSMatchOptions(
                candidate_radius_m=args.hpms_candidate_radius_m,
                overlap_buffer_m=args.hpms_overlap_buffer_m,
                minimum_overlap_ratio=args.hpms_minimum_overlap_ratio,
                maximum_orientation_delta_deg=(
                    args.hpms_maximum_orientation_delta_deg
                ),
                high_confidence_distance_m=(
                    args.hpms_high_confidence_distance_m
                ),
                high_confidence_overlap_ratio=(
                    args.hpms_high_confidence_overlap_ratio
                ),
                high_confidence_orientation_delta_deg=(
                    args.hpms_high_confidence_orientation_delta_deg
                ),
                ambiguity_distance_margin_m=(
                    args.hpms_ambiguity_distance_margin_m
                ),
                ambiguity_overlap_margin=args.hpms_ambiguity_overlap_margin,
            ),
        )
    if hpms_edge_evidence is not None:
        validate_hpms_edge_matches(hpms_edge_evidence)
    elif args.require_hpms_match:
        raise FileNotFoundError(
            f"Required HPMS source/match is unavailable for {slug}; provide "
            "--hpms-source or a normalized file under --hpms-edge-evidence-root"
        )

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
    if args.refresh_protected_connectivity:
        _step(
            "customer-roundtrip",
            "label every frozen edge projection against the reference directed SCC",
        )
        refresh_footprint_access_connectivity(
            graph_path=graph,
            output_dir=access_dir,
        )

    facility_dir = cle_dir / "infrastructure"
    facility_manifest = facility_dir / "facility_manifest.json"
    refresh_facilities = (
        args.refresh_facilities or args.refresh_protected_connectivity
    )
    if refresh_facilities and facility_dir.exists():
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
    speed_layer_rebuilt = False
    if speed_manifest.exists():
        recorded_speed = _read_json(speed_manifest)
        expected_moves_profile = _read_json(args.moves_speed_profile)
        recorded_hpms = recorded_speed.get("hpms_edge_evidence")
        expected_hpms_sha = (
            sha256_file(hpms_edge_evidence)
            if hpms_edge_evidence is not None
            else None
        )
        recorded_hpms_sha = (
            recorded_hpms.get("sha256")
            if isinstance(recorded_hpms, dict)
            else None
        )
        speed_is_current = (
            recorded_speed.get("schema") == "evrptw_directed_speed_profiles_v6"
            and recorded_speed.get("graph", {}).get("sha256") == sha256_file(graph)
            and recorded_hpms_sha == expected_hpms_sha
            and recorded_speed.get("reference_speed_contract", {}).get("profile_id")
            == expected_moves_profile.get("profile_id")
        )
        if not speed_is_current:
            _step(
                "speed",
                "discard stale generated profile after graph/HPMS/MOVES contract change",
            )
            shutil.rmtree(speed_dir)
    if not speed_manifest.exists():
        _step(
            "speed",
            "build HPMS/OSM legal speeds and MOVES5 weekday/weekend references",
        )
        build_legal_speed_layer(
            city_slug=slug,
            graph_path=graph,
            output_dir=speed_dir,
            hpms_edge_evidence_path=hpms_edge_evidence,
            moves_speed_profile_path=args.moves_speed_profile,
            vehicle_speed_cap_kph=args.vehicle_speed_cap_kph,
        )
        speed_layer_rebuilt = True
    else:
        _step("speed", "reuse directed legal-speed layer")

    final_manifest = cle_dir / "manifest.json"
    if (refresh_facilities or speed_layer_rebuilt) and final_manifest.exists():
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
