"""Assemble and verify one City Logistics Environment (CLE)."""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from .util import sha256_file, write_json


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _is_within(path: Path, root: Path) -> bool:
    """Return whether *path* resolves inside *root* (including the root itself)."""

    resolved_path = path.resolve()
    resolved_root = root.resolve()
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _packaged_source_registry(
    source_registry_path: Path,
    packaged_paths: dict[str, Path],
    cle_dir: Path,
    city_slug: str,
) -> dict[str, Any]:
    """Remove machine-local build paths while retaining hashes and packaged locations."""

    sources: dict[str, Any] = {}
    if source_registry_path.exists():
        original = _read_json(source_registry_path)
        for name, record in original.get("sources", {}).items():
            sources[name] = {
                "sha256": record.get("sha256"),
                "availability": "build_provenance_only",
            }
    for name, path in packaged_paths.items():
        sources[name] = {
            "sha256": sha256_file(path),
            "availability": "packaged",
            "path": str(path.relative_to(cle_dir)),
        }
    return {
        "schema": "evrptw_cle_source_registry_v2",
        "city_slug": city_slug,
        "path_policy": (
            "Release packages omit machine-local build paths; packaged paths are relative "
            "to the CLE root."
        ),
        "sources": sources,
    }


def _prepare_customer_layer(
    access_candidates_path: Path,
    output_path: Path,
    candidate_access_threshold_m: float,
) -> dict[str, Any]:
    frame = gpd.read_parquet(access_candidates_path)
    geometry_candidate = frame["geometry_evidence_tier"].isin(
        {"G1_containment", "G2_near_area_consistent"}
    )
    g2_pending = frame["geometry_evidence_tier"].eq("G2_near_area_consistent")
    road_anchor_present = frame["physical_edge_id"].notna() & pd.to_numeric(
        frame["road_access_distance_m"], errors="coerce"
    ).notna()
    road_distance_qa_flag = (
        pd.to_numeric(frame["road_access_distance_m"], errors="coerce")
        > candidate_access_threshold_m
    )
    frame["geometry_rule_candidate_eligible"] = geometry_candidate
    frame["geometry_core_eligible"] = geometry_candidate & ~g2_pending
    frame["road_access_candidate_eligible"] = road_anchor_present
    frame["road_access_distance_qa_flag"] = road_distance_qa_flag
    frame["road_access_default_eligible"] = False
    frame["cle_candidate_eligible"] = geometry_candidate & road_anchor_present
    frame["customer_release_eligible"] = False
    frame["access_qa_reference_m"] = candidate_access_threshold_m

    reasons = []
    for is_g2, anchor_ok, distance_flag in zip(
        g2_pending, road_anchor_present, road_distance_qa_flag, strict=True
    ):
        row_reasons = []
        if is_g2:
            row_reasons.append("g2_manual_geometry_audit_pending")
        if not anchor_ok:
            row_reasons.append("no_eligible_physical_road_anchor")
        if distance_flag:
            row_reasons.append("road_access_distance_tail_requires_review")
        reasons.append(";".join(row_reasons))
    frame["quarantine_reason"] = reasons
    frame["active_customer"] = False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    units = pd.to_numeric(frame["residential_units"], errors="coerce").fillna(0)
    candidate = frame["cle_candidate_eligible"]
    return {
        "location_count": len(frame),
        "candidate_eligible_location_count": int(candidate.sum()),
        "candidate_eligible_modeled_unit_share_within_geometry_resolved_table": (
            float(units.loc[candidate].sum() / units.sum()) if float(units.sum()) else 0.0
        ),
        "g1_location_count": int((frame["geometry_evidence_tier"] == "G1_containment").sum()),
        "g2_manual_audit_pending_location_count": int(g2_pending.sum()),
        "road_access_distance_qa_flag_count": int(road_distance_qa_flag.sum()),
        "release_eligible_location_count": 0,
        "service_location_type_counts": {
            str(key): int(value)
            for key, value in frame["service_location_type"].value_counts().sort_index().items()
        },
    }


def assemble_cle(
    *,
    city_slug: str,
    city_label: str,
    cle_dir: Path,
    admin_boundary_path: Path,
    service_boundary_path: Path,
    road_manifest_path: Path,
    graph_path: Path,
    building_manifest_path: Path,
    nsi_manifest_path: Path,
    spatial_manifest_path: Path,
    nearest_audit_path: Path,
    access_audit_path: Path,
    access_candidates_path: Path,
    service_access_nodes_path: Path,
    road_projection_nodes_path: Path,
    service_access_connectors_path: Path,
    facility_manifest_path: Path,
    speed_manifest_path: Path,
    candidate_access_threshold_m: float = 200.0,
) -> dict[str, Any]:
    """Assemble every available layer and make unresolved release gates explicit."""

    final_manifest_path = cle_dir / "manifest.json"
    if final_manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite CLE manifest: {final_manifest_path}")
    road_manifest = _read_json(road_manifest_path)
    building_manifest = _read_json(building_manifest_path)
    nsi_manifest = _read_json(nsi_manifest_path)
    spatial_manifest = _read_json(spatial_manifest_path)
    nearest_audit = _read_json(nearest_audit_path)
    access_audit = _read_json(access_audit_path)
    facility_manifest = _read_json(facility_manifest_path)
    speed_manifest = _read_json(speed_manifest_path)
    manifests = (
        road_manifest,
        building_manifest,
        nsi_manifest,
        spatial_manifest,
        facility_manifest,
        speed_manifest,
    )
    for payload in manifests:
        payload_slug = payload.get("city_slug")
        if payload_slug is not None and payload_slug != city_slug:
            raise ValueError(f"Layer city_slug {payload_slug!r} differs from {city_slug!r}")
    graph_sha = sha256_file(graph_path)
    nsi_graph_superseded = nsi_manifest.get("road_graph", {}).get("sha256") != graph_sha
    if access_audit.get("inputs", {}).get("graph_sha256") != graph_sha:
        raise ValueError("Customer access layer graph hash differs from requested operational graph")
    if facility_manifest.get("inputs", {}).get("graph", {}).get("sha256") != graph_sha:
        raise ValueError("Facility layer graph hash differs from requested operational graph")
    if speed_manifest.get("graph", {}).get("sha256") != graph_sha:
        raise ValueError("Speed layer graph hash differs from requested operational graph")

    boundary_dir = cle_dir / "boundary"
    graph_dir = cle_dir / "graph"
    service_dir = cle_dir / "service_locations"
    qa_dir = cle_dir / "qa"
    for directory in (boundary_dir, graph_dir, service_dir, qa_dir):
        directory.mkdir(parents=True, exist_ok=True)
    admin_output = boundary_dir / "admin_boundary.geojson"
    service_output = boundary_dir / "service_boundary.geojson"
    shutil.copy2(admin_boundary_path, admin_output)
    shutil.copy2(service_boundary_path, service_output)
    graph_reference_path = graph_dir / "graph_reference.json"
    write_json(
        graph_reference_path,
        {
            "schema": "evrptw_graph_reference_v1",
            "city_slug": city_slug,
            "operational_graph": _source(graph_path),
            "road_manifest": _source(road_manifest_path),
            "packaging_status": (
                "CLE references the verified road artifact; release packaging may copy it"
            ),
        },
    )

    customer_output = service_dir / "latent_locations.parquet"
    customer_summary = _prepare_customer_layer(
        access_candidates_path, customer_output, candidate_access_threshold_m
    )
    access_outputs = {
        "service_access_nodes": service_dir / "service_access_nodes.parquet",
        "road_projection_nodes": service_dir / "road_projection_nodes.parquet",
        "service_access_connectors": service_dir / "service_access_connectors.parquet",
    }
    for source, destination in (
        (service_access_nodes_path, access_outputs["service_access_nodes"]),
        (road_projection_nodes_path, access_outputs["road_projection_nodes"]),
        (service_access_connectors_path, access_outputs["service_access_connectors"]),
    ):
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
    source_paths = {
        "admin_boundary": admin_boundary_path,
        "service_boundary": service_boundary_path,
        "road_manifest": road_manifest_path,
        "operational_graph": graph_path,
        "building_manifest": building_manifest_path,
        "nsi_manifest": nsi_manifest_path,
        "spatial_match_manifest": spatial_manifest_path,
        "nearest_footprint_audit": nearest_audit_path,
        "road_access_audit": access_audit_path,
        "road_access_candidates": access_candidates_path,
        "service_access_nodes": service_access_nodes_path,
        "road_projection_nodes": road_projection_nodes_path,
        "service_access_connectors": service_access_connectors_path,
        "facility_manifest": facility_manifest_path,
        "speed_manifest": speed_manifest_path,
    }
    source_registry = {
        "schema": "evrptw_cle_source_registry_v1",
        "city_slug": city_slug,
        "sources": {name: _source(path) for name, path in source_paths.items()},
    }
    source_registry_path = cle_dir / "source_registry.json"
    write_json(source_registry_path, source_registry)

    geometry_candidate = nearest_audit["summary"]["frozen_development_rule_candidate"]
    access_200 = access_audit["summary"]["threshold_sensitivity"][
        str(int(candidate_access_threshold_m))
    ]
    gates = [
        {
            "gate": "road_boundary_and_connectivity",
            "status": "passed",
            "evidence": "frozen road manifest and operational graph are present and hash-aligned",
        },
        {
            "gate": "microsoft_building_extraction",
            "status": "passed",
            "evidence": {
                "building_count": building_manifest["summary"]["building_count"],
                "source_sha256": building_manifest["source"]["sha256"],
            },
        },
        {
            "gate": "nsi_source_and_classification",
            "status": "passed",
            "evidence": {
                "ordinary_residential_record_count": nsi_manifest["record_audit"][
                    "ordinary_residential_record_count"
                ],
                "raw_tile_count": nsi_manifest["source"]["tile_count"],
                "legacy_preliminary_road_anchor_superseded": nsi_graph_superseded,
                "current_access_graph_sha256": graph_sha,
            },
        },
        {
            "gate": "microsoft_nsi_geometry",
            "status": "blocked",
            "candidate_metric": {
                "unit_weighted_share": geometry_candidate[
                    "candidate_core_modeled_unit_share"
                ],
                "rule": "G1 containment plus G2 <=10m and area factor <=4",
            },
            "blocker": "G2 manual geometry audit has not passed",
        },
        {
            "gate": "customer_road_access_review",
            "status": "blocked",
            "qa_metric": {
                "distance_reference_m": candidate_access_threshold_m,
                "unit_weighted_share": access_200["covered_modeled_unit_share"],
            },
            "blocker": (
                "distance is not a deletion rule; stratified access plausibility review and "
                "the unresolved-access ledger must be completed"
            ),
        },
        {
            "gate": "customer_virtual_access_materialization",
            "status": "passed",
            "evidence": {
                "service_access_node_count": access_audit["access_contract_row_counts"][
                    "service_access_nodes"
                ],
                "unique_road_projection_node_count": access_audit[
                    "access_contract_row_counts"
                ]["road_projection_nodes"],
                "connector_symmetry": "equal_both_directions",
                "edge_split_contract": "directed u/v/key and fractional offsets retained",
            },
        },
        {
            "gate": "depot_release",
            "status": "blocked",
            "candidate_count": facility_manifest["depots"]["candidate_eligible_count"],
            "blocker": "no candidate has current last-mile-function verification",
        },
        {
            "gate": "charging_release",
            "status": "blocked",
            "candidate_count": facility_manifest["charging"]["candidate_eligible_count"],
            "blocker": "station power and reference-vehicle compatibility are not fully verified",
        },
        {
            "gate": "legal_speed",
            "status": "passed_with_imputation",
            "observed_edge_share": speed_manifest["observed_osm_maxspeed_edge_share"],
            "imputed_edge_count": speed_manifest["imputed_edge_count"],
        },
        {
            "gate": "reference_speed_profile",
            "status": "passed",
            "evidence": speed_manifest.get("reference_speed_contract", {}),
            "stage_2_boundary": (
                "weekday/weekend instance-static operational realizations are generated "
                "when instances are created, not required in the CLE"
            ),
        },
        {
            "gate": "delivery_communities",
            "status": "blocked",
            "blocker": "community/territory layer has not been generated",
        },
        {
            "gate": "amazon_operational_profiles",
            "status": "not_applicable_stage_2",
            "evidence": "package, service-time, and time-window calibration is not CLE state",
        },
        {
            "gate": "reference_vehicle_profile",
            "status": "optional_profile_not_applied",
            "evidence": (
                "no vehicle speed cap is applied unless a versioned profile supplies one"
            ),
        },
    ]
    blockers = [item for item in gates if item["status"] == "blocked"]
    report = {
        "schema": "evrptw_cle_build_report_v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "city_slug": city_slug,
        "city_label": city_label,
        "technical_pipeline_reached_assembler": True,
        "release_eligible": False,
        "passed_or_materialized_gate_count": len(gates) - len(blockers),
        "blocked_gate_count": len(blockers),
        "gates": gates,
        "customer_layer": customer_summary,
    }
    report_path = qa_dir / "cle_report.json"
    write_json(report_path, report)

    output_paths = {
        "admin_boundary": admin_output,
        "service_boundary": service_output,
        "graph_reference": graph_reference_path,
        "latent_locations": customer_output,
        **access_outputs,
        "chargers": cle_dir / "infrastructure/chargers.parquet",
        "depots": cle_dir / "infrastructure/depots.parquet",
        "facility_manifest": facility_manifest_path,
        "directed_legal_speeds": cle_dir / "profiles/directed_legal_speeds.parquet",
        "speed_manifest": speed_manifest_path,
        "source_registry": source_registry_path,
        "qa_report": report_path,
    }
    operational_scenarios = cle_dir / "profiles/static_operational_scenarios.parquet"
    if operational_scenarios.exists():
        output_paths["static_operational_scenarios"] = operational_scenarios
    for output_key, name in (
        ("static_speed_route_audit_report", "static_speed_route_audit.json"),
        ("static_speed_route_samples", "static_speed_route_samples.csv"),
        ("static_speed_route_audit_plot", "static_speed_route_audit.png"),
    ):
        path = qa_dir / name
        if path.exists():
            output_paths[output_key] = path
    local_access_report = qa_dir / "local_access/service_access_local_route_audit.json"
    local_access_samples = qa_dir / "local_access/service_access_local_routes.csv"
    if local_access_report.exists():
        output_paths["service_access_local_route_audit"] = local_access_report
    if local_access_samples.exists():
        output_paths["service_access_local_route_samples"] = local_access_samples
    missing = [name for name, path in output_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"City Logistics Environment outputs are missing: {missing}")
    manifest = {
        "schema": "evrptw_city_logistics_environment_v1",
        "status": "cle_build_complete_release_gates_open",
        "generated_utc": datetime.now(UTC).isoformat(),
        "city_slug": city_slug,
        "city_label": city_label,
        "release_eligible": False,
        "release_blocker_count": len(blockers),
        "release_blockers": [item["gate"] for item in blockers],
        "layer_counts": {
            "road_nodes": road_manifest["operational_connectivity"][
                "operational_node_count"
            ],
            "road_edges": road_manifest["operational_connectivity"][
                "operational_directed_edge_count"
            ],
            "microsoft_buildings": building_manifest["summary"]["building_count"],
            "nsi_ordinary_residential_records": nsi_manifest["record_audit"][
                "ordinary_residential_record_count"
            ],
            "latent_service_location_candidates": customer_summary["location_count"],
            "unique_road_projection_nodes": access_audit["access_contract_row_counts"][
                "road_projection_nodes"
            ],
            "service_access_connectors": access_audit["access_contract_row_counts"][
                "service_access_connectors"
            ],
            "charger_candidates": facility_manifest["charging"]["candidate_eligible_count"],
            "depot_candidates": facility_manifest["depots"]["candidate_eligible_count"],
            "strict_depot_candidates": facility_manifest["depots"].get(
                "strict_candidate_eligible_count", 0
            ),
            "optional_depot_candidates": facility_manifest["depots"].get(
                "optional_candidate_eligible_count", 0
            ),
            "directed_legal_speed_edges": speed_manifest["edge_count"],
            "static_operational_scenarios": speed_manifest.get("scenario_count", 0),
            "static_operational_scenario_edge_rows": speed_manifest.get(
                "scenario_edge_row_count", 0
            ),
        },
        "outputs": {
            name: str(path.relative_to(cle_dir)) for name, path in output_paths.items()
        },
        "output_sha256": {name: sha256_file(path) for name, path in output_paths.items()},
    }
    write_json(final_manifest_path, manifest)
    return manifest


def package_cle(
    *,
    source_cle_dir: Path,
    graph_path: Path,
    road_manifest_path: Path,
    destination_cle_dir: Path,
) -> dict[str, Any]:
    """Create an atomic, self-contained CLE package from a verified work artifact.

    The source CLE remains a debug/build artifact. The destination is created only
    after its copied tables, packaged road graph, relative references, hashes, and
    semantic invariants pass the strict portability verifier.
    """

    source_cle_dir = source_cle_dir.resolve()
    graph_path = graph_path.resolve()
    road_manifest_path = road_manifest_path.resolve()
    destination_cle_dir = destination_cle_dir.resolve()
    source_verification = verify_cle(source_cle_dir)
    if not source_verification["passed"]:
        raise ValueError(
            "Source CLE failed technical verification: "
            f"{source_verification['errors']}"
        )
    for required in (graph_path, road_manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required road artifact is missing: {required}")
    source_candidate_manifest_sha256 = sha256_file(source_cle_dir / "manifest.json")
    expected_input_hashes = {
        "operational_graph": sha256_file(graph_path),
        "road_manifest": sha256_file(road_manifest_path),
    }
    if destination_cle_dir.exists():
        existing = verify_cle(destination_cle_dir, require_portable=True)
        existing_manifest_path = destination_cle_dir / "manifest.json"
        existing_manifest = (
            _read_json(existing_manifest_path) if existing_manifest_path.exists() else {}
        )
        same_candidate = (
            existing_manifest.get("package", {}).get(
                "source_candidate_manifest_sha256"
            )
            == source_candidate_manifest_sha256
        )
        same_roads = all(
            existing_manifest.get("output_sha256", {}).get(name) == digest
            for name, digest in expected_input_hashes.items()
        )
        if existing["passed"] and same_candidate and same_roads:
            return {
                **existing,
                "packaging_action": "reused_existing_verified_package",
            }
        raise FileExistsError(
            "Refusing to overwrite an existing invalid or stale CLE package: "
            f"{destination_cle_dir}"
        )

    destination_cle_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = destination_cle_dir.parent / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f"{destination_cle_dir.name}-", dir=staging_parent)
    )
    try:
        shutil.copytree(source_cle_dir, staging_dir, dirs_exist_ok=True)
        graph_dir = staging_dir / "graph"
        graph_dir.mkdir(parents=True, exist_ok=True)
        packaged_graph = graph_dir / "graph_operational.graphml"
        packaged_road_manifest = graph_dir / "road_manifest.json"
        shutil.copy2(graph_path, packaged_graph)
        shutil.copy2(road_manifest_path, packaged_road_manifest)

        graph_reference_path = graph_dir / "graph_reference.json"
        write_json(
            graph_reference_path,
            {
                "schema": "evrptw_graph_reference_v2",
                "city_slug": destination_cle_dir.name,
                "path_base": "graph_reference_directory",
                "operational_graph": {
                    "path": packaged_graph.name,
                    "sha256": sha256_file(packaged_graph),
                },
                "road_manifest": {
                    "path": packaged_road_manifest.name,
                    "sha256": sha256_file(packaged_road_manifest),
                },
                "packaging_status": "self_contained",
            },
        )

        source_registry_path = staging_dir / "source_registry.json"
        packaged_registry = _packaged_source_registry(
            source_registry_path,
            {
                "operational_graph": packaged_graph,
                "road_manifest": packaged_road_manifest,
                "admin_boundary": staging_dir / "boundary/admin_boundary.geojson",
                "service_boundary": staging_dir / "boundary/service_boundary.geojson",
                "facility_manifest": staging_dir
                / "infrastructure/facility_manifest.json",
                "speed_manifest": staging_dir / "profiles/speed_manifest.json",
            },
            staging_dir,
            destination_cle_dir.name,
        )
        write_json(source_registry_path, packaged_registry)

        manifest_path = staging_dir / "manifest.json"
        manifest = _read_json(manifest_path)
        manifest["status"] = "portable_package_pending_verification"
        manifest["technical_verification_passed"] = True
        manifest["portable_package_verified"] = False
        manifest["package"] = {
            "schema": "evrptw_cle_portable_package_v1",
            "generated_utc": datetime.now(UTC).isoformat(),
            "portable": True,
            "runtime_path_policy": "relative_to_cle_root",
            "source_candidate_manifest_sha256": source_candidate_manifest_sha256,
        }
        manifest["outputs"].update(
            {
                "graph_reference": "graph/graph_reference.json",
                "operational_graph": "graph/graph_operational.graphml",
                "road_manifest": "graph/road_manifest.json",
                "source_registry": "source_registry.json",
            }
        )
        for name in (
            "graph_reference",
            "operational_graph",
            "road_manifest",
            "source_registry",
        ):
            manifest["output_sha256"][name] = sha256_file(
                staging_dir / manifest["outputs"][name]
            )
        write_json(manifest_path, manifest)

        verification = verify_cle(staging_dir, require_portable=True)
        if not verification["passed"]:
            raise ValueError(
                "Staged CLE package failed strict portability verification: "
                f"{verification['errors']}"
            )
        manifest["status"] = (
            "portable_package_verified_release_eligible"
            if manifest.get("release_eligible")
            else "portable_package_verified_release_gates_open"
        )
        manifest["portable_package_verified"] = True
        write_json(manifest_path, manifest)
        final_verification = verify_cle(staging_dir, require_portable=True)
        if not final_verification["passed"]:
            raise ValueError(
                "Final CLE package failed strict portability verification: "
                f"{final_verification['errors']}"
            )
        staging_dir.replace(destination_cle_dir)
        return {
            **final_verification,
            "packaging_action": "created_verified_package",
            "destination": str(destination_cle_dir),
        }
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def _verify_portability(cle_dir: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_outputs = ("operational_graph", "road_manifest", "graph_reference")
    outputs = manifest.get("outputs", {})
    for name, relative in outputs.items():
        path_value = Path(str(relative))
        if path_value.is_absolute():
            errors.append(f"absolute runtime output path for {name}: {relative}")
            continue
        if not _is_within(cle_dir / path_value, cle_dir):
            errors.append(f"runtime output escapes CLE root for {name}: {relative}")
    for name in required_outputs:
        if name not in outputs:
            errors.append(f"portable manifest omits {name}")

    graph_reference_relative = outputs.get("graph_reference")
    if graph_reference_relative:
        graph_reference_path = cle_dir / str(graph_reference_relative)
        if graph_reference_path.exists():
            reference = _read_json(graph_reference_path)
            if reference.get("path_base") != "graph_reference_directory":
                errors.append("graph reference does not declare a relative path base")
            for name in ("operational_graph", "road_manifest"):
                record = reference.get(name, {})
                relative = Path(str(record.get("path", "")))
                if relative.is_absolute():
                    errors.append(f"graph reference uses an absolute {name} path")
                    continue
                referenced = graph_reference_path.parent / relative
                if not _is_within(referenced, cle_dir):
                    errors.append(f"graph reference {name} escapes CLE root")
                elif not referenced.is_file():
                    errors.append(f"graph reference {name} is missing")
                elif sha256_file(referenced) != record.get("sha256"):
                    errors.append(f"graph reference SHA-256 mismatch for {name}")
    source_registry_relative = outputs.get("source_registry")
    if source_registry_relative:
        source_registry_path = cle_dir / str(source_registry_relative)
        if source_registry_path.exists():
            registry = _read_json(source_registry_path)
            for name, record in registry.get("sources", {}).items():
                path_value = record.get("path")
                if not path_value:
                    continue
                relative = Path(str(path_value))
                if relative.is_absolute():
                    errors.append(f"source registry uses an absolute path for {name}")
                elif not _is_within(source_registry_path.parent / relative, cle_dir):
                    errors.append(f"source registry path escapes CLE root for {name}")
    if not manifest.get("package", {}).get("portable"):
        errors.append("manifest does not declare a portable package")
    return errors


def verify_cle(cle_dir: Path, *, require_portable: bool = False) -> dict[str, Any]:
    manifest_path = cle_dir / "manifest.json"
    if not manifest_path.exists():
        return {"passed": False, "errors": ["manifest.json is missing"]}
    manifest = _read_json(manifest_path)
    errors: list[str] = []
    warnings: list[str] = []
    for name, relative in manifest.get("outputs", {}).items():
        path = cle_dir / relative
        if not path.exists():
            errors.append(f"missing output {name}: {relative}")
            continue
        if sha256_file(path) != manifest.get("output_sha256", {}).get(name):
            errors.append(f"SHA-256 mismatch for {name}")
    source_registry_path = cle_dir / manifest.get("outputs", {}).get(
        "source_registry", "source_registry.json"
    )
    if source_registry_path.exists():
        source_registry = _read_json(source_registry_path)
        for name, record in source_registry.get("sources", {}).items():
            path_value = record.get("path")
            if not path_value:
                if record.get("availability") != "build_provenance_only":
                    warnings.append(f"source has no checkable path: {name}")
                continue
            source_path = Path(str(path_value))
            if not source_path.is_absolute():
                source_path = source_registry_path.parent / source_path
            if not source_path.exists():
                warnings.append(f"source unavailable for live hash check: {name}")
            elif sha256_file(source_path) != record.get("sha256"):
                errors.append(f"source SHA-256 mismatch for {name}")
    location_path = cle_dir / manifest.get("outputs", {}).get(
        "latent_locations", "service_locations/latent_locations.parquet"
    )
    if location_path.exists():
        locations = gpd.read_parquet(location_path)
        expected_locations = manifest.get("layer_counts", {}).get(
            "latent_service_location_candidates"
        )
        if expected_locations is not None and len(locations) != expected_locations:
            errors.append("latent service-location count differs from manifest")
        if locations["latent_service_location_id"].duplicated().any():
            errors.append("latent_service_location_id is not unique")
        if locations["active_customer"].astype(bool).any():
            errors.append("Stage 1 CLE contains active customers")
        if locations["customer_release_eligible"].astype(bool).any():
            errors.append("release-gated CLE contains release-eligible customers")
        for output_name, id_column, expected_key in (
            (
                "service_access_nodes",
                "service_access_node_id",
                "latent_service_location_candidates",
            ),
            (
                "road_projection_nodes",
                "road_projection_node_id",
                "unique_road_projection_nodes",
            ),
            (
                "service_access_connectors",
                "service_access_connector_id",
                "service_access_connectors",
            ),
        ):
            relative = manifest.get("outputs", {}).get(output_name)
            if not relative:
                errors.append(f"manifest omits {output_name}")
                continue
            frame = pd.read_parquet(cle_dir / relative)
            expected = manifest.get("layer_counts", {}).get(expected_key)
            if expected is not None and len(frame) != expected:
                errors.append(f"{output_name} row count differs from manifest")
            if frame[id_column].duplicated().any():
                errors.append(f"{id_column} is not unique")
    charger_path = cle_dir / manifest.get("outputs", {}).get(
        "chargers", "infrastructure/chargers.parquet"
    )
    if charger_path.exists():
        chargers = gpd.read_parquet(charger_path)
        if chargers["charger_id"].duplicated().any():
            errors.append("charger_id is not unique")
        if chargers["charger_release_eligible"].astype(bool).any():
            errors.append("release-gated CLE contains release-eligible chargers")
    depot_path = cle_dir / manifest.get("outputs", {}).get(
        "depots", "infrastructure/depots.parquet"
    )
    if depot_path.exists():
        depots = gpd.read_parquet(depot_path)
        if depots["candidate_id"].duplicated().any():
            errors.append("depot candidate_id is not unique")
        if depots["depot_release_eligible"].astype(bool).any():
            errors.append("release-gated CLE contains release-eligible depots")
    speed_path = cle_dir / manifest.get("outputs", {}).get(
        "directed_legal_speeds", "profiles/directed_legal_speeds.parquet"
    )
    if speed_path.exists():
        speeds = pd.read_parquet(speed_path)
        expected_edges = manifest.get("layer_counts", {}).get("directed_legal_speed_edges")
        if expected_edges is not None and len(speeds) != expected_edges:
            errors.append("directed legal-speed edge count differs from manifest")
        if speeds["edge_id"].duplicated().any():
            errors.append("directed legal-speed edge_id is not unique")
        required_numeric = ["length_m", "speed_limit_kph", "legal_travel_time_s"]
        if "reference_speed_kph" in speeds.columns:
            required_numeric.extend(["reference_speed_kph", "reference_travel_time_s"])
            if (speeds["reference_speed_kph"] > speeds["speed_limit_kph"] + 1e-9).any():
                errors.append("reference speed exceeds legal speed")
        numeric = speeds[required_numeric].to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or (numeric <= 0).any():
            errors.append("legal-speed layer contains nonpositive or nonfinite values")
    scenario_path = cle_dir / manifest.get("outputs", {}).get(
        "static_operational_scenarios", "profiles/static_operational_scenarios.parquet"
    )
    if "static_operational_scenarios" in manifest.get("outputs", {}) and scenario_path.exists():
        scenarios = pd.read_parquet(scenario_path)
        expected_rows = manifest.get("layer_counts", {}).get(
            "static_operational_scenario_edge_rows"
        )
        if expected_rows is not None and len(scenarios) != expected_rows:
            errors.append("static operational scenario row count differs from manifest")
        scenario_numeric = scenarios[
            ["operational_variation_factor", "speed_kph", "travel_time_s"]
        ].to_numpy(dtype=float)
        if not np.isfinite(scenario_numeric).all() or (scenario_numeric <= 0).any():
            errors.append("static operational scenarios contain invalid numeric values")
        if (scenarios["speed_kph"] > scenarios["speed_limit_kph"] + 1e-9).any():
            errors.append("static operational scenario speed exceeds legal speed")
    report_path = cle_dir / manifest.get("outputs", {}).get(
        "qa_report", "qa/cle_report.json"
    )
    if report_path.exists():
        report = _read_json(report_path)
        if report.get("release_eligible") is not False:
            errors.append("QA report incorrectly marks the CLE release eligible")
        if report.get("blocked_gate_count") != manifest.get("release_blocker_count"):
            errors.append("QA and manifest release-blocker counts differ")
    portability_errors = _verify_portability(cle_dir, manifest)
    technical_verification_passed = not errors
    if require_portable:
        errors.extend(portability_errors)
    return {
        "passed": not errors,
        "technical_verification_passed": technical_verification_passed,
        "portable": not portability_errors,
        "portability_errors": portability_errors,
        "errors": errors,
        "warnings": warnings,
        "status": manifest.get("status"),
        "release_eligible": manifest.get("release_eligible"),
        "release_blocker_count": manifest.get("release_blocker_count"),
        "layer_counts": manifest.get("layer_counts"),
    }
