#!/usr/bin/env python3
"""Build paper-ready, provenance-backed tables for a portable CLE cohort."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from evrptw_cle.util import sha256_file, write_json


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gate(report: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in report["gates"] if item["gate"] == name)


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_tables(cle_root: Path, output_dir: Path, *, replace: bool) -> dict[str, Any]:
    if output_dir.exists():
        if not replace:
            raise FileExistsError(f"Appendix table directory exists: {output_dir}")
        shutil.rmtree(output_dir)
    city_root = cle_root / "cities"
    city_dirs = sorted(
        path
        for path in city_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if not city_dirs:
        raise FileNotFoundError(f"No portable city packages under {city_root}")

    scale_rows: list[dict[str, Any]] = []
    road_rows: list[dict[str, Any]] = []
    service_rows: list[dict[str, Any]] = []
    facility_rows: list[dict[str, Any]] = []
    speed_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    source_manifests: list[dict[str, str]] = []
    for city_dir in city_dirs:
        manifest_path = city_dir / "manifest.json"
        report_path = city_dir / "qa/cle_report.json"
        road_path = city_dir / "graph/road_manifest.json"
        facility_path = city_dir / "infrastructure/facility_manifest.json"
        speed_path = city_dir / "profiles/speed_manifest.json"
        manifest = _read(manifest_path)
        report = _read(report_path)
        road = _read(road_path)
        facility = _read(facility_path)
        speed = _read(speed_path)
        counts = manifest["layer_counts"]
        customer = report["customer_layer"]
        connectivity = _gate(report, "protected_anchor_roundtrip_connectivity")[
            "evidence"
        ]

        scale_rows.append(
            {
                "city": manifest["city_label"],
                "road_nodes": counts["road_nodes"],
                "directed_edges": counts["road_edges"],
                "latent_service_locations": counts[
                    "latent_service_location_candidates"
                ],
                "default_service_pool": counts[
                    "default_instance_eligible_service_locations"
                ],
                "charger_candidates": counts["charger_candidates"],
                "depot_candidates": counts["depot_candidates"],
            }
        )
        operational = road["operational_connectivity"]
        fallback = operational.get("small_isolated_component_fallback") or {}
        road_rows.append(
            {
                "city": manifest["city_label"],
                "buffer_km": operational["selected_buffer_km"],
                "raw_node_coverage_pct": 100 * operational["city_node_coverage"],
                "raw_road_length_coverage_pct": 100
                * operational["city_physical_road_length_coverage"],
                "coverage_gate_mode": operational.get(
                    "coverage_gate_mode", "primary_raw_coverage"
                ),
                "gate_node_coverage_pct": 100
                * operational.get(
                    "coverage_gate_city_node_coverage",
                    operational["city_node_coverage"],
                ),
                "gate_road_length_coverage_pct": 100
                * operational.get(
                    "coverage_gate_city_physical_road_length_coverage",
                    operational["city_physical_road_length_coverage"],
                ),
                "fallback_skipped_components": fallback.get(
                    "auto_skipped_component_count", 0
                ),
                "fallback_skipped_nodes": fallback.get("auto_skipped_node_count", 0),
                "fallback_skipped_road_m": fallback.get(
                    "auto_skipped_physical_road_length_m", 0.0
                ),
                "largest_scc_node_pct": 100
                * connectivity["reference_scc_node_share"],
                "customer_scc_quarantine": connectivity[
                    "customer_roundtrip_quarantined_count"
                ],
                "depot_scc_quarantine": connectivity[
                    "depot_roundtrip_quarantined_count"
                ],
                "charger_scc_quarantine": connectivity[
                    "charger_roundtrip_quarantined_count"
                ],
            }
        )
        types = customer["service_location_type_counts"]
        service_rows.append(
            {
                "city": manifest["city_label"],
                "nsi_residential_records": counts[
                    "nsi_ordinary_residential_records"
                ],
                "g1_locations": customer["g1_location_count"],
                "g2_pending_locations": customer[
                    "g2_manual_audit_pending_location_count"
                ],
                "road_distance_gt_200m": customer[
                    "road_access_distance_qa_flag_count"
                ],
                "scc_quarantined_locations": customer[
                    "protected_roundtrip_quarantined_location_count"
                ],
                "house": types.get("house", 0),
                "manufactured_home": types.get("manufactured_home", 0),
                "small_apt": types.get("small_apt", 0),
                "medium_apt": types.get("medium_apt", 0),
                "large_apt": types.get("large_apt", 0),
            }
        )
        charging = facility["charging"]
        depots = facility["depots"]
        facility_rows.append(
            {
                "city": manifest["city_label"],
                "afdc_sites_inside_boundary": charging[
                    "inside_boundary_public_available_site_count"
                ],
                "charger_exact_geometry": charging[
                    "coordinate_exact_geometry_count"
                ],
                "charger_address_corroborated": charging[
                    "coordinate_address_only_count"
                ],
                "charger_uncorroborated": charging[
                    "coordinate_uncorroborated_count"
                ],
                "charger_distance_gt_250m": charging[
                    "road_access_distance_qa_flag_count"
                ],
                "charger_candidate_pool": charging["candidate_eligible_count"],
                "depot_retained": depots["retained_candidate_count"],
                "depot_strict": depots["strict_candidate_eligible_count"],
                "depot_optional": depots["optional_candidate_eligible_count"],
                "depot_candidate_pool": depots["candidate_eligible_count"],
            }
        )
        speed_rows.append(
            {
                "city": manifest["city_label"],
                "directed_edges": speed["edge_count"],
                "observed_osm_maxspeed_pct": 100
                * speed["observed_osm_maxspeed_edge_share"],
                "imputed_speed_edges": speed["imputed_edge_count"],
                "hpms_matched_edges": speed.get(
                    "observed_hpms_speed_limit_edge_count", 0
                ),
                "reference_profile": speed.get("reference_speed_contract", {}).get(
                    "profile_id", ""
                ),
            }
        )
        blocked_gates = [
            str(item["gate"])
            for item in report["gates"]
            if item.get("status") == "blocked"
        ]
        validation_rows.append(
            {
                "city": manifest["city_label"],
                "technical_verification_passed": manifest[
                    "technical_verification_passed"
                ],
                "portable_package_verified": manifest[
                    "portable_package_verified"
                ],
                "release_eligible": manifest["release_eligible"],
                "release_blocker_count": manifest["release_blocker_count"],
                "blocked_gates": ";".join(blocked_gates),
            }
        )
        source_manifests.append(
            {
                "city_slug": manifest["city_slug"],
                "manifest": str(manifest_path.relative_to(cle_root)),
                "sha256": sha256_file(manifest_path),
            }
        )

    frames = {
        "city_scale": pd.DataFrame(scale_rows),
        "road_connectivity": pd.DataFrame(road_rows),
        "service_locations": pd.DataFrame(service_rows),
        "facilities": pd.DataFrame(facility_rows),
        "speed_evidence": pd.DataFrame(speed_rows),
        "validation_status": pd.DataFrame(validation_rows),
    }
    output_dir.mkdir(parents=True)
    outputs: dict[str, str] = {}
    for name, frame in frames.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        outputs[name] = path.name

    overview = frames["city_scale"]
    totals = {
        column: int(overview[column].sum())
        for column in overview.columns
        if column != "city"
    }
    document = [
        "# CLE cohort tables for the dataset and benchmark appendix",
        "",
        "These tables are generated from the portable CLE manifests, not copied from logs.",
        "Distance values of 200 m and 250 m are QA references, not deletion rules.",
        "SCC-quarantined source locations remain in the provenance layer but are excluded",
        "from the default benchmark candidate pool.",
    ]
    for name, frame in frames.items():
        document.extend(["", f"## {name.replace('_', ' ').title()}", "", _markdown_table(frame)])
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(document) + "\n", encoding="utf-8")
    outputs["readme"] = readme_path.name

    payload = {
        "schema": "evrptw_cle_appendix_tables_v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "city_count": len(city_dirs),
        "cohort_totals": totals,
        "display_note": {
            "minimum_raw_node_coverage": _percent(
                min(row["raw_node_coverage_pct"] for row in road_rows) / 100
            ),
            "minimum_raw_road_length_coverage": _percent(
                min(row["raw_road_length_coverage_pct"] for row in road_rows) / 100
            ),
            "minimum_gate_node_coverage": _percent(
                min(row["gate_node_coverage_pct"] for row in road_rows) / 100
            ),
            "minimum_gate_road_length_coverage": _percent(
                min(row["gate_road_length_coverage_pct"] for row in road_rows) / 100
            ),
        },
        "source_manifests": source_manifests,
        "outputs": outputs,
    }
    manifest_path = output_dir / "appendix_tables.json"
    write_json(manifest_path, payload)
    payload["output_sha256"] = {
        name: sha256_file(output_dir / relative) for name, relative in outputs.items()
    }
    write_json(manifest_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cle-root",
        type=Path,
        default=Path("../EVRPTW_Dataset/CLE_v1/us_11city"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    root = args.cle_root.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "appendix_tables"
    )
    result = build_tables(root, output, replace=args.replace)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
