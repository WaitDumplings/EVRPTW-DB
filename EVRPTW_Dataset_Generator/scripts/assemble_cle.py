#!/usr/bin/env python3
"""Assemble or verify one City Logistics Environment (CLE)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evrptw_cle.cle import (
    assemble_cle,
    verify_cle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-slug", required=True)
    parser.add_argument("--city-label", required=True)
    parser.add_argument("--cle-dir", type=Path, required=True)
    parser.add_argument("--admin-boundary", type=Path)
    parser.add_argument("--service-boundary", type=Path)
    parser.add_argument("--road-manifest", type=Path)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--building-manifest", type=Path)
    parser.add_argument("--nsi-manifest", type=Path)
    parser.add_argument("--spatial-manifest", type=Path)
    parser.add_argument("--nearest-audit", type=Path)
    parser.add_argument("--access-audit", type=Path)
    parser.add_argument("--access-candidates", type=Path)
    parser.add_argument("--service-access-nodes", type=Path)
    parser.add_argument("--road-projection-nodes", type=Path)
    parser.add_argument("--service-access-connectors", type=Path)
    parser.add_argument("--facility-manifest", type=Path)
    parser.add_argument("--speed-manifest", type=Path)
    parser.add_argument("--candidate-access-threshold-m", type=float, default=200.0)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--require-portable",
        action="store_true",
        help="Fail verification unless all runtime paths are packaged inside the CLE.",
    )
    args = parser.parse_args()
    if args.verify_only:
        result = verify_cle(args.cle_dir, require_portable=args.require_portable)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        raise SystemExit(0 if result["passed"] else 1)
    required = {
        "admin_boundary": args.admin_boundary,
        "service_boundary": args.service_boundary,
        "road_manifest": args.road_manifest,
        "graph": args.graph,
        "building_manifest": args.building_manifest,
        "nsi_manifest": args.nsi_manifest,
        "spatial_manifest": args.spatial_manifest,
        "nearest_audit": args.nearest_audit,
        "access_audit": args.access_audit,
        "access_candidates": args.access_candidates,
        "service_access_nodes": args.service_access_nodes,
        "road_projection_nodes": args.road_projection_nodes,
        "service_access_connectors": args.service_access_connectors,
        "facility_manifest": args.facility_manifest,
        "speed_manifest": args.speed_manifest,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"assembly requires: {', '.join(missing)}")
    result = assemble_cle(
        city_slug=args.city_slug,
        city_label=args.city_label,
        cle_dir=args.cle_dir,
        admin_boundary_path=args.admin_boundary,
        service_boundary_path=args.service_boundary,
        road_manifest_path=args.road_manifest,
        graph_path=args.graph,
        building_manifest_path=args.building_manifest,
        nsi_manifest_path=args.nsi_manifest,
        spatial_manifest_path=args.spatial_manifest,
        nearest_audit_path=args.nearest_audit,
        access_audit_path=args.access_audit,
        access_candidates_path=args.access_candidates,
        service_access_nodes_path=args.service_access_nodes,
        road_projection_nodes_path=args.road_projection_nodes,
        service_access_connectors_path=args.service_access_connectors,
        facility_manifest_path=args.facility_manifest,
        speed_manifest_path=args.speed_manifest,
        candidate_access_threshold_m=args.candidate_access_threshold_m,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
