from __future__ import annotations

import argparse
import json
from pathlib import Path

from .buildings import extract_building_footprints
from .connectivity import COMPONENT_POLICIES
from .nsi import (
    DEFAULT_NSI_API_URL,
    NSICustomerOptions,
    build_nsi_customer_cle,
    verify_nsi_customer_cle,
)
from .pipeline import BuildOptions, audit_existing_graph, build_city, change_policy
from .release import build_release_index
from .verification import verify_city_output


def _add_build_arguments(parser: argparse.ArgumentParser, include_city: bool = True) -> None:
    if include_city:
        parser.add_argument(
            "--city",
            required=True,
            help='Qualified city query, e.g. "Boston, Massachusetts, USA"',
        )
    parser.add_argument("--output-root", type=Path, default=Path("data/cities"))
    parser.add_argument("--slug")
    parser.add_argument("--boundary-file", type=Path)
    parser.add_argument("--query-mask-file", type=Path)
    parser.add_argument("--which-result", type=int, default=1)
    parser.add_argument("--component-policy", choices=COMPONENT_POLICIES, default="all")
    parser.add_argument("--network-type", default="drive")
    parser.add_argument("--overpass-url", default="https://overpass-api.de/api")
    parser.add_argument("--request-timeout-s", type=int, default=600)
    parser.add_argument("--max-query-area-km2", type=float, default=100.0)
    parser.add_argument(
        "--overpass-rate-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the endpoint's /status slot protocol; disable for incompatible mirrors",
    )
    parser.add_argument("--query-buffer-m", type=float, default=5_000.0)
    parser.add_argument("--query-simplify-m", type=float, default=100.0)
    parser.add_argument(
        "--query-component-min-area-km2",
        type=float,
        default=0.0,
        help="Development-only query-mask pruning; 0 preserves every polygon component",
    )
    parser.add_argument(
        "--pbf-file",
        type=Path,
        help="Optional frozen regional .osm.pbf; bypasses the live Overpass backend",
    )
    parser.add_argument(
        "--pbf-source-url",
        help="Public source URL recorded in provenance for --pbf-file",
    )
    parser.add_argument(
        "--keep-pbf-intermediates",
        action="store_true",
        help="Retain city extract/XML intermediates for debugging",
    )
    parser.add_argument(
        "--build-operational-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build the connected actual-OSM routing graph in addition to the raw city graph",
    )
    parser.add_argument(
        "--routing-buffer-ladder-km",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 5.0, 10.0, 20.0],
        help="Increasing routing-envelope buffers; the smallest passing value is selected",
    )
    parser.add_argument("--min-retained-node-coverage", type=float, default=0.99)
    parser.add_argument("--min-retained-road-length-coverage", type=float, default=0.995)
    parser.add_argument(
        "--auto-skip-component-node-threshold",
        type=int,
        default=100,
        help=(
            "After exhausting the real-OSM buffer ladder, allow still-uncovered weak "
            "components smaller than this exclusive node threshold to be skipped"
        ),
    )
    parser.add_argument("--micro-component-node-threshold", type=int, default=10)
    parser.add_argument("--micro-component-length-km-threshold", type=float, default=1.0)


def _options_from_args(
    args,
    city_query: str | None = None,
    slug: str | None = None,
    boundary_file: Path | None = None,
    query_mask_file: Path | None = None,
    pbf_file: Path | None = None,
    pbf_source_url: str | None = None,
) -> BuildOptions:
    return BuildOptions(
        city_query=city_query or args.city,
        output_root=args.output_root,
        slug=slug if slug is not None else args.slug,
        boundary_file=boundary_file if boundary_file is not None else args.boundary_file,
        query_mask_file=query_mask_file if query_mask_file is not None else args.query_mask_file,
        which_result=args.which_result,
        component_policy=args.component_policy,
        network_type=args.network_type,
        overpass_url=args.overpass_url,
        request_timeout_s=args.request_timeout_s,
        max_query_area_km2=args.max_query_area_km2,
        overpass_rate_limit=args.overpass_rate_limit,
        query_buffer_m=args.query_buffer_m,
        query_simplify_m=args.query_simplify_m,
        query_component_min_area_km2=args.query_component_min_area_km2,
        pbf_file=pbf_file if pbf_file is not None else args.pbf_file,
        pbf_source_url=(pbf_source_url if pbf_source_url is not None else args.pbf_source_url),
        keep_pbf_intermediates=args.keep_pbf_intermediates,
        build_operational_graph=args.build_operational_graph,
        routing_buffer_ladder_km=tuple(args.routing_buffer_ladder_km),
        min_retained_node_coverage=args.min_retained_node_coverage,
        min_retained_road_length_coverage=args.min_retained_road_length_coverage,
        auto_skip_component_node_threshold=args.auto_skip_component_node_threshold,
        micro_component_node_threshold=args.micro_component_node_threshold,
        micro_component_length_km_threshold=args.micro_component_length_km_threshold,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evrptw-cle",
        description="Build directed OSM road graphs with connectivity audits for CLEs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Resolve one city and download its road graph")
    _add_build_arguments(build)

    batch = subparsers.add_parser("batch", help="Build every city in a preset JSON file")
    batch.add_argument("--preset", type=Path, required=True)
    batch.add_argument("--continue-on-error", action="store_true")
    batch.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a city when its manifest and complete graph already exist",
    )
    _add_build_arguments(batch, include_city=False)

    audit = subparsers.add_parser("audit", help="Audit and visualize an existing GraphML")
    audit.add_argument("--graph", type=Path, required=True)
    audit.add_argument("--boundary", type=Path, required=True)
    audit.add_argument(
        "--display-boundary",
        type=Path,
        help="Optional land-only mask for visualization; boundary remains the QA clip",
    )
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--city-label", required=True)
    audit.add_argument("--component-policy", choices=COMPONENT_POLICIES, default="all")

    policy = subparsers.add_parser(
        "set-policy",
        help="Choose all components or the largest weak component without re-downloading OSM",
    )
    policy.add_argument("--city-dir", type=Path, required=True)
    policy.add_argument("--component-policy", choices=COMPONENT_POLICIES, required=True)

    index = subparsers.add_parser("release-index", help="Build a checksum index for a preset")
    index.add_argument("--preset", type=Path, required=True)
    index.add_argument("--city-root", type=Path, default=Path("data/cities"))
    index.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="Verify checksums and graph counts")
    verify.add_argument("--city-dir", type=Path, required=True)

    buildings = subparsers.add_parser(
        "extract-buildings",
        help="Stream a state USBuildingFootprints GeoJSON into city-level GeoParquet",
    )
    buildings.add_argument("--source", type=Path, required=True)
    buildings.add_argument("--preset", type=Path, required=True)
    buildings.add_argument("--output-root", type=Path, default=Path("data/buildings"))
    buildings.add_argument("--batch-size", type=int, default=50_000)
    buildings.add_argument("--density-grid-m", type=float, default=500.0)

    nsi = subparsers.add_parser(
        "build-nsi-customers",
        help="Build an NSI residential latent-location pool and map it to an OSM graph",
    )
    nsi.add_argument("--city-slug", required=True)
    nsi.add_argument("--city-label", required=True)
    nsi.add_argument("--boundary", type=Path, required=True)
    nsi.add_argument("--graph", type=Path, required=True)
    nsi.add_argument("--output-dir", type=Path, required=True)
    nsi.add_argument("--area-crs", required=True)
    nsi.add_argument("--api-url", default=DEFAULT_NSI_API_URL)
    nsi.add_argument("--tile-size-m", type=float, default=5_000.0)
    nsi.add_argument("--density-grid-m", type=float, default=500.0)
    nsi.add_argument("--workers", type=int, default=4)
    nsi.add_argument("--timeout-s", type=int, default=300)
    nsi.add_argument("--retries", type=int, default=4)

    verify_nsi = subparsers.add_parser(
        "verify-nsi-customers",
        help="Verify hashes and semantic gates for an NSI customer CLE layer",
    )
    verify_nsi.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.command == "build":
        manifest = build_city(_options_from_args(args))
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return
    if args.command == "batch":
        preset = json.loads(args.preset.read_text(encoding="utf-8"))
        failures = []
        for item in preset["cities"]:
            try:
                city_dir = args.output_root / item["slug"]
                if (
                    args.skip_existing
                    and (city_dir / "manifest.json").exists()
                    and (city_dir / "graph_all.graphml").exists()
                    and (
                        not args.build_operational_graph
                        or (city_dir / "graph_operational.graphml").exists()
                    )
                ):
                    print(f"SKIP {item['slug']} existing complete graph")
                    continue
                boundary_file = (
                    (args.preset.parent / item["boundary_file"]).resolve()
                    if item.get("boundary_file")
                    else None
                )
                query_mask_file = (
                    (args.preset.parent / item["query_mask_file"]).resolve()
                    if item.get("query_mask_file")
                    else None
                )
                pbf_file = (
                    (args.preset.parent / item["pbf_file"]).resolve()
                    if item.get("pbf_file")
                    else None
                )
                manifest = build_city(
                    _options_from_args(
                        args,
                        city_query=item["query"],
                        slug=item["slug"],
                        boundary_file=boundary_file,
                        query_mask_file=query_mask_file,
                        pbf_file=pbf_file,
                        pbf_source_url=item.get("pbf_source_url"),
                    )
                )
                operational = manifest.get("operational_connectivity") or {}
                print(
                    f"DONE {item['slug']} raw={manifest['connectivity']['weak_component_count']} WCC "
                    f"operational_buffer={operational.get('selected_buffer_km')}km "
                    f"coverage={operational.get('city_node_coverage', 0.0):.6f}"
                )
            except Exception as error:
                failures.append({"slug": item["slug"], "error": repr(error)})
                print(f"FAILED {item['slug']}: {error}")
                if not args.continue_on_error:
                    raise
        if failures:
            raise SystemExit(json.dumps({"failures": failures}, indent=2))
        return
    if args.command == "audit":
        manifest = audit_existing_graph(
            args.graph,
            args.boundary,
            args.output_dir,
            args.city_label,
            args.component_policy,
            args.display_boundary,
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return
    if args.command == "set-policy":
        manifest = change_policy(args.city_dir, args.component_policy)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return
    if args.command == "release-index":
        payload = build_release_index(args.preset, args.city_root, args.output)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    if args.command == "verify":
        result = verify_city_output(args.city_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if not result["passed"]:
            raise SystemExit(1)
        return
    if args.command == "extract-buildings":
        manifest = extract_building_footprints(
            source_path=args.source,
            preset_path=args.preset,
            output_root=args.output_root,
            batch_size=args.batch_size,
            density_grid_m=args.density_grid_m,
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return
    if args.command == "build-nsi-customers":
        manifest = build_nsi_customer_cle(
            NSICustomerOptions(
                city_slug=args.city_slug,
                city_label=args.city_label,
                boundary_file=args.boundary,
                graph_file=args.graph,
                output_dir=args.output_dir,
                area_crs=args.area_crs,
                api_url=args.api_url,
                tile_size_m=args.tile_size_m,
                density_grid_m=args.density_grid_m,
                workers=args.workers,
                timeout_s=args.timeout_s,
                retries=args.retries,
            )
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return
    if args.command == "verify-nsi-customers":
        result = verify_nsi_customer_cle(args.output_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if not result["passed"]:
            raise SystemExit(1)
        return
    parser.error(f"Unhandled command {args.command}")


if __name__ == "__main__":
    main()
