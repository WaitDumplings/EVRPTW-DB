from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import osmnx as ox

from .boundary import load_optional_query_mask, resolve_boundary
from .connectivity import COMPONENT_POLICIES, apply_component_policy, audit_and_label
from .operational import OperationalCoverageError, OperationalPolicy, select_operational_graph
from .osm_graph import configure_osmnx, download_drive_graph, load_drive_graph_from_pbf
from .util import sha256_file, slugify, write_json
from .visualization import render_connectivity_map, render_operational_map


@dataclass(frozen=True)
class BuildOptions:
    city_query: str
    output_root: Path
    slug: str | None = None
    boundary_file: Path | None = None
    query_mask_file: Path | None = None
    which_result: int = 1
    component_policy: str = "all"
    network_type: str = "drive"
    overpass_url: str = "https://overpass-api.de/api"
    request_timeout_s: int = 600
    max_query_area_km2: float = 100.0
    overpass_rate_limit: bool = True
    query_buffer_m: float = 5_000.0
    query_simplify_m: float = 100.0
    query_component_min_area_km2: float = 0.0
    pbf_file: Path | None = None
    pbf_source_url: str | None = None
    keep_pbf_intermediates: bool = False
    build_operational_graph: bool = True
    routing_buffer_ladder_km: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)
    min_retained_node_coverage: float = 0.99
    min_retained_road_length_coverage: float = 0.995
    auto_skip_component_node_threshold: int = 100
    micro_component_node_threshold: int = 10
    micro_component_length_km_threshold: float = 1.0


def _validate_options(options: BuildOptions) -> None:
    if options.component_policy not in COMPONENT_POLICIES:
        raise ValueError(f"component_policy must be one of {COMPONENT_POLICIES}")
    if options.which_result < 1:
        raise ValueError("which_result is 1-based and must be positive")
    if options.max_query_area_km2 <= 0:
        raise ValueError("max_query_area_km2 must be positive")
    OperationalPolicy(
        buffer_ladder_km=options.routing_buffer_ladder_km,
        min_node_coverage=options.min_retained_node_coverage,
        min_road_length_coverage=options.min_retained_road_length_coverage,
        auto_skip_component_node_threshold=options.auto_skip_component_node_threshold,
        micro_component_node_threshold=options.micro_component_node_threshold,
        micro_component_length_km_threshold=options.micro_component_length_km_threshold,
    ).validate()


def _save_boundary(boundary: gpd.GeoDataFrame, path: Path) -> None:
    boundary.to_file(path, driver="GeoJSON")


def _graph_boundary_qa(graph, boundary: gpd.GeoDataFrame) -> dict[str, Any]:
    nodes, _ = ox.convert.graph_to_gdfs(graph)
    polygon = boundary.geometry.iloc[0]
    outside = int((~nodes.geometry.intersects(polygon)).sum())
    return {
        "nodes_outside_exact_boundary": outside,
        "exact_boundary_node_test_passed": outside == 0,
    }


def _write_selected_graph(graph, city_dir: Path, policy: str) -> str:
    if policy == "all":
        return "graph_all.graphml"
    selected = apply_component_policy(graph, policy)
    name = "graph_largest_weak.graphml"
    ox.save_graphml(selected, city_dir / name)
    return name


def finalize_graph(
    graph,
    boundary: gpd.GeoDataFrame,
    city_dir: Path,
    city_label: str,
    component_policy: str,
    provenance: dict[str, Any] | None = None,
    display_boundary: gpd.GeoDataFrame | None = None,
    envelope_graph=None,
    operational_policy: OperationalPolicy | None = None,
) -> dict[str, Any]:
    """Audit, label, save, select, and visualize a directed city graph."""
    city_dir.mkdir(parents=True, exist_ok=True)
    if component_policy not in COMPONENT_POLICIES:
        raise ValueError(f"component_policy must be one of {COMPONENT_POLICIES}")
    audit = audit_and_label(graph)
    operational = None
    if envelope_graph is not None and operational_policy is not None:
        operational = select_operational_graph(
            envelope_graph,
            graph,
            audit,
            boundary,
            operational_policy,
        )
    audit.components.to_csv(city_dir / "components.csv", index=False)
    write_json(city_dir / "connectivity.json", audit.summary)
    _save_boundary(boundary, city_dir / "boundary.geojson")
    map_boundary = display_boundary if display_boundary is not None else boundary
    _save_boundary(map_boundary, city_dir / "map_boundary.geojson")

    all_graph_path = city_dir / "graph_all.graphml"
    ox.save_graphml(graph, all_graph_path)
    raw_selected_graph = _write_selected_graph(graph, city_dir, component_policy)
    visual = render_connectivity_map(
        graph,
        map_boundary,
        audit.components,
        city_dir,
        city_label,
    )
    boundary_qa = _graph_boundary_qa(graph, boundary)
    checksums = {
        "boundary.geojson": sha256_file(city_dir / "boundary.geojson"),
        "map_boundary.geojson": sha256_file(city_dir / "map_boundary.geojson"),
        "graph_all.graphml": sha256_file(all_graph_path),
        "components.csv": sha256_file(city_dir / "components.csv"),
        "connectivity.json": sha256_file(city_dir / "connectivity.json"),
    }
    if raw_selected_graph != all_graph_path.name:
        checksums[raw_selected_graph] = sha256_file(city_dir / raw_selected_graph)

    selected_graph = raw_selected_graph
    operational_payload = None
    operational_visual = None
    if operational is not None:
        operational_path = city_dir / "graph_operational.graphml"
        ox.save_graphml(operational.graph, operational_path)
        operational.audit.components.to_csv(
            city_dir / "operational_components.csv",
            index=False,
        )
        write_json(city_dir / "operational_connectivity.json", operational.summary)
        operational_visual = render_operational_map(
            operational.graph,
            map_boundary,
            city_dir,
            city_label,
            operational.summary,
        )
        checksums.update(
            {
                "graph_operational.graphml": sha256_file(operational_path),
                "operational_components.csv": sha256_file(city_dir / "operational_components.csv"),
                "operational_connectivity.json": sha256_file(
                    city_dir / "operational_connectivity.json"
                ),
            }
        )
        selected_graph = operational_path.name
        operational_payload = operational.summary

    manifest = {
        "schema": "evrptw_city_operational_road_graph_v3",
        "city_label": city_label,
        "created_utc": datetime.now(UTC).isoformat(),
        "graph_semantics": {
            "raw": "directed OSM MultiDiGraph clipped to the exact city boundary",
            "operational": (
                "one weakly connected actual-OSM routing graph; outside-city roads are "
                "transit-only; no synthetic connector edges"
                if operational_payload is not None
                else None
            ),
        },
        "component_policy": component_policy,
        "raw_component_policy": component_policy,
        "raw_selected_graph": raw_selected_graph,
        "selected_graph": selected_graph,
        "selected_graph_role": (
            "operational_routing" if operational_payload is not None else "raw_city_audit"
        ),
        "all_graph": all_graph_path.name,
        "raw_city_graph": all_graph_path.name,
        "operational_graph": (
            "graph_operational.graphml" if operational_payload is not None else None
        ),
        "connectivity": audit.summary,
        "operational_connectivity": operational_payload,
        "boundary_qa": boundary_qa,
        "visualization": visual,
        "operational_visualization": operational_visual,
        "checksums": checksums,
        "provenance": provenance or {},
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "osmnx": ox.__version__,
            "networkx": nx.__version__,
        },
    }
    write_json(city_dir / "manifest.json", manifest)
    return manifest


def build_city(options: BuildOptions) -> dict[str, Any]:
    _validate_options(options)
    slug = options.slug or slugify(options.city_query.split(",")[0])
    city_dir = options.output_root / slug
    city_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = options.output_root / ".cache" / slug

    boundary, boundary_source = resolve_boundary(
        options.city_query,
        options.boundary_file,
        options.which_result,
    )
    query_mask, query_mask_source = load_optional_query_mask(
        options.query_mask_file,
        boundary,
    )
    if options.pbf_file is None:
        configure_osmnx(
            cache_dir,
            options.overpass_url,
            options.request_timeout_s,
            options.max_query_area_km2,
            options.overpass_rate_limit,
        )

    extraction_attempts_m = [options.query_buffer_m]
    if options.build_operational_graph:
        extraction_attempts_m.extend(
            buffer_km * 1_000.0
            for buffer_km in options.routing_buffer_ladder_km
            if buffer_km * 1_000.0 > options.query_buffer_m
        )
    extraction_attempts_m = sorted(set(extraction_attempts_m))
    last_error: OperationalCoverageError | None = None

    for attempt_index, extraction_buffer_m in enumerate(extraction_attempts_m):
        if options.pbf_file is None:
            graph, envelope_graph, graph_query = download_drive_graph(
                boundary,
                query_mask,
                network_type=options.network_type,
                retain_all=True,
                query_buffer_m=extraction_buffer_m,
                query_simplify_m=options.query_simplify_m,
                query_component_min_area_km2=options.query_component_min_area_km2,
            )
            osm_provenance = {
                "source": "OpenStreetMap via OSMnx/Overpass",
                "snapshot_status": ("live API response; use a frozen PBF for a benchmark release"),
                "overpass_url": options.overpass_url,
                **graph_query,
            }
        else:
            graph, envelope_graph, graph_query = load_drive_graph_from_pbf(
                boundary,
                query_mask,
                pbf_file=options.pbf_file,
                work_dir=options.output_root / ".work" / slug,
                query_buffer_m=extraction_buffer_m,
                query_simplify_m=options.query_simplify_m,
                query_component_min_area_km2=options.query_component_min_area_km2,
                keep_intermediates=options.keep_pbf_intermediates,
            )
            osm_provenance = {
                "source": "OpenStreetMap Geofabrik/local PBF via Osmium and OSMnx",
                "snapshot_status": "frozen local PBF input",
                "pbf_file": str(options.pbf_file),
                "pbf_sha256": sha256_file(options.pbf_file),
                "pbf_source_url": options.pbf_source_url,
                **graph_query,
            }

        available_ladder = tuple(
            buffer_km
            for buffer_km in options.routing_buffer_ladder_km
            if buffer_km * 1_000.0 <= extraction_buffer_m
        )
        provenance = {
            "city_query": options.city_query,
            "boundary_source": boundary_source,
            "query_mask_source": query_mask_source,
            "osm": osm_provenance,
            "operational_extraction_attempts_m": extraction_attempts_m[: attempt_index + 1],
            "build_options": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(options).items()
            },
        }
        try:
            return finalize_graph(
                graph,
                boundary,
                city_dir,
                options.city_query,
                options.component_policy,
                provenance,
                query_mask,
                envelope_graph=(envelope_graph if options.build_operational_graph else None),
                operational_policy=(
                    OperationalPolicy(
                        buffer_ladder_km=available_ladder,
                        min_node_coverage=options.min_retained_node_coverage,
                        min_road_length_coverage=(options.min_retained_road_length_coverage),
                        auto_skip_component_node_threshold=(
                            options.auto_skip_component_node_threshold
                            if attempt_index == len(extraction_attempts_m) - 1
                            else 1
                        ),
                        micro_component_node_threshold=(options.micro_component_node_threshold),
                        micro_component_length_km_threshold=(
                            options.micro_component_length_km_threshold
                        ),
                    )
                    if options.build_operational_graph
                    else None
                ),
            )
        except OperationalCoverageError as error:
            last_error = error
            if attempt_index == len(extraction_attempts_m) - 1:
                raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("City build exhausted its extraction attempts without a result")


def audit_existing_graph(
    graph_file: Path,
    boundary_file: Path,
    output_dir: Path,
    city_label: str,
    component_policy: str,
    display_boundary_file: Path | None = None,
) -> dict[str, Any]:
    graph = ox.load_graphml(graph_file)
    boundary = gpd.read_file(boundary_file).to_crs("EPSG:4326")
    display_boundary = (
        gpd.read_file(display_boundary_file).to_crs("EPSG:4326")
        if display_boundary_file is not None
        else boundary
    )
    provenance = {
        "mode": "audit_existing_graph",
        "input_graph": str(graph_file),
        "input_graph_sha256": sha256_file(graph_file),
        "input_boundary": str(boundary_file),
        "input_boundary_sha256": sha256_file(boundary_file),
        "input_display_boundary": str(display_boundary_file or boundary_file),
        "input_display_boundary_sha256": sha256_file(display_boundary_file or boundary_file),
    }
    return finalize_graph(
        graph,
        boundary,
        output_dir,
        city_label,
        component_policy,
        provenance,
        display_boundary,
    )


def change_policy(city_dir: Path, policy: str) -> dict[str, Any]:
    if policy not in COMPONENT_POLICIES:
        raise ValueError(f"policy must be one of {COMPONENT_POLICIES}")
    graph_path = city_dir / "graph_all.graphml"
    if not graph_path.exists():
        raise FileNotFoundError(graph_path)
    graph = ox.load_graphml(graph_path)
    selected_graph = _write_selected_graph(graph, city_dir, policy)
    manifest_path = city_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["component_policy"] = policy
    manifest["raw_component_policy"] = policy
    manifest["raw_selected_graph"] = selected_graph
    if not manifest.get("operational_graph"):
        manifest["selected_graph"] = selected_graph
    if selected_graph != "graph_all.graphml":
        manifest["checksums"][selected_graph] = sha256_file(city_dir / selected_graph)
    write_json(manifest_path, manifest)
    return manifest
