#!/usr/bin/env python3
"""Build a compact, reproducible review map for one assembled cle.

The exact graph and full latent-location tables remain in the cle.  This
script creates a deterministic display payload only: physical roads and latent
locations are stratified/downsampled, while all depot and charger records are
shown.  The payload therefore supports visual review but is never an analytical
substitute for the source layers.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, MultiLineString, mapping
from shapely.geometry import shape as shapely_shape

from evrptw_cle.util import sha256_file, write_json

SERVICE_TYPE_ORDER = (
    "house",
    "manufactured_home",
    "small_apt",
    "medium_apt",
    "large_apt",
)
SERVICE_TYPE_CODE = {name: index for index, name in enumerate(SERVICE_TYPE_ORDER)}
SERVICE_SAMPLE_CAPS = {
    "house": 3_500,
    "manufactured_home": 700,
    "small_apt": 1_200,
    "medium_apt": 1_000,
    "large_apt": 1_000,
}
ROAD_SAMPLE_CAPS = {
    ("major", False): 1_500,
    ("major", True): 1_000,
    ("mid", False): 1_500,
    ("mid", True): 700,
    ("local", False): 900,
    ("local", True): 400,
}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _first_tag(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else "unknown"
    text = str(value)
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
        if isinstance(parsed, list) and parsed:
            return str(parsed[0])
    return text


def _road_group(highway: Any) -> str:
    tag = _first_tag(highway)
    if tag.startswith(("motorway", "trunk", "primary")):
        return "major"
    if tag.startswith(("secondary", "tertiary")):
        return "mid"
    return "local"


def _stable_score(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def _stable_take(frame: pd.DataFrame, cap: int, key_columns: list[str]) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.copy()
    keys = frame[key_columns].astype(str).agg("|".join, axis=1)
    scores = keys.map(_stable_score)
    return frame.loc[scores.nsmallest(cap).index].copy()


def _round_or_none(value: Any, digits: int) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(numeric) else round(float(numeric), digits)


def _deduplicate_physical_roads(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    frame = edges.reset_index().copy()
    frame["transit_only_bool"] = frame["transit_only"].map(_bool)
    frame["road_group"] = frame["highway"].map(_road_group)
    # Shapely normalization makes reverse-direction copies share one key.
    frame["physical_geometry_key"] = frame.geometry.map(
        lambda geom: geom.normalize().wkb_hex if geom is not None else ""
    )
    return frame.drop_duplicates(["physical_geometry_key", "transit_only_bool"])


def _select_roads(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    physical = _deduplicate_physical_roads(edges)
    selected: list[gpd.GeoDataFrame] = []
    for (group, transit_only), cap in ROAD_SAMPLE_CAPS.items():
        subset = physical[
            physical["road_group"].eq(group)
            & physical["transit_only_bool"].eq(transit_only)
        ]
        selected.append(_stable_take(subset, cap, ["u", "v", "key", "physical_geometry_key"]))
    result = pd.concat(selected, ignore_index=True)
    return gpd.GeoDataFrame(result, geometry="geometry", crs=edges.crs)


def _line_parts(geometry: Any) -> list[LineString]:
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, MultiLineString):
        return list(geometry.geoms)
    return []


def _compact_roads(frame: gpd.GeoDataFrame) -> list[list[Any]]:
    projected = frame.to_crs(frame.estimate_utm_crs())
    projected["geometry"] = projected.geometry.simplify(12.0, preserve_topology=True)
    simplified = projected.to_crs("EPSG:4326")
    rows: list[list[Any]] = []
    group_code = {"major": 0, "mid": 1, "local": 2}
    for row in simplified.itertuples(index=False):
        for part in _line_parts(row.geometry):
            coordinates = [
                [round(float(lon), 5), round(float(lat), 5)] for lon, lat in part.coords
            ]
            if len(coordinates) >= 2:
                rows.append(
                    [group_code[str(row.road_group)], int(bool(row.transit_only_bool)), coordinates]
                )
    return rows


def _select_service_locations(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    selected: list[gpd.GeoDataFrame] = []
    for location_type in SERVICE_TYPE_ORDER:
        subset = frame[frame["service_location_type"].eq(location_type)]
        selected.append(
            _stable_take(
                subset,
                SERVICE_SAMPLE_CAPS[location_type],
                ["latent_service_location_id"],
            )
        )
    result = pd.concat(selected, ignore_index=True)
    return gpd.GeoDataFrame(result, geometry="geometry", crs=frame.crs)


def _compact_service_locations(frame: gpd.GeoDataFrame) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for row in frame.itertuples(index=False):
        rows.append(
            [
                round(float(row.location_lon), 5),
                round(float(row.location_lat), 5),
                SERVICE_TYPE_CODE[str(row.service_location_type)],
                int(row.residential_units),
                _round_or_none(row.road_access_distance_m, 1),
                round(float(row.road_anchor_lon), 5),
                round(float(row.road_anchor_lat), 5),
                0 if str(row.geometry_evidence_tier) == "G1_containment" else 1,
                int(bool(row.cle_candidate_eligible)),
            ]
        )
    return rows


def _compact_depots(frame: gpd.GeoDataFrame) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for row in frame.itertuples(index=False):
        rows.append(
            [
                round(float(row.longitude), 5),
                round(float(row.latitude), 5),
                str(row.facility_name or "Unnamed depot candidate"),
                str(row.evidence_tier),
                int(bool(row.depot_candidate_eligible)),
                _round_or_none(row.road_access_distance_m, 1),
            ]
        )
    return rows


def _compact_chargers(frame: gpd.GeoDataFrame) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for _, row in frame.iterrows():
        ports = int(row["l1_ports"]) + int(row["l2_ports"]) + int(row["dc_fast_ports"])
        rows.append(
            [
                round(float(row["Longitude"]), 5),
                round(float(row["Latitude"]), 5),
                str(row["Station Name"]),
                str(row["reference_charge_mode"]),
                int(bool(row["charger_candidate_eligible"])),
                _round_or_none(row["road_access_distance_m"], 1),
                ports,
            ]
        )
    return rows


def _render_fragment(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"""<div id="sd-cle-review-v1">
  <div class="sd-meta" aria-label="City Logistics Environment layer counts"></div>
  <div class="sd-controls" aria-label="Map layers">
    <label><input type="checkbox" data-layer="roads" checked> Roads</label>
    <label><input type="checkbox" data-layer="services" checked> Latent service locations</label>
    <label><input type="checkbox" data-layer="depots" checked> Depot candidates</label>
    <label><input type="checkbox" data-layer="chargers" checked> Charging stations</label>
    <button class="btn btn-secondary" type="button" data-action="reset">Reset view</button>
  </div>
  <div class="sd-map-wrap">
    <svg viewBox="0 0 900 760" role="img" aria-label="San Diego cle review map"></svg>
  </div>
  <div class="sd-legend" aria-label="Map legend"></div>
  <div class="sd-detail" aria-live="polite">Select a service location or facility to inspect it.</div>
</div>
<style>
#sd-cle-review-v1{{width:100%;color:var(--foreground)}}
#sd-cle-review-v1 .sd-meta{{margin:0 0 10px;color:var(--muted-foreground)}}
#sd-cle-review-v1 .sd-controls{{display:flex;align-items:center;gap:10px 16px;flex-wrap:wrap;margin:0 0 10px}}
#sd-cle-review-v1 .sd-controls label{{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}}
#sd-cle-review-v1 .sd-map-wrap{{width:100%;height:clamp(460px,82vw,720px);overflow:hidden;background:color-mix(in srgb,var(--muted) 25%,transparent);border:1px solid var(--border);border-radius:8px}}
#sd-cle-review-v1 svg{{display:block;width:100%;height:100%;touch-action:none;cursor:grab}}
#sd-cle-review-v1 svg:active{{cursor:grabbing}}
#sd-cle-review-v1 path,#sd-cle-review-v1 line{{vector-effect:non-scaling-stroke}}
#sd-cle-review-v1 .sd-legend{{display:flex;gap:8px 14px;flex-wrap:wrap;margin:10px 0;color:var(--muted-foreground)}}
#sd-cle-review-v1 .sd-legend span{{display:inline-flex;align-items:center;gap:5px}}
#sd-cle-review-v1 .swatch{{width:10px;height:10px;border-radius:50%;display:inline-block;background:var(--swatch)}}
#sd-cle-review-v1 .line-swatch{{width:18px;height:0;border-top:2px solid var(--swatch);display:inline-block}}
#sd-cle-review-v1 .sd-detail{{min-height:22px;color:var(--foreground)}}
@media(max-width:520px){{#sd-cle-review-v1 .sd-map-wrap{{height:500px}}}}
</style>
<script type="module">
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";
const root=document.getElementById("sd-cle-review-v1");
const data={data};
const svg=d3.select(root).select("svg");
const width=900,height=760;
const projection=d3.geoMercator().fitExtent([[24,24],[width-24,height-24]],{{type:"Feature",properties:{{}},geometry:data.boundary}});
const geoPath=d3.geoPath(projection);
const zoomLayer=svg.append("g");
const roadLayer=zoomLayer.append("g").attr("data-map-layer","roads");
const serviceLayer=zoomLayer.append("g").attr("data-map-layer","services");
const depotLayer=zoomLayer.append("g").attr("data-map-layer","depots");
const chargerLayer=zoomLayer.append("g").attr("data-map-layer","chargers");
const selectionLayer=zoomLayer.append("g").attr("data-map-layer","selection");
const css=name=>getComputedStyle(root).getPropertyValue(name).trim();
const colors={{house:css("--viz-series-1"),manufactured_home:css("--viz-series-2"),small_apt:css("--viz-series-3"),medium_apt:css("--viz-series-4"),large_apt:css("--viz-series-5"),transit:css("--viz-series-6"),inside:css("--muted-foreground"),boundary:css("--foreground"),charger:css("--viz-series-2"),depot:css("--viz-series-6")}};
const typeNames=["house","manufactured_home","small_apt","medium_apt","large_apt"];
const typeLabels=["House","Manufactured home","Small apt","Medium apt","Large apt"];
const roadLine=d3.line().x(d=>projection(d)[0]).y(d=>projection(d)[1]);
roadLayer.selectAll("path.road").data(data.roads).join("path")
  .attr("class","road").attr("d",d=>roadLine(d[2])).attr("fill","none")
  .attr("stroke",d=>d[1]?colors.transit:colors.inside)
  .attr("stroke-width",d=>d[0]===0?1.25:d[0]===1?.75:.45)
  .attr("stroke-opacity",d=>d[1]?.55:d[0]===2?.22:.42);
roadLayer.append("path").datum({{type:"Feature",geometry:data.boundary}}).attr("d",geoPath)
  .attr("fill","none").attr("stroke",colors.boundary).attr("stroke-width",1.5).attr("stroke-opacity",.85);
const detail=root.querySelector(".sd-detail");
const zoom=d3.zoom().scaleExtent([1,35]).on("zoom",event=>zoomLayer.attr("transform",event.transform));
svg.call(zoom);
function inspectService(event,d){{
  event.stopPropagation();
  selectionLayer.selectAll("*").remove();
  const p=projection([d[0],d[1]]),a=projection([d[5],d[6]]);
  selectionLayer.append("line").attr("x1",p[0]).attr("y1",p[1]).attr("x2",a[0]).attr("y2",a[1]).attr("stroke",colors.boundary).attr("stroke-width",2.2);
  selectionLayer.append("circle").attr("cx",a[0]).attr("cy",a[1]).attr("r",3.2).attr("fill",colors.boundary);
  detail.textContent=`${{typeLabels[d[2]]}} · ${{d[3]}} modeled unit${{d[3]===1?"":"s"}} · road access ${{d[4]}} m · ${{d[7]===0?"G1 containment":"G2 candidate"}} · ${{d[8]?"candidate eligible":"quarantined"}}`;
  if(d3.zoomTransform(svg.node()).k<7) svg.transition().duration(450).call(zoom.transform,d3.zoomIdentity.translate(width/2,height/2).scale(12).translate(-p[0],-p[1]));
}}
serviceLayer.selectAll("circle.service").data(data.services).join("circle")
  .attr("class","service").attr("cx",d=>projection([d[0],d[1]])[0]).attr("cy",d=>projection([d[0],d[1]])[1])
  .attr("r",d=>d[2]>=3?2.25:1.6).attr("fill",d=>colors[typeNames[d[2]]]).attr("fill-opacity",.72)
  .attr("stroke",d=>d[8]?"none":colors.boundary).attr("stroke-width",.45).style("cursor","pointer").on("click",inspectService);
function inspectFacility(kind,event,d){{
  event.stopPropagation(); selectionLayer.selectAll("*").remove();
  if(kind==="depot") detail.textContent=`Depot candidate · ${{d[2]}} · ${{d[3]}} · road access ${{d[5]}} m · ${{d[4]?"candidate eligible":"audit-only"}}`;
  else detail.textContent=`Charging station · ${{d[2]}} · ${{d[3]}} · ${{d[6]}} ports · road access ${{d[5]}} m · ${{d[4]?"candidate eligible":"excluded"}}`;
}}
depotLayer.selectAll("rect.depot").data(data.depots).join("rect").attr("class","depot")
  .attr("x",d=>projection([d[0],d[1]])[0]-3.6).attr("y",d=>projection([d[0],d[1]])[1]-3.6).attr("width",7.2).attr("height",7.2)
  .attr("transform",d=>`rotate(45 ${{projection([d[0],d[1]])[0]}} ${{projection([d[0],d[1]])[1]}})`)
  .attr("fill",colors.depot).attr("fill-opacity",d=>d[4]?.95:.22).attr("stroke",colors.boundary).attr("stroke-width",.6).style("cursor","pointer").on("click",(e,d)=>inspectFacility("depot",e,d));
chargerLayer.selectAll("circle.charger").data(data.chargers).join("circle").attr("class","charger")
  .attr("cx",d=>projection([d[0],d[1]])[0]).attr("cy",d=>projection([d[0],d[1]])[1]).attr("r",2.5)
  .attr("fill",d=>d[4]?colors.charger:"none").attr("fill-opacity",.7).attr("stroke",colors.charger).attr("stroke-width",.8).style("cursor","pointer").on("click",(e,d)=>inspectFacility("charger",e,d));
svg.on("click",()=>{{selectionLayer.selectAll("*").remove();detail.textContent="Select a service location or facility to inspect it.";}});
root.querySelectorAll('input[data-layer]').forEach(input=>input.addEventListener("change",()=>{{zoomLayer.select(`[data-map-layer="${{input.dataset.layer}}"]`).style("display",input.checked?null:"none");if(input.dataset.layer==="services"&&!input.checked)selectionLayer.selectAll("*").remove();}}));
root.querySelector('[data-action="reset"]').addEventListener("click",()=>svg.transition().duration(350).call(zoom.transform,d3.zoomIdentity));
const full=data.stats;
root.querySelector(".sd-meta").textContent=`${{full.service_total.toLocaleString()}} latent locations · ${{full.depot_eligible}}/${{full.depot_total}} depot candidates eligible · ${{full.charger_eligible}}/${{full.charger_total}} charging candidates eligible · ${{full.selected_buffer_km}} km routing buffer · ${{(full.road_coverage*100).toFixed(3)}}% city-road coverage`;
const legendItems=[
  ["line",colors.inside,"Inside-city road sample"],["line",colors.transit,"Transit-only road sample"],
  ...typeLabels.map((label,i)=>["dot",colors[typeNames[i]],label]),["dot",colors.depot,"Depot"],["dot",colors.charger,"Charger"]
];
root.querySelector(".sd-legend").innerHTML=legendItems.map(d=>`<span><i class="${{d[0]==="line"?"line-swatch":"swatch"}}" style="--swatch:${{d[1]}}"></i>${{d[2]}}</span>`).join("");
</script>
"""


def _render_static_preview(payload: dict[str, Any], output_path: Path) -> None:
    colors = {
        "inside": "#8b98a1",
        "transit": "#dc7f36",
        "house": "#2176ae",
        "manufactured_home": "#7b61a8",
        "small_apt": "#2a9d8f",
        "medium_apt": "#e9a23b",
        "large_apt": "#c94c4c",
        "depot": "#6f42c1",
        "charger": "#00a6a6",
    }
    figure, axis = plt.subplots(figsize=(10, 11), dpi=170)
    for group, transit_only, coordinates in payload["roads"]:
        xs = [coordinate[0] for coordinate in coordinates]
        ys = [coordinate[1] for coordinate in coordinates]
        width = {0: 0.55, 1: 0.34, 2: 0.2}[group]
        axis.plot(
            xs,
            ys,
            color=colors["transit" if transit_only else "inside"],
            linewidth=width,
            alpha=0.55 if transit_only else 0.35,
            zorder=1,
        )
    boundary = shapely_shape(payload["boundary"])
    boundary_frame = gpd.GeoSeries([boundary], crs="EPSG:4326")
    boundary_frame.boundary.plot(ax=axis, color="#27333a", linewidth=0.9, zorder=2)
    services = pd.DataFrame(
        payload["services"],
        columns=["lon", "lat", "type", "units", "access", "anchor_lon", "anchor_lat", "tier", "eligible"],
    )
    for type_code, type_name in enumerate(SERVICE_TYPE_ORDER):
        subset = services[services["type"].eq(type_code)]
        axis.scatter(
            subset["lon"],
            subset["lat"],
            s=2.2 if type_code < 3 else 4.0,
            color=colors[type_name],
            alpha=0.55,
            linewidths=0,
            label=f"{type_name.replace('_', ' ').title()} sample",
            zorder=3,
        )
    depots = pd.DataFrame(
        payload["depots"], columns=["lon", "lat", "name", "tier", "eligible", "access"]
    )
    depot_eligible = depots[depots["eligible"].eq(1)]
    axis.scatter(
        depot_eligible["lon"],
        depot_eligible["lat"],
        s=34,
        marker="D",
        color=colors["depot"],
        edgecolors="#27333a",
        linewidths=0.45,
        label="Eligible depot candidate",
        zorder=5,
    )
    chargers = pd.DataFrame(
        payload["chargers"],
        columns=["lon", "lat", "name", "mode", "eligible", "access", "ports"],
    )
    charger_eligible = chargers[chargers["eligible"].eq(1)]
    axis.scatter(
        charger_eligible["lon"],
        charger_eligible["lat"],
        s=7,
        facecolors="none",
        edgecolors=colors["charger"],
        linewidths=0.55,
        label="Eligible charging candidate",
        zorder=4,
    )
    min_x, min_y, max_x, max_y = boundary.bounds
    pad_x = (max_x - min_x) * 0.07
    pad_y = (max_y - min_y) * 0.04
    axis.set_xlim(min_x - pad_x, max_x + pad_x)
    axis.set_ylim(min_y - pad_y, max_y + pad_y)
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.set_title(
        "San Diego cle review template v1\n"
        "Display samples only; full graph and Parquet layers remain authoritative",
        loc="left",
        fontsize=13,
        fontweight="normal",
    )
    axis.legend(
        loc="lower left",
        fontsize=7.5,
        frameon=True,
        ncol=2,
        markerscale=1.8,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def build_review_map(
    cle_dir: Path,
    road_city_dir: Path,
    output_dir: Path,
    inline_html: Path,
) -> dict[str, Any]:
    cle_manifest = json.loads((cle_dir / "manifest.json").read_text())
    road_manifest = json.loads((road_city_dir / "manifest.json").read_text())
    graph_path = road_city_dir / "graph_operational.graphml"
    graph = ox.load_graphml(graph_path)
    edges = ox.convert.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True)
    selected_roads = _select_roads(edges)

    services = gpd.read_parquet(cle_dir / "service_locations/latent_locations.parquet")
    depots = gpd.read_parquet(cle_dir / "infrastructure/depots.parquet")
    chargers = gpd.read_parquet(cle_dir / "infrastructure/chargers.parquet")
    selected_services = _select_service_locations(services)
    boundary = gpd.read_file(cle_dir / "boundary/admin_boundary.geojson").to_crs(
        "EPSG:4326"
    )
    boundary_projected = boundary.to_crs(boundary.estimate_utm_crs())
    boundary_projected["geometry"] = boundary_projected.geometry.simplify(
        25.0, preserve_topology=True
    )
    boundary_geometry = mapping(boundary_projected.to_crs("EPSG:4326").geometry.iloc[0])

    operational = road_manifest["operational_connectivity"]
    payload = {
        "schema": "evrptw_cle_review_payload_v1",
        "boundary": boundary_geometry,
        "roads": _compact_roads(selected_roads),
        "services": _compact_service_locations(selected_services),
        "depots": _compact_depots(depots),
        "chargers": _compact_chargers(chargers),
        "stats": {
            "service_total": len(services),
            "service_sample": len(selected_services),
            "service_type_totals": {
                str(key): int(value)
                for key, value in services["service_location_type"].value_counts().items()
            },
            "depot_total": len(depots),
            "depot_eligible": int(depots["depot_candidate_eligible"].fillna(False).sum()),
            "charger_total": len(chargers),
            "charger_eligible": int(
                chargers["charger_candidate_eligible"].fillna(False).sum()
            ),
            "operational_nodes": graph.number_of_nodes(),
            "operational_directed_edges": graph.number_of_edges(),
            "road_display_features": len(selected_roads),
            "selected_buffer_km": operational["selected_buffer_km"],
            "node_coverage": operational["city_node_coverage"],
            "road_coverage": operational["city_physical_road_length_coverage"],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "review_payload.json"
    payload_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    preview_path = output_dir / "review_map.png"
    _render_static_preview(payload, preview_path)

    template_manifest = {
        "schema": "evrptw_city_cle_review_template_v1",
        "city_slug": cle_manifest["city_slug"],
        "city_label": cle_manifest["city_label"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "template_role": (
            "reviewable Stage-1 cle template; not an instance and not release eligible"
        ),
        "source_cle": {
            "path": str(cle_dir.resolve()),
            "manifest_sha256": sha256_file(cle_dir / "manifest.json"),
            "status": cle_manifest["status"],
            "release_blockers": cle_manifest["release_blockers"],
        },
        "source_operational_graph": {
            "path": str(graph_path.resolve()),
            "sha256": sha256_file(graph_path),
            "node_count": graph.number_of_nodes(),
            "directed_edge_count": graph.number_of_edges(),
        },
        "layer_counts": cle_manifest["layer_counts"],
        "visual_review_contract": {
            "roads": "deterministic stratified physical-road sample; full graph remains authoritative",
            "service_locations": (
                "deterministic type-stratified point sample; full parquet remains authoritative"
            ),
            "depots": "all audit rows shown; eligibility encoded separately",
            "chargers": "all AFDC rows shown; candidate eligibility encoded separately",
            "customer_connector": (
                "clicking a sampled service location displays its exact stored access connector"
            ),
        },
        "review_payload": {
            "path": str(payload_path.resolve()),
            "sha256": sha256_file(payload_path),
            "bytes": payload_path.stat().st_size,
        },
        "static_preview": {
            "path": str(preview_path.resolve()),
            "sha256": sha256_file(preview_path),
        },
    }
    template_manifest_path = output_dir / "template_manifest.json"
    write_json(template_manifest_path, template_manifest)
    inline_html.parent.mkdir(parents=True, exist_ok=True)
    inline_html.write_text(_render_fragment(payload), encoding="utf-8")
    return {
        "template_manifest": str(template_manifest_path),
        "review_payload": str(payload_path),
        "static_preview": str(preview_path),
        "inline_html": str(inline_html),
        "inline_html_bytes": inline_html.stat().st_size,
        "stats": payload["stats"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cle-dir", type=Path, required=True)
    parser.add_argument("--road-city-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inline-html", type=Path, required=True)
    args = parser.parse_args()
    result = build_review_map(
        args.cle_dir,
        args.road_city_dir,
        args.output_dir,
        args.inline_html,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
