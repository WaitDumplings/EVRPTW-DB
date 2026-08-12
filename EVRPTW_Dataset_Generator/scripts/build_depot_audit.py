#!/usr/bin/env python3
"""Run and aggregate the OSM depot-candidate audit for a city preset."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from evrptw_cle.util import sha256_file


def _run_city(
    script: Path,
    repo_root: Path,
    item: dict[str, Any],
    boundary_root: Path,
    city_root: Path,
    output_dir: Path,
    snapshot_date: str,
    max_road_snap_m: float,
    min_warehouse_area_m2: float,
    max_tier_b_on_map: int,
) -> None:
    command = [
        sys.executable,
        str(script),
        "--repo-root",
        str(repo_root),
        "--city-slug",
        item["slug"],
        "--city-label",
        item.get("display_name", item["query"].split(",")[0]),
        "--boundary-root",
        str(boundary_root),
        "--city-root",
        str(city_root),
        "--output-dir",
        str(output_dir),
        "--snapshot-date",
        snapshot_date,
        "--max-road-snap-m",
        str(max_road_snap_m),
        "--min-warehouse-area-m2",
        str(min_warehouse_area_m2),
        "--max-tier-b-on-map",
        str(max_tier_b_on_map),
    ]
    subprocess.run(command, check=True)


def _summary_row(
    item: dict[str, Any], summary: dict[str, Any], candidates: pd.DataFrame, land_km2: float
) -> dict[str, Any]:
    retained = int(summary["retained_candidate_count"])
    tier_a = int(summary["tier_a_count"])
    operational = int(summary["operational_eligible_count"])
    tier_a_operational = int(summary["tier_a_operational_eligible_count"])
    area_known = candidates.loc[candidates["facility_area_known"].astype(bool)]
    tier_a_frame = candidates.loc[candidates["evidence_tier"] == "A_osm_explicit"]
    return {
        "city_slug": item["slug"],
        "city_label": item.get("display_name", item["query"].split(",")[0]),
        "census_place_geoid": item["census_place_geoid"],
        "land_area_km2": round(land_km2, 3),
        "raw_tier_a_count": int(summary["raw_tier_counts"].get("A_osm_explicit", 0)),
        "raw_tier_b_count": int(summary["raw_tier_counts"].get("B_warehouse_proxy", 0)),
        "raw_tier_c_excluded_count": int(
            summary["raw_tier_counts"].get("C_industrial_proxy", 0)
        ),
        "retained_candidate_count": retained,
        "tier_a_count": tier_a,
        "tier_b_count": int(summary["tier_b_count"]),
        "dispatch_function_signal_count": int(summary["dispatch_function_signal_count"]),
        "carrier_facility_signal_count": int(summary["carrier_facility_signal_count"]),
        "cle_candidate_eligible_count": int(
            summary["cle_candidate_eligible_count"]
        ),
        "operational_eligible_count": operational,
        "tier_a_operational_eligible_count": tier_a_operational,
        "strict_candidate_eligible_count": int(
            summary["strict_candidate_eligible_count"]
        ),
        "optional_candidate_eligible_count": int(
            summary["optional_candidate_eligible_count"]
        ),
        "retained_operational_eligible_rate": round(operational / retained, 6)
        if retained
        else 0.0,
        "tier_a_operational_eligible_rate": round(tier_a_operational / tier_a, 6)
        if tier_a
        else 0.0,
        "retained_candidates_per_100_km2": round(retained / land_km2 * 100, 3),
        "facility_area_known_count": len(area_known),
        "facility_area_median_m2": round(float(area_known["facility_area_m2"].median()), 1)
        if len(area_known)
        else None,
        "tier_a_named_carrier_count": int(tier_a_frame["carrier"].ne("").sum()),
        "depot_release_eligible_count": int(summary["depot_release_eligible_count"]),
        "largest_directed_scc_node_share": round(
            float(summary["graph_gate"]["largest_strong_component_node_share"]), 6
        ),
        "pbf_replication_timestamp_utc": summary["source"][
            "pbf_replication_timestamp_utc"
        ],
    }


def _overview_html(city_stats: pd.DataFrame, output_dir: Path) -> str:
    rows = []
    for row in city_stats.to_dict(orient="records"):
        slug = row["city_slug"]
        map_name = f"{slug}_depot_candidate_audit.html"
        rows.append(
            "<tr>"
            f"<td><a href=\"{map_name}\">{row['city_label']}</a></td>"
            f"<td>{row['tier_a_count']:,}</td>"
            f"<td>{row['dispatch_function_signal_count']:,}</td>"
            f"<td>{row['tier_b_count']:,}</td>"
            f"<td>{row['strict_candidate_eligible_count']:,}</td>"
            f"<td>{row['optional_candidate_eligible_count']:,}</td>"
            f"<td>{row['cle_candidate_eligible_count']:,}</td>"
            f"<td>{row['retained_operational_eligible_rate']:.1%}</td>"
            f"<td>{row['depot_release_eligible_count']:,}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CLE cohort depot candidate audit</title>
<style>
body{{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#18212b;background:#f7f8fa}}
.card{{max-width:1120px;margin:auto;background:white;border:1px solid #d9dee5;border-radius:12px;padding:22px}}
h1{{margin:0 0 8px;font-size:24px}} p{{color:#4d5968}} table{{width:100%;border-collapse:collapse;margin-top:18px}}
th,td{{padding:9px 10px;border-bottom:1px solid #e7eaf0;text-align:right}} th{{background:#f1f4f7}}
th:first-child,td:first-child{{text-align:left}} a{{color:#075f85}} .warn{{border-left:4px solid #d97706;padding-left:10px}}
</style></head><body><main class="card"><h1>CLE cohort OSM depot-candidate audit</h1>
<p class="warn">These are reproducible OSM candidates. Tier A is stronger mapped evidence, not proof of current last-mile operations. Release-eligible depot count remains zero until external or manual verification.</p>
<table><thead><tr><th>City / map</th><th>Tier A</th><th>Dispatch signal</th><th>Tier B proxy</th><th>Strict eligible</th><th>Optional eligible</th><th>Total cle</th><th>Retained road eligible rate</th><th>Externally verified</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p>Directory: {output_dir.name}. Click a city name to inspect its candidates and directed-road anchors.</p>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--boundary-root", type=Path, default=Path("boundaries/us-11city-2025")
    )
    parser.add_argument("--city-root", type=Path, default=Path("data/cities"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit aggregate output directory; defaults to analysis/depot_preview/<date>",
    )
    parser.add_argument("--snapshot-date", default=datetime.now(UTC).date().isoformat())
    parser.add_argument("--max-road-snap-m", type=float, default=250.0)
    parser.add_argument("--min-warehouse-area-m2", type=float, default=1_000.0)
    parser.add_argument("--max-tier-b-on-map", type=int, default=5_000)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    preset_path = args.preset if args.preset.is_absolute() else repo_root / args.preset
    boundary_root = (
        args.boundary_root
        if args.boundary_root.is_absolute()
        else repo_root / args.boundary_root
    )
    city_root = args.city_root if args.city_root.is_absolute() else repo_root / args.city_root
    preset = json.loads(preset_path.read_text(encoding="utf-8"))
    output_dir = (
        args.output_dir
        if args.output_dir and args.output_dir.is_absolute()
        else repo_root / args.output_dir
        if args.output_dir
        else repo_root / "analysis/depot_preview" / args.snapshot_date
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    city_script = Path(__file__).resolve().parent / "build_osm_depot_preview.py"

    for item in preset["cities"]:
        summary_path = output_dir / f"{item['slug']}_depot_summary.json"
        candidate_path = output_dir / f"{item['slug']}_depot_candidates.csv"
        graph_path = city_root / item["slug"] / "graph_operational.graphml"
        if args.skip_existing and summary_path.exists() and candidate_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            gate = summary.get("graph_gate", {})
            graph_matches = (
                gate.get("graph_path") == str(graph_path.resolve())
                and gate.get("graph_sha256") == sha256_file(graph_path)
            )
            if graph_matches:
                print(f"SKIP {item['slug']} graph-aligned", flush=True)
                continue
            print(f"REBUILD {item['slug']} graph changed", flush=True)
        print(f"DEPOT AUDIT {item['slug']}", flush=True)
        _run_city(
            city_script,
            repo_root,
            item,
            boundary_root,
            city_root,
            output_dir,
            args.snapshot_date,
            args.max_road_snap_m,
            args.min_warehouse_area_m2,
            args.max_tier_b_on_map,
        )

    city_rows = []
    candidate_frames = []
    for item in preset["cities"]:
        slug = item["slug"]
        summary = json.loads(
            (output_dir / f"{slug}_depot_summary.json").read_text(encoding="utf-8")
        )
        candidates = pd.read_csv(output_dir / f"{slug}_depot_candidates.csv")
        candidate_frames.append(candidates)
        boundary_metadata = json.loads(
            (boundary_root / slug / "metadata.json").read_text(encoding="utf-8")
        )
        city_rows.append(
            _summary_row(
                item,
                summary,
                candidates,
                float(boundary_metadata["land_mask_qa"]["derived_land_area_km2"]),
            )
        )
    city_stats = pd.DataFrame(city_rows)
    all_candidates = pd.concat(candidate_frames, ignore_index=True)
    cle_candidates = all_candidates.loc[
        all_candidates["cle_candidate_eligible"].astype(bool)
    ].copy()
    carrier_stats = (
        all_candidates.loc[all_candidates["carrier"].fillna("").ne("")]
        .groupby(["city_slug", "carrier"], as_index=False)
        .agg(
            candidate_count=("candidate_id", "size"),
            operational_eligible_count=("operational_eligible", "sum"),
        )
        .sort_values(["city_slug", "candidate_count", "carrier"], ascending=[True, False, True])
    )
    city_stats.to_csv(output_dir / "cle_cohort_depot_city_statistics.csv", index=False)
    carrier_stats.to_csv(output_dir / "cle_cohort_depot_carrier_statistics.csv", index=False)
    all_candidates.to_csv(output_dir / "cle_cohort_depot_candidates_all.csv", index=False)
    cle_columns = [
        "candidate_id",
        "city_slug",
        "facility_name",
        "carrier",
        "evidence_tier",
        "evidence_reason",
        "function_signal",
        "strict_candidate_class",
        "depot_evidence_class",
        "verification_status",
        "last_mile_function_verified",
        "benchmark_depot_class",
        "strict_candidate_eligible",
        "optional_candidate_eligible",
        "source_osm_id",
        "source_osm_url",
        "facility_geometry_type",
        "facility_area_m2",
        "facility_area_known",
        "longitude",
        "latitude",
        "address",
        "road_anchor_node",
        "road_anchor_longitude",
        "road_anchor_latitude",
        "road_snap_distance_m",
        "road_anchor_strategy",
        "operational_eligible",
    ]
    cle_candidates[cle_columns].to_csv(
        output_dir / "cle_cohort_depot_candidates.csv", index=False
    )
    totals = {
        column: int(city_stats[column].sum())
        for column in (
            "raw_tier_a_count",
            "raw_tier_b_count",
            "raw_tier_c_excluded_count",
            "retained_candidate_count",
            "tier_a_count",
            "tier_b_count",
            "dispatch_function_signal_count",
            "carrier_facility_signal_count",
            "cle_candidate_eligible_count",
            "operational_eligible_count",
            "tier_a_operational_eligible_count",
            "strict_candidate_eligible_count",
            "optional_candidate_eligible_count",
            "depot_release_eligible_count",
        )
    }
    aggregate = {
        "schema": "evrptw_cle_cohort_depot_candidate_audit_v3",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "preset_id": preset["preset_id"],
        "snapshot_date": args.snapshot_date,
        "city_count": len(city_stats),
        "classification_policy": {
            "min_warehouse_area_m2": args.min_warehouse_area_m2,
            "strict_pool": "road-eligible A_osm_explicit",
            "optional_pool": "road-eligible B_warehouse_proxy",
        },
        "totals": totals,
        "release_semantics": (
            "All rows are candidate evidence. depot_release_eligible remains false until "
            "current last-mile operating function and access are verified."
        ),
        "outputs": {
            "city_statistics": "cle_cohort_depot_city_statistics.csv",
            "carrier_statistics": "cle_cohort_depot_carrier_statistics.csv",
            "candidate_table": "cle_cohort_depot_candidates_all.csv",
            "cle_candidate_table": "cle_cohort_depot_candidates.csv",
            "overview": "cle_cohort_depot_candidate_overview.html",
        },
    }
    (output_dir / "cle_cohort_depot_summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "cle_cohort_depot_candidate_overview.html").write_text(
        _overview_html(city_stats, output_dir), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
