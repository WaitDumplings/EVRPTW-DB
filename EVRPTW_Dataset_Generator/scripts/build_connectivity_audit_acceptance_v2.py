#!/usr/bin/env python3
"""Build the bounded C1b/R2-v2 replayable connectivity acceptance package."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import folium
import geopandas as gpd
import osmnx as ox
import pandas as pd
from folium.plugins import PolyLineTextPath
from shapely.geometry import LineString, mapping

from evrptw_cle.customer_access import build_eligible_physical_edges
from evrptw_cle.protected_connectivity import (
    build_directed_component_index,
    projection_reference_access,
)
from evrptw_stage2.connectivity_acceptance import (
    ACCEPTANCE_SCHEMA,
    concentration_summary,
    directed_ref_keys,
    h64_rank,
    json_records,
    manual_review_gate,
    primary_pf2_support,
    select_h64_samples,
)
from evrptw_stage2.profile import load_reference_profile
from evrptw_stage2.provenance import resolve_git_provenance
from evrptw_stage2.reader import load_portable_cle
from evrptw_stage2.road_state import build_family_road_state
from evrptw_stage2.routing import PhysicalRoadNetwork
from evrptw_stage2.selection import (
    _charger_roster_terminal_index,
    _connectivity_quarantine,
    _star_terminal_index,
)


H64_NAMESPACE = "EVRPTW-DB:C1b:R2-v2:H64:v1"
KNOWN_REASONS = {
    "stage1_no_reference_scc_inbound_access",
    "stage1_no_reference_scc_outbound_access",
    "node_unreachable_from_depot",
    "node_cannot_return_to_depot",
    "turn_unreachable_from_depot",
    "turn_cannot_return_to_depot",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _reason_list(value: object) -> list[str]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return sorted(map(str, parsed if isinstance(parsed, list) else [parsed]))
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return sorted(map(str, value))
    raise ValueError(f"Malformed reason code collection: {value!r}")


def _audit_input(raw: pd.DataFrame, terminal_kind: str) -> pd.DataFrame:
    if terminal_kind == "customer":
        mask = (
            raw["geometry_core_eligible"].fillna(False).astype(bool)
            & raw["physical_edge_id"].notna()
            & pd.to_numeric(raw["road_access_distance_m"], errors="coerce").notna()
        )
    else:
        mask = (
            raw["restricted_public"].ne(True)
            & raw["reference_charge_mode"].ne("unsupported_or_unresolved")
            & raw["coordinate_candidate_eligible"].fillna(False).astype(bool)
        )
    return raw.loc[mask].copy()


def _raw_tables(cle: Any) -> dict[str, pd.DataFrame]:
    common = [
        "physical_edge_id",
        "directed_projection_offsets",
        "directed_edge_ref_count",
        "protected_inbound_access_eligible",
        "protected_outbound_access_eligible",
        "anchor_scc_id",
        "highway",
        "road_projection_fraction_from_physical_start",
        "road_anchor_lat",
        "road_anchor_lon",
    ]
    return {
        "customer": pd.read_parquet(
            cle.service_locations_path,
            columns=[
                "latent_service_location_id",
                "geometry_core_eligible",
                "road_access_distance_m",
                "access_layer",
                "location_lat",
                "location_lon",
                *common,
            ],
        ),
        "charging_station": pd.read_parquet(
            cle.chargers_path,
            columns=[
                "charger_id",
                "restricted_public",
                "reference_charge_mode",
                "coordinate_candidate_eligible",
                "resolved_latitude",
                "resolved_longitude",
                *common,
            ],
        ),
    }


def _row_osmids(refs: object) -> list[str]:
    return sorted(
        {
            str(osmid)
            for ref in json_records(refs)
            for osmid in ref.get("osmid", [])
        }
    )


def _stage1_certificates(
    cle: Any,
    ledger: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame], gpd.GeoDataFrame]:
    catalog = build_eligible_physical_edges(cle.graph_path, "EPSG:4326")
    graph = ox.load_graphml(cle.graph_path)
    component_index = build_directed_component_index(graph)
    catalog_by_id = catalog.set_index("physical_edge_id", drop=False)
    graph_keys = {
        (str(u), str(v), str(key)) for u, v, key in graph.edges(keys=True)
    }
    raw_tables = _raw_tables(cle)
    identifiers = {
        "customer": "latent_service_location_id",
        "charging_station": "charger_id",
    }
    records: list[dict[str, Any]] = []
    for terminal_kind, id_column in identifiers.items():
        raw = raw_tables[terminal_kind].copy()
        raw[id_column] = raw[id_column].astype(str)
        raw_by_id = raw.drop_duplicates(id_column).set_index(id_column, drop=False)
        subset = ledger.loc[
            ledger["terminal_kind"].astype(str).eq(terminal_kind)
            & ledger["quarantine_stages"].map(
                lambda value: "stage1_directional" in _reason_list(value)
            )
        ]
        for item in subset.to_dict(orient="records"):
            source_id = str(item["source_id"])
            row = raw_by_id.loc[source_id]
            physical_edge_id = str(row["physical_edge_id"])
            catalog_present = physical_edge_id in catalog_by_id.index
            catalog_row = catalog_by_id.loc[physical_edge_id] if catalog_present else None
            stored_keys = directed_ref_keys(row["directed_projection_offsets"])
            catalog_keys = (
                directed_ref_keys(catalog_row["directed_edge_refs"])
                if catalog_present
                else ()
            )
            all_stored_refs_checked = bool(stored_keys) and all(
                key in graph_keys for key in stored_keys
            )
            no_omitted_legal_ref = stored_keys == catalog_keys
            inbound, outbound = projection_reference_access(
                str(row["directed_projection_offsets"]), component_index
            )
            inbound2, outbound2 = projection_reference_access(
                str(row["directed_projection_offsets"]), component_index
            )
            expected_reasons = []
            if not inbound:
                expected_reasons.append("stage1_no_reference_scc_inbound_access")
            if not outbound:
                expected_reasons.append("stage1_no_reference_scc_outbound_access")
            second_reasons = []
            if not inbound2:
                second_reasons.append("stage1_no_reference_scc_inbound_access")
            if not outbound2:
                second_reasons.append("stage1_no_reference_scc_outbound_access")
            replay_signature_1 = json.dumps(
                {
                    "inbound": inbound,
                    "outbound": outbound,
                    "reason_codes": expected_reasons,
                },
                sort_keys=True,
            )
            replay_signature_2 = json.dumps(
                {
                    "inbound": inbound2,
                    "outbound": outbound2,
                    "reason_codes": second_reasons,
                },
                sort_keys=True,
            )
            ledger_reasons = _reason_list(item["reason_codes"])
            count_matches = (
                len(stored_keys)
                == int(row["directed_edge_ref_count"])
                == len(catalog_keys)
            )
            reason_known = bool(expected_reasons) and set(expected_reasons) <= KNOWN_REASONS
            passed = all(
                [
                    str(row.get("access_layer", "operational_public"))
                    == "operational_public",
                    catalog_present,
                    all_stored_refs_checked,
                    no_omitted_legal_ref,
                    count_matches,
                    inbound == bool(row["protected_inbound_access_eligible"]),
                    outbound == bool(row["protected_outbound_access_eligible"]),
                    inbound == bool(item["stage1_inbound_access_eligible"]),
                    outbound == bool(item["stage1_outbound_access_eligible"]),
                    expected_reasons == ledger_reasons,
                    reason_known,
                    replay_signature_1 == replay_signature_2,
                ]
            )
            records.append(
                {
                    "city_slug": cle.city_slug,
                    "terminal_kind": terminal_kind,
                    "source_id": source_id,
                    "audit_stage": "stage1_directional",
                    "depot_id": None,
                    "reason_codes": expected_reasons,
                    "certificate_passed": passed,
                    "all_stored_directed_access_refs_checked": all_stored_refs_checked,
                    "no_omitted_legal_inbound_or_outbound_ref": no_omitted_legal_ref,
                    "stored_directed_ref_count": len(stored_keys),
                    "catalog_directed_ref_count": len(catalog_keys),
                    "directed_ref_count_matches": count_matches,
                    "recomputed_inbound_access_eligible": inbound,
                    "recomputed_outbound_access_eligible": outbound,
                    "node_outbound_reachable": None,
                    "node_return_reachable": None,
                    "turn_outbound_reachable": None,
                    "turn_return_reachable": None,
                    "turn_only_contract_passed": None,
                    "replay_round_1_signature": replay_signature_1,
                    "replay_round_2_signature": replay_signature_2,
                    "replay_round_2_equal": replay_signature_1 == replay_signature_2,
                    "physical_edge_id": physical_edge_id,
                    "osm_way_ids": _row_osmids(
                        catalog_row["directed_edge_refs"] if catalog_present else "[]"
                    ),
                    "anchor_scc_id": str(row.get("anchor_scc_id", "")),
                    "census_block_group_geoid": item.get("census_block_group_geoid"),
                    "community_id": item.get("community_id"),
                    "split_pool": item.get("split_pool"),
                    "highway": str(row.get("highway", "")),
                    "road_projection_fraction_from_physical_start": row.get(
                        "road_projection_fraction_from_physical_start"
                    ),
                }
            )
    return records, raw_tables, catalog


def _stage2_certificates(
    cle: Any,
    ledger: pd.DataFrame,
    replay_basis: dict[str, Any],
    profile: dict[str, Any],
    catalog: gpd.GeoDataFrame,
) -> list[dict[str, Any]]:
    speeds = pd.read_parquet(cle.speeds_path)
    road_state, _ = build_family_road_state(
        speeds,
        day_type=str(replay_basis["day_type"]),
        road_state_seed=int(replay_basis["road_state_seed"]),
        profile=profile,
    )
    network = PhysicalRoadNetwork.from_files(cle.graph_path, road_state, profile)
    customers = cle.read_service_locations().reset_index(drop=True)
    chargers = cle.read_chargers().reset_index(drop=True)
    depots = cle.read_depots().reset_index(drop=True)
    graph_keys = {
        (str(u), str(v), str(key))
        for u, v, key in zip(
            network.edges["edge_u"], network.edges["edge_v"], network.edges["edge_key"], strict=True
        )
    }
    catalog_by_id = catalog.set_index("physical_edge_id", drop=False)
    records: list[dict[str, Any]] = []
    for (depot_id, terminal_kind), expected in ledger.groupby(
        ["depot_id", "terminal_kind"], sort=True
    ):
        depot_rows = depots.loc[depots["candidate_id"].astype(str).eq(str(depot_id))]
        if len(depot_rows) != 1:
            raise ValueError(f"Cannot resolve unique replay depot {cle.city_slug}/{depot_id}")
        depot = depot_rows.iloc[0]
        if str(terminal_kind) == "customer":
            id_column = "latent_service_location_id"
            source = customers
        else:
            id_column = "charger_id"
            source = chargers
        expected_ids = set(expected["source_id"].astype(str))
        roster = source.loc[source[id_column].astype(str).isin(expected_ids)].copy()
        if len(roster) != len(expected_ids):
            raise ValueError(
                f"R2-v2 replay roster mismatch in {cle.city_slug}/{depot_id}/{terminal_kind}"
            )

        def replay() -> tuple[dict[str, dict[str, Any]], Any]:
            if str(terminal_kind) == "customer":
                terminal_index = _star_terminal_index(depot, roster)
            else:
                terminal_index = _charger_roster_terminal_index(
                    depot, customers.iloc[:0], roster
                )
            star = network.route_depot_star(terminal_index)
            _, bad = _connectivity_quarantine(
                roster,
                star,
                id_column=id_column,
                terminal_kind=str(terminal_kind),
            )
            return {str(item["source_id"]): item for item in bad}, star

        replay1, star1 = replay()
        replay2, star2 = replay()
        expected_by_id = {
            str(item["source_id"]): item for item in expected.to_dict(orient="records")
        }
        roster_by_id = roster.set_index(roster[id_column].astype(str), drop=False)
        for position, source_id in enumerate(roster[id_column].astype(str), start=1):
            item = expected_by_id[source_id]
            raw = roster_by_id.loc[source_id]
            expected_reasons = _reason_list(item["reason_codes"])
            observed_reasons = _reason_list(replay1.get(source_id, {}).get("reason_codes", []))
            second_reasons = _reason_list(replay2.get(source_id, {}).get("reason_codes", []))
            stored_keys = directed_ref_keys(raw["directed_projection_offsets"])
            physical_edge_id = str(raw["physical_edge_id"])
            catalog_keys = (
                directed_ref_keys(catalog_by_id.loc[physical_edge_id]["directed_edge_refs"])
                if physical_edge_id in catalog_by_id.index
                else ()
            )
            ref_count = len(stored_keys)
            all_access_states_checked = bool(stored_keys) and all(
                key in graph_keys for key in stored_keys
            )
            no_omitted_legal_ref = stored_keys == catalog_keys
            count_matches = (
                ref_count == int(raw["directed_edge_ref_count"]) == len(catalog_keys)
            )
            node_out = bool(star1.node_outbound_reachable[position])
            node_back = bool(star1.node_return_reachable[position])
            turn_out = bool(star1.turn_outbound_reachable[position])
            turn_back = bool(star1.turn_return_reachable[position])
            node_out2 = bool(star2.node_outbound_reachable[position])
            node_back2 = bool(star2.node_return_reachable[position])
            turn_out2 = bool(star2.turn_outbound_reachable[position])
            turn_back2 = bool(star2.turn_return_reachable[position])
            replay_signature_1 = json.dumps(
                {
                    "node_outbound": node_out,
                    "node_return": node_back,
                    "reason_codes": observed_reasons,
                    "turn_outbound": turn_out,
                    "turn_return": turn_back,
                },
                sort_keys=True,
            )
            replay_signature_2 = json.dumps(
                {
                    "node_outbound": node_out2,
                    "node_return": node_back2,
                    "reason_codes": second_reasons,
                    "turn_outbound": turn_out2,
                    "turn_return": turn_back2,
                },
                sort_keys=True,
            )
            turn_only = not any(reason.startswith("node_") for reason in expected_reasons)
            turn_only_contract = (
                node_out
                and node_back
                and (not turn_out or not turn_back)
                if turn_only
                else True
            )
            reason_known = bool(expected_reasons) and set(expected_reasons) <= KNOWN_REASONS
            replay_equal = replay_signature_1 == replay_signature_2
            passed = all(
                [
                    observed_reasons == expected_reasons,
                    replay_equal,
                    all_access_states_checked,
                    reason_known,
                    turn_only_contract,
                    no_omitted_legal_ref,
                    count_matches,
                ]
            )
            records.append(
                {
                    "city_slug": cle.city_slug,
                    "terminal_kind": str(terminal_kind),
                    "source_id": source_id,
                    "audit_stage": "stage2_exact_preflight",
                    "depot_id": str(depot_id),
                    "reason_codes": observed_reasons,
                    "certificate_passed": passed,
                    "all_stored_directed_access_refs_checked": all_access_states_checked,
                    "no_omitted_legal_inbound_or_outbound_ref": no_omitted_legal_ref,
                    "stored_directed_ref_count": ref_count,
                    "catalog_directed_ref_count": len(catalog_keys),
                    "directed_ref_count_matches": count_matches,
                    "recomputed_inbound_access_eligible": None,
                    "recomputed_outbound_access_eligible": None,
                    "node_outbound_reachable": node_out,
                    "node_return_reachable": node_back,
                    "turn_outbound_reachable": turn_out,
                    "turn_return_reachable": turn_back,
                    "turn_only_contract_passed": turn_only_contract,
                    "replay_round_1_signature": replay_signature_1,
                    "replay_round_2_signature": replay_signature_2,
                    "replay_round_2_equal": replay_equal,
                    "physical_edge_id": str(raw["physical_edge_id"]),
                    "osm_way_ids": _row_osmids(raw["directed_projection_offsets"]),
                    "anchor_scc_id": str(raw.get("anchor_scc_id", "")),
                    "census_block_group_geoid": item.get("census_block_group_geoid"),
                    "community_id": item.get("community_id"),
                    "split_pool": item.get("split_pool"),
                    "highway": str(raw.get("highway", "")),
                    "road_projection_fraction_from_physical_start": raw.get(
                        "road_projection_fraction_from_physical_start"
                    ),
                }
            )
    return records


def _concentration(
    city: str,
    raw_tables: dict[str, pd.DataFrame],
    aggregate_ledger: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    id_columns = {
        "customer": "latent_service_location_id",
        "charging_station": "charger_id",
    }
    for terminal_kind, id_column in id_columns.items():
        inputs = _audit_input(raw_tables[terminal_kind], terminal_kind)
        quarantine = aggregate_ledger.loc[
            aggregate_ledger["terminal_kind"].astype(str).eq(terminal_kind)
        ].copy()
        source_ids = set(quarantine["source_id"].astype(str))
        bad = inputs.loc[inputs[id_column].astype(str).isin(source_ids)].copy()
        quarantine["_stages"] = quarantine["quarantine_stages"].map(_reason_list)
        stage1_ids = set(
            quarantine.loc[
                quarantine["_stages"].map(lambda value: "stage1_directional" in value),
                "source_id",
            ].astype(str)
        )
        stage2_ids = set(
            quarantine.loc[
                quarantine["_stages"].map(
                    lambda value: "stage2_exact_preflight" in value
                ),
                "source_id",
            ].astype(str)
        )
        stage_counts = {
            "stage1_unique_terminal_count": len(stage1_ids),
            "stage1_rate": len(stage1_ids) / len(inputs) if len(inputs) else 0.0,
            "stage2_unique_terminal_count": len(stage2_ids),
            "stage2_rate": len(stage2_ids) / len(inputs) if len(inputs) else 0.0,
            "union_unique_terminal_count": len(source_ids),
            "union_rate": len(source_ids) / len(inputs) if len(inputs) else 0.0,
        }
        reasons_by_id = {
            str(item["source_id"]): _reason_list(item["reason_codes"])
            for item in quarantine.to_dict(orient="records")
        }

        def decorate(
            summary: dict[str, Any],
            reason_bad: pd.DataFrame,
            reason: str,
        ) -> dict[str, Any]:
            reason_ids = set(reason_bad[id_column].astype(str))
            summary.update(
                {
                    "city_slug": city,
                    "terminal_kind": terminal_kind,
                    "reason_code": reason,
                    **stage_counts,
                    "unique_osm_way_count": len(
                        {
                            osmid
                            for value in reason_bad["directed_projection_offsets"]
                            for osmid in _row_osmids(value)
                        }
                    ),
                    "unique_cbg_count": int(
                        quarantine.loc[
                            quarantine["source_id"].astype(str).isin(reason_ids),
                            "census_block_group_geoid",
                        ]
                        .dropna()
                        .astype(str)
                        .nunique()
                    ),
                    "unique_community_count": int(
                        quarantine.loc[
                            quarantine["source_id"].astype(str).isin(reason_ids),
                            "community_id",
                        ]
                        .dropna()
                        .astype(str)
                        .nunique()
                    ),
                    "unique_scc_count": int(
                        reason_bad["anchor_scc_id"].astype(str).nunique()
                    ),
                }
            )
            return summary

        rows.append(
            decorate(
                concentration_summary(inputs, bad, id_column=id_column),
                bad,
                "__all_union__",
            )
        )
        bad["reason_code"] = bad[id_column].astype(str).map(reasons_by_id)
        exploded = bad.explode("reason_code")
        for reason, reason_bad in exploded.groupby("reason_code", sort=True):
            rows.append(
                decorate(
                    concentration_summary(inputs, reason_bad, id_column=id_column),
                    reason_bad,
                    str(reason),
                )
            )
    return rows


def _sample_manifest(certificates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    stage1 = certificates.loc[certificates["audit_stage"].eq("stage1_directional")].copy()
    stage1["reason_code"] = stage1["reason_codes"]
    stage1 = stage1.explode("reason_code")
    customer_sample, customer_coverage = select_h64_samples(
        stage1.loc[stage1["terminal_kind"].eq("customer")],
        id_column="source_id",
        group_columns=["city_slug", "reason_code"],
        minimum_per_group=5,
        namespace=H64_NAMESPACE + ":stage1_customer",
    )
    customer_sample["sample_category"] = "stage1_customer_reason"

    charger_unique = certificates.loc[
        certificates["terminal_kind"].eq("charging_station")
    ].copy()
    charger_unique["reason_code"] = charger_unique["reason_codes"]
    charger_unique = charger_unique.explode("reason_code")
    take_all_chargers = charger_unique["source_id"].nunique() <= 500
    charger_sample, charger_coverage = select_h64_samples(
        charger_unique,
        id_column="source_id",
        group_columns=["city_slug", "reason_code"],
        minimum_per_group=10,
        namespace=H64_NAMESPACE + ":charger",
        take_all=take_all_chargers,
    )
    charger_sample["sample_category"] = (
        "all_quarantined_chargers"
        if take_all_chargers
        else "quarantined_charger_reason"
    )

    stage2 = certificates.loc[
        certificates["audit_stage"].eq("stage2_exact_preflight")
        & certificates["terminal_kind"].eq("customer")
        & certificates["node_outbound_reachable"].eq(True)
        & certificates["node_return_reachable"].eq(True)
        & (
            certificates["turn_outbound_reachable"].eq(False)
            | certificates["turn_return_reachable"].eq(False)
        )
    ].drop_duplicates(["city_slug", "source_id"]).copy()
    stage2["reason_code"] = "stage2_turn_only"
    turn_sample, turn_coverage = select_h64_samples(
        stage2,
        id_column="source_id",
        group_columns=["city_slug", "reason_code"],
        minimum_per_group=5,
        namespace=H64_NAMESPACE + ":stage2_turn_only",
    )
    turn_sample["sample_category"] = "stage2_turn_only"
    major_edge_samples = []
    major_edge_coverage = []
    for city, city_rows in stage2.groupby("city_slug", sort=True):
        edge_counts = city_rows["physical_edge_id"].astype(str).value_counts()
        major_edges = edge_counts.iloc[: min(5, len(edge_counts))].index.tolist()
        selected_edge_ids = set(
            turn_sample.loc[
                turn_sample["city_slug"].astype(str).eq(str(city)),
                "physical_edge_id",
            ].astype(str)
        )
        selected_major_edge_ids = set()
        for edge_id in major_edges:
            candidates = city_rows.loc[
                city_rows["physical_edge_id"].astype(str).eq(str(edge_id))
            ].copy()
            candidates["h64_rank"] = [
                h64_rank(
                    H64_NAMESPACE + ":stage2_major_edge",
                    city,
                    edge_id,
                    source_id,
                )
                for source_id in candidates["source_id"].astype(str)
            ]
            representative = candidates.sort_values(
                ["h64_rank", "source_id"]
            ).iloc[[0]]
            major_edge_samples.append(representative)
            selected_major_edge_ids.update(
                representative["physical_edge_id"].astype(str)
            )
        represented = selected_edge_ids | selected_major_edge_ids
        major_edge_coverage.append(
            {
                "city_slug": str(city),
                "major_physical_edge_ids": list(map(str, major_edges)),
                "all_major_edges_represented": set(map(str, major_edges))
                <= represented,
            }
        )
    if major_edge_samples:
        major = pd.concat(major_edge_samples, ignore_index=True)
        major["sample_category"] = "stage2_turn_only_major_edge"
        turn_sample = pd.concat([turn_sample, major], ignore_index=True)
    sample = pd.concat(
        [customer_sample, charger_sample, turn_sample], ignore_index=True
    ).drop_duplicates(["sample_category", "city_slug", "source_id", "reason_code"])
    coverage = customer_coverage + charger_coverage + turn_coverage
    return sample, {
        "namespace": H64_NAMESPACE,
        "coverage_rows": coverage,
        "major_edge_coverage": major_edge_coverage,
        "coverage_passed": bool(
            coverage
            and all(row["passed"] for row in coverage)
            and all(row["all_major_edges_represented"] for row in major_edge_coverage)
        ),
        "selected_unique_sample_count": sample["source_id"].astype(str).nunique(),
        "selected_sample_row_count": len(sample),
    }


def _write_maps(
    sample: pd.DataFrame,
    cle_root: Path,
    catalogs: dict[str, gpd.GeoDataFrame],
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for city, rows in sample.groupby("city_slug", sort=True):
        cle = load_portable_cle(cle_root, str(city), mode="non_release_pilot")
        raw_tables = _raw_tables(cle)
        raw_by_kind = {
            "customer": raw_tables["customer"].assign(
                _id=raw_tables["customer"]["latent_service_location_id"].astype(str)
            ).set_index("_id"),
            "charging_station": raw_tables["charging_station"].assign(
                _id=raw_tables["charging_station"]["charger_id"].astype(str)
            ).set_index("_id"),
        }
        center = [
            float(pd.to_numeric(raw_tables["customer"]["location_lat"]).mean()),
            float(pd.to_numeric(raw_tables["customer"]["location_lon"]).mean()),
        ]
        city_map = folium.Map(location=center, zoom_start=9, control_scale=True)
        features = []
        catalog = catalogs[str(city)].set_index("physical_edge_id", drop=False)
        for sample_row in rows.to_dict(orient="records"):
            source_id = str(sample_row["source_id"])
            terminal_kind = str(sample_row["terminal_kind"])
            raw = raw_by_kind[terminal_kind].loc[source_id]
            lat = float(raw["location_lat"] if terminal_kind == "customer" else raw["resolved_latitude"])
            lon = float(raw["location_lon"] if terminal_kind == "customer" else raw["resolved_longitude"])
            anchor_lat = float(raw["road_anchor_lat"])
            anchor_lon = float(raw["road_anchor_lon"])
            popup = (
                f"{source_id}<br>{sample_row['sample_category']}<br>"
                f"reasons={sample_row['reason_code']}<br>"
                f"refs={raw['directed_projection_offsets']}"
            )
            folium.Marker([lat, lon], popup=popup).add_to(city_map)
            folium.PolyLine([[lat, lon], [anchor_lat, anchor_lon]], color="orange").add_to(city_map)
            edge = catalog.loc[str(raw["physical_edge_id"])]
            folium.GeoJson(mapping(edge.geometry), tooltip=str(raw["physical_edge_id"])).add_to(city_map)
            edge_parts = (
                list(edge.geometry.geoms)
                if hasattr(edge.geometry, "geoms")
                else [edge.geometry]
            )
            for ref in json_records(raw["directed_projection_offsets"]):
                for edge_part in edge_parts:
                    coordinates = list(edge_part.coords)
                    if ref.get("geometry_orientation") == "reverse_of_physical":
                        coordinates.reverse()
                    directed_line = folium.PolyLine(
                        [[latitude, longitude] for longitude, latitude in coordinates],
                        color="blue",
                        weight=3,
                        opacity=0.75,
                        tooltip=f"{ref['u']}->{ref['v']} key={ref['key']}",
                    ).add_to(city_map)
                    PolyLineTextPath(
                        directed_line,
                        "  >  ",
                        repeat=True,
                        offset=7,
                        attributes={"fill": "blue", "font-weight": "bold"},
                    ).add_to(city_map)
            features.extend(
                [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [lon, lat]},
                        "properties": {"source_id": source_id, "kind": terminal_kind, "popup": popup},
                    },
                    {
                        "type": "Feature",
                        "geometry": mapping(LineString([(lon, lat), (anchor_lon, anchor_lat)])),
                        "properties": {"source_id": source_id, "kind": "projection_connector"},
                    },
                    {
                        "type": "Feature",
                        "geometry": mapping(edge.geometry),
                        "properties": {
                            "source_id": source_id,
                            "kind": "physical_edge",
                            "physical_edge_id": str(raw["physical_edge_id"]),
                            "directed_access_refs": str(raw["directed_projection_offsets"]),
                        },
                    },
                ]
            )
        html_path = output_dir / f"{city}.html"
        geojson_path = output_dir / f"{city}.geojson"
        city_map.save(html_path)
        _write_json(geojson_path, {"type": "FeatureCollection", "features": features})
        artifacts.extend([str(html_path), str(geojson_path)])
    return artifacts


def _materialization_exclusion(root: Path, stage2_customer_ids: set[str]) -> dict[str, Any]:
    paths = sorted(root.rglob("terminal_index.parquet"))
    appearances = []
    for path in paths:
        frame = pd.read_parquet(path, columns=["terminal_kind", "source_id"])
        leaked = frame.loc[
            frame["terminal_kind"].astype(str).eq("customer")
            & frame["source_id"].astype(str).isin(stage2_customer_ids)
        ]
        for source_id in leaked["source_id"].astype(str):
            appearances.append({"path": str(path), "source_id": source_id})
    return {
        "materialized_view_count_checked": len(paths),
        "stage2_quarantined_customer_appearance_count": len(appearances),
        "appearance_examples": appearances[:20],
        "passed": not appearances,
        "runtime_recheck_required_after_each_materialization": True,
        "selector_contract": "mask_before_territory_activation_sampling_materialization",
    }


def run_acceptance(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    code = resolve_git_provenance(
        Path(__file__).resolve().parents[2],
        require_clean=True,
        require_branch="stage2-repair-candidate",
    )
    c1 = json.loads(args.connectivity_audit.read_text(encoding="utf-8"))
    if c1.get("schema") != "cle_evrptw_phase_c1_terminal_connectivity_audit_v3":
        raise ValueError("R2-v2 requires the C1 v3 schema")
    if c1.get("rule_id") != "layered_stage1_pre_split_stage2_family_mask_v1":
        raise ValueError("R2-v2 requires the frozen layered quarantine rule")
    if c1.get("code_provenance", {}).get("code_commit") != code["code_commit"]:
        raise ValueError("C1 and R2-v2 must bind to the same clean candidate commit")
    if c1.get("r2_v1", {}).get("outcome") != "triggered_stop_and_review":
        raise ValueError("R2-v2 must preserve the observed R2-v1 triggered_stop_and_review")
    aggregate = pd.read_parquet(Path(c1["ledger"]))
    stage2_ledger = pd.read_parquet(Path(c1["family_depot_ledger"]))
    profile = load_reference_profile(args.profile, official=False)
    certificates: list[dict[str, Any]] = []
    concentrations: list[dict[str, Any]] = []
    catalogs: dict[str, gpd.GeoDataFrame] = {}
    city_reports = []
    for city_report in c1["cities"]:
        city = str(city_report["city_slug"])
        cle = load_portable_cle(args.cle_root, city, mode="non_release_pilot")
        city_aggregate = aggregate.loc[aggregate["city_slug"].astype(str).eq(city)]
        stage1_records, raw_tables, catalog = _stage1_certificates(cle, city_aggregate)
        city_stage2 = stage2_ledger.loc[stage2_ledger["city_slug"].astype(str).eq(city)]
        stage2_records = _stage2_certificates(
            cle, city_stage2, city_report["stage2_replay_basis"], profile, catalog
        )
        certificates.extend(stage1_records + stage2_records)
        concentrations.extend(_concentration(city, raw_tables, city_aggregate))
        catalogs[city] = catalog
        city_reports.append(
            {
                "city_slug": city,
                "stage1_certificate_count": len(stage1_records),
                "stage2_depot_terminal_certificate_count": len(stage2_records),
                "all_certificates_passed": all(
                    item["certificate_passed"] for item in stage1_records + stage2_records
                ),
            }
        )
    certificate_frame = pd.DataFrame.from_records(certificates)
    certificate_path = args.output.with_suffix(".certificates.parquet")
    certificate_key = [
        "city_slug",
        "audit_stage",
        "terminal_kind",
        "source_id",
        "depot_id",
    ]
    sorted_certificates = certificate_frame.sort_values(
        certificate_key,
        na_position="first",
    ).reset_index(drop=True)
    sorted_certificates.to_parquet(certificate_path, index=False)

    def replay_digest(signature_column: str) -> str:
        replay = sorted_certificates[
            [*certificate_key, signature_column]
        ].rename(columns={signature_column: "replay_signature"})
        payload = replay.to_json(orient="records", date_format="iso").encode(
            "utf-8"
        )
        return hashlib.sha256(payload).hexdigest()

    replay_digest_1 = replay_digest("replay_round_1_signature")
    replay_digest_2 = replay_digest("replay_round_2_signature")
    concentration_path = args.output.with_suffix(".concentration.json")
    union_concentrations = [
        row for row in concentrations if row["reason_code"] == "__all_union__"
    ]
    concentration_complete = len(union_concentrations) == 20
    _write_json(
        concentration_path,
        {
            "schema": "cle_evrptw_connectivity_concentration_v1",
            "passed": concentration_complete,
            "rows": concentrations,
        },
    )
    sample, sample_report = _sample_manifest(certificate_frame)
    sample_path = args.output.with_suffix(".h64_samples.parquet")
    sample.to_parquet(sample_path, index=False)
    map_dir = args.output.parent / "connectivity_h64_maps"
    map_artifacts = _write_maps(sample, args.cle_root, catalogs, map_dir)
    sample_report["sample_manifest"] = str(sample_path)
    sample_sha256 = _sha256(sample_path)
    sample_report["sample_manifest_sha256"] = sample_sha256
    sample_report["map_artifacts"] = map_artifacts
    review_template_path = args.output.with_suffix(".h64_review_template.json")
    _write_json(
        review_template_path,
        {
            "schema": "cle_evrptw_connectivity_h64_manual_review_v1",
            "code_commit": code["code_commit"],
            "sample_manifest": str(sample_path),
            "sample_manifest_sha256": sample_sha256,
            "reviewed_sample_ids": sorted(set(sample["source_id"].astype(str))),
            "reviewer_signoff_id": "",
            "findings": {
                "ignored_valid_access_option_count": -1,
                "incorrect_road_or_projection_semantics_count": -1,
                "certificate_replay_disagreement_count": -1,
            },
        },
    )
    sample_report["manual_review_template"] = str(review_template_path)
    manual = manual_review_gate(
        args.manual_review,
        sample["source_id"].astype(str),
        sample_manifest_sha256=sample_sha256,
        code_commit=code["code_commit"],
    )
    pf1 = {
        "passed": all(city["pf1"]["passed"] for city in c1["cities"]),
        "cities": [
            {"city_slug": city["city_slug"], "passed": city["pf1"]["passed"]}
            for city in c1["cities"]
        ],
    }
    capacity = {
        "passed": all(
            city["stage2_post_mask_capacity"]["passed"] for city in c1["cities"]
        ),
        "rows": [
            {"city_slug": city["city_slug"], **row}
            for city in c1["cities"]
            for row in city["stage2_post_mask_capacity"]["rows"]
        ],
    }
    cohort = json.loads(args.cohort_split.read_text(encoding="utf-8"))
    pf2 = primary_pf2_support(cohort)
    stage2_turn_customer_ids = set(
        certificate_frame.loc[
            certificate_frame["audit_stage"].eq("stage2_exact_preflight")
            & certificate_frame["terminal_kind"].eq("customer")
            & certificate_frame["node_outbound_reachable"].eq(True)
            & certificate_frame["node_return_reachable"].eq(True)
            & (
                certificate_frame["turn_outbound_reachable"].eq(False)
                | certificate_frame["turn_return_reachable"].eq(False)
            ),
            "source_id",
        ].astype(str)
    )
    materialization = _materialization_exclusion(
        args.materialized_root, stage2_turn_customer_ids
    )
    summary_count_rows = []
    for city_report in c1["cities"]:
        city = str(city_report["city_slug"])
        for terminal_kind in ("customer", "charging_station"):
            city_aggregate = aggregate.loc[
                aggregate["city_slug"].astype(str).eq(city)
                & aggregate["terminal_kind"].astype(str).eq(terminal_kind)
            ]
            stages_by_row = city_aggregate["quarantine_stages"].map(_reason_list)
            observed_stage1 = int(
                city_aggregate.loc[
                    stages_by_row.map(lambda value: "stage1_directional" in value),
                    "source_id",
                ]
                .astype(str)
                .nunique()
            )
            city_stage2 = certificate_frame.loc[
                certificate_frame["city_slug"].astype(str).eq(city)
                & certificate_frame["terminal_kind"].astype(str).eq(terminal_kind)
                & certificate_frame["audit_stage"].eq("stage2_exact_preflight")
            ]
            node_ids = set(
                city_stage2.loc[
                    city_stage2["reason_codes"].map(
                        lambda value: any(
                            reason.startswith("node_")
                            for reason in _reason_list(value)
                        )
                    ),
                    "source_id",
                ].astype(str)
            )
            turn_ids = set(
                city_stage2.loc[
                    city_stage2["reason_codes"].map(
                        lambda value: any(
                            reason.startswith("turn_")
                            for reason in _reason_list(value)
                        )
                    ),
                    "source_id",
                ].astype(str)
            )
            expected = city_report[terminal_kind]
            row = {
                "city_slug": city,
                "terminal_kind": terminal_kind,
                "stage1_ledger_unique_count": observed_stage1,
                "stage1_summary_unique_count": int(
                    expected["stage1_directional_quarantine"][
                        "unique_terminal_count"
                    ]
                ),
                "stage2_node_certificate_unique_count": len(node_ids),
                "stage2_node_summary_unique_count": int(
                    expected["stage2_node_quarantine"]["unique_terminal_count"]
                ),
                "stage2_turn_certificate_unique_count": len(turn_ids),
                "stage2_turn_summary_unique_count": int(
                    expected["stage2_turn_quarantine"]["unique_terminal_count"]
                ),
                "union_ledger_unique_count": int(
                    city_aggregate["source_id"].astype(str).nunique()
                ),
                "union_summary_unique_count": int(
                    expected["stage1_or_stage2_union_quarantine"][
                        "unique_terminal_count"
                    ]
                ),
            }
            row["passed"] = (
                row["stage1_ledger_unique_count"]
                == row["stage1_summary_unique_count"]
                and row["stage2_node_certificate_unique_count"]
                == row["stage2_node_summary_unique_count"]
                and row["stage2_turn_certificate_unique_count"]
                == row["stage2_turn_summary_unique_count"]
                and row["union_ledger_unique_count"]
                == row["union_summary_unique_count"]
            )
            summary_count_rows.append(row)

    count_consistency = {
        "stage1_ledger_unique_count": int(
            aggregate.loc[
                aggregate["quarantine_stages"].map(
                    lambda value: "stage1_directional" in _reason_list(value)
                ),
                ["city_slug", "terminal_kind", "source_id"],
            ]
            .drop_duplicates()
            .shape[0]
        ),
        "stage1_certificate_count": int(
            certificate_frame["audit_stage"].eq("stage1_directional").sum()
        ),
        "stage2_ledger_row_count": len(stage2_ledger),
        "stage2_certificate_count": int(
            certificate_frame["audit_stage"].eq("stage2_exact_preflight").sum()
        ),
        "city_terminal_summary_rows": summary_count_rows,
    }
    count_consistency["passed"] = (
        count_consistency["stage1_ledger_unique_count"]
        == count_consistency["stage1_certificate_count"]
        and count_consistency["stage2_ledger_row_count"]
        == count_consistency["stage2_certificate_count"]
        and len(summary_count_rows) == 20
        and all(row["passed"] for row in summary_count_rows)
    )
    automated_assertions = {
        "all_full_replay_certificates_passed": bool(
            len(certificate_frame)
            and certificate_frame["certificate_passed"].astype(bool).all()
        ),
        "ledger_mask_summary_counts_are_exact": count_consistency["passed"],
        "deterministic_replay_digest_equal": replay_digest_1 == replay_digest_2,
        "reason_and_concentration_report_complete": concentration_complete,
        "h64_sample_coverage_passed": sample_report["coverage_passed"],
        "pf1_all_cities_passed": pf1["passed"],
        "every_planned_family_has_post_mask_capacity": capacity["passed"],
        "primary_pf2_support_unaffected": pf2["passed"],
        "stage2_turn_only_customer_materialization_count_zero": materialization["passed"],
        "c1_structural_contract_passed": bool(c1.get("structural_contract_passed")),
        "r2_v1_failure_preserved": c1["r2_v1"]["outcome"] == "triggered_stop_and_review",
    }
    automated_passed = all(automated_assertions.values())
    passed = automated_passed and manual["passed"]
    report = {
        "schema": ACCEPTANCE_SCHEMA,
        "code_provenance": code,
        "passed": passed,
        "c2_allowed": passed,
        "rule_id": "r2_v2_replayable_connectivity_certificate_gate_v1",
        "r2_v1_provenance": c1["r2_v1"],
        "automated_gate": {
            "passed": automated_passed,
            "assertions": automated_assertions,
        },
        "manual_h64_gate": manual,
        "city_certificates": city_reports,
        "count_consistency": count_consistency,
        "deterministic_replay": {
            "round_1_sha256": replay_digest_1,
            "round_2_sha256": replay_digest_2,
            "passed": replay_digest_1 == replay_digest_2,
        },
        "h64_samples": sample_report,
        "pf1": pf1,
        "post_mask_capacity": capacity,
        "primary_pf2_support": pf2,
        "materialization_exclusion": materialization,
        "inputs": {
            "connectivity_audit": str(args.connectivity_audit),
            "connectivity_audit_sha256": _sha256(args.connectivity_audit),
            "cohort_split": str(args.cohort_split),
            "cohort_split_sha256": _sha256(args.cohort_split),
        },
        "artifacts": {
            "certificates": str(certificate_path),
            "certificates_sha256": _sha256(certificate_path),
            "concentration": str(concentration_path),
            "concentration_sha256": _sha256(concentration_path),
        },
        "failure_semantics": "any_failed_certificate_sample_capacity_pf_or_manual_check_forbids_C2",
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cle-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--connectivity-audit", type=Path, required=True)
    parser.add_argument("--cohort-split", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--manual-review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_acceptance(args)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
