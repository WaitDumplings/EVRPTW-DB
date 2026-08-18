#!/usr/bin/env python3
"""Phase C1: exact unique-terminal connectivity and PF-1 audit for the pilot plan."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from evrptw_stage2.profile import load_reference_profile
from evrptw_stage2.provenance import resolve_git_provenance
from evrptw_stage2.reader import load_portable_cle
from evrptw_stage2.release_discipline import (
    CHARGER_QUARANTINE_RATE_LIMIT,
    CUSTOMER_QUARANTINE_RATE_LIMIT,
    quarantine_rate_summary,
)
from evrptw_stage2.road_state import build_family_road_state
from evrptw_stage2.routing import PhysicalRoadNetwork
from evrptw_stage2.selection import (
    _charger_roster_terminal_index,
    _connectivity_quarantine,
    _select_depot_group,
    _star_terminal_index,
)


KNOWN_IDS = {
    "houston": {"msft_nsi_msft_usbf_houston_004823097"},
    "phoenix": {
        "msft_nsi_msft_usbf_phoenix_000722556",
        "msft_nsi_msft_usbf_phoenix_001951371",
        "msft_nsi_msft_usbf_phoenix_001738061",
    },
    "los-angeles": {
        "afdc_113090",
        "afdc_159025",
        "afdc_160987",
        "afdc_176457",
        "afdc_176458",
        "afdc_176459",
        "afdc_179517",
    },
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_families(plan_root: Path) -> pd.DataFrame:
    parts = [
        pd.read_parquet(path)
        for path in sorted(plan_root.rglob("family_index.parquet"))
    ]
    if not parts:
        raise FileNotFoundError(f"No family plan under {plan_root}")
    return pd.concat(parts, ignore_index=True).drop_duplicates("family_id")


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    text = str(value).strip()
    return text or None


def _stage1_customer_sets(
    cle: Any,
    *,
    block_group_path: Path,
) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    """Build the city-level pre-split Stage-1 customer audit universe.

    Connectivity quarantine precedes customer splitting. Consequently every
    source/geometry/road-anchor candidate enters the denominator, while every
    directional quarantine is retained in the ledger with a null split pool.
    """

    raw = gpd.read_parquet(cle.service_locations_path)
    base = raw.loc[
        raw["geometry_core_eligible"].fillna(False).astype(bool)
        & raw["physical_edge_id"].notna()
        & pd.to_numeric(raw["road_access_distance_m"], errors="coerce").notna()
    ].copy()
    input_series = base["latent_service_location_id"].astype(str)
    if input_series.duplicated().any():
        raise ValueError(f"Duplicate pre-split customer IDs in {cle.city_slug}")
    input_ids = set(input_series)
    bad = base.loc[~base["protected_roundtrip_eligible"].fillna(False).astype(bool)].copy()
    if bad.empty:
        return input_ids, set(), []

    points = bad.to_crs("EPSG:4326")
    if not set(points.geometry.geom_type.dropna().unique()) <= {"Point", "MultiPoint"}:
        metric = points.to_crs(points.estimate_utm_crs())
        metric.geometry = metric.geometry.representative_point()
        points = metric.to_crs("EPSG:4326")
    block_groups = gpd.read_file(block_group_path).to_crs("EPSG:4326")
    geoid_field = next(
        (field for field in ("GEOID", "GEOIDFQ", "geoid", "GEOID20") if field in block_groups),
        None,
    )
    if geoid_field is None:
        raise ValueError("Census block-group source has no recognized GEOID field")
    joined = gpd.sjoin(
        points,
        block_groups[[geoid_field, "geometry"]],
        how="left",
        predicate="within",
    )
    if joined.index.duplicated().any():
        raise ValueError(
            f"Cannot assign each Stage-1 quarantined customer in {cle.city_slug} "
            "to at most one block group"
        )
    joined["census_block_group_geoid"] = joined[geoid_field].astype("string")
    joined["community_id"] = [
        (
            f"{cle.city_slug}:bg:{geoid}:scc:{scc}"
            if _optional_text(geoid) is not None and _optional_text(scc) is not None
            else None
        )
        for geoid, scc in zip(
            joined["census_block_group_geoid"], joined["anchor_scc_id"], strict=True
        )
    ]
    bad_ids = set(joined["latent_service_location_id"].astype(str))
    records = []
    for row in joined.to_dict(orient="records"):
        reasons = []
        if not bool(row["protected_inbound_access_eligible"]):
            reasons.append("stage1_no_reference_scc_inbound_access")
        if not bool(row["protected_outbound_access_eligible"]):
            reasons.append("stage1_no_reference_scc_outbound_access")
        records.append(
            {
                "city_slug": cle.city_slug,
                "terminal_kind": "customer",
                "source_id": str(row["latent_service_location_id"]),
                "audit_stage": "stage1_directional",
                "reason_codes": reasons,
                "physical_edge_id": str(row["physical_edge_id"]),
                "anchor_scc_id": _optional_text(row.get("anchor_scc_id")),
                "directed_edge_ref_count": row.get("directed_edge_ref_count"),
                "directed_projection_offsets": _optional_text(
                    row.get("directed_projection_offsets")
                ),
                "census_block_group_geoid": _optional_text(
                    row.get("census_block_group_geoid")
                ),
                "community_id": _optional_text(row.get("community_id")),
                "split_pool": None,
                "split_assignment_status": "excluded_pre_split_connectivity",
                "stage1_generation_eligible": False,
                "family_connectivity_eligible": False,
                "stage1_inbound_access_eligible": bool(
                    row["protected_inbound_access_eligible"]
                ),
                "stage1_outbound_access_eligible": bool(
                    row["protected_outbound_access_eligible"]
                ),
                "stage2_node_outbound_reachable": None,
                "stage2_node_return_reachable": None,
                "stage2_turn_outbound_reachable": None,
                "stage2_turn_return_reachable": None,
                "depot_id": None,
            }
        )
    if len(records) != len(bad_ids):
        raise ValueError(
            f"Stage-1 customer quarantine ledger is not one-row-per-ID in {cle.city_slug}"
        )
    return input_ids, bad_ids, records


def _stage1_charger_sets(cle: Any) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    raw = pd.read_parquet(cle.chargers_path)
    base_mask = (
        raw["restricted_public"].ne(True)
        & raw["reference_charge_mode"].ne("unsupported_or_unresolved")
        & raw["coordinate_candidate_eligible"].fillna(False).astype(bool)
    )
    base = raw.loc[base_mask].copy()
    input_series = base["charger_id"].astype(str)
    if input_series.duplicated().any():
        raise ValueError(f"Duplicate pre-split charger IDs in {cle.city_slug}")
    bad = base.loc[~base["protected_roundtrip_eligible"].fillna(False).astype(bool)]
    input_ids = set(input_series)
    bad_ids = set(bad["charger_id"].astype(str))
    records = []
    for row in bad.to_dict(orient="records"):
        reasons = []
        if not bool(row["protected_inbound_access_eligible"]):
            reasons.append("stage1_no_reference_scc_inbound_access")
        if not bool(row["protected_outbound_access_eligible"]):
            reasons.append("stage1_no_reference_scc_outbound_access")
        records.append(
            {
                "city_slug": cle.city_slug,
                "terminal_kind": "charging_station",
                "source_id": str(row["charger_id"]),
                "audit_stage": "stage1_directional",
                "reason_codes": reasons,
                "physical_edge_id": str(row["physical_edge_id"]),
                "anchor_scc_id": _optional_text(row.get("anchor_scc_id")),
                "directed_edge_ref_count": row.get("directed_edge_ref_count"),
                "directed_projection_offsets": _optional_text(
                    row.get("directed_projection_offsets")
                ),
                "census_block_group_geoid": None,
                "community_id": None,
                "split_pool": None,
                "split_assignment_status": "not_applicable_terminal_kind",
                "stage1_generation_eligible": None,
                "family_connectivity_eligible": False,
                "stage1_inbound_access_eligible": bool(
                    row["protected_inbound_access_eligible"]
                ),
                "stage1_outbound_access_eligible": bool(
                    row["protected_outbound_access_eligible"]
                ),
                "stage2_node_outbound_reachable": None,
                "stage2_node_return_reachable": None,
                "stage2_turn_outbound_reachable": None,
                "stage2_turn_return_reachable": None,
                "depot_id": None,
            }
        )
    if len(records) != len(bad_ids):
        raise ValueError(
            f"Stage-1 charger quarantine ledger is not one-row-per-ID in {cle.city_slug}"
        )
    return input_ids, bad_ids, records


def _customer_split_contract(
    eligible_customer_ids: set[str],
    split: pd.DataFrame,
    stage1_quarantined_customer_ids: set[str],
    stage2_quarantined_customer_ids: set[str],
) -> dict[str, Any]:
    required = {"latent_service_location_id", "customer_pool"}
    missing_columns = required - set(split.columns)
    if missing_columns:
        raise ValueError(f"Customer split lacks columns: {sorted(missing_columns)}")
    split_ids = split["latent_service_location_id"].astype(str)
    duplicate_ids = set(split_ids.loc[split_ids.duplicated(keep=False)])
    split_id_set = set(split_ids)
    invalid_pool_ids = set(
        split_ids.loc[~split["customer_pool"].isin({"train", "heldout"})]
    )
    missing_eligible = eligible_customer_ids - split_id_set
    unexpected_split = split_id_set - eligible_customer_ids
    stage1_overlap = stage1_quarantined_customer_ids & split_id_set
    stage2_missing = stage2_quarantined_customer_ids - split_id_set
    stage2_pool_ids = stage2_quarantined_customer_ids & split_id_set
    assertions = {
        "every_stage1_generation_eligible_customer_has_exactly_one_frozen_pool": (
            not duplicate_ids
            and not invalid_pool_ids
            and not missing_eligible
            and not unexpected_split
        ),
        "every_stage1_quarantined_customer_is_excluded_before_split": (
            not stage1_overlap
        ),
        "every_stage2_only_quarantined_customer_retains_one_frozen_pool": (
            not stage2_missing
            and len(stage2_pool_ids) == len(stage2_quarantined_customer_ids)
        ),
        "leakage_never_drops_eligible_customer_for_missing_pool": (
            not missing_eligible
        ),
    }
    return {
        "schema": "cle_evrptw_layered_customer_split_contract_v2",
        "passed": all(assertions.values()),
        "assertions": assertions,
        "eligible_customer_count": len(eligible_customer_ids),
        "split_customer_count": len(split_id_set),
        "duplicate_split_customer_ids": sorted(duplicate_ids)[:10],
        "invalid_pool_customer_ids": sorted(invalid_pool_ids)[:10],
        "missing_eligible_customer_ids": sorted(missing_eligible)[:10],
        "unexpected_split_customer_ids": sorted(unexpected_split)[:10],
        "stage1_quarantined_customer_ids_in_split": sorted(stage1_overlap)[:10],
        "stage2_quarantined_customer_ids_missing_from_split": sorted(stage2_missing)[:10],
        "stage2_quarantined_customer_retained_pool_count": len(stage2_pool_ids),
    }


def _stage2_ledger_record(
    item: dict[str, Any],
    *,
    city_slug: str,
    depot_id: str,
    split_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_id = str(item["source_id"])
    reasons = set(map(str, item["reason_codes"]))
    split_row = split_lookup.get(source_id, {})
    is_customer = str(item["terminal_kind"]) == "customer"
    split_pool = _optional_text(split_row.get("customer_pool")) if is_customer else None
    if is_customer and split_pool not in {"train", "heldout"}:
        raise ValueError(
            f"Stage-2 quarantined customer lacks a frozen C0 pool: {source_id}"
        )
    return {
        **item,
        "city_slug": city_slug,
        "source_id": source_id,
        "audit_stage": "stage2_exact_preflight",
        "reason_codes": sorted(reasons),
        "census_block_group_geoid": (
            _optional_text(split_row.get("census_block_group_geoid"))
            if is_customer
            else None
        ),
        "community_id": (
            _optional_text(split_row.get("community_id")) if is_customer else None
        ),
        "split_pool": split_pool,
        "split_assignment_status": (
            "assigned_before_stage2_turn_preflight"
            if is_customer
            else "not_applicable_terminal_kind"
        ),
        "stage1_generation_eligible": True if is_customer else None,
        "family_connectivity_eligible": False,
        "stage1_inbound_access_eligible": None,
        "stage1_outbound_access_eligible": None,
        "stage2_node_outbound_reachable": (
            "node_unreachable_from_depot" not in reasons
        ),
        "stage2_node_return_reachable": (
            "node_cannot_return_to_depot" not in reasons
        ),
        "stage2_turn_outbound_reachable": (
            "turn_unreachable_from_depot" not in reasons
        ),
        "stage2_turn_return_reachable": (
            "turn_cannot_return_to_depot" not in reasons
        ),
        "depot_id": depot_id,
    }


def _aggregate_quarantine_ledger(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    boolean_fields = (
        "stage1_inbound_access_eligible",
        "stage1_outbound_access_eligible",
        "stage2_node_outbound_reachable",
        "stage2_node_return_reachable",
        "stage2_turn_outbound_reachable",
        "stage2_turn_return_reachable",
    )
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            str(record["city_slug"]),
            str(record["terminal_kind"]),
            str(record["source_id"]),
        )
        groups.setdefault(key, []).append(record)

    result = []
    for key, items in sorted(groups.items()):
        first = dict(items[0])
        first["audit_stage"] = "connectivity_union"
        first["quarantine_stages"] = sorted(
            {str(item["audit_stage"]) for item in items}
        )
        first["reason_codes"] = sorted(
            {
                str(reason)
                for item in items
                for reason in item.get("reason_codes", [])
            }
        )
        first["depot_ids"] = sorted(
            {
                str(item["depot_id"])
                for item in items
                if _optional_text(item.get("depot_id")) is not None
            }
        )
        first["depot_id"] = None
        for field in boolean_fields:
            values = [
                bool(item[field])
                for item in items
                if item.get(field) is not None
            ]
            first[field] = all(values) if values else None
        stage1_items = [
            item for item in items if item["audit_stage"] == "stage1_directional"
        ]
        stage2_items = [
            item for item in items if item["audit_stage"] == "stage2_exact_preflight"
        ]
        if stage1_items:
            first["split_pool"] = None
            first["split_assignment_status"] = (
                "excluded_pre_split_connectivity"
                if first["terminal_kind"] == "customer"
                else "not_applicable_terminal_kind"
            )
            first["stage1_generation_eligible"] = (
                False if first["terminal_kind"] == "customer" else None
            )
        elif stage2_items and first["terminal_kind"] == "customer":
            pools = {str(item["split_pool"]) for item in stage2_items}
            if len(pools) != 1 or not pools <= {"train", "heldout"}:
                raise ValueError(
                    f"Stage-2 customer has inconsistent frozen pools: {key}: {pools}"
                )
            first["split_pool"] = next(iter(pools))
            first["split_assignment_status"] = "assigned_before_stage2_turn_preflight"
            first["stage1_generation_eligible"] = True
        first["family_connectivity_eligible"] = False
        result.append(first)
    return result


def _quarantine_ledger_contract(
    city_slug: str,
    ledger: list[dict[str, Any]],
    expected_stage1_customer_ids: set[str],
    expected_stage2_only_customer_ids: set[str],
    expected_missing_community_ids: set[str] | None = None,
) -> dict[str, Any]:
    customers = [
        item
        for item in ledger
        if item["city_slug"] == city_slug and item["terminal_kind"] == "customer"
    ]
    observed = [str(item["source_id"]) for item in customers]
    observed_set = set(observed)
    expected_customer_ids = (
        expected_stage1_customer_ids | expected_stage2_only_customer_ids
    )
    observed_missing_community_ids = {
        str(item["source_id"])
        for item in customers
        if item.get("community_id") is None
    }
    expected_missing_community_ids = set(expected_missing_community_ids or set())
    by_id = {str(item["source_id"]): item for item in customers}
    observed_stage1 = {
        source_id
        for source_id, item in by_id.items()
        if "stage1_directional" in item.get("quarantine_stages", [])
    }
    observed_stage2_only = {
        source_id
        for source_id, item in by_id.items()
        if item.get("quarantine_stages") == ["stage2_exact_preflight"]
    }
    assertions = {
        "every_quarantined_customer_appears_exactly_once_in_city_ledger": (
            len(observed) == len(observed_set) and observed_set == expected_customer_ids
        ),
        "r2_numerator_includes_quarantined_customers_with_missing_community_ids": (
            expected_missing_community_ids <= observed_set
            and expected_missing_community_ids <= observed_missing_community_ids
        ),
        "every_stage1_quarantined_customer_has_null_split_pool": all(
            by_id[source_id].get("split_pool") is None
            for source_id in expected_stage1_customer_ids & observed_set
        ),
        "every_stage1_quarantined_customer_is_excluded_before_split": all(
            by_id[source_id].get("split_assignment_status")
            == "excluded_pre_split_connectivity"
            and by_id[source_id].get("stage1_generation_eligible") is False
            and by_id[source_id].get("family_connectivity_eligible") is False
            for source_id in expected_stage1_customer_ids & observed_set
        ),
        "every_stage2_only_customer_retains_one_frozen_pool_and_is_family_masked": all(
            by_id[source_id].get("split_pool") in {"train", "heldout"}
            and by_id[source_id].get("split_assignment_status")
            == "assigned_before_stage2_turn_preflight"
            and by_id[source_id].get("stage1_generation_eligible") is True
            and by_id[source_id].get("family_connectivity_eligible") is False
            for source_id in expected_stage2_only_customer_ids & observed_set
        ),
        "stage_membership_matches_exactly": (
            observed_stage1 == expected_stage1_customer_ids
            and observed_stage2_only == expected_stage2_only_customer_ids
        ),
    }
    return {
        "schema": "cle_evrptw_layered_quarantine_ledger_contract_v2",
        "passed": all(assertions.values()),
        "assertions": assertions,
        "expected_unique_customer_count": len(expected_customer_ids),
        "observed_unique_customer_count": len(observed_set),
        "expected_stage1_customer_count": len(expected_stage1_customer_ids),
        "expected_stage2_only_customer_count": len(expected_stage2_only_customer_ids),
        "expected_missing_community_customer_count": len(
            expected_missing_community_ids
        ),
        "observed_missing_community_customer_count": len(
            observed_missing_community_ids
        ),
        "missing_expected_communityless_customer_ids": sorted(
            expected_missing_community_ids - observed_missing_community_ids
        )[:10],
        "missing_customer_ids": sorted(expected_customer_ids - observed_set)[:10],
        "unexpected_customer_ids": sorted(observed_set - expected_customer_ids)[:10],
    }


def _reason_sets(ledger: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    node: set[str] = set()
    turn: set[str] = set()
    for item in ledger:
        reasons = set(map(str, item["reason_codes"]))
        if any(reason.startswith("node_") for reason in reasons):
            node.add(str(item["source_id"]))
        if any(reason.startswith("turn_") for reason in reasons):
            turn.add(str(item["source_id"]))
    return node, turn


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    code_provenance = resolve_git_provenance(
        Path(__file__).resolve().parents[2],
        require_clean=True,
        require_branch="stage2-repair-candidate",
    )
    profile = load_reference_profile(args.profile, official=False)
    all_stage2_ledger: list[dict[str, Any]] = []
    families = _load_families(args.plan_root)
    plan_registry = json.loads(
        (args.plan_root / "split_registry.json").read_text(encoding="utf-8")
    )
    if plan_registry.get("code_provenance", {}).get("code_commit") != code_provenance[
        "code_commit"
    ]:
        raise ValueError("C1 pilot plan is not bound to the current clean candidate commit")
    expected_tracks = {"train", "validation"}
    if set(families["track_id"].astype(str)) != expected_tracks:
        raise ValueError("Phase C1 accepts only the frozen train/validation pilot plan")
    source_preset = json.loads(args.block_group_preset.read_text(encoding="utf-8"))
    all_ledger: list[dict[str, Any]] = []
    city_reports: list[dict[str, Any]] = []
    for city, city_families in families.groupby("city_slug", sort=True):
        city_started = time.perf_counter()
        cle = load_portable_cle(args.cle_root, str(city), mode="non_release_pilot")
        state = source_preset["city_to_state"][str(city)]
        block_group_path = args.block_group_source_dir / (
            f"tl_{int(source_preset['vintage'])}_{source_preset['states'][state]}_bg.zip"
        )
        customer_input, stage1_customers, stage1_customer_ledger = _stage1_customer_sets(
            cle,
            block_group_path=block_group_path,
        )
        charger_input, stage1_chargers, stage1_charger_ledger = _stage1_charger_sets(cle)
        city_raw_ledger = stage1_customer_ledger + stage1_charger_ledger

        split = pd.read_parquet(
            args.split_root / str(city) / "customer_split_manifest.parquet"
        )
        customers = cle.read_service_locations().reset_index(drop=True)
        eligible_customer_ids = set(
            customers["latent_service_location_id"].astype(str)
        )
        split_lookup = (
            split.assign(
                _source_id=split["latent_service_location_id"].astype(str)
            )
            .drop_duplicates("_source_id")
            .set_index("_source_id")
            .to_dict(orient="index")
        )
        chargers = cle.read_chargers().reset_index(drop=True)
        family_depot_ids: dict[str, str] = {}
        depots = cle.read_depots().reset_index(drop=True)
        directed_speeds = pd.read_parquet(cle.speeds_path)
        first_family = city_families.sort_values("family_id").iloc[0]
        road_state, _ = build_family_road_state(
            directed_speeds,
            day_type=str(first_family["day_type"]),
            road_state_seed=int(first_family["road_state_seed"]),
            profile=profile,
        )
        network = PhysicalRoadNetwork.from_files(cle.graph_path, road_state, profile)
        selected_depots: dict[str, pd.Series] = {}
        for family in city_families.itertuples(index=False):
            depot, _ = _select_depot_group(
                depots,
                seed=int(family.depot_seed),
                track=str(profile["stage2_spatial"]["depot_track"]),
            )
            family_depot_ids[str(family.family_id)] = str(depot["candidate_id"])
            selected_depots[str(depot["candidate_id"])] = depot

        customer_ledger: list[dict[str, Any]] = []
        charger_ledger: list[dict[str, Any]] = []
        pf1_rows: list[dict[str, Any]] = []
        energy_coefficient = float(profile["energy"]["specific_energy_consumption_kwh_per_km"])
        battery = float(profile["energy"]["battery_capacity_kwh"])
        for depot_id, depot in sorted(selected_depots.items()):
            customer_star = network.route_depot_star(_star_terminal_index(depot, customers))
            _, bad_customers = _connectivity_quarantine(
                customers,
                customer_star,
                id_column="latent_service_location_id",
                terminal_kind="customer",
            )
            charger_star = network.route_depot_star(
                _charger_roster_terminal_index(depot, customers.iloc[:0], chargers)
            )
            _, bad_chargers = _connectivity_quarantine(
                chargers,
                charger_star,
                id_column="charger_id",
                terminal_kind="charging_station",
            )
            for item in bad_customers:
                customer_ledger.append(
                    _stage2_ledger_record(
                        item,
                        city_slug=str(city),
                        depot_id=depot_id,
                        split_lookup=split_lookup,
                    )
                )
            for item in bad_chargers:
                charger_ledger.append(
                    _stage2_ledger_record(
                        item,
                        city_slug=str(city),
                        depot_id=depot_id,
                        split_lookup={},
                    )
                )
            direct_energy = (
                charger_star.connectivity_eligible[1:]
                & (charger_star.outbound_distance_km[1:] * energy_coefficient <= battery + 1e-9)
                & (charger_star.inbound_distance_km[1:] * energy_coefficient <= battery + 1e-9)
            )
            pf1_rows.append(
                {
                    "depot_id": depot_id,
                    "exact_direct_bidirectional_energy_lower_bound_count": int(direct_energy.sum()),
                    "required_count": 50,
                    "passed": int(direct_energy.sum()) >= 50,
                    "lower_bound_semantics": (
                        "each counted CS directly communicates with depot in both directions "
                        "within one battery; therefore it belongs to the multihop communicating set"
                    ),
                }
            )
        depot_family_ids: dict[str, list[str]] = {}
        for family_id, depot_id in family_depot_ids.items():
            depot_family_ids.setdefault(depot_id, []).append(family_id)
        for item in customer_ledger + charger_ledger:
            item["family_ids"] = sorted(depot_family_ids[str(item["depot_id"])])
        all_stage2_ledger.extend(customer_ledger + charger_ledger)
        customer_node, customer_turn = _reason_sets(customer_ledger)
        charger_node, charger_turn = _reason_sets(charger_ledger)
        stage2_only_customers = (customer_node | customer_turn) - stage1_customers
        customer_union = stage1_customers | customer_node | customer_turn
        charger_union = stage1_chargers | charger_node | charger_turn
        customer_bad_by_depot: dict[str, set[str]] = {}
        for item in customer_ledger:
            customer_bad_by_depot.setdefault(str(item["depot_id"]), set()).add(
                str(item["source_id"])
            )
        capacity_rows = []
        for family in city_families.itertuples(index=False):
            pool = str(family.customer_pool)
            pool_ids = set(
                split.loc[
                    split["customer_pool"].astype(str).eq(pool),
                    "latent_service_location_id",
                ].astype(str)
            )
            bad_ids = customer_bad_by_depot.get(family_depot_ids[str(family.family_id)], set())
            eligible_after_mask = len(pool_ids - bad_ids)
            required = int(family.parent_customer_count)
            capacity_rows.append(
                {
                    "family_id": str(family.family_id),
                    "depot_id": family_depot_ids[str(family.family_id)],
                    "customer_pool": pool,
                    "input_pool_count": len(pool_ids),
                    "stage2_masked_count": len(pool_ids & bad_ids),
                    "eligible_after_stage2_mask_count": eligible_after_mask,
                    "required_customer_count": required,
                    "passed": eligible_after_mask >= required,
                }
            )
        city_raw_ledger.extend(customer_ledger + charger_ledger)
        city_ledger = _aggregate_quarantine_ledger(city_raw_ledger)
        all_ledger.extend(city_ledger)

        expected_missing_community_ids = {
            str(item["source_id"])
            for item in city_raw_ledger
            if item["terminal_kind"] == "customer"
            and item.get("community_id") is None
        }

        split_contract = _customer_split_contract(
            eligible_customer_ids,
            split,
            stage1_customers,
            stage2_only_customers,
        )
        ledger_contract = _quarantine_ledger_contract(
            str(city),
            city_ledger,
            stage1_customers,
            stage2_only_customers,
            expected_missing_community_ids,
        )
        customer_rates = quarantine_rate_summary(
            customer_input,
            stage1_directional_ids=stage1_customers,
            stage2_node_ids=customer_node,
            stage2_turn_ids=customer_turn,
            rate_limit=CUSTOMER_QUARANTINE_RATE_LIMIT,
        )
        charger_rates = quarantine_rate_summary(
            charger_input,
            stage1_directional_ids=stage1_chargers,
            stage2_node_ids=charger_node,
            stage2_turn_ids=charger_turn,
            rate_limit=CHARGER_QUARANTINE_RATE_LIMIT,
        )
        observed = customer_union | charger_union
        known = {
            terminal_id: {"quarantined": terminal_id in observed}
            for terminal_id in sorted(KNOWN_IDS.get(str(city), set()))
        }
        city_reports.append(
            {
                "city_slug": str(city),
                "audited_pilot_family_count": len(city_families),
                "unique_selected_depot_count": len(selected_depots),
                "customer_audit_universe": {
                    "rule_id": "layered_stage1_pre_split_stage2_family_mask_v1",
                    "denominator_stage": (
                        "after non-connectivity source/geometry/road-anchor eligibility "
                        "and before Stage-1/Stage-2 connectivity filtering"
                    ),
                    "pre_split_unique_customer_count": len(customer_input),
                    "generation_eligible_unique_customer_count": len(
                        eligible_customer_ids
                    ),
                },
                "stage2_replay_basis": {
                    "family_id": str(first_family["family_id"]),
                    "day_type": str(first_family["day_type"]),
                    "road_state_seed": int(first_family["road_state_seed"]),
                    "topology_semantics": "canonical turn reachability is speed invariant",
                },
                "customer": customer_rates,
                "charging_station": charger_rates,
                "customer_split_contract": split_contract,
                "quarantine_ledger_contract": ledger_contract,
                "stage2_post_mask_capacity": {
                    "schema": "cle_evrptw_stage2_post_mask_capacity_v1",
                    "rows": capacity_rows,
                    "passed": bool(capacity_rows and all(row["passed"] for row in capacity_rows)),
                },
                "pf1": {
                    "schema": "cle_evrptw_pf1_exact_lower_bound_v1",
                    "rows": pf1_rows,
                    "passed": bool(pf1_rows and all(row["passed"] for row in pf1_rows)),
                },
                "known_terminal_audit": known,
                "elapsed_seconds": time.perf_counter() - city_started,
            }
        )

    r2_v1_fixed_rate_all_passed = all(
        city["customer"]["passed"] and city["charging_station"]["passed"]
        for city in city_reports
    )
    structural_contract_passed = all(
        city["customer_split_contract"]["passed"]
        and city["quarantine_ledger_contract"]["passed"]
        and city["stage2_post_mask_capacity"]["passed"]
        and city["pf1"]["passed"]
        and all(item["quarantined"] for item in city["known_terminal_audit"].values())
        for city in city_reports
    )
    passed = r2_v1_fixed_rate_all_passed and structural_contract_passed
    ledger_path = args.output.with_suffix(".ledger.parquet")
    ledger_columns = [
        "city_slug",
        "terminal_kind",
        "source_id",
        "audit_stage",
        "quarantine_stages",
        "reason_codes",
        "physical_edge_id",
        "anchor_scc_id",
        "directed_edge_ref_count",
        "directed_projection_offsets",
        "census_block_group_geoid",
        "community_id",
        "split_pool",
        "split_assignment_status",
        "stage1_generation_eligible",
        "family_connectivity_eligible",
        "stage1_inbound_access_eligible",
        "stage1_outbound_access_eligible",
        "stage2_node_outbound_reachable",
        "stage2_node_return_reachable",
        "stage2_turn_outbound_reachable",
        "stage2_turn_return_reachable",
        "depot_ids",
    ]
    pd.DataFrame.from_records(all_ledger, columns=ledger_columns).to_parquet(
        ledger_path, index=False
    )
    stage2_ledger_path = args.output.with_suffix(".family_depot_ledger.parquet")
    pd.DataFrame.from_records(all_stage2_ledger).sort_values(
        ["city_slug", "terminal_kind", "depot_id", "source_id"]
    ).to_parquet(stage2_ledger_path, index=False)
    report = {
        "schema": "cle_evrptw_phase_c1_terminal_connectivity_audit_v3",
        "code_provenance": code_provenance,
        "passed": passed,
        "structural_contract_passed": structural_contract_passed,
        "rule_id": "layered_stage1_pre_split_stage2_family_mask_v1",
        "policy": "preserve_r2_v1_trigger_and_require_r2_v2_certificate_acceptance",
        "r2_v1": {
            "fixed_rate_all_passed": r2_v1_fixed_rate_all_passed,
            "outcome": (
                "within_original_stop_review_trigger"
                if r2_v1_fixed_rate_all_passed
                else "triggered_stop_and_review"
            ),
            "active_acceptance_rule": False,
            "superseded_by": "r2_v2_replayable_connectivity_certificate_gate_v1",
            "raw_rates_are_mandatory_report_only": True,
        },
        "r2_v2": {
            "status": "requires_connectivity_audit_acceptance_v2",
            "c2_allowed": False,
        },
        "customer_split_semantics": {
            "stage1_quarantine": {
                "split_pool": None,
                "split_assignment_status": "excluded_pre_split_connectivity",
                "stage1_generation_eligible": False,
            },
            "stage2_only_quarantine": {
                "split_pool": "retain_exact_frozen_C0_train_or_heldout_value",
                "split_assignment_status": "assigned_before_stage2_turn_preflight",
                "stage1_generation_eligible": True,
                "family_connectivity_eligible": False,
                "mask_stage": "before_territory_activation_sampling_and_materialization",
            },
            "eligible_split_universe": (
                "Stage-1 connectivity-eligible customers; complete-community 80/20"
            ),
        },
        "cities": city_reports,
        "ledger": str(ledger_path),
        "family_depot_ledger": str(stage2_ledger_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cle-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--block-group-preset", type=Path, required=True)
    parser.add_argument("--block-group-source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(args)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
