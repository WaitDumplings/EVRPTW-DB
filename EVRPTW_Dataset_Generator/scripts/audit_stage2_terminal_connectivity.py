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


def _stage1_customer_sets(
    cle: Any,
    *,
    split_root: Path,
    block_group_path: Path,
) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    raw = gpd.read_parquet(cle.service_locations_path)
    base = raw.loc[
        raw["geometry_core_eligible"].fillna(False).astype(bool)
        & raw["physical_edge_id"].notna()
        & pd.to_numeric(raw["road_access_distance_m"], errors="coerce").notna()
    ].copy()
    split = pd.read_parquet(
        split_root / cle.city_slug / "customer_split_manifest.parquet"
    )
    train_ids = set(
        split.loc[split["customer_pool"].eq("train"), "latent_service_location_id"]
        .astype(str)
    )
    bad = base.loc[~base["protected_roundtrip_eligible"].fillna(False).astype(bool)].copy()
    if bad.empty:
        return train_ids, set(), []

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
    if joined.index.duplicated().any() or joined[geoid_field].isna().any():
        raise ValueError(
            f"Cannot assign every Stage-1 quarantined customer in {cle.city_slug} "
            "to exactly one block group"
        )
    joined["community_id"] = [
        f"{cle.city_slug}:bg:{geoid}:scc:{scc}"
        for geoid, scc in zip(joined[geoid_field].astype(str), joined["anchor_scc_id"].astype(str))
    ]
    communities = pd.read_parquet(split_root / cle.city_slug / "community_manifest.parquet")
    pool = communities.set_index("community_id")["customer_pool"]
    joined["customer_pool"] = joined["community_id"].map(pool)
    if joined["customer_pool"].isna().any():
        missing = joined.loc[
            joined["customer_pool"].isna(), "latent_service_location_id"
        ].astype(str).tolist()
        raise ValueError(
            "Stage-1 quarantined customers occupy communities absent from the frozen "
            f"split; stop-and-review: {missing[:10]}"
        )
    bad_train = joined.loc[joined["customer_pool"].eq("train")].copy()
    bad_ids = set(bad_train["latent_service_location_id"].astype(str))
    records = []
    for row in bad_train.itertuples(index=False):
        reasons = []
        if not bool(row.protected_inbound_access_eligible):
            reasons.append("stage1_no_reference_scc_inbound_access")
        if not bool(row.protected_outbound_access_eligible):
            reasons.append("stage1_no_reference_scc_outbound_access")
        records.append(
            {
                "city_slug": cle.city_slug,
                "terminal_kind": "customer",
                "source_id": str(row.latent_service_location_id),
                "audit_stage": "stage1_directional",
                "reason_codes": reasons,
                "physical_edge_id": str(row.physical_edge_id),
            }
        )
    return train_ids | bad_ids, bad_ids, records


def _stage1_charger_sets(cle: Any) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    raw = pd.read_parquet(cle.chargers_path)
    base_mask = (
        raw["restricted_public"].ne(True)
        & raw["reference_charge_mode"].ne("unsupported_or_unresolved")
        & raw["coordinate_candidate_eligible"].fillna(False).astype(bool)
    )
    base = raw.loc[base_mask].copy()
    bad = base.loc[~base["protected_roundtrip_eligible"].fillna(False).astype(bool)]
    input_ids = set(base["charger_id"].astype(str))
    bad_ids = set(bad["charger_id"].astype(str))
    records = []
    for row in bad.itertuples(index=False):
        reasons = []
        if not bool(row.protected_inbound_access_eligible):
            reasons.append("stage1_no_reference_scc_inbound_access")
        if not bool(row.protected_outbound_access_eligible):
            reasons.append("stage1_no_reference_scc_outbound_access")
        records.append(
            {
                "city_slug": cle.city_slug,
                "terminal_kind": "charging_station",
                "source_id": str(row.charger_id),
                "audit_stage": "stage1_directional",
                "reason_codes": reasons,
                "physical_edge_id": str(row.physical_edge_id),
            }
        )
    return input_ids, bad_ids, records


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
            split_root=args.split_root,
            block_group_path=block_group_path,
        )
        charger_input, stage1_chargers, stage1_charger_ledger = _stage1_charger_sets(cle)
        all_ledger.extend(stage1_customer_ledger)
        all_ledger.extend(stage1_charger_ledger)

        split = pd.read_parquet(
            args.split_root / str(city) / "customer_split_manifest.parquet"
        )
        train_ids = set(
            split.loc[split["customer_pool"].eq("train"), "latent_service_location_id"]
            .astype(str)
        )
        customers = cle.read_service_locations()
        customers = customers.loc[
            customers["latent_service_location_id"].astype(str).isin(train_ids)
        ].reset_index(drop=True)
        chargers = cle.read_chargers().reset_index(drop=True)
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
                customer_ledger.append({**item, "city_slug": str(city), "depot_id": depot_id})
            for item in bad_chargers:
                charger_ledger.append({**item, "city_slug": str(city), "depot_id": depot_id})
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
        all_ledger.extend(
            {**item, "audit_stage": "stage2_exact_preflight"}
            for item in customer_ledger + charger_ledger
        )
        customer_node, customer_turn = _reason_sets(customer_ledger)
        charger_node, charger_turn = _reason_sets(charger_ledger)
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
        observed = stage1_customers | stage1_chargers | customer_node | customer_turn | charger_node | charger_turn
        known = {
            terminal_id: {"quarantined": terminal_id in observed}
            for terminal_id in sorted(KNOWN_IDS.get(str(city), set()))
        }
        city_reports.append(
            {
                "city_slug": str(city),
                "audited_pilot_family_count": len(city_families),
                "unique_selected_depot_count": len(selected_depots),
                "customer": customer_rates,
                "charging_station": charger_rates,
                "pf1": {
                    "schema": "cle_evrptw_pf1_exact_lower_bound_v1",
                    "rows": pf1_rows,
                    "passed": bool(pf1_rows and all(row["passed"] for row in pf1_rows)),
                },
                "known_terminal_audit": known,
                "elapsed_seconds": time.perf_counter() - city_started,
            }
        )

    passed = all(
        city["customer"]["passed"]
        and city["charging_station"]["passed"]
        and city["pf1"]["passed"]
        and all(item["quarantined"] for item in city["known_terminal_audit"].values())
        for city in city_reports
    )
    ledger_path = args.output.with_suffix(".ledger.parquet")
    ledger_columns = [
        "city_slug",
        "terminal_kind",
        "source_id",
        "audit_stage",
        "reason_codes",
        "physical_edge_id",
        "depot_id",
    ]
    pd.DataFrame.from_records(all_ledger, columns=ledger_columns).to_parquet(
        ledger_path, index=False
    )
    report = {
        "schema": "cle_evrptw_phase_c1_terminal_connectivity_audit_v1",
        "code_provenance": code_provenance,
        "passed": passed,
        "policy": "stop_and_review_on_rate_pf1_or_known_id_failure",
        "cities": city_reports,
        "ledger": str(ledger_path),
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
