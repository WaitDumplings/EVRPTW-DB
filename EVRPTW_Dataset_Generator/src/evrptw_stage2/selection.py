"""Deterministic family-level terminal activation for portable CLE packages."""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .amazon import AmazonStage2Artifacts
from .planning import derive_seed
from .reader import PortableCLE
from .release_discipline import roster_fingerprint
from .routing import DepotTerminalStar, PhysicalRoadNetwork, TerminalConnectivityError
from .spatial_activation import (
    SpatialActivationError,
    activate_spatial_customers,
    radial_decile_support_contract,
)


JOINT_SUPPORT_CONTRACT_ID = "c3_joint_spatial_support_v1"


class JointSupportConsistencyError(RuntimeError):
    """A C3-approved capacity contract did not replay during materialization."""

    retryable = False

    def __init__(
        self,
        message: str,
        *,
        capacity_contract_fingerprint: str,
    ) -> None:
        super().__init__(f"C3_ACTIVATION_CONSISTENCY_BUG: {message}")
        self.capacity_contract_fingerprint = capacity_contract_fingerprint
        self.roster_fingerprint = capacity_contract_fingerprint


def sample_day_type(profile: Mapping[str, Any], family_seed: int) -> str:
    cfg = profile["day_type"]
    weekday = int(cfg["weekday_weight"])
    weekend = int(cfg["weekend_weight"])
    rng = np.random.default_rng(derive_seed(family_seed, "day_type"))
    return "weekday" if int(rng.integers(weekday + weekend)) < weekday else "weekend"




def _haversine_km(
    source_lon: np.ndarray,
    source_lat: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
) -> np.ndarray:
    lon1 = np.radians(source_lon)[:, None]
    lat1 = np.radians(source_lat)[:, None]
    lon2 = np.radians(target_lon)[None, :]
    lat2 = np.radians(target_lat)[None, :]
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1
    value = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
        delta_lon / 2.0
    ) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(value)))



def battery_feasible_communicating_mask(
    infrastructure_energy_kwh: np.ndarray,
    *,
    battery_capacity_kwh: float,
) -> np.ndarray:
    """Return depot/CS nodes reachable from and able to return to the depot."""

    energy = np.asarray(infrastructure_energy_kwh, dtype=float)
    if energy.ndim != 2 or energy.shape[0] != energy.shape[1]:
        raise ValueError("Infrastructure energy closure must be square")
    adjacency = np.isfinite(energy) & (energy <= float(battery_capacity_kwh) + 1e-9)
    np.fill_diagonal(adjacency, True)

    def reachable(matrix: np.ndarray) -> np.ndarray:
        seen = np.zeros(len(matrix), dtype=bool)
        seen[0] = True
        frontier = [0]
        while frontier:
            node = frontier.pop()
            for neighbor in np.flatnonzero(matrix[node] & ~seen):
                seen[int(neighbor)] = True
                frontier.append(int(neighbor))
        return seen

    return reachable(adjacency) & reachable(adjacency.T)


def road_time_replacement_deltas(
    running_time_matrix_s: np.ndarray,
    *,
    customer_indices: np.ndarray,
    charger_indices: np.ndarray,
) -> np.ndarray:
    """Compute clamped directed customer/CS replacement-time increments."""

    times = np.asarray(running_time_matrix_s, dtype=float)
    customers = np.asarray(customer_indices, dtype=int)
    chargers = np.asarray(charger_indices, dtype=int)
    outbound = (
        times[np.ix_(customers, chargers)]
        + times[np.ix_(chargers, np.asarray([0], dtype=int))].T
        - times[customers, 0][:, None]
    )
    inbound = (
        times[np.ix_(np.asarray([0], dtype=int), chargers)]
        + times[np.ix_(chargers, customers)].T
        - times[0, customers][:, None]
    )
    return np.maximum(0.0, np.minimum(outbound, inbound))


def select_road_time_charger_indices(
    deltas_s: np.ndarray,
    *,
    count: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministic greedy mean/P90/max coverage over eligible CS deltas."""

    deltas = np.asarray(deltas_s, dtype=float)
    if deltas.ndim != 2 or not deltas.shape[0]:
        raise ValueError("Road-time charger deltas must be a nonempty 2-D array")
    if deltas.shape[1] < count:
        raise ValueError(
            f"Bidirectionally eligible charger roster has {deltas.shape[1]} sites; "
            f"requested {count}"
        )
    current = np.full(deltas.shape[0], np.inf, dtype=float)
    available = np.ones(deltas.shape[1], dtype=bool)
    tie = np.asarray(
        [derive_seed(seed, "road_time_charger", index) for index in range(deltas.shape[1])],
        dtype=np.uint64,
    )
    selected: list[int] = []
    for _ in range(count):
        candidates = np.flatnonzero(available)
        proposed = np.minimum(current[:, None], deltas[:, candidates])
        mean = proposed.mean(axis=0)
        p90 = np.quantile(proposed, 0.90, axis=0)
        maximum = proposed.max(axis=0)
        local = min(
            range(len(candidates)),
            key=lambda index: (
                float(mean[index] + 0.25 * p90[index] + 0.10 * maximum[index]),
                int(tie[candidates[index]]),
                int(candidates[index]),
            ),
        )
        chosen = int(candidates[local])
        selected.append(chosen)
        available[chosen] = False
        current = np.minimum(current, deltas[:, chosen])
    return np.asarray(selected, dtype=np.int32), {
        "policy": "bidirectional_energy_roster_road_time_greedy_v1",
        "eligible_roster_count": int(deltas.shape[1]),
        "selected_count": int(count),
        "objective": "mean_delta_plus_0.25_p90_plus_0.10_max",
        "mean_nearest_replacement_delta_s": float(current.mean()),
        "p95_nearest_replacement_delta_s": float(np.quantile(current, 0.95)),
        "max_nearest_replacement_delta_s": float(current.max()),
    }


def haversine_legacy_comparison_audit(
    customers: pd.DataFrame,
    eligible_chargers: pd.DataFrame,
    eligible_deltas_s: np.ndarray,
    *,
    road_time_selected_positions: np.ndarray,
    count: int,
    seed: int,
) -> dict[str, Any]:
    """Emulate the retired geographic selector strictly for pilot reporting."""

    distances = _haversine_km(
        customers["location_lon"].to_numpy(dtype=float),
        customers["location_lat"].to_numpy(dtype=float),
        eligible_chargers["resolved_longitude"].to_numpy(dtype=float),
        eligible_chargers["resolved_latitude"].to_numpy(dtype=float),
    )
    available = np.ones(len(eligible_chargers), dtype=bool)
    current = np.full(len(customers), np.inf, dtype=float)
    tie = np.asarray(
        [derive_seed(seed, "haversine_legacy_audit", index) for index in range(len(eligible_chargers))],
        dtype=np.uint64,
    )
    legacy: list[int] = []
    for _ in range(count):
        candidates = np.flatnonzero(available)
        proposed = np.minimum(current[:, None], distances[:, candidates])
        mean = proposed.mean(axis=0)
        p90 = np.quantile(proposed, 0.90, axis=0)
        maximum = proposed.max(axis=0)
        local = min(
            range(len(candidates)),
            key=lambda index: (
                float(mean[index] + 0.25 * p90[index] + 0.10 * maximum[index]),
                int(tie[candidates[index]]),
                int(candidates[index]),
            ),
        )
        chosen = int(candidates[local])
        legacy.append(chosen)
        available[chosen] = False
        current = np.minimum(current, distances[:, chosen])

    deltas = np.asarray(eligible_deltas_s, dtype=float)
    road_positions = np.asarray(road_time_selected_positions, dtype=int)
    legacy_positions = np.asarray(legacy, dtype=int)

    def delta_summary(positions: np.ndarray) -> dict[str, float]:
        nearest = deltas[:, positions].min(axis=1)
        return {
            "mean_s": float(nearest.mean()),
            "p95_s": float(np.quantile(nearest, 0.95)),
            "max_s": float(nearest.max()),
            "objective_mean_plus_0.25_p90_plus_0.10_max_s": float(
                nearest.mean()
                + 0.25 * np.quantile(nearest, 0.90)
                + 0.10 * nearest.max()
            ),
        }

    return {
        "role": "pilot_report_only_retired_selector_never_used_for_generation",
        "legacy_emulation": "active_customer_haversine_greedy_mean_p90_max_v1",
        "road_time_delta": delta_summary(road_positions),
        "haversine_legacy_road_time_delta": delta_summary(legacy_positions),
        "selected_roster_overlap_count": len(set(road_positions) & set(legacy_positions)),
        "selected_count": int(count),
    }



def _active_customer_charger_diagnostic(
    customers: pd.DataFrame,
    chargers: pd.DataFrame,
) -> dict[str, Any]:
    distances = _haversine_km(
        customers["location_lon"].to_numpy(dtype=float),
        customers["location_lat"].to_numpy(dtype=float),
        chargers["resolved_longitude"].to_numpy(dtype=float),
        chargers["resolved_latitude"].to_numpy(dtype=float),
    )
    nearest = distances.min(axis=1)
    return {
        "semantics": "post-selection QA only; not used to choose charging stations",
        "mean_active_customer_to_nearest_selected_charger_km": float(nearest.mean()),
        "p90_active_customer_to_nearest_selected_charger_km": float(
            np.quantile(nearest, 0.90)
        ),
        "max_active_customer_to_nearest_selected_charger_km": float(nearest.max()),
    }


def _resolve_charging_power(
    chargers: pd.DataFrame,
    *,
    profile: Mapping[str, Any],
    generation_mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = chargers.copy()
    charging = profile["charging"]
    mode_caps = {
        "dc_fast": float(charging["dc_vehicle_cap_kw"]),
        "ac_level2": float(charging["ac_l2_vehicle_cap_kw"]),
    }
    observed = pd.to_numeric(result["station_power_kw"], errors="coerce")
    mode_medians = {
        str(key): float(value)
        for key, value in charging.get("national_mode_medians_kw", {}).items()
    }
    resolved: list[float] = []
    sources: list[str] = []
    for mode, value in zip(result["reference_charge_mode"].astype(str), observed):
        if mode not in mode_caps:
            raise ValueError(f"Unsupported charger reference mode: {mode!r}")
        cap = mode_caps[mode]
        if pd.notna(value) and float(value) > 0.0:
            resolved.append(min(float(value), cap))
            sources.append("reported_station_power_capped_by_vehicle")
        elif mode in mode_medians and pd.notna(mode_medians[mode]):
            resolved.append(min(float(mode_medians[mode]), cap))
            sources.append("national_mode_median_capped_by_vehicle")
        else:
            raise ValueError(
                f"NATIONAL_MODE_MEDIAN_UNAVAILABLE: no reported station power and no "
                f"frozen national median for {mode}; generation is blocked"
            )
    result["effective_charging_power_kw"] = np.asarray(resolved, dtype=np.float32)
    result["effective_charging_power_source"] = sources
    return result, {
        "reported_power_count": int(sum(source.startswith("reported") for source in sources)),
        "national_mode_median_count": int(
            sum(source.startswith("national_mode") for source in sources)
        ),
        "vehicle_mode_cap_fallback_count": 0,
        "generation_mode": generation_mode,
        "frozen_national_mode_power_medians_kw": {
            str(key): float(value) for key, value in mode_medians.items() if pd.notna(value)
        },
    }


def _common_terminal_record(
    row: Mapping[str, Any],
    *,
    terminal_index: int,
    terminal_kind: str,
    source_id: str,
    longitude: float,
    latitude: float,
) -> dict[str, Any]:
    return {
        "terminal_index": int(terminal_index),
        "terminal_kind": terminal_kind,
        "source_id": source_id,
        "longitude": float(longitude),
        "latitude": float(latitude),
        "physical_edge_id": str(row["physical_edge_id"]),
        "directed_projection_offsets": str(row["directed_projection_offsets"]),
        "connector_length_m": float(row["connector_length_m"]),
        "road_projection_node_id": str(row["road_projection_node_id"]),
        "access_node_id": str(
            row.get("service_access_node_id", row.get("facility_access_node_id", ""))
        ),
        "anchor_scc_id": str(row["anchor_scc_id"]),
    }



def depot_candidate_order(
    depots: pd.DataFrame,
    *,
    seed: int,
    track: str,
) -> tuple[list[pd.Series], dict[str, Any]]:
    """Return canonical access points with the legacy choice ranked first."""

    if track not in {"strict", "practical"}:
        raise ValueError("depot_track must be strict or practical")
    frame = depots.copy()
    if track == "strict":
        frame = frame.loc[frame["strict_depot_candidate_eligible"].astype(bool)].copy()
    else:
        frame = frame.loc[
            frame["strict_depot_candidate_eligible"].astype(bool)
            | frame["optional_depot_candidate_eligible"].astype(bool)
        ].copy()
    if frame.empty:
        raise ValueError(f"No eligible {track} depot candidates")
    if "facility_group_id" not in frame:
        frame["facility_group_id"] = frame["candidate_id"].astype(str)
        grouping_source = "candidate_singleton_fallback"
    else:
        frame["facility_group_id"] = frame["facility_group_id"].astype(str)
        grouping_source = "cle_physical_facility_group"
    group_ids = sorted(map(str, frame["facility_group_id"].unique()))
    rng = np.random.default_rng(seed)
    legacy_group_id = str(group_ids[int(rng.integers(len(group_ids)))])
    remaining = sorted(
        (value for value in group_ids if value != legacy_group_id),
        key=lambda value: (
            derive_seed(seed, "depot_group_candidate", value),
            value,
        ),
    )
    ordered_groups = [legacy_group_id, *remaining]
    result: list[pd.Series] = []
    group_details: dict[str, dict[str, Any]] = {}
    for group_id in ordered_groups:
        group = frame.loc[frame["facility_group_id"].eq(group_id)].copy()
        tier_a = group.loc[group["strict_depot_candidate_eligible"].astype(bool)].copy()
        access_pool = tier_a if not tier_a.empty else group
        access_pool["_rank"] = [
            derive_seed(seed, "depot_access_point", candidate_id)
            for candidate_id in access_pool["candidate_id"].astype(str)
        ]
        result.append(
            access_pool.sort_values(["_rank", "candidate_id"], kind="stable").iloc[0]
        )
        group_details[group_id] = {
            "selected_group_candidate_count": len(group),
            "selected_group_tier_a_count": len(tier_a),
            "selected_access_tier_a_preferred": bool(len(tier_a)),
        }
    return result, {
        "policy": "uniform_physical_group_then_tier_a_preferred_access_v1",
        "track": track,
        "grouping_source": grouping_source,
        "eligible_group_count": len(group_ids),
        "legacy_first_facility_group_id": legacy_group_id,
        "candidate_group_order": ordered_groups,
        "group_details": group_details,
        "area_used_as_hard_gate": False,
        "area_used_as_sampling_weight": False,
    }


def _select_depot_group(
    depots: pd.DataFrame,
    *,
    seed: int,
    track: str,
) -> tuple[pd.Series, dict[str, Any]]:
    """Select the first candidate under the frozen legacy-compatible rank."""

    ordered, metadata = depot_candidate_order(depots, seed=seed, track=track)
    depot = ordered[0]
    group_id = str(depot.get("facility_group_id", depot["candidate_id"]))
    return depot, {
        **metadata,
        "selected_facility_group_id": group_id,
        **metadata["group_details"][group_id],
    }


def _star_terminal_index(depot: pd.Series, customers: pd.DataFrame) -> pd.DataFrame:
    records = [
        _common_terminal_record(
            depot,
            terminal_index=0,
            terminal_kind="depot",
            source_id=str(depot["candidate_id"]),
            longitude=float(depot["longitude"]),
            latitude=float(depot["latitude"]),
        )
    ]
    for position, row in customers.reset_index(drop=True).iterrows():
        records.append(
            _common_terminal_record(
                row,
                terminal_index=position + 1,
                terminal_kind="customer",
                source_id=str(row["latent_service_location_id"]),
                longitude=float(row["location_lon"]),
                latitude=float(row["location_lat"]),
            )
        )
    return pd.DataFrame.from_records(records)


def _charger_roster_terminal_index(
    depot: pd.Series,
    customers: pd.DataFrame,
    chargers: pd.DataFrame,
) -> pd.DataFrame:
    records = _star_terminal_index(depot, customers).to_dict("records")
    start = 1 + len(customers)
    for position, row in chargers.reset_index(drop=True).iterrows():
        records.append(
            _common_terminal_record(
                row,
                terminal_index=start + position,
                terminal_kind="charging_station",
                source_id=str(row["charger_id"]),
                longitude=float(row["resolved_longitude"]),
                latitude=float(row["resolved_latitude"]),
            )
        )
    return pd.DataFrame.from_records(records)


def _connectivity_quarantine(
    frame: pd.DataFrame,
    star: DepotTerminalStar,
    *,
    id_column: str,
    terminal_kind: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return the exact eligible mask and deterministic bad-terminal ledger."""

    if len(star.connectivity_eligible) != len(frame) + 1:
        raise ValueError("Depot-star connectivity mask does not align with its roster")
    eligible = star.connectivity_eligible[1:].copy()
    ledger: list[dict[str, Any]] = []
    masks = {
        "node_unreachable_from_depot": star.node_outbound_reachable[1:],
        "node_cannot_return_to_depot": star.node_return_reachable[1:],
        "turn_unreachable_from_depot": star.turn_outbound_reachable[1:],
        "turn_cannot_return_to_depot": star.turn_return_reachable[1:],
    }
    for position in np.flatnonzero(~eligible):
        row = frame.iloc[int(position)]
        reasons = [name for name, values in masks.items() if not bool(values[position])]
        ledger.append(
            {
                "terminal_kind": terminal_kind,
                "source_id": str(row[id_column]),
                "reason_codes": reasons,
                "physical_edge_id": str(row["physical_edge_id"]),
                "anchor_scc_id": str(row.get("anchor_scc_id", "")),
                "directed_edge_ref_count": int(row["directed_edge_ref_count"]),
                "directed_projection_offsets": str(row["directed_projection_offsets"]),
            }
        )
    return eligible, ledger


def _road_time_adjacency(
    adjacency: pd.DataFrame,
    network: PhysicalRoadNetwork,
) -> pd.DataFrame:
    if adjacency.empty:
        return adjacency.assign(crossing_time_s=pd.Series(dtype=float))
    edge_times = network.edges[
        ["edge_u", "edge_v", "edge_key", "edge_travel_time_s"]
    ].copy()
    for column in ("edge_u", "edge_v", "edge_key"):
        edge_times[column] = edge_times[column].astype(str)
        adjacency[column] = adjacency[column].astype(str)
    result = adjacency.merge(
        edge_times,
        on=["edge_u", "edge_v", "edge_key"],
        how="left",
        validate="many_to_one",
    )
    if result["edge_travel_time_s"].isna().any():
        missing = int(result["edge_travel_time_s"].isna().sum())
        raise ValueError(f"Community adjacency references {missing} absent road-state edges")
    result["crossing_time_s"] = result["edge_travel_time_s"].astype(float)
    return result


def encode_structure_source_id(source_ids: list[str] | tuple[str, ...]) -> str:
    values = list(map(str, source_ids))
    return values[0] if len(values) == 1 else json.dumps(values, separators=(",", ":"))


def decode_structure_source_id(value: Any) -> list[str]:
    text = str(value)
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("Frozen structure source list is empty or malformed")
        return list(map(str, parsed))
    return [text]


def capacity_contract_fingerprint(
    *,
    family: Mapping[str, Any],
    selected_depot_id: str,
    selected_structure_source_ids: list[str] | tuple[str, ...],
    required_decile_counts: list[int] | tuple[int, ...],
    available_decile_counts: list[int] | tuple[int, ...],
) -> str:
    payload = {
        "contract_id": JOINT_SUPPORT_CONTRACT_ID,
        "family_id": str(family["family_id"]),
        "city_slug": str(family["city_slug"]),
        "track_id": str(family["track_id"]),
        "day_type": str(family["day_type"]),
        "customer_pool": str(family["customer_pool"]),
        "parent_customer_count": int(family["parent_customer_count"]),
        "depot_seed": int(family["depot_seed"]),
        "customer_superset_seed": int(family["customer_superset_seed"]),
        "road_state_seed": int(family["road_state_seed"]),
        "selected_depot_id": str(selected_depot_id),
        "selected_structure_source_ids": sorted(
            map(str, selected_structure_source_ids)
        ),
        "required_decile_counts": list(map(int, required_decile_counts)),
        "available_decile_counts": list(map(int, available_decile_counts)),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "ccf_" + hashlib.blake2b(encoded, digest_size=16).hexdigest()


def _frozen_or_default_depot(
    depots: pd.DataFrame,
    *,
    family: Mapping[str, Any],
    depot_track: str,
) -> tuple[pd.Series, dict[str, Any]]:
    ordered, metadata = depot_candidate_order(
        depots,
        seed=int(family["depot_seed"]),
        track=depot_track,
    )
    selected_id = family.get("selected_depot_id")
    if selected_id is None or (
        isinstance(selected_id, float) and np.isnan(selected_id)
    ):
        depot = ordered[0]
    else:
        selected_text = str(selected_id)
        depot = next(
            (
                candidate
                for candidate in ordered
                if str(candidate["candidate_id"]) == selected_text
            ),
            None,
        )
        if depot is None:
            raise JointSupportConsistencyError(
                f"frozen depot {selected_text!r} is no longer eligible",
                capacity_contract_fingerprint=str(
                    family.get("capacity_contract_fingerprint") or "missing"
                ),
            )
    group_id = str(depot.get("facility_group_id", depot["candidate_id"]))
    return depot, {
        **metadata,
        "selected_facility_group_id": group_id,
        **metadata["group_details"][group_id],
    }


def _prepare_customer_territory(
    cle: PortableCLE,
    *,
    family: Mapping[str, Any],
    depot: pd.Series,
    structure_metadata: Mapping[str, Any],
    customer_split_path: str,
    profile: Mapping[str, Any],
    network: PhysicalRoadNetwork,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    customer_count = int(family["parent_customer_count"])
    customers = cle.read_service_locations().copy()
    split = pd.read_parquet(customer_split_path)
    split_columns = [
        "latent_service_location_id",
        "community_id",
        "customer_pool",
        "road_connectivity_subgroup",
    ]
    missing = set(split_columns) - set(split.columns)
    if missing:
        raise ValueError(f"Customer split ledger is missing columns: {sorted(missing)}")
    customers = customers.merge(
        split[split_columns],
        on="latent_service_location_id",
        how="inner",
        validate="one_to_one",
    )
    requested_pool = str(family["customer_pool"])
    if requested_pool in {"train", "heldout"}:
        customers = customers.loc[customers["customer_pool"].eq(requested_pool)].copy()
    elif requested_pool != "all_release_eligible":
        raise ValueError(f"Unsupported customer pool: {requested_pool!r}")
    customers = customers.reset_index(drop=True)
    split_pool_count = len(customers)
    customer_roster_fingerprint = roster_fingerprint(
        customers["latent_service_location_id"],
        depot_id=depot["candidate_id"],
        terminal_kind="customer",
    )
    star = network.route_depot_star(_star_terminal_index(depot, customers))
    connectivity_mask, quarantine = _connectivity_quarantine(
        customers,
        star,
        id_column="latent_service_location_id",
        terminal_kind="customer",
    )
    customers["depot_running_time_s"] = star.outbound_time_s[1:]
    customers["depot_return_time_s"] = star.inbound_time_s[1:]
    customers["depot_outbound_distance_km"] = star.outbound_distance_km[1:]
    customers["depot_return_distance_km"] = star.inbound_distance_km[1:]
    customers = customers.loc[connectivity_mask].reset_index(drop=True)
    if len(customers) < customer_count:
        raise TerminalConnectivityError(
            "NONRETRYABLE_TERMINAL_CONNECTIVITY: customer roster has "
            f"{len(customers)} bidirectionally node/turn-reachable candidates after "
            f"quarantining {len(quarantine)}; N={customer_count}",
            roster_fingerprint=customer_roster_fingerprint,
        )
    t_env = float(structure_metadata["source_t_env_s"])
    before_energy = customers.loc[customers["depot_running_time_s"].le(t_env)].copy()
    specific_energy = float(profile["energy"]["specific_energy_consumption_kwh_per_km"])
    battery = float(profile["energy"]["battery_capacity_kwh"])
    roundtrip_energy = (
        before_energy["depot_outbound_distance_km"]
        + before_energy["depot_return_distance_km"]
    ) * specific_energy
    territory = before_energy.loc[roundtrip_energy.le(battery + 1e-9)].copy()
    territory["direct_roundtrip_energy_kwh"] = roundtrip_energy.loc[territory.index]
    territory["radial_decile"] = np.searchsorted(
        np.asarray(structure_metadata["source_radial_decile_edges_s"], dtype=float)[1:-1],
        territory["depot_running_time_s"].to_numpy(dtype=float),
        side="right",
    ).astype(np.int8)
    if len(territory) < customer_count:
        raise SpatialActivationError(
            "TERRITORY_TOO_SMALL",
            f"structure_source_ids={structure_metadata['structure_source_ids']}, "
            f"split={split_pool_count}, time_envelope={len(before_energy)}, "
            f"direct_energy={len(territory)}, N={customer_count}",
        )
    return territory, {
        "requested_pool": requested_pool,
        "split_pool_count": split_pool_count,
        "connectivity_eligible_count": len(customers),
        "connectivity_quarantine_count": len(quarantine),
        "connectivity_quarantine_ledger": quarantine,
        "depot_star_report": star.report,
        "customer_roster_fingerprint": customer_roster_fingerprint,
        "time_envelope_count": len(before_energy),
        "territory_count": len(territory),
    }


def assess_joint_spatial_support_pair(
    cle: PortableCLE,
    *,
    family: Mapping[str, Any],
    selected_depot_id: str,
    selected_structure_source_ids: list[str] | tuple[str, ...],
    customer_split_path: str,
    community_adjacency_path: str,
    profile: Mapping[str, Any],
    network: PhysicalRoadNetwork,
    amazon: AmazonStage2Artifacts,
    community_adjacency_cache: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Run C3-A and C3-B for one frozen depot x source candidate pair."""

    frozen_family = {
        **dict(family),
        "selected_depot_id": str(selected_depot_id),
    }
    source_pool = amazon.pool_for_track(str(family["track_id"]))
    structure_targets, structure_metadata = amazon.structure_source(
        day_type=str(family["day_type"]),
        customer_count=int(family["parent_customer_count"]),
        seed=int(family["customer_superset_seed"]),
        pool=source_pool,
        track_id=str(family["track_id"]),
        allow_composite=str(family["parent_scale_id"]) == "cus2000",
        selected_source_ids=list(selected_structure_source_ids),
    )
    spatial_cfg = profile.get("stage2_spatial", {})
    depot, _ = _frozen_or_default_depot(
        cle.read_depots().reset_index(drop=True),
        family=frozen_family,
        depot_track=str(spatial_cfg.get("depot_track", "practical")),
    )
    territory, territory_report = _prepare_customer_territory(
        cle,
        family=family,
        depot=depot,
        structure_metadata=structure_metadata,
        customer_split_path=customer_split_path,
        profile=profile,
        network=network,
    )
    _, quota_metadata, _ = radial_decile_support_contract(
        territory,
        structure_targets,
        customer_count=int(family["parent_customer_count"]),
        seed=int(family["customer_superset_seed"]),
    )
    adjacency_cache_key = f"{cle.city_slug}:{family['day_type']}"
    adjacency = (
        community_adjacency_cache.get(adjacency_cache_key)
        if community_adjacency_cache is not None
        else None
    )
    if adjacency is None:
        adjacency = _road_time_adjacency(
            pd.read_parquet(community_adjacency_path), network
        )
        if community_adjacency_cache is not None:
            community_adjacency_cache[adjacency_cache_key] = adjacency
    activation = activate_spatial_customers(
        territory,
        adjacency,
        structure_targets,
        customer_count=int(family["parent_customer_count"]),
        seed=int(family["customer_superset_seed"]),
        region_redraw_cap=int(spatial_cfg.get("region_redraw_cap", 3)),
    )
    required = list(map(int, quota_metadata["required_decile_counts"]))
    available = list(map(int, quota_metadata["available_decile_counts"]))
    fingerprint = capacity_contract_fingerprint(
        family=family,
        selected_depot_id=str(depot["candidate_id"]),
        selected_structure_source_ids=list(selected_structure_source_ids),
        required_decile_counts=required,
        available_decile_counts=available,
    )
    return {
        "joint_support_contract_id": JOINT_SUPPORT_CONTRACT_ID,
        "aggregate_gate_passed": True,
        "exact_gate_passed": True,
        "selected_depot_id": str(depot["candidate_id"]),
        "selected_structure_source_id": encode_structure_source_id(
            selected_structure_source_ids
        ),
        "required_decile_counts": required,
        "available_decile_counts": available,
        "capacity_contract_fingerprint": fingerprint,
        "territory": territory_report,
        "activation_region_attempts_used": int(
            activation.metadata["region_attempts_used"]
        ),
    }


def select_family_terminals_v2(
    cle: PortableCLE,
    *,
    family: Mapping[str, Any],
    customer_split_path: str,
    community_adjacency_path: str,
    profile: Mapping[str, Any],
    network: PhysicalRoadNetwork,
    amazon: AmazonStage2Artifacts,
    community_adjacency_cache: dict[str, pd.DataFrame] | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Select a depot, one Amazon-structured customer parent and relevant CSs."""

    performance_profile: list[dict[str, Any]] = []

    def begin(stage: str, **details: Any) -> tuple[float, float]:
        if progress_callback is not None:
            progress_callback(stage, {"status": "started", **details})
        return time.perf_counter(), time.process_time()

    def finish(
        stage: str,
        started: tuple[float, float],
        **details: Any,
    ) -> None:
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        event = {
            "stage": stage,
            "status": "completed",
            "wall_seconds": time.perf_counter() - started[0],
            "cpu_seconds": time.process_time() - started[1],
            "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
            **details,
        }
        performance_profile.append(event)
        if progress_callback is not None:
            progress_callback(stage, event)

    stage_started = begin("structure_and_depot_preflight")
    family_seed = int(family["family_seed"])
    customer_count = int(family["parent_customer_count"])
    charger_count = int(family["parent_charging_station_count"])
    day_type = str(family.get("day_type") or sample_day_type(profile, family_seed))
    source_pool = amazon.pool_for_track(str(family["track_id"]))
    allow_composite = str(family["parent_scale_id"]) == "cus2000"
    frozen_source_value = family.get("selected_structure_source_id")
    frozen_source_ids = (
        None
        if frozen_source_value is None
        or (isinstance(frozen_source_value, float) and np.isnan(frozen_source_value))
        else decode_structure_source_id(frozen_source_value)
    )
    structure_targets, structure_metadata = amazon.structure_source(
        day_type=day_type,
        customer_count=customer_count,
        seed=int(family["customer_superset_seed"]),
        pool=source_pool,
        track_id=str(family["track_id"]),
        allow_composite=allow_composite,
        selected_source_ids=frozen_source_ids,
    )
    spatial_cfg = profile.get("stage2_spatial", {})
    depot_track = str(spatial_cfg.get("depot_track", "practical"))
    depots = cle.read_depots().reset_index(drop=True)
    depot, depot_metadata = _frozen_or_default_depot(
        depots,
        family=family,
        depot_track=depot_track,
    )
    finish(
        "structure_and_depot_preflight", stage_started, depot_count=len(depots)
    )

    stage_started = begin("customer_preflight")
    territory, territory_report = _prepare_customer_territory(
        cle,
        family=family,
        depot=depot,
        structure_metadata=structure_metadata,
        customer_split_path=customer_split_path,
        profile=profile,
        network=network,
    )
    finish(
        "customer_preflight",
        stage_started,
        split_pool_count=territory_report["split_pool_count"],
        connectivity_eligible_count=territory_report["connectivity_eligible_count"],
        territory_count=len(territory),
    )
    stage_started = begin("customer_spatial_activation", territory_count=len(territory))
    amazon_nn = pd.to_numeric(
        structure_targets.get("amazon_route_nearest_neighbor_time_s"), errors="coerce"
    ).dropna()
    source_route_ids = set(structure_targets["route_id"].astype(str))
    amazon_pair_reference = amazon.route_spatial_reference.loc[
        amazon.route_spatial_reference["route_id"].astype(str).isin(source_route_ids)
    ]
    adjacency_cache_key = f"{cle.city_slug}:{day_type}"
    adjacency = (
        community_adjacency_cache.get(adjacency_cache_key)
        if community_adjacency_cache is not None
        else None
    )
    if adjacency is None:
        adjacency = _road_time_adjacency(
            pd.read_parquet(community_adjacency_path), network
        )
        if community_adjacency_cache is not None:
            community_adjacency_cache[adjacency_cache_key] = adjacency
    planned_fingerprint_raw = family.get("capacity_contract_fingerprint")
    planned_fingerprint = (
        None
        if planned_fingerprint_raw is None
        or (
            isinstance(planned_fingerprint_raw, float)
            and np.isnan(planned_fingerprint_raw)
        )
        else str(planned_fingerprint_raw)
    )
    try:
        activation = activate_spatial_customers(
            territory,
            adjacency,
            structure_targets,
            customer_count=customer_count,
            seed=int(family["customer_superset_seed"]),
            region_redraw_cap=int(spatial_cfg.get("region_redraw_cap", 3)),
            progress_callback=(
                (
                    lambda substage, details: progress_callback(
                        f"customer_spatial_activation.{substage}", details
                    )
                )
                if progress_callback is not None
                else None
            ),
        )
    except SpatialActivationError as error:
        if planned_fingerprint is not None:
            raise JointSupportConsistencyError(
                f"approved pair failed exact activation with {error.code}",
                capacity_contract_fingerprint=planned_fingerprint,
            ) from error
        raise
    quota_report = activation.metadata["quota"]
    replay_fingerprint = capacity_contract_fingerprint(
        family=family,
        selected_depot_id=str(depot["candidate_id"]),
        selected_structure_source_ids=list(
            structure_metadata["structure_source_ids"]
        ),
        required_decile_counts=list(quota_report["required_decile_counts"]),
        available_decile_counts=list(quota_report["available_decile_counts"]),
    )
    if planned_fingerprint is not None and replay_fingerprint != planned_fingerprint:
        raise JointSupportConsistencyError(
            "approved pair replayed with a different capacity contract "
            f"(planned={planned_fingerprint}, actual={replay_fingerprint})",
            capacity_contract_fingerprint=replay_fingerprint,
        )
    selected_customers = activation.customers.reset_index(drop=True)
    route_quota_used = {
        str(route_id): int(count)
        for route_id, count in sorted(
            selected_customers["structure_route_id"]
            .astype(str)
            .value_counts()
            .items()
        )
    }
    if (
        any(count <= 0 for count in route_quota_used.values())
        or sum(route_quota_used.values()) != customer_count
    ):
        raise RuntimeError("positive source-route quotas do not sum to customer_count")
    structure_metadata["route_count_used"] = len(route_quota_used)
    structure_metadata["route_quota_used"] = route_quota_used
    structure_metadata["route_count_used_semantics"] = (
        "number_of_distinct_source_routes_with_positive_customer_quota"
    )
    finish(
        "customer_spatial_activation",
        stage_started,
        selected_customer_count=len(selected_customers),
        adjacency_row_count=len(adjacency),
    )

    stage_started = begin("charger_preflight")
    chargers = cle.read_chargers().reset_index(drop=True)
    charger_candidate_roster_count = len(chargers)
    charger_roster_fingerprint = roster_fingerprint(
        chargers["charger_id"],
        depot_id=depot["candidate_id"],
        terminal_kind="charging_station",
    )
    charger_star = network.route_depot_star(
        _charger_roster_terminal_index(depot, selected_customers.iloc[:0], chargers)
    )
    charger_connectivity_mask, charger_connectivity_quarantine = (
        _connectivity_quarantine(
            chargers,
            charger_star,
            id_column="charger_id",
            terminal_kind="charging_station",
        )
    )
    chargers = chargers.loc[charger_connectivity_mask].reset_index(drop=True)
    if len(chargers) < charger_count:
        raise TerminalConnectivityError(
            "NONRETRYABLE_TERMINAL_CONNECTIVITY: charger roster has "
            f"{len(chargers)} bidirectionally node/turn-reachable candidates after "
            f"quarantining {len(charger_connectivity_quarantine)}; K={charger_count}",
            roster_fingerprint=charger_roster_fingerprint,
        )
    finish(
        "charger_preflight",
        stage_started,
        charger_input_count=charger_candidate_roster_count,
        charger_connectivity_eligible_count=len(chargers),
    )
    stage_started = begin(
        "charger_roster_batched_dijkstra",
        terminal_count=1 + len(selected_customers) + len(chargers),
    )
    roster_terminals = _charger_roster_terminal_index(depot, selected_customers, chargers)
    roster_matrices = network.route_terminals(roster_terminals)
    finish(
        "charger_roster_batched_dijkstra",
        stage_started,
        routing_workload=roster_matrices.report,
    )
    stage_started = begin("energy_closure_and_charger_selection")
    customer_indices = np.arange(1, 1 + customer_count, dtype=np.int32)
    charger_indices = np.arange(
        1 + customer_count,
        1 + customer_count + len(chargers),
        dtype=np.int32,
    )
    infrastructure_indices = np.concatenate(
        [np.asarray([0], dtype=np.int32), charger_indices]
    )
    specific_energy = float(profile["energy"]["specific_energy_consumption_kwh_per_km"])
    infrastructure_energy = roster_matrices.running_time_path_distance_km[
        np.ix_(infrastructure_indices, infrastructure_indices)
    ].astype(float) * specific_energy
    communicating = battery_feasible_communicating_mask(
        infrastructure_energy,
        battery_capacity_kwh=float(profile["energy"]["battery_capacity_kwh"]),
    )
    eligible_positions = np.flatnonzero(communicating[1:])
    eligible_terminal_indices = charger_indices[eligible_positions]
    eligible_deltas = road_time_replacement_deltas(
        roster_matrices.running_time_shortest_matrix_s,
        customer_indices=customer_indices,
        charger_indices=eligible_terminal_indices,
    )
    chosen_eligible_positions, charger_metadata = select_road_time_charger_indices(
        eligible_deltas,
        count=charger_count,
        seed=int(family["charger_seed"]),
    )
    eligible_chargers = chargers.iloc[eligible_positions].copy().reset_index(drop=True)
    charger_metadata["road_time_vs_haversine_legacy_audit"] = (
        haversine_legacy_comparison_audit(
            selected_customers,
            eligible_chargers,
            eligible_deltas,
            road_time_selected_positions=chosen_eligible_positions,
            count=charger_count,
            seed=int(family["charger_seed"]),
        )
    )
    selected_positions = eligible_positions[chosen_eligible_positions]
    selected_chargers = chargers.iloc[selected_positions].copy().reset_index(drop=True)
    selected_chargers["charger_selection_rank"] = np.arange(
        1, len(selected_chargers) + 1, dtype=np.int32
    )
    charger_metadata.update(
        {
            "candidate_roster_count": charger_candidate_roster_count,
            "connectivity_eligible_roster_count": len(chargers),
            "bidirectional_energy_eligible_count": int(len(eligible_positions)),
            "eligible_charger_ids": chargers.iloc[eligible_positions]["charger_id"]
            .astype(str)
            .tolist(),
            "selected_roster_positions": selected_positions.astype(int).tolist(),
        }
    )
    selected_chargers, power_metadata = _resolve_charging_power(
        selected_chargers,
        profile=profile,
        generation_mode=cle.mode,
    )
    charger_active_customer_diagnostic = _active_customer_charger_diagnostic(
        selected_customers,
        selected_chargers,
    )
    finish(
        "energy_closure_and_charger_selection",
        stage_started,
        energy_eligible_charger_count=len(eligible_positions),
        selected_charger_count=len(selected_chargers),
    )

    stage_started = begin("terminal_index_construction")
    records = [
        {
            **_common_terminal_record(
                depot,
                terminal_index=0,
                terminal_kind="depot",
                source_id=str(depot["candidate_id"]),
                longitude=float(depot["longitude"]),
                latitude=float(depot["latitude"]),
            ),
            "parent_customer_position": pd.NA,
            "community_id": pd.NA,
            "sampling_cluster_id": pd.NA,
            "structure_route_id": pd.NA,
            "activation_decile": pd.NA,
            "service_location_type": pd.NA,
            "residential_unit_band": pd.NA,
            "residential_units": pd.NA,
            "depot_running_time_s": 0.0,
            "charger_selection_rank": pd.NA,
            "reference_charge_mode": pd.NA,
            "effective_charging_power_kw": pd.NA,
            "effective_charging_power_source": pd.NA,
        }
    ]
    for position, row in selected_customers.iterrows():
        records.append(
            {
                **_common_terminal_record(
                    row,
                    terminal_index=1 + position,
                    terminal_kind="customer",
                    source_id=str(row["latent_service_location_id"]),
                    longitude=float(row["location_lon"]),
                    latitude=float(row["location_lat"]),
                ),
                "parent_customer_position": position,
                "community_id": str(row["community_id"]),
                "sampling_cluster_id": str(row["sampling_cluster_id"]),
                "structure_route_id": str(row["structure_route_id"]),
                "activation_decile": int(row["activation_decile"]),
                "service_location_type": str(row["service_location_type"]),
                "residential_unit_band": str(row["residential_unit_band"]),
                "residential_units": int(row["residential_units"]),
                "depot_running_time_s": float(row["depot_running_time_s"]),
                "charger_selection_rank": pd.NA,
                "reference_charge_mode": pd.NA,
                "effective_charging_power_kw": pd.NA,
                "effective_charging_power_source": pd.NA,
            }
        )
    for rank, row in selected_chargers.iterrows():
        records.append(
            {
                **_common_terminal_record(
                    row,
                    terminal_index=1 + customer_count + rank,
                    terminal_kind="charging_station",
                    source_id=str(row["charger_id"]),
                    longitude=float(row["resolved_longitude"]),
                    latitude=float(row["resolved_latitude"]),
                ),
                "parent_customer_position": pd.NA,
                "community_id": pd.NA,
                "sampling_cluster_id": pd.NA,
                "structure_route_id": pd.NA,
                "activation_decile": pd.NA,
                "service_location_type": pd.NA,
                "residential_unit_band": pd.NA,
                "residential_units": pd.NA,
                "depot_running_time_s": pd.NA,
                "charger_selection_rank": rank + 1,
                "reference_charge_mode": str(row["reference_charge_mode"]),
                "effective_charging_power_kw": float(row["effective_charging_power_kw"]),
                "effective_charging_power_source": str(row["effective_charging_power_source"]),
            }
        )
    terminal_index = pd.DataFrame.from_records(records)
    finish(
        "terminal_index_construction", stage_started,
        terminal_count=len(terminal_index),
    )
    metadata = {
        "schema": "cle_evrptw_family_terminal_selection_v3",
        "family_id": str(family["family_id"]),
        "city_slug": cle.city_slug,
        "day_type": day_type,
        "day_type_source": "preallocated_family_slot",
        "customer_pool": territory_report["requested_pool"],
        "parent_customer_count": customer_count,
        "parent_charging_station_count": charger_count,
        "selected_depot_id": str(depot["candidate_id"]),
        "selected_depot_evidence_tier": str(depot["evidence_tier"]),
        "depot_selection": depot_metadata,
        "terminal_connectivity": {
            "schema": "cle_evrptw_terminal_connectivity_quarantine_v2",
            "policy": "depot_bidirectional_node_and_canonical_turn_topology_v1",
            "customer_input_count": territory_report["split_pool_count"],
            "customer_eligible_count": territory_report["connectivity_eligible_count"],
            "customer_quarantined_count": territory_report[
                "connectivity_quarantine_count"
            ],
            "customer_quarantine_ledger": territory_report[
                "connectivity_quarantine_ledger"
            ],
            "customer_depot_star": territory_report["depot_star_report"],
            "charger_input_count": charger_candidate_roster_count,
            "charger_eligible_count": int(charger_connectivity_mask.sum()),
            "charger_quarantined_count": len(charger_connectivity_quarantine),
            "charger_quarantine_ledger": charger_connectivity_quarantine,
            "charger_depot_star": charger_star.report,
            "customer_split_semantics": (
                "retain frozen C0 train/heldout assignment; apply only a "
                "family/depot connectivity eligibility mask"
            ),
            "applied_before_territory_and_full_terminal_closure": True,
            "applied_before_family_activation_territory_capacity_customer_sampling_and_materialization": (
                True
            ),
            "post_mask_capacity_failure_semantics": (
                "nonretryable fixed-roster failure; changing family seed cannot bypass"
            ),
        },
        "territory": {
            "policy": "amazon_per_source_q99_directed_network_time_and_direct_energy_screen_v2",
            "source_t_env_s": float(structure_metadata["source_t_env_s"]),
            "source_radial_decile_edges_s": structure_metadata[
                "source_radial_decile_edges_s"
            ],
            "split_legal_pool_count": territory_report["split_pool_count"],
            "time_envelope_pool_count": territory_report["time_envelope_count"],
            "energy_screen_pool_count": len(territory),
            "energy_screen_removed_count": territory_report["time_envelope_count"]
            - len(territory),
            "energy_screen_removed_share": (
                (
                    territory_report["time_envelope_count"] - len(territory)
                )
                / territory_report["time_envelope_count"]
                if territory_report["time_envelope_count"]
                else 0.0
            ),
            "energy_screen_semantics": "direct_depot_customer_depot_sufficient_condition",
            "pool_floor": 1.0,
            "territory_reserve_ratio": len(territory) / customer_count,
            "depot_star": territory_report["depot_star_report"],
        },
        "joint_spatial_support": {
            "joint_support_contract_id": JOINT_SUPPORT_CONTRACT_ID,
            "capacity_contract_fingerprint": replay_fingerprint,
            "planned_capacity_contract_fingerprint": planned_fingerprint,
            "replay_matches_plan": (
                planned_fingerprint is None
                or replay_fingerprint == planned_fingerprint
            ),
        },
        "amazon_structure_source": structure_metadata,
        "amazon_spatial_reference": {
            "nearest_neighbor_time_s": amazon_nn.astype(float).tolist(),
            "within_route_pairwise_time_p50_s": pd.to_numeric(
                amazon_pair_reference["within_route_pairwise_time_p50_s"], errors="coerce"
            ).dropna().astype(float).tolist(),
            "within_route_pairwise_time_p90_s": pd.to_numeric(
                amazon_pair_reference["within_route_pairwise_time_p90_s"], errors="coerce"
            ).dropna().astype(float).tolist(),
            "coordinate_space_only": True,
            "cross_route_centroid_separation_normative": False,
        },
        "spatial_activation": activation.metadata,
        "charger_selection": {
            **charger_metadata,
            "core_count": 0,
            "core_reason": "territory_direct_roundtrip_energy_screen_makes_core_empty",
            "fill_semantics": "active_community_and_depot_corridor_proxy_coverage",
        },
        "charger_active_customer_coverage_diagnostic": charger_active_customer_diagnostic,
        "charging_power_resolution": power_metadata,
        "performance_profile": performance_profile,
        "non_release_pilot": cle.non_release_pilot,
    }
    baseline_columns = [
        "latent_service_location_id",
        "community_id",
        "radial_decile",
        "depot_running_time_s",
    ]
    return terminal_index, metadata, activation.radial_baseline[baseline_columns]
