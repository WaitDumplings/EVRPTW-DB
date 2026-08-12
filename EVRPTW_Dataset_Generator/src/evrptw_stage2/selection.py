"""Deterministic family-level terminal activation for portable CLE packages."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .planning import derive_seed
from .reader import PortableCLE


def sample_day_type(profile: Mapping[str, Any], family_seed: int) -> str:
    cfg = profile["day_type"]
    weekday = int(cfg["weekday_weight"])
    weekend = int(cfg["weekend_weight"])
    rng = np.random.default_rng(derive_seed(family_seed, "day_type"))
    return "weekday" if int(rng.integers(weekday + weekend)) < weekday else "weekend"


def _select_customer_rows(
    customers: pd.DataFrame,
    *,
    count: int,
    day_type: str,
    profile: Mapping[str, Any],
    seed: int,
    original_pool_size: int,
    depot_catchment_radius_km: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(customers) < count:
        raise ValueError(f"Customer pool has {len(customers)} locations; requested {count}")
    cfg = profile["customer_activation"]
    rng = np.random.default_rng(seed)
    catchment_cfg = cfg["depot_catchment"]
    if "depot_haversine_km" not in customers.columns:
        raise ValueError("Customer catchment rows must include depot_haversine_km")
    target = float(
        np.clip(
            rng.lognormal(
                math.log(float(cfg["target_active_locations_per_community"])),
                float(cfg["target_lognormal_sigma"]),
            ),
            float(cfg["target_min"]),
            float(cfg["target_max"]),
        )
    )
    groups = (
        customers.groupby("community_id", sort=True)
        .size()
        .rename("eligible_location_count")
        .reset_index()
    )
    community_target = max(
        int(cfg["minimum_active_communities"]),
        math.ceil(count / max(target, 1.0)),
        math.ceil(math.log2(count + 1)),
    )
    community_target = min(community_target, len(groups), count)
    activity = rng.lognormal(
        0.0, float(cfg["community_activity_lognormal_sigma"]), size=len(groups)
    )
    group_weights = activity * groups["eligible_location_count"].to_numpy(dtype=float)
    group_mean_distance = (
        customers.groupby("community_id", sort=True)["depot_haversine_km"].mean().reindex(
            groups["community_id"]
        )
    )
    group_weights *= np.exp(
        -group_mean_distance.to_numpy(dtype=float) / float(catchment_cfg["distance_decay_km"])
    )
    available = list(range(len(groups)))
    selected_group_indices: list[int] = []
    selected_capacity = 0
    capacity_target = math.ceil(count * float(cfg["community_capacity_buffer"]))
    while available and (
        len(selected_group_indices) < community_target or selected_capacity < capacity_target
    ):
        probabilities = group_weights[np.asarray(available, dtype=int)]
        probabilities /= probabilities.sum()
        position = int(rng.choice(len(available), p=probabilities))
        group_index = int(available.pop(position))
        selected_group_indices.append(group_index)
        selected_capacity += int(groups.iloc[group_index]["eligible_location_count"])
    if selected_capacity < count:
        raise ValueError(
            f"Selected communities contain {selected_capacity} locations; requested {count}"
        )

    selected_communities = groups.iloc[selected_group_indices]["community_id"].astype(str)
    candidate_mask = customers["community_id"].astype(str).isin(set(selected_communities))
    candidates = customers.loc[candidate_mask].copy().reset_index(drop=True)
    units = pd.to_numeric(candidates["residential_units"], errors="coerce").fillna(1.0)
    units = units.clip(lower=1.0, upper=5000.0).to_numpy(dtype=float)
    per_unit_probability = float(cfg["per_unit_order_probability"][day_type])
    location_activation = 1.0 - np.power(1.0 - per_unit_probability, units)
    community_activity = {
        str(groups.iloc[index]["community_id"]): float(activity[index])
        for index in selected_group_indices
    }
    spatial_multiplier = candidates["community_id"].astype(str).map(community_activity).to_numpy()
    weights = np.maximum(location_activation * spatial_multiplier, 1e-12)

    # Preserve at least one active service location per selected community; the
    # remaining locations are sampled by unit-aware daily activation weight.
    chosen: list[int] = []
    if len(selected_communities) <= count:
        for community_id in selected_communities:
            indices = np.flatnonzero(candidates["community_id"].astype(str).eq(community_id))
            probabilities = weights[indices] / weights[indices].sum()
            chosen.append(int(rng.choice(indices, p=probabilities)))
    chosen = list(dict.fromkeys(chosen))
    remaining = count - len(chosen)
    residual_weights = weights.copy()
    if chosen:
        residual_weights[np.asarray(chosen, dtype=int)] = 0.0
    if remaining:
        residual_weights /= residual_weights.sum()
        chosen.extend(
            map(
                int,
                rng.choice(len(candidates), size=remaining, replace=False, p=residual_weights),
            )
        )
    selected = candidates.iloc[np.asarray(chosen, dtype=int)].copy().reset_index(drop=True)
    permutation = rng.permutation(len(selected))
    selected = selected.iloc[permutation].reset_index(drop=True)
    metadata = {
        "policy": "complete_community_activity_then_unit_aware_location_sampling_v1",
        "eligible_pool_size": original_pool_size,
        "depot_catchment_pool_size": len(customers),
        "depot_catchment_radius_km": depot_catchment_radius_km,
        "selected_customer_depot_haversine_km_mean": float(
            selected["depot_haversine_km"].mean()
        ),
        "selected_customer_depot_haversine_km_max": float(
            selected["depot_haversine_km"].max()
        ),
        "selected_location_count": len(selected),
        "sampled_target_active_locations_per_community": target,
        "selected_community_count": len(selected_group_indices),
        "selected_community_capacity": selected_capacity,
        "per_unit_order_probability": per_unit_probability,
        "selected_service_location_type_counts": {
            str(key): int(value)
            for key, value in selected["service_location_type"].value_counts().items()
        },
    }
    return selected, metadata


def _depot_catchment_rows(
    customers: pd.DataFrame,
    *,
    count: int,
    profile: Mapping[str, Any],
    depot_longitude: float,
    depot_latitude: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Freeze the infrastructure catchment before CS or daily-customer selection."""
    if len(customers) < count:
        raise ValueError(f"Customer pool has {len(customers)} locations; requested {count}")
    catchment_cfg = profile["customer_activation"]["depot_catchment"]
    depot_distances = _haversine_km(
        customers["location_lon"].to_numpy(dtype=float),
        customers["location_lat"].to_numpy(dtype=float),
        np.asarray([depot_longitude], dtype=float),
        np.asarray([depot_latitude], dtype=float),
    )[:, 0]
    minimum_pool = max(
        count,
        math.ceil(count * float(catchment_cfg["minimum_pool_buffer"])),
    )
    radius = float(catchment_cfg["start_radius_km"])
    maximum_radius = float(catchment_cfg["max_radius_km"])
    step = float(catchment_cfg["expansion_step_km"])
    catchment_mask = depot_distances <= radius
    while int(catchment_mask.sum()) < minimum_pool and radius < maximum_radius:
        radius = min(maximum_radius, radius + step)
        catchment_mask = depot_distances <= radius
    if int(catchment_mask.sum()) < count:
        raise ValueError(
            f"Depot catchment has {int(catchment_mask.sum())} locations; requested {count}"
        )
    result = customers.loc[catchment_mask].copy()
    result["depot_haversine_km"] = depot_distances[catchment_mask]
    return result, {
        "eligible_pool_size": len(customers),
        "depot_catchment_pool_size": len(result),
        "depot_catchment_radius_km": radius,
        "minimum_required_pool_size": minimum_pool,
    }


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


def _select_charger_rows(
    chargers: pd.DataFrame,
    reference_points: pd.DataFrame,
    *,
    count: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(chargers) < count:
        raise ValueError(f"Charger pool has {len(chargers)} sites; requested {count}")
    distances = _haversine_km(
        reference_points["reference_longitude"].to_numpy(dtype=float),
        reference_points["reference_latitude"].to_numpy(dtype=float),
        chargers["resolved_longitude"].to_numpy(dtype=float),
        chargers["resolved_latitude"].to_numpy(dtype=float),
    )
    weights = reference_points["reference_weight"].to_numpy(dtype=float)
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    tie_breaker = rng.random(len(chargers)) * 1e-9
    current = np.full(len(reference_points), np.inf, dtype=float)
    available = np.ones(len(chargers), dtype=bool)
    selected: list[int] = []
    for _ in range(count):
        candidate_indices = np.flatnonzero(available)
        proposed = np.minimum(current[:, None], distances[:, candidate_indices])
        score = (
            (proposed * weights[:, None]).sum(axis=0)
            + 0.25 * np.quantile(proposed, 0.90, axis=0)
            + 0.10 * proposed.max(axis=0)
            + tie_breaker[candidate_indices]
        )
        chosen = int(candidate_indices[int(np.argmin(score))])
        selected.append(chosen)
        available[chosen] = False
        current = np.minimum(current, distances[:, chosen])
    result = chargers.iloc[np.asarray(selected, dtype=int)].copy().reset_index(drop=True)
    result["charger_selection_rank"] = np.arange(1, len(result) + 1, dtype=np.int32)
    metadata = {
        "policy": "nested_depot_catchment_community_coverage_greedy_v1",
        "candidate_pool_size": len(chargers),
        "reference_point_count": len(reference_points),
        "reference_point_semantics": (
            "complete-community centroids weighted by eligible latent-location count; "
            "independent of daily active customer IDs"
        ),
        "selected_count": len(result),
        "weighted_mean_reference_to_nearest_selected_charger_km": float(
            np.sum(current * weights)
        ),
        "p90_reference_to_nearest_selected_charger_km": float(
            np.quantile(current, 0.90)
        ),
        "max_reference_to_nearest_selected_charger_km": float(current.max()),
    }
    return result, metadata


def _community_reference_points(customers: pd.DataFrame) -> pd.DataFrame:
    return (
        customers.groupby("community_id", sort=True)
        .agg(
            reference_longitude=("location_lon", "mean"),
            reference_latitude=("location_lat", "mean"),
            reference_weight=("latent_service_location_id", "size"),
        )
        .reset_index()
    )


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
    city_chargers: pd.DataFrame | None = None,
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
    median_source = result if city_chargers is None else city_chargers
    city_observed = pd.to_numeric(median_source["station_power_kw"], errors="coerce")
    mode_medians = city_observed.groupby(median_source["reference_charge_mode"]).median().to_dict()
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
            sources.append("city_mode_median_capped_by_vehicle")
        elif generation_mode in {"research", "non_release_pilot"}:
            resolved.append(cap)
            sources.append(f"{generation_mode}_vehicle_mode_cap_fallback")
        else:
            raise ValueError(
                f"No reported or city-mode median station power is available for {mode}; "
                "official generation is blocked."
            )
    result["effective_charging_power_kw"] = np.asarray(resolved, dtype=np.float32)
    result["effective_charging_power_source"] = sources
    return result, {
        "reported_power_count": int(sum(source.startswith("reported") for source in sources)),
        "city_mode_median_count": int(sum(source.startswith("city_mode") for source in sources)),
        "vehicle_mode_cap_fallback_count": int(
            sum(source.endswith("vehicle_mode_cap_fallback") for source in sources)
        ),
        "generation_mode": generation_mode,
        "city_mode_reported_power_medians_kw": {
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


def select_family_terminals(
    cle: PortableCLE,
    *,
    family: Mapping[str, Any],
    customer_split_path: str,
    profile: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    family_seed = int(family["family_seed"])
    customer_count = int(family["parent_customer_count"])
    charger_count = int(family["parent_charging_station_count"])
    day_type = sample_day_type(profile, family_seed)

    depot_columns = [
        "candidate_id",
        "longitude",
        "latitude",
        "strict_depot_candidate_eligible",
        "optional_depot_candidate_eligible",
        "evidence_tier",
        "depot_evidence_class",
        "physical_edge_id",
        "directed_projection_offsets",
        "connector_length_m",
        "road_projection_node_id",
        "facility_access_node_id",
        "anchor_scc_id",
    ]
    depots = cle.read_depots(columns=depot_columns).reset_index(drop=True)
    depot_rng = np.random.default_rng(int(family["depot_seed"]))
    depot_weights = np.where(depots["strict_depot_candidate_eligible"].astype(bool), 2.0, 1.0)
    depot_weights /= depot_weights.sum()
    depot = depots.iloc[int(depot_rng.choice(len(depots), p=depot_weights))]

    customer_columns = [
        "latent_service_location_id",
        "service_location_type",
        "residential_unit_band",
        "residential_units",
        "location_lon",
        "location_lat",
        "physical_edge_id",
        "directed_projection_offsets",
        "connector_length_m",
        "road_projection_node_id",
        "service_access_node_id",
        "anchor_scc_id",
    ]
    customers = cle.read_service_locations(columns=customer_columns)
    split = pd.read_parquet(customer_split_path)
    required_split = {"latent_service_location_id", "community_id", "customer_pool"}
    missing = required_split - set(split.columns)
    if missing:
        raise ValueError(f"Customer split ledger is missing columns: {sorted(missing)}")
    split = split[sorted(required_split)].copy()
    customers = customers.merge(
        split,
        on="latent_service_location_id",
        how="inner",
        validate="one_to_one",
    )
    requested_pool = str(family["customer_pool"])
    if requested_pool in {"train", "heldout"}:
        customers = customers.loc[customers["customer_pool"].eq(requested_pool)].copy()
    elif requested_pool != "all_release_eligible":
        raise ValueError(f"Unsupported customer pool: {requested_pool!r}")
    original_customer_pool_size = len(customers)
    catchment_customers, catchment_metadata = _depot_catchment_rows(
        customers,
        count=customer_count,
        profile=profile,
        depot_longitude=float(depot["longitude"]),
        depot_latitude=float(depot["latitude"]),
    )

    # Infrastructure is selected from the frozen depot catchment before the
    # daily active-customer set is sampled. Community centroids avoid leaking
    # exact daily customer IDs into the charger-selection policy.
    charger_columns = [
        "charger_id",
        "resolved_longitude",
        "resolved_latitude",
        "reference_charge_mode",
        "station_power_kw",
        "station_power_status",
        "l2_ports",
        "dc_fast_ports",
        "physical_edge_id",
        "directed_projection_offsets",
        "connector_length_m",
        "road_projection_node_id",
        "facility_access_node_id",
        "anchor_scc_id",
    ]
    chargers = cle.read_chargers(columns=charger_columns).reset_index(drop=True)
    selected_chargers, charger_metadata = _select_charger_rows(
        chargers,
        _community_reference_points(catchment_customers),
        count=charger_count,
        seed=int(family["charger_seed"]),
    )
    selected_chargers, power_metadata = _resolve_charging_power(
        selected_chargers,
        city_chargers=chargers,
        profile=profile,
        generation_mode=cle.mode,
    )
    selected_customers, customer_metadata = _select_customer_rows(
        catchment_customers,
        count=customer_count,
        day_type=day_type,
        profile=profile,
        seed=int(family["customer_superset_seed"]),
        original_pool_size=original_customer_pool_size,
        depot_catchment_radius_km=float(catchment_metadata["depot_catchment_radius_km"]),
    )
    charger_active_customer_diagnostic = _active_customer_charger_diagnostic(
        selected_customers,
        selected_chargers,
    )

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
            "service_location_type": pd.NA,
            "residential_unit_band": pd.NA,
            "residential_units": pd.NA,
            "charger_selection_rank": pd.NA,
            "reference_charge_mode": pd.NA,
            "effective_charging_power_kw": pd.NA,
            "effective_charging_power_source": pd.NA,
        }
    ]
    for position, (_, row) in enumerate(selected_customers.iterrows()):
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
                "service_location_type": str(row["service_location_type"]),
                "residential_unit_band": str(row["residential_unit_band"]),
                "residential_units": int(row["residential_units"]),
                "charger_selection_rank": pd.NA,
                "reference_charge_mode": pd.NA,
                "effective_charging_power_kw": pd.NA,
                "effective_charging_power_source": pd.NA,
            }
        )
    for rank, (_, row) in enumerate(selected_chargers.iterrows(), start=1):
        records.append(
            {
                **_common_terminal_record(
                    row,
                    terminal_index=1 + customer_count + rank - 1,
                    terminal_kind="charging_station",
                    source_id=str(row["charger_id"]),
                    longitude=float(row["resolved_longitude"]),
                    latitude=float(row["resolved_latitude"]),
                ),
                "parent_customer_position": pd.NA,
                "community_id": pd.NA,
                "service_location_type": pd.NA,
                "residential_unit_band": pd.NA,
                "residential_units": pd.NA,
                "charger_selection_rank": rank,
                "reference_charge_mode": str(row["reference_charge_mode"]),
                "effective_charging_power_kw": float(row["effective_charging_power_kw"]),
                "effective_charging_power_source": str(row["effective_charging_power_source"]),
            }
        )
    terminal_index = pd.DataFrame.from_records(records)
    metadata = {
        "schema": "cle_evrptw_family_terminal_selection_v1",
        "family_id": str(family["family_id"]),
        "city_slug": cle.city_slug,
        "day_type": day_type,
        "customer_pool": requested_pool,
        "parent_customer_count": customer_count,
        "parent_charging_station_count": charger_count,
        "selected_depot_id": str(depot["candidate_id"]),
        "selected_depot_evidence_tier": str(depot["evidence_tier"]),
        "customer_selection": customer_metadata,
        "charger_selection": charger_metadata,
        "charger_active_customer_coverage_diagnostic": charger_active_customer_diagnostic,
        "charging_power_resolution": power_metadata,
        "non_release_pilot": cle.non_release_pilot,
    }
    return terminal_index, metadata
