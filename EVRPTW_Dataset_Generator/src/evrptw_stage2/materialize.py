"""Materialize one matrix family and all of its deterministic scale views."""

from __future__ import annotations

import json
import os
import resource
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .amazon import AmazonStage2Artifacts
from .config import Stage2Config
from .metrics import build_phase1_family_metrics
from .orders import (
    FULL_CS_TO_DEPOT_CACHE_CONTRACT,
    build_view_attributes_from_amazon,
    match_amazon_order_templates,
)
from .reader import PortableCLE
from .road_state import build_family_road_state
from .routing import PhysicalRoadNetwork, RoutingMatrices
from .selection import (
    road_time_replacement_deltas,
    select_family_terminals_v2,
    select_road_time_charger_indices,
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _make_progress_emitter(
    callback: Callable[[str, Mapping[str, Any]], None] | None,
) -> Callable[..., None]:
    """Forward nested profile details without reserving their field names."""

    def progress(event_name: str, **details: Any) -> None:
        if callback is not None:
            callback(event_name, details)

    return progress


def _cle_reference(cle: PortableCLE) -> dict[str, Any]:
    manifest_path = cle.root / "manifest.json"
    return {
        "contract_root": "EVRPTW_Dataset/CLE_v2/us_11city",
        "city_relative_path": str(Path("cities") / cle.city_slug),
        "city_manifest_schema": str(cle.manifest.get("schema", "")),
        "city_manifest_size_bytes": int(manifest_path.stat().st_size),
        "content_digest_validation_performed": False,
        "connectivity_contract_id": str(
            cle.manifest.get("connectivity_contract", {}).get("id", "")
        ),
    }


def view_parent_terminal_indices(
    view: Mapping[str, Any],
    *,
    parent_customer_count: int,
    parent_charging_station_count: int,
    running_time_matrix_s: np.ndarray | None = None,
) -> np.ndarray:
    customer_count = int(view["customer_count"])
    charger_count = int(view["charging_station_count"])
    branch_index = int(view["branch_index"])
    if parent_customer_count % customer_count:
        raise ValueError("Parent customer count is not divisible by view customer count")
    branch_count = parent_customer_count // customer_count
    if not 0 <= branch_index < branch_count:
        raise ValueError(
            f"branch_index={branch_index} outside [0, {branch_count}) for {view['scale_id']}"
        )
    if charger_count > parent_charging_station_count:
        raise ValueError("View requests more charging stations than its matrix parent")
    start = branch_index * customer_count
    customer_indices = 1 + np.arange(start, start + customer_count, dtype=np.int32)
    charger_start = 1 + parent_customer_count
    parent_charger_indices = charger_start + np.arange(
        parent_charging_station_count, dtype=np.int32
    )
    if charger_count == parent_charging_station_count:
        charger_indices = parent_charger_indices
    else:
        if running_time_matrix_s is None:
            raise ValueError("Child charger reselection requires the parent running-time matrix")
        deltas = road_time_replacement_deltas(
            running_time_matrix_s,
            customer_indices=customer_indices,
            charger_indices=parent_charger_indices,
        )
        chosen, _ = select_road_time_charger_indices(
            deltas,
            count=charger_count,
            seed=int(view["view_seed"]),
        )
        charger_indices = parent_charger_indices[chosen]
    return np.concatenate([np.asarray([0], dtype=np.int32), customer_indices, charger_indices])


def _matrix_payload(matrices: RoutingMatrices) -> dict[str, np.ndarray]:
    return {
        "distance_matrix_km": matrices.distance_matrix_km,
        "distance_path_travel_time_s": matrices.distance_path_travel_time_s,
        "running_time_shortest_matrix_s": matrices.running_time_shortest_matrix_s,
        "running_time_path_distance_km": matrices.running_time_path_distance_km,
    }


def _view_spatial_metrics(
    customer_rows: pd.DataFrame,
    customer_time_s: np.ndarray,
    baseline_rows: pd.DataFrame,
    *,
    partition_tree: Mapping[str, Any],
    scale_id: str,
    branch_index: int,
) -> dict[str, Any]:
    times = np.asarray(customer_time_s, dtype=float).copy()
    np.fill_diagonal(times, np.inf)
    nearest = times.min(axis=1)
    regions = customer_rows["sampling_cluster_id"].astype(str).to_numpy()
    region_p50: list[float] = []
    region_p90: list[float] = []
    for region in sorted(set(regions)):
        local = np.flatnonzero(regions == region)
        if len(local) < 2:
            continue
        sub = np.asarray(customer_time_s, dtype=float)[np.ix_(local, local)]
        directed = sub[~np.eye(len(local), dtype=bool)]
        region_p50.append(float(np.quantile(directed, 0.50)))
        region_p90.append(float(np.quantile(directed, 0.90)))

    audit: dict[str, Any] | None = None
    namespace_child: tuple[str, int] | None = None
    if scale_id == "cus500":
        namespace_child = ("tree_1000_to_500", branch_index)
    elif scale_id == "cus100":
        namespace_child = (f"tree_500_{branch_index // 5}_to_100", branch_index % 5)
    elif scale_id == "cus50":
        namespace_child = (
            f"tree_100_{branch_index // 10}_{(branch_index // 2) % 5}_to_50",
            branch_index % 2,
        )
    elif scale_id == "cus1000" and int(partition_tree["parent_customer_count"]) == 2000:
        namespace_child = ("scalability_2000_to_1000", branch_index)
    if namespace_child is not None:
        namespace, child_index = namespace_child
        for level in partition_tree.get("partition_levels", []):
            if level["namespace"] == namespace:
                audit = dict(level["children"][child_index])
                audit["partition_namespace"] = namespace
                break
        if audit is None:
            raise ValueError(f"Missing region-first partition audit for {namespace_child}")

    def concentration(frame: pd.DataFrame) -> dict[str, Any]:
        counts = frame["community_id"].astype(str).value_counts()
        shares = counts.to_numpy(dtype=float) / max(len(frame), 1)
        return {
            "community_count": int(len(counts)),
            "largest_community_share": float(shares.max()) if len(shares) else 0.0,
            "community_hhi": float(np.square(shares).sum()),
        }

    return {
        "m2_outgoing_network_nearest_neighbor_s": {
            "mean": float(nearest.mean()),
            "p50": float(np.quantile(nearest, 0.50)),
            "p90": float(np.quantile(nearest, 0.90)),
        },
        "m3_directed_within_region_s": {
            "region_p50_values": region_p50,
            "region_p90_values": region_p90,
        },
        "m4_region_first_partition": audit or {
            "parent_view": True,
            "region_count": int(len(set(regions))),
        },
        "m5_community_concentration": {
            "generated": concentration(customer_rows),
            "same_count_uniform_baseline": concentration(baseline_rows),
            "diagnostic_only": True,
        },
    }


def materialize_family(
    cle: PortableCLE,
    *,
    config: Stage2Config,
    profile: dict[str, Any],
    family: Mapping[str, Any],
    views: pd.DataFrame,
    customer_split_path: str | Path,
    community_adjacency_path: str | Path,
    amazon_artifacts: AmazonStage2Artifacts,
    output_root: str | Path,
    routing_topology_cache: dict[str, PhysicalRoadNetwork] | None = None,
    community_adjacency_cache: dict[str, pd.DataFrame] | None = None,
    code_provenance: Mapping[str, Any] | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    family_id = str(family["family_id"])
    if views.empty:
        raise ValueError(f"No scale views were supplied for family {family_id}")
    if set(views["family_id"].astype(str)) != {family_id}:
        raise ValueError("View rows do not belong exclusively to the requested family")
    final_dir = Path(output_root) / "families" / family_id
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite materialized family: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    materialization_started = time.perf_counter()
    stage_timings: dict[str, float] = {}
    performance_profile: list[dict[str, Any]] = []
    progress = _make_progress_emitter(progress_callback)

    def finish_profile(
        stage: str,
        wall_started: float,
        cpu_started: float,
        **details: Any,
    ) -> None:
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        event = {
            "stage": stage,
            "status": "completed",
            "wall_seconds": time.perf_counter() - wall_started,
            "cpu_seconds": time.process_time() - cpu_started,
            "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
            **details,
        }
        performance_profile.append(event)
        progress(f"{stage}.completed", **event)

    progress("road_state")
    stage_started = time.perf_counter()
    directed_speeds = pd.read_parquet(cle.speeds_path)
    road_state, road_state_report = build_family_road_state(
        directed_speeds,
        day_type=str(family["day_type"]),
        road_state_seed=int(family["road_state_seed"]),
        profile=profile,
    )
    stage_timings["road_state"] = time.perf_counter() - stage_started
    progress("routing_topology")
    stage_started = time.perf_counter()
    cached_network = (
        routing_topology_cache.get(cle.city_slug) if routing_topology_cache is not None else None
    )
    if cached_network is None:
        network = PhysicalRoadNetwork.from_files(cle.graph_path, road_state, profile)
        if routing_topology_cache is not None:
            routing_topology_cache[cle.city_slug] = network
    else:
        network = cached_network.with_road_state(road_state, profile)
    stage_timings["routing_topology"] = time.perf_counter() - stage_started
    progress("terminal_selection")
    stage_started = time.perf_counter()
    stage_cpu_started = time.process_time()
    terminal_index, selection_report, radial_baseline = select_family_terminals_v2(
        cle,
        family=family,
        customer_split_path=str(customer_split_path),
        community_adjacency_path=str(community_adjacency_path),
        profile=profile,
        network=network,
        amazon=amazon_artifacts,
        community_adjacency_cache=community_adjacency_cache,
        progress_callback=(
            lambda stage, details: progress(f"terminal_selection.{stage}", **details)
        ),
    )
    stage_timings["terminal_selection"] = time.perf_counter() - stage_started
    finish_profile(
        "terminal_selection",
        stage_started,
        stage_cpu_started,
        terminal_count=len(terminal_index),
        nested_profile=selection_report["performance_profile"],
    )
    progress("parent_matrix_closure", terminal_count=len(terminal_index))
    stage_started = time.perf_counter()
    stage_cpu_started = time.process_time()
    matrices = network.route_terminals(terminal_index)
    stage_timings["parent_matrix_closure"] = time.perf_counter() - stage_started
    finish_profile(
        "matrix_construction",
        stage_started,
        stage_cpu_started,
        terminal_count=len(terminal_index),
        routing_workload=matrices.report,
    )
    matrix_payload = _matrix_payload(matrices)
    parent_customer_count = int(family["parent_customer_count"])
    parent_charger_rows = terminal_index.loc[
        terminal_index["terminal_kind"].eq("charging_station")
    ]
    parent_charging_power = parent_charger_rows[
        "effective_charging_power_kw"
    ].to_numpy(dtype=np.float32)
    source_pool = amazon_artifacts.pool_for_track(str(family["track_id"]))
    allow_composite = str(family["parent_scale_id"]) == "cus2000"
    order_source_candidates = amazon_artifacts.order_sources(
        day_type=str(selection_report["day_type"]),
        customer_count=parent_customer_count,
        seed=int(family["family_seed"]),
        pool=source_pool,
        track_id=str(family["track_id"]),
        allow_composite=allow_composite,
    )
    if not order_source_candidates:
        raise ValueError(
            "PF2_ORDER_UNSUPPORTED: no single-day or same-station composite order source"
        )
    progress("order_matching", source_candidate_count=len(order_source_candidates))
    stage_started = time.perf_counter()
    matched_templates, order_source_report = match_amazon_order_templates(
        customer_count=parent_customer_count,
        order_sources=[
            (source, amazon_artifacts.templates_for_source(source))
            for source in order_source_candidates
        ],
        matching_seed=int(family["family_seed"]),
        operating_start_s=config.operating_horizon_start_s,
        operating_end_s=config.operating_horizon_end_s,
        running_time_matrix_s=matrices.running_time_shortest_matrix_s,
        running_time_path_distance_matrix_km=matrices.running_time_path_distance_km,
        charging_power_kw=parent_charging_power,
        profile=profile,
    )
    structure_mode = str(
        selection_report["amazon_structure_source"]["structure_source_mode"]
    )
    order_mode = str(order_source_report["selected_order_source_mode"])
    if not allow_composite and (
        structure_mode != "SINGLE_STRUCTURE_DAY" or order_mode != "SINGLE_ORDER_DAY"
    ):
        raise ValueError(
            "PRIMARY_COMPOSITE_SOURCE_REJECTED: "
            f"scale={family['parent_scale_id']}, modes={(structure_mode, order_mode)}"
        )
    order_source_report["source_mode"] = [structure_mode, order_mode]
    order_source_report["source_pool"] = source_pool
    order_source_report["release_role"] = (
        "report_only" if allow_composite else "primary_candidate"
    )
    structure_ids = set(
        map(
            str,
            selection_report["amazon_structure_source"]["structure_source_ids"],
        )
    )
    order_ids = set(map(str, order_source_report["selected_station_day_ids"]))
    structure_stations = {value.split(":", 1)[0] for value in structure_ids}
    order_stations = {value.split(":", 1)[0] for value in order_ids}
    if structure_ids == order_ids:
        relationship = "same_station_day"
    elif structure_stations & order_stations:
        relationship = "same_station"
    else:
        relationship = "independent_station_day"
    order_source_report["structure_order_source_relationship"] = relationship
    stage_timings["order_matching"] = time.perf_counter() - stage_started
    terminal_index["order_template_id"] = pd.NA
    terminal_index["order_station_day_id"] = pd.NA
    terminal_index["order_source_mode"] = pd.NA
    customer_terminal_rows = np.arange(1, 1 + parent_customer_count)
    terminal_index.loc[customer_terminal_rows, "order_template_id"] = matched_templates[
        "template_id"
    ].astype(str).to_numpy()
    terminal_index.loc[customer_terminal_rows, "order_station_day_id"] = matched_templates[
        "station_day_id"
    ].astype(str).to_numpy()
    terminal_index.loc[customer_terminal_rows, "order_source_mode"] = order_source_report[
        "selected_order_source_mode"
    ]

    with tempfile.TemporaryDirectory(prefix=f".{family_id}-", dir=final_dir.parent) as temp_name:
        temp_dir = Path(temp_name)
        matrix_dir = temp_dir / "matrices"
        matrix_dir.mkdir()
        terminal_index.to_parquet(temp_dir / "terminal_index.parquet", index=False)
        progress("parent_metrics")
        stage_started = time.perf_counter()
        phase1_metrics, phase1_observations, phase1_region_pairs = (
            build_phase1_family_metrics(
                family_manifest_fields={
                    "family_id": family_id,
                    "family_cohort_id": str(family["family_cohort_id"]),
                    "city_slug": cle.city_slug,
                    "day_type": str(selection_report["day_type"]),
                    "parent_scale_id": str(family["parent_scale_id"]),
                    "materialization_attempt_number": int(
                        family.get("materialization_attempt_number", 0)
                    ),
                },
                terminal_index=terminal_index,
                running_time_matrix_s=matrices.running_time_shortest_matrix_s,
                radial_baseline=radial_baseline,
                selection_report=selection_report,
            )
        )
        _write_json(temp_dir / "phase1_metrics.json", phase1_metrics)
        phase1_observations.to_parquet(
            temp_dir / "phase1_observations.parquet", index=False
        )
        phase1_region_pairs.to_parquet(
            temp_dir / "phase1_region_pair_metrics.parquet", index=False
        )
        stage_timings["parent_metrics"] = time.perf_counter() - stage_started
        progress("view_materialization", view_count=len(views))
        stage_started = time.perf_counter()
        matrix_files: dict[str, str] = {}
        for name, array in matrix_payload.items():
            relative = f"matrices/{name}.npy"
            np.save(temp_dir / relative, np.asarray(array, dtype=np.float32), allow_pickle=False)
            matrix_files[name] = relative

        view_manifests: list[dict[str, Any]] = []
        for _, view_row in views.sort_values("view_id").iterrows():
            view = view_row.to_dict()
            indices = view_parent_terminal_indices(
                view,
                parent_customer_count=int(family["parent_customer_count"]),
                parent_charging_station_count=int(family["parent_charging_station_count"]),
                running_time_matrix_s=matrices.running_time_shortest_matrix_s,
            )
            customer_count = int(view["customer_count"])
            view_terminals = terminal_index.iloc[indices].reset_index(drop=True)
            customer_rows = view_terminals.iloc[1 : 1 + customer_count]
            charger_rows = view_terminals.loc[
                view_terminals["terminal_kind"].eq("charging_station")
            ]
            charging_power = charger_rows["effective_charging_power_kw"].to_numpy(dtype=np.float32)
            running_time = matrices.running_time_shortest_matrix_s[np.ix_(indices, indices)]
            running_distance = matrices.running_time_path_distance_km[np.ix_(indices, indices)]
            parent_customer_positions = indices[1 : 1 + customer_count] - 1
            baseline_rows = radial_baseline.iloc[
                parent_customer_positions.astype(int)
            ].reset_index(drop=True)
            view_templates = matched_templates.iloc[
                parent_customer_positions.astype(int)
            ].reset_index(drop=True)
            attributes = build_view_attributes_from_amazon(
                customer_rows,
                view_templates,
                day_type=str(selection_report["day_type"]),
                operating_start_s=config.operating_horizon_start_s,
                operating_end_s=config.operating_horizon_end_s,
                running_time_matrix_s=running_time,
                running_time_path_distance_matrix_km=running_distance,
                charging_power_kw=charging_power,
                profile=profile,
                order_source_report=order_source_report,
            )
            spatial_metrics = _view_spatial_metrics(
                customer_rows,
                running_time[1 : 1 + customer_count, 1 : 1 + customer_count],
                baseline_rows,
                partition_tree=selection_report["spatial_activation"]["view_tree"],
                scale_id=str(view["scale_id"]),
                branch_index=int(view["branch_index"]),
            )
            view_id = str(view["view_id"])
            view_dir = temp_dir / "views" / view_id
            view_dir.mkdir(parents=True)
            np.save(view_dir / "terminal_parent_indices.npy", indices, allow_pickle=False)
            np.savez_compressed(
                view_dir / "customer_attributes.npz",
                package_counts=attributes.package_counts,
                demands_cm3=attributes.demands_cm3,
                service_time_s=attributes.service_time_s,
                time_windows_s=attributes.time_windows_s,
                feasible_arrival_time_s=attributes.feasible_arrival_time_s,
                feasible_return_duration_s=attributes.feasible_return_duration_s,
                feasibility_requires_charging=attributes.feasibility_requires_charging,
                feasibility_charging_visit_count=(attributes.feasibility_charging_visit_count),
                feasibility_inbound_full_state_terminal_index=(
                    attributes.feasibility_inbound_full_state_terminal_index
                ),
                feasibility_first_post_customer_charger_terminal_index=(
                    attributes.feasibility_first_post_customer_charger_terminal_index
                ),
                feasibility_energy_margin_kwh=attributes.feasibility_energy_margin_kwh,
                order_sampling_attempts=attributes.order_sampling_attempts,
            )
            np.savez_compressed(
                view_dir / "charging_attributes.npz",
                charging_power_kw=charging_power,
                full_cs_to_depot_time_s=attributes.full_cs_to_depot_time_s,
            )
            view_manifest = {
                "schema": "cle_evrptw_materialized_view_v4",
                "view_id": view_id,
                "family_id": family_id,
                "consumer_cohort_id": str(view["consumer_cohort_id"]),
                "split_id": str(view["split_id"]),
                "track_id": str(view["track_id"]),
                "city_slug": cle.city_slug,
                "scale_id": str(view["scale_id"]),
                "day_type": str(selection_report["day_type"]),
                "customer_count": customer_count,
                "charging_station_count": int(view["charging_station_count"]),
                "operating_horizon_s": [
                    config.operating_horizon_start_s,
                    config.operating_horizon_end_s,
                ],
                "matrix_storage": "parent_index_view",
                "charger_selection": {
                    "policy": "child_road_time_reselection_from_eligible_parent_roster_v1",
                    "parent_terminal_indices": indices[
                        1 + customer_count :
                    ].astype(int).tolist(),
                    "prefix_semantics": False,
                },
                "parent_matrix_files": matrix_files,
                "terminal_parent_indices": "terminal_parent_indices.npy",
                "customer_attributes": "customer_attributes.npz",
                "charging_attributes": "charging_attributes.npz",
                "full_cs_to_depot_cache": dict(FULL_CS_TO_DEPOT_CACHE_CONTRACT),
                "vehicle": {
                    "vehicle_id": str(profile["vehicle"]["vehicle_id"]),
                    "battery_capacity_kwh": float(profile["energy"]["battery_capacity_kwh"]),
                    "cargo_capacity_cm3": float(profile["vehicle"]["cargo_capacity_cm3"]),
                    "unlimited_fleet": bool(profile["vehicle"]["unlimited_fleet"]),
                    "specific_energy_consumption_kwh_per_km": float(
                        profile["energy"]["specific_energy_consumption_kwh_per_km"]
                    ),
                },
                "energy_model": dict(profile["energy"]),
                "charging_policy": dict(profile["charging"]),
                "runtime_mask_stored": False,
                "attribute_report": attributes.report,
                "spatial_metrics": spatial_metrics,
                "order_source": order_source_report,
                "order_template_ids_inherited_from_parent": True,
                "generation_mode": cle.mode,
                "non_release_pilot": cle.non_release_pilot,
                "materialization_attempt_number": int(
                    family.get("materialization_attempt_number", 0)
                ),
                "materialization_attempt_seed": int(
                    family.get("materialization_attempt_seed", family["family_seed"])
                ),
            }
            _write_json(view_dir / "view_manifest.json", view_manifest)
            view_manifests.append(view_manifest)
        stage_timings["view_materialization"] = time.perf_counter() - stage_started

        byte_counts = {
            name: int((temp_dir / relative).stat().st_size)
            for name, relative in matrix_files.items()
        }
        stage_timings["total_before_atomic_commit"] = (
            time.perf_counter() - materialization_started
        )
        manifest = {
            "schema": "cle_evrptw_materialized_matrix_family_v3",
            "stage2_generation_contract": "stage2_repair_v2_1_final",
            "code_provenance": dict(code_provenance or {"status": "unbound_test_fixture"}),
            "cle_reference": _cle_reference(cle),
            "family_id": family_id,
            "family_cohort_id": str(family["family_cohort_id"]),
            "city_slug": cle.city_slug,
            "day_type": str(selection_report["day_type"]),
            "parent_scale_id": str(family["parent_scale_id"]),
            "parent_customer_count": int(family["parent_customer_count"]),
            "parent_charging_station_count": int(family["parent_charging_station_count"]),
            "terminal_count": len(terminal_index),
            "matrix_dtype": "float32",
            "matrix_files": matrix_files,
            "matrix_file_bytes": byte_counts,
            "matrix_total_bytes": int(sum(byte_counts.values())),
            "stored_matrix_count": len(matrix_files),
            "derived_matrix_names": ["distance_path_energy_kwh", "running_time_path_energy_kwh"],
            "terminal_index": "terminal_index.parquet",
            "view_count": len(view_manifests),
            "view_ids": [item["view_id"] for item in view_manifests],
            "selection_report": selection_report,
            "order_source_report": order_source_report,
            "order_template_assignment": "terminal_index.parquet",
            "road_state_report": road_state_report,
            "routing_report": matrices.report,
            "road_state_storage": "deterministic_reconstruction_from_seed_cle_and_profile",
            "road_state_seed": int(family["road_state_seed"]),
            "base_family_seed": int(family.get("base_family_seed", family["family_seed"])),
            "materialization_attempt_number": int(family.get("materialization_attempt_number", 0)),
            "materialization_attempt_seed": int(
                family.get("materialization_attempt_seed", family["family_seed"])
            ),
            "reference_profile_id": str(profile["profile_id"]),
            "reference_profile_status": str(profile["profile_status"]),
            "vehicle": {
                "vehicle_id": str(profile["vehicle"]["vehicle_id"]),
                "battery_capacity_kwh": float(profile["energy"]["battery_capacity_kwh"]),
                "cargo_capacity_cm3": float(profile["vehicle"]["cargo_capacity_cm3"]),
                "unlimited_fleet": bool(profile["vehicle"]["unlimited_fleet"]),
                "specific_energy_consumption_kwh_per_km": float(
                    profile["energy"]["specific_energy_consumption_kwh_per_km"]
                ),
            },
            "energy_model": dict(profile["energy"]),
            "charging_policy": dict(profile["charging"]),
            "generation_mode": cle.mode,
            "non_release_pilot": cle.non_release_pilot,
            "materialization_status": "complete",
            "stage_timings_seconds": stage_timings,
            "performance_profile": performance_profile,
            "phase1_metrics": "phase1_metrics.json",
            "phase1_observations": "phase1_observations.parquet",
            "phase1_region_pair_metrics": "phase1_region_pair_metrics.parquet",
            "phase1_metric_files": {
                "phase1_metrics": "phase1_metrics.json",
                "phase1_observations": "phase1_observations.parquet",
                "phase1_region_pair_metrics": "phase1_region_pair_metrics.parquet",
            },
        }
        _write_json(temp_dir / "family_manifest.json", manifest)
        progress("atomic_publish")
        os.replace(temp_dir, final_dir)
    progress("complete")
    return manifest
