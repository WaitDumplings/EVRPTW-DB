"""Materialize one matrix family and all of its deterministic scale views."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Stage2Config
from .orders import FULL_CS_TO_DEPOT_CACHE_CONTRACT, build_view_attributes
from .reader import PortableCLE
from .road_state import build_family_road_state
from .routing import PhysicalRoadNetwork, RoutingMatrices
from .selection import select_family_terminals


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def view_parent_terminal_indices(
    view: Mapping[str, Any],
    *,
    parent_customer_count: int,
    parent_charging_station_count: int,
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
    charger_indices = charger_start + np.arange(charger_count, dtype=np.int32)
    return np.concatenate(
        [np.asarray([0], dtype=np.int32), customer_indices, charger_indices]
    )


def _matrix_payload(matrices: RoutingMatrices) -> dict[str, np.ndarray]:
    return {
        "distance_matrix_km": matrices.distance_matrix_km,
        "distance_path_travel_time_s": matrices.distance_path_travel_time_s,
        "distance_path_energy_kwh": matrices.distance_path_energy_kwh,
        "running_time_shortest_matrix_s": matrices.running_time_shortest_matrix_s,
        "running_time_path_distance_km": matrices.running_time_path_distance_km,
        "running_time_path_energy_kwh": matrices.running_time_path_energy_kwh,
    }


def materialize_family(
    cle: PortableCLE,
    *,
    config: Stage2Config,
    profile: dict[str, Any],
    family: Mapping[str, Any],
    views: pd.DataFrame,
    customer_split_path: str | Path,
    output_root: str | Path,
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

    terminal_index, selection_report = select_family_terminals(
        cle,
        family=family,
        customer_split_path=str(customer_split_path),
        profile=profile,
    )
    directed_speeds = pd.read_parquet(cle.speeds_path)
    road_state, road_state_report = build_family_road_state(
        directed_speeds,
        day_type=str(selection_report["day_type"]),
        road_state_seed=int(family["road_state_seed"]),
        profile=profile,
    )
    network = PhysicalRoadNetwork.from_files(cle.graph_path, road_state, profile)
    matrices = network.route_terminals(terminal_index)
    matrix_payload = _matrix_payload(matrices)

    with tempfile.TemporaryDirectory(prefix=f".{family_id}-", dir=final_dir.parent) as temp_name:
        temp_dir = Path(temp_name)
        matrix_dir = temp_dir / "matrices"
        matrix_dir.mkdir()
        terminal_index.to_parquet(temp_dir / "terminal_index.parquet", index=False)
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
            )
            customer_count = int(view["customer_count"])
            view_terminals = terminal_index.iloc[indices].reset_index(drop=True)
            customer_rows = view_terminals.iloc[1 : 1 + customer_count]
            charger_rows = view_terminals.loc[
                view_terminals["terminal_kind"].eq("charging_station")
            ]
            charging_power = charger_rows["effective_charging_power_kw"].to_numpy(
                dtype=np.float32
            )
            running_time = matrices.running_time_shortest_matrix_s[np.ix_(indices, indices)]
            running_energy = matrices.running_time_path_energy_kwh[np.ix_(indices, indices)]
            attributes = build_view_attributes(
                customer_rows,
                day_type=str(selection_report["day_type"]),
                package_seed=int(view["package_seed"]),
                service_time_seed=int(view["service_time_seed"]),
                time_window_seed=int(view["time_window_seed"]),
                operating_start_s=config.operating_horizon_start_s,
                operating_end_s=config.operating_horizon_end_s,
                running_time_matrix_s=running_time,
                running_time_energy_matrix_kwh=running_energy,
                charging_power_kw=charging_power,
                profile=profile,
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
                feasibility_charging_visit_count=(
                    attributes.feasibility_charging_visit_count
                ),
                feasibility_inbound_full_state_terminal_index=(
                    attributes.feasibility_inbound_full_state_terminal_index
                ),
                feasibility_first_post_customer_charger_terminal_index=(
                    attributes.feasibility_first_post_customer_charger_terminal_index
                ),
                feasibility_energy_margin_kwh=attributes.feasibility_energy_margin_kwh,
            )
            np.savez_compressed(
                view_dir / "charging_attributes.npz",
                charging_power_kw=charging_power,
                full_cs_to_depot_time_s=attributes.full_cs_to_depot_time_s,
            )
            view_manifest = {
                "schema": "cle_evrptw_materialized_view_v2",
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
                },
                "charging_policy": dict(profile["charging"]),
                "runtime_mask_stored": False,
                "attribute_report": attributes.report,
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

        byte_counts = {
            name: int((temp_dir / relative).stat().st_size)
            for name, relative in matrix_files.items()
        }
        manifest = {
            "schema": "cle_evrptw_materialized_matrix_family_v1",
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
            "terminal_index": "terminal_index.parquet",
            "view_count": len(view_manifests),
            "view_ids": [item["view_id"] for item in view_manifests],
            "selection_report": selection_report,
            "road_state_report": road_state_report,
            "routing_report": matrices.report,
            "road_state_storage": "deterministic_reconstruction_from_seed_cle_and_profile",
            "road_state_seed": int(family["road_state_seed"]),
            "base_family_seed": int(family.get("base_family_seed", family["family_seed"])),
            "materialization_attempt_number": int(
                family.get("materialization_attempt_number", 0)
            ),
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
            },
            "charging_policy": dict(profile["charging"]),
            "non_release_pilot": cle.non_release_pilot,
            "materialization_status": "complete",
        }
        _write_json(temp_dir / "family_manifest.json", manifest)
        os.replace(temp_dir, final_dir)
    return manifest
