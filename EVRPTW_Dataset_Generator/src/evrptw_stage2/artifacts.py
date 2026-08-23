"""Consumer-facing loader and structural verifier for materialized Stage-2 views."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import STAGE2_GENERATION_CONTRACT
from .metrics import PHASE1_FAMILY_METRICS_SCHEMA
from .orders import (
    FULL_CS_TO_DEPOT_CACHE_CONTRACT,
    _single_customer_certificates,
)

STORED_MATRIX_NAMES = (
    "distance_matrix_km",
    "distance_path_travel_time_s",
    "running_time_shortest_matrix_s",
    "running_time_path_distance_km",
)
DERIVED_MATRIX_NAMES = (
    "distance_path_energy_kwh",
    "running_time_path_energy_kwh",
)
MATRIX_NAMES = (*STORED_MATRIX_NAMES, *DERIVED_MATRIX_NAMES)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_materialized_view(
    family_dir: str | Path,
    view_id: str,
    *,
    mmap_mode: str | None = "r",
) -> dict[str, Any]:
    root = Path(family_dir)
    family_manifest = _read_json(root / "family_manifest.json")
    view_root = root / "views" / view_id
    view_manifest = _read_json(view_root / "view_manifest.json")
    if view_manifest["family_id"] != family_manifest["family_id"]:
        raise ValueError("View/family manifest ID mismatch")
    parent_indices = np.load(
        view_root / view_manifest["terminal_parent_indices"],
        allow_pickle=False,
    ).astype(np.int32, copy=False)
    terminal_index = pd.read_parquet(root / family_manifest["terminal_index"])
    terminals = terminal_index.iloc[parent_indices].reset_index(drop=True)
    matrices: dict[str, np.ndarray] = {}
    for name in STORED_MATRIX_NAMES:
        parent = np.load(
            root / family_manifest["matrix_files"][name],
            mmap_mode=mmap_mode,
            allow_pickle=False,
        )
        matrices[name] = np.asarray(parent[np.ix_(parent_indices, parent_indices)])
    specific_energy = float(
        family_manifest["energy_model"]["specific_energy_consumption_kwh_per_km"]
    )
    matrices["distance_path_energy_kwh"] = (
        matrices["distance_matrix_km"] * specific_energy
    ).astype(np.float32)
    matrices["running_time_path_energy_kwh"] = (
        matrices["running_time_path_distance_km"] * specific_energy
    ).astype(np.float32)
    with np.load(view_root / view_manifest["customer_attributes"], allow_pickle=False) as data:
        attributes = {key: data[key] for key in data.files}
    customer_count = int(view_manifest["customer_count"])
    charger_count = int(view_manifest["charging_station_count"])
    cache_contract = view_manifest.get("full_cs_to_depot_cache", {})
    if cache_contract != FULL_CS_TO_DEPOT_CACHE_CONTRACT:
        raise ValueError("Full-CS-to-depot cache contract mismatch")
    with np.load(
        view_root / view_manifest["charging_attributes"], allow_pickle=False
    ) as data:
        charging_power = data["charging_power_kw"].astype(np.float32, copy=False)
        full_cs_to_depot_time = data["full_cs_to_depot_time_s"].astype(
            np.float32, copy=False
        )
    coordinates = terminals[["longitude", "latitude"]].to_numpy(dtype=np.float32)
    customer_terminals = terminals.iloc[1 : 1 + customer_count]
    def _customer_strings(column: str) -> np.ndarray:
        if column not in customer_terminals:
            return np.full(customer_count, "", dtype=str)
        return customer_terminals[column].fillna("").astype(str).to_numpy()

    payload = {
        "schema": "cle_evrptw_loaded_view_v3",
        "instance_id": view_id,
        "family_id": str(family_manifest["family_id"]),
        "city_slug": str(family_manifest["city_slug"]),
        "day_type": str(view_manifest["day_type"]),
        "working_start_s": int(view_manifest["operating_horizon_s"][0]),
        "working_end_s": int(view_manifest["operating_horizon_s"][1]),
        "depot": coordinates[0],
        "customers": coordinates[1 : 1 + customer_count],
        "charging_stations": coordinates[1 + customer_count :],
        "terminal_source_ids": terminals["source_id"].astype(str).to_numpy(),
        "order_template_ids": _customer_strings("order_template_id"),
        "order_station_day_ids": _customer_strings("order_station_day_id"),
        "order_template_source_modes": _customer_strings("order_source_mode"),
        "terminal_parent_indices": parent_indices,
        "package_counts": attributes["package_counts"].astype(np.int32, copy=False),
        "demands_cm3": attributes["demands_cm3"].astype(np.float32, copy=False),
        "service_time_s": attributes["service_time_s"].astype(np.float32, copy=False),
        "tw_s": attributes["time_windows_s"].astype(np.float32, copy=False),
        "order_sampling_attempts": attributes.get(
            "order_sampling_attempts",
            np.ones(customer_count, dtype=np.int16),
        ).astype(np.int16, copy=False),
        "feasibility_certificate": {
            "certificate_service_arrival_time_s": attributes[
                "feasible_arrival_time_s"
            ].astype(
                np.float32, copy=False
            ),
            "return_duration_s": attributes["feasible_return_duration_s"].astype(
                np.float32, copy=False
            ),
            "requires_charging": attributes["feasibility_requires_charging"].astype(
                bool, copy=False
            ),
            "charging_visit_count": attributes[
                "feasibility_charging_visit_count"
            ].astype(np.int16, copy=False),
            "inbound_full_state_terminal_index": attributes[
                "feasibility_inbound_full_state_terminal_index"
            ].astype(np.int32, copy=False),
            "first_post_customer_charger_terminal_index": attributes[
                "feasibility_first_post_customer_charger_terminal_index"
            ].astype(np.int32, copy=False),
            "customer_transition_energy_margin_kwh": attributes[
                "feasibility_energy_margin_kwh"
            ].astype(np.float32, copy=False),
        },
        "charging_power_kw": charging_power,
        "full_cs_to_depot_time_s": full_cs_to_depot_time,
        "vehicle": dict(view_manifest["vehicle"]),
        "charging_policy": dict(view_manifest["charging_policy"]),
        "runtime_mask": None,
        "metadata": {
            "scale_id": view_manifest["scale_id"],
            "consumer_cohort_id": view_manifest["consumer_cohort_id"],
            "split_id": view_manifest["split_id"],
            "track_id": view_manifest["track_id"],
            "generation_mode": view_manifest.get(
                "generation_mode",
                "non_release_pilot" if view_manifest["non_release_pilot"] else "official",
            ),
            "non_release_pilot": bool(view_manifest["non_release_pilot"]),
            "reference_profile_id": family_manifest["reference_profile_id"],
            "source_view_schema": str(view_manifest["schema"]),
            "full_cs_to_depot_cache_source": "stored",
            "energy_matrix_source": "derived_from_path_distance",
        },
        **matrices,
    }
    if payload["customers"].shape != (customer_count, 2):
        raise ValueError("Loaded customer coordinate shape mismatch")
    if payload["charging_stations"].shape != (charger_count, 2):
        raise ValueError("Loaded charging-station coordinate shape mismatch")
    if payload["charging_power_kw"].shape != (charger_count,):
        raise ValueError("Loaded charging-power shape mismatch")
    if payload["full_cs_to_depot_time_s"].shape != (charger_count,):
        raise ValueError("Loaded full-CS-to-depot cache shape mismatch")
    return payload


def _verify_view_feasibility_certificate(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    customer_count = len(payload["customers"])
    certificate = payload["feasibility_certificate"]
    if any(len(values) != customer_count for values in certificate.values()):
        return ["feasibility-certificate length mismatch"]

    arrival = certificate["certificate_service_arrival_time_s"]
    return_duration = certificate["return_duration_s"]
    margin = certificate["customer_transition_energy_margin_kwh"]
    if not np.isfinite(arrival).all():
        errors.append("non-finite feasibility arrival time")
    if not np.isfinite(return_duration).all():
        errors.append("non-finite feasibility return duration")
    if np.any(margin < -1e-5):
        errors.append("negative feasibility energy margin")
    cached_return = payload["full_cs_to_depot_time_s"]
    if np.isnan(cached_return).any():
        errors.append("full-CS-to-depot cache contains NaN")
    if np.any(cached_return < -1e-5):
        errors.append("full-CS-to-depot cache contains negative time")

    vehicle = payload["vehicle"]
    charging = payload["charging_policy"]
    recomputed = _single_customer_certificates(
        customer_count=customer_count,
        running_time_matrix_s=payload["running_time_shortest_matrix_s"],
        running_time_path_distance_matrix_km=payload[
            "running_time_path_distance_km"
        ],
        specific_energy_consumption_kwh_per_km=float(
            vehicle["specific_energy_consumption_kwh_per_km"]
        ),
        charging_power_kw=payload["charging_power_kw"],
        battery_capacity_kwh=float(vehicle["battery_capacity_kwh"]),
        charging_power_derating_factor=float(
            charging["charging_power_derating_factor"]
        ),
    )
    expected_arrival = (
        float(payload["working_start_s"]) + recomputed.arrival_elapsed_s
    ).astype(np.float32)
    float_checks = (
        ("service-arrival time", arrival, expected_arrival),
        ("return duration", return_duration, recomputed.return_duration_s),
        (
            "energy margin",
            margin,
            recomputed.customer_transition_energy_margin_kwh,
        ),
    )
    for label, stored, expected in float_checks:
        if not np.allclose(stored, expected, rtol=1e-6, atol=1e-4, equal_nan=False):
            errors.append(f"stored feasibility {label} does not match recomputation")
    exact_checks = (
        (
            "requires-charging flags",
            certificate["requires_charging"],
            recomputed.requires_charging,
        ),
        (
            "charging-visit counts",
            certificate["charging_visit_count"],
            recomputed.charging_visit_count,
        ),
        (
            "inbound full-state indices",
            certificate["inbound_full_state_terminal_index"],
            recomputed.inbound_full_state_terminal_index,
        ),
        (
            "first post-customer charger indices",
            certificate["first_post_customer_charger_terminal_index"],
            recomputed.first_post_customer_charger_terminal_index,
        ),
    )
    for label, stored, expected in exact_checks:
        if not np.array_equal(stored, expected):
            errors.append(f"stored feasibility {label} do not match recomputation")
    if not np.allclose(
        cached_return,
        recomputed.full_cs_to_depot_time_s,
        rtol=1e-6,
        atol=1e-4,
        equal_nan=False,
    ):
        errors.append("stored full-CS-to-depot cache does not match recomputation")

    service_start = np.maximum(arrival, payload["tw_s"][:, 0])
    if np.any(service_start > payload["tw_s"][:, 1] + 1e-5):
        errors.append("certificate violates a customer time window")
    route_end = service_start + payload["service_time_s"] + return_duration
    if np.any(route_end > float(payload["working_end_s"]) + 1e-5):
        errors.append("certificate returns after the operating horizon")
    if np.any(payload["demands_cm3"] > float(vehicle["cargo_capacity_cm3"]) + 1e-5):
        errors.append("one-customer demand exceeds vehicle volume capacity")
    if np.any(payload["charging_power_kw"] <= 0.0):
        errors.append("charging power must be positive")
    return errors


def verify_materialized_family(family_dir: str | Path) -> dict[str, Any]:
    root = Path(family_dir)
    manifest = _read_json(root / "family_manifest.json")
    errors: list[str] = []
    warnings: list[str] = []
    terminal_index = pd.read_parquet(root / manifest["terminal_index"])
    terminal_count = int(manifest["terminal_count"])
    if len(terminal_index) != terminal_count:
        errors.append("terminal_index row count does not match family manifest")
    if terminal_index["source_id"].astype(str).duplicated().any():
        errors.append("terminal_index contains duplicate source IDs")
    current_contract = (
        manifest.get("stage2_generation_contract") == STAGE2_GENERATION_CONTRACT
    )
    customer_count = int(manifest.get("parent_customer_count", 0))
    customer_rows = terminal_index.iloc[1 : 1 + customer_count]
    if current_contract:
        for column in (
            "order_template_id",
            "order_station_day_id",
            "order_source_mode",
        ):
            if column not in customer_rows:
                errors.append(f"terminal_index is missing {column}")
            elif customer_rows[column].isna().any():
                errors.append(f"terminal_index has missing customer {column}")

    phase1_paths = manifest.get("phase1_metric_files", {})
    expected_phase1 = {
        "phase1_metrics",
        "phase1_observations",
        "phase1_region_pair_metrics",
    }
    if current_contract and set(phase1_paths) != expected_phase1:
        errors.append("family manifest does not declare the three Phase-1 metric files")
    elif phase1_paths:
        for label, relative in phase1_paths.items():
            if not (root / relative).is_file():
                errors.append(f"missing {label} file")
        metrics_path = root / phase1_paths["phase1_metrics"]
        observations_path = root / phase1_paths["phase1_observations"]
        if metrics_path.is_file():
            phase1 = _read_json(metrics_path)
            if phase1.get("schema") != PHASE1_FAMILY_METRICS_SCHEMA:
                errors.append("Phase-1 family metric schema mismatch")
            if phase1.get("family_id") != manifest.get("family_id"):
                errors.append("Phase-1 family metric family_id mismatch")
            if not bool(phase1.get("hard_gates", {}).get("passed", False)):
                errors.append("Phase-1 hard correctness gates did not pass")
        if observations_path.is_file():
            phase1_observations = pd.read_parquet(observations_path)
            if len(phase1_observations) != customer_count:
                errors.append("Phase-1 observation row count does not match parent customers")
    matrix_metrics: dict[str, Any] = {}
    stored_names = tuple(manifest["matrix_files"])
    expected_stored = set(STORED_MATRIX_NAMES)
    if (
        manifest.get("schema") == "cle_evrptw_materialized_matrix_family_v3"
        and set(stored_names) != expected_stored
    ):
        errors.append("v2 family must persist exactly the four contracted matrices")
    for name in stored_names:
        path = root / manifest["matrix_files"][name]
        if not path.is_file():
            errors.append(f"missing matrix: {name}")
            continue
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        if matrix.shape != (terminal_count, terminal_count):
            errors.append(f"{name} shape is {matrix.shape}, expected {(terminal_count, terminal_count)}")
            continue
        if matrix.dtype != np.float32:
            errors.append(f"{name} dtype is {matrix.dtype}, expected float32")
        if not np.isfinite(matrix).all():
            errors.append(f"{name} contains non-finite values")
        if np.any(matrix < -1e-7):
            errors.append(f"{name} contains negative values")
        if np.max(np.abs(np.diag(matrix))) > 1e-5:
            errors.append(f"{name} diagonal is not zero")
        off_diagonal = ~np.eye(terminal_count, dtype=bool)
        matrix_metrics[name] = {
            "min_off_diagonal": float(np.min(matrix[off_diagonal])),
            "median_off_diagonal": float(np.median(matrix[off_diagonal])),
            "max_off_diagonal": float(np.max(matrix[off_diagonal])),
            "asymmetric_pair_fraction": float(
                np.mean(np.abs(matrix - matrix.T)[off_diagonal] > 1e-6)
            ),
        }
    view_ids = list(map(str, manifest["view_ids"]))
    if len(view_ids) != int(manifest["view_count"]):
        errors.append("view_count does not match view_ids")
    scale_counts: dict[str, int] = {}
    certified_customer_count = 0
    requires_charging_customer_count = 0
    maximum_charging_visit_count = 0
    stored_full_cs_cache_view_count = 0
    unreachable_full_cs_return_count = 0
    for view_id in view_ids:
        try:
            payload = load_materialized_view(root, view_id)
            scale_id = str(payload["metadata"]["scale_id"])
            scale_counts[scale_id] = scale_counts.get(scale_id, 0) + 1
            customer_count = len(payload["customers"])
            terminal_view_count = 1 + customer_count + len(payload["charging_stations"])
            for name in MATRIX_NAMES:
                if payload[name].shape != (terminal_view_count, terminal_view_count):
                    errors.append(f"{view_id}: {name} view shape mismatch")
            specific_energy = float(
                payload["vehicle"]["specific_energy_consumption_kwh_per_km"]
            )
            if not np.allclose(
                payload["distance_path_energy_kwh"],
                payload["distance_matrix_km"] * specific_energy,
                rtol=1e-6,
                atol=1e-5,
            ):
                errors.append(f"{view_id}: distance-path energy is not linear in distance")
            if not np.allclose(
                payload["running_time_path_energy_kwh"],
                payload["running_time_path_distance_km"] * specific_energy,
                rtol=1e-6,
                atol=1e-5,
            ):
                errors.append(
                    f"{view_id}: running-time-path energy is not linear in distance"
                )
            if len(payload["package_counts"]) != customer_count:
                errors.append(f"{view_id}: package count length mismatch")
            if len(payload["demands_cm3"]) != customer_count:
                errors.append(f"{view_id}: demand length mismatch")
            if payload["tw_s"].shape != (customer_count, 2):
                errors.append(f"{view_id}: time-window shape mismatch")
            certificate = payload["feasibility_certificate"]
            for error in _verify_view_feasibility_certificate(payload):
                errors.append(f"{view_id}: {error}")
            stored_full_cs_cache_view_count += 1
            unreachable_full_cs_return_count += int(
                (~np.isfinite(payload["full_cs_to_depot_time_s"])).sum()
            )
            certified_customer_count += customer_count
            requires_charging_customer_count += int(
                certificate["requires_charging"].sum()
            )
            if customer_count:
                maximum_charging_visit_count = max(
                    maximum_charging_visit_count,
                    int(certificate["charging_visit_count"].max()),
                )
            if payload["runtime_mask"] is not None:
                errors.append(f"{view_id}: runtime mask must not be stored")
        except Exception as error:  # noqa: BLE001 - verifier reports every broken view it can.
            errors.append(f"{view_id}: {error}")
    if bool(manifest.get("non_release_pilot", False)):
        warnings.append("Family is a non-release pilot and cannot be published as an official split.")
    if manifest.get("reference_profile_status") != "release_calibrated":
        warnings.append("Reference operations profile is not release calibrated.")
    return {
        "schema": "cle_evrptw_materialized_family_verification_v1",
        "family_id": str(manifest["family_id"]),
        "city_slug": str(manifest["city_slug"]),
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "terminal_count": terminal_count,
        "view_count": len(view_ids),
        "view_counts_by_scale": scale_counts,
        "certified_customer_count": certified_customer_count,
        "requires_charging_customer_count": requires_charging_customer_count,
        "maximum_charging_visit_count": maximum_charging_visit_count,
        "stored_full_cs_cache_view_count": stored_full_cs_cache_view_count,
        "unreachable_full_cs_return_count": unreachable_full_cs_return_count,
        "matrix_total_bytes": int(manifest["matrix_total_bytes"]),
        "matrix_metrics": matrix_metrics,
    }
