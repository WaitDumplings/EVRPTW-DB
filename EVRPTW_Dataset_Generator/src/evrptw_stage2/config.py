"""Versioned configuration contract for CLE-backed Stage-2 generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScaleConfig:
    scale_id: str
    customers: int
    charging_stations: int
    role: str
    train_views: int
    validation_instances: int
    test_instances: dict[str, int]

    @property
    def terminal_count(self) -> int:
        return 1 + self.customers + self.charging_stations


@dataclass(frozen=True)
class Stage2Config:
    schema: str
    dataset_id: str
    benchmark_version: str
    master_seed: int
    train_cities: tuple[str, ...]
    heldout_city: str
    operating_horizon_start_s: int
    operating_horizon_end_s: int
    weekday_weight: int
    weekend_weight: int
    demand_unit: str
    capacity_unit: str
    heldout_community_fraction: float
    community_partition_version: str
    matrix_dtype: str
    matrix_path_policy: str
    running_time_path_policy: str
    matrix_bundle_id: str
    stored_parent_matrix_count: int
    parent_scale_id: str
    train_parent_family_count: int
    validation_parent_family_count: int
    core_test_parent_family_count: int
    scalability_parent_family_count: int
    vehicle_dc_cap_kw: float
    vehicle_ac_l2_cap_kw: float
    scales: tuple[ScaleConfig, ...]
    raw: dict[str, Any]

    def scale(self, scale_id: str) -> ScaleConfig:
        for scale in self.scales:
            if scale.scale_id == scale_id:
                return scale
        raise KeyError(f"Unknown scale_id: {scale_id}")

    def validate(self) -> None:
        errors: list[str] = []
        if self.schema != "cle_evrptw_stage2_config_v2":
            errors.append(f"Unsupported config schema: {self.schema!r}")
        if len(self.train_cities) != 10 or len(set(self.train_cities)) != 10:
            errors.append("train_cities must contain ten unique city slugs")
        if self.heldout_city in self.train_cities:
            errors.append("heldout_city must not occur in train_cities")
        if self.operating_horizon_start_s != 8 * 3600:
            errors.append("V2 operating horizon must start at 08:00")
        if self.operating_horizon_end_s != 24 * 3600:
            errors.append("V2 operating horizon must end at 24:00")
        if (self.weekday_weight, self.weekend_weight) != (5, 2):
            errors.append("V2 weekday/weekend weights must be 5:2")
        if self.demand_unit != "cm3" or self.capacity_unit != "cm3":
            errors.append("V2 demand and capacity must both use cm3")
        if not 0.0 < self.heldout_community_fraction < 1.0:
            errors.append("heldout_community_fraction must be in (0, 1)")
        if self.matrix_dtype != "float32":
            errors.append("V2 matrix_dtype must be float32")
        if self.matrix_path_policy != "distance_shortest_directed_physical_path_v1":
            errors.append("V2 matrix path policy must be the frozen distance-shortest policy")
        if (
            self.running_time_path_policy
            != "directed_zero_turn_shortest_running_time_v3"
        ):
            errors.append("V2 running-time path policy must be canonical zero-turn")
        if self.matrix_bundle_id != "dual_path_four_matrix_v3":
            errors.append("V2 matrix bundle must be dual_path_four_matrix_v3")
        if self.stored_parent_matrix_count != 4:
            errors.append("V2 dual-path matrix families must contain four parent matrices")
        expected = {
            "cus50": (50, 10, 100_000, 500),
            "cus100": (100, 20, 50_000, 500),
            "cus500": (500, 50, 10_000, 500),
            "cus1000": (1000, 50, 5_000, 500),
            "cus2000": (2000, 50, 0, 0),
        }
        actual_ids = {scale.scale_id for scale in self.scales}
        if actual_ids != set(expected):
            errors.append(f"V2 scales must be {sorted(expected)}, got {sorted(actual_ids)}")
        for scale in self.scales:
            exp = expected.get(scale.scale_id)
            if exp and (
                scale.customers,
                scale.charging_stations,
                scale.train_views,
                scale.validation_instances,
            ) != exp:
                errors.append(
                    f"{scale.scale_id} contract mismatch: got "
                    f"{(scale.customers, scale.charging_stations, scale.train_views, scale.validation_instances)}, "
                    f"expected {exp}"
                )
            if scale.train_views and scale.customers * scale.train_views != 5_000_000:
                errors.append(
                    f"{scale.scale_id} violates the 5M active-customer exposure contract"
                )
        if self.parent_scale_id != "cus1000" or self.train_parent_family_count != 5_000:
            errors.append("V2 train storage must use 5,000 cus1000 parent families")
        if self.validation_parent_family_count != 500:
            errors.append("V2 validation must use 500 parent families")
        if self.core_test_parent_family_count != 500:
            errors.append("Each V2 core test track must use 500 parent families")
        if self.scalability_parent_family_count != 500:
            errors.append("V2 Cus2000 scalability test must use 500 parent families")
        if self.vehicle_dc_cap_kw != 100.0 or self.vehicle_ac_l2_cap_kw != 11.0:
            errors.append("V2 reference charge caps must be DC 100 kW and AC L2 11 kW")
        source = self.raw.get("amazon_source", {})
        if source.get("cohort_split_config") != "configs/amazon_cohort_split_v1.json":
            errors.append("V2 must use the frozen Amazon cohort split")
        if source.get("primary_source_mode") != ["SINGLE_STRUCTURE_DAY", "SINGLE_ORDER_DAY"]:
            errors.append("V2 primary source mode must be single-day for structure and orders")
        acceptance = self.raw.get("acceptance", {})
        if acceptance.get("realism_gate") != "station_block_q90_m2_m3_v1":
            errors.append("V2 must freeze the station-block Q90 M2/M3 gate")
        output = self.raw.get("output", {})
        if output.get("calibration_root") != "Calibration_v2" or output.get("instances_root") != "Instances_v2":
            errors.append("V2 outputs must use Calibration_v2 and Instances_v2")
        if errors:
            raise ValueError("Invalid Stage-2 config:\n- " + "\n- ".join(errors))


def _scale_from_dict(payload: dict[str, Any]) -> ScaleConfig:
    return ScaleConfig(
        scale_id=str(payload["scale_id"]),
        customers=int(payload["customers"]),
        charging_stations=int(payload["charging_stations"]),
        role=str(payload["role"]),
        train_views=int(payload.get("train_views", 0)),
        validation_instances=int(payload.get("validation_instances", 0)),
        test_instances={
            str(key): int(value) for key, value in payload.get("test_instances", {}).items()
        },
    )


def load_stage2_config(path: str | Path) -> Stage2Config:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    horizon = payload["operating_horizon"]
    day_type = payload["day_type"]
    split = payload["customer_split"]
    family = payload["matrix_family"]
    vehicle = payload["reference_vehicle"]
    config = Stage2Config(
        schema=str(payload["schema"]),
        dataset_id=str(payload["dataset_id"]),
        benchmark_version=str(payload["benchmark_version"]),
        master_seed=int(payload["master_seed"]),
        train_cities=tuple(map(str, payload["cities"]["train"])),
        heldout_city=str(payload["cities"]["heldout"]),
        operating_horizon_start_s=int(horizon["start_s"]),
        operating_horizon_end_s=int(horizon["end_s"]),
        weekday_weight=int(day_type["weekday_weight"]),
        weekend_weight=int(day_type["weekend_weight"]),
        demand_unit=str(payload["demand"]["unit"]),
        capacity_unit=str(payload["demand"]["vehicle_capacity_unit"]),
        heldout_community_fraction=float(split["heldout_fraction"]),
        community_partition_version=str(split["partition_version"]),
        matrix_dtype=str(family["dtype"]),
        matrix_path_policy=str(family["path_policy_id"]),
        running_time_path_policy=str(family["running_time_path_policy_id"]),
        matrix_bundle_id=str(family["matrix_bundle_id"]),
        stored_parent_matrix_count=int(family["stored_parent_matrix_count"]),
        parent_scale_id=str(family["parent_scale_id"]),
        train_parent_family_count=int(family["train_parent_families"]),
        validation_parent_family_count=int(family["validation_parent_families"]),
        core_test_parent_family_count=int(family["core_test_parent_families_per_track"]),
        scalability_parent_family_count=int(family["scalability_parent_families"]),
        vehicle_dc_cap_kw=float(vehicle["dc_cap_kw"]),
        vehicle_ac_l2_cap_kw=float(vehicle["ac_l2_cap_kw"]),
        scales=tuple(_scale_from_dict(item) for item in payload["scales"]),
        raw=payload,
    )
    config.validate()
    return config
