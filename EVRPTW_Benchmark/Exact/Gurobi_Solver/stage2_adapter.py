from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from evrptw_core.schema import EVRPTWInstance
from evrptw_stage2.artifacts import load_materialized_view
from evrptw_stage2.contracts import STAGE2_GENERATION_CONTRACT


FAMILY_SCHEMA = "cle_evrptw_materialized_matrix_family_v3"
VIEW_SCHEMA = "cle_evrptw_materialized_view_v4"
VIEW_MATRIX_STORAGE = "parent_index_view"
REQUIRED_VIEW_INDEX_COLUMNS = {
    "view_id",
    "family_id",
    "consumer_cohort_id",
    "split_id",
    "track_id",
    "city_slug",
    "scale_id",
    "customer_count",
    "charging_station_count",
    "family_cohort_id",
    "terminal_count",
    "view_seed",
}


@dataclass(frozen=True)
class Stage2ViewTask:
    """A lightweight reference to one materialized Stage-2 view."""

    index_path: str
    family_dir: str
    view_id: str
    family_id: str
    consumer_cohort_id: str
    split_id: str
    track_id: str
    city_slug: str
    scale_id: str
    customer_count: int
    charging_station_count: int
    row_position: int
    family_cohort_id: str = ""
    terminal_count: int = 0
    view_seed: int = 0

    @property
    def instance_id(self) -> str:
        return self.view_id

    @property
    def scale_label(self) -> str:
        suffix = self.scale_id.lower().removeprefix("cus")
        return f"Cus{int(suffix)}"


def discover_view_indices(dataset_path: str | Path) -> list[Path]:
    root = Path(dataset_path)
    if root.is_file():
        if root.name == "view_index.parquet":
            return [root]
        if root.suffix.lower() == ".parquet":
            raise ValueError(
                "Stage-2 input must use the canonical filename "
                f"view_index.parquet, got {root.name}"
            )
        return []
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(root.rglob("view_index.parquet"))


def infer_family_root(index_path: str | Path) -> Path | None:
    """Resolve the canonical sibling materialized/families directory."""

    path = Path(index_path).resolve()
    for ancestor in path.parents:
        if ancestor.name == "generation_plan":
            candidate = ancestor.parent / "materialized" / "families"
            if candidate.is_dir():
                return candidate
        candidate = ancestor / "materialized" / "families"
        if candidate.is_dir():
            return candidate
    return None


def read_stage2_tasks(
    dataset_path: str | Path,
    *,
    family_root: str | Path | None = None,
) -> list[Stage2ViewTask]:
    index_paths = discover_view_indices(dataset_path)
    if not index_paths:
        return []

    explicit_family_root = Path(family_root).resolve() if family_root else None
    tasks: list[Stage2ViewTask] = []
    position = 0
    for index_path in index_paths:
        resolved_family_root = explicit_family_root or infer_family_root(index_path)
        if resolved_family_root is None:
            raise ValueError(
                "Could not infer materialized family root for "
                f"{index_path}. Pass --family_root explicitly."
            )
        frame = pd.read_parquet(index_path)
        missing = sorted(REQUIRED_VIEW_INDEX_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(f"{index_path} is missing view-index columns: {missing}")
        if frame["view_id"].astype(str).duplicated().any():
            raise ValueError(f"{index_path} contains duplicate view_id values")
        for row in frame.to_dict(orient="records"):
            family_id = str(row["family_id"])
            family_dir = resolved_family_root / family_id
            tasks.append(
                Stage2ViewTask(
                    index_path=str(index_path.resolve()),
                    family_dir=str(family_dir),
                    view_id=str(row["view_id"]),
                    family_id=family_id,
                    consumer_cohort_id=str(row["consumer_cohort_id"]),
                    split_id=str(row["split_id"]),
                    track_id=str(row["track_id"]),
                    city_slug=str(row["city_slug"]),
                    scale_id=str(row["scale_id"]),
                    customer_count=int(row["customer_count"]),
                    charging_station_count=int(row["charging_station_count"]),
                    row_position=position,
                    family_cohort_id=str(row["family_cohort_id"]),
                    terminal_count=int(row["terminal_count"]),
                    view_seed=int(row["view_seed"]),
                )
            )
            position += 1
    return tasks


def missing_family_directories(tasks: Iterable[Stage2ViewTask]) -> list[Path]:
    return sorted(
        {
            Path(task.family_dir)
            for task in tasks
            if not (Path(task.family_dir) / "family_manifest.json").is_file()
        }
    )


def load_stage2_instance(task: Stage2ViewTask) -> EVRPTWInstance:
    family_dir = Path(task.family_dir)
    family_manifest = json.loads(
        (family_dir / "family_manifest.json").read_text(encoding="utf-8")
    )
    view_manifest = json.loads(
        (family_dir / "views" / task.view_id / "view_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if family_manifest.get("schema") != FAMILY_SCHEMA:
        raise ValueError(
            f"Unsupported Stage-2 family schema: {family_manifest.get('schema')!r}"
        )
    if view_manifest.get("schema") != VIEW_SCHEMA:
        raise ValueError(
            f"Unsupported Stage-2 view schema: {view_manifest.get('schema')!r}"
        )
    if (
        family_manifest.get("stage2_generation_contract")
        != STAGE2_GENERATION_CONTRACT
    ):
        raise ValueError("Stage-2 generation contract mismatch")
    if view_manifest.get("matrix_storage") != VIEW_MATRIX_STORAGE:
        raise ValueError("Stage-2 view does not use parent_index_view matrix storage")
    if family_manifest.get("materialization_status") != "complete":
        raise ValueError("Stage-2 family is not completely materialized")
    if str(view_manifest.get("view_id")) != task.view_id:
        raise ValueError("View manifest ID does not match view index")
    if str(view_manifest.get("family_id")) != task.family_id:
        raise ValueError("View manifest family ID does not match view index")

    payload = load_materialized_view(task.family_dir, task.view_id)
    if payload["instance_id"] != task.view_id:
        raise ValueError("Loaded Stage-2 view ID does not match view index")
    if payload["family_id"] != task.family_id:
        raise ValueError("Loaded Stage-2 family ID does not match view index")
    if len(payload["customers"]) != task.customer_count:
        raise ValueError("Loaded customer count does not match view index")
    if len(payload["charging_stations"]) != task.charging_station_count:
        raise ValueError("Loaded charging-station count does not match view index")
    if task.terminal_count != 1 + task.customer_count + task.charging_station_count:
        raise ValueError("View-index terminal count is inconsistent")

    vehicle = dict(payload["vehicle"])
    vehicle["charging_power_derating_factor"] = float(
        payload["charging_policy"]["charging_power_derating_factor"]
    )
    metadata: dict[str, Any] = {
        **dict(payload["metadata"]),
        "family_id": task.family_id,
        "view_id": task.view_id,
        "city_slug": task.city_slug,
        "split_id": task.split_id,
        "track_id": task.track_id,
        "consumer_cohort_id": task.consumer_cohort_id,
        "family_cohort_id": task.family_cohort_id,
        "view_seed": task.view_seed,
        "source_view_index": task.index_path,
        "source_family_dir": task.family_dir,
        "source_family_schema": family_manifest["schema"],
        "source_view_schema": view_manifest["schema"],
        "stage2_generation_contract": family_manifest[
            "stage2_generation_contract"
        ],
        "matrix_storage": view_manifest["matrix_storage"],
        "metric_contract": {
            "objective": "distance_matrix_km",
            "travel_time": "running_time_shortest_matrix_s",
            "energy": "running_time_path_energy_kwh",
        },
    }
    canonical_payload: dict[str, Any] = {
        "instance_id": task.view_id,
        "region_id": task.city_slug,
        "mother_board_id": task.family_id,
        "operating_day_id": task.family_id,
        "day_type": payload["day_type"],
        "working_start_s": payload["working_start_s"],
        "working_end_s": payload["working_end_s"],
        "depot": payload["depot"],
        "customers": payload["customers"],
        "charging_stations": payload["charging_stations"],
        "distance_matrix_km": payload["distance_matrix_km"],
        "demands_cm3": payload["demands_cm3"],
        "package_counts": payload["package_counts"],
        "service_time_s": payload["service_time_s"],
        "tw_s": payload["tw_s"],
        "cs_time_to_depot_s": payload["full_cs_to_depot_time_s"],
        "vehicle": vehicle,
        # These canonical compatibility fields carry the V1 exact-solver
        # contract.  The original Stage-2 names remain below for provenance.
        "raw_travel_time_matrix_s": payload["running_time_shortest_matrix_s"],
        "shortest_time_matrix_s": payload["running_time_shortest_matrix_s"],
        "energy_matrix_kwh": payload["running_time_path_energy_kwh"],
        "speed_profile": {
            "matrix_source": "running_time_shortest_matrix_s",
            "reference_profile_id": payload["metadata"]["reference_profile_id"],
        },
        "cs_activation": {
            "charging_power_kw": payload["charging_power_kw"],
        },
        "metadata": metadata,
        "charging_power_kw": payload["charging_power_kw"],
        "charging_policy": payload["charging_policy"],
        "running_time_shortest_matrix_s": payload[
            "running_time_shortest_matrix_s"
        ],
        "running_time_path_energy_kwh": payload[
            "running_time_path_energy_kwh"
        ],
        "running_time_path_distance_km": payload[
            "running_time_path_distance_km"
        ],
        "distance_path_travel_time_s": payload["distance_path_travel_time_s"],
        "full_cs_to_depot_time_s": payload["full_cs_to_depot_time_s"],
        "terminal_parent_indices": payload["terminal_parent_indices"],
        "feasibility_certificate": payload.get("feasibility_certificate"),
    }
    return EVRPTWInstance.from_dict(canonical_payload)
