from __future__ import annotations

import math
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from evrptw_core.schema import EVRPTWInstance, merge_route_sequences
from evrptw_stage2.artifacts import load_materialized_view


DEFAULT_CHECKPOINTS_S = (60.0, 300.0, 900.0, 3600.0, 7200.0)
DEFAULT_TIME_LIMIT_S = 7200.0

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
}


class SolverTimeLimit(TimeoutError):
    pass


@contextmanager
def hard_time_limit(seconds: float):
    """Interrupt pure-Python heuristic work at the experiment wall-clock cap."""

    duration = max(0.001, float(seconds))
    previous_handler = signal.getsignal(signal.SIGALRM)

    def raise_timeout(signum, frame):  # noqa: ARG001
        raise SolverTimeLimit(f"solver wall-clock limit reached after {duration:g}s")

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, duration)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


@dataclass(frozen=True)
class Stage2ViewTask:
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

    @property
    def instance_id(self) -> str:
        return self.view_id

    @property
    def scale_label(self) -> str:
        return normalize_scale(self.scale_id)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Stage2ViewTask":
        return cls(**data)


def normalize_scale(value: str | int) -> str:
    raw = str(value).strip()
    suffix = raw[3:] if raw.lower().startswith("cus") else raw
    return f"Cus{int(suffix)}"


def parse_scales(raw: str) -> set[str]:
    return {normalize_scale(item) for item in raw.split(",") if item.strip()}


def parse_checkpoints(raw: str) -> tuple[float, ...]:
    if not raw.strip():
        return tuple()
    values = tuple(sorted({float(item) for item in raw.split(",") if item.strip()}))
    if not values or any(value < 0.0 for value in values):
        raise ValueError("checkpoints must be a non-empty list of non-negative seconds")
    return values


def resolve_schedule(
    checkpoints_s: Iterable[float],
    time_limit_s: float | None,
) -> tuple[tuple[float, ...], float]:
    checkpoints = tuple(sorted({float(value) for value in checkpoints_s}))
    if not checkpoints:
        if time_limit_s is None:
            return DEFAULT_CHECKPOINTS_S, DEFAULT_TIME_LIMIT_S
        limit = float(time_limit_s)
        if limit <= 0.0:
            raise ValueError("time_limit_s must be positive")
        selected = tuple(value for value in DEFAULT_CHECKPOINTS_S if value <= limit)
        if not selected or selected[-1] != limit:
            selected = (*selected, limit)
        return selected, limit

    requested_limit = checkpoints[-1] if time_limit_s is None else float(time_limit_s)
    if requested_limit <= 0.0:
        raise ValueError("time_limit_s must be positive")
    # Explicit checkpoints are part of the experiment contract, so the run
    # cannot stop before the last one.
    return checkpoints, max(requested_limit, checkpoints[-1])


def checkpoint_label(checkpoint_s: float) -> str:
    value = float(checkpoint_s)
    return f"{int(value)}s" if value.is_integer() else f"{value:g}s".replace(".", "p")


def discover_view_indices(dataset_path: str | Path) -> list[Path]:
    root = Path(dataset_path)
    if root.is_file():
        if root.name == "view_index.parquet":
            return [root]
        return []
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(root.rglob("view_index.parquet"))


def infer_family_root(index_path: str | Path) -> Path | None:
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
    explicit_root = Path(family_root).resolve() if family_root else None
    tasks: list[Stage2ViewTask] = []
    position = 0
    for index_path in discover_view_indices(dataset_path):
        resolved_root = explicit_root or infer_family_root(index_path)
        if resolved_root is None:
            raise ValueError(
                f"Cannot infer materialized/families for {index_path}; pass --family_root"
            )
        frame = pd.read_parquet(index_path)
        missing = sorted(REQUIRED_VIEW_INDEX_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(f"{index_path} is missing columns: {missing}")
        if frame["view_id"].astype(str).duplicated().any():
            raise ValueError(f"{index_path} contains duplicate view_id values")
        for row in frame.to_dict(orient="records"):
            family_id = str(row["family_id"])
            tasks.append(
                Stage2ViewTask(
                    index_path=str(index_path.resolve()),
                    family_dir=str((resolved_root / family_id).resolve()),
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
    payload = load_materialized_view(task.family_dir, task.view_id)
    if str(payload["instance_id"]) != task.view_id:
        raise ValueError("loaded Stage-2 view_id does not match view index")
    if str(payload["family_id"]) != task.family_id:
        raise ValueError("loaded Stage-2 family_id does not match view index")
    if len(payload["customers"]) != task.customer_count:
        raise ValueError("loaded customer count does not match view index")
    if len(payload["charging_stations"]) != task.charging_station_count:
        raise ValueError("loaded charging-station count does not match view index")

    vehicle = dict(payload["vehicle"])
    vehicle["charging_efficiency"] = float(
        payload["charging_policy"]["charging_efficiency"]
    )
    metadata = {
        **dict(payload["metadata"]),
        "family_id": task.family_id,
        "view_id": task.view_id,
        "city_slug": task.city_slug,
        "split_id": task.split_id,
        "track_id": task.track_id,
        "consumer_cohort_id": task.consumer_cohort_id,
        "source_view_index": task.index_path,
        "source_family_dir": task.family_dir,
        "metric_contract": {
            "objective": "distance_matrix_km",
            "travel_time": "running_time_shortest_matrix_s",
            "energy": "running_time_path_energy_kwh",
        },
    }
    return EVRPTWInstance.from_dict(
        {
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
            "raw_travel_time_matrix_s": payload["running_time_shortest_matrix_s"],
            "shortest_time_matrix_s": payload["running_time_shortest_matrix_s"],
            "energy_matrix_kwh": payload["running_time_path_energy_kwh"],
            "speed_profile": {
                "matrix_source": "running_time_shortest_matrix_s",
                "reference_profile_id": payload["metadata"]["reference_profile_id"],
            },
            "cs_activation": {"charging_power_kw": payload["charging_power_kw"]},
            "metadata": metadata,
            "charging_power_kw": payload["charging_power_kw"],
            "charging_policy": payload["charging_policy"],
            "running_time_shortest_matrix_s": payload["running_time_shortest_matrix_s"],
            "running_time_path_energy_kwh": payload["running_time_path_energy_kwh"],
            "running_time_path_distance_km": payload["running_time_path_distance_km"],
            "distance_path_travel_time_s": payload["distance_path_travel_time_s"],
            "full_cs_to_depot_time_s": payload["full_cs_to_depot_time_s"],
            "terminal_parent_indices": payload["terminal_parent_indices"],
        }
    )


def build_input_tasks(
    dataset_path: str | Path,
    *,
    family_root: str | Path | None = None,
    scales: set[str] | None = None,
    max_instances: int | None = None,
) -> list[dict[str, Any]]:
    scale_filter = scales or set()
    stage2 = read_stage2_tasks(dataset_path, family_root=family_root)
    if stage2:
        selected = [task for task in stage2 if not scale_filter or task.scale_label in scale_filter]
        if max_instances is not None:
            selected = selected[: int(max_instances)]
        missing = missing_family_directories(selected)
        if missing:
            preview = ", ".join(str(path) for path in missing[:5])
            raise FileNotFoundError(f"missing {len(missing)} Stage-2 family directories: {preview}")
        return [{"input_kind": "stage2", "stage2_task": task.to_dict()} for task in selected]

    raise ValueError(
        "No Stage-2 view_index.parquet was found. Metaheuristic runners only "
        "accept the current Stage-2 view-index/materialized-family layout."
    )


def load_input_task(task: dict[str, Any]) -> tuple[EVRPTWInstance, dict[str, str]]:
    if task["input_kind"] == "stage2":
        ref = Stage2ViewTask.from_dict(task["stage2_task"])
        instance = load_stage2_instance(ref)
        return instance, {
            "file": ref.index_path,
            "family_id": ref.family_id,
            "city_slug": ref.city_slug,
            "split_id": ref.split_id,
            "track_id": ref.track_id,
            "scale_id": ref.scale_label,
        }
    raise ValueError(f"unsupported input task kind: {task.get('input_kind')}")


def running_time_matrix_s(instance: EVRPTWInstance) -> np.ndarray:
    matrix = instance.raw.get("running_time_shortest_matrix_s")
    if matrix is None:
        matrix = instance.raw_travel_time_matrix_s
    if matrix is None:
        raise ValueError(
            "Stage-2 instance is missing running_time_shortest_matrix_s"
        )
    out = np.asarray(matrix, dtype=np.float64)
    if out.shape != (instance.num_terminals, instance.num_terminals):
        raise ValueError(f"travel-time matrix has invalid shape {out.shape}")
    return out


def running_time_energy_matrix_kwh(instance: EVRPTWInstance) -> np.ndarray:
    matrix = instance.raw.get("running_time_path_energy_kwh")
    if matrix is None:
        matrix = instance.energy_matrix_kwh
    if matrix is None:
        raise ValueError(
            "Stage-2 instance is missing running_time_path_energy_kwh"
        )
    out = np.asarray(matrix, dtype=np.float64)
    if out.shape != (instance.num_terminals, instance.num_terminals):
        raise ValueError(f"energy matrix has invalid shape {out.shape}")
    return out


def charging_profile(instance: EVRPTWInstance) -> tuple[np.ndarray, float, str]:
    count = instance.num_charging_stations
    raw = instance.raw.get("charging_power_kw")
    source = "charging_power_kw"
    if raw is None:
        raw = instance.cs_activation.get("charging_power_kw")
        source = "cs_activation.charging_power_kw"
    policy = instance.raw.get("charging_policy", {})
    efficiency = float(
        policy.get("charging_efficiency", instance.vehicle.get("charging_efficiency", 1.0))
    )
    if not 0.0 < efficiency <= 1.0:
        raise ValueError(f"charging_efficiency must be in (0, 1], got {efficiency}")
    if raw is None and count:
        raise ValueError("Stage-2 charging stations require per-station charging_power_kw")
    power = np.asarray([] if raw is None else raw, dtype=np.float64)
    if power.shape != (count,) or np.any(~np.isfinite(power)) or np.any(power <= 0.0):
        raise ValueError(f"charging_power_kw must contain {count} finite positive values")
    return power, efficiency, source


def validate_routes(
    instance: EVRPTWInstance,
    routes: list[list[int]],
    *,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Replay a solution with the same full-charge contract as Stage 2."""

    n = instance.num_customers
    first_station = n + 1
    power, efficiency, power_source = charging_profile(instance)
    distance = np.asarray(instance.distance_matrix_km, dtype=float)
    travel = running_time_matrix_s(instance)
    energy = running_time_energy_matrix_kwh(instance)
    battery_capacity = float(instance.vehicle["battery_capacity_kwh"])
    cargo_capacity = float(instance.vehicle["cargo_capacity_cm3"])
    customer_visits = np.zeros(n, dtype=np.int32)
    violations: list[str] = []
    total_distance = 0.0
    total_charging_time = 0.0
    charging_visits = 0

    for route_index, raw_route in enumerate(routes):
        route = [int(node) for node in raw_route]
        if len(route) < 3 or route[0] != 0 or route[-1] != 0:
            violations.append(f"route {route_index} must start and end at depot 0")
            continue
        if any(node < 0 or node >= instance.num_terminals for node in route):
            violations.append(f"route {route_index} contains an invalid terminal")
            continue
        current_time = float(instance.working_start_s)
        battery = battery_capacity
        load = 0.0
        for origin, destination in zip(route, route[1:]):
            metrics = (
                float(distance[origin, destination]),
                float(travel[origin, destination]),
                float(energy[origin, destination]),
            )
            if not all(math.isfinite(value) for value in metrics):
                violations.append(f"route {route_index} contains a non-finite arc")
                break
            total_distance += metrics[0]
            current_time += metrics[1]
            battery -= metrics[2]
            if battery < -tolerance:
                violations.append(f"route {route_index} arc {origin}->{destination} exceeds battery")
            if 1 <= destination <= n:
                customer = destination - 1
                customer_visits[customer] += 1
                current_time = max(current_time, float(instance.tw_s[customer, 0]))
                if current_time > float(instance.tw_s[customer, 1]) + tolerance:
                    violations.append(f"route {route_index} misses customer {destination} time window")
                current_time += float(instance.service_time_s[customer])
                load += float(instance.demands_cm3[customer])
                if load > cargo_capacity + tolerance:
                    violations.append(f"route {route_index} exceeds cargo capacity")
            elif destination >= first_station:
                station = destination - first_station
                charge_s = max(0.0, battery_capacity - battery) / (efficiency * power[station]) * 3600.0
                current_time += charge_s
                total_charging_time += charge_s
                charging_visits += 1
                battery = battery_capacity
        if current_time > float(instance.working_end_s) + tolerance:
            violations.append(f"route {route_index} returns after operating horizon")

    missing = (np.flatnonzero(customer_visits == 0) + 1).tolist()
    duplicate = (np.flatnonzero(customer_visits > 1) + 1).tolist()
    if missing:
        violations.append(f"customers not served: {missing}")
    if duplicate:
        violations.append(f"customers served more than once: {duplicate}")
    return {
        "passed": not violations,
        "violations": violations,
        "objective_distance_km": total_distance,
        "charging_visit_count": charging_visits,
        "total_charging_time_s": total_charging_time,
        "charging_power_source": power_source,
        "charging_efficiency": efficiency,
    }


class IncumbentEventRecorder:
    """Record improvements and build strict at-or-before checkpoint snapshots.

    An event is timestamped when the solver reports the incumbent. A checkpoint
    only sees events whose timestamp is <= that checkpoint. This deliberately
    prevents a late solution from being backfilled into an earlier checkpoint.
    Final-incumbent forward filling is allowed only for checkpoints after a
    solver that naturally terminated early.
    """

    def __init__(self, checkpoints_s: Iterable[float], time_limit_s: float) -> None:
        self.checkpoints_s = tuple(sorted({float(value) for value in checkpoints_s}))
        self.time_limit_s = float(time_limit_s)
        self._best_event: dict[str, Any] | None = None
        self._first_feasible_time_s: float | None = None
        self._sealed_events: dict[float, dict[str, Any] | None] = {}

    @staticmethod
    def _copy_event(event: dict[str, Any] | None) -> dict[str, Any] | None:
        if event is None:
            return None
        return {
            **event,
            "routes": [list(route) for route in event["routes"]],
            "route_sequence": list(event["route_sequence"]),
        }

    def observe(self, elapsed_s: float, objective: float, routes: list[list[int]]) -> None:
        elapsed = float(elapsed_s)
        value = float(objective)
        clean_routes = [list(map(int, route)) for route in routes]
        if elapsed < 0.0 or elapsed > self.time_limit_s:
            return
        if not math.isfinite(value) or not clean_routes:
            return
        # Seal checkpoints strictly before this event using the previous
        # incumbent. Keeping only five sealed snapshots plus the current best
        # avoids retaining every large Cus1000/Cus2000 improvement route.
        for checkpoint in self.checkpoints_s:
            if checkpoint < elapsed and checkpoint not in self._sealed_events:
                self._sealed_events[checkpoint] = self._copy_event(self._best_event)
        if (
            self._best_event is not None
            and value >= float(self._best_event["objective_distance_km"]) - 1e-9
        ):
            return
        self._best_event = {
            "event_time_s": elapsed,
            "objective_distance_km": value,
            "routes": clean_routes,
            "route_sequence": merge_route_sequences(clean_routes),
            "vehicle_count": len(clean_routes),
        }
        if self._first_feasible_time_s is None:
            self._first_feasible_time_s = elapsed

    @property
    def first_feasible_time_s(self) -> float | None:
        return self._first_feasible_time_s

    @property
    def best_event(self) -> dict[str, Any] | None:
        return self._copy_event(self._best_event)

    def snapshots(
        self,
        *,
        runtime_s: float,
        natural_completion: bool,
        final_status: str,
    ) -> list[dict[str, Any]]:
        runtime = min(float(runtime_s), self.time_limit_s)
        snapshots: list[dict[str, Any]] = []
        for checkpoint in self.checkpoints_s:
            reached = runtime >= checkpoint
            if reached:
                if checkpoint in self._sealed_events:
                    event = self._copy_event(self._sealed_events[checkpoint])
                elif (
                    self._best_event is not None
                    and self._best_event["event_time_s"] <= checkpoint
                ):
                    event = self._copy_event(self._best_event)
                else:
                    event = None
                source = "checkpoint_incumbent" if event else "checkpoint_no_incumbent"
                elapsed = checkpoint
            elif natural_completion:
                event = (
                    self._copy_event(self._best_event)
                    if self._best_event is not None
                    and self._best_event["event_time_s"] <= runtime
                    else None
                )
                source = "final_after_early_stop" if event else "final_no_incumbent"
                elapsed = runtime
            else:
                event = None
                source = "checkpoint_not_reached"
                elapsed = runtime
            has_incumbent = event is not None
            if has_incumbent:
                benchmark_status = "INCUMBENT_AVAILABLE"
            elif not reached:
                benchmark_status = final_status
            elif checkpoint >= self.time_limit_s and self._best_event is None:
                benchmark_status = final_status
            else:
                benchmark_status = "NO_INCUMBENT_YET"
            snapshots.append(
                {
                    "checkpoint_s": checkpoint,
                    "elapsed_s": elapsed,
                    "reached_checkpoint": reached,
                    "status": (
                        final_status
                        if (not reached or checkpoint >= self.time_limit_s)
                        else "RUNNING"
                    ),
                    "benchmark_status": benchmark_status,
                    "has_incumbent": has_incumbent,
                    "objective_distance_km": None if event is None else event["objective_distance_km"],
                    "vehicle_count": None if event is None else event["vehicle_count"],
                    "routes": [] if event is None else [list(route) for route in event["routes"]],
                    "route_sequence": [] if event is None else list(event["route_sequence"]),
                    "incumbent_event_time_s": None if event is None else event["event_time_s"],
                    "source": source,
                }
            )
        return snapshots
