from __future__ import annotations

import hashlib
import json
import math
import signal
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import pandas as pd

from evrptw_core.schema import EVRPTWInstance, merge_route_sequences
from evrptw_stage2.artifacts import load_materialized_view
from evrptw_stage2.contracts import STAGE2_GENERATION_CONTRACT


DEFAULT_CHECKPOINTS_S = (60.0, 300.0, 900.0, 3600.0, 7200.0)
DEFAULT_TIME_LIMIT_S = 7200.0
SEED_SCHEME = "blake2b_view_id_v1"
TIME_BUDGET_ITERATION_CEILING = 2_147_483_647
ALGORITHM_TIMING_SCOPE = "adapter_solver_constructor_and_solve"
RUN_CONTRACT_SCHEMA = "evrptw_meta_run_contract_v3"
CANONICAL_REPLAY_PROFILE_ID = "full_charge_derated_strict_route_v3"
FAMILY_SCHEMA = "cle_evrptw_materialized_matrix_family_v3"
VIEW_SCHEMA = "cle_evrptw_materialized_view_v4"
VIEW_MATRIX_STORAGE = "parent_index_view"
DATA_IDENTITY_MODE = "deterministic_stage2_ids_no_content_hash_v1"

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
    family_cohort_id: str = ""
    terminal_count: int = 0
    view_seed: int = 0

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


def stable_view_seed(base_seed: int, view_id: str) -> int:
    """Return a shard/order-independent NumPy-compatible seed for a view.

    Python's built-in hash is intentionally randomized between processes, so a
    cryptographic digest is used as the experiment contract.  The domain tag
    permits a future seed scheme without silently changing existing runs.
    """

    payload = f"{SEED_SCHEME}\0{int(base_seed)}\0{str(view_id)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32)


def stable_view_shard(view_id: str, shard_count: int) -> int:
    """Assign a view to a stable shard independent of index ordering."""

    count = int(shard_count)
    if count <= 0:
        raise ValueError("shard_count must be positive")
    payload = f"evrptw-meta-shard-v1\0{str(view_id)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % count


def resolve_optional_iteration_budget(
    requested: int | None,
) -> tuple[int, str]:
    """Resolve a smoke-test iteration cap or a wall-clock-driven ceiling."""

    if requested is None:
        return TIME_BUDGET_ITERATION_CEILING, "wall_clock"
    value = int(requested)
    if value < 0:
        raise ValueError("iteration budget must be non-negative")
    return value, "iteration_limited"


def _portable_dataset_path_identity(raw_path: str, anchor: str) -> str:
    """Drop host-specific prefixes while retaining the dataset-relative path."""

    parts = Path(str(raw_path)).parts
    try:
        position = parts.index(anchor)
    except ValueError:
        return Path(str(raw_path)).name
    return Path(*parts[position:]).as_posix()


def build_run_contract(
    task: dict[str, Any],
    *,
    algorithm_name: str,
    algorithm_profile_id: str,
    base_seed: int,
    solver_parameters: dict[str, Any],
) -> tuple[str, str]:
    """Build the portable semantic identity of one benchmark invocation.

    Execution layout (worker count, queue depth, slice/shard selection and save
    directory) is intentionally absent.  Consequently the same view can be
    resumed on another server, while any budget, seed, profile, solver setting
    or dataset-row identity change requires a fresh solve.
    """

    reference = dict(task.get("stage2_task", {}))
    contract = {
        "schema": RUN_CONTRACT_SCHEMA,
        "algorithm": {
            "name": str(algorithm_name),
            "profile_id": str(algorithm_profile_id),
        },
        "budget": {
            "time_limit_s": float(task["time_limit_s"]),
            "checkpoints_s": [
                float(value) for value in task.get("checkpoints_s", ())
            ],
        },
        "randomness": {
            "base_seed": int(base_seed),
            "seed_scheme": str(task["seed_scheme"]),
            "instance_seed": int(task["seed"]),
        },
        "timing_scope": ALGORITHM_TIMING_SCOPE,
        "canonical_replay_profile_id": CANONICAL_REPLAY_PROFILE_ID,
        "data_identity": {
            "input_kind": str(task.get("input_kind", "")),
            "view_id": str(reference.get("view_id", "")),
            "family_id": str(reference.get("family_id", "")),
            "family_cohort_id": str(reference.get("family_cohort_id", "")),
            "consumer_cohort_id": str(reference.get("consumer_cohort_id", "")),
            "split_id": str(reference.get("split_id", "")),
            "track_id": str(reference.get("track_id", "")),
            "city_slug": str(reference.get("city_slug", "")),
            "scale_id": str(reference.get("scale_id", "")),
            "customer_count": int(reference.get("customer_count", 0)),
            "charging_station_count": int(
                reference.get("charging_station_count", 0)
            ),
            "terminal_count": int(reference.get("terminal_count", 0)),
            "view_seed": int(reference.get("view_seed", 0)),
            "identity_mode": DATA_IDENTITY_MODE,
            "expected_family_schema": FAMILY_SCHEMA,
            "expected_view_schema": VIEW_SCHEMA,
            "expected_generation_contract": STAGE2_GENERATION_CONTRACT,
            "expected_matrix_storage": VIEW_MATRIX_STORAGE,
            "view_index_identity": _portable_dataset_path_identity(
                str(reference.get("index_path", "")), "generation_plan"
            ),
            "family_directory_identity": _portable_dataset_path_identity(
                str(reference.get("family_dir", "")), "materialized"
            ),
        },
        "solver_parameters": dict(solver_parameters),
    }
    contract_json = json.dumps(
        contract,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(contract_json.encode("utf-8")).hexdigest()
    return fingerprint, contract_json


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
    family_path = family_dir / "family_manifest.json"
    view_path = family_dir / "views" / task.view_id / "view_manifest.json"
    family_manifest = json.loads(family_path.read_text(encoding="utf-8"))
    view_manifest = json.loads(view_path.read_text(encoding="utf-8"))
    if family_manifest.get("schema") != FAMILY_SCHEMA:
        raise ValueError(
            f"unsupported Stage-2 family schema: {family_manifest.get('schema')!r}"
        )
    if view_manifest.get("schema") != VIEW_SCHEMA:
        raise ValueError(
            f"unsupported Stage-2 view schema: {view_manifest.get('schema')!r}"
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
        raise ValueError("view manifest ID does not match view index")
    if str(view_manifest.get("family_id")) != task.family_id:
        raise ValueError("view manifest family ID does not match view index")

    payload = load_materialized_view(task.family_dir, task.view_id)
    if str(payload["instance_id"]) != task.view_id:
        raise ValueError("loaded Stage-2 view_id does not match view index")
    if str(payload["family_id"]) != task.family_id:
        raise ValueError("loaded Stage-2 family_id does not match view index")
    if len(payload["customers"]) != task.customer_count:
        raise ValueError("loaded customer count does not match view index")
    if len(payload["charging_stations"]) != task.charging_station_count:
        raise ValueError("loaded charging-station count does not match view index")
    if task.terminal_count != 1 + task.customer_count + task.charging_station_count:
        raise ValueError("view-index terminal count is inconsistent")

    vehicle = dict(payload["vehicle"])
    vehicle["charging_power_derating_factor"] = float(
        payload["charging_policy"]["charging_power_derating_factor"]
    )
    metadata = {
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
            # Preserve the generator's constructive feasibility witness for
            # heuristic warm starts and independent diagnostics.  It remains
            # raw metadata; benchmark route replay is still authoritative.
            "feasibility_certificate": payload.get("feasibility_certificate"),
        }
    )


def build_input_tasks(
    dataset_path: str | Path,
    *,
    family_root: str | Path | None = None,
    scales: set[str] | None = None,
    max_instances: int | None = None,
    start_index: int = 0,
    end_index: int | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
) -> list[dict[str, Any]]:
    start = int(start_index)
    end = None if end_index is None else int(end_index)
    count = int(shard_count)
    shard = int(shard_index)
    if start < 0:
        raise ValueError("start_index must be non-negative")
    if end is not None and end < start:
        raise ValueError("end_index must be greater than or equal to start_index")
    if count <= 0:
        raise ValueError("shard_count must be positive")
    if shard < 0 or shard >= count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    if max_instances is not None and int(max_instances) < 0:
        raise ValueError("max_instances must be non-negative")

    scale_filter = scales or set()
    stage2 = read_stage2_tasks(dataset_path, family_root=family_root)
    if stage2:
        selected = [task for task in stage2 if not scale_filter or task.scale_label in scale_filter]
        selected = selected[start:end]
        if count > 1:
            selected = [
                task
                for task in selected
                if stable_view_shard(task.view_id, count) == shard
            ]
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


def bounded_process_map(
    solve_function: Callable[[dict[str, Any]], dict[str, Any]],
    tasks: Iterable[dict[str, Any]],
    *,
    workers: int,
    max_in_flight: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Map tasks without placing the complete experiment in the process queue."""

    worker_count = max(1, int(workers))
    if worker_count == 1:
        for task in tasks:
            yield solve_function(task)
        return

    limit = worker_count * 2 if max_in_flight is None else int(max_in_flight)
    if limit <= 0:
        raise ValueError("max_in_flight must be positive")
    task_iterator = iter(tasks)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        pending = set()
        for _ in range(limit):
            try:
                task = next(task_iterator)
            except StopIteration:
                break
            pending.add(executor.submit(solve_function, task))

        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                yield future.result()
                try:
                    task = next(task_iterator)
                except StopIteration:
                    continue
                pending.add(executor.submit(solve_function, task))


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
    # Stage-2 matrices are float32 and can be hundreds of MiB.  Casting to
    # float64 here used to duplicate the complete matrix for every incumbent
    # replay.  Individual arc values are converted to Python float by the
    # replay loop, so retaining the source dtype is numerically identical.
    out = np.asarray(matrix)
    if out.shape != (instance.num_terminals, instance.num_terminals):
        raise ValueError(f"travel-time matrix has invalid shape {out.shape}")
    if not np.issubdtype(out.dtype, np.number):
        raise ValueError(f"travel-time matrix has non-numeric dtype {out.dtype}")
    return out


def running_time_energy_matrix_kwh(instance: EVRPTWInstance) -> np.ndarray:
    matrix = instance.raw.get("running_time_path_energy_kwh")
    if matrix is None:
        matrix = instance.energy_matrix_kwh
    if matrix is None:
        raise ValueError(
            "Stage-2 instance is missing running_time_path_energy_kwh"
        )
    out = np.asarray(matrix)
    if out.shape != (instance.num_terminals, instance.num_terminals):
        raise ValueError(f"energy matrix has invalid shape {out.shape}")
    if not np.issubdtype(out.dtype, np.number):
        raise ValueError(f"energy matrix has non-numeric dtype {out.dtype}")
    return out


def charging_profile(instance: EVRPTWInstance) -> tuple[np.ndarray, float, str]:
    count = instance.num_charging_stations
    raw = instance.raw.get("charging_power_kw")
    source = "charging_power_kw"
    if raw is None:
        raw = instance.cs_activation.get("charging_power_kw")
        source = "cs_activation.charging_power_kw"
    policy = instance.raw.get("charging_policy", {})
    if "charging_power_derating_factor" in policy:
        if "charging_efficiency" in policy:
            raise ValueError("charging policy cannot define both derating and efficiency")
        power_factor = float(policy["charging_power_derating_factor"])
    elif "charging_efficiency" in policy:
        power_factor = float(policy["charging_efficiency"])
    elif "charging_power_derating_factor" in instance.vehicle:
        power_factor = float(instance.vehicle["charging_power_derating_factor"])
    else:
        power_factor = float(instance.vehicle.get("charging_efficiency", 1.0))
    if not 0.0 < power_factor <= 1.0:
        raise ValueError(
            f"charging power factor must be in (0, 1], got {power_factor}"
        )
    if raw is None and count:
        raise ValueError("Stage-2 charging stations require per-station charging_power_kw")
    power = np.asarray([] if raw is None else raw, dtype=np.float64)
    if power.shape != (count,) or np.any(~np.isfinite(power)) or np.any(power <= 0.0):
        raise ValueError(f"charging_power_kw must contain {count} finite positive values")
    return power, power_factor, source


def validate_routes(
    instance: EVRPTWInstance,
    routes: list[list[int]],
    *,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Replay a solution with the same full-charge contract as Stage 2."""

    n = instance.num_customers
    first_station = n + 1
    power, power_factor, power_source = charging_profile(instance)
    distance = np.asarray(instance.distance_matrix_km)
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
        if 0 in route[1:-1]:
            violations.append(f"route {route_index} contains an internal depot visit")
            continue
        if not any(1 <= node <= n for node in route[1:-1]):
            violations.append(f"route {route_index} contains no customer")
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
                charge_s = max(0.0, battery_capacity - battery) / (power_factor * power[station]) * 3600.0
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
        "charging_power_derating_factor": power_factor,
    }


def certificate_singleton_routes(
    instance: EVRPTWInstance,
) -> list[list[int]] | None:
    """Reconstruct and replay the Stage-2 constructive feasibility witness.

    The stored certificate identifies the full-battery infrastructure state
    immediately before each customer and the first charger after it.  The
    intermediate depot/charger hops are deliberately not stored, so they are
    reconstructed from the same directed full-state cache used by Stage 2.
    A route set is returned only after canonical benchmark replay accepts it;
    malformed, stale, or absent certificates are treated as no warm start.
    """

    certificate = instance.raw.get("feasibility_certificate")
    if not isinstance(certificate, dict) or instance.num_customers <= 0:
        return None

    required = (
        "inbound_full_state_terminal_index",
        "first_post_customer_charger_terminal_index",
    )
    if any(name not in certificate for name in required):
        return None
    try:
        inbound = np.asarray(
            certificate["inbound_full_state_terminal_index"], dtype=np.int64
        )
        post_customer = np.asarray(
            certificate["first_post_customer_charger_terminal_index"],
            dtype=np.int64,
        )
        if inbound.shape != (instance.num_customers,) or post_customer.shape != (
            instance.num_customers,
        ):
            return None

        # Keep this internal generator dependency local: normal replay and
        # runners that do not request a certificate avoid importing SciPy.
        from evrptw_stage2.orders import _build_full_state_route_cache

        power_kw, power_factor, _ = charging_profile(instance)
        cache = _build_full_state_route_cache(
            customer_count=instance.num_customers,
            running_time_matrix_s=running_time_matrix_s(instance),
            running_time_energy_matrix_kwh=running_time_energy_matrix_kwh(instance),
            charging_power_kw=power_kw,
            battery_capacity_kwh=float(instance.vehicle["battery_capacity_kwh"]),
            charging_power_derating_factor=power_factor,
        )
        full_nodes = np.asarray(cache.terminal_indices, dtype=np.int64)
        full_position = {int(node): pos for pos, node in enumerate(full_nodes)}

        def depot_to(position: int) -> list[int]:
            cursor = int(position)
            reversed_positions = [cursor]
            for _ in range(len(full_nodes)):
                if cursor == 0:
                    return [
                        int(full_nodes[pos]) for pos in reversed(reversed_positions)
                    ]
                cursor = int(cache.from_depot_predecessor[cursor])
                if cursor < 0 or cursor >= len(full_nodes):
                    break
                reversed_positions.append(cursor)
            raise ValueError("broken depot-to-customer certificate predecessor chain")

        def to_depot(position: int) -> list[int]:
            cursor = int(position)
            positions = [cursor]
            for _ in range(len(full_nodes)):
                if cursor == 0:
                    return [int(full_nodes[pos]) for pos in positions]
                cursor = int(cache.to_depot_reverse_predecessor[cursor])
                if cursor < 0 or cursor >= len(full_nodes):
                    break
                positions.append(cursor)
            raise ValueError("broken customer-to-depot certificate predecessor chain")

        routes: list[list[int]] = []
        first_station = 1 + instance.num_customers
        for customer_offset in range(instance.num_customers):
            customer_node = customer_offset + 1
            inbound_node = int(inbound[customer_offset])
            if inbound_node not in full_position:
                return None
            route = depot_to(full_position[inbound_node])
            route.append(customer_node)
            post_node = int(post_customer[customer_offset])
            if post_node < 0:
                route.append(0)
            else:
                if post_node not in full_position or post_node < first_station:
                    return None
                outbound = to_depot(full_position[post_node])
                route.extend(outbound)
            routes.append(route)

        stored_visits = certificate.get("charging_visit_count")
        if stored_visits is not None:
            expected_visits = np.asarray(stored_visits, dtype=np.int64)
            actual_visits = np.asarray(
                [sum(node >= first_station for node in route) for route in routes],
                dtype=np.int64,
            )
            if expected_visits.shape != actual_visits.shape or not np.array_equal(
                expected_visits, actual_visits
            ):
                return None

        audit = validate_routes(instance, routes)
        return routes if audit["passed"] else None
    except (KeyError, TypeError, ValueError, IndexError, ImportError):
        return None


class IncumbentReplayCache:
    """Reuse replay only when a solver reports the exact same route again.

    VNS-TS reports its current global solution at conservative outer-iteration
    boundaries even when that incumbent has not changed.  Replaying a large
    route at every boundary is unnecessary.  This one-entry cache deliberately
    does *not* trust the solver objective and does not skip any new route
    sequence, so a real improvement can never be lost.
    """

    def __init__(self, instance: EVRPTWInstance) -> None:
        self.instance = instance
        self._route_key: tuple[tuple[int, ...], ...] | None = None
        self._audit: dict[str, Any] | None = None
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _copy_audit(audit: dict[str, Any]) -> dict[str, Any]:
        return {
            **audit,
            "violations": list(audit.get("violations", [])),
        }

    def validate(self, routes: list[list[int]]) -> dict[str, Any]:
        key = tuple(tuple(int(node) for node in route) for route in routes)
        if key == self._route_key and self._audit is not None:
            self.hits += 1
            return self._copy_audit(self._audit)
        clean_routes = [list(route) for route in key]
        audit = validate_routes(self.instance, clean_routes)
        self._route_key = key
        self._audit = self._copy_audit(audit)
        self.misses += 1
        return self._copy_audit(audit)


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
            and value >= float(self._best_event["objective_distance_km"])
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
