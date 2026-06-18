from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "EVRPTW_Dataset" / "Geo_AC_v1" / "release" / "dataset_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "EVRPTW_Dataset" / "Geo_AC_v1" / "release" / "classical_dataset_v1"

PROBLEMS = ("cvrp", "vrptw")
SPLITS = ("train", "val", "eval")
SCALES = (15, 50, 100)
SPLIT_COUNTS = {
    "train": 5000,
    "val": 1000,
    "eval": 400,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive paired Geo-CVRP-v1 and Geo-VRPTW-v1 datasets from Geo-AC-v1 EVRP instances."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite to rebuild it.")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def read_bundle(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("rb") as f:
        header = pickle.load(f)
        instances = [pickle.load(f) for _ in range(int(header["num_instances"]))]
    return header, instances


def write_bundle(path: Path, header: dict[str, Any], instances: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(header, f, protocol=pickle.HIGHEST_PROTOCOL)
        for instance in instances:
            pickle.dump(instance, f, protocol=pickle.HIGHEST_PROTOCOL)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _slice_terminal_array(value: Any, stop: int) -> Any:
    if value is None:
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return value
    if arr.ndim == 0 or arr.shape[0] < stop:
        return value
    return arr[:stop].copy()


def _travel_time_matrix_s(distance_matrix_km: np.ndarray, speed_profile: dict[str, Any], vehicle: dict[str, Any]) -> np.ndarray:
    speed = float(speed_profile.get("effective_speed_kmh") or vehicle.get("design_speed_kmh") or 40.0)
    return (np.asarray(distance_matrix_km, dtype=np.float32) / max(speed, 1e-9) * 3600.0).astype(np.float32)


def _vehicle_for_classical(source_vehicle: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(source_vehicle.get("name", "delivery_vehicle")),
        "cargo_capacity_cm3": float(source_vehicle.get("cargo_capacity_cm3", np.inf)),
    }


def _base_metadata(
    source: dict[str, Any],
    problem: str,
    split: str,
    scale: int,
) -> dict[str, Any]:
    source_meta = dict(source.get("metadata", {}) or {})
    source_key = source_meta.get("instance_key")
    instance_id = str(source["instance_id"])
    problem_name = problem.upper()
    key = f"{problem}/{split}/Cus{scale}/{instance_id}"
    metadata = dict(source_meta)
    metadata.update(
        {
            "problem_class": problem_name,
            "dataset_family": f"Geo-{problem_name}",
            "dataset_version": f"Geo-{problem_name}-v1",
            "release_profile": "NA-US-20",
            "instance_key": key,
            "reference_solution_key": key if split == "val" else None,
            "source_evrptw_instance_key": source_key,
            "source_evrptw_instance_id": instance_id,
            "source_evrptw_problem_class": "EVRPTW-D",
            "source_evrptw_dataset_version": "Geo-AC-v1",
            "derivation_policy": (
                "Paired derivation from Geo-AC-v1 EVRP instances using the same depot, active customers, "
                "demands, and customer distance matrix; EV charging/battery terminals and constraints are removed."
            ),
            "release_split": split,
            "release_scale": f"Cus{scale}",
        }
    )
    terminal_node_ids = source_meta.get("terminal_node_ids")
    sliced_terminal_node_ids = _slice_terminal_array(terminal_node_ids, scale + 1)
    if sliced_terminal_node_ids is not None:
        metadata["terminal_node_ids"] = sliced_terminal_node_ids
    if "time_matrix_storage" in metadata:
        metadata["time_matrix_storage"] = {
            "distance_matrix_km": True,
            "travel_time_matrix_s": problem == "vrptw",
            "raw_travel_time_matrix_s": False,
            "ev_transition_time_matrix_s": False,
            "shortest_time_matrix_s": False,
        }
    return metadata


def derive_instance(source: dict[str, Any], problem: str, split: str, scale: int) -> dict[str, Any]:
    num_customers = int(scale)
    stop = num_customers + 1
    distance = np.asarray(source["distance_matrix_km"], dtype=np.float32)[:stop, :stop].copy()
    source_vehicle = dict(source.get("vehicle", {}) or {})
    metadata = _base_metadata(source, problem, split, scale)
    if problem == "cvrp":
        metadata["removed_constraints"] = ["time_windows", "service_times", "battery", "charging_stations"]
        metadata["objective_unit"] = "km"
    elif problem == "vrptw":
        metadata["removed_constraints"] = ["battery", "charging_stations"]
        metadata["objective_unit"] = "km"
    else:
        raise ValueError(f"Unsupported problem: {problem}")

    out = {
        "problem_class": problem.upper(),
        "instance_id": str(source["instance_id"]),
        "region_id": str(source["region_id"]),
        "mother_board_id": source.get("mother_board_id"),
        "operating_day_id": source.get("operating_day_id"),
        "active_customer_ids": np.asarray(source["active_customer_ids"], dtype=np.int32).copy(),
        "depot": np.asarray(source["depot"], dtype=np.float32).copy(),
        "customers": np.asarray(source["customers"], dtype=np.float32).copy(),
        "distance_matrix_km": distance,
        "demands_cm3": np.asarray(source["demands_cm3"], dtype=np.float32).copy(),
        "package_counts": np.asarray(source["package_counts"], dtype=np.int32).copy(),
        "vehicle": _vehicle_for_classical(source_vehicle),
        "metadata": metadata,
    }
    if problem == "vrptw":
        speed_profile = dict(source.get("speed_profile", {}) or {})
        out.update(
            {
                "day_type": source.get("day_type"),
                "working_start_s": int(source["working_start_s"]),
                "working_end_s": int(source["working_end_s"]),
                "service_time_s": np.asarray(source["service_time_s"], dtype=np.float32).copy(),
                "tw_s": np.asarray(source["tw_s"], dtype=np.float32).copy(),
                "speed_profile": speed_profile,
                "travel_time_matrix_s": _travel_time_matrix_s(distance, speed_profile, source_vehicle),
            }
        )
    return out


def derived_header(
    source_header: dict[str, Any],
    problem: str,
    split: str,
    scale: int,
    num_instances: int,
) -> dict[str, Any]:
    source_meta = dict(source_header.get("dataset_metadata", {}) or {})
    territories = list(source_meta.get("territories", []))
    problem_name = problem.upper()
    metadata = {
        "dataset_family": f"Geo-{problem_name}",
        "dataset_version": f"Geo-{problem_name}-v1",
        "source_dataset_family": "EVRP-TW-D",
        "source_dataset_version": "Geo-AC-v1",
        "release_profile": "NA-US-20",
        "problem_class": problem_name,
        "split": split,
        "scale_name": f"Cus{scale}",
        "num_customers": int(scale),
        "territory_count": len(territories),
        "territories": territories,
        "paired_with_source_evrptw": True,
        "framing": "real-geography semi-synthetic classical VRP benchmark derived from Geo-AC-v1",
        "terminal_order": "0=depot, 1..N=customers",
    }
    if problem == "vrptw":
        metadata["time_model"] = "travel_time_matrix_s plus service_time_s and tw_s"
    return {
        "format": "classical_vrp_instance_bundle_v1",
        "dataset_metadata": metadata,
        "num_instances": int(num_instances),
        "num_customers": int(scale),
    }


def build_problem(source_root: Path, output_root: Path, problem: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for scale in SCALES:
            source_path = source_root / "dataset" / split / f"Cus{scale}" / "instances.pkl"
            source_header, source_instances = read_bundle(source_path)
            if len(source_instances) != SPLIT_COUNTS[split]:
                raise ValueError(f"{source_path} has {len(source_instances)} instances; expected {SPLIT_COUNTS[split]}.")
            derived = [derive_instance(item, problem, split, scale) for item in source_instances]
            out_dir = output_root / problem / "dataset" / split / f"Cus{scale}"
            out_path = out_dir / "instances.pkl"
            header = derived_header(source_header, problem, split, scale, len(derived))
            write_bundle(out_path, header, derived)
            digest = sha256_file(out_path)
            metadata = {
                **header["dataset_metadata"],
                "num_instances": len(derived),
                "instances_per_territory": SPLIT_COUNTS[split] // 20,
                "bundle_file": "instances.pkl",
                "bundle_sha256": digest,
                "source_bundle": str(source_path.relative_to(source_root)),
                "instance_key_field": "metadata.instance_key",
                "reference_solution_key_field": "metadata.reference_solution_key",
                "reference_solution_key_policy": "Only validation instances use a non-null key; train/eval are null.",
            }
            with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            rows.append(
                {
                    "problem": problem,
                    "split": split,
                    "scale": f"Cus{scale}",
                    "num_instances": len(derived),
                    "bundle": str(out_path.relative_to(output_root)),
                    "sha256": digest,
                }
            )
    return rows


def write_solution_templates(problem_root: Path, problem: str) -> None:
    val_dir = problem_root / "reference_solutions" / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    readme = f"""# Validation Reference Solutions

This directory is reserved for validation reference solutions for Geo-{problem.upper()}-v1.

Recommended summary file:

```text
solutions.csv
```

Use `solutions_template.csv` as the schema template. Route-detail files can be
placed under:

```text
routes/{{scale}}/{{instance_id}}.json
```

Route sequences use local terminal indices:

```text
0                         depot
1 ... N                   active customers
```

Problem instances should remain immutable. Store solver outputs as sidecar
files in this directory and join by `instance_key`.
"""
    (val_dir / "README.md").write_text(readme, encoding="utf-8")
    (val_dir / "solutions_template.csv").write_text(
        "instance_key,split,scale,instance_id,region_id,status,objective,is_certified_optimal,"
        "lower_bound,optimality_gap,runtime_sec,solver_name,solver_version,time_limit_sec,"
        "vehicle_count,solution_path,notes\n",
        encoding="utf-8",
    )
    route_template: dict[str, Any] = {
        "instance_key": f"{problem}/val/Cus15/val_Cus15_000000",
        "split": "val",
        "scale": "Cus15",
        "instance_id": "val_Cus15_000000",
        "region_id": "atlanta_ga_fulton_county",
        "problem_class": problem.upper(),
        "status": "optimal",
        "objective": 123.45,
        "objective_unit": "km",
        "is_certified_optimal": True,
        "lower_bound": 123.45,
        "optimality_gap": 0.0,
        "runtime_sec": 32.4,
        "solver_name": "solver_name",
        "solver_version": "solver_version",
        "time_limit_sec": 900,
        "vehicle_count": 2,
        "terminal_index_convention": {
            "0": "depot",
            "1_to_N": "active customers",
        },
        "routes": [
            {
                "vehicle_id": 0,
                "terminal_sequence": [0, 1, 5, 3, 0],
            }
        ],
        "notes": "Template values are illustrative; replace with solver output.",
    }
    if problem == "vrptw":
        route_template["routes"][0].update(
            {
                "arrival_time_s": [30840, 32210, 35100, 37420, 40200],
                "departure_time_s": [30840, 32320, 35200, 37510, 40200],
                "load_cm3": [0.0, 6730.2, 8760.6, 10200.0, 10200.0],
            }
        )
    (val_dir / "route_template.json").write_text(json.dumps(route_template, indent=2) + "\n", encoding="utf-8")


def write_problem_readme(problem_root: Path, problem: str) -> None:
    problem_name = problem.upper()
    if problem == "cvrp":
        retained = "depot, customers, road-distance matrix, demand, package counts, and vehicle cargo capacity"
        removed = "time windows, service times, battery constraints, and charging stations"
    else:
        retained = (
            "depot, customers, road-distance matrix, travel-time matrix, demand, package counts, "
            "service times, time windows, working horizon, and vehicle cargo capacity"
        )
        removed = "battery constraints and charging stations"
    readme = f"""# Geo-{problem_name}-v1 / NA-US-20

This is a paired classical VRP dataset derived from `../dataset_v1`, the
Geo-AC-v1 EVRP-TW-D release.

Each instance reuses the same service territory, depot, active customers, demand
attributes, and customer road-distance submatrix as the source EVRP instance.
The derived Geo-{problem_name}-v1 instance retains {retained}. It removes
{removed}.

## Layout

```text
dataset/
  train/Cus15|Cus50|Cus100/instances.pkl
  val/Cus15|Cus50|Cus100/instances.pkl
  eval/Cus15|Cus50|Cus100/instances.pkl
reference_solutions/
  val/
    README.md
    solutions_template.csv
    route_template.json
```

## Splits

| split | Cus15 | Cus50 | Cus100 | total |
|---|---:|---:|---:|---:|
| train | 5,000 | 5,000 | 5,000 | 15,000 |
| val | 1,000 | 1,000 | 1,000 | 3,000 |
| eval | 400 | 400 | 400 | 1,200 |

Total instances: 19,200

## Pairing

Every instance has:

- `metadata.instance_key`: canonical key for this derived dataset.
- `metadata.source_evrptw_instance_key`: the paired source Geo-AC-v1 EVRP key.

Terminal order:

```text
0                         depot
1 ... N                   active customers
```

`distance_matrix_km` is the customer/depot submatrix from the source EVRP
instance. Charging-station terminals are not present in this derived dataset.
"""
    (problem_root / "README.md").write_text(readme, encoding="utf-8")


def write_root_files(output_root: Path, rows: list[dict[str, Any]], source_root: Path) -> None:
    total_by_problem = {}
    for problem in PROBLEMS:
        total_by_problem[problem] = sum(int(row["num_instances"]) for row in rows if row["problem"] == problem)
    readme = f"""# Classical Geo-VRP Dataset v1

This release contains two paired classical VRP datasets derived from the
Geo-AC-v1 EVRP-TW-D `dataset_v1` release:

- `cvrp/`: Geo-CVRP-v1
- `vrptw/`: Geo-VRPTW-v1

Both variants use the same real-geography service territories, depots, active
customers, demands, and customer road-distance matrices as the source EVRP
instances. EV-specific charging-station and battery constraints are removed.

## Instance Counts

| problem | train | val | eval | total |
|---|---:|---:|---:|---:|
| CVRP | 15,000 | 3,000 | 1,200 | {total_by_problem["cvrp"]:,} |
| VRPTW | 15,000 | 3,000 | 1,200 | {total_by_problem["vrptw"]:,} |

Scales are Cus15, Cus50, and Cus100.

## Source

Source EVRP release:

```text
{source_root}
```

Instances are self-contained and include coordinates plus terminal distance
matrices. Use `metadata.source_evrptw_instance_key` to join any classical
instance back to its paired EVRP source instance.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "release_name": "classical_dataset_v1",
        "profile_name": "NA-US-20",
        "source_dataset": str(source_root),
        "problems": {
            "cvrp": {
                "dataset_family": "Geo-CVRP",
                "dataset_version": "Geo-CVRP-v1",
                "total_instances": total_by_problem["cvrp"],
            },
            "vrptw": {
                "dataset_family": "Geo-VRPTW",
                "dataset_version": "Geo-VRPTW-v1",
                "total_instances": total_by_problem["vrptw"],
            },
        },
        "splits": rows,
    }
    with (output_root / "release_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with (output_root / "bundle_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["problem", "split", "scale", "num_instances", "bundle", "sha256"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    ensure_output(output_root, bool(args.overwrite))
    all_rows: list[dict[str, Any]] = []
    for problem in PROBLEMS:
        rows = build_problem(source_root, output_root, problem)
        all_rows.extend(rows)
        problem_root = output_root / problem
        write_solution_templates(problem_root, problem)
        write_problem_readme(problem_root, problem)
    write_root_files(output_root, all_rows, source_root)
    print(json.dumps({"output_root": str(output_root), "bundles": len(all_rows)}, indent=2))


if __name__ == "__main__":
    main()
