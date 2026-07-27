from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_ROOT = REPO_ROOT / "EVRPTW_Dataset" / "Geo_AC_v1" / "routing_problem_dataset_work"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "EVRPTW_Dataset" / "Geo_AC_v1" / "release" / "routing_dataset_v2"
PROBLEMS = ("evrptw", "vrptw", "cvrp")
SPLITS = ("train", "val")
SCALES = (15, 50, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build independent EVRPTW/VRPTW/CVRP train-val bundles from per-problem per-territory source bundles."
    )
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--evrptw-source-root", type=Path, default=None)
    parser.add_argument("--vrptw-source-root", type=Path, default=None)
    parser.add_argument("--cvrp-source-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--problems", default=",".join(PROBLEMS))
    parser.add_argument("--scales", default=",".join(str(x) for x in SCALES))
    parser.add_argument("--train-source-split", default="train")
    parser.add_argument("--val-source-split", default="val")
    parser.add_argument("--train-per-territory", type=int, default=150)
    parser.add_argument("--val-per-territory", type=int, default=50)
    parser.add_argument(
        "--derive-raw-time-if-missing",
        action="store_true",
        help="Backfill raw_travel_time_matrix_s from distance/effective speed when source bundles do not store it.",
    )
    return parser.parse_args()


def ensure_empty_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite to rebuild it.")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(token.strip()) for token in raw.split(",") if token.strip())


def parse_csv_tokens(raw: str) -> tuple[str, ...]:
    out = tuple(token.strip().lower() for token in raw.split(",") if token.strip())
    unknown = sorted(set(out) - set(PROBLEMS))
    if unknown:
        raise ValueError(f"Unsupported problems: {unknown}; supported={PROBLEMS}")
    return out


def read_bundle(path: Path, limit: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("rb") as f:
        header = pickle.load(f)
        available = int(header["num_instances"])
        count = available if limit is None else min(int(limit), available)
        instances = [pickle.load(f) for _ in range(count)]
    if limit is not None and len(instances) < int(limit):
        raise ValueError(f"{path} has {len(instances)} instances, expected at least {limit}.")
    return header, instances


def write_bundle(path: Path, header: dict[str, Any], instances: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(header, f, protocol=pickle.HIGHEST_PROTOCOL)
        for instance in instances:
            pickle.dump(instance, f, protocol=pickle.HIGHEST_PROTOCOL)


def source_paths(source_root: Path, source_split: str, scale: int) -> list[Path]:
    paths = sorted((source_root / source_split).glob(f"*/Cus_{scale}/instances.pkl"))
    if not paths:
        raise FileNotFoundError(f"No source bundles found for {source_split}/Cus_{scale} under {source_root}")
    return paths


def _copy_array(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    return deepcopy(value)


def _copy_instance(source: dict[str, Any]) -> dict[str, Any]:
    return {key: _copy_array(value) for key, value in source.items()}


def _speed_for_time_derivation(source: dict[str, Any]) -> float:
    speed_profile = dict(source.get("speed_profile", {}) or {})
    vehicle = dict(source.get("vehicle", {}) or {})
    speed = (
        speed_profile.get("effective_speed_kmh")
        or speed_profile.get("base_effective_speed_kmh")
        or vehicle.get("design_speed_kmh")
        or 40.0
    )
    return float(speed)


def ensure_raw_time_matrix(source: dict[str, Any], derive_if_missing: bool) -> dict[str, Any]:
    out = _copy_instance(source)
    if out.get("raw_travel_time_matrix_s") is None and derive_if_missing:
        distance = np.asarray(out["distance_matrix_km"], dtype=np.float32)
        speed = _speed_for_time_derivation(out)
        out["raw_travel_time_matrix_s"] = (distance / max(speed, 1e-9) * 3600.0).astype(np.float32)
        metadata = dict(out.get("metadata", {}) or {})
        metadata["raw_travel_time_backfill"] = {
            "source": "distance_matrix_km / speed_profile.effective_speed_kmh",
            "speed_kmh": speed,
        }
        storage = dict(metadata.get("time_matrix_storage", {}) or {})
        storage["raw_travel_time_matrix_s"] = True
        metadata["time_matrix_storage"] = storage
        out["metadata"] = metadata
    return out


def normalize_metadata(
    instance: dict[str, Any],
    problem: str,
    split: str,
    scale: int,
    global_index: int,
    source_bundle: Path,
    source_root: Path,
    source_instance_id: str,
) -> dict[str, Any]:
    problem_name = problem.upper()
    instance_id = f"{split}_Cus{scale}_{global_index:06d}"
    key = f"{problem}/{split}/Cus{scale}/{instance_id}"
    metadata = dict(instance.get("metadata", {}) or {})
    metadata.update(
        {
            "problem_class": problem_name,
            "routing_problem": problem,
            "dataset_family": f"Geo-{problem_name}",
            "dataset_version": f"Geo-{problem_name}-v2",
            "release_profile": "NA-US-20",
            "release_split": split,
            "release_scale": f"Cus{scale}",
            "release_global_index": int(global_index),
            "instance_key": key,
            "reference_solution_key": key if split == "val" else None,
            "source_problem_instance_id": source_instance_id,
            "source_problem_region_id": str(instance.get("region_id", "")),
            "source_problem_bundle": str(source_bundle.relative_to(source_root)),
            "terminal_order": "0=depot, 1..N=customers, EVRPTW only: N+1..N+M=charging stations",
        }
    )
    return metadata


def derive_evrptw(
    source: dict[str, Any],
    split: str,
    scale: int,
    global_index: int,
    source_bundle: Path,
    source_root: Path,
    derive_raw_time: bool,
) -> dict[str, Any]:
    out = ensure_raw_time_matrix(source, derive_raw_time)
    source_id = str(out.get("instance_id", f"source_{global_index:06d}"))
    out["problem_class"] = "EVRPTW"
    out["instance_id"] = f"{split}_Cus{scale}_{global_index:06d}"
    out["metadata"] = normalize_metadata(out, "evrptw", split, scale, global_index, source_bundle, source_root, source_id)
    return out


def _vehicle_for_classical(source_vehicle: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(source_vehicle.get("name", "delivery_vehicle")),
        "cargo_capacity_cm3": float(source_vehicle.get("cargo_capacity_cm3", np.inf)),
    }


def derive_classical(
    source: dict[str, Any],
    problem: str,
    split: str,
    scale: int,
    global_index: int,
    source_bundle: Path,
    source_root: Path,
    derive_raw_time: bool,
) -> dict[str, Any]:
    source = ensure_raw_time_matrix(source, derive_raw_time)
    source_id = str(source.get("instance_id", f"source_{global_index:06d}"))
    stop = int(scale) + 1
    metadata = normalize_metadata(source, problem, split, scale, global_index, source_bundle, source_root, source_id)
    if problem == "cvrp":
        metadata["removed_constraints"] = ["time_windows", "service_times", "battery", "charging_stations"]
    elif problem == "vrptw":
        metadata["removed_constraints"] = ["battery", "charging_stations"]
    else:
        raise ValueError(f"Unsupported classical problem: {problem}")
    metadata["source_problem_instance_key"] = source.get("metadata", {}).get("instance_key")

    out = {
        "problem_class": problem.upper(),
        "instance_id": f"{split}_Cus{scale}_{global_index:06d}",
        "region_id": str(source["region_id"]),
        "mother_board_id": source.get("mother_board_id"),
        "operating_day_id": source.get("operating_day_id"),
        "active_customer_ids": np.asarray(source["active_customer_ids"], dtype=np.int32)[:scale].copy(),
        "depot": np.asarray(source["depot"], dtype=np.float32).copy(),
        "customers": np.asarray(source["customers"], dtype=np.float32)[:scale].copy(),
        "distance_matrix_km": np.asarray(source["distance_matrix_km"], dtype=np.float32)[:stop, :stop].copy(),
        "demands_cm3": np.asarray(source["demands_cm3"], dtype=np.float32)[:scale].copy(),
        "package_counts": np.asarray(source["package_counts"], dtype=np.int32)[:scale].copy(),
        "vehicle": _vehicle_for_classical(dict(source.get("vehicle", {}) or {})),
        "metadata": metadata,
    }
    if problem == "vrptw":
        raw_time = source.get("raw_travel_time_matrix_s")
        if raw_time is None:
            raise ValueError("VRPTW derivation requires raw_travel_time_matrix_s or --derive-raw-time-if-missing.")
        out.update(
            {
                "day_type": source.get("day_type"),
                "working_start_s": int(source["working_start_s"]),
                "working_end_s": int(source["working_end_s"]),
                "service_time_s": np.asarray(source["service_time_s"], dtype=np.float32)[:scale].copy(),
                "tw_s": np.asarray(source["tw_s"], dtype=np.float32)[:scale].copy(),
                "speed_profile": dict(source.get("speed_profile", {}) or {}),
                "travel_time_matrix_s": np.asarray(raw_time, dtype=np.float32)[:stop, :stop].copy(),
            }
        )
    return out


def build_header(
    problem: str,
    split: str,
    scale: int,
    num_instances: int,
    num_cs: int | None,
    territories: list[str],
    source_bundles: list[str],
) -> dict[str, Any]:
    problem_name = problem.upper()
    metadata = {
        "dataset_family": f"Geo-{problem_name}",
        "dataset_version": f"Geo-{problem_name}-v2",
        "source_dataset_version": "Geo-AC-v1",
        "release_profile": "NA-US-20",
        "problem_class": problem_name,
        "split": split,
        "scale_name": f"Cus{scale}",
        "num_customers": int(scale),
        "territory_count": len(territories),
        "territories": territories,
        "source_bundles": source_bundles,
        "terminal_order": "0=depot, 1..N=customers, EVRPTW only: N+1..N+M=charging stations",
        "reference_solution_key_policy": "Validation instances use a non-null key; train instances use null.",
        "sampling_policy": "Problem variants share service-territory mother boards but are independently sampled active days.",
    }
    if num_cs is not None:
        metadata["num_charging_stations"] = int(num_cs)
    if problem == "evrptw":
        metadata["time_matrix"] = "raw_travel_time_matrix_s when present; distance_matrix_km is objective basis"
    elif problem == "vrptw":
        metadata["time_matrix"] = "travel_time_matrix_s inherited from the independently sampled VRPTW source raw_travel_time_matrix_s"
    return {
        "format": "routing_problem_instance_bundle_v2",
        "dataset_metadata": metadata,
        "num_instances": int(num_instances),
        "num_customers": int(scale),
        **({"num_charging_stations": int(num_cs)} if num_cs is not None else {}),
    }


def build_split_scale(
    source_root: Path,
    output_root: Path,
    problem: str,
    split: str,
    source_split: str,
    scale: int,
    per_territory: int,
    derive_raw_time: bool,
) -> dict[str, Any]:
    instances: list[dict[str, Any]] = []
    territories: list[str] = []
    source_bundle_rel: list[str] = []
    num_cs_values: set[int] = set()
    for path in source_paths(source_root, source_split, scale):
        header, source_instances = read_bundle(path, limit=per_territory)
        territory_id = path.parts[-3]
        territories.append(territory_id)
        source_bundle_rel.append(str(path.relative_to(source_root)))
        if "num_charging_stations" in header:
            num_cs_values.add(int(header["num_charging_stations"]))
        for source in source_instances:
            global_index = len(instances)
            if problem == "evrptw":
                item = derive_evrptw(source, split, scale, global_index, path, source_root, derive_raw_time)
            else:
                item = derive_classical(source, problem, split, scale, global_index, path, source_root, derive_raw_time)
            instances.append(item)

    if problem == "evrptw":
        if len(num_cs_values) != 1:
            raise ValueError(f"{split}/Cus{scale} has inconsistent EVRPTW active CS counts: {sorted(num_cs_values)}")
        num_cs = int(next(iter(num_cs_values)))
    else:
        num_cs = None
    out_dir = output_root / problem / "dataset" / split / f"Cus{scale}"
    out_path = out_dir / "instances.pkl"
    header = build_header(problem, split, scale, len(instances), num_cs, territories, source_bundle_rel)
    write_bundle(out_path, header, instances)
    digest = sha256_file(out_path)
    metadata = {
        **header["dataset_metadata"],
        "num_instances": int(len(instances)),
        "instances_per_territory": int(per_territory),
        "bundle_file": "instances.pkl",
        "bundle_sha256": digest,
        "instance_key_field": "metadata.instance_key",
        "reference_solution_key_field": "metadata.reference_solution_key",
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return {
        "problem": problem,
        "split": split,
        "scale": f"Cus{scale}",
        "num_instances": len(instances),
        "instances_per_territory": int(per_territory),
        "territory_count": len(territories),
        "bundle": str(out_path.relative_to(output_root)),
        "sha256": digest,
    }


def write_solution_templates(problem_root: Path, problem: str) -> None:
    solution_dir = problem_root / "reference_solutions" / "val"
    solution_dir.mkdir(parents=True, exist_ok=True)
    (solution_dir / "solutions_template.csv").write_text(
        "instance_key,split,scale,instance_id,region_id,status,objective,is_certified_optimal,"
        "lower_bound,optimality_gap,runtime_sec,solver_name,solver_version,time_limit_sec,"
        "vehicle_count,solution_path,notes\n",
        encoding="utf-8",
    )
    route = {
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
        "runtime_sec": 30.0,
        "solver_name": "solver_name",
        "solver_version": "solver_version",
        "time_limit_sec": 900.0,
        "vehicle_count": 2,
        "terminal_index_convention": {
            "0": "depot",
            "1_to_N": "active customers",
            "N_plus_1_to_N_plus_M": "charging stations for EVRPTW only",
        },
        "routes": [{"vehicle_id": 0, "terminal_sequence": [0, 1, 5, 3, 0]}],
    }
    (solution_dir / "route_template.json").write_text(json.dumps(route, indent=2) + "\n", encoding="utf-8")
    readme = f"""# Validation Reference Solutions

This directory is reserved for Geo-{problem.upper()}-v2 validation reference solutions.

Join solutions to instances by `instance_key`. Training bundles intentionally use
`metadata.reference_solution_key = None`; validation bundles use the canonical
validation key.
"""
    (solution_dir / "README.md").write_text(readme, encoding="utf-8")


def write_readmes(output_root: Path, rows: list[dict[str, Any]]) -> None:
    totals: dict[str, dict[str, int]] = {}
    for row in rows:
        totals.setdefault(row["problem"], {"train": 0, "val": 0})
        totals[row["problem"]][row["split"]] += int(row["num_instances"])
    root_readme = f"""# Routing Problem Dataset v2

This release contains real-geography semi-synthetic routing datasets:

- `evrptw/`: electric VRP with time windows and charging stations
- `vrptw/`: classical VRP with time windows, derived from the same EVRPTW active days
- `cvrp/`: capacitated VRP, derived from the same EVRPTW active days

Each problem has `train` and `val` splits for Cus15, Cus50, and Cus100.
The three problem variants share the same NA-US-20 service-territory mother
boards, but their active-day instances are independently sampled with different
random seeds. CVRP/VRPTW are not created by deleting fields from the EVRPTW
instances.

| problem | train | val |
|---|---:|---:|
| EVRPTW | {totals.get("evrptw", {}).get("train", 0):,} | {totals.get("evrptw", {}).get("val", 0):,} |
| VRPTW | {totals.get("vrptw", {}).get("train", 0):,} | {totals.get("vrptw", {}).get("val", 0):,} |
| CVRP | {totals.get("cvrp", {}).get("train", 0):,} | {totals.get("cvrp", {}).get("val", 0):,} |

Directory layout:

```text
{{problem}}/dataset/{{split}}/Cus15/instances.pkl
{{problem}}/dataset/{{split}}/Cus50/instances.pkl
{{problem}}/dataset/{{split}}/Cus100/instances.pkl
{{problem}}/reference_solutions/val/
```

Use these metadata fields for auditing source provenance:

- `metadata.instance_key`
- `metadata.source_problem_instance_id`
- `region_id`
- `active_customer_ids`

EVRPTW instances store `distance_matrix_km` and, for this release, persist
`raw_travel_time_matrix_s`. VRPTW instances store the customer/depot submatrix
as `travel_time_matrix_s`. CVRP instances retain only distance, demand, and
cargo-capacity semantics.
"""
    (output_root / "README.md").write_text(root_readme, encoding="utf-8")
    for problem in PROBLEMS:
        problem_root = output_root / problem
        if not problem_root.exists():
            continue
        (problem_root / "README.md").write_text(
            f"# Geo-{problem.upper()}-v2\n\n"
            "Instances are stored in `dataset/train` and `dataset/val` by scale.\n"
            "Validation solution templates are under `reference_solutions/val`.\n",
            encoding="utf-8",
        )


def write_manifest(output_root: Path, rows: list[dict[str, Any]], source_roots: dict[str, str]) -> None:
    with (output_root / "bundle_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "problem",
                "split",
                "scale",
                "num_instances",
                "instances_per_territory",
                "territory_count",
                "bundle",
                "sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "release_name": "routing_dataset_v2",
        "release_profile": "NA-US-20",
        "source_roots": source_roots,
        "problems": list(sorted({row["problem"] for row in rows})),
        "splits": ["train", "val"],
        "scales": ["Cus15", "Cus50", "Cus100"],
        "bundles": rows,
    }
    with (output_root / "release_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    ensure_empty_output(output_root, bool(args.overwrite))
    problems = parse_csv_tokens(args.problems)
    scales = parse_csv_ints(args.scales)
    work_root = args.work_root.resolve()
    source_roots = {
        "evrptw": (args.evrptw_source_root or (work_root / "evrptw_by_territory")).resolve(),
        "vrptw": (args.vrptw_source_root or (work_root / "vrptw_by_territory")).resolve(),
        "cvrp": (args.cvrp_source_root or (work_root / "cvrp_by_territory")).resolve(),
    }
    split_specs = {
        "train": (args.train_source_split, int(args.train_per_territory)),
        "val": (args.val_source_split, int(args.val_per_territory)),
    }
    rows: list[dict[str, Any]] = []
    for problem in problems:
        source_root = source_roots[problem]
        for split, (source_split, per_territory) in split_specs.items():
            for scale in scales:
                rows.append(
                    build_split_scale(
                        source_root=source_root,
                        output_root=output_root,
                        problem=problem,
                        split=split,
                        source_split=source_split,
                        scale=int(scale),
                        per_territory=per_territory,
                        derive_raw_time=bool(args.derive_raw_time_if_missing),
                    )
                )
        write_solution_templates(output_root / problem, problem)
    write_readmes(output_root, rows)
    write_manifest(output_root, rows, {key: str(value) for key, value in source_roots.items() if key in problems})
    print(json.dumps({"output_root": str(output_root), "bundles": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
