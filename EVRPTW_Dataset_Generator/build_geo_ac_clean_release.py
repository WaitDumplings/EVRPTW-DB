from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import shutil
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "EVRPTW_Dataset" / "Geo_AC_v1" / "benchmark_splits_v1"
DEFAULT_RAW_ROOT = REPO_ROOT / "EVRPTW_Dataset" / "Geo_AC_v1" / "source_data_na_us20"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "EVRPTW_Dataset"
    / "Geo_AC_v1"
    / "release"
    / "EVRP_TW_D_Geo_AC_v1_NA_US20_clean"
)
DEFAULT_CITY_CONFIG = REPO_ROOT / "EVRPTW_Dataset_Generator" / "configs" / "geo_ac_v1_na_us20.with_sources.yaml"


SPLIT_MAP = {
    "offline_train_standard": "train",
    "validation_reference": "val",
    "eval_standard": "eval",
}
SPLIT_INSTANCE_COUNTS = {
    "train": 250,
    "val": 50,
    "eval": 20,
}
SCALES = (5, 15, 50, 100)
RAW_CSVS = (
    "road_nodes.csv",
    "road_edges.csv",
    "customer_seed.csv",
    "latent_customer.csv",
    "charging_station.csv",
    "depot_candidate.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean Geo-AC-v1 release layout.")
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--city-config", type=Path, default=DEFAULT_CITY_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
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


def bundle_paths_for_scale(benchmark_root: Path, source_split: str, scale: int) -> list[Path]:
    pattern = benchmark_root / source_split
    paths = sorted(pattern.glob(f"*/Cus_{scale}/instances.pkl"))
    if not paths:
        raise FileNotFoundError(f"No bundles found for {source_split}/Cus_{scale} under {benchmark_root}")
    return paths


def normalize_instance_id(split: str, scale: int, global_index: int) -> str:
    return f"{split}_Cus{scale}_{global_index:06d}"


def consolidated_header(
    split: str,
    source_split: str,
    scale: int,
    num_cs: int,
    total: int,
    territories: list[str],
    source_bundles: list[str],
) -> dict[str, Any]:
    metadata = {
        "dataset_family": "EVRP-TW-D",
        "dataset_version": "Geo-AC-v1",
        "release_profile": "NA-US-20",
        "split": split,
        "source_split": source_split,
        "scale_name": f"Cus{scale}",
        "num_customers": int(scale),
        "num_charging_stations": int(num_cs),
        "territory_count": len(territories),
        "territories": territories,
        "source_bundles": source_bundles,
        "framing": "real-geography semi-synthetic EVRPTW benchmark",
        "instance_territory_field": "region_id",
    }
    return {
        "format": "evrptw_instance_bundle_v1",
        "dataset_metadata": metadata,
        "num_instances": int(total),
        "num_customers": int(scale),
        "num_charging_stations": int(num_cs),
    }


def consolidate_dataset(benchmark_root: Path, output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_split, split in SPLIT_MAP.items():
        for scale in SCALES:
            paths = bundle_paths_for_scale(benchmark_root, source_split, scale)
            instances: list[dict[str, Any]] = []
            territories: list[str] = []
            source_bundles: list[str] = []
            num_cs_values = set()
            expected_per_territory = SPLIT_INSTANCE_COUNTS[split]
            for path in paths:
                header, source_instances = read_bundle(path)
                territory_id = path.parts[-3]
                territories.append(territory_id)
                source_bundles.append(str(path.relative_to(benchmark_root)))
                num_cs_values.add(int(header["num_charging_stations"]))
                if int(header["num_instances"]) != expected_per_territory:
                    raise ValueError(f"{path} has {header['num_instances']} instances, expected {expected_per_territory}.")
                for item in source_instances:
                    global_index = len(instances)
                    item = dict(item)
                    original_id = str(item.get("instance_id", f"instance_{global_index:06d}"))
                    item["instance_id"] = normalize_instance_id(split, scale, global_index)
                    metadata = dict(item.get("metadata", {}) or {})
                    metadata.update(
                        {
                            "release_split": split,
                            "release_scale": f"Cus{scale}",
                            "release_global_index": int(global_index),
                            "instance_key": f"{split}/Cus{scale}/{normalize_instance_id(split, scale, global_index)}",
                            "reference_solution_key": (
                                f"{split}/Cus{scale}/{normalize_instance_id(split, scale, global_index)}"
                                if split == "val"
                                else None
                            ),
                            "source_split": source_split,
                            "source_territory_id": territory_id,
                            "source_instance_id": original_id,
                            "source_bundle": str(path.relative_to(benchmark_root)),
                        }
                    )
                    item["metadata"] = metadata
                    instances.append(item)
            if len(num_cs_values) != 1:
                raise ValueError(f"{source_split}/Cus_{scale} has inconsistent active CS counts: {sorted(num_cs_values)}")
            num_cs = int(next(iter(num_cs_values)))
            out_dir = output_root / "dataset" / split / f"Cus{scale}"
            out_path = out_dir / "instances.pkl"
            header = consolidated_header(
                split=split,
                source_split=source_split,
                scale=scale,
                num_cs=num_cs,
                total=len(instances),
                territories=territories,
                source_bundles=source_bundles,
            )
            write_bundle(out_path, header, instances)
            metadata = {
                "dataset_family": "EVRP-TW-D",
                "dataset_version": "Geo-AC-v1",
                "release_profile": "NA-US-20",
                "split": split,
                "source_split": source_split,
                "scale": f"Cus{scale}",
                "num_customers": int(scale),
                "num_charging_stations": num_cs,
                "territory_count": len(territories),
                "territories": territories,
                "instances_per_territory": expected_per_territory,
                "num_instances": len(instances),
                "bundle_file": "instances.pkl",
                "bundle_sha256": sha256_file(out_path),
                "instance_key_field": "metadata.instance_key",
                "reference_solution_key_field": "metadata.reference_solution_key",
                "reference_solution_key_policy": "Only validation instances use a non-null key; train/eval are null.",
                "source_bundles": source_bundles,
            }
            with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            rows.append(
                {
                    "split": split,
                    "scale": f"Cus{scale}",
                    "num_instances": len(instances),
                    "num_customers": int(scale),
                    "num_charging_stations": num_cs,
                    "bundle": str(out_path.relative_to(output_root)),
                    "sha256": metadata["bundle_sha256"],
                }
            )
    return rows


def copy_raw_data(raw_root: Path, output_root: Path) -> list[dict[str, Any]]:
    raw_out = output_root / "raw_data"
    raw_out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for territory_dir in sorted(path for path in raw_root.iterdir() if path.is_dir() and (path / "normalized").exists()):
        territory_id = territory_dir.name
        out_dir = raw_out / territory_id
        out_dir.mkdir(parents=True, exist_ok=True)
        row = {"territory_id": territory_id}
        for filename in RAW_CSVS:
            src = territory_dir / "normalized" / filename
            if not src.exists():
                raise FileNotFoundError(src)
            dst = out_dir / filename
            shutil.copy2(src, dst)
            with dst.open("r", encoding="utf-8", newline="") as f:
                row[filename.replace(".csv", "_rows")] = max(sum(1 for _ in f) - 1, 0)
        qa = territory_dir / "qa" / "qa_summary.json"
        if qa.exists():
            shutil.copy2(qa, out_dir / "qa_summary.json")
        rows.append(row)
    with (raw_out / "territories.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to rewrite release configs.")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to rewrite release configs.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def write_industrial_dataset(benchmark_root: Path, output_root: Path, city_config: Path) -> None:
    industrial_root = output_root / "industrial_dataset"
    config_dir = industrial_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    source_profiles = benchmark_root / "generator_profiles"
    shutil.copy2(source_profiles / "industrial_train_profile_template.yaml", config_dir / "industrial_train_profile_template.yaml")
    shutil.copy2(source_profiles / "cus1000_train_profile.yaml", config_dir / "cus1000_example.yaml")

    cfg = load_yaml(city_config)
    for territory in cfg.get("territories", []):
        territory_id = territory["territory_id"]
        territory["data_root"] = f"../../raw_data/{territory_id}"
        territory["source_files"] = {
            "road_nodes_csv": "road_nodes.csv",
            "road_edges_csv": "road_edges.csv",
            "customer_seed_csv": "customer_seed.csv",
            "charging_station_csv": "charging_station.csv",
            "depot_candidate_csv": "depot_candidate.csv",
            "latent_customer_csv": "latent_customer.csv",
        }
    write_yaml(config_dir / "geo_ac_v1_na_us20.release_sources.yaml", cfg)


def write_text_files(output_root: Path, split_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]]) -> None:
    total_standard = sum(int(row["num_instances"]) for row in split_rows)
    readme = f"""# EVRP-TW-D / Geo-AC-v1 / NA-US-20

This is the clean release layout for the Geo-AC-v1 real-geography
semi-synthetic EVRP-TW-D dataset.

## Layout

```text
dataset/
  train/
    Cus5/instances.pkl
    Cus5/metadata.json
    Cus15/
    Cus50/
    Cus100/
  val/
  eval/
raw_data/
  {{territory_id}}/
    road_nodes.csv
    road_edges.csv
    customer_seed.csv
    latent_customer.csv
    charging_station.csv
    depot_candidate.csv
industrial_dataset/
  configs/
    industrial_train_profile_template.yaml
    cus1000_example.yaml
    geo_ac_v1_na_us20.release_sources.yaml
reference_solutions/
  val/
    README.md
    solutions_template.csv
    solutions.csv          # to be added after solving
```

The `dataset/` directory is intended for direct loading. Each split/scale has
one consolidated `instances.pkl` bundle and one `metadata.json` file. Instances
retain their source service territory in the `region_id` field and in
`metadata.source_territory_id`.

Each instance has stable identity fields:

- `instance_id`: globally unique within this release, for example
  `val_Cus50_000123`.
- `metadata.instance_key`: canonical join key, formatted as
  `split/CusN/instance_id`, for example `val/Cus50/val_Cus50_000123`.
- `metadata.reference_solution_key`: non-null only for validation instances;
  train and eval instances use `None`.

Reference solutions should be stored as sidecar files under
`reference_solutions/val/` and joined by `instance_key`. The problem bundles
should remain immutable after publication.

## Standard Splits

| split | scales | instances | solutions |
|---|---|---:|---|
| `train` | Cus5, Cus15, Cus50, Cus100 | 20,000 | not included |
| `val` | Cus5, Cus15, Cus50, Cus100 | 4,000 | validation references may be added later |
| `eval` | Cus5, Cus15, Cus50, Cus100 | 1,600 | intentionally withheld |

Total standard instances: {total_standard:,}

## Instance Node Convention

Each instance stores one active operating day. The terminal order used by
`distance_matrix_km` is:

```text
0                         depot
1 ... N                   active customers
N+1 ... N+M               active charging stations
```

For example, a Cus5 instance with 3 active charging stations has a `9 x 9`
matrix. `distance_matrix_km[i, j]` is the road-network distance between
terminal `i` and terminal `j` in kilometers. Customer connector distances from
the sampled customer position to its snapped road node are already included.

The active IDs map back to the raw territory CSVs:

- `active_customer_ids[k]` is the row index of customer `k` in
  `raw_data/{{territory_id}}/latent_customer.csv`.
- `active_cs_ids[k]` is the row index of charging station `k` in
  `raw_data/{{territory_id}}/charging_station.csv`.
- Depot provenance is stored in `metadata.depot_catchment`, including the
  selected depot road node and candidate metadata.
- `metadata.terminal_node_ids` stores the snapped road-node IDs in the same
  terminal order as the matrix.

The release stores `distance_matrix_km` by default. Travel time can be derived
from `speed_profile.effective_speed_kmh`; demand, service time, time windows,
vehicle capacity, battery capacity, energy consumption, and charging power are
stored directly in each instance.

## Raw Data

`raw_data/` contains the normalized public-data CSV inputs for {len(raw_rows)}
service territories. These are the source tables used to build service
territories and operating-day instances.

## Industrial Dataset

`industrial_dataset/` contains generator configuration templates for train-only
large-scale profiles. Industrial instances are not materialized by default.
Use `--scales N:M`, where `N` is active customers and `M` is active charging
stations.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")

    description = """# Dataset Description

EVRP-TW-D / Geo-AC-v1 / NA-US-20 is a real-geography semi-synthetic benchmark
for the Electric Vehicle Routing Problem with Time Windows. The release contains
fixed train, validation, and held-out evaluation operating-day instances for
20 North American service territories.

The raw geospatial layer is derived from public sources: Census/TIGER
geographies, ACS occupied housing-unit counts, OSM/OSMnx and TIGER/Overture
road data, NREL/AFDC public charging station data, and open depot/industrial
candidate sources. Demand, service time, and time-window attributes are
generated by the EVRP-TW-D generator using calibrated operational distributions.

Validation reference solutions can be added under the validation solution
directory in future releases. Held-out evaluation solutions are intentionally
not included.
"""
    (output_root / "DATASET_DESCRIPTION.md").write_text(description, encoding="utf-8")

    raw_readme = """# Raw Data

This directory contains normalized public-data CSV files by service territory.
Each territory directory contains:

- `road_nodes.csv`
- `road_edges.csv`
- `customer_seed.csv`
- `latent_customer.csv`
- `charging_station.csv`
- `depot_candidate.csv`

`territories.csv` summarizes row counts for the territory-level CSVs.
"""
    (output_root / "raw_data" / "README.md").write_text(raw_readme, encoding="utf-8")

    industrial_readme = """# Industrial Dataset Profiles

Industrial-scale train-only instances are generator-supported rather than
materialized in the default release.

Use `configs/industrial_train_profile_template.yaml` for general settings and
`configs/cus1000_example.yaml` as a concrete example. The release source config
`configs/geo_ac_v1_na_us20.release_sources.yaml` points to this package's
`raw_data/` directory.

Example:

```bash
conda run -n maojie python -m EVRPTW_Dataset_Generator.prepare_geospatial_benchmark_suite \\
  --city-config industrial_dataset/configs/geo_ac_v1_na_us20.release_sources.yaml \\
  --output-root industrial_dataset/generated \\
  --seed 20260901 \\
  --split-name offline_industrial \\
  --scales 1000:100 \\
  --instances-per-scale 50 \\
  --require-real-sources \\
  --skip-existing
```
"""
    (output_root / "industrial_dataset" / "README.md").write_text(industrial_readme, encoding="utf-8")

    solution_dir = output_root / "reference_solutions" / "val"
    solution_dir.mkdir(parents=True, exist_ok=True)
    solution_readme = """# Validation Reference Solutions

This directory is reserved for global/reference solutions for the validation
split only.

Recommended file:

```text
solutions.csv
```

Use `solutions_template.csv` as the schema template. Recommended columns:

```text
instance_key,split,scale,instance_id,region_id,status,objective,is_certified_optimal,
lower_bound,optimality_gap,runtime_sec,solver_name,solver_version,
time_limit_sec,solution_path,notes
```

For validation instances, `instance_key` should match
`metadata.reference_solution_key` in the corresponding `instances.pkl` record.
Train and eval instances intentionally have `metadata.reference_solution_key =
None`.

Keep problem instances immutable. Add or update validation solutions here as
sidecar files instead of writing solver results into `dataset/val/*/instances.pkl`.

Recommended route-detail files can be placed under:

```text
routes/{scale}/{instance_id}.json
```

Use `route_template.json` as the route-detail schema template.

Use local terminal indices in route sequences:

```text
0                         depot
1 ... N                   active customers
N+1 ... N+M               active charging stations
```

Example route-detail shape:

```json
{
  "instance_key": "val/Cus5/val_Cus5_000000",
  "objective": 123.45,
  "status": "optimal",
  "routes": [
    {
      "vehicle_id": 0,
      "terminal_sequence": [0, 1, 7, 3, 0],
      "arrival_time_s": [30840, 32210, 35100, 37420, 40200],
      "departure_time_s": [30840, 32320, 36900, 37510, 40200],
      "battery_arrival_kwh": [126.0, 121.2, 108.4, 126.0, 119.1],
      "battery_departure_kwh": [126.0, 121.2, 126.0, 126.0, 119.1],
      "load_cm3": [0, 6730.2, 6730.2, 8760.6, 8760.6]
    }
  ]
}
```

The exact objective definition should match the benchmark solver/evaluator.
Do not store full road polylines in the reference solution unless a later
visualization release needs them; the instance-level terminal distance matrix is
the benchmark graph abstraction.

The held-out `dataset/eval/` split should not include solution records in the
public dataset release.
"""
    (solution_dir / "README.md").write_text(solution_readme, encoding="utf-8")
    (solution_dir / "solutions_template.csv").write_text(
        "instance_key,split,scale,instance_id,region_id,status,objective,is_certified_optimal,"
        "lower_bound,optimality_gap,runtime_sec,solver_name,solver_version,time_limit_sec,"
        "solution_path,notes\n",
        encoding="utf-8",
    )
    route_template = {
        "instance_key": "val/Cus5/val_Cus5_000000",
        "split": "val",
        "scale": "Cus5",
        "instance_id": "val_Cus5_000000",
        "region_id": "atlanta_ga_fulton_county",
        "status": "optimal",
        "objective": 123.45,
        "objective_unit": "km",
        "is_certified_optimal": True,
        "lower_bound": 123.45,
        "optimality_gap": 0.0,
        "runtime_sec": 32.4,
        "solver_name": "gurobi",
        "solver_version": "11.0.0",
        "time_limit_sec": 900,
        "terminal_index_convention": {
            "0": "depot",
            "1_to_N": "active customers",
            "N_plus_1_to_N_plus_M": "active charging stations",
        },
        "routes": [
            {
                "vehicle_id": 0,
                "terminal_sequence": [0, 1, 7, 3, 0],
                "arrival_time_s": [30840, 32210, 35100, 37420, 40200],
                "departure_time_s": [30840, 32320, 36900, 37510, 40200],
                "battery_arrival_kwh": [126.0, 121.2, 108.4, 126.0, 119.1],
                "battery_departure_kwh": [126.0, 121.2, 126.0, 126.0, 119.1],
                "load_cm3": [0.0, 6730.2, 6730.2, 8760.6, 8760.6],
            }
        ],
        "notes": "Template values are illustrative; replace with solver output.",
    }
    (solution_dir / "route_template.json").write_text(json.dumps(route_template, indent=2) + "\n", encoding="utf-8")


def write_release_manifest(output_root: Path, split_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]]) -> None:
    manifest = {
        "dataset_family": "EVRP-TW-D",
        "dataset_version": "Geo-AC-v1",
        "profile_name": "NA-US-20",
        "layout": "clean_consolidated_release_v1",
        "standard_instance_count": sum(int(row["num_instances"]) for row in split_rows),
        "territory_count": len(raw_rows),
        "splits": split_rows,
        "raw_data_territories": raw_rows,
        "industrial_dataset": {
            "materialized_by_default": False,
            "config_template": "industrial_dataset/configs/industrial_train_profile_template.yaml",
            "example_profile": "industrial_dataset/configs/cus1000_example.yaml",
            "release_city_config": "industrial_dataset/configs/geo_ac_v1_na_us20.release_sources.yaml",
        },
        "reference_solutions": {
            "validation_only": "reference_solutions/val",
            "schema_template": "reference_solutions/val/solutions_template.csv",
            "eval_solutions_included": False,
        },
        "instance_identity": {
            "instance_id": "Globally unique within this release, e.g. val_Cus50_000123.",
            "instance_key": "Canonical join key stored at metadata.instance_key, formatted as {split}/Cus{N}/{instance_id}.",
            "reference_solution_key": "Stored at metadata.reference_solution_key. Non-null only for validation instances; train/eval use null.",
            "solution_join_policy": "Reference solutions should be stored as sidecar files under reference_solutions/val and joined by instance_key/reference_solution_key; problem bundles should remain immutable.",
        },
    }
    with (output_root / "release_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    ensure_empty_output(output_root, bool(args.overwrite))
    split_rows = consolidate_dataset(args.benchmark_root.resolve(), output_root)
    raw_rows = copy_raw_data(args.raw_root.resolve(), output_root)
    write_industrial_dataset(args.benchmark_root.resolve(), output_root, args.city_config.resolve())
    write_text_files(output_root, split_rows, raw_rows)
    write_release_manifest(output_root, split_rows, raw_rows)
    print(json.dumps({"output_root": str(output_root), "standard_instances": sum(row["num_instances"] for row in split_rows)}, indent=2))


if __name__ == "__main__":
    main()
