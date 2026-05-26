from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_ROOT = REPO_ROOT / "EVRPTW_Dataset_Generator"
sys.path.insert(0, str(GENERATOR_ROOT))

from evrptw_hierarchy.generation.generator import HierarchyDatasetGenerator
from evrptw_hierarchy.generation.territory_pool import generate_service_territory_pool


SUITES: dict[str, tuple[int, int]] = {
    "AC_Tiny_5": (5, 3),
    "AC_Small_15": (15, 3),
    "AC_Medium_50": (50, 10),
    "AC_Large_100": (100, 20),
    "AC_XLarge_1000": (1000, 120),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare EVRPTW-D/AC-v1 service-territory pools and fixed evaluation suites.")
    parser.add_argument("--config-path", type=Path, default=GENERATOR_ROOT / "configs/amazon_hierarchy.yaml")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "EVRPTW_Dataset" / "AC_v1")
    parser.add_argument("--train-territory-pool-path", type=Path, default=None)
    parser.add_argument("--eval-territory-pool-path", type=Path, default=None)
    parser.add_argument("--train-territories", type=int, default=None)
    parser.add_argument("--eval-territories", type=int, default=256)
    parser.add_argument("--territory-pool-path", type=Path, default=None, help="Legacy alias for the training service-territory pool path.")
    parser.add_argument("--num-territories", type=int, default=None, help="Legacy alias for --train-territories.")
    parser.add_argument("--active-territories", type=int, default=None, help="Legacy alias for the number of eval service territories to load.")
    parser.add_argument("--latent-customer-pool-size", type=int, default=5000)
    parser.add_argument("--cs-candidate-pool-size", type=int, default=120)
    parser.add_argument("--num-instances", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--pool-workers", type=int, default=8)
    parser.add_argument("--skip-train-pool", action="store_true")
    parser.add_argument("--skip-eval-pool", action="store_true")
    parser.add_argument("--skip-eval-instances", "--skip-eval", dest="skip_eval_instances", action="store_true")
    parser.add_argument("--skip-pool", action="store_true", help="Legacy alias: skip both train and eval service-territory pool generation.")
    parser.add_argument("--suite", action="append", choices=sorted(SUITES), help="Generate only selected suite(s). Can be repeated.")
    parser.add_argument("--plots", action="store_true", help="Also write preview plots. Disabled by default for large eval generation.")
    return parser.parse_args()


def dataset_metadata(args: argparse.Namespace, suite_name: str, num_customers: int, num_cs: int, pool_path: Path) -> dict[str, object]:
    eval_territories = int(args.active_territories or args.eval_territories)
    return {
        "dataset_family": "EVRPTW-D",
        "dataset_version": "AC-v1",
        "calibration_profile": "Amazon-Calibrated",
        "split": "eval",
        "suite_name": suite_name,
        "num_instances": int(args.num_instances),
        "num_customers": int(num_customers),
        "num_charging_stations": int(num_cs),
        "num_service_territories": eval_territories,
        "latent_customer_pool_size": int(args.latent_customer_pool_size),
        "cs_candidate_pool_size": int(args.cs_candidate_pool_size),
        "service_territory_pool_path": str(pool_path),
        "terminology": {
            "EVRPTW-D": "Dataset family for generated EVRP-TW benchmark instances.",
            "AC-v1": "Amazon-Calibrated first public profile.",
            "service_territory_graph": "Stable region-level delivery-station territory used to sample operating days.",
        },
    }


def file_size_bytes(path: str | Path | None) -> int:
    if path is None:
        return 0
    p = Path(path)
    if not p.exists() or not p.is_file():
        return 0
    return int(p.stat().st_size)


def write_timing_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "split",
        "suite_name",
        "num_service_territories",
        "num_instances",
        "num_customers",
        "num_charging_stations",
        "wall_time_s",
        "output_path",
        "output_size_bytes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    train_territories = int(args.train_territories or args.num_territories or 1024)
    eval_territories = int(args.active_territories or args.eval_territories)
    train_pool_path = Path(args.train_territory_pool_path or args.territory_pool_path or (dataset_root / "train"))
    eval_pool_path = Path(args.eval_territory_pool_path or (dataset_root / "eval"))
    dataset_root.mkdir(parents=True, exist_ok=True)
    timing_rows: list[dict[str, object]] = []

    if not (args.skip_pool or args.skip_train_pool):
        start = time.perf_counter()
        train_manifest = generate_service_territory_pool(
            config_path=args.config_path,
            save_path=train_pool_path,
            num_territories=train_territories,
            latent_customer_pool_size=args.latent_customer_pool_size,
            cs_candidate_pool_size=args.cs_candidate_pool_size,
            seed=args.seed,
            num_workers=args.pool_workers,
        )
        elapsed = time.perf_counter() - start
        timing_rows.append({
            "stage": "service_territory_pool",
            "split": "train",
            "suite_name": "",
            "num_service_territories": train_territories,
            "num_instances": "",
            "num_customers": "",
            "num_charging_stations": "",
            "wall_time_s": round(elapsed, 6),
            "output_path": train_manifest["bundle_path"],
            "output_size_bytes": file_size_bytes(train_manifest["bundle_path"]),
        })
        print(json.dumps({"train_service_territory_pool": train_manifest["pool_path"], "num_service_territories": train_territories, "wall_time_s": round(elapsed, 3)}, indent=2))

    if not (args.skip_pool or args.skip_eval_pool):
        start = time.perf_counter()
        eval_manifest = generate_service_territory_pool(
            config_path=args.config_path,
            save_path=eval_pool_path,
            num_territories=eval_territories,
            latent_customer_pool_size=args.latent_customer_pool_size,
            cs_candidate_pool_size=args.cs_candidate_pool_size,
            seed=int(args.seed) + 10_000,
            num_workers=args.pool_workers,
        )
        elapsed = time.perf_counter() - start
        timing_rows.append({
            "stage": "service_territory_pool",
            "split": "eval",
            "suite_name": "",
            "num_service_territories": eval_territories,
            "num_instances": "",
            "num_customers": "",
            "num_charging_stations": "",
            "wall_time_s": round(elapsed, 6),
            "output_path": eval_manifest["bundle_path"],
            "output_size_bytes": file_size_bytes(eval_manifest["bundle_path"]),
        })
        print(json.dumps({"eval_service_territory_pool": eval_manifest["pool_path"], "num_service_territories": eval_territories, "wall_time_s": round(elapsed, 3)}, indent=2))

    if args.skip_eval_instances:
        write_timing_csv(dataset_root / "generation_timing.csv", timing_rows)
        return

    selected = args.suite or list(SUITES)
    for offset, suite_name in enumerate(selected):
        num_customers, num_cs = SUITES[suite_name]
        suite_path = eval_pool_path / suite_name
        generator = HierarchyDatasetGenerator.from_config_path(args.config_path, seed=int(args.seed) + 1000 + offset)
        generator.load_region_pool(
            pool_path=eval_pool_path,
            num_regions=eval_territories,
            shuffle=True,
            replacement_policy="cycle",
        )
        start = time.perf_counter()
        summary = generator.generate(
            save_path=suite_path,
            num_instances=args.num_instances,
            num_customers=num_customers,
            num_charging_stations=num_cs,
            num_regions=eval_territories,
            mother_num_customers=args.latent_customer_pool_size,
            mother_num_charging_stations=args.cs_candidate_pool_size,
            region_reuse_limit=200,
            save_plots=bool(args.plots),
            save_regions=False,
            dataset_metadata=dataset_metadata(args, suite_name, num_customers, num_cs, eval_pool_path),
            clear_oracle_after_instance=True,
        )
        elapsed = time.perf_counter() - start
        timing_rows.append({
            "stage": "operating_day_instances",
            "split": "eval",
            "suite_name": suite_name,
            "num_service_territories": eval_territories,
            "num_instances": int(args.num_instances),
            "num_customers": int(num_customers),
            "num_charging_stations": int(num_cs),
            "wall_time_s": round(elapsed, 6),
            "output_path": summary["instances_path"],
            "output_size_bytes": file_size_bytes(summary["instances_path"]),
        })
        print(json.dumps({"suite": suite_name, "instances_path": summary["instances_path"], "num_instances": summary["num_instances"], "wall_time_s": round(elapsed, 3)}, indent=2))

    write_timing_csv(dataset_root / "generation_timing.csv", timing_rows)
    manifest = {
        "dataset_family": "EVRPTW-D",
        "dataset_version": "AC-v1",
        "calibration_profile": "Amazon-Calibrated",
        "train_service_territory_pool": str(train_pool_path / "service_territory_pool.pkl"),
        "eval_service_territory_pool": str(eval_pool_path / "service_territory_pool.pkl"),
        "train_num_service_territories": train_territories,
        "eval_num_service_territories": eval_territories,
        "eval_num_instances_per_suite": int(args.num_instances),
        "suites": {name: {"num_customers": values[0], "num_charging_stations": values[1]} for name, values in SUITES.items()},
        "timing_csv": str(dataset_root / "generation_timing.csv"),
        "layout": {
            "train": "Reusable training service-territory pool only; training instances are sampled online.",
            "eval": "Held-out service-territory pool plus fixed operating-day instance bundles.",
        },
    }
    with (dataset_root / "dataset_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
