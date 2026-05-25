from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_ROOT = REPO_ROOT / "EVRPTW_Dataset_Generator"
sys.path.insert(0, str(GENERATOR_ROOT))

from evrptw_hierarchy.generation.generator import HierarchyDatasetGenerator


def default_save_path(args: argparse.Namespace) -> Path:
    return (
        REPO_ROOT
        / "EVRPTW_Dataset"
        / "Amazon_Calibrated_v1"
        / "region_pools"
        / f"mother_N{args.mother_num_customers}_CS{args.mother_num_charging_stations}_R{args.num_regions}_seed{args.seed}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare reusable EVRPTW mother-board / region pools.")
    parser.add_argument("--config-path", type=Path, default=GENERATOR_ROOT / "configs/amazon_hierarchy.yaml")
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--num-regions", type=int, default=256)
    parser.add_argument("--mother-num-customers", type=int, default=5000)
    parser.add_argument("--mother-num-charging-stations", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260525)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_path = args.save_path or default_save_path(args)
    generator = HierarchyDatasetGenerator.from_config_path(args.config_path, seed=args.seed)
    manifest = generator.save_region_pool(
        save_path=save_path,
        num_regions=args.num_regions,
        mother_num_customers=args.mother_num_customers,
        mother_num_charging_stations=args.mother_num_charging_stations,
    )
    print(json.dumps({
        "pool_path": manifest["pool_path"],
        "num_regions": manifest["num_regions"],
        "mother_num_customers": manifest["mother_num_customers"],
        "mother_num_charging_stations": manifest["mother_num_charging_stations"],
        "seed": manifest["seed"],
    }, indent=2))


if __name__ == "__main__":
    main()
