from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_ROOT = REPO_ROOT / "EVRPTW_Dataset_Generator"
sys.path.insert(0, str(GENERATOR_ROOT))

from evrptw_hierarchy.generation.territory_pool import generate_service_territory_pool


def default_save_path(args: argparse.Namespace) -> Path:
    dataset_root = args.dataset_root or (REPO_ROOT / "EVRPTW_Dataset" / "AC_v1")
    return Path(dataset_root) / "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a reusable EVRPTW-D training service-territory pool.")
    parser.add_argument("--config-path", type=Path, default=GENERATOR_ROOT / "configs/amazon_hierarchy.yaml")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "EVRPTW_Dataset" / "AC_v1")
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--num-territories", "--num-regions", dest="num_regions", type=int, default=1024)
    parser.add_argument("--latent-customer-pool-size", "--mother-num-customers", dest="mother_num_customers", type=int, default=5000)
    parser.add_argument("--cs-candidate-pool-size", "--mother-num-charging-stations", dest="mother_num_charging_stations", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_path = args.save_path or default_save_path(args)
    manifest = generate_service_territory_pool(
        config_path=args.config_path,
        save_path=save_path,
        num_territories=args.num_regions,
        latent_customer_pool_size=args.mother_num_customers,
        cs_candidate_pool_size=args.mother_num_charging_stations,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    print(json.dumps({
        "pool_path": manifest["pool_path"],
        "pool_type": manifest.get("pool_type", "service_territory_pool"),
        "num_service_territories": manifest.get("num_service_territories", manifest.get("num_regions")),
        "latent_customer_pool_size": manifest.get("latent_customer_pool_size", manifest.get("mother_num_customers")),
        "cs_candidate_pool_size": manifest.get("cs_candidate_pool_size", manifest.get("mother_num_charging_stations")),
        "seed": manifest["seed"],
    }, indent=2))


if __name__ == "__main__":
    main()
