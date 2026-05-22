from __future__ import annotations

import argparse
from pathlib import Path

from evrptw_hierarchy.generation.generator import HierarchyDatasetGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EVRP-TW-Hierarchy-D daily instances from region mother boards.")
    parser.add_argument("--config_path", type=str, default="configs/amazon_hierarchy.yaml")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--num_instances", type=int, default=1000)
    parser.add_argument("--num_customers", type=int, default=1800)
    parser.add_argument("--num_charging_stations", type=int, default=12)
    parser.add_argument("--num_regions", type=int, default=10)
    parser.add_argument("--mother_num_customers", type=int, default=5000)
    parser.add_argument("--mother_num_charging_stations", type=int, default=120)
    parser.add_argument("--region_reuse_limit", type=int, default=200)
    parser.add_argument("--max_attempts_per_instance", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    generator = HierarchyDatasetGenerator.from_config_path(config_path, seed=args.seed)
    summary = generator.generate(
        save_path=args.save_path,
        num_instances=args.num_instances,
        num_customers=args.num_customers,
        num_charging_stations=args.num_charging_stations,
        num_regions=args.num_regions,
        mother_num_customers=args.mother_num_customers,
        mother_num_charging_stations=args.mother_num_charging_stations,
        region_reuse_limit=args.region_reuse_limit,
        max_attempts_per_instance=args.max_attempts_per_instance,
        save_plots=not args.no_plots,
    )
    print(f"Generated {summary['num_instances']} instances")
    print(f"Instances: {summary['instances_dir']}")
    print(f"Metadata: {Path(args.save_path) / 'metadata'}")
    print(f"Reports: {Path(args.save_path) / 'analysis_outputs'}")


if __name__ == "__main__":
    main()
