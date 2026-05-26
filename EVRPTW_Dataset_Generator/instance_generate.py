from __future__ import annotations

import argparse
from pathlib import Path

from evrptw_hierarchy.generation.generator import HierarchyDatasetGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EVRPTW-D daily instances from service-territory graphs.")
    parser.add_argument("--config_path", type=str, default="configs/amazon_hierarchy.yaml")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--num_instances", type=int, default=1000)
    parser.add_argument("--num_customers", type=int, default=1800)
    parser.add_argument("--num_charging_stations", type=int, default=12)
    parser.add_argument("--num_regions", "--num_service_territories", dest="num_regions", type=int, default=10)
    parser.add_argument("--mother_num_customers", "--latent_customer_pool_size", dest="mother_num_customers", type=int, default=None)
    parser.add_argument("--mother_num_charging_stations", "--cs_candidate_pool_size", dest="mother_num_charging_stations", type=int, default=120)
    parser.add_argument("--region_reuse_limit", "--territory_reuse_limit", dest="region_reuse_limit", type=int, default=200)
    parser.add_argument("--max_attempts_per_instance", type=int, default=None)
    parser.add_argument("--territory_pool_path", "--region_pool_path", dest="territory_pool_path", type=str, default=None)
    parser.add_argument("--territory_pool_shuffle", action="store_true", default=True)
    parser.add_argument("--no_territory_pool_shuffle", action="store_false", dest="territory_pool_shuffle")
    parser.add_argument("--territory_pool_replacement_policy", type=str, default="cycle", choices=["cycle", "generate"])
    parser.add_argument("--dataset_family", type=str, default="EVRPTW-D")
    parser.add_argument("--dataset_version", type=str, default="AC-v1")
    parser.add_argument("--calibration_profile", type=str, default="Amazon-Calibrated")
    parser.add_argument("--suite_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--no_save_regions", action="store_true")
    parser.add_argument("--clear_oracle_after_instance", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    generator = HierarchyDatasetGenerator.from_config_path(config_path, seed=args.seed)
    mother_num_customers = args.mother_num_customers
    if mother_num_customers is None:
        mother_num_customers = int(generator.config.get("region", {}).get("default_mother_num_customers", 5000))

    if args.territory_pool_path not in (None, ""):
        generator.load_region_pool(
            pool_path=args.territory_pool_path,
            num_regions=args.num_regions,
            shuffle=bool(args.territory_pool_shuffle),
            replacement_policy=str(args.territory_pool_replacement_policy),
        )

    dataset_metadata = {
        "dataset_family": args.dataset_family,
        "dataset_version": args.dataset_version,
        "calibration_profile": args.calibration_profile,
        "suite_name": args.suite_name,
        "num_instances": int(args.num_instances),
        "num_customers": int(args.num_customers),
        "num_charging_stations": int(args.num_charging_stations),
        "num_service_territories": int(args.num_regions),
        "latent_customer_pool_size": int(mother_num_customers),
        "cs_candidate_pool_size": int(args.mother_num_charging_stations),
        "service_territory_pool_path": str(args.territory_pool_path or ""),
        "terminology": {
            "service_territory_graph": "Stable city/region/delivery-station service territory.",
            "operating_day_instance": "Active daily EVRP-TW instance sampled from one service territory.",
            "legacy_internal_fields": ["mother_board_id", "mother_board_pool_size", "region_pool_path"],
        },
    }

    summary = generator.generate(
        save_path=args.save_path,
        num_instances=args.num_instances,
        num_customers=args.num_customers,
        num_charging_stations=args.num_charging_stations,
        num_regions=args.num_regions,
        mother_num_customers=mother_num_customers,
        mother_num_charging_stations=args.mother_num_charging_stations,
        region_reuse_limit=args.region_reuse_limit,
        max_attempts_per_instance=args.max_attempts_per_instance,
        save_plots=not args.no_plots,
        save_regions=not args.no_save_regions,
        dataset_metadata=dataset_metadata,
        clear_oracle_after_instance=bool(args.clear_oracle_after_instance),
    )
    print(f"Generated {summary['num_instances']} instances")
    print(f"Instances: {summary['instances_dir']}")
    print(f"Metadata: {Path(args.save_path) / 'metadata'}")
    print(f"Reports: {Path(args.save_path) / 'analysis_outputs'}")


if __name__ == "__main__":
    main()
