from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_ROOT = REPO_ROOT / "EVRPTW_Dataset_Generator"
sys.path.insert(0, str(GENERATOR_ROOT))

from evrptw_hierarchy.configs.config import deep_update, load_yaml, vehicle_from_config
from evrptw_hierarchy.core.models import RegionBoard, RegionUsage
from evrptw_hierarchy.generation.generator import HierarchyDatasetGenerator
from evrptw_hierarchy.generation.territory_pool import SERVICE_TERRITORY_POOL_FORMAT
from evrptw_hierarchy.geospatial import GeospatialTerritoryBuilder
from evrptw_hierarchy.io.persistence import ensure_dir
from evrptw_hierarchy.validation.reports import summarize_region


REAL_SOURCE_KEYS = (
    "road_nodes_csv",
    "road_edges_csv",
    "customer_seed_csv",
    "charging_station_csv",
    "depot_candidate_csv",
)
OPTIONAL_REAL_SOURCE_KEYS = ("latent_customer_csv",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Geo-AC-v1 county-container/depot-catchment benchmark suites.")
    parser.add_argument("--city-config", type=Path, default=GENERATOR_ROOT / "configs/geo_ac_v1_us10.yaml")
    parser.add_argument("--base-config", type=Path, default=GENERATOR_ROOT / "configs/amazon_hierarchy.yaml")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "EVRPTW_Dataset" / "Geo_AC_v1")
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--territory-limit", type=int, default=None)
    parser.add_argument("--territory-id", action="append", default=None)
    parser.add_argument("--scales", default="", help="Comma-separated customer scales. Defaults to config scales.")
    parser.add_argument(
        "--split-name",
        default="eval",
        help="Dataset split directory and metadata label, e.g. train, validation_reference, eval, offline_industrial.",
    )
    parser.add_argument("--instances-per-scale", type=int, default=None)
    parser.add_argument("--latent-customer-pool-size", type=int, default=None)
    parser.add_argument("--cs-candidate-pool-size", type=int, default=None)
    parser.add_argument("--skip-instances", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Skip a territory/scale split when instances.pkl already has the requested count.")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--require-real-sources", action="store_true", help="Fail if any territory would use fallback synthetic geospatial scaffolds.")
    return parser.parse_args()


def _fresh_usage(board: RegionBoard) -> RegionUsage:
    return RegionUsage(
        region_id=board.region_id,
        sampled_days=0,
        customer_activation_counts=np.zeros(len(board.customers), dtype=np.int32),
        cluster_activation_counts=np.zeros(len(board.cluster_centers), dtype=np.int32),
    )


def _resolve_path(path: str | Path | None, base: Path) -> str | None:
    if path in (None, ""):
        return None
    p = Path(path)
    return str(p if p.is_absolute() else base / p)


def _resolve_source_files(spec: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    out = dict(spec)
    sources = dict(spec.get("source_files", {}) or {})
    data_root = _resolve_path(out.get("data_root"), config_dir)
    base = Path(data_root) if data_root is not None else config_dir
    out["data_root"] = data_root
    out["source_files"] = {key: _resolve_path(value, base) for key, value in sources.items()}
    return out


def _selected_territories(cfg: dict[str, Any], args: argparse.Namespace, config_dir: Path) -> list[dict[str, Any]]:
    territories = [_resolve_source_files(item, config_dir) for item in cfg.get("territories", [])]
    if args.territory_id:
        keep = set(args.territory_id)
        territories = [item for item in territories if str(item.get("territory_id")) in keep]
    if args.territory_limit is not None:
        territories = territories[: int(args.territory_limit)]
    if not territories:
        raise ValueError("No territories selected.")
    return territories


def _scale_map(cfg: dict[str, Any], args: argparse.Namespace) -> dict[int, int]:
    raw = cfg.get("scales", {}) or {5: 3, 15: 3, 50: 10, 100: 20}
    out = {int(k): int(v) for k, v in raw.items()}
    if args.scales.strip():
        requested: dict[int, int] = {}
        for item in args.scales.split(","):
            token = item.strip()
            if not token:
                continue
            if ":" in token:
                scale_raw, cs_raw = token.split(":", 1)
                requested[int(scale_raw.strip())] = int(cs_raw.strip())
            else:
                scale = int(token)
                if scale in out:
                    requested[scale] = out[scale]
                else:
                    requested[scale] = max(3, min(160, int(round(scale * 0.1))))
        out = {scale: requested[scale] for scale in sorted(requested)}
    if not out:
        raise ValueError("No valid scales selected.")
    return out


def _csv_has_rows(path: str | Path | None) -> bool:
    if path in (None, ""):
        return False
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return False
    with p.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in f) > 1


def _require_real_source_files(spec: dict[str, Any]) -> None:
    sources = spec.get("source_files", {}) or {}
    required = list(REAL_SOURCE_KEYS)
    required.extend(key for key in OPTIONAL_REAL_SOURCE_KEYS if key in sources)
    missing = [key for key in required if not _csv_has_rows(sources.get(key))]
    if missing:
        detail = {key: sources.get(key, "") for key in missing}
        raise FileNotFoundError(
            f"{spec.get('territory_id')} is missing non-empty real geospatial source CSVs: {detail}"
        )


def _config_for_spec(base_cfg: dict[str, Any], suite_cfg: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    cfg = deep_update(base_cfg, suite_cfg.get("config_overrides", {}) or {})
    cfg = deep_update(cfg, spec.get("config_overrides", {}) or {})
    cfg = deep_update(
        cfg,
        {
            "geospatial": {
                "depot_catchment": spec.get(
                    "depot_catchment",
                    suite_cfg.get("depot_catchment", {"start_radius_km": 40.0, "max_radius_km": 55.0}),
                )
            }
        },
    )
    return cfg


def _write_pool(root: Path, board: RegionBoard, suite_cfg: dict[str, Any], spec: dict[str, Any], seed: int) -> dict[str, Any]:
    ensure_dir(root)
    bundle_path = root / "service_territory_pool.pkl"
    header = {
        "format": SERVICE_TERRITORY_POOL_FORMAT,
        "dataset_family": "EVRPTW-D",
        "dataset_version": "Geo-AC-v1",
        "calibration_profile": "Amazon-Calibrated real-geography semi-synthetic",
        "num_service_territories": 1,
        "latent_customer_pool_size": int(len(board.customers)),
        "cs_candidate_pool_size": int(len(board.charging_stations)),
        "seed": int(seed),
        "territory_id": board.region_id,
    }
    with bundle_path.open("wb") as f:
        pickle.dump(header, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(board.to_pickle_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)
    row = summarize_region(board, _fresh_usage(board))
    manifest = {
        "dataset_family": "EVRPTW-D",
        "dataset_version": "Geo-AC-v1",
        "profile_name": suite_cfg.get("profile_name", "Geo-AC-v1 / NA-US-10"),
        "pool_type": "geospatial_service_territory_pool",
        "pool_path": str(root),
        "bundle_path": str(bundle_path),
        "format": SERVICE_TERRITORY_POOL_FORMAT,
        "territory": {
            "territory_id": board.region_id,
            "display_name": spec.get("display_name", board.region_id),
            "county_name": spec.get("county_name", ""),
            "state": spec.get("state", ""),
            "county_fips": str(spec.get("county_fips", "")),
        },
        "num_service_territories": 1,
        "latent_customer_pool_size": int(len(board.customers)),
        "cs_candidate_pool_size": int(len(board.charging_stations)),
        "num_depot_candidates": int(0 if board.depot_candidate_node_ids is None else len(board.depot_candidate_node_ids)),
        "region_row": row,
        "source_mode": board.metadata.get("source_mode", ""),
        "source_files": spec.get("source_files", {}),
        "notes": [
            "County container is used for data acquisition and provenance.",
            "Daily instances sample an anonymous depot candidate and use a road-distance catchment.",
            "Orders, demand, service time, and time windows remain Amazon-calibrated semi-synthetic attributes.",
        ],
    }
    with (root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _dataset_metadata(
    suite_cfg: dict[str, Any],
    spec: dict[str, Any],
    split_name: str,
    scale: int,
    num_cs: int,
    pool_path: Path,
) -> dict[str, Any]:
    return {
        "dataset_family": "EVRPTW-D",
        "dataset_version": "Geo-AC-v1",
        "profile_name": suite_cfg.get("profile_name", "Geo-AC-v1 / NA-US-10"),
        "split": split_name,
        "territory_id": spec["territory_id"],
        "display_name": spec.get("display_name", spec["territory_id"]),
        "county_name": spec.get("county_name", ""),
        "state": spec.get("state", ""),
        "county_fips": str(spec.get("county_fips", "")),
        "suite_name": f"Cus_{scale}",
        "num_customers": int(scale),
        "num_charging_stations": int(num_cs),
        "service_territory_pool_path": str(pool_path),
        "framing": "real-geography semi-synthetic EVRPTW benchmark",
        "terminology": {
            "county_container": "County used to acquire and publish public geospatial source data.",
            "depot_catchment": "Road-distance service area around one anonymous depot candidate selected for an operating day.",
        },
    }


def _write_timing_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "territory_id",
        "suite_name",
        "num_instances",
        "num_customers",
        "num_charging_stations",
        "wall_time_s",
        "output_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _existing_bundle_count(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with path.open("rb") as f:
            header = pickle.load(f)
    except Exception:
        return None
    if not isinstance(header, dict) or header.get("format") != "evrptw_instance_bundle_v1":
        return None
    try:
        return int(header.get("num_instances"))
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    city_cfg = load_yaml(args.city_config)
    base_cfg = load_yaml(args.base_config)
    config_dir = args.city_config.resolve().parent
    output_root = Path(args.output_root)
    split_name = str(args.split_name).strip() or "eval"
    territories = _selected_territories(city_cfg, args, config_dir)
    scales = _scale_map(city_cfg, args)
    instances_per_scale = int(args.instances_per_scale or city_cfg.get("instances_per_scale", 100))
    timing_rows: list[dict[str, Any]] = []

    for territory_index, spec_raw in enumerate(territories):
        spec = dict(spec_raw)
        if args.latent_customer_pool_size is not None:
            spec["latent_customer_pool_size"] = int(args.latent_customer_pool_size)
        if args.cs_candidate_pool_size is not None:
            spec["cs_candidate_pool_size"] = int(args.cs_candidate_pool_size)
        if args.require_real_sources:
            _require_real_source_files(spec)
        seed = int(args.seed) + territory_index * 10_000
        cfg = _config_for_spec(base_cfg, city_cfg, spec)
        vehicle = vehicle_from_config(cfg)
        rng = np.random.default_rng(seed)
        builder = GeospatialTerritoryBuilder(cfg, vehicle, rng)
        start = time.perf_counter()
        board = builder.build(spec, territory_index)
        if args.require_real_sources and board.metadata.get("source_mode") != "standard_geospatial_csv":
            raise RuntimeError(
                f"{spec['territory_id']} did not build from standard geospatial CSVs; "
                f"source_mode={board.metadata.get('source_mode')!r}"
            )
        territory_root = ensure_dir(output_root / "service_territories" / str(spec["territory_id"]))
        manifest = _write_pool(territory_root, board, city_cfg, spec, seed)
        elapsed = time.perf_counter() - start
        timing_rows.append({
            "stage": "geospatial_service_territory",
            "territory_id": spec["territory_id"],
            "suite_name": "",
            "num_instances": "",
            "num_customers": "",
            "num_charging_stations": "",
            "wall_time_s": round(elapsed, 6),
            "output_path": manifest["bundle_path"],
        })
        print(json.dumps({"territory": spec["territory_id"], "pool": manifest["bundle_path"], "wall_time_s": round(elapsed, 3)}, indent=2))

        if args.skip_instances:
            continue

        for scale, num_cs in scales.items():
            suite_path = output_root / split_name / str(spec["territory_id"]) / f"Cus_{scale}"
            existing_count = _existing_bundle_count(suite_path / "instances.pkl")
            if args.skip_existing and existing_count == int(instances_per_scale):
                timing_rows.append({
                    "stage": "operating_day_instances_skipped_existing",
                    "territory_id": spec["territory_id"],
                    "suite_name": f"Cus_{scale}",
                    "num_instances": int(instances_per_scale),
                    "num_customers": int(scale),
                    "num_charging_stations": int(num_cs),
                    "wall_time_s": 0.0,
                    "output_path": str(suite_path / "instances.pkl"),
                })
                print(json.dumps({
                    "territory": spec["territory_id"],
                    "suite": f"Cus_{scale}",
                    "instances_path": str(suite_path / "instances.pkl"),
                    "skipped_existing": True,
                }, indent=2))
                continue
            generator = HierarchyDatasetGenerator(cfg, seed=seed + int(scale))
            generator.boards = [board]
            generator.usages = [_fresh_usage(board)]
            generator.precomputed_boards = [board]
            generator.precomputed_replacement_policy = "cycle"
            generator.next_precomputed_index = 0
            region_reuse_limit = max(
                int(spec.get("region_reuse_limit", city_cfg.get("region_reuse_limit", 200))),
                int(instances_per_scale) + 1,
            )
            start = time.perf_counter()
            summary = generator.generate(
                save_path=suite_path,
                num_instances=instances_per_scale,
                num_customers=int(scale),
                num_charging_stations=int(num_cs),
                num_regions=1,
                mother_num_customers=int(len(board.customers)),
                mother_num_charging_stations=int(len(board.charging_stations)),
                region_reuse_limit=region_reuse_limit,
                max_attempts_per_instance=spec.get("max_attempts_per_instance", city_cfg.get("max_attempts_per_instance", None)),
                save_plots=bool(args.plots),
                save_regions=False,
                dataset_metadata=_dataset_metadata(city_cfg, spec, split_name, int(scale), int(num_cs), territory_root),
                clear_oracle_after_instance=False,
            )
            elapsed = time.perf_counter() - start
            timing_rows.append({
                "stage": "operating_day_instances",
                "territory_id": spec["territory_id"],
                "suite_name": f"Cus_{scale}",
                "num_instances": int(instances_per_scale),
                "num_customers": int(scale),
                "num_charging_stations": int(num_cs),
                "wall_time_s": round(elapsed, 6),
                "output_path": summary["instances_path"],
            })
            print(json.dumps({"territory": spec["territory_id"], "suite": f"Cus_{scale}", "instances_path": summary["instances_path"], "wall_time_s": round(elapsed, 3)}, indent=2))

    _write_timing_csv(output_root / "generation_timing.csv", timing_rows)
    manifest = {
        "dataset_family": "EVRPTW-D",
        "dataset_version": "Geo-AC-v1",
        "profile_name": city_cfg.get("profile_name", "Geo-AC-v1 / NA-US-10"),
        "split_name": split_name,
        "territory_count": len(territories),
        "territories": [
            {
                "territory_id": spec["territory_id"],
                "display_name": spec.get("display_name", spec["territory_id"]),
                "county_name": spec.get("county_name", ""),
                "state": spec.get("state", ""),
                "county_fips": str(spec.get("county_fips", "")),
            }
            for spec in territories
        ],
        "scales": {str(k): int(v) for k, v in scales.items()},
        "instances_per_scale": int(instances_per_scale),
        "timing_csv": str(output_root / "generation_timing.csv"),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "dataset_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
