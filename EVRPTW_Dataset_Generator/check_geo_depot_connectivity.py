from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_ROOT = REPO_ROOT / "EVRPTW_Dataset_Generator"
sys.path.insert(0, str(GENERATOR_ROOT))

from evrptw_hierarchy.configs.config import deep_update, load_yaml, vehicle_from_config
from evrptw_hierarchy.geospatial import GeospatialTerritoryBuilder
from evrptw_hierarchy.graph.distance_oracle import DistanceOracle

from prepare_geospatial_benchmark_suite import _resolve_source_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check depot-candidate road connectivity for Geo-AC territories.")
    parser.add_argument("--city-config", type=Path, default=GENERATOR_ROOT / "configs/geo_ac_v1_na_us20.with_sources.yaml")
    parser.add_argument("--base-config", type=Path, default=GENERATOR_ROOT / "configs/amazon_hierarchy.yaml")
    parser.add_argument("--scale", type=int, default=100)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def config_for_spec(base_cfg: dict[str, Any], suite_cfg: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
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


def main() -> None:
    args = parse_args()
    city_cfg = load_yaml(args.city_config)
    base_cfg = load_yaml(args.base_config)
    config_dir = args.city_config.resolve().parent
    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(city_cfg.get("territories", [])):
        spec = _resolve_source_files(raw, config_dir)
        cfg = config_for_spec(base_cfg, city_cfg, spec)
        vehicle = vehicle_from_config(cfg)
        board = GeospatialTerritoryBuilder(cfg, vehicle, np.random.default_rng(20260526 + idx)).build(spec, idx)
        if board.depot_candidate_node_ids is None or len(board.depot_candidate_node_ids) == 0:
            depot_nodes = np.asarray([board.depot_node_id], dtype=np.int32)
        else:
            depot_nodes = np.asarray(board.depot_candidate_node_ids, dtype=np.int32)
        oracle = DistanceOracle(len(board.road_nodes), board.road_edges, board.road_edge_lengths_km)
        dist = oracle.matrix_between(depot_nodes, board.customer_node_ids).astype(np.float32)
        finite_counts = np.sum(np.isfinite(dist), axis=1).astype(int)
        within_40 = np.sum(np.isfinite(dist) & (dist <= 40.0), axis=1).astype(int)
        within_55 = np.sum(np.isfinite(dist) & (dist <= 55.0), axis=1).astype(int)
        row = {
            "territory_id": spec["territory_id"],
            "customers": int(len(board.customers)),
            "depots": int(depot_nodes.size),
            "max_finite": int(finite_counts.max()) if finite_counts.size else 0,
            "max_within_40": int(within_40.max()) if within_40.size else 0,
            "max_within_55": int(within_55.max()) if within_55.size else 0,
            "depots_ge_scale": int(np.count_nonzero(within_55 >= int(args.scale))),
            "finite_counts": finite_counts.tolist(),
            "within_55_counts": within_55.tolist(),
        }
        rows.append(row)
        print(json.dumps(row))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
