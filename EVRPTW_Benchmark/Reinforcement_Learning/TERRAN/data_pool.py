from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATOR_ROOT = REPO_ROOT / "EVRPTW_Dataset_Generator"
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))
sys.path.insert(0, str(GENERATOR_ROOT))

from evrptw_core.schema import EVRPTWInstance
from evrptw_hierarchy.generation.generator import HierarchyDatasetGenerator


@dataclass
class OnlineInstancePool:
    """In-memory mother-board pool for online TERRAN training."""

    config_path: str | Path
    num_regions: int = 32
    mother_num_customers: int = 5000
    mother_num_charging_stations: int = 120
    num_customers: int = 15
    num_charging_stations: int = 3
    region_reuse_limit: int = 200
    seed: int | None = None
    max_attempts_per_instance: int | None = None
    region_pool_path: str | Path | None = None
    region_pool_shuffle: bool = True
    region_pool_replacement_policy: str = "cycle"

    def __post_init__(self) -> None:
        path = Path(self.config_path)
        if not path.is_absolute():
            path = GENERATOR_ROOT / path
        self.config_path = path
        self.generator = HierarchyDatasetGenerator.from_config_path(path, seed=self.seed)
        self.region_pool_status = "generated_online"
        loaded_precomputed = False
        if self.region_pool_path not in (None, ""):
            pool_path = Path(self.region_pool_path)
            if not pool_path.is_absolute():
                pool_path = REPO_ROOT / pool_path
            try:
                self.generator.load_region_pool(
                    pool_path=pool_path,
                    num_regions=int(self.num_regions),
                    shuffle=bool(self.region_pool_shuffle),
                    replacement_policy=str(self.region_pool_replacement_policy),
                )
                if len(self.generator.boards) >= int(self.num_regions):
                    loaded_precomputed = True
                    self.region_pool_status = f"loaded_precomputed:{pool_path}"
                else:
                    self.region_pool_status = (
                        f"precomputed_pool_insufficient:{pool_path}:"
                        f"{len(self.generator.boards)}<{int(self.num_regions)}"
                    )
            except Exception as exc:  # Optional acceleration path; training must remain robust.
                self.region_pool_status = f"precomputed_pool_failed:{self.region_pool_path}:{exc}"

        if not loaded_precomputed:
            self.generator = HierarchyDatasetGenerator.from_config_path(path, seed=self.seed)
            self.generator.prepare_region_pool(
                num_regions=self.num_regions,
                mother_num_customers=self.mother_num_customers,
                mother_num_charging_stations=self.mother_num_charging_stations,
            )
        self.sample_count = 0

    def sample(self) -> EVRPTWInstance:
        active = self.generator.sample_active_instance(
            num_customers=self.num_customers,
            num_charging_stations=self.num_charging_stations,
            region_reuse_limit=self.region_reuse_limit,
            mother_num_customers=self.mother_num_customers,
            mother_num_charging_stations=self.mother_num_charging_stations,
            instance_index=self.sample_count,
            max_attempts_per_instance=self.max_attempts_per_instance,
        )
        self.sample_count += 1
        return EVRPTWInstance.from_dict(active.to_pickle_dict())

    def usage_summary(self) -> list[dict[str, Any]]:
        rows = []
        for board, usage in zip(self.generator.boards, self.generator.usages):
            rows.append(
                {
                    "region_id": board.region_id,
                    "sampled_days": usage.sampled_days,
                    "customer_exposure_rate": usage.customer_exposure_rate,
                    "recent_mean_jaccard_distance": usage.recent_mean_jaccard_distance,
                    "cluster_exposure_entropy": usage.cluster_exposure_entropy,
                    "region_pool_status": self.region_pool_status,
                }
            )
        return rows


__all__ = ["OnlineInstancePool"]
