from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATOR_ROOT = REPO_ROOT / "EVRPTW_Dataset_Generator"
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))
sys.path.insert(0, str(GENERATOR_ROOT))

from evrptw_core.io import load_instances
from evrptw_core.schema import EVRPTWInstance

from ..common import Stage2TaskPool
from ..common.data_pass import seeded_pass_order
from ..common.training_stream import read_stream_view_ids


@dataclass
class Stage2TERRANPool:
    """No-regeneration sampler over frozen Stage-2 view records."""

    dataset_path: str | Path
    family_root: str | Path | None = None
    scale: str | int | None = None
    split_ids: str | None = "train"
    track_ids: str | None = "train"
    city_slugs: str | None = None
    seed: int = 1234
    cache_size: int = 4
    completed_data_passes: int = 0
    training_stream_path: str | Path | None = None
    representation: str = "G"
    euclidean_manifest: str | Path | None = None

    def __post_init__(self) -> None:
        self.pool = Stage2TaskPool(
            dataset_path=self.dataset_path,
            family_root=self.family_root,
            scale=self.scale,
            split_ids=self.split_ids,
            track_ids=self.track_ids,
            city_slugs=self.city_slugs,
            seed=self.seed,
            cache_size=self.cache_size,
            representation=self.representation,
            euclidean_manifest=self.euclidean_manifest,
        )
        self._stream_view_ids = (
            read_stream_view_ids(self.training_stream_path)
            if self.training_stream_path is not None
            else None
        )
        self._task_by_view_id = {task.view_id: task for task in self.pool.tasks}
        if self._stream_view_ids is not None:
            missing = sorted(set(self._stream_view_ids).difference(self._task_by_view_id))
            if missing:
                raise ValueError(
                    f"TERRAN training stream contains IDs outside its pool: {missing[:3]}"
                )
            self.sample_count = 0
            self._order = None
        else:
            self.sample_count = int(self.completed_data_passes) * len(self.pool)
            self._order = seeded_pass_order(
                len(self.pool), self.seed, int(self.completed_data_passes) + 1
            )
        self.region_pool_status = f"stage2_frozen:{Path(self.dataset_path)}"

    def sample(self) -> EVRPTWInstance:
        if self._stream_view_ids is not None:
            if self.sample_count >= len(self._stream_view_ids):
                raise RuntimeError("TERRAN exhausted the registered training ID stream")
            view_id = self._stream_view_ids[self.sample_count]
            self.sample_count += 1
            return self.pool.instance(self._task_by_view_id[view_id])
        offset = self.sample_count % len(self.pool)
        if self.sample_count and offset == 0:
            data_pass = self.sample_count // len(self.pool) + 1
            self._order = seeded_pass_order(len(self.pool), self.seed, data_pass)
        task = self.pool.tasks[int(self._order[offset])]
        self.sample_count += 1
        return self.pool.instance(task)

    def usage_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "region_id": "stage2_frozen",
                "sampled_days": self.sample_count,
                "customer_exposure_rate": "",
                "recent_mean_jaccard_distance": "",
                "cluster_exposure_entropy": "",
                "region_pool_status": self.region_pool_status,
                "dataset_size": len(self.pool),
                "sample_mode": (
                    "registered_stratified_stream_with_replacement"
                    if self._stream_view_ids is not None
                    else "seeded_shuffle_cycle_without_replacement"
                ),
            }
        ]

    def close(self, terminate: bool = False) -> None:
        del terminate


@dataclass
class FixedDatasetInstancePool:
    """Reusable sampler over a fixed EVRPTW-D instance bundle.

    The public dataset release stores train/val/eval as consolidated
    ``instances.pkl`` streams. For RL training we keep the train split immutable
    and sample operating-day instances from that finite bundle instead of
    generating new active days online.
    """

    dataset_path: str | Path
    num_customers: int | None = None
    num_charging_stations: int | None = None
    seed: int | None = None
    sample_mode: str = "shuffle_cycle"

    def __post_init__(self) -> None:
        path = Path(self.dataset_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        self.dataset_path = path
        self.instances = load_instances(
            path,
            num_customers=self.num_customers,
            num_charging_stations=self.num_charging_stations,
        )
        if not self.instances:
            raise FileNotFoundError(
                f"No EVRPTW instances found under {path} "
                f"for Cus{self.num_customers}/CS{self.num_charging_stations}"
            )
        self.sample_mode = str(self.sample_mode or "shuffle_cycle").lower()
        if self.sample_mode not in {"shuffle_cycle", "cycle", "random"}:
            raise ValueError("fixed dataset sample_mode must be one of: shuffle_cycle, cycle, random")
        import numpy as np

        self.rng = np.random.default_rng(self.seed)
        self.order = np.arange(len(self.instances), dtype=np.int64)
        if self.sample_mode == "shuffle_cycle":
            self.rng.shuffle(self.order)
        self.cursor = 0
        self.sample_count = 0
        self.region_pool_status = f"fixed_dataset:{path}"
        self._reward_scale_cache: dict[str, float] = {}

    def sample(self) -> EVRPTWInstance:
        if self.sample_mode == "random":
            idx = int(self.rng.integers(0, len(self.instances)))
        else:
            if self.cursor >= len(self.order):
                self.cursor = 0
                if self.sample_mode == "shuffle_cycle":
                    self.rng.shuffle(self.order)
            idx = int(self.order[self.cursor])
            self.cursor += 1
        self.sample_count += 1
        return self.instances[idx]

    def reward_distance_scale_km(self, mode: str = "single_customer_repair_median") -> float:
        """Compute a train-set reward distance scale in km.

        This keeps reward magnitude consistent across the fixed training split.
        Per-instance reward normalization remains available in the env by using
        non-``dataset_`` modes directly.
        """

        import numpy as np

        mode = str(mode)
        if mode in self._reward_scale_cache:
            return self._reward_scale_cache[mode]
        values: list[float] = []
        if mode == "max_edge":
            for instance in self.instances:
                dist = np.asarray(instance.distance_matrix_km, dtype=np.float64)
                finite = dist[np.isfinite(dist)]
                if finite.size:
                    values.append(float(finite.max()))
        elif mode in {
            "single_customer_repair_sum",
            "single_customer_repair_mean",
            "single_customer_repair_median",
        }:
            per_instance: list[float] = []
            all_customer_repairs: list[float] = []
            for instance in self.instances:
                n = int(instance.num_customers)
                dist = np.asarray(instance.distance_matrix_km, dtype=np.float64)
                repairs = dist[0, 1 : n + 1] + dist[1 : n + 1, 0]
                repairs = repairs[np.isfinite(repairs)]
                if not repairs.size:
                    continue
                if mode == "single_customer_repair_sum":
                    per_instance.append(float(repairs.sum()))
                else:
                    all_customer_repairs.extend(float(x) for x in repairs)
            if mode == "single_customer_repair_sum":
                values = per_instance
            else:
                values = all_customer_repairs
        else:
            raise ValueError(f"Unsupported dataset reward scale mode: {mode}")
        if not values:
            scale = 1.0
        elif mode.endswith(("_mean", "_sum")):
            scale = float(np.mean(values))
        elif mode.endswith("_median"):
            scale = float(np.median(values))
        else:
            scale = float(np.max(values))
        scale = max(scale, 1e-9)
        self._reward_scale_cache[mode] = scale
        return scale

    def usage_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "region_id": "fixed_dataset",
                "sampled_days": self.sample_count,
                "customer_exposure_rate": "",
                "recent_mean_jaccard_distance": "",
                "cluster_exposure_entropy": "",
                "region_pool_status": self.region_pool_status,
                "dataset_size": len(self.instances),
                "sample_mode": self.sample_mode,
            }
        ]

    def close(self, terminate: bool = False) -> None:
        del terminate


@dataclass
class OnlineInstancePool:
    """In-memory service-territory pool for online TERRAN training."""

    config_path: str | Path
    num_regions: int = 32
    mother_num_customers: int = 5000
    mother_num_charging_stations: int = 120
    num_customers: int = 15
    num_charging_stations: int = 3
    region_reuse_limit: int = 200
    seed: int | None = None
    max_attempts_per_instance: int | None = None
    territory_pool_path: str | Path | None = None
    region_pool_path: str | Path | None = None
    region_pool_shuffle: bool = True
    region_pool_replacement_policy: str = "cycle"

    def __post_init__(self) -> None:
        try:
            from evrptw_hierarchy.generation.generator import (
                HierarchyDatasetGenerator,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "the retired online evrptw_hierarchy generator is unavailable; "
                "use stage2_dataset_path for canonical TERRAN training"
            ) from exc
        path = Path(self.config_path)
        if not path.is_absolute():
            path = GENERATOR_ROOT / path
        self.config_path = path
        self.generator = HierarchyDatasetGenerator.from_config_path(path, seed=self.seed)
        self.region_pool_status = "generated_online"
        loaded_precomputed = False
        pool_source = self.territory_pool_path if self.territory_pool_path not in (None, "") else self.region_pool_path
        if pool_source not in (None, ""):
            pool_path = Path(pool_source)
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
            except Exception as exc:  # noqa: BLE001 - optional retired legacy path
                self.region_pool_status = f"precomputed_pool_failed:{pool_source}:{exc}"

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


__all__ = [
    "FixedDatasetInstancePool",
    "OnlineInstancePool",
    "Stage2TERRANPool",
]
