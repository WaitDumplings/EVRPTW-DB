from __future__ import annotations

import sys
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Core"))
sys.path.insert(0, str(REPO_ROOT / "EVRPTW_Dataset_Generator" / "src"))

from evrptw_core.schema import EVRPTWInstance

from EVRPTW_Benchmark.MetaHeuristics.benchmark_common import (
    Stage2ViewTask,
    load_stage2_instance,
    normalize_scale,
    read_stage2_tasks,
)
from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_Env import (
    EVRPTWVectorEnvFast,
)
from .data_pass import pass_batches
from .euclidean import euclidean_instance, load_euclidean_manifest
from .training_stream import read_stream_view_ids


def _csv_set(value: str | None) -> set[str] | None:
    if value is None or not value.strip():
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


@dataclass
class Stage2TaskPool:
    """Lazy, fixed-terminal-count pool of materialized Stage-2 views."""

    dataset_path: str | Path
    family_root: str | Path | None = None
    scale: str | int | None = None
    split_ids: str | None = None
    track_ids: str | None = None
    city_slugs: str | None = None
    seed: int = 1234
    cache_size: int = 4
    representation: str = "G"
    euclidean_manifest: str | Path | None = None

    def __post_init__(self) -> None:
        tasks = read_stage2_tasks(self.dataset_path, family_root=self.family_root)
        expected_scale = None if self.scale is None else normalize_scale(self.scale)
        split_ids = _csv_set(self.split_ids)
        track_ids = _csv_set(self.track_ids)
        city_slugs = _csv_set(self.city_slugs)
        self.tasks = [
            task
            for task in tasks
            if (expected_scale is None or task.scale_label == expected_scale)
            and (split_ids is None or task.split_id in split_ids)
            and (track_ids is None or task.track_id in track_ids)
            and (city_slugs is None or task.city_slug in city_slugs)
        ]
        if not self.tasks:
            raise ValueError("no Stage-2 views match the requested learning-data filters")
        self.representation = str(self.representation).upper()
        if self.representation not in {"E", "G"}:
            raise ValueError("representation must be E or G")
        if self.representation == "E" and self.euclidean_manifest is None:
            raise ValueError("E representation requires a Euclidean calibration manifest")
        self._euclidean = (
            load_euclidean_manifest(self.euclidean_manifest)
            if self.representation == "E"
            else None
        )
        terminal_counts = {task.terminal_count for task in self.tasks}
        if len(terminal_counts) != 1:
            raise ValueError(
                "one learning batch pool must have a fixed customer/charger terminal count"
            )
        self.rng = np.random.default_rng(int(self.seed))
        self._cache: OrderedDict[str, EVRPTWInstance] = OrderedDict()
        self._task_by_view_id = {task.view_id: task for task in self.tasks}

    def __len__(self) -> int:
        return len(self.tasks)

    def instance(self, task: Stage2ViewTask) -> EVRPTWInstance:
        cached = self._cache.pop(task.view_id, None)
        if cached is None:
            cached = load_stage2_instance(task)
            if self._euclidean is not None:
                cached = euclidean_instance(cached, self._euclidean)
        self._cache[task.view_id] = cached
        while len(self._cache) > max(0, int(self.cache_size)):
            self._cache.popitem(last=False)
        return cached

    def sample(self, batch_size: int) -> list[EVRPTWInstance]:
        indices = self.rng.integers(0, len(self.tasks), size=int(batch_size))
        return [self.instance(self.tasks[int(index)]) for index in indices]

    def first(self, limit: int | None = None) -> Iterable[EVRPTWInstance]:
        tasks = self.tasks if limit is None else self.tasks[: int(limit)]
        for task in tasks:
            yield self.instance(task)

    def data_pass_batches(
        self,
        data_pass: int,
        physical_batch_size: int,
    ) -> Iterable[list[EVRPTWInstance]]:
        """Yield every filtered view exactly once in a deterministic shuffle."""

        for tasks in pass_batches(
            self.tasks,
            seed=self.seed,
            data_pass=data_pass,
            physical_batch_size=physical_batch_size,
        ):
            yield [self.instance(task) for task in tasks]

    def stream_batches(
        self,
        stream_path: str | Path,
        physical_batch_size: int,
        *,
        start: int = 0,
        stop: int | None = None,
    ) -> Iterable[list[EVRPTWInstance]]:
        """Consume an explicit method-independent training ID stream."""

        if int(physical_batch_size) <= 0:
            raise ValueError("physical_batch_size must be positive")
        view_ids = read_stream_view_ids(stream_path, start=start, stop=stop)
        missing = sorted(set(view_ids).difference(self._task_by_view_id))
        if missing:
            raise ValueError(
                f"training stream contains {len(missing)} IDs outside the filtered pool: "
                f"{missing[:3]}"
            )
        for offset in range(0, len(view_ids), int(physical_batch_size)):
            yield [
                self.instance(self._task_by_view_id[view_id])
                for view_id in view_ids[offset : offset + int(physical_batch_size)]
            ]


def make_envs(
    instances: Iterable[EVRPTWInstance],
    *,
    n_traj: int,
    info_level: str,
    use_jit_mask: bool = True,
) -> list[EVRPTWVectorEnvFast]:
    """Create environments under the canonical Stage-2 physical contract."""

    return [
        EVRPTWVectorEnvFast(
            instance=instance,
            n_traj=int(n_traj),
            reward_mode="distance",
            charging_mode="station_power_full",
            matrix_mode="canonical",
            info_level=info_level,
            use_jit_mask=use_jit_mask,
        )
        for instance in instances
    ]


__all__ = ["Stage2TaskPool", "make_envs"]
