from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from evrptw_hierarchy.configs.config import load_yaml, vehicle_from_config
from evrptw_hierarchy.core.models import RegionBoard, RegionUsage, VehicleConfig
from evrptw_hierarchy.generation.region_generator import RegionGenerator
from evrptw_hierarchy.graph.distance_oracle import DistanceOracle
from evrptw_hierarchy.io.persistence import ensure_dir, save_pickle
from evrptw_hierarchy.sampling.active_day import ActiveDaySampler
from evrptw_hierarchy.validation.reports import summarize_instance, summarize_region, write_reports
from evrptw_hierarchy.visualization.plots import write_region_svg


@dataclass
class HierarchyDatasetGenerator:
    config: dict[str, Any]
    seed: int | None = None
    vehicle: VehicleConfig = field(init=False)
    rng: np.random.Generator = field(init=False)
    boards: list[RegionBoard] = field(default_factory=list, init=False)
    usages: list[RegionUsage] = field(default_factory=list, init=False)
    oracles: dict[str, DistanceOracle] = field(default_factory=dict, init=False)
    next_region_serial: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        cfg_seed = self.config.get("seed")
        actual_seed = self.seed if self.seed is not None else cfg_seed
        self.rng = np.random.default_rng(None if actual_seed in (None, "") else int(actual_seed))
        self.vehicle = vehicle_from_config(self.config)

    @classmethod
    def from_config_path(cls, config_path: str | Path, seed: int | None = None) -> "HierarchyDatasetGenerator":
        return cls(load_yaml(config_path), seed=seed)

    def _create_region(self, slot_index: int, mother_num_customers: int, mother_num_charging_stations: int) -> tuple[RegionBoard, RegionUsage]:
        generator = RegionGenerator(self.config, self.vehicle, self.rng)
        serial = self.next_region_serial
        self.next_region_serial += 1
        board = generator.generate(serial, mother_num_customers, mother_num_charging_stations)
        usage = RegionUsage(
            region_id=board.region_id,
            sampled_days=0,
            customer_activation_counts=np.zeros(len(board.customers), dtype=np.int32),
            cluster_activation_counts=np.zeros(len(board.cluster_centers), dtype=np.int32),
        )
        if slot_index < len(self.boards):
            old_id = self.boards[slot_index].region_id
            self.oracles.pop(old_id, None)
            self.boards[slot_index] = board
            self.usages[slot_index] = usage
        else:
            self.boards.append(board)
            self.usages.append(usage)
        return board, usage

    def _oracle_for(self, board: RegionBoard) -> DistanceOracle:
        if board.region_id not in self.oracles:
            sp_cfg = self.config.get("shortest_path", {})
            terminal_node_ids = np.concatenate([
                np.asarray([board.depot_node_id], dtype=np.int32),
                board.customer_node_ids.astype(np.int32),
                board.cs_node_ids.astype(np.int32),
            ])
            mode = str(sp_cfg.get("oracle_mode", "auto"))
            estimated_mb = terminal_node_ids.size * terminal_node_ids.size * 4.0 / (1024.0 * 1024.0)
            use_terminal = mode == "terminal_matrix" or (
                mode == "auto"
                and terminal_node_ids.size <= int(sp_cfg.get("terminal_matrix_max_terminals", 8000))
                and estimated_mb <= float(sp_cfg.get("terminal_matrix_max_mb", 512.0))
            )
            self.oracles[board.region_id] = DistanceOracle(
                len(board.road_nodes),
                board.road_edges,
                board.road_edge_lengths_km,
                terminal_node_ids=terminal_node_ids,
                use_terminal_matrix=use_terminal,
            )
        return self.oracles[board.region_id]

    def _is_stale(self, usage: RegionUsage, region_reuse_limit: int) -> bool:
        fresh = self.config.get("freshness", {})
        if usage.sampled_days >= int(region_reuse_limit):
            return True
        if bool(fresh.get("use_exposure_rule", True)):
            if usage.customer_exposure_rate >= float(fresh.get("customer_exposure_rate_threshold", 0.85)):
                return True
        if bool(fresh.get("use_jaccard_rule", True)):
            min_recent = int(fresh.get("min_recent_days", 10))
            if len(usage.recent_active_customer_sets) >= min_recent:
                if usage.recent_mean_jaccard_distance <= float(fresh.get("recent_jaccard_diversity_threshold", 0.65)):
                    return True
        return False

    def _select_region_slot(self, region_reuse_limit: int) -> int:
        eligible = [idx for idx, usage in enumerate(self.usages) if not self._is_stale(usage, region_reuse_limit)]
        if not eligible:
            return int(np.argmin([usage.sampled_days for usage in self.usages]))
        return min(eligible, key=lambda idx: self.usages[idx].sampled_days)

    def _save_regions(self, save_path: Path) -> None:
        region_dir = ensure_dir(save_path / "regions")
        for board in self.boards:
            save_pickle(region_dir / f"{board.region_id}_board.pkl", board)

    def generate(
        self,
        save_path: str | Path,
        num_instances: int,
        num_customers: int,
        num_charging_stations: int,
        num_regions: int,
        mother_num_customers: int,
        mother_num_charging_stations: int,
        region_reuse_limit: int,
        max_attempts_per_instance: int | None = None,
        save_plots: bool = True,
    ) -> dict[str, Any]:
        root = Path(save_path)
        ensure_dir(root)
        instance_dir = ensure_dir(root / "instances" / f"Cus_{int(num_customers)}_CS_{int(num_charging_stations)}")
        plots_dir = ensure_dir(root / "analysis_outputs" / "plots")

        while len(self.boards) < int(num_regions):
            slot = len(self.boards)
            self._create_region(slot, int(mother_num_customers), int(mother_num_charging_stations))
        self._save_regions(root)

        sampler = ActiveDaySampler(self.config, self.vehicle, self.rng)
        max_attempts = int(max_attempts_per_instance or self.config.get("generation", {}).get("max_attempts_per_instance", 30))
        failed_attempt_rows: list[dict[str, Any]] = []
        instance_rows: list[dict[str, Any]] = []
        plot_limit = int(self.config.get("visualization", {}).get("max_instance_plots", 10))

        for instance_index in range(int(num_instances)):
            outer_attempts = int(self.config.get("generation", {}).get("max_region_attempts_per_instance", 5))
            last_error = "not_started"
            instance = None
            for outer_attempt in range(outer_attempts):
                slot = self._select_region_slot(int(region_reuse_limit))
                if self._is_stale(self.usages[slot], int(region_reuse_limit)):
                    self._create_region(slot, int(mother_num_customers), int(mother_num_charging_stations))
                    self._save_regions(root)
                board = self.boards[slot]
                usage = self.usages[slot]
                oracle = self._oracle_for(board)
                try:
                    instance = sampler.build_instance(
                        board=board,
                        usage_index=usage.sampled_days + 1,
                        instance_index=instance_index,
                        num_customers=int(num_customers),
                        num_charging_stations=int(num_charging_stations),
                        max_attempts=max_attempts,
                        oracle=oracle,
                    )
                    usage.record_day(instance.active_customer_ids, board.cluster_labels, int(self.config.get("freshness", {}).get("recent_window", 30)))
                    break
                except Exception as exc:  # Keep generation robust; failed candidates are reported, not saved.
                    last_error = str(exc)
                    failed_attempt_rows.append({
                        "instance_index": instance_index,
                        "outer_attempt": outer_attempt + 1,
                        "region_id": board.region_id,
                        "error": last_error,
                    })
                    self._create_region(slot, int(mother_num_customers), int(mother_num_charging_stations))
                    self._save_regions(root)
            if instance is None:
                raise RuntimeError(f"Failed to generate instance_{instance_index:06d}: {last_error}")

            save_pickle(instance_dir / f"{instance.instance_id}.pkl", instance)
            row = summarize_instance(instance)
            instance_rows.append(row)
            if save_plots and instance_index < plot_limit:
                write_region_svg(self.boards[slot], plots_dir / f"{instance.instance_id}_active_day.svg", instance=instance)

        self._save_regions(root)
        region_rows = [summarize_region(board, usage) for board, usage in zip(self.boards, self.usages)]
        write_reports(root, region_rows, instance_rows, failed_attempt_rows)
        if save_plots:
            for idx, board in enumerate(self.boards[:plot_limit]):
                write_region_svg(board, plots_dir / f"{board.region_id}_mother_board.svg")

        return {
            "save_path": str(root),
            "instances_dir": str(instance_dir),
            "num_instances": int(num_instances),
            "num_regions_in_pool": int(len(self.boards)),
            "generated_instance_rows": instance_rows,
            "region_rows": region_rows,
            "failed_attempt_rows": failed_attempt_rows,
        }
