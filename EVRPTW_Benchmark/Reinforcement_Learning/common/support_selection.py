from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_DESCRIPTOR_COLUMNS = (
    "m1_radial.proposal_distribution.mean",
    "m1_radial.proposal_distribution.p50",
    "m1_radial.proposal_distribution.p90",
    "m2_network_nearest_neighbor.generated.mean",
    "m2_network_nearest_neighbor.generated.p50",
    "m2_network_nearest_neighbor.generated.p90",
    "m3_within_region_pairwise_time.generated_region_p50_distribution.mean",
    "m3_within_region_pairwise_time.generated_region_p50_distribution.p90",
    "m3_within_region_pairwise_time.generated_region_p90_distribution.mean",
    "m3_within_region_pairwise_time.generated_region_p90_distribution.p90",
    "m4_region_structure.region_count",
    "m4_region_structure.region_size_distribution.mean",
    "m4_region_structure.region_size_distribution.p50",
    "m4_region_structure.region_size_distribution.p90",
    "m5_community_concentration.proposal.active_community_count",
    "m5_community_concentration.proposal.community_hhi",
    "m5_community_concentration.proposal.largest_share",
)


@dataclass(frozen=True)
class SupportSelection:
    full_family_ids: tuple[str, ...]
    random_family_ids: tuple[str, ...]
    coverage_family_ids: tuple[str, ...]
    manifest: dict


def _largest_remainder(counts: pd.Series, total: int) -> dict[tuple[str, str], int]:
    exact = counts.astype(float) * (int(total) / float(counts.sum()))
    base = np.floor(exact).astype(int)
    left = int(total) - int(base.sum())
    order = sorted(counts.index, key=lambda key: (-(exact[key] - base[key]), str(key)))
    answer = {key: int(base[key]) for key in counts.index}
    for key in order[:left]:
        answer[key] += 1
    return answer


def _farthest_first(matrix: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if count >= len(matrix):
        return np.arange(len(matrix), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    jitter = rng.uniform(0.0, 1.0e-12, size=len(matrix))
    centroid = matrix.mean(axis=0)
    centroid_d2 = ((matrix - centroid) ** 2).sum(axis=1)
    selected = [int(np.argmax(centroid_d2 + jitter))]
    min_d2 = ((matrix - matrix[selected[0]]) ** 2).sum(axis=1)
    min_d2[selected[0]] = -1.0
    while len(selected) < count:
        choice = int(np.argmax(min_d2 + jitter))
        selected.append(choice)
        min_d2 = np.minimum(min_d2, ((matrix - matrix[choice]) ** 2).sum(axis=1))
        min_d2[np.asarray(selected, dtype=np.int64)] = -1.0
    return np.asarray(selected, dtype=np.int64)


def _coverage_radius(matrix: np.ndarray, selected: np.ndarray) -> tuple[float, float]:
    nearest = np.full(len(matrix), np.inf, dtype=np.float64)
    for index in selected:
        nearest = np.minimum(nearest, np.sqrt(((matrix - matrix[index]) ** 2).sum(axis=1)))
    return float(nearest.mean()), float(np.quantile(nearest, 0.95))


def select_parent_family_supports(
    train_index: pd.DataFrame,
    family_metrics: pd.DataFrame,
    *,
    fraction: float = 0.10,
    seed: int = 73129,
    descriptor_columns: Iterable[str] = DEFAULT_DESCRIPTOR_COLUMNS,
) -> SupportSelection:
    required = {"family_id", "city_slug", "day_type", "split_id", "track_id"}
    missing = sorted(required.difference(train_index.columns))
    if missing:
        raise ValueError(f"training index is missing columns: {missing}")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("support fraction must be in (0, 1]")
    train = train_index.loc[
        (train_index["split_id"].astype(str) == "train")
        & (train_index["track_id"].astype(str) == "train")
    ].copy()
    if train.empty:
        raise ValueError("training support is empty")
    family_meta_rows = []
    for family_id, group in train.groupby("family_id", sort=True):
        cities = sorted(set(group["city_slug"].astype(str)))
        days = sorted(set(group["day_type"].astype(str)))
        if len(cities) != 1 or not days:
            raise ValueError(f"parent family {family_id} has invalid city/day metadata")
        # A Stage-2 parent family intentionally contains its frozen 5:2
        # weekday/weekend views. It remains one statistical/support unit.
        family_day_group = days[0] if len(days) == 1 else "mixed_5_2"
        family_meta_rows.append((str(family_id), cities[0], family_day_group))
    family_meta = pd.DataFrame(family_meta_rows, columns=["family_id", "city_slug", "day_type"])
    if "family_id" not in family_metrics.columns:
        raise ValueError("family metrics must contain family_id")
    metrics = family_metrics.copy()
    metrics["family_id"] = metrics["family_id"].astype(str)
    if metrics["family_id"].duplicated().any():
        raise ValueError("family metrics contain duplicate family_id rows")
    columns = [name for name in descriptor_columns if name in metrics.columns]
    if len(columns) < 5:
        raise ValueError(
            "fewer than five registered pre-solver descriptors are available; "
            f"found {columns}"
        )
    frame = family_meta.merge(
        metrics[["family_id", *columns]], on="family_id", how="left", validate="one_to_one"
    )
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad = frame.loc[~np.isfinite(values).all(axis=1), "family_id"].tolist()
        raise ValueError(f"non-finite registered descriptors for parent families: {bad[:5]}")
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0.0] = 1.0
    standardized = (values - mean) / std
    strata_labels = frame["city_slug"].astype(str) + "\x1f" + frame["day_type"].astype(str)
    one_hot = pd.get_dummies(strata_labels, dtype=float).to_numpy(dtype=np.float64)
    feature_matrix = np.concatenate([standardized, one_hot], axis=1)
    target = max(1, int(round(len(frame) * float(fraction))))
    quotas = _largest_remainder(
        frame.groupby(["city_slug", "day_type"], sort=True).size(), target
    )
    random_selected: list[int] = []
    coverage_selected: list[int] = []
    cities = frame["city_slug"].astype(str).to_numpy()
    days = frame["day_type"].astype(str).to_numpy()
    for stratum_offset, (key, quota) in enumerate(quotas.items()):
        positions = np.flatnonzero((cities == str(key[0])) & (days == str(key[1])))
        if quota > len(positions):
            raise ValueError(f"support quota exceeds stratum size for {key}")
        if quota == 0:
            continue
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, stratum_offset, 0x52414E44])
        )
        random_selected.extend(
            positions[rng.choice(len(positions), size=quota, replace=False)].tolist()
        )
        local = _farthest_first(
            feature_matrix[positions], quota, seed + stratum_offset * 1009
        )
        coverage_selected.extend(positions[local].tolist())
    random_idx = np.asarray(sorted(random_selected), dtype=np.int64)
    coverage_idx = np.asarray(sorted(coverage_selected), dtype=np.int64)
    random_radius = _coverage_radius(feature_matrix, random_idx)
    coverage_radius = _coverage_radius(feature_matrix, coverage_idx)
    manifest = {
        "schema": "drl_parent_family_support_selection_v1",
        "selection_unit": "parent_family",
        "fraction": float(fraction),
        "seed": int(seed),
        "full_parent_family_count": len(frame),
        "selected_parent_family_count": target,
        "descriptor_columns": columns,
        "standardization_scope": "full_training_support_only",
        "selection_uses_validation_or_test": False,
        "algorithm": "city_day_stratified_random_and_farthest_first_v1",
        "random_mean_nearest_support_distance": random_radius[0],
        "random_p95_nearest_support_distance": random_radius[1],
        "coverage_mean_nearest_support_distance": coverage_radius[0],
        "coverage_p95_nearest_support_distance": coverage_radius[1],
        "file_hash_validation_performed": False,
    }
    return SupportSelection(
        full_family_ids=tuple(frame["family_id"].astype(str)),
        random_family_ids=tuple(frame.iloc[random_idx]["family_id"].astype(str)),
        coverage_family_ids=tuple(frame.iloc[coverage_idx]["family_id"].astype(str)),
        manifest=manifest,
    )


__all__ = ["DEFAULT_DESCRIPTOR_COLUMNS", "SupportSelection", "select_parent_family_supports"]
