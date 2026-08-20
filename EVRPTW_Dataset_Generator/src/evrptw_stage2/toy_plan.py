"""Atomically reduce a full Stage-2 plan to the frozen 75 x 2 toy scope."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from .toy import TOY_SCHEMA, toy_family_ids


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame.groupby(column, sort=True).size().items()
    }


def prune_generation_plan_to_toy(
    plan_root: str | Path,
    manifest: dict[str, Any],
    *,
    manifest_path: str | Path,
) -> dict[str, Any]:
    if manifest.get("schema") != TOY_SCHEMA:
        raise ValueError("Cannot prune a plan with an unsupported toy manifest")
    selected_ids = set(toy_family_ids(manifest))
    root = Path(plan_root)
    registry_path = root / "split_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    family_paths = sorted(root.rglob("family_index.parquet"))
    view_paths = sorted(root.rglob("view_index.parquet"))
    if not family_paths or not view_paths:
        raise ValueError("Generation plan lacks family/view index partitions")

    original_families = pd.concat(
        [pd.read_parquet(path) for path in family_paths], ignore_index=True
    ).drop_duplicates("family_id")
    if len(original_families) not in {150, 7_500}:
        raise ValueError(
            "Toy pruning requires either the original 7,500 plan or an already-pruned "
            f"150-family plan; got {len(original_families)}"
        )
    missing = sorted(selected_ids - set(original_families["family_id"].astype(str)))
    if missing:
        raise ValueError(f"Toy manifest families are absent from the plan: {missing}")

    original_scale_counts = _counts(original_families, "parent_scale_id")
    original_estimates = dict(registry.get("estimated_parent_matrix_bytes_by_scale") or {})
    selected_family_frames: list[pd.DataFrame] = []
    for path in family_paths:
        frame = pd.read_parquet(path)
        frame = frame.loc[frame["family_id"].astype(str).isin(selected_ids)].copy()
        frame["non_release_pilot"] = True
        selected_family_frames.append(frame)
        _atomic_parquet(path, frame)

    selected_view_frames: list[pd.DataFrame] = []
    for path in view_paths:
        frame = pd.read_parquet(path)
        frame = frame.loc[frame["family_id"].astype(str).isin(selected_ids)].copy()
        frame["non_release_pilot"] = True
        selected_view_frames.append(frame)
        _atomic_parquet(path, frame)

    families = pd.concat(selected_family_frames, ignore_index=True).drop_duplicates(
        "family_id"
    )
    views = pd.concat(selected_view_frames, ignore_index=True).drop_duplicates("view_id")
    if set(families["family_id"].astype(str)) != selected_ids or len(families) != 150:
        raise AssertionError("Pruned family plan does not equal the toy manifest")
    if set(views["family_id"].astype(str)) != selected_ids:
        raise AssertionError("Pruned view plan does not cover every toy family")

    source_summary = registry.get("toy_source_full_plan") or {
        "family_count": int(registry.get("family_count", len(original_families))),
        "view_count": int(registry.get("view_count", 0)),
        "family_counts_by_cohort": registry.get("family_counts_by_cohort", {}),
        "view_counts_by_consumer_cohort": registry.get(
            "view_counts_by_consumer_cohort", {}
        ),
        "view_counts_by_scale": registry.get("view_counts_by_scale", {}),
        "estimated_parent_matrix_bytes_by_scale": original_estimates,
        "estimated_parent_matrix_bytes_total": registry.get(
            "estimated_parent_matrix_bytes_total"
        ),
    }
    selected_scale_counts = _counts(families, "parent_scale_id")
    estimate_by_scale = {}
    for scale_id, count in selected_scale_counts.items():
        source_count = int(original_scale_counts.get(scale_id, 0))
        source_bytes = int(original_estimates.get(scale_id, 0))
        estimate_by_scale[scale_id] = (
            int(round(source_bytes / source_count * count)) if source_count else 0
        )
    registry.update(
        {
            "dataset_id": (
                str(registry.get("dataset_id", "CLE_EVRPTW_v2"))
                if str(registry.get("dataset_id", "")).endswith("_toy_75x2")
                else f"{registry.get('dataset_id', 'CLE_EVRPTW_v2')}_toy_75x2"
            ),
            "non_release_pilot": True,
            "official_counts": False,
            "release_eligible": False,
            "family_count": int(len(families)),
            "view_count": int(len(views)),
            "family_counts_by_cohort": _counts(families, "family_cohort_id"),
            "view_counts_by_consumer_cohort": _counts(views, "consumer_cohort_id"),
            "view_counts_by_scale": _counts(views, "scale_id"),
            "estimated_parent_matrix_bytes_by_scale": estimate_by_scale,
            "estimated_parent_matrix_bytes_total": int(sum(estimate_by_scale.values())),
            "toy_source_full_plan": source_summary,
            "toy_contract": {
                "schema": TOY_SCHEMA,
                "benchmark_role": "non_release_full_path_toy",
                "template_count": 2,
                "families_per_template": 75,
                "family_count": 150,
                "manifest": str(Path(manifest_path).resolve()),
                "hash_validation_performed": False,
            },
        }
    )
    registry.pop("joint_spatial_support", None)
    _atomic_json(registry_path, registry)
    return {
        "family_count": int(len(families)),
        "view_count": int(len(views)),
        "family_counts_by_cohort": registry["family_counts_by_cohort"],
        "view_counts_by_consumer_cohort": registry[
            "view_counts_by_consumer_cohort"
        ],
        "view_counts_by_scale": registry["view_counts_by_scale"],
    }
