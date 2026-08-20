"""Deterministic 75 x 2 full-path toy cohort selection."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


TOY_SCHEMA = "evrptw_stage2_full_path_toy_manifest_v1"
TOY_TEMPLATE_IDS = ("toy-a", "toy-b")
TOY_FAMILIES_PER_TEMPLATE = 75
TRACK_QUOTAS = {
    "train": 25,
    "validation": 10,
    "test1_new_seed": 10,
    "test2_heldout_locations": 10,
    "test3_heldout_city": 10,
    "unseen_scale_same_cities": 10,
}
WEEKEND_QUOTAS = {
    "train": 7,
    "validation": 3,
    "test1_new_seed": 3,
    "test2_heldout_locations": 3,
    "test3_heldout_city": 2,
    "unseen_scale_same_cities": 3,
}


def _round_robin_by_city(
    rows: pd.DataFrame,
    *,
    count: int,
    template_id: str,
) -> list[dict[str, Any]]:
    if count == 0:
        return []
    buckets: dict[str, list[dict[str, Any]]] = {}
    for city, city_rows in rows.groupby("city_slug", sort=True):
        records = city_rows.to_dict("records")
        records.sort(key=lambda row: str(row["family_id"]))
        buckets[str(city)] = records
    cities = sorted(buckets)
    if not cities:
        raise ValueError("Toy stratum has no candidate cities")
    offset = (1 if template_id.startswith("toy-b") else 0) % len(cities)
    cities = cities[offset:] + cities[:offset]
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < count:
        city = cities[cursor % len(cities)]
        if buckets[city]:
            selected.append(buckets[city].pop(0))
        elif not any(buckets.values()):
            raise ValueError(f"Toy stratum has fewer than {count} candidates")
        cursor += 1
    return selected


def build_full_path_toy_manifest(
    families: pd.DataFrame,
    *,
    code_provenance: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "family_id",
        "city_slug",
        "track_id",
        "day_type",
        "parent_scale_id",
    }
    missing = sorted(required - set(families.columns))
    if missing:
        raise ValueError(f"Generation plan lacks toy-selection columns: {missing}")
    families = families.drop_duplicates("family_id").copy()
    if len(families) != 7_500:
        raise ValueError(
            "Full-path toy selection requires the 7,500-family plan; "
            f"got {len(families)}"
        )
    if set(families["track_id"].astype(str)) != set(TRACK_QUOTAS):
        raise ValueError("Full plan track roster differs from the frozen toy contract")

    used: set[str] = set()
    templates: list[dict[str, Any]] = []
    for template_id in TOY_TEMPLATE_IDS:
        selected: list[dict[str, Any]] = []
        available = families.loc[~families["family_id"].astype(str).isin(used)]
        for track_id, track_count in TRACK_QUOTAS.items():
            track_rows = available.loc[available["track_id"].astype(str).eq(track_id)]
            weekend_count = WEEKEND_QUOTAS[track_id]
            weekday_count = track_count - weekend_count
            for day_type, count in (("weekday", weekday_count), ("weekend", weekend_count)):
                day_rows = track_rows.loc[track_rows["day_type"].astype(str).eq(day_type)]
                chosen = _round_robin_by_city(
                    day_rows,
                    count=count,
                    template_id=f"{template_id}|{track_id}|{day_type}",
                )
                selected.extend(chosen)
                chosen_ids = {str(row["family_id"]) for row in chosen}
                available = available.loc[
                    ~available["family_id"].astype(str).isin(chosen_ids)
                ]
        if len(selected) != TOY_FAMILIES_PER_TEMPLATE:
            raise AssertionError("Toy template size drifted from 75")
        selected_ids = {str(row["family_id"]) for row in selected}
        if selected_ids & used:
            raise AssertionError("Toy templates are not family-disjoint")
        used.update(selected_ids)
        selected.sort(key=lambda row: str(row["family_id"]))
        templates.append(
            {
                "template_id": template_id,
                "family_count": len(selected),
                "track_counts": dict(
                    sorted(Counter(str(row["track_id"]) for row in selected).items())
                ),
                "day_type_counts": dict(
                    sorted(Counter(str(row["day_type"]) for row in selected).items())
                ),
                "city_counts": dict(
                    sorted(Counter(str(row["city_slug"]) for row in selected).items())
                ),
                "parent_scale_counts": dict(
                    sorted(Counter(str(row["parent_scale_id"]) for row in selected).items())
                ),
                "families": [
                    {
                        "family_id": str(row["family_id"]),
                        "city_slug": str(row["city_slug"]),
                        "track_id": str(row["track_id"]),
                        "day_type": str(row["day_type"]),
                        "parent_scale_id": str(row["parent_scale_id"]),
                    }
                    for row in selected
                ],
            }
        )

    selected_rows = [row for template in templates for row in template["families"]]
    city_set = {str(row["city_slug"]) for row in selected_rows}
    track_set = {str(row["track_id"]) for row in selected_rows}
    day_set = {str(row["day_type"]) for row in selected_rows}
    if len(city_set) != 11 or track_set != set(TRACK_QUOTAS) or day_set != {"weekday", "weekend"}:
        raise AssertionError("Toy manifest does not cover all frozen full-path dimensions")
    return {
        "schema": TOY_SCHEMA,
        "status": "frozen",
        "release_eligible": False,
        "benchmark_role": "non_release_full_path_toy",
        "benchmark_positioning": "infrastructure_grounded_semi_synthetic_not_fully_real",
        "benchmark_description": "infrastructure-grounded semi-synthetic",
        "hash_validation_performed": False,
        "template_count": len(templates),
        "families_per_template": TOY_FAMILIES_PER_TEMPLATE,
        "family_count": len(selected_rows),
        "templates_pairwise_family_disjoint": True,
        "covered_city_count": len(city_set),
        "covered_tracks": sorted(track_set),
        "covered_day_types": sorted(day_set),
        "code_provenance": code_provenance,
        "templates": templates,
    }


def toy_family_ids(manifest: dict[str, Any]) -> tuple[str, ...]:
    if manifest.get("schema") != TOY_SCHEMA:
        raise ValueError(f"Unsupported toy manifest schema: {manifest.get('schema')!r}")
    templates = manifest.get("templates", [])
    if [item.get("template_id") for item in templates] != list(TOY_TEMPLATE_IDS):
        raise ValueError("Toy manifest template roster is not frozen toy-a/toy-b")
    ids = tuple(
        str(row["family_id"])
        for template in templates
        for row in template.get("families", [])
    )
    if len(ids) != 150 or len(set(ids)) != 150:
        raise ValueError("Toy manifest must contain 150 unique family IDs")
    if any(int(template.get("family_count", -1)) != 75 for template in templates):
        raise ValueError("Each toy template must contain exactly 75 families")
    return ids


def load_full_path_toy_manifest(
    path: str | Path,
    *,
    code_commit: str | None = None,
) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    toy_family_ids(manifest)
    if (
        code_commit is not None
        and manifest.get("code_provenance", {}).get("code_commit") != code_commit
    ):
        raise ValueError("Toy manifest belongs to a different executable commit")
    return manifest


def write_full_path_toy_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)
