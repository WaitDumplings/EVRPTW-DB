from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json

import pandas as pd
import pytest

import evrptw_stage2.parallel as parallel
from evrptw_stage2.toy import (
    TRACK_QUOTAS,
    build_full_path_toy_manifest,
    load_full_path_toy_manifest,
    toy_family_ids,
    write_full_path_toy_manifest,
)
from evrptw_stage2.toy_plan import prune_generation_plan_to_toy

ROOT = Path(__file__).parents[1]
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "full_path_toy_stage2_runner", ROOT / "scripts" / "build_stage2_instances.py"
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


def _full_plan() -> pd.DataFrame:
    training_cities = [f"city-{index:02d}" for index in range(10)]
    rows = []
    ordinal = 0

    def extend(track_id: str, city: str, count: int, parent_scale_id: str) -> None:
        nonlocal ordinal
        for index in range(count):
            rows.append(
                {
                    "family_id": f"mf_{ordinal:08d}",
                    "city_slug": city,
                    "track_id": track_id,
                    "day_type": "weekend" if index % 7 in {5, 6} else "weekday",
                    "parent_scale_id": parent_scale_id,
                }
            )
            ordinal += 1

    for city in training_cities:
        extend("train", city, 500, "cus1000")
        extend("validation", city, 50, "cus1000")
        extend("test1_new_seed", city, 50, "cus1000")
        extend("test2_heldout_locations", city, 50, "cus1000")
        extend("unseen_scale_same_cities", city, 50, "cus2000")
    extend("test3_heldout_city", "heldout-city", 500, "cus1000")
    frame = pd.DataFrame(rows)
    assert len(frame) == 7_500
    return frame


def test_full_path_toy_is_deterministic_disjoint_and_covers_frozen_branches() -> None:
    families = _full_plan()
    provenance = {"code_commit": "commit-a"}
    first = build_full_path_toy_manifest(families, code_provenance=provenance)
    second = build_full_path_toy_manifest(families, code_provenance=provenance)
    assert first == second
    assert first["family_count"] == 150
    assert first["template_count"] == 2
    assert first["families_per_template"] == 75
    assert first["covered_city_count"] == 11
    assert set(first["covered_tracks"]) == set(TRACK_QUOTAS)
    assert set(first["covered_day_types"]) == {"weekday", "weekend"}
    ids = toy_family_ids(first)
    assert len(ids) == len(set(ids)) == 150
    for template in first["templates"]:
        assert template["family_count"] == 75
        assert template["track_counts"] == TRACK_QUOTAS
        assert template["day_type_counts"] == {"weekday": 54, "weekend": 21}
        assert len(template["city_counts"]) == 11
        assert set(template["parent_scale_counts"]) == {"cus1000", "cus2000"}


def test_toy_manifest_is_executable_commit_bound(tmp_path: Path) -> None:
    manifest = build_full_path_toy_manifest(
        _full_plan(), code_provenance={"code_commit": "commit-a"}
    )
    path = tmp_path / "toy.json"
    write_full_path_toy_manifest(path, manifest)
    assert load_full_path_toy_manifest(path, code_commit="commit-a") == manifest
    with pytest.raises(ValueError, match="different executable commit"):
        load_full_path_toy_manifest(path, code_commit="commit-b")


def test_toy_plan_pruning_is_non_release_exact_and_resume_safe(tmp_path: Path) -> None:
    plan_root = tmp_path / "generation_plan"
    plan_root.mkdir()
    families = _full_plan().copy()
    families["family_cohort_id"] = families["track_id"]
    families["non_release_pilot"] = False
    views = families[["family_id", "track_id", "parent_scale_id"]].copy()
    views["view_id"] = "view_" + views["family_id"].astype(str)
    views["consumer_cohort_id"] = views["track_id"]
    views["scale_id"] = views["parent_scale_id"]
    views["non_release_pilot"] = False
    families.to_parquet(plan_root / "family_index.parquet", index=False)
    views.to_parquet(plan_root / "view_index.parquet", index=False)
    registry = {
        "dataset_id": "CLE_EVRPTW_v2",
        "family_count": 7_500,
        "view_count": 7_500,
        "family_counts_by_cohort": families.groupby("family_cohort_id").size().to_dict(),
        "view_counts_by_consumer_cohort": views.groupby("consumer_cohort_id").size().to_dict(),
        "view_counts_by_scale": views.groupby("scale_id").size().to_dict(),
        "estimated_parent_matrix_bytes_by_scale": {
            "cus1000": 1_000_000_000,
            "cus2000": 2_000_000_000,
        },
        "estimated_parent_matrix_bytes_total": 3_000_000_000,
        "joint_spatial_support": {"status": "stale"},
    }
    (plan_root / "split_registry.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )
    manifest = build_full_path_toy_manifest(
        families, code_provenance={"code_commit": "commit-a"}
    )
    manifest_path = tmp_path / "toy.json"
    first = prune_generation_plan_to_toy(
        plan_root, manifest, manifest_path=manifest_path
    )
    second = prune_generation_plan_to_toy(
        plan_root, manifest, manifest_path=manifest_path
    )
    pruned_families = pd.read_parquet(plan_root / "family_index.parquet")
    pruned_views = pd.read_parquet(plan_root / "view_index.parquet")
    pruned_registry = json.loads(
        (plan_root / "split_registry.json").read_text(encoding="utf-8")
    )
    assert first == second
    assert len(pruned_families) == len(pruned_views) == 150
    assert pruned_families["non_release_pilot"].all()
    assert pruned_views["non_release_pilot"].all()
    assert set(pruned_families["family_id"]) == set(toy_family_ids(manifest))
    assert pruned_registry["dataset_id"] == "CLE_EVRPTW_v2_toy_75x2"
    assert pruned_registry["family_count"] == pruned_registry["view_count"] == 150
    assert pruned_registry["release_eligible"] is False
    assert pruned_registry["official_counts"] is False
    assert "joint_spatial_support" not in pruned_registry


def test_materialization_worker_receives_frozen_candidate_contract(monkeypatch) -> None:
    observed = {}

    class ExpectedStop(RuntimeError):
        pass

    monkeypatch.setattr(parallel, "load_stage2_config", lambda _path: object())
    monkeypatch.setattr(
        parallel,
        "load_reference_profile",
        lambda _path, *, official: {"official": official},
    )

    def fake_load(_root, _city, *, mode, official_cle_contract):
        observed.update(
            mode=mode,
            official_cle_contract=official_cle_contract,
        )
        raise ExpectedStop

    monkeypatch.setattr(parallel, "load_portable_cle", fake_load)
    with pytest.raises(ExpectedStop):
        parallel.materialize_family_chunk(
            {
                "heartbeat_path": None,
                "families": [{"family": {"family_id": "mf_test"}}],
                "config_path": "config.json",
                "profile_path": "profile.json",
                "mode": "official_toy",
                "official_cle_contract": "frozen_technical_candidate_v1",
                "city_slug": "chicago",
                "cle_root": "cle",
            }
        )
    assert observed == {
        "mode": "official_toy",
        "official_cle_contract": "frozen_technical_candidate_v1",
    }


def test_materialization_worker_reads_capsule_from_canonical_root(
    monkeypatch, tmp_path: Path
) -> None:
    observed = {}

    class ExpectedStop(OSError):
        pass

    monkeypatch.setattr(parallel, "load_stage2_config", lambda _path: object())
    monkeypatch.setattr(
        parallel,
        "load_reference_profile",
        lambda _path, *, official: {"official": official},
    )
    monkeypatch.setattr(
        parallel, "load_portable_cle", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        parallel, "load_amazon_stage2_artifacts", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        parallel,
        "materialization_attempt_inputs",
        lambda family, views, *, attempt_number: (family, views),
    )

    def fake_materialize(_cle, **kwargs):
        observed.update(
            output_root=Path(kwargs["output_root"]),
            selection_capsule_root=Path(kwargs["selection_capsule_root"]),
        )
        raise ExpectedStop

    monkeypatch.setattr(parallel, "materialize_family", fake_materialize)
    canonical = tmp_path / "instance-root"
    inflight = canonical / ".inflight" / "mf_test" / "attempt-000" / "materialized"
    with pytest.raises(ExpectedStop):
        parallel.materialize_family_chunk(
            {
                "heartbeat_path": None,
                "families": [
                    {"family": {"family_id": "mf_test"}, "views": []}
                ],
                "chunk_id": "chicago:00000",
                "config_path": "config.json",
                "profile_path": "profile.json",
                "mode": "official_toy",
                "official_cle_contract": "frozen_technical_candidate_v1",
                "city_slug": "chicago",
                "cle_root": "cle",
                "output_root": str(canonical),
                "materialized_output_root": str(inflight),
                "customer_split_path": "split.parquet",
                "community_adjacency_path": "adjacency.parquet",
                "amazon_artifact_root": "amazon",
                "amazon_cohort_split_path": "cohort.json",
                "max_attempts_per_family": 1,
            }
        )
    assert observed == {
        "output_root": inflight,
        "selection_capsule_root": canonical,
    }


def test_builder_places_candidate_contract_in_every_worker_envelope(tmp_path: Path) -> None:
    families = pd.DataFrame(
        [{"family_id": "mf_test", "city_slug": "chicago"}]
    )
    views = pd.DataFrame(
        [{"family_id": "mf_test", "view_id": "view_test"}]
    )
    args = SimpleNamespace(
        families_per_worker_task=1,
        config=tmp_path / "config.json",
        profile=tmp_path / "profile.json",
        cle_root=tmp_path / "cle",
        mode="official_toy",
        official_cle_contract="frozen_technical_candidate_v1",
        output_root=tmp_path / "output",
        amazon_artifact_root=tmp_path / "amazon",
        amazon_cohort_split_path=tmp_path / "cohort.json",
        max_attempts_per_family=4,
        code_provenance={"code_commit": "commit-a"},
    )
    tasks = RUNNER._build_materialization_tasks(families, views, args=args)
    assert len(tasks) == 1
    assert tasks[0]["mode"] == "official_toy"
    assert (
        tasks[0]["official_cle_contract"]
        == "frozen_technical_candidate_v1"
    )
