from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "compare_stage2_c0_plans.py"
SPEC = importlib.util.spec_from_file_location("compare_c0", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "stage2_runner", ROOT / "scripts" / "build_stage2_instances.py"
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)
C3_SPEC = importlib.util.spec_from_file_location(
    "stage2_c3", ROOT / "scripts" / "apply_stage2_joint_support_gate.py"
)
assert C3_SPEC is not None and C3_SPEC.loader is not None
C3 = importlib.util.module_from_spec(C3_SPEC)
C3_SPEC.loader.exec_module(C3)


def test_supervisor_plan_envelope_normalizes_arrow_numpy_values() -> None:
    value = {
        "required_decile_counts": np.asarray([1, 2, 3], dtype=np.int64),
        "candidate_depot_count": np.int64(7),
        "optional": np.nan,
    }
    normalized = RUNNER._json_safe_plan_value(value)
    assert normalized == {
        "required_decile_counts": [1, 2, 3],
        "candidate_depot_count": 7,
        "optional": None,
    }
    json.dumps(normalized)


def test_c3_c2_binding_requires_exact_c0_evidence_for_inheritance(
    tmp_path: Path,
) -> None:
    c2_path = tmp_path / "c2.json"
    c2_path.write_text(
        json.dumps(
            {
                "schema": "cle_evrptw_phase_c2_release_preflight_v1",
                "passed": True,
                "code_provenance": {"code_commit": "a" * 40},
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        c2_report=c2_path,
        c0_comparison=None,
        plan_root=tmp_path / "generation_plan",
    )
    with pytest.raises(ValueError, match="--c0-comparison"):
        C3._validate_c2_evidence(
            args,
            {"code_commit": "b" * 40},
        )
    binding = C3._validate_c2_evidence(
        args,
        {"code_commit": "a" * 40},
    )
    assert binding["mode"] == "same_commit"


def test_verifier_exception_writes_terminal_failed_state(tmp_path: Path) -> None:
    report = {
        "status": "verifying",
        "passed": None,
        "last_completed_stage": "materialization",
        "planned_family_ids": ["f2", "f1"],
        "materialized": [{"family_id": "f1"}, {"family_id": "f2"}],
        "verified": [{"family_id": "f1", "passed": True}],
        "unresolved_family_ids": ["f2"],
    }
    error = RuntimeError("synthetic verifier exception")
    RUNNER._mark_run_report_failed(
        report,
        error,
        last_completed_stage="verification",
    )
    path = tmp_path / "stage2_run_report.json"
    RUNNER._write_json(path, report)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["passed"] is False
    assert persisted["planned_family_ids"] == ["f1", "f2"]
    assert persisted["materialized_family_ids"] == ["f1", "f2"]
    assert persisted["verified_family_ids"] == ["f1"]
    assert persisted["unresolved_family_ids"] == ["f2"]
    assert persisted["exception"] == {
        "type": "RuntimeError",
        "message": "synthetic verifier exception",
    }
    assert persisted["last_completed_stage"] == "verification"
    assert not list(tmp_path.glob(".stage2_run_report.json.*.tmp"))


def test_c3_plan_update_persists_required_contract_fields_atomically(
    tmp_path: Path,
) -> None:
    plan_root = tmp_path / "generation_plan"
    family_path = plan_root / "core" / "train" / "family_index.parquet"
    family_path.parent.mkdir(parents=True)
    original = pd.DataFrame(
        {
            "family_id": ["f1", "f2"],
            "family_cohort_id": ["core/train", "core/train"],
        }
    )
    original.to_parquet(family_path, index=False)
    registry_path = plan_root / "split_registry.json"
    registry_path.write_text(
        json.dumps({"schema": "cle_evrptw_generation_plan_v3"}),
        encoding="utf-8",
    )
    updates = {
        "f1": {
            "joint_support_contract_id": "c3_joint_spatial_support_v1",
            "candidate_depot_count": 3,
            "candidate_structure_source_count": 4,
            "joint_pair_count": 12,
            "aggregate_gate_pass_count": 2,
            "exact_gate_pass_count": 1,
            "selected_depot_id": "d1",
            "selected_structure_source_id": "s1",
            "required_decile_counts": [1] * 10,
            "available_decile_counts": [2] * 10,
            "capacity_contract_fingerprint": "ccf-1",
            "rejected_pair_reason_counts": '{"SPATIAL_QUOTA_UNSUPPORTED":1}',
        }
    }
    report_path = tmp_path / "c3.json"
    C3._apply_updates(
        plan_root,
        [(family_path, original)],
        updates,
        c3_report=report_path,
        code_provenance={"code_commit": "a" * 40},
        full_plan=False,
    )
    persisted = pd.read_parquet(family_path)
    row = persisted.loc[persisted["family_id"].eq("f1")].iloc[0]
    assert row["selected_depot_id"] == "d1"
    assert list(row["required_decile_counts"]) == [1] * 10
    assert row["capacity_contract_fingerprint"] == "ccf-1"
    untouched = persisted.loc[persisted["family_id"].eq("f2")].iloc[0]
    assert pd.isna(untouched["selected_depot_id"])
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["joint_spatial_support"]["status"] == (
        "passed_targeted_gate_only"
    )
    assert not list(family_path.parent.glob(".family_index.parquet.*.parquet"))


def _write_plan(root: Path, *, mutate_view: bool = False) -> None:
    split_rows = []
    family_rows = []
    view_rows = []
    for city_index in range(10):
        city = f"city-{city_index}"
        split_rows.append(
            {
                "city_slug": city,
                "latent_service_location_id": f"customer-{city_index}",
                "customer_pool": "train",
            }
        )
        for track in ("train", "validation"):
            for slot in range(7):
                family_id = f"{city}-{track}-{slot}"
                day_type = "weekday" if slot < 5 else "weekend"
                family_rows.append(
                    {
                        "family_id": family_id,
                        "city_slug": city,
                        "track_id": track,
                        "day_type": day_type,
                    }
                )
    for index in range(2_590):
        view_rows.append(
            {
                "view_id": f"view-{index}",
                "family_id": family_rows[index % len(family_rows)]["family_id"],
                "value": index + (1 if mutate_view and index == 0 else 0),
            }
        )
    split_path = root / "customer_splits" / "all" / "customer_split_manifest.parquet"
    family_path = root / "generation_plan" / "core" / "family_index.parquet"
    view_path = root / "generation_plan" / "core" / "view_index.parquet"
    split_path.parent.mkdir(parents=True)
    family_path.parent.mkdir(parents=True)
    pd.DataFrame(split_rows).to_parquet(split_path, index=False)
    pd.DataFrame(family_rows).to_parquet(family_path, index=False)
    pd.DataFrame(view_rows).to_parquet(view_path, index=False)


def test_c0_exact_comparison_accepts_exact_and_rejects_changed_view(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    exact = tmp_path / "exact"
    changed = tmp_path / "changed"
    _write_plan(baseline)
    _write_plan(exact)
    _write_plan(changed, mutate_view=True)
    accepted = COMPARE.compare_c0(baseline, exact)
    rejected = COMPARE.compare_c0(baseline, changed)
    assert accepted["passed"]
    assert all(accepted["fixed_counts"].values())
    assert not rejected["passed"]
    assert not rejected["view_registry"]["passed"]


def test_reuse_frozen_customer_split_copies_approved_artifacts(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    source = baseline / "customer_splits" / "city"
    source.mkdir(parents=True)
    report = {"schema": "cle_evrptw_customer_split_report_v1", "city_slug": "city"}
    (source / "customer_split_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    payloads = {
        "customer_split_manifest.parquet": b"split",
        "community_manifest.parquet": b"communities",
        "community_adjacency.parquet": b"adjacency",
    }
    for name, payload in payloads.items():
        (source / name).write_bytes(payload)

    output = tmp_path / "candidate"
    reused = RUNNER._reuse_frozen_customer_split(baseline, output, "city")
    assert reused["frozen_split_reused"] is True
    assert reused["frozen_split_source"] == str(source.resolve())
    for name, payload in payloads.items():
        assert (output / "customer_splits" / "city" / name).read_bytes() == payload
