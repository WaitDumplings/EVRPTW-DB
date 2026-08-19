from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from evrptw_stage2.materialize import _cle_reference
from evrptw_stage2.profile import load_reference_profile
from evrptw_stage2.promotion import (
    PILOT_REPORT_SCHEMA,
    build_pilot_acceptance_report,
    promote_reference_profile,
    sha256_file,
)
from evrptw_stage2.provenance import RevisionDisciplineError, resolve_git_provenance
from evrptw_stage2.reader import PortableCLE
from evrptw_stage2.release_discipline import (
    FAMILY_STAGE_LIMIT_S,
    PILOT_FIRST_PROJECTION_S,
    PilotStopController,
    classify_la_smoke,
    quarantine_rate_summary,
)


ROOT = Path(__file__).parents[1]
REPOSITORY_ROOT = ROOT.parent


def test_fixed_cohort_launchers_propagate_cle_v2_and_not_v1() -> None:
    paths = [
        REPOSITORY_ROOT / "generate_cle.sh",
        REPOSITORY_ROOT / "generate_instances.sh",
        REPOSITORY_ROOT / "create_dataset_archive.sh",
        ROOT / "scripts" / "export_stage2_slim.sh",
        ROOT / "scripts" / "restore_stage2_instances.sh",
    ]
    for path in paths:
        content = path.read_text(encoding="utf-8")
        assert "CLE_v2/us_11city" in content
        assert "CLE_v1/us_11city" not in content


def test_clean_candidate_revision_is_required(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-b", "stage2-repair-candidate"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("v1", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=tmp_path, check=True)
    provenance = resolve_git_provenance(
        tmp_path, require_clean=True, require_branch="stage2-repair-candidate"
    )
    assert len(provenance["code_commit"]) == 40
    assert provenance["working_tree_clean"]
    tracked.write_text("dirty", encoding="utf-8")
    with pytest.raises(RevisionDisciplineError, match="clean working tree"):
        resolve_git_provenance(
            tmp_path, require_clean=True, require_branch="stage2-repair-candidate"
        )


def test_quarantine_rate_uses_unique_union_and_frozen_denominator() -> None:
    report = quarantine_rate_summary(
        ["a", "a", "b", "c", "d"],
        stage1_directional_ids=["a", "a"],
        stage2_node_ids=["b"],
        stage2_turn_ids=["a", "c", "c"],
        rate_limit=0.75,
    )
    assert report["audit_input_unique_terminal_count"] == 4
    assert report["stage1_directional_quarantine"]["unique_terminal_count"] == 1
    assert report["stage2_node_quarantine"]["unique_terminal_count"] == 1
    assert report["stage2_turn_quarantine"]["unique_terminal_count"] == 2
    assert report["stage1_or_stage2_union_quarantine"]["unique_terminal_count"] == 3
    assert report["stage1_or_stage2_union_quarantine"]["rate"] == pytest.approx(0.75)
    assert report["passed"]


def test_quarantine_rate_rejects_empty_or_outside_denominator() -> None:
    with pytest.raises(ValueError, match="denominator is empty"):
        quarantine_rate_summary([], stage1_directional_ids=[], stage2_turn_ids=[], rate_limit=0.1)
    with pytest.raises(ValueError, match="outside audit input"):
        quarantine_rate_summary(
            ["a"], stage1_directional_ids=[], stage2_turn_ids=["b"], rate_limit=0.1
        )


@pytest.mark.parametrize(
    ("terminal", "total", "status"),
    [
        (3600.0, 7200.0, "GREEN"),
        (3600.0001, 7200.0, "AMBER"),
        (7200.0, 7200.0, "AMBER"),
        (7200.0001, 1.0, "RED"),
        (1.0, 7200.0001, "RED"),
    ],
)
def test_la_smoke_rule_has_no_boundary_gap(terminal: float, total: float, status: str) -> None:
    assert classify_la_smoke(
        terminal_selection_s=terminal, family_total_s=total
    )["status"] == status


def test_pilot_stop_rule_covers_timeout_projection_zero_and_duplicate() -> None:
    timeout = PilotStopController(planned_family_count=140, started_monotonic=0.0)
    timeout.observe_chunk(
        {
            "materialized": [
                {
                    "family_id": "f1",
                    "materialization_seconds": FAMILY_STAGE_LIMIT_S + 1,
                    "stage_timings_seconds": {"routing": FAMILY_STAGE_LIMIT_S + 2},
                }
            ]
        }
    )
    assert timeout.stopped

    zero = PilotStopController(planned_family_count=140, started_monotonic=0.0)
    zero.poll(PILOT_FIRST_PROJECTION_S)
    assert zero.stop_reasons[0]["reason_code"] == "no_completed_family_after_4h"

    projected = PilotStopController(planned_family_count=140, started_monotonic=0.0)
    projected.completed_family_count = 10
    projected.poll(PILOT_FIRST_PROJECTION_S)
    assert projected.stop_reasons[0]["reason_code"] == "projected_total_exceeded_36h"

    duplicate = PilotStopController(planned_family_count=140, started_monotonic=0.0)
    rejection = {
        "family_id": "f2",
        "retryable": False,
        "roster_fingerprint": "r" * 64,
        "reason_code": "terminal_connectivity",
    }
    duplicate.observe_chunk({"rejected_attempts": [rejection], "unresolved_family_ids": ["f2"]})
    assert not duplicate.stopped
    duplicate.observe_chunk({"rejected_attempts": [rejection], "unresolved_family_ids": []})
    assert duplicate.stop_reasons[0]["reason_code"] == "duplicate_nonretryable_signature"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_pilot_report_and_profile_promotion_require_all_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialized = [
        {"family_id": f"f{index}", "status": "materialized"}
        for index in range(140)
    ]
    verified = [
        {"family_id": f"f{index}", "passed": True} for index in range(140)
    ]
    run_path = tmp_path / "run.json"
    phase1_path = tmp_path / "phase1.json"
    operational_path = tmp_path / "operational.json"
    spatial_path = tmp_path / "spatial.json"
    q90_path = tmp_path / "q90_v1.json"
    connectivity_path = tmp_path / "connectivity.json"
    preflight_path = tmp_path / "preflight.json"
    smoke_path = tmp_path / "smoke.json"
    connectivity_acceptance_path = tmp_path / "connectivity_acceptance.json"
    sensitivity_path = tmp_path / "sensitivity.json"
    generation_commit = "a" * 40
    post_commit = "b" * 40
    reviewed_commit = "c" * 40
    smoke_commit = "d" * 40
    acceptance_commit = "e" * 40
    _write_json(
        run_path,
        {
            "mode": "non_release_pilot",
            "passed": True,
            "execution": {"selected_family_count": 140},
            "run_discipline": {
                "schema": "cle_evrptw_pilot_stop_discipline_v1",
                "stopped": False,
            },
            "materialized": materialized,
            "verified": verified,
            "unresolved_family_ids": [],
            "code_provenance": {"code_commit": generation_commit},
            "generation_code_commit": generation_commit,
            "reconciliation_code_commit": post_commit,
            "reconciled": True,
            "family_artifacts_modified": False,
            "original_exception": {"type": "BrokenPipeError"},
        },
    )
    _write_json(phase1_path, {"all_hard_gates_passed": True})
    _write_json(
        operational_path,
        {
            "schema": "amazon_operational_transfer_acceptance_v2",
            "passed": True,
            "family_artifacts_modified": False,
            "hash_validation_performed": False,
            "code_provenance": {"code_commit": acceptance_commit},
        },
    )
    _write_json(
        spatial_path,
        {
            "schema": "cross_city_spatial_diagnostic_v1",
            "status": "complete_report_only",
            "hard_gate": False,
            "contributes_to_operational_acceptance": False,
            "construct_validity_review": {
                "triggered_by_q90_v1_failure": True,
                "threshold_changed": False,
            },
            "code_provenance": {"code_commit": acceptance_commit},
        },
    )
    _write_json(
        q90_path,
        {
            "schema": "evrptw_station_block_q90_gate_v1",
            "release_calibrated": False,
            "code_provenance": {"code_commit": post_commit},
        },
    )
    _write_json(
        connectivity_path,
        {
            "schema": "cle_evrptw_phase_c1_terminal_connectivity_audit_v3",
            "passed": False,
            "structural_contract_passed": True,
            "r2_v1": {"outcome": "triggered_stop_and_review"},
            "code_provenance": {"code_commit": reviewed_commit},
        },
    )
    _write_json(
        connectivity_acceptance_path,
        {
            "schema": "cle_evrptw_connectivity_audit_acceptance_v2",
            "passed": True,
            "c2_allowed": True,
            "inputs": {"connectivity_audit": str(connectivity_path)},
            "automated_gate": {"passed": True},
            "manual_h64_gate": {
                "passed": True,
                "reviewer_signoff_id": "signed-review",
            },
            "code_provenance": {"code_commit": reviewed_commit},
        },
    )
    _write_json(
        preflight_path,
        {
            "passed": True,
            "code_provenance": {"code_commit": reviewed_commit},
        },
    )
    _write_json(
        smoke_path,
        {
            "run_discipline": {"status": "GREEN", "pilot_allowed": True},
            "code_provenance": {"code_commit": smoke_commit},
        },
    )
    _write_json(
        sensitivity_path,
        {
            "schema": "evrptw_charging_derating_sensitivity_v1",
            "factors": [0.85, 0.9, 0.95],
            "rows": [{}, {}, {}],
            "code_provenance": {"code_commit": post_commit},
        },
    )

    def reject_hashing(_path: str | Path) -> str:
        raise AssertionError("pilot acceptance must not hash evidence files")

    monkeypatch.setattr("evrptw_stage2.promotion.sha256_file", reject_hashing)
    arguments = {
        "run_report_path": run_path,
        "phase1_report_path": phase1_path,
        "operational_transfer_report_path": operational_path,
        "spatial_diagnostic_path": spatial_path,
        "historical_q90_report_path": q90_path,
        "connectivity_audit_path": connectivity_path,
        "connectivity_acceptance_path": connectivity_acceptance_path,
        "release_preflight_path": preflight_path,
        "la_smoke_report_path": smoke_path,
        "charging_sensitivity_path": sensitivity_path,
        "acceptance_code_provenance": {"code_commit": acceptance_commit},
    }
    report = build_pilot_acceptance_report(**arguments)
    assert report["schema"] == PILOT_REPORT_SCHEMA
    assert report["passed"]
    assert report["hash_validation_performed"] is False
    assert report["evidence_inventory_method"] == "path_size_mtime_ns_no_hash"
    assert all("sha256" not in item for item in report["evidence"].values())
    assert len(set(filter(None, report["evidence_code_commits"].values()))) == 5

    operational = json.loads(operational_path.read_text(encoding="utf-8"))
    operational["passed"] = False
    _write_json(operational_path, operational)
    red_report = build_pilot_acceptance_report(**arguments)
    assert red_report["passed"] is False
    assert [
        label for label, passed in red_report["checks"].items() if not passed
    ] == ["amazon_operational_transfer_v2_passed"]

    profile_path = ROOT / "configs" / "us_reference_instance_profile_v2.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="release_calibrated"):
        load_reference_profile(profile_path, official=True)
    promoted = promote_reference_profile(
        profile,
        pilot_report=report,
        pilot_report_sha256="b" * 64,
        acceptance_config_sha256="c" * 64,
        advisor_signoff_id="advisor-approval-1",
    )
    registry = ROOT / "configs" / profile["charging"]["national_mode_median_registry"]
    copied_registry = tmp_path / registry.name
    copied_registry.write_bytes(registry.read_bytes())
    promoted_path = tmp_path / "promoted.json"
    promoted_path.write_text(json.dumps(promoted), encoding="utf-8")
    loaded = load_reference_profile(promoted_path, official=True)
    assert loaded["profile_status"] == "release_calibrated"
    with pytest.raises(ValueError, match="advisor sign-off"):
        promote_reference_profile(
            profile,
            pilot_report=report,
            pilot_report_sha256=sha256_file(run_path),
            acceptance_config_sha256="c" * 64,
            advisor_signoff_id="",
        )


def test_instance_manifest_cle_reference_is_v2_and_content_bound(tmp_path: Path) -> None:
    city_root = tmp_path / "cities" / "x"
    city_root.mkdir(parents=True)
    manifest_path = city_root / "manifest.json"
    manifest = {"connectivity_contract": {"id": "directed_projection_roundtrip_v2"}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cle = PortableCLE(
        root=city_root,
        city_slug="x",
        mode="non_release_pilot",
        manifest=manifest,
        graph_path=city_root / "g",
        service_locations_path=city_root / "s",
        depots_path=city_root / "d",
        chargers_path=city_root / "c",
        speeds_path=city_root / "v",
        customer_eligibility_field="eligible",
        depot_eligibility_field="eligible",
        charger_eligibility_field="eligible",
        warnings=(),
    )
    reference = _cle_reference(cle)
    assert reference["contract_root"] == "EVRPTW_Dataset/CLE_v2/us_11city"
    assert reference["city_relative_path"] == "cities/x"
    assert len(reference["city_manifest_sha256"]) == 64
