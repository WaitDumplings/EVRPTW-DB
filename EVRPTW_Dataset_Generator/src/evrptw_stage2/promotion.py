"""Pilot evidence bundle and advisor-gated profile promotion (R-6)."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


PILOT_REPORT_SCHEMA = "cle_evrptw_stage2_pilot_acceptance_report_v3"
PROMOTION_SCHEMA = "evrptw_profile_acceptance_promotion_v1"


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _code_commit(payload: Mapping[str, Any]) -> str:
    return str(dict(payload.get("code_provenance") or {}).get("code_commit", ""))


def _valid_commit(value: str) -> bool:
    return len(value) == 40 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def build_pilot_acceptance_report(
    *,
    run_report_path: str | Path,
    phase1_report_path: str | Path,
    operational_transfer_report_path: str | Path,
    spatial_diagnostic_path: str | Path,
    historical_q90_report_path: str | Path,
    connectivity_audit_path: str | Path,
    connectivity_acceptance_path: str | Path,
    release_preflight_path: str | Path,
    la_smoke_report_path: str | Path,
    charging_sensitivity_path: str | Path,
    acceptance_code_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine pilot evidence without file hashing and fail closed on gates."""

    paths = {
        "stage2_run_report": Path(run_report_path),
        "phase1_report": Path(phase1_report_path),
        "operational_transfer": Path(operational_transfer_report_path),
        "spatial_diagnostic": Path(spatial_diagnostic_path),
        "historical_q90_v1": Path(historical_q90_report_path),
        "connectivity_audit": Path(connectivity_audit_path),
        "connectivity_acceptance": Path(connectivity_acceptance_path),
        "release_preflight": Path(release_preflight_path),
        "la_smoke_report": Path(la_smoke_report_path),
        "charging_sensitivity": Path(charging_sensitivity_path),
    }
    run = _read(paths["stage2_run_report"])
    phase1 = _read(paths["phase1_report"])
    operational = _read(paths["operational_transfer"])
    spatial = _read(paths["spatial_diagnostic"])
    historical_q90 = _read(paths["historical_q90_v1"])
    connectivity = _read(paths["connectivity_audit"])
    connectivity_acceptance = _read(paths["connectivity_acceptance"])
    preflight = _read(paths["release_preflight"])
    smoke_run = _read(paths["la_smoke_report"])
    smoke = dict(smoke_run.get("run_discipline") or {})
    sensitivity = _read(paths["charging_sensitivity"])

    generation_commit = str(
        run.get("generation_code_commit") or _code_commit(run)
    )
    reconciliation_commit = str(run.get("reconciliation_code_commit") or "")
    operational_commit = _code_commit(operational)
    spatial_commit = _code_commit(spatial)
    historical_q90_commit = _code_commit(historical_q90)
    sensitivity_commit = _code_commit(sensitivity)
    connectivity_commit = _code_commit(connectivity)
    connectivity_acceptance_commit = _code_commit(connectivity_acceptance)
    preflight_commit = _code_commit(preflight)
    smoke_commit = _code_commit(smoke_run)
    acceptance_commit = str(
        dict(acceptance_code_provenance or {}).get("code_commit", "")
    )
    evidence_commits = {
        "generation": generation_commit,
        "reconciliation": reconciliation_commit,
        "operational_transfer_v2": operational_commit,
        "spatial_diagnostic_v1": spatial_commit,
        "historical_q90_v1": historical_q90_commit,
        "charging_sensitivity": sensitivity_commit,
        "connectivity_audit": connectivity_commit,
        "connectivity_acceptance": connectivity_acceptance_commit,
        "release_preflight": preflight_commit,
        "la_smoke": smoke_commit,
        "acceptance_builder": acceptance_commit or None,
    }

    successful = [
        item
        for item in run.get("materialized", [])
        if item.get("status") in {"materialized", "reused_verified"}
    ]
    verified = [
        item for item in run.get("verified", []) if bool(item.get("passed"))
    ]
    discipline = dict(run.get("run_discipline") or {})
    manual_h64 = dict(
        connectivity_acceptance.get("manual_h64_gate") or {}
    )
    automated_r2v2 = dict(
        connectivity_acceptance.get("automated_gate") or {}
    )
    declared_audit = str(
        dict(connectivity_acceptance.get("inputs") or {}).get(
            "connectivity_audit", ""
        )
    )
    declared_audit_name = Path(declared_audit).name if declared_audit else ""

    checks = {
        "run_mode_is_non_release_pilot": run.get("mode") == "non_release_pilot",
        "pilot_discipline_recorded": (
            discipline.get("schema") == "cle_evrptw_pilot_stop_discipline_v1"
        ),
        "pilot_not_stopped": discipline.get("stopped") is False,
        "planned_140": (
            int(run.get("execution", {}).get("selected_family_count", -1)) == 140
        ),
        "materialized_140": len(successful) == 140,
        "verified_140": len(verified) == 140,
        "no_unresolved": not run.get("unresolved_family_ids", []),
        "runner_passed": run.get("passed") is True,
        "reconciliation_passed": (
            run.get("reconciled") is True
            and run.get("family_artifacts_modified") is False
            and run.get("original_exception", {}).get("type") == "BrokenPipeError"
        ),
        "phase1_all_hard_gates_passed": (
            phase1.get("all_hard_gates_passed") is True
        ),
        "amazon_operational_transfer_v2_passed": (
            operational.get("schema") == "amazon_operational_transfer_acceptance_v2"
            and operational.get("passed") is True
            and operational.get("family_artifacts_modified") is False
            and operational.get("hash_validation_performed") is False
        ),
        "spatial_diagnostic_v1_complete_report_only": (
            spatial.get("schema") == "cross_city_spatial_diagnostic_v1"
            and spatial.get("status") == "complete_report_only"
            and spatial.get("hard_gate") is False
            and spatial.get("contributes_to_operational_acceptance") is False
        ),
        "historical_q90_v1_failure_preserved": (
            historical_q90.get("schema") == "evrptw_station_block_q90_gate_v1"
            and historical_q90.get("release_calibrated") is False
            and spatial.get("construct_validity_review", {}).get(
                "triggered_by_q90_v1_failure"
            ) is True
            and spatial.get("construct_validity_review", {}).get(
                "threshold_changed"
            ) is False
        ),
        "connectivity_c1_structural_contract_passed": (
            connectivity.get("schema")
            == "cle_evrptw_phase_c1_terminal_connectivity_audit_v3"
            and connectivity.get("structural_contract_passed") is True
            and connectivity.get("r2_v1", {}).get("outcome")
            == "triggered_stop_and_review"
        ),
        "connectivity_r2_v2_automated_and_h64_passed": (
            connectivity_acceptance.get("schema")
            == "cle_evrptw_connectivity_audit_acceptance_v2"
            and connectivity_acceptance.get("passed") is True
            and connectivity_acceptance.get("c2_allowed") is True
            and automated_r2v2.get("passed") is True
            and manual_h64.get("passed") is True
            and bool(manual_h64.get("reviewer_signoff_id"))
        ),
        "connectivity_acceptance_declares_audit_path": (
            declared_audit_name == paths["connectivity_audit"].name
        ),
        "amazon_h3_pf_c2_passed": preflight.get("passed") is True,
        "la_smoke_green_or_amber": (
            smoke.get("status") in {"GREEN", "AMBER"}
            and smoke.get("pilot_allowed") is True
        ),
        "charging_sensitivity_complete": (
            sensitivity.get("schema")
            == "evrptw_charging_derating_sensitivity_v1"
            and sensitivity.get("factors") == [0.85, 0.9, 0.95]
            and len(sensitivity.get("rows", [])) == 3
        ),
        "generation_reconciliation_lineage_complete": (
            run.get("reconciled") is True
            and _valid_commit(generation_commit)
            and _valid_commit(reconciliation_commit)
            and generation_commit == _code_commit(run)
        ),
        "historical_post_evaluation_uses_reconciliation_commit": (
            _valid_commit(reconciliation_commit)
            and historical_q90_commit == reconciliation_commit
            and sensitivity_commit == reconciliation_commit
        ),
        "d5_v2_evidence_uses_acceptance_revision": (
            _valid_commit(operational_commit)
            and operational_commit == spatial_commit
            and operational_commit == acceptance_commit
        ),
        "reviewed_preflight_lineage_complete": (
            _valid_commit(connectivity_commit)
            and connectivity_acceptance_commit == connectivity_commit
            and preflight_commit == connectivity_commit
            and _valid_commit(smoke_commit)
        ),
    }
    evidence = {}
    for label, path in paths.items():
        stat = path.resolve(strict=True).stat()
        evidence[label] = {
            "path": str(path.resolve()),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    core = {
        "schema": PILOT_REPORT_SCHEMA,
        "checks": checks,
        "passed": all(checks.values()),
        "planned_family_count": 140,
        "materialized_family_count": len(successful),
        "verified_family_count": len(verified),
        "unresolved_family_ids": list(run.get("unresolved_family_ids", [])),
        "code_provenance": dict(run.get("code_provenance", {})),
        "acceptance_code_provenance": dict(acceptance_code_provenance or {}),
        "evidence_code_commits": evidence_commits,
        "evidence": evidence,
        "evidence_inventory_method": "path_size_mtime_ns_no_hash",
        "hash_validation_performed": False,
    }
    report_commit = acceptance_commit or reconciliation_commit
    core["pilot_report_id"] = (
        f"pilot_{generation_commit[:8]}_{reconciliation_commit[:8]}_"
        f"{report_commit[:8]}_140family"
    )
    return core


def promote_reference_profile(
    profile: Mapping[str, Any],
    *,
    pilot_report: Mapping[str, Any],
    pilot_report_sha256: str,
    acceptance_config_sha256: str,
    advisor_signoff_id: str,
) -> dict[str, Any]:
    """Return a promoted copy; the caller must commit it before official generation."""

    if profile.get("profile_status") != "candidate_calibration":
        raise ValueError("Only candidate_calibration profiles can be promoted")
    if bool(profile.get("official_generation_eligible", False)):
        raise ValueError("Candidate profile is already marked official")
    if pilot_report.get("schema") != PILOT_REPORT_SCHEMA or pilot_report.get("passed") is not True:
        raise ValueError("Profile promotion requires a complete passing 140-family pilot report")
    if not str(advisor_signoff_id).strip():
        raise ValueError("Profile promotion requires an explicit advisor sign-off ID")
    for label, value in (
        ("pilot_report_sha256", pilot_report_sha256),
        ("acceptance_config_sha256", acceptance_config_sha256),
    ):
        digest = str(value).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"{label} must be a SHA256 digest")
    promoted = deepcopy(dict(profile))
    promoted["profile_status"] = "release_calibrated"
    promoted["official_generation_eligible"] = True
    promoted["acceptance_promotion"] = {
        "schema": PROMOTION_SCHEMA,
        "pilot_report_id": str(pilot_report["pilot_report_id"]),
        "pilot_report_sha256": str(pilot_report_sha256).lower(),
        "acceptance_config_sha256": str(acceptance_config_sha256).lower(),
        "advisor_signoff_id": str(advisor_signoff_id).strip(),
        "required_next_step": "commit_and_push_clean_acceptance_revision_before_full_run",
    }
    return promoted
