"""Pilot evidence bundle and advisor-gated profile promotion (R-6)."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


PILOT_REPORT_SCHEMA = "cle_evrptw_stage2_pilot_acceptance_report_v1"
PROMOTION_SCHEMA = "evrptw_profile_acceptance_promotion_v1"


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_pilot_acceptance_report(
    *,
    run_report_path: str | Path,
    phase1_report_path: str | Path,
    q90_report_path: str | Path,
    connectivity_audit_path: str | Path,
    release_preflight_path: str | Path,
    la_smoke_report_path: str | Path,
    charging_sensitivity_path: str | Path,
) -> dict[str, Any]:
    """Combine immutable pilot evidence and fail closed on any release gate."""

    paths = {
        "stage2_run_report": Path(run_report_path),
        "phase1_report": Path(phase1_report_path),
        "q90_report": Path(q90_report_path),
        "connectivity_audit": Path(connectivity_audit_path),
        "release_preflight": Path(release_preflight_path),
        "la_smoke_report": Path(la_smoke_report_path),
        "charging_sensitivity": Path(charging_sensitivity_path),
    }
    run = _read(paths["stage2_run_report"])
    phase1 = _read(paths["phase1_report"])
    q90 = _read(paths["q90_report"])
    connectivity = _read(paths["connectivity_audit"])
    preflight = _read(paths["release_preflight"])
    smoke_run = _read(paths["la_smoke_report"])
    smoke = smoke_run.get("run_discipline", {})
    sensitivity = _read(paths["charging_sensitivity"])
    code_commit = str(run.get("code_provenance", {}).get("code_commit", ""))
    evidence_commits = [
        str(payload.get("code_provenance", {}).get("code_commit", ""))
        for payload in (connectivity, preflight, smoke_run, q90, sensitivity)
    ]
    successful = [
        item
        for item in run.get("materialized", [])
        if item.get("status") in {"materialized", "reused_verified"}
    ]
    verified = [item for item in run.get("verified", []) if bool(item.get("passed"))]
    discipline = run.get("run_discipline", {})
    checks = {
        "run_mode_is_non_release_pilot": run.get("mode") == "non_release_pilot",
        "pilot_discipline_recorded": (
            discipline.get("schema") == "cle_evrptw_pilot_stop_discipline_v1"
        ),
        "pilot_not_stopped": discipline.get("stopped") is False,
        "planned_140": int(run.get("execution", {}).get("selected_family_count", -1)) == 140,
        "materialized_140": len(successful) == 140,
        "verified_140": len(verified) == 140,
        "no_unresolved": not run.get("unresolved_family_ids", []),
        "runner_passed": run.get("passed") is True,
        "phase1_all_hard_gates_passed": phase1.get("all_hard_gates_passed") is True,
        "q90_release_calibrated": q90.get("release_calibrated") is True,
        "connectivity_c1_passed": connectivity.get("passed") is True,
        "amazon_h3_pf_c2_passed": preflight.get("passed") is True,
        "la_smoke_green_or_amber": (
            smoke.get("status") in {"GREEN", "AMBER"}
            and smoke.get("pilot_allowed") is True
        ),
        "charging_sensitivity_complete": (
            sensitivity.get("schema") == "evrptw_charging_derating_sensitivity_v1"
            and sensitivity.get("factors") == [0.85, 0.9, 0.95]
            and len(sensitivity.get("rows", [])) == 3
        ),
        "all_evidence_uses_one_candidate_commit": (
            len(code_commit) == 40 and all(commit == code_commit for commit in evidence_commits)
        ),
    }
    evidence = {
        label: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for label, path in paths.items()
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
        "evidence": evidence,
    }
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    core["pilot_report_id"] = "pilot_" + hashlib.sha256(canonical).hexdigest()[:24]
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
