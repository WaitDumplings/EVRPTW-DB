from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "stage2_release_preflight",
    SCRIPTS / "run_stage2_release_preflight.py",
)
assert SPEC is not None and SPEC.loader is not None
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


def test_c2_requires_passed_same_commit_r2_v2_before_other_work(tmp_path: Path) -> None:
    commit = "a" * 40
    c1_path = tmp_path / "c1.json"
    acceptance_path = tmp_path / "acceptance.json"
    c1_path.write_text(
        json.dumps(
            {
                "schema": "cle_evrptw_phase_c1_terminal_connectivity_audit_v3",
                "rule_id": "layered_stage1_pre_split_stage2_family_mask_v1",
                "code_provenance": {"code_commit": commit},
            }
        ),
        encoding="utf-8",
    )
    acceptance = {
        "schema": "cle_evrptw_connectivity_audit_acceptance_v2",
        "rule_id": "r2_v2_replayable_connectivity_certificate_gate_v1",
        "code_provenance": {"code_commit": commit},
        "r2_v1_provenance": {"outcome": "triggered_stop_and_review"},
        "passed": False,
        "c2_allowed": False,
    }
    acceptance_path.write_text(json.dumps(acceptance), encoding="utf-8")
    args = argparse.Namespace(
        connectivity_audit=c1_path,
        connectivity_acceptance=acceptance_path,
    )
    with pytest.raises(ValueError, match="did not authorize C2"):
        PREFLIGHT._validated_connectivity_inputs(args, {"code_commit": commit})

    acceptance["passed"] = True
    acceptance["c2_allowed"] = True
    acceptance_path.write_text(json.dumps(acceptance), encoding="utf-8")
    c1, accepted = PREFLIGHT._validated_connectivity_inputs(
        args, {"code_commit": commit}
    )
    assert c1["schema"].endswith("_v3")
    assert accepted["c2_allowed"]
