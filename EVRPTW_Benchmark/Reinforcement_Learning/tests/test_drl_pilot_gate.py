from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_drl_pilot_gate.py"
SPEC = importlib.util.spec_from_file_location("drl_pilot_gate", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_runtime_budget_requires_explicit_approval(tmp_path: Path) -> None:
    protocol = {
        "protocol_id": "runtime-gate-test",
        "hardware": {"rtx_2080_ti": {"memory_gate_gib": 9.5}},
        "pilot": {
            "required_methods": [],
            "full_runtime_budget_approved": False,
        },
    }
    report = MODULE.build_report(tmp_path, tmp_path / "evidence", protocol)
    assert report["passed"] is False
    assert report["full_protocol_recommendation"] == "STOP_FOR_REVIEW"

    protocol["pilot"]["full_runtime_budget_approved"] = True
    report = MODULE.build_report(tmp_path, tmp_path / "evidence", protocol)
    assert report["passed"] is True
    assert report["full_protocol_recommendation"] == "unchanged"
