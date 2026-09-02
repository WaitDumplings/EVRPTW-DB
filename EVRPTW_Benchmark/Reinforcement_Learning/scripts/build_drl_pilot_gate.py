#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest(root: Path, method: str, scale: str, stage: str) -> Path | None:
    matches = sorted(root.glob(f"R/{method}/{scale}/seed_1234/*/pilot/{stage}"))
    return matches[-1] if matches else None


def build_report(output_root: Path, evidence_root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    methods = protocol["pilot"]["required_methods"]
    gate_bytes = int(float(protocol["hardware"]["rtx_2080_ti"]["memory_gate_gib"]) * 1024**3)
    checks.append(
        {
            "method": "global",
            "check": "full_runtime_budget_approved",
            "passed": bool(protocol["pilot"].get("full_runtime_budget_approved")),
            "artifact": str(
                ROOT / "configs" / "drl_experiment_protocol_v1.yaml"
            ),
        }
    )
    for method in methods:
        for scale, stage in (("Cus100", "short_optimization"), ("Cus500", "memory"), ("Cus1000", "memory")):
            directory = _latest(output_root, method, scale, stage)
            result_path = directory / "training_result.json" if directory else Path("missing")
            validation_path = directory / "validation_summary.json" if directory else Path("missing")
            checkpoint_path = directory / "checkpoint_selected.pt" if directory else Path("missing")
            result = _json(result_path) if result_path.is_file() else {}
            passed = bool(
                result.get("status") in {"passed", "pilot_partial"}
                and checkpoint_path.is_file()
                and validation_path.is_file()
                and math.isfinite(float(result.get("wall_time_s", result.get("completed_at", 0.0))))
            )
            if scale == "Cus500" and result.get("peak_gpu_memory_bytes") is not None:
                passed = passed and int(result["peak_gpu_memory_bytes"]) <= gate_bytes
            checks.append(
                {
                    "method": method,
                    "check": f"{scale}_{stage}",
                    "passed": passed,
                    "artifact": str(directory) if directory else None,
                    "peak_gpu_memory_bytes": result.get("peak_gpu_memory_bytes"),
                }
            )
        for name in ("greedy_verifier", "best50_equality", "data_pass_resume", "wall_time_estimate"):
            path = evidence_root / f"{method}__{name}.json"
            payload = _json(path) if path.is_file() else {}
            checks.append(
                {
                    "method": method,
                    "check": name,
                    "passed": bool(payload.get("passed")),
                    "artifact": str(path),
                }
            )
    return {
        "schema": "drl_pilot_gate_report_v1",
        "protocol_id": protocol["protocol_id"],
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "full_protocol_recommendation": "unchanged" if all(check["passed"] for check in checks) else "STOP_FOR_REVIEW",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen DRL pilot gate.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs" / "drl_experiment_protocol_v1.yaml")
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    report = build_report(args.output_root, args.evidence_root, protocol)
    destination = args.output_root / "pilot_gate_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(destination)}, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
