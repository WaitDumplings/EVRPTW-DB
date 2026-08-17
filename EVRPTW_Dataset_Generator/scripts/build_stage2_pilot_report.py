#!/usr/bin/env python3
"""Build the immutable R-6 pilot acceptance bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evrptw_stage2.promotion import build_pilot_acceptance_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--phase1-report", type=Path, required=True)
    parser.add_argument("--q90-report", type=Path, required=True)
    parser.add_argument("--connectivity-audit", type=Path, required=True)
    parser.add_argument("--release-preflight", type=Path, required=True)
    parser.add_argument("--la-smoke-report", type=Path, required=True)
    parser.add_argument("--charging-sensitivity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_pilot_acceptance_report(
        run_report_path=args.run_report,
        phase1_report_path=args.phase1_report,
        q90_report_path=args.q90_report,
        connectivity_audit_path=args.connectivity_audit,
        release_preflight_path=args.release_preflight,
        la_smoke_report_path=args.la_smoke_report,
        charging_sensitivity_path=args.charging_sensitivity,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
