#!/usr/bin/env python3
"""Evaluate D-5 v2 on an existing pilot without regenerating artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from evrptw_stage2.operational_acceptance import write_d5_v2_reports
from evrptw_stage2.provenance import resolve_git_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-root", type=Path, required=True)
    parser.add_argument("--amazon-artifact-root", type=Path, required=True)
    parser.add_argument("--cohort-split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--q90-v1", type=Path, required=True)
    parser.add_argument("--operational-output", type=Path, required=True)
    parser.add_argument("--spatial-output", type=Path, required=True)
    parser.add_argument("--preserved-q90-output", type=Path, required=True)
    args = parser.parse_args()
    provenance = resolve_git_provenance(
        Path(__file__).resolve().parents[2],
        require_clean=True,
        require_branch="stage2-repair-candidate",
    )
    operational, spatial = write_d5_v2_reports(
        instance_root=args.instance_root,
        amazon_artifact_root=args.amazon_artifact_root,
        cohort_split_path=args.cohort_split,
        config_path=args.config,
        q90_v1_path=args.q90_v1,
        operational_output=args.operational_output,
        spatial_output=args.spatial_output,
        preserved_q90_output=args.preserved_q90_output,
        code_provenance=provenance,
    )
    print(
        "D-5 v2 operational acceptance: "
        f"status={operational['status']} passed={operational['passed']}"
    )
    print(f"Operational report: {args.operational_output.resolve()}")
    print(
        "Spatial diagnostic: "
        f"status={spatial['status']} hard_gate={spatial['hard_gate']}"
    )
    print(f"Spatial report: {args.spatial_output.resolve()}")
    print(f"Historical Q90 v1 preserved: {args.preserved_q90_output.resolve()}")
    if not operational["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
