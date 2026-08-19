#!/usr/bin/env python3
"""Read-only Stage-2 acceptance v3 and operational diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from evrptw_stage2.construct_acceptance import write_construct_valid_acceptance
from evrptw_stage2.provenance import resolve_git_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-root", type=Path, required=True)
    parser.add_argument("--amazon-artifact-root", type=Path, required=True)
    parser.add_argument("--cohort-split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--acceptance-output", type=Path, required=True)
    parser.add_argument("--diagnostics-output", type=Path, required=True)
    args = parser.parse_args()
    provenance = resolve_git_provenance(
        Path(__file__).resolve().parents[2],
        require_clean=True,
        require_branch="stage2-repair-candidate",
    )
    acceptance, diagnostics = write_construct_valid_acceptance(
        instance_root=args.instance_root,
        amazon_artifact_root=args.amazon_artifact_root,
        cohort_split_path=args.cohort_split,
        config_path=args.config,
        acceptance_output=args.acceptance_output,
        diagnostics_output=args.diagnostics_output,
        code_provenance=provenance,
    )
    print(f"Stage-2 acceptance v3: status={acceptance['status']} passed={acceptance['passed']}")
    print(f"Acceptance report: {args.acceptance_output.resolve()}")
    print(f"Diagnostics: rows={diagnostics['row_count']} hard_gate={diagnostics['hard_gate']}")
    print(f"Diagnostics report: {args.diagnostics_output.resolve()}")
    if not acceptance["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
