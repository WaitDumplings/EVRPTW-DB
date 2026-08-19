#!/usr/bin/env python3
"""Reconcile a verified Stage-2 pilot after report-control failure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evrptw_stage2.provenance import resolve_git_provenance
from evrptw_stage2.reconciliation import reconcile_existing_pilot


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--expected-family-count", type=int, default=140)
    parser.add_argument(
        "--require-branch",
        default="stage2-repair-candidate",
        help="Clean branch required for authoritative reconciliation.",
    )
    parser.add_argument(
        "--expected-generation-commit",
        required=True,
        help="Exact generation commit recorded in the original RED report.",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    repository_root = (
        args.repository_root.resolve()
        if args.repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    provenance = resolve_git_provenance(
        repository_root,
        require_clean=True,
        require_branch=args.require_branch,
    )
    result = reconcile_existing_pilot(
        args.output_root,
        reconciliation_code_provenance=provenance,
        expected_family_count=args.expected_family_count,
        expected_generation_commit=args.expected_generation_commit,
        workers=args.workers,
    )
    try:
        print(f"Stage-2 reconciliation status: {result['status']}")
        print(
            f"planned={result['planned_count']} "
            f"materialized={result['materialized_count']} "
            f"verified={result['verified_count']}"
        )
        print(
            "report="
            + str(
                args.output_root.resolve()
                / "reports"
                / "report_reconciliation_v1.json"
            )
        )
    except BrokenPipeError:
        pass
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
