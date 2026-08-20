#!/usr/bin/env python3
"""Freeze two disjoint 75-family full-path toy cohorts from the official plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evrptw_stage2.provenance import resolve_git_provenance
from evrptw_stage2.toy import (
    build_full_path_toy_manifest,
    write_full_path_toy_manifest,
)
from evrptw_stage2.toy_plan import prune_generation_plan_to_toy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    provenance = resolve_git_provenance(
        repo_root,
        require_clean=True,
        require_branch="stage2-repair-candidate",
    )
    registry = json.loads((args.plan_root / "split_registry.json").read_text(encoding="utf-8"))
    if registry.get("code_provenance", {}).get("code_commit") != provenance["code_commit"]:
        raise ValueError("Generation plan belongs to a different executable commit")
    paths = sorted(args.plan_root.rglob("family_index.parquet"))
    if not paths:
        raise ValueError(f"No family plan partitions found under {args.plan_root}")
    families = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    manifest = build_full_path_toy_manifest(families, code_provenance=provenance)
    pruning = prune_generation_plan_to_toy(
        args.plan_root, manifest, manifest_path=args.output
    )
    write_full_path_toy_manifest(args.output, manifest)
    print(
        f"Full-path toy manifest: templates={manifest['template_count']} "
        f"families={manifest['family_count']} cities={manifest['covered_city_count']} "
        f"views={pruning['view_count']}"
    )
    print(f"Manifest: {args.output.resolve()}")


if __name__ == "__main__":
    main()
