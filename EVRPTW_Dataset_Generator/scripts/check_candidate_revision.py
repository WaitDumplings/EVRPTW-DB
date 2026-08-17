#!/usr/bin/env python3
"""Reject generation unless the repository is one clean candidate commit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evrptw_stage2.provenance import resolve_git_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--require-branch", default="stage2-repair-candidate")
    args = parser.parse_args()
    print(
        json.dumps(
            resolve_git_provenance(
                args.repo_root,
                require_clean=True,
                require_branch=args.require_branch,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

