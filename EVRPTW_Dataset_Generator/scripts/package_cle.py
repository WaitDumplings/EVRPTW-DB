#!/usr/bin/env python3
"""Package one technical CLE work artifact as a self-contained CLE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evrptw_cle.cle import package_cle, verify_cle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cle", type=Path)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--road-manifest", type=Path)
    parser.add_argument("--destination-cle", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = verify_cle(args.destination_cle, require_portable=True)
    else:
        missing = [
            name
            for name, value in {
                "source-cle": args.source_cle,
                "graph": args.graph,
                "road-manifest": args.road_manifest,
            }.items()
            if value is None
        ]
        if missing:
            parser.error(f"packaging requires: {', '.join(missing)}")
        result = package_cle(
            source_cle_dir=args.source_cle,
            graph_path=args.graph,
            road_manifest_path=args.road_manifest,
            destination_cle_dir=args.destination_cle,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
