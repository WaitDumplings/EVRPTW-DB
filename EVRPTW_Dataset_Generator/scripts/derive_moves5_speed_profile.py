#!/usr/bin/env python3
"""Derive the compact CLE speed-retention profile from a MOVES5 SQL dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evrptw_cle.moves_speed import derive_moves_speed_profile_from_sql


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moves-sql", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = derive_moves_speed_profile_from_sql(args.moves_sql)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(profile, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(profile, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
