#!/usr/bin/env python3
"""Run EVRPTW-DB Release Gate 1 and optionally write its JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evrptw_cle.boundary_gate import (
    DEFAULT_MAX_LAND_AREA_RELATIVE_ERROR,
    verify_boundary_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--boundary-root", type=Path, required=True)
    parser.add_argument("--city-root", type=Path, required=True)
    parser.add_argument("--expected-city-count", type=int, default=10)
    parser.add_argument(
        "--max-land-area-relative-error",
        type=float,
        default=DEFAULT_MAX_LAND_AREA_RELATIVE_ERROR,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = verify_boundary_gate(
        args.preset,
        args.boundary_root,
        args.city_root,
        expected_city_count=args.expected_city_count,
        max_land_area_relative_error=args.max_land_area_relative_error,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    summary = report["summary"]
    print(
        f"{report['gate_id']}: {'PASS' if report['passed'] else 'FAIL'} "
        f"({summary['passed_city_count']}/{summary['checked_city_count']} cities)",
        file=sys.stderr,
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
