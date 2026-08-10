#!/usr/bin/env python3
"""Build one city's directed legal-speed layer and report traffic-calibration blockers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evrptw_cle.speed import build_legal_speed_layer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-slug", required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_legal_speed_layer(
        city_slug=args.city_slug,
        graph_path=args.graph,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
