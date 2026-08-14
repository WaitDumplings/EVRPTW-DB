"""Build normalized HPMS-to-directed-OSM edge evidence for one city."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evrptw_cle.hpms_match import HPMSMatchOptions, build_hpms_edge_matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-slug", required=True)
    parser.add_argument("--hpms", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-radius-m", type=float, default=75.0)
    parser.add_argument("--overlap-buffer-m", type=float, default=25.0)
    parser.add_argument("--minimum-overlap-ratio", type=float, default=0.20)
    parser.add_argument("--maximum-orientation-delta-deg", type=float, default=30.0)
    parser.add_argument("--high-confidence-distance-m", type=float, default=25.0)
    parser.add_argument("--high-confidence-overlap-ratio", type=float, default=0.50)
    parser.add_argument(
        "--high-confidence-orientation-delta-deg", type=float, default=15.0
    )
    parser.add_argument("--ambiguity-distance-margin-m", type=float, default=10.0)
    parser.add_argument("--ambiguity-overlap-margin", type=float, default=0.20)
    args = parser.parse_args()
    options = HPMSMatchOptions(
        candidate_radius_m=args.candidate_radius_m,
        overlap_buffer_m=args.overlap_buffer_m,
        minimum_overlap_ratio=args.minimum_overlap_ratio,
        maximum_orientation_delta_deg=args.maximum_orientation_delta_deg,
        high_confidence_distance_m=args.high_confidence_distance_m,
        high_confidence_overlap_ratio=args.high_confidence_overlap_ratio,
        high_confidence_orientation_delta_deg=(
            args.high_confidence_orientation_delta_deg
        ),
        ambiguity_distance_margin_m=args.ambiguity_distance_margin_m,
        ambiguity_overlap_margin=args.ambiguity_overlap_margin,
    )
    result = build_hpms_edge_matches(
        city_slug=args.city_slug,
        hpms_path=args.hpms,
        graph_path=args.graph,
        boundary_path=args.boundary,
        output_path=args.output,
        options=options,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
