#!/usr/bin/env python3
"""Create an advisor-approved release-calibrated profile; never run implicitly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evrptw_stage2.promotion import promote_reference_profile, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--acceptance-config", type=Path, required=True)
    parser.add_argument("--advisor-signoff-id", required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    args = parser.parse_args()
    if args.output_profile.exists():
        raise FileExistsError(f"Refusing to overwrite profile: {args.output_profile}")
    profile = json.loads(args.candidate_profile.read_text(encoding="utf-8"))
    pilot = json.loads(args.pilot_report.read_text(encoding="utf-8"))
    promoted = promote_reference_profile(
        profile,
        pilot_report=pilot,
        pilot_report_sha256=sha256_file(args.pilot_report),
        acceptance_config_sha256=sha256_file(args.acceptance_config),
        advisor_signoff_id=args.advisor_signoff_id,
    )
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_text(
        json.dumps(promoted, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote promoted profile: {args.output_profile}")
    print("Commit and push this profile, restore a clean tree, then use a new official output root.")


if __name__ == "__main__":
    main()
