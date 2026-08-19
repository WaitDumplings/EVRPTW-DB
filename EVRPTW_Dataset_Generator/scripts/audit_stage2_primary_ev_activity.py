#!/usr/bin/env python3
"""Run the deterministic route-level EV activity gate on primary pilot views."""

from __future__ import annotations

import argparse
from pathlib import Path

from evrptw_stage2.ev_activity import write_primary_pilot_ev_activity_audit
from evrptw_stage2.provenance import resolve_git_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance = resolve_git_provenance(
        Path(__file__).resolve().parents[2],
        require_clean=True,
        require_branch="stage2-repair-candidate",
    )
    report = write_primary_pilot_ev_activity_audit(
        instance_root=args.instance_root,
        output=args.output,
        code_provenance=provenance,
    )
    print(
        "EV activity audit: "
        f"status={report['status']} views={report['completed_view_count']}/{report['view_count']} "
        f"cs_visits={report['charging_station_visit_count']} "
        f"binding_routes={report['battery_binding_route_count_without_cs']} "
        f"effect_views={report['energy_ablation_effect_view_count']}"
    )
    print(f"Report: {args.output.resolve()}")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
