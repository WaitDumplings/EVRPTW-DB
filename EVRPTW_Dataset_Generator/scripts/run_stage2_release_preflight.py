#!/usr/bin/env python3
"""Phase C2: deterministic Amazon H3/PF/leakage/5:2 release preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from build_amazon_cohort_split import build_split
from evrptw_stage2.provenance import resolve_git_provenance


PRIMARY_SCALES = (100, 500, 1_000)
DAY_TYPES = ("weekday", "weekend")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_families(plan_root: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(path) for path in sorted(plan_root.rglob("family_index.parquet"))]
    if not parts:
        raise FileNotFoundError(f"No family plan under {plan_root}")
    return pd.concat(parts, ignore_index=True).drop_duplicates("family_id")


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    code_provenance = resolve_git_provenance(
        Path(__file__).resolve().parents[2],
        require_clean=True,
        require_branch="stage2-repair-candidate",
    )
    frozen = json.loads(args.cohort_split.read_text(encoding="utf-8"))
    rebuilt = build_split(args.amazon_artifact_root, seed=int(frozen["frozen_seed"]))
    h3 = {
        "selection_id": frozen.get("metric_holdout", {}).get("selection_id"),
        "station_codes": frozen.get("metric_holdout", {}).get("station_codes"),
        "deterministic_search_station_codes": rebuilt["metric_holdout"]["station_codes"],
    }
    h3["passed"] = (
        h3["selection_id"] == "H3"
        and h3["station_codes"] == h3["deterministic_search_station_codes"]
        and h3["station_codes"] == ["DCH2", "DLA9", "DSE2"]
    )
    leakage = dict(frozen.get("leakage_assertions", {}))
    leakage_passed = len(leakage) == 4 and all(bool(value) for value in leakage.values())

    support_rows = []
    for pool in ("GEN-TRAIN", "GEN-EVAL", "METRIC-HOLDOUT"):
        for day_type in DAY_TYPES:
            values = frozen["support"][pool]["by_day_type"][day_type]
            for scale in PRIMARY_SCALES:
                structure = int(values[f"single_structure_days_ge_{scale}"])
                order = int(values[f"single_order_days_ge_{scale}"])
                support_rows.append(
                    {
                        "pool": pool,
                        "day_type": day_type,
                        "customer_count": scale,
                        "single_structure_day_count": structure,
                        "single_order_day_count": order,
                        "passed": structure > 0 and order > 0,
                    }
                )
    eval_allocation = frozen.get("evaluation_track_allocation", {})
    pf2_passed = (
        all(row["passed"] for row in support_rows)
        and bool(eval_allocation.get("station_day_ledgers_pairwise_disjoint_and_exhaustive"))
        and not bool(eval_allocation.get("exact_template_reuse_between_evaluation_tracks", True))
    )

    c1 = json.loads(args.connectivity_audit.read_text(encoding="utf-8"))
    if c1.get("schema") != "cle_evrptw_phase_c1_terminal_connectivity_audit_v2":
        raise ValueError("C2 requires the C1 pre-split connectivity audit v2 schema")
    if c1.get("rule_id") != "connectivity_quarantine_precedes_customer_split_v1":
        raise ValueError("C2 requires the frozen C1-Q1 pre-split quarantine rule")
    if c1.get("code_provenance", {}).get("code_commit") != code_provenance["code_commit"]:
        raise ValueError("C1 report is not bound to the current clean candidate commit")
    plan_registry = json.loads(
        (args.plan_root / "split_registry.json").read_text(encoding="utf-8")
    )
    if plan_registry.get("code_provenance", {}).get("code_commit") != code_provenance[
        "code_commit"
    ]:
        raise ValueError("C2 pilot plan is not bound to the current clean candidate commit")
    pf1_rows = [
        {
            "city_slug": city["city_slug"],
            "passed": bool(city["pf1"]["passed"]),
            "minimum_exact_lower_bound_count": min(
                row["exact_direct_bidirectional_energy_lower_bound_count"]
                for row in city["pf1"]["rows"]
            ),
        }
        for city in c1["cities"]
    ]
    pf1_passed = bool(c1.get("passed")) and all(row["passed"] for row in pf1_rows)

    families = _load_families(args.plan_root)
    slot_rows = []
    slot_passed = True
    for (city, track), rows in families.groupby(["city_slug", "track_id"], sort=True):
        counts = rows["day_type"].value_counts().to_dict()
        row = {
            "city_slug": str(city),
            "track_id": str(track),
            "family_count": len(rows),
            "weekday_count": int(counts.get("weekday", 0)),
            "weekend_count": int(counts.get("weekend", 0)),
        }
        row["passed"] = (
            row["family_count"] == 7
            and row["weekday_count"] == 5
            and row["weekend_count"] == 2
        )
        slot_passed &= bool(row["passed"])
        slot_rows.append(row)
    slot_passed &= (
        len(families) == 140
        and set(families["track_id"].astype(str)) == {"train", "validation"}
        and families["city_slug"].nunique() == 10
    )

    passed = h3["passed"] and leakage_passed and pf1_passed and pf2_passed and slot_passed
    return {
        "schema": "cle_evrptw_phase_c2_release_preflight_v1",
        "code_provenance": code_provenance,
        "passed": bool(passed),
        "h3": h3,
        "pf1": {"passed": pf1_passed, "cities": pf1_rows},
        "pf2_and_metric_holdout_support": {
            "passed": pf2_passed,
            "rows": support_rows,
        },
        "leakage_assertions": {"passed": leakage_passed, "values": leakage},
        "slot_ledger_5_to_2": {"passed": bool(slot_passed), "rows": slot_rows},
        "inputs": {
            "cohort_split": str(args.cohort_split),
            "cohort_split_sha256": _sha256(args.cohort_split),
            "connectivity_audit": str(args.connectivity_audit),
            "connectivity_audit_sha256": _sha256(args.connectivity_audit),
        },
        "failure_semantics": "any_zero_or_failed_primary_cell_forbids_smoke_and_pilot",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amazon-artifact-root", type=Path, required=True)
    parser.add_argument("--cohort-split", type=Path, required=True)
    parser.add_argument("--connectivity-audit", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_preflight(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
