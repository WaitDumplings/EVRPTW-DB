"""Resumable Stage-2 runner for community splits, plans, families, and QA."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from evrptw_stage2.artifacts import verify_materialized_family
from evrptw_stage2.community import build_customer_split
from evrptw_stage2.config import load_stage2_config
from evrptw_stage2.materialize import materialize_family
from evrptw_stage2.planning import (
    build_generation_plan,
    derive_seed,
    materialization_attempt_inputs,
    write_generation_plan,
)
from evrptw_stage2.profile import load_reference_profile
from evrptw_stage2.reader import load_portable_cle

STAGES = ("preflight", "splits", "plan", "materialize", "verify")


def _process_peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--cle-root", type=Path, required=True)
    parser.add_argument("--block-group-preset", type=Path, required=True)
    parser.add_argument("--block-group-source-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("official", "non_release_pilot"), default="official"
    )
    parser.add_argument("--cities", nargs="+")
    parser.add_argument("--tracks", nargs="+")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--pilot-families-per-city", type=int)
    parser.add_argument("--max-families", type=int)
    parser.add_argument("--family-ids", nargs="+")
    parser.add_argument("--max-attempts-per-family", type=int, default=1)
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_plan(plan_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    family_parts = [pd.read_parquet(path) for path in sorted(plan_root.rglob("family_index.parquet"))]
    view_parts = [pd.read_parquet(path) for path in sorted(plan_root.rglob("view_index.parquet"))]
    if not family_parts or not view_parts:
        raise FileNotFoundError(f"Incomplete plan under {plan_root}")
    return (
        pd.concat(family_parts, ignore_index=True).drop_duplicates("family_id"),
        pd.concat(view_parts, ignore_index=True).drop_duplicates("view_id"),
        json.loads((plan_root / "split_registry.json").read_text(encoding="utf-8")),
    )


def main() -> None:
    run_started = time.perf_counter()
    args = make_parser().parse_args()
    if args.max_attempts_per_family <= 0:
        raise ValueError("--max-attempts-per-family must be positive")
    config = load_stage2_config(args.config)
    non_release = args.mode == "non_release_pilot"
    profile = load_reference_profile(args.profile, official=not non_release)
    all_cities = (*config.train_cities, config.heldout_city)
    cities = tuple(dict.fromkeys(args.cities or all_cities))
    if non_release and args.pilot_families_per_city is None:
        raise ValueError("non_release_pilot requires --pilot-families-per-city")
    if not non_release and args.pilot_families_per_city is not None:
        raise ValueError("Official generation cannot use pilot family counts")
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_report: dict[str, Any] = {
        "schema": "cle_evrptw_stage2_run_report_v1",
        "mode": args.mode,
        "cities": list(cities),
        "stages": list(args.stages),
        "preflight": [],
        "splits": {},
        "materialized": [],
        "rejected_attempts": [],
        "unresolved_family_ids": [],
        "verified": [],
    }

    cle_by_city = {}
    for city in cities:
        cle = load_portable_cle(args.cle_root, city, mode=args.mode)
        cle_by_city[city] = cle
        run_report["preflight"].append(cle.eligibility_summary())

    source_preset = json.loads(args.block_group_preset.read_text(encoding="utf-8"))
    if "splits" in args.stages:
        for city, cle in cle_by_city.items():
            state = source_preset["city_to_state"][city]
            state_fips = source_preset["states"][state]
            vintage = int(source_preset["vintage"])
            block_groups = args.block_group_source_dir / f"tl_{vintage}_{state_fips}_bg.zip"
            split_dir = args.output_root / "customer_splits" / city
            report_path = split_dir / "customer_split_report.json"
            if report_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
            else:
                report = build_customer_split(
                    cle,
                    block_groups_path=block_groups,
                    output_dir=split_dir,
                    split_seed=config.master_seed,
                    heldout_fraction=config.heldout_community_fraction,
                    partition_version=config.community_partition_version,
                )
            run_report["splits"][city] = report

    plan_root = args.output_root / "generation_plan"
    needs_plan = bool({"plan", "materialize", "verify"} & set(args.stages))
    if "plan" in args.stages and not (plan_root / "split_registry.json").is_file():
        families, views, registry = build_generation_plan(
            config,
            available_cities=cities,
            pilot_families_per_city=args.pilot_families_per_city,
            include_tracks=args.tracks,
            non_release_pilot=non_release,
        )
        registry["cle_preflight"] = run_report["preflight"]
        write_generation_plan(plan_root, families, views, registry)
    if not needs_plan:
        run_report["passed"] = True
        _write_json(args.output_root / "stage2_run_report.json", run_report)
        print(json.dumps(run_report, indent=2, sort_keys=True, ensure_ascii=False))
        return
    families, views, registry = _load_plan(plan_root)
    run_report["generation_plan"] = registry

    selected_families = families.sort_values("family_id")
    if args.family_ids:
        requested_family_ids = set(map(str, args.family_ids))
        selected_families = selected_families.loc[
            selected_families["family_id"].astype(str).isin(requested_family_ids)
        ]
        missing_family_ids = sorted(
            requested_family_ids - set(selected_families["family_id"].astype(str))
        )
        if missing_family_ids:
            raise ValueError(f"Requested family IDs are absent from the plan: {missing_family_ids}")
    if args.max_families is not None:
        selected_families = selected_families.iloc[: int(args.max_families)]
    if "materialize" in args.stages:
        for _, family_row in selected_families.iterrows():
            family = family_row.to_dict()
            family_id = str(family["family_id"])
            family_dir = args.output_root / "materialized" / "families" / family_id
            if family_dir.is_dir():
                verification_started = time.perf_counter()
                verification = verify_materialized_family(family_dir)
                verification_seconds = time.perf_counter() - verification_started
                if not verification["passed"]:
                    raise ValueError(f"Existing family {family_id} failed verification")
                run_report["materialized"].append(
                    {
                        "family_id": family_id,
                        "status": "reused_verified",
                        "verification_seconds": verification_seconds,
                        "matrix_total_bytes": int(verification["matrix_total_bytes"]),
                        "process_peak_rss_bytes_after": _process_peak_rss_bytes(),
                    }
                )
                continue
            city = str(family["city_slug"])
            family_views = views.loc[views["family_id"].astype(str).eq(family_id)]
            rejection_path = args.output_root / "rejections" / f"{family_id}.json"
            rejection_payload = (
                json.loads(rejection_path.read_text(encoding="utf-8"))
                if rejection_path.is_file()
                else {"schema": "cle_evrptw_family_rejection_ledger_v1", "attempts": []}
            )
            first_attempt_number = len(rejection_payload["attempts"])
            materialized = False
            for offset in range(args.max_attempts_per_family):
                attempt_number = first_attempt_number + offset
                attempt_family, attempt_views = materialization_attempt_inputs(
                    family,
                    family_views,
                    attempt_number=attempt_number,
                )
                materialization_started = time.perf_counter()
                try:
                    manifest = materialize_family(
                        cle_by_city[city],
                        config=config,
                        profile=profile,
                        family=attempt_family,
                        views=attempt_views,
                        customer_split_path=(
                            args.output_root
                            / "customer_splits"
                            / city
                            / "customer_split_manifest.parquet"
                        ),
                        output_root=args.output_root / "materialized",
                    )
                except Exception as error:  # noqa: BLE001 - persist every failed attempt.
                    rejection = {
                        "family_id": family_id,
                        "city_slug": city,
                        "attempt_number": attempt_number,
                        "attempt_seed": int(attempt_family["materialization_attempt_seed"]),
                        "next_attempt_seed": derive_seed(
                            int(family["family_seed"]),
                            "materialization_attempt",
                            attempt_number + 1,
                        ),
                        "error_type": type(error).__name__,
                        "reason": str(error),
                        "elapsed_seconds": time.perf_counter() - materialization_started,
                    }
                    rejection_payload["family_id"] = family_id
                    rejection_payload["attempts"].append(rejection)
                    _write_json(rejection_path, rejection_payload)
                    run_report["rejected_attempts"].append(rejection)
                    continue
                materialization_seconds = time.perf_counter() - materialization_started
                run_report["materialized"].append(
                    {
                        "family_id": family_id,
                        "status": "materialized",
                        "materialization_attempt_number": attempt_number,
                        "materialization_attempt_seed": int(
                            attempt_family["materialization_attempt_seed"]
                        ),
                        "matrix_total_bytes": manifest["matrix_total_bytes"],
                        "materialization_seconds": materialization_seconds,
                        "terminal_pair_throughput_per_second": (
                            int(manifest["terminal_count"]) ** 2 / materialization_seconds
                        ),
                        "process_peak_rss_bytes_after": _process_peak_rss_bytes(),
                    }
                )
                materialized = True
                break
            if not materialized:
                run_report["unresolved_family_ids"].append(family_id)

    if "verify" in args.stages:
        for _, family_row in selected_families.iterrows():
            family_id = str(family_row["family_id"])
            family_dir = args.output_root / "materialized" / "families" / family_id
            if not family_dir.is_dir():
                run_report["verified"].append(
                    {
                        "family_id": family_id,
                        "passed": False,
                        "errors": ["materialized family directory is missing"],
                        "warnings": [],
                    }
                )
                continue
            verification_started = time.perf_counter()
            verification = verify_materialized_family(family_dir)
            verification["verification_seconds"] = (
                time.perf_counter() - verification_started
            )
            run_report["verified"].append(verification)
            if not verification["passed"]:
                raise ValueError(f"Materialized family {family_id} failed verification")

    run_report["passed"] = not run_report["unresolved_family_ids"] and all(
        bool(item["passed"]) for item in run_report["verified"]
    )
    run_report["performance"] = {
        "run_wall_seconds": time.perf_counter() - run_started,
        "process_peak_rss_bytes": _process_peak_rss_bytes(),
        "rss_semantics": "maximum resident set size for the runner process",
    }
    _write_json(args.output_root / "stage2_run_report.json", run_report)
    print(json.dumps(run_report, indent=2, sort_keys=True, ensure_ascii=False))
    if not run_report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
