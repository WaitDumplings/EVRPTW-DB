"""Command-line entry point for CLE-backed Stage-2 preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .artifacts import verify_materialized_family
from .community import build_customer_split
from .config import load_stage2_config
from .materialize import materialize_family
from .planning import build_generation_plan, write_generation_plan
from .profile import load_reference_profile
from .reader import CLEEligibilityError, load_portable_cle


def _write_or_print(payload: object, output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if output is None:
        print(rendered, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cle-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("official", "non_release_pilot"),
        default="official",
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evrptw-stage2",
        description="Validate CLE inputs and plan deterministic EVRPTW Stage-2 artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Audit CLE readiness for Stage 2")
    _add_common(preflight)
    preflight.add_argument("--cities", nargs="+")
    preflight.add_argument("--output", type=Path)

    split = subparsers.add_parser(
        "build-customer-split",
        help="Create the deterministic complete-community customer ledger",
    )
    _add_common(split)
    split.add_argument("--city", required=True)
    split.add_argument("--block-groups", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)

    plan = subparsers.add_parser(
        "plan", help="Write matrix-family and scale-view indices without materializing matrices"
    )
    _add_common(plan)
    plan.add_argument("--output-root", type=Path, required=True)
    plan.add_argument("--cities", nargs="+")
    plan.add_argument("--tracks", nargs="+")
    plan.add_argument("--pilot-families-per-city", type=int)

    materialize = subparsers.add_parser(
        "materialize-family",
        help="Materialize one planned matrix family and all of its scale views",
    )
    _add_common(materialize)
    materialize.add_argument("--profile", type=Path, required=True)
    materialize.add_argument("--plan-root", type=Path, required=True)
    materialize.add_argument("--family-id", required=True)
    materialize.add_argument("--customer-split", type=Path, required=True)
    materialize.add_argument("--output-root", type=Path, required=True)

    verify_family = subparsers.add_parser(
        "verify-family", help="Structurally verify one materialized Stage-2 matrix family"
    )
    verify_family.add_argument("--family-dir", type=Path, required=True)
    verify_family.add_argument("--output", type=Path)
    return parser


def _selected_cities(config, requested: list[str] | None) -> tuple[str, ...]:
    if requested:
        return tuple(dict.fromkeys(requested))
    return (*config.train_cities, config.heldout_city)


def _planned_family(plan_root: Path, family_id: str) -> tuple[dict[str, object], pd.DataFrame]:
    family_parts = [pd.read_parquet(path) for path in sorted(plan_root.rglob("family_index.parquet"))]
    view_parts = [pd.read_parquet(path) for path in sorted(plan_root.rglob("view_index.parquet"))]
    if not family_parts or not view_parts:
        raise FileNotFoundError(f"Plan root has no family/view indices: {plan_root}")
    families = pd.concat(family_parts, ignore_index=True)
    families = families.loc[families["family_id"].astype(str).eq(family_id)]
    families = families.drop_duplicates("family_id")
    if len(families) != 1:
        raise ValueError(f"Expected exactly one planned family {family_id!r}; found {len(families)}")
    views = pd.concat(view_parts, ignore_index=True)
    views = views.loc[views["family_id"].astype(str).eq(family_id)].drop_duplicates("view_id")
    if views.empty:
        raise ValueError(f"No scale views reference planned family {family_id!r}")
    return families.iloc[0].to_dict(), views


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.command == "verify-family":
        report = verify_materialized_family(args.family_dir)
        _write_or_print(report, args.output)
        if not report["passed"]:
            raise SystemExit(2)
        return

    config = load_stage2_config(args.config)

    if args.command == "preflight":
        records: list[dict[str, object]] = []
        failures: list[dict[str, str]] = []
        for city in _selected_cities(config, args.cities):
            try:
                cle = load_portable_cle(args.cle_root, city, mode=args.mode)
                records.append(cle.eligibility_summary())
            except (CLEEligibilityError, KeyError, ValueError) as error:
                failures.append({"city_slug": city, "error": str(error)})
        payload = {
            "schema": "cle_evrptw_stage2_preflight_v1",
            "dataset_id": config.dataset_id,
            "mode": args.mode,
            "passed_city_count": len(records),
            "failed_city_count": len(failures),
            "cities": records,
            "failures": failures,
            "ready": not failures,
            "official_generation_allowed": args.mode == "official" and not failures,
        }
        _write_or_print(payload, args.output)
        if failures:
            raise SystemExit(2)
        return

    if args.command == "build-customer-split":
        cle = load_portable_cle(args.cle_root, args.city, mode=args.mode)
        report = build_customer_split(
            cle,
            block_groups_path=args.block_groups,
            output_dir=args.output_dir,
            split_seed=config.master_seed,
            heldout_fraction=config.heldout_community_fraction,
            partition_version=config.community_partition_version,
        )
        _write_or_print(report, None)
        return

    if args.command == "plan":
        cities = _selected_cities(config, args.cities)
        readiness = []
        for city in cities:
            cle = load_portable_cle(args.cle_root, city, mode=args.mode)
            readiness.append(cle.eligibility_summary())
        non_release = args.mode == "non_release_pilot"
        if non_release and args.pilot_families_per_city is None:
            parser.error("non_release_pilot planning requires --pilot-families-per-city")
        if not non_release and args.pilot_families_per_city is not None:
            parser.error("--pilot-families-per-city is only legal in non_release_pilot mode")
        family_frame, view_frame, registry = build_generation_plan(
            config,
            available_cities=cities,
            pilot_families_per_city=args.pilot_families_per_city,
            include_tracks=args.tracks,
            non_release_pilot=non_release,
        )
        registry["cle_preflight"] = readiness
        written = write_generation_plan(args.output_root, family_frame, view_frame, registry)
        _write_or_print(written, None)
        return

    if args.command == "materialize-family":
        family, views = _planned_family(args.plan_root, args.family_id)
        city = str(family["city_slug"])
        cle = load_portable_cle(args.cle_root, city, mode=args.mode)
        profile = load_reference_profile(args.profile, official=args.mode == "official")
        manifest = materialize_family(
            cle,
            config=config,
            profile=profile,
            family=family,
            views=views,
            customer_split_path=args.customer_split,
            output_root=args.output_root,
        )
        _write_or_print(manifest, None)
        return

    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
