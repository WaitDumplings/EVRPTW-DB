#!/usr/bin/env python3
"""Phase C3: freeze a feasible depot x Amazon structure-source pair per family."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from evrptw_stage2.amazon import load_amazon_stage2_artifacts
from evrptw_stage2.profile import load_reference_profile
from evrptw_stage2.provenance import resolve_git_provenance
from evrptw_stage2.reader import load_portable_cle
from evrptw_stage2.road_state import build_family_road_state
from evrptw_stage2.routing import PhysicalRoadNetwork
from evrptw_stage2.selection import (
    JOINT_SUPPORT_CONTRACT_ID,
    assess_joint_spatial_support_pair,
    depot_candidate_order,
    prepare_customer_split_roster,
)
from evrptw_stage2.spatial_activation import SpatialActivationError


C3_SCHEMA = "cle_evrptw_phase_c3_joint_spatial_support_v1"
AGGREGATE_FAILURE_CODES = {
    "PF2_STRUCTURE_UNSUPPORTED",
    "PF2_STRUCTURE_UPSCALING_FORBIDDEN",
    "SPATIAL_QUOTA_UNSUPPORTED",
    "TERRITORY_TOO_SMALL",
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_family_parts(plan_root: Path) -> list[tuple[Path, pd.DataFrame]]:
    parts = [
        (path, pd.read_parquet(path))
        for path in sorted(plan_root.rglob("family_index.parquet"))
    ]
    if not parts:
        raise FileNotFoundError(f"No family plan under {plan_root}")
    return parts


def _reason(error: Exception) -> tuple[str, str]:
    code = str(getattr(error, "code", type(error).__name__))
    return code, str(error)


def _validate_c2_evidence(
    args: argparse.Namespace,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if getattr(args, "mode", None) == "official":
        profile = load_reference_profile(args.profile, official=True)
        promotion = dict(profile.get("acceptance_promotion") or {})
        if (
            promotion.get("schema")
            != "evrptw_profile_acceptance_promotion_v2_no_hash"
            or promotion.get("construct_acceptance_schema")
            != "stage2_acceptance_v3_construct_valid"
            or promotion.get("ev_activity_audit_schema")
            != "stage2_primary_view_ev_activity_audit_v1"
            or promotion.get("hash_validation_performed") is not False
            or not str(promotion.get("advisor_signoff_id", ""))
        ):
            raise ValueError(
                "Official C3 requires a no-hash profile promoted from acceptance v3 "
                "and the primary-view EV activity gate"
            )
        return {
            "mode": "profile_bound_stage2_acceptance_v3",
            "construct_acceptance_schema": promotion[
                "construct_acceptance_schema"
            ],
            "construct_acceptance_code_commit": promotion.get(
                "construct_acceptance_code_commit"
            ),
            "ev_activity_audit_schema": promotion["ev_activity_audit_schema"],
            "ev_activity_code_commit": promotion.get("ev_activity_code_commit"),
            "advisor_signoff_id": promotion["advisor_signoff_id"],
            "hash_validation_performed": False,
        }
    if args.c2_report is None:
        raise ValueError("Non-official C3 requires --c2-report")
    c2 = json.loads(args.c2_report.read_text(encoding="utf-8"))
    if c2.get("schema") != "cle_evrptw_phase_c2_release_preflight_v1":
        raise ValueError("C3 requires the frozen C2 release-preflight schema")
    if not c2.get("passed"):
        raise ValueError("C3 is forbidden because C2 did not pass")
    baseline_commit = str(c2.get("code_provenance", {}).get("code_commit", ""))
    current_commit = str(provenance["code_commit"])
    if baseline_commit == current_commit:
        return {
            "mode": "same_commit",
            "c2_commit": baseline_commit,
            "current_commit": current_commit,
        }
    if args.c0_comparison is None:
        raise ValueError(
            "Inherited C2 evidence requires --c0-comparison bound to this fresh root"
        )
    comparison = json.loads(args.c0_comparison.read_text(encoding="utf-8"))
    required_counts = {
        "family_count_140",
        "ten_city_split_membership",
        "twenty_city_track_slots_are_5_to_2",
        "view_count_2590",
    }
    if (
        comparison.get("schema")
        != "cle_evrptw_stage2_c0_exact_comparison_v1"
        or not comparison.get("passed")
        or not required_counts.issubset(comparison.get("fixed_counts", {}))
        or not all(
            bool(comparison["fixed_counts"][key]) for key in required_counts
        )
    ):
        raise ValueError("C3 inherited C2 requires an exact passing C0 comparison")
    candidate_root = Path(str(comparison["candidate_root"]))
    if not candidate_root.is_absolute():
        candidate_root = (Path.cwd() / candidate_root).resolve()
    if candidate_root != args.plan_root.parent.resolve():
        raise ValueError(
            "C0 comparison candidate root does not match the current C3 plan root"
        )
    ancestry = subprocess.run(
        [
            "git",
            "-C",
            str(Path(__file__).resolve().parents[2]),
            "merge-base",
            "--is-ancestor",
            baseline_commit,
            current_commit,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestry.returncode:
        raise ValueError("Inherited C2 commit is not an ancestor of current commit")
    return {
        "mode": "reviewer_authorized_frozen_c2_inheritance",
        "basis": (
            "CLE_v2, C0 customer split and R2-v2 certificates are frozen; "
            "fresh C0 plan/split/view comparison is exact"
        ),
        "c2_commit": baseline_commit,
        "current_commit": current_commit,
        "c0_comparison": str(args.c0_comparison.resolve()),
    }


def _apply_updates(
    plan_root: Path,
    parts: list[tuple[Path, pd.DataFrame]],
    updates: dict[str, dict[str, Any]],
    *,
    c3_report: Path,
    code_provenance: dict[str, Any],
    full_plan: bool,
) -> None:
    columns = sorted({key for values in updates.values() for key in values})
    for path, original in parts:
        frame = original.copy()
        for column in columns:
            if column not in frame:
                frame[column] = pd.Series([None] * len(frame), dtype=object)
        changed = False
        for index, family_id in frame["family_id"].astype(str).items():
            values = updates.get(family_id)
            if values is None:
                continue
            changed = True
            for key, value in values.items():
                frame.at[index, key] = value
        if changed:
            _atomic_parquet(path, frame)

    registry_path = plan_root / "split_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["joint_spatial_support"] = {
        "contract_id": JOINT_SUPPORT_CONTRACT_ID,
        "schema": C3_SCHEMA,
        "status": "passed_full_plan" if full_plan else "passed_targeted_gate_only",
        "covered_family_count": len(updates),
        "report": str(c3_report.resolve()),
        "code_provenance": code_provenance,
    }
    _atomic_json(registry_path, registry)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[2]
    provenance = resolve_git_provenance(
        repo_root,
        require_clean=True,
        require_branch="stage2-repair-candidate",
    )
    c2_evidence_binding = _validate_c2_evidence(args, provenance)

    parts = _load_family_parts(args.plan_root)
    families = pd.concat([frame for _, frame in parts], ignore_index=True)
    families = families.drop_duplicates("family_id").sort_values(
        ["city_slug", "family_id"]
    )
    requested = set(map(str, args.family_ids or []))
    if requested:
        missing = sorted(requested - set(families["family_id"].astype(str)))
        if missing:
            raise ValueError(f"Requested C3 families are absent from plan: {missing}")
        families = families.loc[families["family_id"].astype(str).isin(requested)]
    requested_cities = set(map(str, args.cities or []))
    if requested_cities:
        missing_cities = sorted(
            requested_cities - set(families["city_slug"].astype(str))
        )
        if missing_cities:
            raise ValueError(f"Requested C3 cities are absent from plan: {missing_cities}")
        families = families.loc[
            families["city_slug"].astype(str).isin(requested_cities)
        ]
    full_plan = len(families) == sum(len(frame) for _, frame in parts)
    if not full_plan and not args.targeted_gate:
        raise ValueError("Partial C3 requires --targeted-gate")

    profile = load_reference_profile(args.profile, official=args.mode == "official")
    amazon = load_amazon_stage2_artifacts(
        args.amazon_artifact_root,
        cohort_split_path=args.amazon_cohort_split,
    )
    updates: dict[str, dict[str, Any]] = {}
    family_reports: list[dict[str, Any]] = []
    topology_cache: dict[str, PhysicalRoadNetwork] = {}
    cle_cache: dict[str, Any] = {}
    speeds_cache: dict[str, pd.DataFrame] = {}
    depots_cache: dict[str, pd.DataFrame] = {}
    customer_roster_cache: dict[str, pd.DataFrame] = {}

    for family_row in families.to_dict("records"):
        family_id = str(family_row["family_id"])
        city = str(family_row["city_slug"])
        family_started = time.perf_counter()
        cle = cle_cache.get(city)
        if cle is None:
            cle = load_portable_cle(
                args.cle_root,
                city,
                mode=args.mode,
                official_cle_contract=args.official_cle_contract,
            )
            cle_cache[city] = cle
            speeds_cache[city] = pd.read_parquet(cle.speeds_path)
            depots_cache[city] = cle.read_depots().reset_index(drop=True)
            customer_roster_cache[city] = prepare_customer_split_roster(
                cle,
                str(
                    args.customer_split_root
                    / city
                    / "customer_split_manifest.parquet"
                ),
            )
        speeds = speeds_cache[city]
        road_state, _ = build_family_road_state(
            speeds,
            day_type=str(family_row["day_type"]),
            road_state_seed=int(family_row["road_state_seed"]),
            profile=profile,
        )
        cached = topology_cache.get(city)
        if cached is None:
            network = PhysicalRoadNetwork.from_files(cle.graph_path, road_state, profile)
            topology_cache[city] = network
        else:
            network = cached.with_road_state(road_state, profile)

        depot_track = str(
            profile.get("stage2_spatial", {}).get("depot_track", "practical")
        )
        depot_candidates, _ = depot_candidate_order(
            depots_cache[city],
            seed=int(family_row["depot_seed"]),
            track=depot_track,
        )
        source_pool = amazon.pool_for_track(str(family_row["track_id"]))
        source_candidates = amazon.structure_source_candidates(
            day_type=str(family_row["day_type"]),
            customer_count=int(family_row["parent_customer_count"]),
            seed=int(family_row["customer_superset_seed"]),
            pool=source_pool,
            track_id=str(family_row["track_id"]),
            allow_composite=str(family_row["parent_scale_id"]) == "cus2000",
        )
        pair_count = len(depot_candidates) * len(source_candidates)
        rejected = Counter()
        attempted_pairs: list[dict[str, Any]] = []
        aggregate_pass_count = 0
        exact_pass_count = 0
        selected: dict[str, Any] | None = None
        # Road-state weights change for every family. Cache only within this
        # family so adjacency and depot-star results never cross road states.
        family_adjacency_cache: dict[str, pd.DataFrame] = {}
        depot_star_cache: dict[
            tuple[str, str], tuple[pd.DataFrame, dict[str, Any]]
        ] = {}

        for depot_rank, depot in enumerate(depot_candidates):
            for source_rank, source in enumerate(source_candidates):
                depot_id = str(depot["candidate_id"])
                source_ids = list(map(str, source["structure_source_ids"]))
                try:
                    support = assess_joint_spatial_support_pair(
                        cle,
                        family=family_row,
                        selected_depot_id=depot_id,
                        selected_structure_source_ids=source_ids,
                        customer_split_path=str(
                            args.customer_split_root
                            / city
                            / "customer_split_manifest.parquet"
                        ),
                        community_adjacency_path=str(
                            args.customer_split_root
                            / city
                            / "community_adjacency.parquet"
                        ),
                        profile=profile,
                        network=network,
                        amazon=amazon,
                        community_adjacency_cache=family_adjacency_cache,
                        customer_split_roster=customer_roster_cache[city],
                        depots=depots_cache[city],
                        depot_star_cache=depot_star_cache,
                    )
                except Exception as error:
                    code, detail = _reason(error)
                    rejected[code] += 1
                    aggregate_passed = not (
                        isinstance(error, SpatialActivationError)
                        and code in AGGREGATE_FAILURE_CODES
                    )
                    if aggregate_passed:
                        aggregate_pass_count += 1
                    attempted_pairs.append(
                        {
                            "depot_rank": depot_rank,
                            "structure_source_rank": source_rank,
                            "depot_id": depot_id,
                            "structure_source_ids": source_ids,
                            "aggregate_gate_passed": aggregate_passed,
                            "exact_gate_passed": False,
                            "reason_code": code,
                            "reason": detail,
                            "diagnostics": getattr(error, "diagnostics", {}),
                        }
                    )
                    continue
                aggregate_pass_count += 1
                exact_pass_count += 1
                selected = support
                attempted_pairs.append(
                    {
                        "depot_rank": depot_rank,
                        "structure_source_rank": source_rank,
                        "depot_id": depot_id,
                        "structure_source_ids": source_ids,
                        "aggregate_gate_passed": True,
                        "exact_gate_passed": True,
                        "reason_code": None,
                    }
                )
                break
            if selected is not None:
                break

        family_report = {
            "family_id": family_id,
            "city_slug": city,
            "track_id": str(family_row["track_id"]),
            "day_type": str(family_row["day_type"]),
            "parent_scale_id": str(family_row["parent_scale_id"]),
            "candidate_depot_count": len(depot_candidates),
            "candidate_structure_source_count": len(source_candidates),
            "joint_pair_count": pair_count,
            "aggregate_gate_pass_count": aggregate_pass_count,
            "exact_gate_pass_count": exact_pass_count,
            "rejected_pair_reason_counts": dict(sorted(rejected.items())),
            "attempted_pairs": attempted_pairs,
            "selected": selected,
            "elapsed_seconds": time.perf_counter() - family_started,
        }
        family_reports.append(family_report)
        if args.progress_output is not None:
            _atomic_json(
                args.progress_output,
                {
                    "schema": "cle_evrptw_c3_progress_v1",
                    "planned": int(len(families)),
                    "completed": int(len(family_reports)),
                    "last_completed_family_id": family_id,
                    "city_slug": city,
                    "failed": selected is None,
                    "updated_monotonic_s": time.monotonic(),
                },
            )
        if selected is None:
            report = {
                "schema": C3_SCHEMA,
                "code_provenance": provenance,
                "c2_evidence_binding": c2_evidence_binding,
                "status": "failed",
                "passed": False,
                "failure_family_id": family_id,
                "families": family_reports,
                "elapsed_seconds": time.perf_counter() - started,
            }
            _atomic_json(args.output, report)
            raise RuntimeError(
                f"C3_PLANNING_HARD_FAIL: no legal depot x structure pair for {family_id}"
            )

        updates[family_id] = {
            "joint_support_contract_id": JOINT_SUPPORT_CONTRACT_ID,
            "candidate_depot_count": len(depot_candidates),
            "candidate_structure_source_count": len(source_candidates),
            "joint_pair_count": pair_count,
            "aggregate_gate_pass_count": aggregate_pass_count,
            "exact_gate_pass_count": exact_pass_count,
            "selected_depot_id": selected["selected_depot_id"],
            "selected_structure_source_id": selected[
                "selected_structure_source_id"
            ],
            "required_decile_counts": selected["required_decile_counts"],
            "available_decile_counts": selected["available_decile_counts"],
            "capacity_contract_fingerprint": selected[
                "capacity_contract_fingerprint"
            ],
            "rejected_pair_reason_counts": json.dumps(
                dict(sorted(rejected.items())),
                sort_keys=True,
                separators=(",", ":"),
            ),
        }

    report = {
        "schema": C3_SCHEMA,
        "code_provenance": provenance,
        "c2_evidence_binding": c2_evidence_binding,
        "status": "applying_plan" if not args.report_only else "passed_report_only",
        "passed": None if not args.report_only else True,
        "full_plan": full_plan,
        "covered_family_count": len(updates),
        "families": family_reports,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output, report)
    if not args.report_only:
        try:
            _apply_updates(
                args.plan_root,
                parts,
                updates,
                c3_report=args.output,
                code_provenance=provenance,
                full_plan=full_plan,
            )
        except BaseException as error:
            report["status"] = "failed"
            report["passed"] = False
            report["exception"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            _atomic_json(args.output, report)
            raise
        report["status"] = "passed"
        report["passed"] = True
        _atomic_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cle-root", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--customer-split-root", type=Path, required=True)
    parser.add_argument("--amazon-artifact-root", type=Path, required=True)
    parser.add_argument("--amazon-cohort-split", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--c2-report", type=Path)
    parser.add_argument("--c0-comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("official", "research", "non_release_pilot"),
        default="research",
    )
    parser.add_argument("--family-ids", nargs="*")
    parser.add_argument("--cities", nargs="*")
    parser.add_argument(
        "--official-cle-contract",
        choices=("strict_release_v1", "frozen_technical_candidate_v1"),
        default="strict_release_v1",
    )
    parser.add_argument("--progress-output", type=Path)
    parser.add_argument("--targeted-gate", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(
        "C3 joint support: "
        f"status={report['status']} passed={report['passed']} "
        f"covered={report['covered_family_count']}"
    )
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
