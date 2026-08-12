"""Resumable Stage-2 runner for community splits, plans, families, and QA."""

from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import resource
import sys
import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import pandas as pd

from evrptw_stage2.artifacts import verify_materialized_family
from evrptw_stage2.community import build_customer_split
from evrptw_stage2.config import load_stage2_config
from evrptw_stage2.materialize import materialize_family
from evrptw_stage2.parallel import materialize_family_chunk, verify_family_path
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
        "--mode",
        choices=("official", "research", "non_release_pilot"),
        default="research",
    )
    parser.add_argument("--cities", nargs="+")
    parser.add_argument("--tracks", nargs="+")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--pilot-families-per-city", type=int)
    parser.add_argument("--max-families", type=int)
    parser.add_argument("--family-ids", nargs="+")
    parser.add_argument("--max-attempts-per-family", type=int, default=1)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Spawn-safe family workers. Use 1 for deterministic serial execution.",
    )
    parser.add_argument(
        "--families-per-worker-task",
        type=int,
        default=25,
        help="Single-city families handled by one worker task with topology reuse.",
    )
    parser.add_argument(
        "--allow-memory-oversubscription",
        action="store_true",
        help="Allow --workers above the conservative physical-memory estimate.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Deterministically split selected families across independent runners.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard assigned to this runner.",
    )
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _run_report_path(args: argparse.Namespace) -> Path:
    if int(args.shard_count) == 1:
        return args.output_root / "stage2_run_report.json"
    return args.output_root / (
        f"stage2_run_report.shard-{int(args.shard_index):03d}-of-{int(args.shard_count):03d}.json"
    )


def _load_plan(plan_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    family_parts = [
        pd.read_parquet(path) for path in sorted(plan_root.rglob("family_index.parquet"))
    ]
    view_parts = [pd.read_parquet(path) for path in sorted(plan_root.rglob("view_index.parquet"))]
    if not family_parts or not view_parts:
        raise FileNotFoundError(f"Incomplete plan under {plan_root}")
    return (
        pd.concat(family_parts, ignore_index=True).drop_duplicates("family_id"),
        pd.concat(view_parts, ignore_index=True).drop_duplicates("view_id"),
        json.loads((plan_root / "split_registry.json").read_text(encoding="utf-8")),
    )


def _physical_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _conservative_worker_limit(physical_memory_bytes: int | None) -> int:
    """Budget 5 GiB per routing worker and retain 4 GiB for parent/OS."""

    if physical_memory_bytes is None:
        return 1
    gib = 1024**3
    return max(1, math.floor((physical_memory_bytes - 4 * gib) / (5 * gib)))


def _build_materialization_tasks(
    selected_families: pd.DataFrame,
    views: pd.DataFrame,
    *,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    chunk_size = int(args.families_per_worker_task)
    for city, city_families in selected_families.groupby("city_slug", sort=True):
        ordered = city_families.sort_values("family_id").reset_index(drop=True)
        for chunk_start in range(0, len(ordered), chunk_size):
            chunk = ordered.iloc[chunk_start : chunk_start + chunk_size]
            families_payload = []
            for family in chunk.to_dict("records"):
                family_id = str(family["family_id"])
                family_views = views.loc[views["family_id"].astype(str).eq(family_id)].sort_values(
                    "view_id"
                )
                families_payload.append(
                    {"family": family, "views": family_views.to_dict("records")}
                )
            tasks.append(
                {
                    "chunk_id": f"{city}:{chunk_start // chunk_size:05d}",
                    "config_path": str(args.config.resolve()),
                    "profile_path": str(args.profile.resolve()),
                    "cle_root": str(args.cle_root.resolve()),
                    "mode": args.mode,
                    "city_slug": str(city),
                    "customer_split_path": str(
                        (
                            args.output_root
                            / "customer_splits"
                            / str(city)
                            / "customer_split_manifest.parquet"
                        ).resolve()
                    ),
                    "output_root": str(args.output_root.resolve()),
                    "max_attempts_per_family": int(args.max_attempts_per_family),
                    "families": families_payload,
                }
            )
    return tasks


def _run_bounded_process_tasks(
    executor: ProcessPoolExecutor,
    function: Any,
    tasks: list[Any],
    *,
    max_in_flight: int,
) -> Iterator[tuple[Any, Any]]:
    """Run spawn tasks without placing the complete release in the process queue."""

    task_iterator = iter(tasks)
    pending: dict[Any, Any] = {}

    def submit_next() -> bool:
        try:
            task = next(task_iterator)
        except StopIteration:
            return False
        pending[executor.submit(function, task)] = task
        return True

    for _ in range(min(max_in_flight, len(tasks))):
        submit_next()
    while pending:
        done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
        for future in done:
            task = pending.pop(future)
            result = future.result()
            submit_next()
            yield task, result


def main() -> None:
    run_started = time.perf_counter()
    args = make_parser().parse_args()
    if args.max_attempts_per_family <= 0:
        raise ValueError("--max-attempts-per-family must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.families_per_worker_task <= 0:
        raise ValueError("--families-per-worker-task must be positive")
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, --shard-count)")
    physical_memory_bytes = _physical_memory_bytes()
    safe_worker_limit = _conservative_worker_limit(physical_memory_bytes)
    if args.workers > safe_worker_limit and not args.allow_memory_oversubscription:
        raise ValueError(
            f"--workers={args.workers} exceeds the conservative memory limit "
            f"of {safe_worker_limit}; use fewer workers or explicitly pass "
            "--allow-memory-oversubscription"
        )
    config = load_stage2_config(args.config)
    non_release = args.mode == "non_release_pilot"
    profile = load_reference_profile(args.profile, official=args.mode == "official")
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
        "execution": {
            "workers": int(args.workers),
            "families_per_worker_task": int(args.families_per_worker_task),
            "multiprocessing_start_method": "spawn" if args.workers > 1 else None,
            "physical_memory_bytes": physical_memory_bytes,
            "conservative_worker_limit": safe_worker_limit,
            "memory_model": "5 GiB per worker plus 4 GiB reserved for parent and OS",
            "memory_oversubscription_allowed": bool(args.allow_memory_oversubscription),
            "shard_count": int(args.shard_count),
            "shard_index": int(args.shard_index),
        },
    }

    source_preset = json.loads(args.block_group_preset.read_text(encoding="utf-8"))
    for city in cities:
        cle = load_portable_cle(args.cle_root, city, mode=args.mode)
        run_report["preflight"].append(cle.eligibility_summary())
        if "splits" in args.stages:
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
        del cle
    gc.collect()

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
        _write_json(_run_report_path(args), run_report)
        print(json.dumps(run_report, indent=2, sort_keys=True, ensure_ascii=False))
        return
    families, views, registry = _load_plan(plan_root)
    run_report["generation_plan"] = registry

    selected_families = families.sort_values(["city_slug", "family_id"])
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
    selected_families = selected_families.iloc[int(args.shard_index) :: int(args.shard_count)]
    if args.max_families is not None:
        selected_families = selected_families.iloc[: int(args.max_families)]
    run_report["execution"]["selected_family_count"] = len(selected_families)
    if "materialize" in args.stages and args.workers > 1:
        tasks = _build_materialization_tasks(
            selected_families,
            views,
            args=args,
        )
        run_report["execution"]["materialization_task_count"] = len(tasks)
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
        ) as executor:
            for _, chunk_result in _run_bounded_process_tasks(
                executor,
                materialize_family_chunk,
                tasks,
                max_in_flight=args.workers * 2,
            ):
                run_report["materialized"].extend(chunk_result["materialized"])
                run_report["rejected_attempts"].extend(chunk_result["rejected_attempts"])
                run_report["unresolved_family_ids"].extend(chunk_result["unresolved_family_ids"])
        run_report["materialized"].sort(key=lambda item: str(item["family_id"]))
        run_report["rejected_attempts"].sort(
            key=lambda item: (str(item["family_id"]), int(item["attempt_number"]))
        )
        run_report["unresolved_family_ids"] = sorted(set(run_report["unresolved_family_ids"]))

    if "materialize" in args.stages and args.workers == 1:
        routing_topology_cache = {}
        cached_city: str | None = None
        active_cle = None
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
            if cached_city != city:
                routing_topology_cache.clear()
                cached_city = city
                active_cle = load_portable_cle(args.cle_root, city, mode=args.mode)
            if active_cle is None:
                raise RuntimeError(f"CLE was not loaded for {city}")
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
                        active_cle,
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
                        routing_topology_cache=routing_topology_cache,
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

    if "verify" in args.stages and args.workers > 1:
        existing: list[tuple[str, Path]] = []
        for _, family_row in selected_families.iterrows():
            family_id = str(family_row["family_id"])
            family_dir = args.output_root / "materialized" / "families" / family_id
            if family_dir.is_dir():
                existing.append((family_id, family_dir))
            else:
                run_report["verified"].append(
                    {
                        "family_id": family_id,
                        "passed": False,
                        "errors": ["materialized family directory is missing"],
                        "warnings": [],
                    }
                )
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
        ) as executor:
            verification_tasks = [str(family_dir) for _, family_dir in existing]
            family_id_by_path = {str(family_dir): family_id for family_id, family_dir in existing}
            for family_dir, verification in _run_bounded_process_tasks(
                executor,
                verify_family_path,
                verification_tasks,
                max_in_flight=args.workers * 4,
            ):
                family_id = family_id_by_path[family_dir]
                verification.setdefault("family_id", family_id)
                run_report["verified"].append(verification)
        run_report["verified"].sort(key=lambda item: str(item["family_id"]))
        failed = [item["family_id"] for item in run_report["verified"] if not item["passed"]]
        if failed:
            raise ValueError(f"Materialized families failed verification: {failed}")

    if "verify" in args.stages and args.workers == 1:
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
            verification["verification_seconds"] = time.perf_counter() - verification_started
            run_report["verified"].append(verification)
            if not verification["passed"]:
                raise ValueError(f"Materialized family {family_id} failed verification")

    run_report["passed"] = not run_report["unresolved_family_ids"] and all(
        bool(item["passed"]) for item in run_report["verified"]
    )
    run_report["performance"] = {
        "run_wall_seconds": time.perf_counter() - run_started,
        "process_peak_rss_bytes": _process_peak_rss_bytes(),
        "worker_peak_rss_bytes_max": max(
            (
                int(item.get("worker_peak_rss_bytes_after", 0))
                for item in run_report["materialized"]
            ),
            default=0,
        ),
        "rss_semantics": (
            "maximum resident set size for the runner process; worker maximum "
            "is the largest individual child peak, not their sum"
        ),
    }
    _write_json(_run_report_path(args), run_report)
    print(json.dumps(run_report, indent=2, sort_keys=True, ensure_ascii=False))
    if not run_report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
