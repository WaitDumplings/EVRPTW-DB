"""Resumable Stage-2 runner for community splits, plans, families, and QA."""

from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import resource
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evrptw_stage2.amazon import load_amazon_stage2_artifacts
from evrptw_stage2.artifacts import verify_materialized_family
from evrptw_stage2.community import build_customer_split
from evrptw_stage2.config import load_stage2_config
from evrptw_stage2.metrics import aggregate_phase1_metrics
from evrptw_stage2.parallel import verify_family_path
from evrptw_stage2.planning import (
    build_generation_plan,
    write_generation_plan,
)
from evrptw_stage2.profile import load_reference_profile
from evrptw_stage2.progress import Stage2ProgressWriter, append_json_event
from evrptw_stage2.provenance import resolve_git_provenance
from evrptw_stage2.reader import load_portable_cle
from evrptw_stage2.release_discipline import (
    PilotStopController,
    classify_la_smoke,
)
from evrptw_stage2.subprocess_parallel import run_supervised_materialization

STAGES = ("preflight", "splits", "plan", "materialize", "verify", "metrics")
_ACTIVE_RUN_REPORT: dict[str, Any] | None = None
_OBSERVABILITY_LEDGER_PATH: Path | None = None
_TERMINAL_REPORT_COMMITTED = False
_ACTIVE_RUN_REPORT_PATH: Path | None = None


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
        "--frozen-split-root",
        type=Path,
        help="Reuse approved customer_splits from this dataset root instead of recomputing CBGs.",
    )
    parser.add_argument("--amazon-artifact-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("official", "research", "non_release_pilot"),
        default="research",
    )
    parser.add_argument(
        "--run-discipline",
        choices=("la_smoke", "targeted_profile", "pilot"),
        help="Required materialization stop discipline for non-release runs.",
    )
    parser.add_argument("--cities", nargs="+")
    parser.add_argument("--tracks", nargs="+")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--pilot-families-per-city", type=int)
    parser.add_argument("--max-families", type=int)
    parser.add_argument("--family-ids", nargs="+")
    parser.add_argument("--max-attempts-per-family", type=int, default=4)
    parser.add_argument(
        "--family-wall-timeout-s",
        type=float,
        default=7200.0,
        help="Monotonic wall deadline for each independently killable family attempt.",
    )
    parser.add_argument(
        "--termination-grace-s",
        type=float,
        default=60.0,
        help="Grace after a hard stop before process-group escalation.",
    )
    parser.add_argument(
        "--runner-exit-slack-s",
        type=float,
        default=30.0,
        help="Final SIGKILL/reap budget after the global in-flight grace.",
    )
    parser.add_argument(
        "--stop-policy",
        choices=("abort_all_inflight_after_grace",),
        default="abort_all_inflight_after_grace",
    )
    parser.add_argument(
        "--full-run-approved",
        action="store_true",
        help="Required for any non-pilot corpus run after calibration approval.",
    )
    parser.add_argument(
        "--official-cle-contract",
        choices=("strict_release_v1", "frozen_technical_candidate_v1"),
        default="strict_release_v1",
        help=(
            "Explicit Stage-1 boundary for official runs. The technical-candidate "
            "contract preserves open CLE manual-review labels in provenance."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Spawn-safe family workers. Use 1 for deterministic serial execution.",
    )
    parser.add_argument(
        "--families-per-worker-task",
        type=int,
        default=1,
        help="Frozen at one: every family must own an independently killable session.",
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
    parser.add_argument(
        "--debug-print-full-report",
        action="store_true",
        help="Explicitly print the complete report JSON instead of a concise summary.",
    )
    return parser


def _write_json(path: Path, payload: Any) -> None:
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

def _record_observability_warning(
    error: BaseException,
    *,
    phase: str,
) -> None:
    if _OBSERVABILITY_LEDGER_PATH is None:
        return
    try:
        append_json_event(
            _OBSERVABILITY_LEDGER_PATH,
            {
                "schema": "cle_evrptw_stage2_observability_warning_v1",
                "warning_type": type(error).__name__,
                "message": str(error),
                "phase": phase,
                "terminal_report_committed": bool(_TERMINAL_REPORT_COMMITTED),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
    except OSError:
        pass


def _emit_run_report(
    report: dict[str, Any],
    report_path: Path,
    *,
    debug_full_report: bool = False,
    stream: Any | None = None,
) -> bool:
    """Best-effort stdout output that can never change the persisted outcome."""

    destination = stream if stream is not None else sys.stdout
    if debug_full_report:
        message = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
    else:
        message = "\n".join(
            (
                f"Stage-2 status: {report.get('status')}",
                f"planned={len(report.get('planned_family_ids') or [])} "
                f"materialized={len(report.get('materialized_family_ids') or [])} "
                f"verified={len(report.get('verified_family_ids') or [])}",
                f"report={report_path}",
            )
        )
    try:
        destination.write(message + "\n")
        destination.flush()
    except BrokenPipeError as error:
        _record_observability_warning(error, phase="stdout_summary")
        return False
    return True



def _mark_run_report_failed(
    report: dict[str, Any],
    error: BaseException,
    *,
    last_completed_stage: str | None = None,
) -> dict[str, Any]:
    """Persistable terminal state for verifier and other uncaught exceptions."""

    materialized_ids = sorted(
        {
            str(item["family_id"])
            for item in report.get("materialized", [])
            if item.get("family_id") is not None
        }
    )
    verified_ids = sorted(
        {
            str(item["family_id"])
            for item in report.get("verified", [])
            if item.get("family_id") is not None and item.get("passed") is True
        }
    )
    report.update(
        {
            "status": "failed",
            "passed": False,
            "planned_family_ids": sorted(
                map(str, report.get("planned_family_ids", []))
            ),
            "materialized_family_ids": materialized_ids,
            "verified_family_ids": verified_ids,
            "unresolved_family_ids": sorted(
                map(str, report.get("unresolved_family_ids", []))
            ),
            "exception": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "last_completed_stage": (
                last_completed_stage
                or str(report.get("last_completed_stage", "unknown"))
            ),
        }
    )
    return report


def _uncaught_run_report_hook(
    error_type: type[BaseException],
    error: BaseException,
    traceback: Any,
) -> None:
    if _TERMINAL_REPORT_COMMITTED:
        _record_observability_warning(error, phase="uncaught_after_terminal_commit")
        if not isinstance(error, BrokenPipeError):
            sys.__excepthook__(error_type, error, traceback)
        return
    if _ACTIVE_RUN_REPORT is not None and _ACTIVE_RUN_REPORT_PATH is not None:
        _mark_run_report_failed(_ACTIVE_RUN_REPORT, error)
        _write_json(_ACTIVE_RUN_REPORT_PATH, _ACTIVE_RUN_REPORT)
    sys.__excepthook__(error_type, error, traceback)


def _run_report_path(args: argparse.Namespace) -> Path:
    if int(args.shard_count) == 1:
        return args.output_root / "stage2_run_report.json"
    return args.output_root / (
        f"stage2_run_report.shard-{int(args.shard_index):03d}-of-{int(args.shard_count):03d}.json"
    )


def _json_safe_plan_value(value: Any) -> Any:
    """Normalize Arrow/NumPy plan values before supervisor JSON envelopes."""

    if isinstance(value, np.ndarray):
        return [_json_safe_plan_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _json_safe_plan_value(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_plan_value(item) for item in value]
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    return value


def _reuse_frozen_customer_split(
    frozen_root: Path,
    output_root: Path,
    city: str,
) -> dict[str, Any]:
    source_base = (
        frozen_root / "customer_splits"
        if (frozen_root / "customer_splits").is_dir()
        else frozen_root
    )
    source = source_base / city
    destination = output_root / "customer_splits" / city
    required = (
        "customer_split_report.json",
        "customer_split_manifest.parquet",
        "community_manifest.parquet",
        "community_adjacency.parquet",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Frozen split is incomplete for {city}: missing {missing} under {source}"
        )
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
    destination_missing = [
        name for name in required if not (destination / name).is_file()
    ]
    if destination_missing:
        raise ValueError(
            f"Reused split destination is incomplete for {city}: {destination_missing}"
        )
    report = json.loads(
        (destination / "customer_split_report.json").read_text(encoding="utf-8")
    )
    report["frozen_split_reused"] = True
    report["frozen_split_source"] = str(source.resolve())
    return report


def _load_plan(plan_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    family_parts = [
        pd.read_parquet(path) for path in sorted(plan_root.rglob("family_index.parquet"))
    ]
    view_parts = [pd.read_parquet(path) for path in sorted(plan_root.rglob("view_index.parquet"))]
    if not family_parts or not view_parts:
        raise FileNotFoundError(f"Incomplete plan under {plan_root}")
    families = pd.concat(family_parts, ignore_index=True).drop_duplicates("family_id")
    views = pd.concat(view_parts, ignore_index=True).drop_duplicates("view_id")
    registry = json.loads(
        (plan_root / "split_registry.json").read_text(encoding="utf-8")
    )
    if registry.get("schema") != "cle_evrptw_generation_plan_v3" or "day_type" not in families:
        raise ValueError(
            f"Stale Stage-2 plan under {plan_root}; remove generated plan/materialized "
            "outputs and rebuild with the v2 spatial contract"
        )
    return families, views, registry


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
            for family_raw in chunk.to_dict("records"):
                family = _json_safe_plan_value(family_raw)
                family_id = str(family["family_id"])
                family_views = views.loc[views["family_id"].astype(str).eq(family_id)].sort_values(
                    "view_id"
                )
                families_payload.append(
                    {
                        "family": family,
                        "views": _json_safe_plan_value(
                            family_views.to_dict("records")
                        ),
                    }
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
                    "community_adjacency_path": str(
                        (
                            args.output_root
                            / "customer_splits"
                            / str(city)
                            / "community_adjacency.parquet"
                        ).resolve()
                    ),
                    "amazon_artifact_root": str(args.amazon_artifact_root.resolve()),
                    "amazon_cohort_split_path": str(
                        args.amazon_cohort_split_path.resolve()
                    ),
                    "output_root": str(args.output_root.resolve()),
                    "max_attempts_per_family": int(args.max_attempts_per_family),
                    "code_provenance": dict(args.code_provenance),
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
    supervisor: PilotStopController | None = None,
    poll_interval_s: float = 60.0,
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
    stop_submitting = False
    while pending:
        done, _ = wait(
            tuple(pending),
            timeout=poll_interval_s if supervisor is not None else None,
            return_when=FIRST_COMPLETED,
        )
        if not done:
            if supervisor is not None:
                supervisor.poll(time.perf_counter())
                stop_submitting = supervisor.stopped
            continue
        for future in done:
            task = pending.pop(future)
            result = future.result()
            if supervisor is not None:
                supervisor.observe_chunk(result)
                supervisor.poll(time.perf_counter())
                stop_submitting = supervisor.stopped
            if not stop_submitting:
                submit_next()
            yield task, result


def main() -> None:
    global _ACTIVE_RUN_REPORT, _ACTIVE_RUN_REPORT_PATH
    global _OBSERVABILITY_LEDGER_PATH, _TERMINAL_REPORT_COMMITTED

    _TERMINAL_REPORT_COMMITTED = False

    run_started = time.perf_counter()
    args = make_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    code_provenance = resolve_git_provenance(
        repo_root,
        require_clean=True,
        require_branch="stage2-repair-candidate",
    )
    args.code_provenance = code_provenance
    if args.max_attempts_per_family <= 0:
        raise ValueError("--max-attempts-per-family must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if "materialize" in args.stages and args.families_per_worker_task != 1:
        raise ValueError(
            "family_process_timeout_and_abort_v2 requires "
            "--families-per-worker-task=1"
        )
    for name in (
        "family_wall_timeout_s",
        "termination_grace_s",
        "runner_exit_slack_s",
    ):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
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
    args.amazon_cohort_split_path = args.config.parent.parent / config.raw[
        "amazon_source"
    ]["cohort_split_config"]
    amazon_artifacts = load_amazon_stage2_artifacts(
        args.amazon_artifact_root,
        cohort_split_path=args.amazon_cohort_split_path,
    )
    non_release = args.mode == "non_release_pilot"
    profile = load_reference_profile(args.profile, official=args.mode == "official")
    if not non_release:
        if args.mode != "official":
            raise ValueError(
                "A full Stage-2 corpus may only use mode=official with a promoted "
                "release_calibrated profile; research-mode full generation is forbidden"
            )
        if not args.full_run_approved:
            raise ValueError(
                "Full Stage-2 generation is frozen pending reviewed pilot evidence; "
                "--full-run-approved is required after explicit approval"
            )
        expected_cle_contract = str(
            profile.get("source_contract", {}).get(
                "cle_eligibility_contract", "strict_release_v1"
            )
        )
        if args.official_cle_contract != expected_cle_contract:
            raise ValueError(
                "Official CLE contract differs from the promoted profile: "
                f"runner={args.official_cle_contract}, profile={expected_cle_contract}"
            )
    all_cities = (*config.train_cities, config.heldout_city)
    cities = tuple(
        dict.fromkeys(args.cities or (config.train_cities if non_release else all_cities))
    )
    if non_release and args.pilot_families_per_city is None:
        raise ValueError("non_release_pilot requires --pilot-families-per-city")
    if non_release:
        forbidden_cities = sorted(set(cities) - set(config.train_cities))
        if forbidden_cities:
            raise ValueError(
                f"Calibration pilot must not access held-out cities: {forbidden_cities}"
            )
        if args.tracks is None:
            args.tracks = ["train", "validation"]
        forbidden_tracks = sorted(set(args.tracks) - {"train", "validation"})
        if forbidden_tracks:
            raise ValueError(
                f"Calibration pilot permits only train/validation tracks: {forbidden_tracks}"
            )
    if not non_release and args.pilot_families_per_city is not None:
        raise ValueError("Official generation cannot use pilot family counts")
    if "materialize" in args.stages and non_release and args.run_discipline is None:
        raise ValueError(
            "Non-release materialization requires --run-discipline "
            "la_smoke, targeted_profile, or pilot"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    _OBSERVABILITY_LEDGER_PATH = (
        args.output_root / "stage2_observability_warnings.jsonl"
    )
    run_report: dict[str, Any] = {
        "schema": "cle_evrptw_stage2_run_report_v2",
        "status": "planned",
        "passed": None,
        "last_completed_stage": "runner_initialization",
        "mode": args.mode,
        "code_provenance": code_provenance,
        "cities": list(cities),
        "stages": list(args.stages),
        "preflight": [],
        "amazon_artifact": {
            "schema": amazon_artifacts.manifest["schema"],
            "artifact_id": amazon_artifacts.manifest["artifact_id"],
            "template_count": int(amazon_artifacts.manifest["template_count"]),
            "station_day_count": int(amazon_artifacts.manifest["station_day_count"]),
            "t_env_s": float(amazon_artifacts.manifest["t_env_s"]),
        },
        "splits": {},
        "materialized": [],
        "rejected_attempts": [],
        "unresolved_family_ids": [],
        "verified": [],
        "execution": {
            "workers": int(args.workers),
            "families_per_worker_task": int(args.families_per_worker_task),
            "materialization_process_model": "subprocess_start_new_session_per_family_attempt",
            "materialization_runtime_contract_id": "family_process_timeout_and_abort_v2",
            "verification_multiprocessing_start_method": "spawn" if args.workers > 1 else None,
            "multiprocessing_start_method": "spawn" if args.workers > 1 else None,
            "physical_memory_bytes": physical_memory_bytes,
            "conservative_worker_limit": safe_worker_limit,
            "memory_model": "5 GiB per worker plus 4 GiB reserved for parent and OS",
            "memory_oversubscription_allowed": bool(args.allow_memory_oversubscription),
            "shard_count": int(args.shard_count),
            "shard_index": int(args.shard_index),
        },
    }
    _ACTIVE_RUN_REPORT = run_report
    _ACTIVE_RUN_REPORT_PATH = _run_report_path(args)
    sys.excepthook = _uncaught_run_report_hook
    _write_json(_ACTIVE_RUN_REPORT_PATH, run_report)

    source_preset = json.loads(args.block_group_preset.read_text(encoding="utf-8"))
    for city in cities:
        cle = load_portable_cle(
            args.cle_root,
            city,
            mode=args.mode,
            official_cle_contract=args.official_cle_contract,
        )
        run_report["preflight"].append(cle.eligibility_summary())
        if "splits" in args.stages:
            if args.frozen_split_root is not None:
                report = _reuse_frozen_customer_split(
                    args.frozen_split_root, args.output_root, city
                )
            else:
                state = source_preset["city_to_state"][city]
                state_fips = source_preset["states"][state]
                vintage = int(source_preset["vintage"])
                block_groups = (
                    args.block_group_source_dir
                    / f"tl_{vintage}_{state_fips}_bg.zip"
                )
                split_dir = args.output_root / "customer_splits" / city
                report_path = split_dir / "customer_split_report.json"
                adjacency_path = split_dir / "community_adjacency.parquet"
                if report_path.is_file() and adjacency_path.is_file():
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                elif report_path.exists() or adjacency_path.exists():
                    raise ValueError(
                        f"Incomplete or stale customer split under {split_dir}; "
                        "remove that generated split directory and rebuild it"
                    )
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
    needs_plan = bool({"plan", "materialize", "verify", "metrics"} & set(args.stages))
    if "plan" in args.stages and not (plan_root / "split_registry.json").is_file():
        families, views, registry = build_generation_plan(
            config,
            available_cities=cities,
            pilot_families_per_city=args.pilot_families_per_city,
            include_tracks=args.tracks,
            non_release_pilot=non_release,
        )
        registry["cle_preflight"] = run_report["preflight"]
        registry["code_provenance"] = code_provenance
        write_generation_plan(plan_root, families, views, registry)
    if not needs_plan:
        run_report["status"] = "planned"
        run_report["passed"] = None
        run_report["last_completed_stage"] = "c0_preflight_and_splits"
        _write_json(_run_report_path(args), run_report)
        _emit_run_report(
            run_report,
            _run_report_path(args),
            debug_full_report=bool(args.debug_print_full_report),
        )
        return
    families, views, registry = _load_plan(plan_root)
    plan_provenance = registry.get("code_provenance", {})
    if plan_provenance.get("code_commit") != code_provenance["code_commit"]:
        raise ValueError(
            "Generation plan belongs to a different or unbound code commit; use a new "
            "output root and rebuild the plan"
        )
    run_report["generation_plan"] = registry
    run_report["planned_family_ids"] = sorted(families["family_id"].astype(str))
    run_report["planned_count"] = len(run_report["planned_family_ids"])
    run_report["last_completed_stage"] = "c0_plan"
    _write_json(_run_report_path(args), run_report)

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
    run_report["planned_family_ids"] = sorted(
        selected_families["family_id"].astype(str)
    )
    run_report["planned_count"] = len(run_report["planned_family_ids"])
    run_report["execution"]["selected_family_count"] = len(selected_families)
    if "materialize" in args.stages:
        required_c3_columns = {
            "joint_support_contract_id",
            "selected_depot_id",
            "selected_structure_source_id",
            "capacity_contract_fingerprint",
        }
        missing_c3_columns = sorted(
            required_c3_columns - set(selected_families.columns)
        )
        if missing_c3_columns:
            raise ValueError(
                "C3 joint spatial-support planning must pass before materialization; "
                f"missing plan fields: {missing_c3_columns}"
            )
        incomplete_c3 = selected_families.loc[
            selected_families[list(required_c3_columns)].isna().any(axis=1)
        ]
        if not incomplete_c3.empty:
            raise ValueError(
                "C3 joint spatial-support planning is incomplete for selected families: "
                + repr(sorted(incomplete_c3["family_id"].astype(str).tolist()))
            )
        c3_registry = registry.get("joint_spatial_support", {})
        if args.run_discipline == "pilot" and c3_registry.get("status") != (
            "passed_full_plan"
        ):
            raise ValueError(
                "The 140-family pilot requires a passed_full_plan C3 registry"
            )
    if not ({"materialize", "verify", "metrics"} & set(args.stages)):
        run_report["status"] = "planned"
        run_report["passed"] = None
        run_report["last_completed_stage"] = "c0_plan"
        _write_json(_run_report_path(args), run_report)
        _emit_run_report(
            run_report,
            _run_report_path(args),
            debug_full_report=bool(args.debug_print_full_report),
        )
        return
    generation_stopped = False
    pilot_supervisor: PilotStopController | None = None
    if "materialize" in args.stages and args.run_discipline == "la_smoke":
        if (
            args.workers != 1
            or args.families_per_worker_task != 1
            or args.max_attempts_per_family != 4
            or len(selected_families) != 1
            or set(selected_families["city_slug"].astype(str)) != {"los-angeles"}
        ):
            raise ValueError(
                "la_smoke requires Los Angeles, exactly one selected family, "
                "workers=1, families-per-worker-task=1, and max-attempts=4"
            )
    if "materialize" in args.stages and args.run_discipline == "targeted_profile":
        if (
            args.workers != 1
            or args.families_per_worker_task != 1
            or args.max_attempts_per_family != 4
            or len(selected_families) != 1
        ):
            raise ValueError(
                "targeted_profile requires exactly one selected family, workers=1, "
                "families-per-worker-task=1, and max-attempts=4"
            )
    if "materialize" in args.stages and args.run_discipline == "pilot":
        expected_cities = set(config.train_cities)
        if (
            args.workers != 12
            or args.families_per_worker_task != 1
            or args.max_attempts_per_family != 4
            or len(selected_families) != 140
            or set(selected_families["city_slug"].astype(str)) != expected_cities
            or set(selected_families["track_id"].astype(str)) != {"train", "validation"}
        ):
            raise ValueError(
                "pilot discipline requires the frozen 140-family ten-city "
                "train/validation plan with workers=12, task=1, attempts=4"
            )
        pilot_supervisor = PilotStopController(
            planned_family_count=len(selected_families),
            started_monotonic=run_started,
        )
    progress_writer = Stage2ProgressWriter(
        args.output_root,
        list(map(str, selected_families["family_id"].tolist())),
    )
    supervised_materialization = "materialize" in args.stages
    if supervised_materialization:
        run_report["status"] = "materializing"
        run_report["passed"] = None
        run_report["last_completed_stage"] = "materialization_preflight"
        _write_json(_run_report_path(args), run_report)
        tasks = _build_materialization_tasks(selected_families, views, args=args)
        run_report["execution"]["materialization_task_count"] = len(tasks)
        supervised = run_supervised_materialization(
            tasks,
            workers=args.workers,
            max_attempts_per_family=args.max_attempts_per_family,
            family_wall_timeout_s=args.family_wall_timeout_s,
            termination_grace_s=args.termination_grace_s,
            runner_exit_slack_s=args.runner_exit_slack_s,
            pilot_controller=pilot_supervisor,
            python_executable=sys.executable,
            working_directory=Path(__file__).resolve().parents[1],
            progress_callback=progress_writer.apply_supervisor_event,
        )
        run_report["runtime_contract"] = supervised["runtime_contract"]
        run_report["runtime_run_id"] = supervised["run_id"]
        run_report["materialized"].extend(supervised["materialized"])
        run_report["rejected_attempts"].extend(supervised["rejected_attempts"])
        run_report["timed_out_attempts"] = supervised["timed_out_attempts"]
        run_report["aborted_attempts"] = supervised["aborted_attempts"]
        run_report["unresolved_family_ids"].extend(
            supervised["unresolved_family_ids"]
        )
        run_report["not_started_family_ids"] = supervised["not_started_family_ids"]
        run_report["hard_stop_triggered"] = bool(
            supervised["runtime_contract"]["hard_stop_triggered"]
        )
        generation_stopped = run_report["hard_stop_triggered"]
        if pilot_supervisor is not None:
            run_report["run_discipline"] = pilot_supervisor.report()
            generation_stopped = generation_stopped or pilot_supervisor.stopped
        run_report["materialized"].sort(key=lambda item: str(item["family_id"]))
        run_report["rejected_attempts"].sort(
            key=lambda item: (str(item["family_id"]), int(item["attempt_number"]))
        )
        run_report["unresolved_family_ids"] = sorted(
            set(run_report["unresolved_family_ids"])
        )
        run_report["unresolved_count"] = len(run_report["unresolved_family_ids"])
        run_report["materialized_family_ids"] = sorted(
            {
                str(item["family_id"])
                for item in run_report["materialized"]
                if item.get("family_id") is not None
            }
        )
        run_report["last_completed_stage"] = "materialization"
        _write_json(_run_report_path(args), run_report)

    if "materialize" in args.stages and args.run_discipline == "la_smoke":
        successful = [
            item
            for item in run_report["materialized"]
            if item.get("status") in {"materialized", "reused_verified"}
        ]
        if len(successful) == 1 and not run_report["unresolved_family_ids"]:
            item = successful[0]
            smoke = classify_la_smoke(
                terminal_selection_s=float(
                    item.get("stage_timings_seconds", {}).get(
                        "terminal_selection", float("inf")
                    )
                ),
                family_total_s=float(item.get("materialization_seconds", float("inf"))),
            )
        else:
            smoke = {
                "schema": "cle_evrptw_la_smoke_stop_rule_v1",
                "status": "RED",
                "pilot_allowed": False,
                "exact_performance_optimization_required": False,
                "reason": "smoke family did not materialize exactly once",
            }
        run_report["run_discipline"] = smoke
        generation_stopped = not bool(smoke["pilot_allowed"])

    if "verify" in args.stages and not generation_stopped:
        run_report["status"] = "verifying"
        run_report["passed"] = None
        _write_json(_run_report_path(args), run_report)

    if "verify" in args.stages and args.workers > 1 and not generation_stopped:
        existing: list[tuple[str, Path]] = []
        for _, family_row in selected_families.iterrows():
            family_id = str(family_row["family_id"])
            family_dir = args.output_root / "materialized" / "families" / family_id
            if family_dir.is_dir():
                existing.append((family_id, family_dir))
            else:
                verification = {
                    "family_id": family_id,
                    "passed": False,
                    "errors": ["materialized family directory is missing"],
                    "warnings": [],
                }
                run_report["verified"].append(verification)
                progress_writer.record_verification(family_id, passed=False)
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
                progress_writer.record_verification(
                    family_id, passed=bool(verification["passed"])
                )
        run_report["verified"].sort(key=lambda item: str(item["family_id"]))
        failed = [item["family_id"] for item in run_report["verified"] if not item["passed"]]
        if failed:
            run_report["last_completed_stage"] = "verification"
            _mark_run_report_failed(
                run_report,
                ValueError(f"Materialized families failed verification: {failed}"),
                last_completed_stage="verification",
            )
            progress_writer.finalize(passed=False)
            _write_json(_run_report_path(args), run_report)
            raise ValueError(f"Materialized families failed verification: {failed}")

    if "verify" in args.stages and args.workers == 1 and not generation_stopped:
        for _, family_row in selected_families.iterrows():
            family_id = str(family_row["family_id"])
            family_dir = args.output_root / "materialized" / "families" / family_id
            if not family_dir.is_dir():
                verification = {
                    "family_id": family_id,
                    "passed": False,
                    "errors": ["materialized family directory is missing"],
                    "warnings": [],
                }
                run_report["verified"].append(verification)
                progress_writer.record_verification(family_id, passed=False)
                continue
            verification_started = time.perf_counter()
            verification = verify_materialized_family(family_dir)
            verification["verification_seconds"] = time.perf_counter() - verification_started
            run_report["verified"].append(verification)
            progress_writer.record_verification(
                family_id, passed=bool(verification["passed"])
            )
            if not verification["passed"]:
                run_report["last_completed_stage"] = "verification"
                _mark_run_report_failed(
                    run_report,
                    ValueError(
                        f"Materialized family {family_id} failed verification"
                    ),
                    last_completed_stage="verification",
                )
                progress_writer.finalize(passed=False)
                _write_json(_run_report_path(args), run_report)
                raise ValueError(f"Materialized family {family_id} failed verification")

    run_report["passed"] = (
        not generation_stopped
        and not run_report["unresolved_family_ids"]
        and all(bool(item["passed"]) for item in run_report["verified"])
    )
    run_report["status"] = "passed" if run_report["passed"] else "failed"
    run_report["verified_family_ids"] = sorted(
        {
            str(item["family_id"])
            for item in run_report["verified"]
            if item.get("family_id") is not None and item.get("passed") is True
        }
    )
    run_report["planned_count"] = len(run_report["planned_family_ids"])
    run_report["materialized_count"] = len(run_report["materialized_family_ids"])
    run_report["verified_count"] = len(run_report["verified_family_ids"])
    run_report["last_completed_stage"] = (
        "verification" if "verify" in args.stages else "materialization"
    )
    if "metrics" in args.stages and not generation_stopped:
        if int(args.shard_count) != 1:
            raise ValueError(
                "The metrics stage requires a complete unsharded output; run "
                "scripts/aggregate_phase1_metrics.py after all shards finish"
            )
        run_report["phase1_metrics"] = aggregate_phase1_metrics(args.output_root)
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
    run_manifest = {
        "schema": "cle_evrptw_stage2_run_manifest_v2",
        "config_schema": config.schema,
        "profile_schema": profile["schema"],
        "mode": args.mode,
        "official_cle_contract": args.official_cle_contract,
        "code_provenance": code_provenance,
        "max_attempts_per_family": int(args.max_attempts_per_family),
        "run_discipline": args.run_discipline,
        "selected_cities": list(cities),
        "selected_tracks": list(args.tracks or []),
        "baseline_solver": {
            "run": False,
            "solver_version": None,
            "time_budget_s": None,
            "soc_tolerance_kwh": None,
        },
        "instance_hash_excludes": [
            "baseline_solver.solver_version",
            "baseline_solver.time_budget_s",
            "baseline_solver.soc_tolerance_kwh",
            "performance",
        ],
        "performance": run_report["performance"],
    }
    report_path = _run_report_path(args)
    run_report["terminal_report_committed"] = True
    run_report["terminal_report_committed_at"] = datetime.now(
        timezone.utc
    ).isoformat()
    progress_writer.finalize(passed=bool(run_report["passed"]))
    if int(args.shard_count) == 1:
        _write_json(args.output_root / "run_manifest.json", run_manifest)
    _write_json(report_path, run_report)
    _TERMINAL_REPORT_COMMITTED = True
    _emit_run_report(
        run_report,
        report_path,
        debug_full_report=bool(args.debug_print_full_report),
    )
    if not run_report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
