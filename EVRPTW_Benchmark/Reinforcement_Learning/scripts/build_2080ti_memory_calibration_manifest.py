#!/usr/bin/env python3
"""Build disposable two-epoch manifests for RTX 2080 Ti memory calibration.

The generated jobs preserve the formal data stream, logical batch, seed,
rollout horizon, validation cohort, and best-of-100 decoding.  Only the
physical microbatch is configurable.  DRL-TS executes one soft and one hard
update so both training stages are covered before the full validation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RQ_ROOT = ROOT / "scripts" / "rq_v1"
SERVER_IDS = ("2080ti_4_1", "2080ti_4_2", "2080ti_3_1")
WAVES = ("cus50", "cus100_g", "cus100_e", "cus100_support")


def _load_jobs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for server_id in SERVER_IDS:
        path = RQ_ROOT / server_id / "jobs.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["calibration_source_server"] = server_id
                rows.append(row)
    if len(rows) != 16 or len({row["job_id"] for row in rows}) != 16:
        raise RuntimeError("expected exactly 16 unique RTX 2080 Ti jobs")
    return rows


def _wave(row: dict[str, Any]) -> str:
    if row["scale"] == "Cus50":
        return "cus50"
    if row["condition"] != "Full-support":
        return "cus100_support"
    if row["representation"] == "E":
        return "cus100_e"
    return "cus100_g"


def _parse_overrides(values: list[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values:
        key, separator, raw = value.rpartition("=")
        if not separator or not key or not raw.isdigit() or int(raw) <= 0:
            raise ValueError(f"invalid batch override: {value!r}")
        parsed[key] = int(raw)
    return parsed


def _batch_for(row: dict[str, Any], overrides: dict[str, int]) -> int:
    method_scale = f"{row['method']}:{row['scale']}"
    batch = int(
        overrides.get(
            row["job_id"],
            overrides.get(method_scale, row["physical_batch_size"]),
        )
    )
    logical = int(row["effective_batch_size"])
    if batch > logical:
        raise ValueError(
            f"physical batch {batch} exceeds logical batch {logical}: "
            f"{row['job_id']}"
        )
    if row["method"] == "terran" and logical % batch:
        raise ValueError(
            f"TERRAN physical batch {batch} must divide logical batch {logical}: "
            f"{row['job_id']}"
        )
    return batch


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _calibration_job(
    source: dict[str, Any], *, batch: int, slot: int
) -> dict[str, Any]:
    row = dict(source)
    original_job_id = str(source["job_id"])
    original_condition = str(source["condition"])
    customers = int(str(source["scale"]).removeprefix("Cus"))
    effective = int(source["effective_batch_size"])
    environments = 2 * effective
    exposures = environments * customers
    tag = f"b{batch}"
    row.update(
        {
            "schema": "drl_rq_memory_calibration_job_v1",
            "job_id": f"memory_calibration__{_slug(original_job_id)}__{tag}",
            "runtime_budget_id": "drl_rq_2080ti_memory_calibration_2epoch_val500_c100_v1",
            "stage": "memory_calibration",
            "condition": (
                f"Memory-calibration__{_slug(original_condition)}__{tag}"
            ),
            "calibration_original_job_id": original_job_id,
            "calibration_original_condition": original_condition,
            "physical_batch_size": batch,
            "training_epochs": 2,
            "minimum_training_epochs": 2,
            "planned_optimizer_updates": 2,
            "target_environments": environments,
            "minimum_target_environments": environments,
            "customer_exposure_budget": exposures,
            "minimum_customer_exposure_budget": exposures,
            "validation_every_epochs": 2,
            "post_minimum_validation_every_epochs": 1,
            "validation_checkpoints": 1,
            "early_stop_patience_validations": 0,
            "early_stop_start_epoch": 1,
            "exposure_checkpoints": [],
            "gpu_hour_checkpoints": [],
            "global_slot": slot,
            "queue_position": 0,
        }
    )
    if row["method"] == "drl_ts":
        row["soft_stage_end_epoch"] = 1
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a disposable 2-epoch + full-validation memory manifest."
    )
    parser.add_argument("--wave", choices=(*WAVES, "all"), required=True)
    parser.add_argument(
        "--method",
        choices=("am_evrptw", "evrptw_rl", "drl_ts", "terran"),
    )
    parser.add_argument(
        "--batch",
        action="append",
        default=[],
        metavar="METHOD:SCALE=N|JOB_ID=N",
        help="Override a method-scale or one exact formal job.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    overrides = _parse_overrides(args.batch)
    selected = [
        row
        for row in _load_jobs()
        if (args.wave == "all" or _wave(row) == args.wave)
        and (args.method is None or row["method"] == args.method)
    ]
    if not selected:
        raise RuntimeError(f"wave contains no jobs: {args.wave}")
    if len(selected) > 4:
        raise RuntimeError("one calibration wave may contain at most four jobs")

    generated = [
        _calibration_job(
            row,
            batch=_batch_for(row, overrides),
            slot=index,
        )
        for index, row in enumerate(selected)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in generated),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema": "drl_rq_memory_calibration_manifest_summary_v1",
                "wave": args.wave,
                "output": str(args.output.resolve()),
                "jobs": [
                    {
                        "formal_job_id": row["calibration_original_job_id"],
                        "slot": row["global_slot"],
                        "physical_batch_size": row["physical_batch_size"],
                    }
                    for row in generated
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
