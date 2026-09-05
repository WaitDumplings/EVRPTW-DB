#!/usr/bin/env python3
"""Execute one disposable RTX 2080 Ti memory-calibration wave."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from . import drl_job_runtime as runtime


ROOT = Path(__file__).resolve().parents[3]
GPU_PATTERN = "RTX 2080 Ti"


def resolve_dataset_root(repo: Path) -> Path:
    requested = os.environ.get("EVRPTW_DATASET_ROOT")
    if requested:
        path = Path(requested)
        return (path if path.is_absolute() else repo / path).resolve()
    search_roots = [repo]
    restore = os.environ.get("EVRPTW_RESTORE_ROOT")
    if restore:
        restore_path = Path(restore)
        search_roots.append(
            (restore_path if restore_path.is_absolute() else repo / restore_path).resolve()
        )
    relative_candidates = (
        Path("EVRPTW_Dataset/Instances_v2/us_11city"),
        Path(
            "EVRPTW_Dataset/Instances_v2/"
            "us_11city_full_clean_v7_bbde5db_20260823"
        ),
    )
    for search_root in search_roots:
        for relative in relative_candidates:
            candidate = search_root / relative
            if (candidate / "generation_plan/core/train/view_index.parquet").is_file():
                return candidate.resolve()
    return (repo / relative_candidates[0]).resolve()


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    jobs = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    slots = [int(job["global_slot"]) for job in jobs]
    if not jobs or len(jobs) > 4 or len(slots) != len(set(slots)):
        raise ValueError("calibration manifest requires one to four unique slots")
    for job in jobs:
        required = {
            "stage": "memory_calibration",
            "training_epochs": 2,
            "minimum_training_epochs": 2,
            "validation_views": 500,
            "validation_candidate_count": 100,
            "validation_every_epochs": 2,
        }
        for key, expected in required.items():
            if job.get(key) != expected:
                raise ValueError(
                    f"unsafe calibration manifest value {key}={job.get(key)!r}; "
                    f"expected {expected!r}"
                )
    return jobs


def _prepare_short_streams(
    jobs: list[dict[str, Any]], output_root: Path
) -> None:
    stream_root = output_root / "streams"
    stream_root.mkdir(parents=True, exist_ok=True)
    cached: dict[tuple[str, int], Path] = {}
    for job in jobs:
        source = Path(job["training_stream_path"])
        if not source.is_absolute():
            source = ROOT / source
        expected_rows = 2 * int(job["effective_batch_size"])
        key = (str(source.resolve()), expected_rows)
        target = cached.get(key)
        if target is None:
            target = stream_root / f"stream_{len(cached):02d}_{expected_rows}.parquet"
            first_batch = next(
                pq.ParquetFile(source).iter_batches(batch_size=expected_rows)
            )
            table = pa.Table.from_batches([first_batch])
            if table.num_rows != expected_rows:
                raise RuntimeError(
                    f"short stream has {table.num_rows} rows; expected {expected_rows}"
                )
            pq.write_table(table, target)
            cached[key] = target
        job["training_stream_path"] = str(target.resolve())


def _run_one(
    job: dict[str, Any],
    *,
    dataset: Path,
    output_root: Path,
    commit: str,
    results: list[dict[str, Any]],
) -> None:
    slot = int(job["global_slot"])
    out = output_root / job["job_id"]
    out.mkdir(parents=True, exist_ok=True)
    context = {
        "repo": ROOT,
        "dataset": dataset,
        "output": output_root,
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        "commit": commit,
    }
    command = runtime.training_command(job, context, out, resume=False)
    (out / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(slot)
    started = time.perf_counter()
    with (out / "stdout.log").open("w", encoding="utf-8") as stdout, (
        out / "stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        peak_process_gpu = 0
        peak_process_cpu = 0
        while process.poll() is None:
            peak_process_gpu = max(
                peak_process_gpu, runtime.process_gpu_memory_bytes(process.pid)
            )
            peak_process_cpu = max(
                peak_process_cpu, runtime.process_rss_bytes(process.pid)
            )
            time.sleep(0.2)
        returncode = process.wait()
    training_path = out / "training_result.json"
    training = (
        json.loads(training_path.read_text(encoding="utf-8"))
        if training_path.is_file()
        else {}
    )
    validation_path = out / "validation_history.jsonl"
    validation_rows = (
        [
            json.loads(line)
            for line in validation_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if validation_path.is_file()
        else []
    )
    validation_candidates = (
        validation_rows[0].get(
            "candidate_count",
            validation_rows[0].get(
                "eval_n_traj", job.get("validation_candidate_count", 0)
            )
        )
        if validation_rows
        else 0
    )
    passed = bool(
        returncode == 0
        and training.get("status") == "passed"
        and int(training.get("completed_training_epochs", 0)) == 2
        and len(validation_rows) == 1
        and int(validation_rows[0].get("instances", 0)) == 500
        and int(validation_candidates) == 100
        and validation_rows[0].get("verifier_summary_passed") is not None
    )
    result = {
        "schema": "drl_rq_2080ti_memory_calibration_result_v1",
        "job_id": job["job_id"],
        "formal_job_id": job["calibration_original_job_id"],
        "method": job["method"],
        "scale": job["scale"],
        "representation": job["representation"],
        "condition": job["calibration_original_condition"],
        "slot": slot,
        "physical_batch_size": int(job["physical_batch_size"]),
        "effective_batch_size": int(job["effective_batch_size"]),
        "returncode": returncode,
        "passed": passed,
        "peak_process_gpu_memory_bytes": peak_process_gpu,
        "peak_process_cpu_memory_bytes": peak_process_cpu,
        "peak_pytorch_allocated_bytes": training.get("peak_gpu_memory_bytes"),
        "wall_time_s": time.perf_counter() - started,
        "training_epochs_completed": training.get("completed_training_epochs"),
        "validation_rows": len(validation_rows),
        "validation_instances": (
            validation_rows[0].get("instances") if validation_rows else None
        ),
        "validation_candidates": validation_candidates or None,
        "validation_verifier_summary_passed": (
            validation_rows[0].get("verifier_summary_passed")
            if validation_rows
            else None
        ),
        "output_dir": str(out),
    }
    runtime.atomic_json(out / "memory_calibration_result.json", result)
    results.append(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one four-GPU 2-epoch + validation memory-calibration wave."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()

    jobs = _load_manifest(args.manifest)
    dataset = (
        args.dataset_root.resolve()
        if args.dataset_root is not None
        else resolve_dataset_root(ROOT)
    )
    if not dataset.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset}")
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if any(
        int(job["global_slot"]) >= len(names)
        or GPU_PATTERN.lower() not in names[int(job["global_slot"])].lower()
        for job in jobs
    ):
        raise RuntimeError(f"calibration requires RTX 2080 Ti slots; found {names}")

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"calibration output root must be fresh or empty: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    _prepare_short_streams(jobs, output_root)
    commit = runtime.git_commit(ROOT)
    results: list[dict[str, Any]] = []
    threads = [
        threading.Thread(
            target=_run_one,
            kwargs={
                "job": job,
                "dataset": dataset,
                "output_root": output_root,
                "commit": commit,
                "results": results,
            },
            daemon=False,
        )
        for job in jobs
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    results.sort(key=lambda row: int(row["slot"]))
    summary = {
        "schema": "drl_rq_2080ti_memory_calibration_wave_v1",
        "manifest": str(args.manifest.resolve()),
        "dataset_root": str(dataset),
        "git_commit": commit,
        "working_tree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        "jobs": len(results),
        "passed": len(results) == len(jobs) and all(row["passed"] for row in results),
        "results": results,
    }
    runtime.atomic_json(output_root / "wave_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
