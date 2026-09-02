#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ENV = (
    "EVRPTW_REPO_ROOT",
    "EVRPTW_DATASET_ROOT",
    "EVRPTW_OUTPUT_ROOT",
    "EVRPTW_CONDA_ENV",
)
STOP = threading.Event()
CHILDREN: dict[int, subprocess.Popen[str]] = {}
CHILD_LOCK = threading.Lock()
OVERFLOW_LOCK = threading.Lock()


def process_rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        pass
    return 0


def process_gpu_memory_bytes(pid: int) -> int:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return 0
    total_mib = 0
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) == pid:
            try:
                total_mib += int(fields[1])
            except ValueError:
                continue
    return total_mib * 1024 * 1024


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def record_a6000_overflow(job: dict[str, Any], context: dict[str, Any]) -> Path:
    overflow = dict(job)
    overflow.update(
        {
            "hardware": "a6000",
            "global_slot": int(job["seed"]) % 2,
            "queue_position": -1,
            "overflow_reason": "OOM_UNCHANGED_CONFIG",
            "source_hardware": job.get("hardware"),
        }
    )
    path = context["output"] / "a6000_overflow_jobs_v1.jsonl"
    with OVERFLOW_LOCK:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(overflow, sort_keys=True) + "\n")
            stream.flush()
    return path


def load_jobs(path: Path, slots: set[int], mode: str) -> list[dict[str, Any]]:
    jobs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    jobs = [job for job in jobs if int(job["global_slot"]) in slots and job.get("enabled", True)]
    if mode in {"pilot"}:
        jobs = [job for job in jobs if job["run_mode"] == "pilot"]
    elif mode in {"full", "resume"}:
        jobs = [job for job in jobs if job["run_mode"] == "full"]
    elif mode == "evaluate":
        jobs = [job for job in jobs if job["run_mode"] == "evaluate"]
    return sorted(jobs, key=lambda job: (int(job["global_slot"]), int(job["queue_position"])))


def git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def preflight(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing required environment variables: {', '.join(missing)}")
    repo = Path(os.environ["EVRPTW_REPO_ROOT"]).resolve()
    dataset = Path(os.environ["EVRPTW_DATASET_ROOT"]).resolve()
    output = Path(os.environ["EVRPTW_OUTPUT_ROOT"]).resolve()
    if not (repo / ".git").exists() or not dataset.is_dir():
        raise RuntimeError("repository or dataset root does not exist")
    expected_env = os.environ["EVRPTW_CONDA_ENV"]
    if Path(sys.prefix).name != expected_env:
        raise RuntimeError(
            f"wrong Python environment: {Path(sys.prefix).name}; expected {expected_env}"
        )
    output.mkdir(parents=True, exist_ok=True)
    probe = output / f".write_probe_{os.getpid()}"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    free = shutil.disk_usage(output).free
    if free < int(args.minimum_free_gib * 1024**3):
        raise RuntimeError(f"output free space is below {args.minimum_free_gib} GiB")
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repo, text=True
    ).strip()
    commit = git_commit(repo)
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).strip()
    if dirty and not args.dry_run:
        raise RuntimeError("working tree must be clean for a non-dry run")
    expected_branch = args.expected_branch
    if expected_branch and branch != expected_branch:
        raise RuntimeError(f"wrong branch: {branch}; expected {expected_branch}")
    if not args.skip_gpu_preflight:
        query = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).splitlines()
        if len(query) != args.local_gpu_count:
            raise RuntimeError(f"expected {args.local_gpu_count} GPUs, found {len(query)}")
        bad = [name for name in query if args.gpu_name_pattern.lower() not in name.lower()]
        if bad:
            raise RuntimeError(f"unexpected GPU model(s): {bad}")
    for job in jobs:
        index_key = "train_index" if job["kind"] in {"train", "pilot"} else "dataset_index"
        path = dataset / job[index_key]
        if not path.is_file():
            raise FileNotFoundError(f"dataset index is missing for {job['job_id']}: {path}")
    context = {
        "repo": repo,
        "dataset": dataset,
        "output": output,
        "branch": branch,
        "commit": commit,
        "free_bytes": free,
        "conda_env": os.environ["EVRPTW_CONDA_ENV"],
    }
    if args.mode == "evaluate" and not args.dry_run:
        missing_checkpoints = [
            str(checkpoint_dir(job, context) / "checkpoint_selected.pt")
            for job in jobs
            if not (checkpoint_dir(job, context) / "checkpoint_selected.pt").is_file()
        ]
        if missing_checkpoints:
            raise FileNotFoundError(
                "evaluation checkpoint dependencies are missing; sync the listed "
                f"training outputs first: {missing_checkpoints[:5]}"
            )
    return context


def output_dir(job: dict[str, Any], context: dict[str, Any]) -> Path:
    root = context["output"] / job["representation"] / job["method"] / job["scale"] / f"seed_{job['seed']}" / context["commit"]
    if job["kind"] in {"eval", "transfer"}:
        root = root / job["test_id"] / job["decode_budget"]
    elif job["kind"] == "pilot":
        root = root / "pilot" / job["stage"]
    return root


def training_command(job: dict[str, Any], context: dict[str, Any], out: Path, resume: bool) -> list[str]:
    dataset = context["dataset"]
    if job["method"] == "terran":
        command = [
            sys.executable,
            "-m",
            job["train_module"],
            "--config",
            str(ROOT / "TERRAN" / "configs" / "stage2_cus100_terran.yaml"),
            "--seed",
            str(job["seed"]),
            "--device",
            "cuda",
            "--stage2-dataset-path",
            str(dataset / job["train_index"]),
            "--stage2-family-root",
            str(dataset / "materialized" / "families"),
            "--stage2-scale",
            job["scale"],
            "--num-customers",
            job["scale"].removeprefix("Cus"),
            "--num-envs-per-gpu",
            str(job["physical_batch_size"]),
            "--data-passes",
            str(job["data_passes"]),
            "--physical-batch-size",
            str(job["physical_batch_size"]),
            "--effective-batch-size",
            str(job["effective_batch_size"]),
            "--validation-dataset-path",
            str(dataset / job["validation_index"]),
            "--validation-family-root",
            str(dataset / "materialized" / "families"),
            "--validation-limit",
            str(job["validation_views"]),
            "--validation-every-passes",
            str(job["validation_every_passes"]),
            "--protocol-id",
            job["protocol_id"],
            "--output-dir",
            str(out),
        ]
    else:
        command = [
            sys.executable,
            "-m",
            job["train_module"],
            "--dataset-path",
            str(dataset / job["train_index"]),
            "--family-root",
            str(dataset / "materialized" / "families"),
            "--scale",
            job["scale"],
            "--split-ids",
            "train",
            "--track-ids",
            "train",
            "--seed",
            str(job["seed"]),
            "--device",
            "cuda",
            "--data-passes",
            str(job["data_passes"]),
            "--physical-batch-size",
            str(job["physical_batch_size"]),
            "--effective-batch-size",
            str(job["effective_batch_size"]),
            "--validation-dataset-path",
            str(dataset / job["validation_index"]),
            "--validation-family-root",
            str(dataset / "materialized" / "families"),
            "--validation-limit",
            str(job["validation_views"]),
            "--validation-every-passes",
            str(job["validation_every_passes"]),
            "--protocol-id",
            job["protocol_id"],
            "--output-dir",
            str(out),
        ]
    if job["run_mode"] == "pilot":
        command.extend(
            [
                "--pilot-mode",
                "--max-batches-per-pass",
                str(job["max_batches_per_pass"]),
            ]
        )
    if resume:
        command.append("--resume")
    return command


def checkpoint_dir(job: dict[str, Any], context: dict[str, Any]) -> Path:
    train = dict(job)
    train["scale"] = job.get("source_scale", job["scale"])
    train["kind"] = "train"
    return output_dir(train, context)


def evaluation_command(job: dict[str, Any], context: dict[str, Any], out: Path) -> list[str]:
    dataset = context["dataset"]
    checkpoint = checkpoint_dir(job, context) / "checkpoint_selected.pt"
    command = [
        sys.executable,
        "-m",
        job["eval_module"],
        "--dataset-path",
        str(dataset / job["dataset_index"]),
        "--family-root",
        str(dataset / "materialized" / "families"),
        "--checkpoint",
        str(checkpoint),
        "--scale",
        job["scale"],
        "--split-ids",
        "test",
        "--track-ids",
        job["track_id"],
        "--candidates",
        str(job["candidate_count"]),
        "--candidate-chunk-size",
        str(job["candidate_chunk_size"]),
        "--seed",
        str(job["seed"]),
        "--limit",
        str(job["expected_views"]),
        "--device",
        "cuda",
        "--output-dir",
        str(out),
    ]
    if job["method"] == "terran":
        command.extend(["--decode-mode", "greedy" if job["decode_type"] == "greedy" else "sample"])
    else:
        command.extend(["--decode-type", job["decode_type"]])
    return command


def command_for(job: dict[str, Any], context: dict[str, Any], out: Path, resume: bool) -> list[str]:
    if "test_command" in job:
        return list(job["test_command"])
    if job["kind"] in {"train", "pilot"}:
        return training_command(job, context, out, resume)
    return evaluation_command(job, context, out)


def job_complete(job: dict[str, Any], out: Path) -> bool:
    result = out / "job_result.json"
    if not result.exists():
        return False
    payload = json.loads(result.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        return False
    if job["kind"] in {"train", "pilot"}:
        return (out / "checkpoint_selected.pt").is_file() and (out / "validation_summary.json").is_file()
    return (out / "summary.csv").is_file() and (out / "routes.jsonl").is_file()


def should_resume_job(job: dict[str, Any], out: Path, requested: bool) -> bool:
    if not requested or job["kind"] not in {"train", "pilot"}:
        return False
    state = (out / "data_pass_state.json").is_file()
    checkpoint = (out / "checkpoint_latest.pt").is_file()
    if state != checkpoint:
        raise RuntimeError(
            f"incomplete resume evidence for {job['job_id']}: "
            f"state={state} checkpoint={checkpoint}"
        )
    return state and checkpoint


def run_job(job: dict[str, Any], context: dict[str, Any], local_gpu: int, resume: bool, dry_run: bool) -> bool:
    out = output_dir(job, context)
    out.mkdir(parents=True, exist_ok=True)
    if job_complete(job, out):
        return True
    resume_this_job = should_resume_job(job, out, resume)
    command = command_for(job, context, out, resume_this_job)
    provenance = {
        "schema": "drl_job_provenance_v1",
        "job": job,
        "command": command,
        "git_commit": context["commit"],
        "git_branch": context["branch"],
        "dataset_release_id": (context["dataset"] / "release_manifest.json").read_text(encoding="utf-8")[:4096]
        if (context["dataset"] / "release_manifest.json").exists()
        else "unavailable",
        "conda_env": context["conda_env"],
        "resume_requested": bool(resume),
        "resumed_from_checkpoint": bool(resume_this_job),
        "started_at": time.time(),
    }
    atomic_json(out / "provenance.json", provenance)
    if dry_run:
        print(json.dumps({"job_id": job["job_id"], "gpu": local_gpu, "command": command}, sort_keys=True))
        return True
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(local_gpu)
    started = time.perf_counter()
    with (out / "stdout.log").open("a", encoding="utf-8") as stdout, (out / "stderr.log").open("a", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=context["repo"], env=env, text=True, stdout=stdout, stderr=stderr, start_new_session=True)
        with CHILD_LOCK:
            CHILDREN[local_gpu] = process
        peak_cpu_memory_bytes = 0
        peak_gpu_memory_bytes = 0
        while process.poll() is None:
            peak_cpu_memory_bytes = max(
                peak_cpu_memory_bytes, process_rss_bytes(process.pid)
            )
            peak_gpu_memory_bytes = max(
                peak_gpu_memory_bytes, process_gpu_memory_bytes(process.pid)
            )
            time.sleep(0.5)
        returncode = process.wait()
        with CHILD_LOCK:
            CHILDREN.pop(local_gpu, None)
    stderr_text = (out / "stderr.log").read_text(encoding="utf-8", errors="replace")
    oom = returncode in {137, -9} or "out of memory" in stderr_text.lower()
    overflow_manifest = None
    if oom and job.get("hardware") == "2080ti" and job.get("scale") == "Cus500":
        overflow_manifest = record_a6000_overflow(job, context)
    passed = returncode == 0 and (
        ((out / "checkpoint_selected.pt").is_file() and (out / "validation_summary.json").is_file())
        if job["kind"] in {"train", "pilot"}
        else ((out / "summary.csv").is_file() and (out / "routes.jsonl").is_file())
    )
    result = {
        "schema": "drl_job_result_v1",
        "job_id": job["job_id"],
        "status": "passed" if passed else "failed",
        "returncode": returncode,
        "wall_time_s": time.perf_counter() - started,
        "peak_cpu_memory_bytes": peak_cpu_memory_bytes,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "completed_at": time.time(),
        "failure_reason": "OOM_UNCHANGED_CONFIG" if oom else None,
        "overflow_manifest": str(overflow_manifest) if overflow_manifest else None,
    }
    atomic_json(out / "job_result.json", result)
    return passed


def worker(slot: int, jobs: list[dict[str, Any]], context: dict[str, Any], local_gpu: int, resume: bool, dry_run: bool, failures: list[str]) -> None:
    for job in jobs:
        if STOP.is_set():
            return
        if not run_job(job, context, local_gpu, resume, dry_run):
            failures.append(job["job_id"])
            return


def handle_signal(signum: int, _frame: Any) -> None:
    STOP.set()
    with CHILD_LOCK:
        children = list(CHILDREN.values())
    for process in children:
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass


def require_pilot_gate(context: dict[str, Any]) -> None:
    report = context["output"] / "pilot_gate_report.json"
    if not report.exists() or not json.loads(report.read_text(encoding="utf-8")).get("passed"):
        raise RuntimeError(f"full mode is blocked until the pilot gate passes: {report}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen DRL manifest queues.")
    parser.add_argument("mode", choices=("pilot", "full", "evaluate", "status", "resume"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--slots", required=True)
    parser.add_argument("--local-gpu-count", type=int, required=True)
    parser.add_argument("--gpu-name-pattern", required=True)
    parser.add_argument("--expected-branch", default="drl-benchmark-adapters")
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--skip-gpu-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slots = {int(value) for value in args.slots.split(",") if value.strip()}
    jobs = load_jobs(args.manifest, slots, args.mode)
    context = preflight(args, jobs)
    if args.mode == "status":
        rows = []
        for mode in ("pilot", "full", "evaluate"):
            for job in load_jobs(args.manifest, slots, mode):
                rows.append({"job_id": job["job_id"], "complete": job_complete(job, output_dir(job, context))})
        print(json.dumps({"jobs": len(rows), "completed": sum(row["complete"] for row in rows), "rows": rows}, sort_keys=True))
        return
    if args.mode in {"full", "resume"}:
        require_pilot_gate(context)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    by_slot = {slot: [job for job in jobs if int(job["global_slot"]) == slot] for slot in slots}
    failures: list[str] = []
    threads = []
    for local_gpu, slot in enumerate(sorted(slots)):
        thread = threading.Thread(target=worker, args=(slot, by_slot[slot], context, local_gpu, args.mode == "resume", args.dry_run, failures), daemon=False)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit(f"failed queues: {failures}")


if __name__ == "__main__":
    main()
