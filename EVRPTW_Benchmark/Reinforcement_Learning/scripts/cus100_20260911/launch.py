#!/usr/bin/env python3
"""The ten Cus100 jobs only; no legacy queues, warm starts or automatic tests."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import platform
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
MANIFEST = HERE / "cus100_seed1234_jobs.jsonl"
RUN_ROOT = "EVRPTW_Benchmark/results/cus100_20260911"
SERVERS = {"2080ti_4_1": 4, "2080ti_4_2": 4, "2080ti_3_1": 3}
ASSIGNMENTS = {
    "TR02": ("2080ti_4_1", 0, "am_evrptw", "G"),
    "TR01": ("2080ti_4_1", 1, "am_evrptw", "E"),
    "TR06": ("2080ti_4_1", 2, "terran", "G"),
    "TR05": ("2080ti_4_1", 3, "terran", "E"),
    "TR04": ("2080ti_4_2", 0, "drl_ts", "G"),
    "TR03": ("2080ti_4_2", 1, "drl_ts", "E"),
    "TR18": ("2080ti_4_2", 2, "evrptw_rl", "G"),
    "TR17": ("2080ti_4_2", 3, "evrptw_rl", "E"),
    "TR10": ("2080ti_3_1", 1, "rrnco", "G"),
    "TR09": ("2080ti_3_1", 2, "rrnco", "E"),
}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value, repo=REPO):
    path = Path(value)
    return path if path.is_absolute() else Path(repo) / path


def dataset_root(job, repo=REPO, data_root=None):
    override = data_root or os.environ.get(
        "CUS100_SYNTHETIC_ROOT" if job["source_kind"] == "terran_synthetic" else "CUS100_ROAD_ROOT"
    )
    return resolve_path(override or job["dataset_root"], repo).resolve()


def output_directory(job, repo=REPO, output_root=None):
    root = resolve_path(output_root or os.environ.get("CUS100_OUTPUT_ROOT", RUN_ROOT), repo)
    return root / "runs" / job["experiment_id"]


def build_command(job, repo=REPO, data_root=None, output_root=None,
                  python=sys.executable, overrides=None):
    """Build one command without launching; overrides support isolated profiling.

    An override named ``output_dir`` is an exact directory. Other overrides
    replace manifest fields (batch, epochs, stream, validation, extra_args).
    """
    repo = Path(repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from EVRPTW_Benchmark.Reinforcement_Learning.scripts.drl_job_runtime import training_command
    selected = copy.deepcopy(job)
    selected.update(overrides or {})
    out = selected.pop("output_dir", None)
    out = Path(out) if out else output_directory(selected, repo, output_root)
    context = {"repo": repo, "dataset": dataset_root(selected, repo, data_root),
               "output": Path(out).parent, "reuse_preverified_training_streams": False}
    # training_command is only a pure CLI builder. The old runtime's manifest,
    # architecture allowlist, scheduler and formal launch gate are not invoked.
    command = training_command(selected, context, out, resume=False)
    command[0] = str(python)
    command.extend(str(item) for item in selected.get("extra_args", []))
    return command


def load_jobs(path=MANIFEST, server=None):
    jobs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(jobs) != 10 or {j["experiment_id"] for j in jobs} != set(ASSIGNMENTS):
        raise ValueError("This manifest must contain exactly the ten declared Cus100 jobs")
    for job in jobs:
        identity = (job["server"], job["gpu"], job["method"], job["representation"])
        if identity != ASSIGNMENTS[job["experiment_id"]]:
            raise ValueError(f"Unexpected assignment for {job['experiment_id']}: {identity}")
        if job["scale"] != "Cus100" or job["seed"] != 1234:
            raise ValueError("Only Cus100, seed 1234 is permitted")
        if job.get("warm_start_source_commit") or job.get("resume"):
            raise ValueError("This round requires fresh output and optimizer state")
        if job["training_trajectory_count"] != 30 or job["validation_candidate_count"] != 30:
            raise ValueError("Training and validation candidate counts must remain 30")
    return [job for job in jobs if server is None or job["server"] == server]


def gpu_inventory():
    fields = ["index", "uuid", "name", "memory.total", "memory.used", "utilization.gpu", "driver_version"]
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=" + ",".join(fields),
                                   "--format=csv,noheader,nounits"], text=True)
    result = []
    for line in raw.strip().splitlines():
        values = [item.strip() for item in line.split(",")]
        row = dict(zip(fields, values))
        row["index"] = int(row["index"])
        result.append(row)
    return result


def gpu_processes():
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                   "--format=csv,noheader,nounits"], text=True)
    return [dict(zip(("gpu_uuid", "pid", "used_memory_mib"),
                     [item.strip() for item in line.split(",")]))
            for line in raw.strip().splitlines() if line.strip()]


def completion_status(record, returncode, stopped):
    if stopped:
        return "stopped"
    if returncode != 0:
        return "failed"
    root = Path(record["output_dir"])
    try:
        result = json.loads((root / "training_result.json").read_text())
        completed = int(result["completed_training_epochs"])
        job = record["job"]
        if result.get("status") not in {"passed", "early_stopped"}:
            raise ValueError("Trainer did not report a valid completion outcome")
        if not job["minimum_training_epochs"] <= completed <= job["training_epochs"]:
            raise ValueError("Trainer completion is outside the declared epoch budget")
        if not any((root / name).is_file() for name in ("best.ckpt", "best_overall.ckpt")):
            raise ValueError("Trainer did not save a selected checkpoint")
        record["completed_training_epochs"] = completed
        record["training_result_status"] = result["status"]
        return "completed"
    except (OSError, ValueError, KeyError) as error:
        record["completion_error"] = str(error)
        return "failed"


def preflight(jobs, manifest, repo=REPO, output_root=None, require_ready=True, verify_payloads=True):
    repo = Path(repo)
    gpus = gpu_inventory()
    by_index = {gpu["index"]: gpu for gpu in gpus}
    processes = gpu_processes()
    checks = []
    hashed = {}
    synthetic_contracts = {}
    for job in jobs:
        selected = by_index.get(job["gpu"])
        if selected is None or "2080 Ti" not in selected["name"]:
            raise RuntimeError(f"GPU {job['gpu']} is not an available RTX 2080 Ti")
        if any(process["gpu_uuid"] == selected["uuid"] for process in processes):
            raise RuntimeError(f"GPU {job['gpu']} is occupied; no existing task will be stopped")
        if require_ready and (job.get("calibration_status") != "passed" or not job.get("enabled")):
            raise RuntimeError(f"{job['experiment_id']} has not passed this round's calibration")
        for field in ("physical_batch_size", "effective_batch_size"):
            if require_ready and (not isinstance(job.get(field), int) or job[field] < 1):
                raise RuntimeError(f"{job['experiment_id']}: {field} is not calibrated")
        if require_ready:
            if job["physical_batch_size"] > job["effective_batch_size"]:
                raise RuntimeError("Physical batch must not exceed effective batch")
            expected_exposure = job["training_epochs"] * job["effective_batch_size"] * 100
            if job.get("customer_exposure_budget") != expected_exposure:
                raise RuntimeError(f"{job['experiment_id']}: inconsistent exposure budget")
        data = dataset_root(job, repo)
        if job["source_kind"] == "terran_synthetic" and str(data) not in synthetic_contracts:
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus100_20260911.data_contract import inspect_synthetic
            synthetic_contracts[str(data)] = inspect_synthetic(data, verify_payloads=verify_payloads)
        if job.get("data_source_manifest_sha256") and (
            synthetic_contracts.get(str(data), {}).get("manifest_sha256") != job["data_source_manifest_sha256"]
        ):
            raise RuntimeError(f"{job['experiment_id']}: synthetic corpus manifest differs from prepared streams")
        paths = {"train_index": data / job["train_index"],
                 "validation_index": data / job["validation_index"],
                 "objective_config_path": resolve_path(job["objective_config_path"], repo),
                 "reward_contract_config_path": resolve_path(job["reward_contract_config_path"], repo)}
        for field in ("training_stream_path", "method_auxiliary_profile_path", "terran_config_path"):
            if job.get(field):
                paths[field] = resolve_path(job[field], repo)
        if require_ready and not job.get("training_stream_path"):
            raise RuntimeError(f"{job['experiment_id']} has no frozen training ID stream")
        for field, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            if str(path) not in hashed:
                hashed[str(path)] = sha256(path)
            expected = job.get(field + "_sha256")
            if expected and hashed[str(path)] != expected:
                raise RuntimeError(f"Changed input {field}: {path}")
        if require_ready and job["source_kind"] == "terran_synthetic":
            reward = json.loads(paths["reward_contract_config_path"].read_text())
            calibration = reward.get("calibration", {})
            source_contract = synthetic_contracts[str(data)]
            if (calibration.get("source_kind") != "terran_synthetic"
                    or calibration.get("source_split") != "train"
                    or calibration.get("pilot") is not False
                    or calibration.get("train_index_sha256") != hashed[str(paths["train_index"])]
                    or calibration.get("corpus_manifest_sha256") != source_contract["manifest_sha256"]):
                raise RuntimeError(f"{job['experiment_id']}: reward calibration is not bound to this frozen synthetic training corpus")
        out = output_directory(job, repo, output_root)
        if require_ready and out.exists() and any(out.iterdir()):
            raise FileExistsError(f"Fresh-run output already contains files: {out}")
        checks.append({"experiment_id": job["experiment_id"], "dataset_root": str(data),
                       "gpu": selected, "output_dir": str(out), "input_paths": {k: str(v) for k, v in paths.items()}})
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).splitlines()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus100_20260911.source_snapshot import capture_source
    snapshot_root = resolve_path(output_root or os.environ.get("CUS100_OUTPUT_ROOT", RUN_ROOT), repo) / "provenance"
    source_snapshot = capture_source(repo, snapshot_root)
    deployed = repo / RUN_ROOT / "deployment_manifest.json"
    if deployed.is_file():
        expected_deployment = json.loads(deployed.read_text())
        if expected_deployment.get("source_version") != source_snapshot["source_version"]:
            from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus100_20260911.deployment_upgrade import ensure_compatible_deployment
            expected_deployment = ensure_compatible_deployment(repo, deployed, manifest, source_snapshot)
        if expected_deployment.get("source_version") != source_snapshot["source_version"]:
            raise RuntimeError("Actual source differs from the transferred deployment snapshot")
        if expected_deployment.get("job_manifest_sha256") != sha256(manifest):
            raise RuntimeError("Job manifest differs from the transferred deployment package")
    package_versions = {}
    for package in ("torch", "numpy", "pandas", "numba", "pyarrow"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    return {"schema": "cus100_preflight_v1", "time": timestamp(), "hostname": socket.gethostname(),
            "repo": str(repo), "commit": commit, "git_status": dirty,
            "manifest_path": str(manifest), "manifest_sha256": sha256(manifest),
            "gpu_inventory": gpus, "gpu_processes": processes, "checks": checks,
            "input_sha256": hashed, "synthetic_contracts": synthetic_contracts,
            "source_snapshot": source_snapshot, "source_version": source_snapshot["source_version"],
            "cpu_count": os.cpu_count(), "python": sys.executable, "python_version": platform.python_version(), "package_versions": package_versions,
            "cpu_threads_per_job": 2}


def worker(args):
    jobs = load_jobs(args.manifest, args.server)
    root = resolve_path(args.output_root or os.environ.get("CUS100_OUTPUT_ROOT", RUN_ROOT))
    launcher = root / "launchers" / args.server
    launcher.mkdir(parents=True, exist_ok=True)
    with (launcher / "launcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit = preflight(jobs, args.manifest, output_root=root)
        write_json(launcher / "preflight.json", audit)
        children = []
        stopping = False

        def stop(signum, frame):
            nonlocal stopping
            stopping = True
            for child, _, _ in children:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        records = []
        write_json(launcher / "status.json", {"status": "starting", "pid": os.getpid(), "time": timestamp()})
        try:
            for job in jobs:
                if stopping:
                    break
                out = output_directory(job, output_root=root)
                out.mkdir(parents=True, exist_ok=False)
                command = build_command(job, output_root=root)
                env = os.environ.copy()
                gpu = next(item for item in audit["gpu_inventory"] if item["index"] == job["gpu"])
                env.update(CUDA_VISIBLE_DEVICES=gpu["uuid"], OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                           OPENBLAS_NUM_THREADS="2", NUMBA_NUM_THREADS="2", PYTHONUNBUFFERED="1")
                stdout = (out / "stdout.log").open("w")
                stderr = (out / "stderr.log").open("w")
                child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                         stdout=stdout, stderr=stderr, start_new_session=True)
                children.append((child, stdout, stderr))
                record = {"experiment_id": job["experiment_id"], "status": "running", "pid": child.pid,
                          "command": command, "job": job, "gpu": gpu, "hostname": socket.gethostname(),
                          "started_at": timestamp(), "output_dir": str(out), "preflight": str(launcher / "preflight.json"),
                          "source_version": audit["source_version"], "source_snapshot": audit["source_snapshot"]["manifest"],
                          "manifest_sha256": audit["manifest_sha256"]}
                records.append(record)
                write_json(out / "launch_record.json", record)
                write_json(launcher / "status.json", {"status": "running", "pid": os.getpid(), "jobs": records})
            while any(child.poll() is None for child, _, _ in children):
                time.sleep(2)
                for (child, _, _), record in zip(children, records):
                    if child.poll() is not None and record["status"] == "running":
                        record.update(status=completion_status(record, child.returncode, stopping),
                                      returncode=child.returncode, finished_at=timestamp())
                        write_json(Path(record["output_dir"]) / "launch_record.json", record)
                        write_json(launcher / "status.json", {"status": "stopping" if stopping else "running", "jobs": records})
            for (child, _, _), record in zip(children, records):
                if record["status"] == "running":
                    record.update(status=completion_status(record, child.returncode, stopping),
                                  returncode=child.returncode, finished_at=timestamp())
                    write_json(Path(record["output_dir"]) / "launch_record.json", record)
            failed = any(record["status"] == "failed" for record in records)
            write_json(launcher / "status.json", {"status": "stopped" if stopping else ("failed" if failed else "completed"),
                                                   "jobs": records, "finished_at": timestamp()})
            return 1 if failed else 0
        except BaseException:
            stop(signal.SIGTERM, None)
            for child, _, _ in children:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
            raise
        finally:
            for _, stdout, stderr in children:
                stdout.close()
                stderr.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", choices=SERVERS, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--mode", choices=("plan", "preflight", "start", "worker"), default="plan")
    args = parser.parse_args()
    jobs = load_jobs(args.manifest, args.server)
    if args.mode == "plan":
        print(json.dumps(jobs, indent=2, sort_keys=True))
    elif args.mode == "preflight":
        print(json.dumps(preflight(jobs, args.manifest, output_root=args.output_root), indent=2, sort_keys=True))
    elif args.mode == "worker":
        return worker(args)
    else:
        audit = preflight(jobs, args.manifest, output_root=args.output_root, verify_payloads=False)
        root = resolve_path(args.output_root or os.environ.get("CUS100_OUTPUT_ROOT", RUN_ROOT))
        launcher = root / "launchers" / args.server
        launcher.mkdir(parents=True, exist_ok=True)
        with (launcher / "start.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (launcher / "launcher.lock").open("a") as worker_lock:
                fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # No shell and no inherited terminal: the launcher survives logout.
            command = [sys.executable, str(Path(__file__).resolve()), "--server", args.server,
                       "--manifest", str(args.manifest.resolve()), "--mode", "worker", "--output-root", str(root)]
            stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            logfile = launcher / f"launcher_{stamp}.log"
            with logfile.open("w") as log:
                child = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write_json(launcher / "started.json", {"pid": child.pid, "time": timestamp(),
                                                    "log": str(logfile), "preflight": audit})
            time.sleep(2)
            if child.poll() is not None:
                raise RuntimeError(f"Launcher exited during startup; see {logfile}")
            print(json.dumps({"status": "launcher_started", "pid": child.pid,
                              "server": args.server, "log": str(logfile),
                              "note": "Actual trainer startup and completion are recorded in status.json"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
