#!/usr/bin/env python3
"""Fresh, guarded TR17/TR18 mean-aggregation retraining on physical GPU 3/2."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus100_20260911 import launch as legacy
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus100_20260911.source_snapshot import capture_source, source_files, source_identity
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913.launch import lock_gpus

DEFAULT_OUTPUT = REPO / "EVRPTW_Benchmark/results/cus100_evrptw_stable_20260914"
ORDER = ("TR18", "TR17")
write_json, timestamp, sha256 = legacy.write_json, legacy.timestamp, legacy.sha256


def resolve_existing(value, *, repo=REPO):
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    current = repo / path
    fallback = repo.parent / "EVRPTW-DB" / path
    return (current if current.exists() or not fallback.exists() else fallback).resolve()


def stable_jobs(*, manifest=legacy.MANIFEST, artifact_root=None, road_root=None, synthetic_root=None):
    old = {row["experiment_id"]: row for row in legacy.load_jobs(manifest)}
    artifacts = resolve_existing(artifact_root or os.environ.get("CUS100_ARTIFACT_ROOT", legacy.RUN_ROOT))
    jobs = []
    for identity in ORDER:
        job = copy.deepcopy(old[identity])
        if (job["method"], job["gpu"], job["server"]) != ("evrptw_rl", 2 if identity == "TR18" else 3, "2080ti_4_2"):
            raise ValueError("Unexpected source experiment assignment")
        source = job["source_kind"]
        override = (road_root or os.environ.get("CUS100_ROAD_ROOT")) if source == "stage2_road" else (synthetic_root or os.environ.get("CUS100_SYNTHETIC_ROOT"))
        job["dataset_root"] = str(resolve_existing(override or job["dataset_root"]))
        for field in ("training_stream_path", "reward_contract_config_path"):
            path = Path(job[field])
            try:
                suffix = path.relative_to(legacy.RUN_ROOT)
            except ValueError:
                job[field] = str(resolve_existing(path))
            else:
                job[field] = str(artifacts / suffix)
        for field in ("objective_config_path", "method_auxiliary_profile_path"):
            job[field] = str(resolve_existing(job[field]))
        job.update(original_experiment_id=identity, experiment_id=identity + "_stable_mean",
                   job_id=identity + "_stable_mean", protocol_id="cus100_evrptw_mean_stability_20260914_v1",
                   calibration_status="pending_local_gpu_probe", enabled=False,
                   stability_adaptation="degree_normalized_structure2vec_mean_v1")
        # Keep the original effective batch, stream and sample budget. Only a
        # measured physical microbatch may change on a card with less free RAM.
        job["extra_args"] = ["--activation-checkpoint-stride", "1", "--graph-aggregation", "mean",
                             "--learning-rate", "0.001", "--ema-warmup-steps", "1000"]
        jobs.append(job)
    return jobs


def available_gpus(jobs):
    inventory = {row["index"]: row for row in legacy.gpu_inventory()}
    processes = legacy.gpu_processes()
    selected = []
    for job in jobs:
        gpu = inventory.get(job["gpu"])
        if gpu is None or "2080 Ti" not in gpu["name"]:
            raise RuntimeError(f"Physical GPU {job['gpu']} is not an RTX 2080 Ti")
        occupied = [row for row in processes if row["gpu_uuid"] == gpu["uuid"]]
        if occupied:
            raise RuntimeError(f"GPU {job['gpu']} is occupied by {occupied}; identify the owner and model first. Use the stop helper only for verified old TR17/TR18; wait for other work to release this GPU. No existing process was stopped.")
        selected.append(gpu)
    return selected


def audit_inputs(jobs):
    hashes, details = {}, []
    for job in jobs:
        root = Path(job["dataset_root"])
        paths = {"train_index": root / job["train_index"], "validation_index": root / job["validation_index"]}
        paths.update({field: Path(job[field]) for field in ("training_stream_path", "objective_config_path", "reward_contract_config_path", "method_auxiliary_profile_path")})
        for field, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"{job['original_experiment_id']} missing {field}: {path}; set CUS100_ROAD_ROOT, CUS100_SYNTHETIC_ROOT or CUS100_ARTIFACT_ROOT to the existing deployment.")
            actual = hashes.setdefault(str(path), sha256(path))
            if actual != job[field + "_sha256"]:
                raise RuntimeError(f"Frozen {field} hash differs: {path}")
        if job["source_kind"] == "terran_synthetic":
            from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus100_20260911.data_contract import inspect_synthetic
            corpus = inspect_synthetic(root, verify_payloads=False)
            if corpus["manifest_sha256"] != job["data_source_manifest_sha256"]:
                raise RuntimeError("Synthetic corpus manifest differs from the original experiment")
        details.append({"experiment_id": job["experiment_id"], "dataset_root": str(root), "inputs": {key: str(value) for key, value in paths.items()}})
    return {"input_sha256": hashes, "checks": details}


def set_flag(command, name, value):
    result = list(command)
    while name in result:
        pos = result.index(name)
        del result[pos:pos + 2]
    return result + [name, str(value)]


def command_for(job, out, *, physical=None, probe=False):
    selected = copy.deepcopy(job)
    selected["physical_batch_size"] = physical or job["physical_batch_size"]
    if probe:
        selected.update(training_epochs=6, minimum_training_epochs=3,
                        post_minimum_validation_every_epochs=3, validation_every_epochs=3,
                        validation_checkpoints=2, validation_views=10,
                        customer_exposure_budget=6 * selected["effective_batch_size"] * 100,
                        minimum_customer_exposure_budget=3 * selected["effective_batch_size"] * 100,
                        early_stop_start_epoch=3, early_stop_patience_validations=5,
                        protocol_id="cus100_evrptw_stability_disposable_probe_v1")
        for flag, value in (("--ema-warmup-steps", 2), ("--baseline-eval-interval", 2), ("--baseline-eval-size", 2)):
            selected["extra_args"] = set_flag(selected["extra_args"], flag, value)
    return legacy.build_command(selected, data_root=selected["dataset_root"], overrides={"output_dir": out})


def require_source(expected):
    if source_identity(source_files(REPO)) != expected:
        raise RuntimeError("Source changed since this retraining launch was armed")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def audit_probe(run):
    import torch
    result = json.loads((run / "training_result.json").read_text())
    if result.get("status") != "passed" or result.get("completed_training_epochs") != 6:
        raise RuntimeError("Smoke training did not complete all six updates")
    for filename in ("best.ckpt", "checkpoint_latest.pt", "checkpoint_epoch_0003.pt"):
        if not (run / filename).is_file() or (run / filename).stat().st_size == 0:
            raise RuntimeError(f"Missing smoke checkpoint: {filename}")
    previous = torch.load(run / "checkpoint_epoch_0003.pt", map_location="cpu", weights_only=False)
    current = torch.load(run / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    def finite_tensors(value):
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
            if not torch.isfinite(value).all():
                raise RuntimeError("Nonfinite smoke model/baseline/optimizer tensors")
        elif isinstance(value, dict):
            for nested in value.values():
                finite_tensors(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                finite_tensors(nested)
    for checkpoint in (previous, current):
        for field in ("model", "baseline", "optimizer"):
            finite_tensors(checkpoint.get(field))
    if not any(not torch.equal(current["model"][name], tensor) for name, tensor in previous["model"].items()):
        raise RuntimeError("Smoke model parameters did not change")
    train = read_rows(run / "logical_epoch_history.jsonl")
    if len(train) != 6 or any(not math.isfinite(row["mean_loss"]) for row in train):
        raise RuntimeError("Smoke loss/epoch history is incomplete or nonfinite")
    diagnostics = read_rows(run / "reward_diagnostics.jsonl")
    if [row["baseline_kind"] for row in diagnostics] != ["paper_ema"] * 2 + ["greedy_rollout"] * 4:
        raise RuntimeError("Smoke did not exercise native EMA-to-greedy transition")
    gradients = []
    for row in diagnostics:
        summary = row.get("gradients", {}).get("pre_clip_norm", {})
        if (not isinstance(summary, dict) or summary.get("count") != 1
                or summary.get("finite_count") != 1 or summary.get("nonfinite_count") != 0
                or summary.get("mean") is None):
            raise RuntimeError("Smoke training gradients are missing or nonfinite")
        gradients.append(float(summary["mean"]))
    if any(not math.isfinite(value) for value in gradients) or sum(value > 1e-8 for value in gradients) < 3:
        raise RuntimeError("Smoke training gradients are nonfinite or ineffective; parameter decay alone is insufficient")
    if [row["optimizer_step"] for row in read_rows(run / "baseline_history.jsonl")] != [4, 6]:
        raise RuntimeError("Smoke did not exercise greedy baseline probes")
    validations = read_rows(run / "validation_history.jsonl")
    if [row["logical_epoch"] for row in validations] != [3, 6] or any(row["instances"] != 10 for row in validations):
        raise RuntimeError("Smoke validation did not complete the two fixed cohorts")
    keys = ("complete_and_feasible", "mean_verified_cost_usd", "mean_verified_distance_km", "mean_verified_vehicle_count")
    if tuple(validations[0].get(key) for key in keys) == tuple(validations[1].get(key) for key in keys):
        raise RuntimeError("Both fixed smoke validations are exactly unchanged; inspect policy diagnostics before formal training")
    return {"completed_updates": 6, "baseline_transition": "checked", "validation_checks": 2,
            "training_pre_clip_gradient_norms": gradients,
            "validation_costs": [row["mean_verified_cost_usd"] for row in validations]}


def last_row(path):
    value = {}
    try:
        with Path(path).open() as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    return value


def compact_status(root):
    path = root / "launchers/2080ti_4_2/status.json"
    if not path.exists():
        return {"status": "not_started", "output_root": str(root)}
    state = json.loads(path.read_text())
    result = {key: state[key] for key in ("status", "pid", "time", "experiment_id", "error", "calibration") if key in state}
    result["jobs"] = []
    for record in state.get("jobs", []):
        run = Path(record["output_dir"])
        training_path = run / "logical_epoch_history.jsonl"
        train = last_row(training_path)
        latest = last_row(run / "validation_history.jsonl")
        best = {}
        if (run / "validation_summary.json").exists():
            best = json.loads((run / "validation_summary.json").read_text())
        def validation(row):
            return {key: row.get(key) for key in ("logical_epoch", "mean_verified_cost_usd", "complete_and_feasible", "instances")}
        result["jobs"].append({"experiment_id": record["experiment_id"], "status": record["status"],
                               "pid": record["pid"], "gpu": record["gpu"]["index"],
                               "epoch": train.get("logical_epoch"),
                               "training_feasibility": train.get("mean_environment_feasible_rate"),
                               "train_write_age_s": round(time.time() - training_path.stat().st_mtime, 1) if training_path.exists() else None,
                               "latest_validation": validation(latest), "best_validation": validation(best)})
    return result


def stop_owned(child):
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=15)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=15)


def environment(gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[key] = "2"
    return env


def run_probe(job, root, gpu, physical, locks, source_sha):
    require_source(source_sha)
    if available_gpus([job])[0]["uuid"] != gpu["uuid"]:
        raise RuntimeError("Physical GPU identity changed")
    run = root / "calibration" / job["experiment_id"] / f"batch_{physical}"
    run.mkdir(parents=True, exist_ok=False)
    command = command_for(job, run, physical=physical, probe=True)
    write_json(run / "command.json", command)
    peak, started = 0.0, time.monotonic()
    heartbeat = 0.0
    with (run / "stdout.log").open("w") as stdout, (run / "stderr.log").open("w") as stderr:
        child = subprocess.Popen(command, cwd=REPO, env=environment(gpu), stdout=stdout, stderr=stderr,
                                 start_new_session=True, pass_fds=tuple(locks))
        try:
            while child.poll() is None:
                for row in legacy.gpu_processes():
                    if row["gpu_uuid"] == gpu["uuid"] and str(row["pid"]) == str(child.pid):
                        try:
                            peak = max(peak, float(row["used_memory_mib"]) / 1024)
                        except ValueError:
                            pass
                if time.monotonic() - heartbeat >= 30:
                    progress_path = run / "logical_epoch_history.jsonl"
                    write_json(root / "launchers/2080ti_4_2/status.json", {
                        "status": "calibrating", "pid": os.getpid(), "time": timestamp(),
                        "experiment_id": job["experiment_id"], "jobs": [],
                        "calibration": {"phase": "six_update_cuda_smoke", "gpu": job["gpu"],
                                        "probe_pid": child.pid, "physical_batch": physical,
                                        "peak_process_gib": peak, "elapsed_s": time.monotonic() - started,
                                        "epoch": last_row(progress_path).get("logical_epoch"),
                                        "train_write_age_s": time.time() - progress_path.stat().st_mtime if progress_path.exists() else None,
                                        "output_dir": str(run)}})
                    heartbeat = time.monotonic()
                if time.monotonic() - started > 7200:
                    raise RuntimeError("GPU smoke exceeded its two-hour timeout")
                time.sleep(1)
        finally:
            stop_owned(child)
    error = (run / "stderr.log").read_text(errors="replace")[-12000:]
    report = {"physical_batch_size": physical, "effective_batch_size": 200,
              "peak_process_gib": peak, "elapsed_s": time.monotonic() - started, "run": str(run)}
    if child.returncode:
        report.update(status="oom" if "CUDA out of memory" in error or "CUDA error: out of memory" in error else "failed", error=error)
    else:
        try:
            if not math.isfinite(peak) or peak <= 0:
                raise RuntimeError("No positive process GPU-memory measurement was collected")
            report.update(audit_probe(run), status="passed")
        except Exception as exc:
            report.update(status="failed", error=str(exc))
    write_json(run / "memory_and_training_audit.json", report)
    return report


def diagnose_probe(job, run, source_sha):
    require_source(source_sha)
    run = Path(run)
    report_path = run / "policy_diagnostics.json"
    command = [sys.executable, str(HERE.parent.parent / "diagnose_evrptw_policy.py"),
               "--checkpoint", str(run / "checkpoint_latest.pt"),
               "--dataset-path", str(Path(job["dataset_root"]) / job["train_index"]),
               "--family-root", str(Path(job["dataset_root"]) / "materialized/families"),
               "--instances", "3", "--state-steps", "0,10,50,100,180,239",
               "--max-steps", "240", "--n-traj", "30", "--seed", "910001234",
               "--require-effective-policy", "--output", str(report_path)]
    write_json(run / "policy_diagnostics_command.json", command)
    with (run / "policy_diagnostics.log").open("w") as log:
        child = subprocess.Popen(command, cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started, heartbeat = time.monotonic(), 0.0
        try:
            while child.poll() is None:
                if time.monotonic() - heartbeat >= 30:
                    write_json(run.parents[2] / "launchers/2080ti_4_2/status.json", {
                        "status": "calibrating", "pid": os.getpid(), "time": timestamp(),
                        "experiment_id": job["experiment_id"], "jobs": [],
                        "calibration": {"phase": "cpu_policy_effectiveness", "gpu": job["gpu"],
                                        "probe_pid": child.pid, "elapsed_s": time.monotonic() - started,
                                        "output_dir": str(run)}})
                    heartbeat = time.monotonic()
                if time.monotonic() - started > 1800:
                    raise RuntimeError("Policy effectiveness diagnostic exceeded 30 minutes")
                time.sleep(1)
        finally:
            stop_owned(child)
    if child.returncode:
        raise RuntimeError(f"{job['experiment_id']} policy learning gate failed; inspect {report_path} and policy_diagnostics.log")
    report = json.loads(report_path.read_text())
    if (not report["gate"]["passed"] or report["counterfactual"]
            or report["effective_graph_aggregation"] != "mean"
            or report["checkpoint_sha256"] != sha256(run / "checkpoint_latest.pt")
            or report["training_index_sha256"] != job["train_index_sha256"]):
        raise RuntimeError("Smoke checkpoint did not pass the actual mean-policy diagnostic gate")
    return {"policy_diagnostics": str(report_path), "policy_gate": report["gate"]}


def calibrate_job(job, root, gpu, locks, source_sha):
    # Preserve effective batch/exposure. Refine by ten instances near the old
    # 200-instance allocation; very small GPUs continue at smaller microbatches.
    for physical in (*range(200, 9, -10), 5, 1):
        report = run_probe(job, root, gpu, physical, locks, source_sha)
        if report["status"] == "failed":
            raise RuntimeError(f"{job['experiment_id']} smoke failed: {report.get('error')}")
        if report["status"] == "passed" and report["peak_process_gib"] <= 10.3:
            diagnostic = diagnose_probe(job, report["run"], source_sha)
            stable = copy.deepcopy(job)
            stable.update(**diagnostic, physical_batch_size=physical, enabled=True, calibration_status="passed",
                          measured_peak_process_gib=report["peak_process_gib"],
                          calibration_probe_path=report["run"],
                          measured_within_target=9.5 <= report["peak_process_gib"] <= 10.3)
            return stable
    raise RuntimeError(f"No tested physical batch fits GPU {job['gpu']}; no formal training started")


def preflight(jobs, root):
    for job in jobs:
        run = root / "runs" / job["experiment_id"]
        if run.exists() and any(run.iterdir()):
            raise FileExistsError(f"Fresh output already exists: {run}; this entry never resumes old checkpoints")
    if (root / "launchers/2080ti_4_2/status.json").exists():
        raise FileExistsError("This output root already has a launcher record; choose a new --output-root")
    return {"time": timestamp(), "hostname": socket.gethostname(), "repo": str(REPO),
            "gpus": available_gpus(jobs), **audit_inputs(jobs)}


def worker(args):
    root = args.output_root.resolve()
    launcher = root / "launchers/2080ti_4_2"
    launcher.mkdir(parents=True, exist_ok=True)
    request = json.loads((launcher / "launch_request.json").read_text())
    jobs = request["jobs"]
    locks, children, records = [], [], []
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        raise KeyboardInterrupt("Stable retraining launcher stopped")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    lock_handle = os.fdopen(os.dup(args.launch_lock_fd), "a") if args.launch_lock_fd is not None else (launcher / "launcher.lock").open("a")
    with lock_handle as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            require_source(request["source_sha256"])
            audit = preflight(jobs, root)
            locks = lock_gpus(audit["gpus"])
            available_gpus(jobs)
            snapshot = capture_source(REPO, root / "provenance")
            audit["source_snapshot"] = snapshot
            write_json(launcher / "preflight.json", audit)
            selected = []
            for job, gpu in zip(jobs, audit["gpus"]):
                write_json(launcher / "status.json", {"status": "calibrating", "pid": os.getpid(), "time": timestamp(), "experiment_id": job["experiment_id"], "jobs": records})
                selected.append(calibrate_job(job, root, gpu, locks, snapshot["source_sha256"]))
            require_source(snapshot["source_sha256"])
            # Recheck both cards after every smoke run has finished and before
            # starting either formal child. Other GPU 0/1 work is irrelevant.
            available_gpus(selected)
            if audit_inputs(selected)["input_sha256"] != audit["input_sha256"]:
                raise RuntimeError("Frozen inputs changed during calibration; formal training was not started")
            write_json(root / "calibrated_jobs.json", selected)
            for job, gpu in zip(selected, audit["gpus"]):
                out = root / "runs" / job["experiment_id"]
                out.mkdir(parents=True, exist_ok=False)
                command = command_for(job, out)
                stdout, stderr = (out / "stdout.log").open("w"), (out / "stderr.log").open("w")
                child = subprocess.Popen(command, cwd=REPO, env=environment(gpu), stdout=stdout, stderr=stderr,
                                         start_new_session=True, pass_fds=tuple(locks))
                children.append((child, stdout, stderr))
                record = {"experiment_id": job["experiment_id"], "original_experiment_id": job["original_experiment_id"],
                          "status": "running", "pid": child.pid, "job": job, "output_dir": str(out),
                          "gpu": gpu, "command": command, "started_at": timestamp(), "source_version": snapshot["source_version"]}
                records.append(record)
                write_json(out / "launch_record.json", record)
            while any(child.poll() is None for child, _, _ in children):
                for (child, _, _), record in zip(children, records):
                    if child.poll() is not None and record["status"] == "running":
                        record.update(status=legacy.completion_status(record, child.returncode, False), returncode=child.returncode, finished_at=timestamp())
                        write_json(Path(record["output_dir"]) / "launch_record.json", record)
                write_json(launcher / "status.json", {"status": "running", "pid": os.getpid(), "time": timestamp(), "jobs": records})
                time.sleep(5)
            for (child, _, _), record in zip(children, records):
                record.update(status=legacy.completion_status(record, child.returncode, False), returncode=child.returncode, finished_at=timestamp())
                write_json(Path(record["output_dir"]) / "launch_record.json", record)
            failed = any(row["status"] != "completed" for row in records)
            write_json(launcher / "status.json", {"status": "failed" if failed else "completed", "jobs": records, "time": timestamp()})
            return int(failed)
        except BaseException as exc:
            for record in records:
                if record["status"] == "running":
                    record.update(status="stopped" if stopping else "failed", error="launcher stopped its own child after: " + str(exc))
                    write_json(Path(record["output_dir"]) / "launch_record.json", record)
            write_json(launcher / "status.json", {"status": "stopped" if stopping else "failed", "pid": os.getpid(), "jobs": records, "error": str(exc), "time": timestamp()})
            raise
        finally:
            for child, stdout, stderr in children:
                stop_owned(child)
                stdout.close()
                stderr.close()
            for descriptor in locks:
                os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("start", "worker", "status", "preflight"), default="start")
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("CUS100_STABLE_OUTPUT_ROOT", DEFAULT_OUTPUT)))
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--road-root", type=Path)
    parser.add_argument("--synthetic-root", type=Path)
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--launch-lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    launcher = args.output_root.resolve() / "launchers/2080ti_4_2"
    if args.mode == "status":
        print(json.dumps(compact_status(args.output_root.resolve()), indent=2))
        return 0
    if args.mode == "worker":
        return worker(args)
    jobs = stable_jobs(artifact_root=args.artifact_root, road_root=args.road_root, synthetic_root=args.synthetic_root)
    if args.mode == "preflight":
        print(json.dumps(preflight(jobs, args.output_root.resolve()), indent=2))
        return 0
    launcher.mkdir(parents=True, exist_ok=True)
    with (launcher / "launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit = preflight(jobs, args.output_root.resolve())
        snapshot = capture_source(REPO, args.output_root.resolve() / "provenance")
        request = {"time": timestamp(), "jobs": jobs, "source_sha256": snapshot["source_sha256"],
                   "hostname": socket.gethostname(), "input_audit": audit}
        write_json(launcher / "launch_request.json", request)
        if args.foreground:
            args.launch_lock_fd = lock.fileno()
            return worker(args)
        command = [sys.executable, str(Path(__file__).resolve()), "--mode", "worker", "--output-root", str(args.output_root.resolve()), "--launch-lock-fd", str(lock.fileno())]
        with (launcher / "launcher.log").open("a") as log:
            child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     start_new_session=True, pass_fds=(lock.fileno(),))
        request.update(pid=child.pid, command=command)
        write_json(launcher / "launch_request.json", request)
    print(json.dumps({"status": "starting", "pid": child.pid, "status_file": str(launcher / "status.json"), "launcher_log": str(launcher / "launcher.log")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
