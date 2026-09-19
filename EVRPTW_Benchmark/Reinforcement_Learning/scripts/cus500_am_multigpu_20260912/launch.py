#!/usr/bin/env python3
"""Launch one Road Cus500 AM model on multiple GPUs, without touching Cus100."""
from __future__ import annotations

import argparse
import csv
import fcntl
import importlib
import io
import json
import math
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

if __package__:
    from .common import CONFIG, HERE, OUTPUT, REPO, TRAIN_INDEX, VAL_INDEX, load_config, resolve_road_root, source_snapshot, timestamp, write_json
    from .prepare import inspect_data, prepare_stream
else:
    from common import CONFIG, HERE, OUTPUT, REPO, TRAIN_INDEX, VAL_INDEX, load_config, resolve_road_root, source_snapshot, timestamp, write_json
    from prepare import inspect_data, prepare_stream

DESKTOP_EXECUTABLE = "/usr/libexec/gnome-remote-desktop-daemon"
DESKTOP_LIMIT_MIB = 1024


def gpu_inventory():
    fields = ["index", "uuid", "name", "memory.total", "memory.used", "utilization.gpu", "driver_version"]
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=" + ",".join(fields),
                                   "--format=csv,noheader,nounits"], text=True)
    result = []
    for values in csv.reader(io.StringIO(raw)):
        if not values:
            continue
        row = dict(zip(fields, [v.strip() for v in values]))
        for key in ("index", "memory.total", "memory.used", "utilization.gpu"):
            row[key] = int(row[key])
        result.append(row)
    return result


def gpu_processes():
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                   "--format=csv,noheader,nounits"], text=True)
    result = []
    for values in csv.reader(io.StringIO(raw)):
        if not values:
            continue
        gpu, raw_pid, memory = [v.strip() for v in values]
        pid = int(raw_pid)
        try:
            executable = os.readlink(f"/proc/{pid}/exe")
        except OSError:
            executable = None
        result.append({"gpu_uuid": gpu, "pid": pid,
                       "used_memory_mib": int(memory) if memory.isdigit() else None,
                       "executable": executable})
    return result


def validate_gpus(config, inventory, processes):
    selected = []
    for physical in config["gpus"]:
        rows = [row for row in inventory if row["index"] == physical]
        if len(rows) != 1 or "2080 Ti" not in rows[0]["name"]:
            raise RuntimeError(f"GPU {physical} is not an available RTX 2080 Ti")
        gpu = dict(rows[0])
        if gpu["memory.used"] >= DESKTOP_LIMIT_MIB:
            raise RuntimeError(f"GPU {physical} already uses {gpu['memory.used']} MiB; no task will be stopped")
        allowed = []
        for process in processes:
            if process["gpu_uuid"] != gpu["uuid"]:
                continue
            memory = process["used_memory_mib"]
            desktop = (physical == 0 and process["executable"] == DESKTOP_EXECUTABLE
                       and memory is not None and 0 <= memory < DESKTOP_LIMIT_MIB)
            if not desktop:
                raise RuntimeError(f"GPU {physical} has compute PID {process['pid']} ({process['executable']}); no task will be stopped")
            allowed.append(process)
        gpu["preserved_desktop_processes"] = allowed
        gpu["free_memory_mib"] = gpu["memory.total"] - gpu["memory.used"]
        required = math.ceil(config["target_process_memory_gib"][1] * 1024) + 128
        if gpu["free_memory_mib"] < required:
            raise RuntimeError(f"GPU {physical} has insufficient free memory: {gpu['free_memory_mib']} MiB < {required} MiB including margin")
        selected.append(gpu)
    return selected


def environment_report():
    versions = {}
    for name in ("torch", "numpy", "pandas", "pyarrow", "scipy", "gymnasium", "numba"):
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "unknown")
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available() or not torch.distributed.is_available() or not torch.distributed.is_nccl_available():
        raise RuntimeError("Selected Python requires CUDA PyTorch with distributed NCCL support")
    module_path = REPO / "EVRPTW_Benchmark/Reinforcement_Learning/AM_EVRPTW/distributed_train.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Multi-GPU trainer missing; update to the Cus500 branch: {module_path}")
    return {"python": sys.executable, "python_version": sys.version, "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "versions": versions, "torch_cuda": torch.version.cuda,
            "cuda_visible_devices_inherited": os.environ.get("CUDA_VISIBLE_DEVICES")}


def build_command(config, root, run, stream, *, python=sys.executable, resume=False):
    command = [str(python), "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc_per_node={config['world_size']}", "--max_restarts=0", "--module",
               "EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.distributed_train"]
    values = {
        "dataset-path": Path(root) / TRAIN_INDEX, "family-root": Path(root) / "materialized/families",
        "scale": "Cus500", "split-ids": "train", "track-ids": "train", "training-representation": "G",
        "seed": config["seed"], "device": "cuda", "batch-size": config["physical_batch_size"],
        "physical-batch-size": config["physical_batch_size"], "effective-batch-size": config["effective_batch_size"],
        "training-epochs": config["training_epochs"], "minimum-training-epochs": config["minimum_training_epochs"],
        "training-rollout-steps": config["training_rollout_steps"], "validation-rollout-steps": config["validation_rollout_steps"],
        "samples-per-instance": config["samples_per_instance"], "training-stream-path": stream["path"],
        "training-stream-contract-sha256": stream["contract"]["sha256"],
        "customer-exposure-budget": config["customer_exposure_budget"],
        "validation-dataset-path": Path(root) / VAL_INDEX,
        "validation-family-root": Path(root) / "materialized/families", "validation-limit": config["validation_limit"],
        "validation-decode-type": "sampling", "validation-candidates": config["validation_candidates"],
        "validation-seed": config["validation_seed"], "validation-every-epochs": config["validation_every_epochs"],
        "post-minimum-validation-every-epochs": config["validation_every_epochs"],
        "validation-checkpoints": config["training_epochs"] // config["validation_every_epochs"],
        "early-stop-patience-validations": config["early_stop_patience_validations"],
        "early-stop-start-epoch": config["early_stop_start_epoch"], "final-validation-limit": 0,
        "objective-config": REPO / config["objective_config"], "reward-contract": REPO / config["reward_contract"],
        "optimizer": "adamw", "weight-decay": config["weight_decay"], "learning-rate": config["learning_rate"],
        "protocol-id": config["protocol_id"], "output-dir": run,
        "distributed-backend": "nccl", "distributed-timeout-seconds": config["distributed_timeout_seconds"],
        "expected-world-size": config["world_size"],
    }
    for key, value in values.items():
        command.extend(["--" + key, str(value)])
    if resume:
        command.append("--resume")
    return command


def preflight(config, root, output, *, resume=False):
    selected = validate_gpus(config, gpu_inventory(), gpu_processes())
    environment = environment_report()
    run = Path(output) / "runs" / config["run_id"]
    if resume:
        if not (run / "checkpoint_latest.pt").is_file():
            raise FileNotFoundError(f"--resume requires an existing checkpoint: {run / 'checkpoint_latest.pt'}")
        result_path = run / "training_result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            if result.get("status") in {"passed", "early_stopped"}:
                raise RuntimeError("This experiment already completed; use a new output root for a new experiment")
    elif run.exists() and any(run.iterdir()):
        raise FileExistsError(f"Output already contains a run: {run}; use explicit --resume or a new CUS500_OUTPUT_ROOT")
    target = Path(output).resolve()
    while not target.exists():
        target = target.parent
    if not os.access(target, os.W_OK | os.X_OK):
        raise PermissionError(f"Output parent is not writable: {target}")
    return {"schema": "cus500_multigpu_preflight_v1", "time": timestamp(), "host": socket.gethostname(),
            "config": config, "data": inspect_data(root, config), "gpus": selected,
            "environment": environment, "output_dir": str(run.resolve()), "resume": resume}


def lock_output(output):
    directory = Path(output) / "launchers/2080ti_3_1"
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / ".launch.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError("A Cus500 launcher is already running in this output directory") from None
    return fd, directory


def validate_resume(previous, current):
    for key in ("config", "data"):
        if previous["preflight"][key] != current["preflight"][key]:
            raise ValueError(f"Resume {key} differs from the original experiment")
    if previous["source"]["source_sha256"] != current["source"]["source_sha256"]:
        raise ValueError("Source bytes changed since the original experiment; inspect before resuming")


def worker(request_path, lock_fd):
    # An inherited flock covers startup, the whole torchrun lifetime, and cleanup.
    os.fstat(lock_fd)
    path = Path(request_path)
    request = json.loads(path.read_text())
    launcher = path.parent
    state_path = launcher / "status.json"
    state = {"schema": "cus500_multigpu_status_v1", "status": "starting", "time": timestamp(),
             "launcher_pid": os.getpid(), "host": socket.gethostname(), "command": request["command"],
             "output_dir": request["preflight"]["output_dir"], "gpus": request["preflight"]["gpus"],
             "config": request["preflight"]["config"], "request": str(path), "returncode": None}
    write_json(state_path, state)
    try:
        if source_snapshot()["source_sha256"] != request["source"]["source_sha256"]:
            raise RuntimeError("Source changed between preflight and worker startup")
        validate_gpus(state["config"], gpu_inventory(), gpu_processes())
        env = os.environ.copy()
        env.update(request["environment"])
        run = Path(state["output_dir"])
        run.mkdir(parents=True, exist_ok=True)
        with (run / "stdout.log").open("ab") as stdout, (run / "stderr.log").open("ab") as stderr:
            child = subprocess.Popen(request["command"], cwd=REPO, env=env, stdout=stdout, stderr=stderr,
                                     pass_fds=(lock_fd,))
            state.update(status="running", pid=child.pid, started_at=timestamp())
            write_json(state_path, state)
            while child.poll() is None:
                state["time"] = timestamp()
                write_json(state_path, state)
                time.sleep(10)
            code = child.returncode
        state.update(returncode=code, finished_at=timestamp(), time=timestamp())
        result_path = run / "training_result.json"
        if code == 0 and result_path.is_file() and (run / "best.ckpt").is_file():
            result = json.loads(result_path.read_text())
            state["training_result"] = result
            state["status"] = "completed" if result.get("status") in {"passed", "early_stopped"} else "failed"
        else:
            state["status"] = "failed"
            if code == 0:
                state["error"] = "Trainer exited without a final training_result.json and best.ckpt"
        write_json(state_path, state)
        return code if code else (0 if state["status"] == "completed" else 1)
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", finished_at=timestamp())
        write_json(state_path, state)
        raise
    finally:
        os.close(lock_fd)


def start(args, config, root, output):
    lock_fd, launcher = lock_output(output)
    handed_off = False
    try:
        audit = preflight(config, root, output, resume=args.resume)
        stream = prepare_stream(root, Path(output) / "artifacts", config, audit=audit["data"])
        source = source_snapshot()
        request = {"schema": "cus500_multigpu_launch_request_v1", "time": timestamp(),
                   "preflight": audit, "stream": stream, "source": source,
                   "command": build_command(config, root, audit["output_dir"], stream, resume=args.resume),
                   "environment": {"CUDA_VISIBLE_DEVICES": ",".join(gpu["uuid"] for gpu in audit["gpus"]),
                                   "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONUNBUFFERED": "1",
                                   "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2"),
                                   "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "2"),
                                   "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "2"),
                                   "NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS", "2")}}
        original = launcher / "launch_request.json"
        if args.resume:
            if not original.is_file():
                raise FileNotFoundError("Cannot resume without original launch_request.json")
            validate_resume(json.loads(original.read_text()), request)
            request_path = launcher / f"resume_request_{time.time_ns()}.json"
        else:
            request_path = original
        write_json(request_path, request)
        print(json.dumps({"world_size": config["world_size"], "physical_gpus": config["gpus"],
                          "local_batch": config["physical_batch_size"], "global_batch": config["effective_batch_size"],
                          "command": shlex.join(request["command"]), "status_file": str(launcher / "status.json")}, indent=2), flush=True)
        if args.foreground:
            handed_off = True
            return worker(request_path, lock_fd)
        with (launcher / "launcher.log").open("ab") as log:
            child = subprocess.Popen([sys.executable, str(HERE / "launch.py"), "--mode", "worker", "--request", str(request_path),
                                      "--lock-fd", str(lock_fd)], cwd=REPO, stdout=log, stderr=log,
                                     start_new_session=True, pass_fds=(lock_fd,))
        print(f"Launcher PID {child.pid}. Training starts in the background; see {launcher / 'status.json'}", flush=True)
        return 0
    finally:
        if not handed_off:
            os.close(lock_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "start", "worker", "status"), default="start")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--road-root")
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("CUS500_OUTPUT_ROOT", OUTPUT)))
    parser.add_argument("--gpus", default=os.environ.get("CUS500_GPUS", "0,1,2"))
    parser.add_argument("--batch-size", type=int, default=os.environ.get("CUS500_BATCH_SIZE"))
    parser.add_argument("--accumulation-steps", type=int, default=os.environ.get("CUS500_ACCUMULATION_STEPS"))
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mode == "worker":
        if args.request is None or args.lock_fd is None:
            parser.error("worker is internal and requires an inherited launch lock")
        return worker(args.request, args.lock_fd)
    output = args.output_root.expanduser().resolve()
    if args.mode == "status":
        path = output / "launchers/2080ti_3_1/status.json"
        print(path.read_text() if path.is_file() else f"No launch status at {path}")
        return 0
    config = load_config(args.config, gpus=args.gpus, batch=args.batch_size, accumulation=args.accumulation_steps)
    root = resolve_road_root(args.road_root)
    if args.mode == "preflight":
        print(json.dumps(preflight(config, root, output, resume=args.resume), indent=2))
        return 0
    return start(args, config, root, output)


if __name__ == "__main__":
    raise SystemExit(main())
