"""Prepare, run and inspect independent TERRAN stable-cost experiments.

Examples (run with the project's Python environment)::

    python -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.stable_cli prepare \
        --scale Cus500 --dataset-root /path/to/us_11city --output-dir /tmp/example
    python -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.stable_cli launch \
        --scale Cus1000 --dataset-root /path/to/us_11city --output-dir /path/to/new/run --gpu 1
    python -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.stable_cli status \
        --output-dir /path/to/new/run

Every launch has a new directory and a resolved configuration. This entry point
does not reuse the old frozen reward contract or finite training ID streams.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE = "EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.stable_cli"
PROFILE_ROOT = Path(__file__).resolve().parent / "configs" / "stable_cost_v1"
ALGORITHM = "stable_cost_v1"


def parse_scale(value: str) -> str:
    match = re.fullmatch(r"(?i:cus)([1-9][0-9]*)", str(value))
    if match is None:
        raise argparse.ArgumentTypeError("scale must be Cus followed by a positive integer, e.g. Cus500")
    return f"Cus{int(match.group(1))}"


def parse_gpus(value: str) -> tuple[int, ...]:
    parts = str(value).split(",")
    if not parts or any(not part.strip().isdigit() for part in parts):
        raise argparse.ArgumentTypeError("GPUs must be comma-separated nonnegative IDs, e.g. 0,1,2,3")
    selected = tuple(int(part.strip()) for part in parts)
    if len(set(selected)) != len(selected):
        raise argparse.ArgumentTypeError("GPU IDs must be unique")
    return selected


def _json_write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _dataset_root(selected: str | None) -> Path:
    if selected or os.environ.get("EVRPTW_DATASET_ROOT"):
        return Path(selected or os.environ["EVRPTW_DATASET_ROOT"]).expanduser().resolve()
    candidates = [REPO_ROOT / "EVRPTW_Dataset/Instances_v2/us_11city",
                  Path("/data/evrptw_runtime/EVRPTW_Dataset/Instances_v2/us_11city")]
    for candidate in candidates:
        if (candidate / "generation_plan/core/train/view_index.parquet").is_file():
            return candidate.resolve()
    raise FileNotFoundError("provide --dataset-root containing generation_plan/core/train/view_index.parquet")


def _index_summary(path: Path, customers: int, *, split: str, track: str) -> dict[str, Any]:
    import pandas as pd

    frame = pd.read_parquet(path, columns=["view_id", "customer_count", "charging_station_count",
                                         "split_id", "track_id"])
    selected = frame[(frame.customer_count == customers) & (frame.split_id == split)
                     & (frame.track_id == track)]
    if selected.empty:
        available = sorted(int(n) for n in frame.customer_count.unique())
        raise ValueError(f"Cus{customers} absent from {path}; available customer counts: {available}")
    if selected.view_id.duplicated().any():
        raise ValueError(f"duplicate view IDs in {path}")
    stations = sorted(int(n) for n in selected.charging_station_count.unique())
    if len(stations) != 1:
        raise ValueError(f"Cus{customers} has varying charging station counts: {stations}")
    return {"path": str(path), "sha256": _sha256(path), "views": len(selected),
            "num_charging_stations": stations[0], "split_id": split, "track_id": track}


def resolve_config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    scale = parse_scale(args.scale)
    customers = int(scale.removeprefix("Cus"))
    machine_profile = getattr(args, "machine_profile", None)
    profile_root = PROFILE_ROOT / machine_profile if machine_profile else PROFILE_ROOT
    exact_profile = profile_root / f"cus{customers}.yaml"
    fallback_profile = False
    if args.config:
        if machine_profile:
            raise ValueError("--config and --machine-profile are mutually exclusive")
        profile = Path(args.config).expanduser().resolve()
    elif exact_profile.is_file():
        profile = exact_profile
    else:
        if machine_profile:
            raise ValueError(f"machine profile {machine_profile} has no {scale} configuration")
        bucket = 100 if customers <= 100 else 500 if customers <= 500 else 1000
        profile = profile_root / f"cus{bucket}.yaml"
        fallback_profile = True
    cfg = copy.deepcopy(yaml.safe_load(profile.read_text()))
    if cfg.get("training", {}).get("algorithm") != ALGORITHM:
        raise ValueError(f"profile must select training.algorithm={ALGORITHM}")
    if cfg.get("model", {}).get("critic_mode") != ALGORITHM:
        raise ValueError(f"profile must select model.critic_mode={ALGORITHM}")
    if cfg.get("reward_contract") or cfg.get("protocol"):
        raise ValueError("stable experiments must not inherit an old reward contract or protocol")
    root = _dataset_root(args.dataset_root)
    train_path = root / "generation_plan/core/train/view_index.parquet"
    val_path = root / "generation_plan/core/val/view_index.parquet"
    train_index = _index_summary(train_path, customers, split="train", track="train")
    val_index = _index_summary(val_path, customers, split="val", track="validation")
    family_root = root / "materialized/families"
    if not family_root.is_dir():
        raise FileNotFoundError(f"materialized family root does not exist: {family_root}")
    cfg["data"].update({"stage2_dataset_path": str(train_path), "stage2_family_root": str(family_root),
                        "stage2_scale": scale, "num_customers": customers,
                        "num_charging_stations": train_index["num_charging_stations"],
                        "training_index_sha256": train_index["sha256"]})
    if cfg["data"].get("stage2_training_stream_path"):
        raise ValueError("stable experiments use their own deterministic sampler")
    evaluation = cfg.setdefault("evaluation", {})
    evaluation.update({"eval_path": str(val_path), "eval_family_root": str(family_root),
                       "eval_scale": scale, "eval_split_ids": "val", "eval_track_ids": "validation",
                       "validation_index_sha256": val_index["sha256"],
                       "eval_seed": int(args.seed) + 910_000_000})
    training = cfg["training"]
    minimum_budget = math.ceil(1.9 * customers)
    if fallback_profile:
        training["rollout_steps"] = minimum_budget
        evaluation["eval_max_steps"] = minimum_budget
    else:
        training["rollout_steps"] = max(int(training["rollout_steps"]), minimum_budget)
        evaluation["eval_max_steps"] = max(int(evaluation["eval_max_steps"]), minimum_budget)
    for option, field in [("epochs", "epochs"), ("physical_batch_size", "num_envs_per_gpu"),
                          ("n_traj", "n_traj"), ("rollout_steps", "rollout_steps"),
                          ("ppo_step_chunk_size", "ppo_step_chunk_size")]:
        value = getattr(args, option, None)
        if value is not None:
            training[field] = int(value)
    physical = int(training["num_envs_per_gpu"])
    selected_gpus = getattr(args, "gpus", None)
    requested_world = getattr(args, "world_size", None)
    if selected_gpus is not None and args.gpu is not None:
        raise ValueError("--gpu and --gpus are mutually exclusive")
    if selected_gpus is not None:
        if requested_world is not None and int(requested_world) != len(selected_gpus):
            raise ValueError("--world-size must equal the number of --gpus")
        world_size = len(selected_gpus)
    else:
        world_size = int(requested_world if requested_world is not None else
                         training.get("distributed_world_size", 1))
    if world_size < 1:
        raise ValueError("--world-size must be positive")
    if args.gpu is not None and world_size != 1:
        raise ValueError("multi-GPU profiles require --gpus or an inherited visible GPU list")
    if world_size != 1 or "distributed_world_size" in training:
        training["distributed_world_size"] = world_size
    effective = int(args.effective_batch_size or training["effective_batch_size"])
    if physical <= 0 or effective <= 0 or effective % (physical * world_size):
        raise ValueError("effective batch must be a positive integer multiple of physical batch times world size")
    training["effective_batch_size"] = effective
    training["logical_microbatches_per_epoch"] = effective // (physical * world_size)
    for key in ["epochs", "n_traj", "rollout_steps", "ppo_step_chunk_size"]:
        if int(training[key]) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if int(training["n_traj"]) < 2:
        raise ValueError("stable feasibility LOO baseline requires training.n_traj >= 2")
    for option, field in [("warm_start_checkpoint", "actor_warm_start"), ("resume", "resume_checkpoint")]:
        checkpoint = getattr(args, option, None)
        if checkpoint:
            training[field] = str(Path(checkpoint).expanduser().resolve(strict=True))
    if getattr(args, "resume", None) and not getattr(args, "warm_start_checkpoint", None):
        training.pop("actor_warm_start", None)
    if getattr(args, "warm_start_checkpoint", None) and not getattr(args, "resume", None):
        training.pop("resume_checkpoint", None)
    training["allow_batch_resize_resume"] = bool(getattr(args, "allow_batch_resize_resume", False))
    if training["allow_batch_resize_resume"] and not training.get("resume_checkpoint"):
        raise ValueError("--allow-batch-resize-resume requires --resume")
    training["allow_dataset_relocation_resume"] = bool(getattr(args, "allow_dataset_relocation_resume", False))
    if training["allow_dataset_relocation_resume"] and not training.get("resume_checkpoint"):
        raise ValueError("--allow-dataset-relocation-resume requires --resume")
    if not isinstance(cfg["objective"], dict):
        objective = Path(cfg["objective"])
        if not objective.is_absolute():
            objective = REPO_ROOT / objective
        cfg["objective"] = json.loads(objective.read_text())
    cfg["output_dir"] = str(Path(args.output_dir).expanduser().resolve())
    cfg["seed"] = int(args.seed)
    resume = None
    if training.get("resume_checkpoint"):
        import torch
        from .stable_trainer import resolve_stable_config, validate_resume_checkpoint

        payload = torch.load(training["resume_checkpoint"], map_location="cpu", weights_only=False)
        resume = validate_resume_checkpoint(payload, resolve_stable_config(cfg), seed=int(args.seed),
                                            source=training["resume_checkpoint"])
    metadata = {"schema": "terran_stable_launch_v1", "algorithm": ALGORITHM,
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(),
                "profile_path": str(profile), "profile_sha256": _sha256(profile),
                "requested_scale": scale, "uses_bucket_profile": fallback_profile,
                "seed": int(args.seed), "gpu": args.gpu, "python": sys.executable,
                "gpus": list(selected_gpus) if selected_gpus is not None else None,
                "world_size": world_size, "machine_profile": machine_profile,
                "objective": cfg["objective"], "dataset_root": str(root),
                "train_index": train_index, "validation_index": val_index,
                "sampling": {"mode": "seeded_shuffle_cycle_without_replacement",
                             "seed": int(args.seed), "uses_old_registered_stream": False,
                             "sampled_ids_artifact": "sampled_view_ids.jsonl"},
                "resolved_config": cfg}
    if resume is not None:
        metadata["resume"] = resume
    for field in ["actor_warm_start", "resume_checkpoint"]:
        if training.get(field):
            source = Path(training[field])
            metadata[field] = {"path": str(source), "sha256": _sha256(source)}
    return cfg, metadata


def prepare(args: argparse.Namespace) -> Path:
    cfg, metadata = resolve_config(args)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    _json_write(output / "provenance.json", metadata)
    return config_path


def run(config_path: Path, *, device: str, gpu: int | None) -> None:
    import fcntl

    cfg = yaml.safe_load(config_path.read_text())
    if cfg.get("training", {}).get("algorithm") != ALGORITHM:
        raise ValueError("resolved config does not select stable_cost_v1")
    for section, path_key, hash_key in [("data", "stage2_dataset_path", "training_index_sha256"),
                                        ("evaluation", "eval_path", "validation_index_sha256")]:
        settings = cfg[section]
        expected_hash = settings.get(hash_key)
        if expected_hash and _sha256(Path(settings[path_key])) != expected_hash:
            raise ValueError(f"{section} index changed since configuration preparation")
    output = Path(cfg["output_dir"])
    expected_world = int(cfg["training"].get("distributed_world_size", 1))
    actual_world = int(os.environ.get("WORLD_SIZE", "1"))
    if actual_world != expected_world:
        raise ValueError(f"configuration requires {expected_world} workers, got {actual_world}; "
                         "use stable_cli launch or torchrun with the matching process count")
    if gpu is not None and actual_world != 1:
        raise ValueError("set visible GPUs on the launcher, not independently in torchrun workers")
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    dist = None
    rank = 0
    worker_pids = [os.getpid()]
    if actual_world > 1:
        import torch
        import torch.distributed as distributed

        dist = distributed
        local_rank = int(os.environ["LOCAL_RANK"])
        if device == "cuda":
            if local_rank >= torch.cuda.device_count():
                raise ValueError("not enough visible CUDA devices for the configured worker count")
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        timeout_s = int(os.environ.get("TERRAN_DISTRIBUTED_TIMEOUT_S", "7200"))
        if timeout_s <= 0:
            raise ValueError("TERRAN_DISTRIBUTED_TIMEOUT_S must be positive")
        dist.init_process_group(backend="nccl" if device.startswith("cuda") else "gloo",
                                timeout=timedelta(seconds=timeout_s))
        rank = dist.get_rank()
        worker_pids = [None] * actual_world
        dist.all_gather_object(worker_pids, os.getpid())
    lock = None
    state = None
    try:
        setup_error = None
        if rank == 0:
            try:
                lock = (output / "run.lock").open("a")
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if (output / "run_state.json").exists():
                    raise FileExistsError("this run has already started; use a fresh output directory")
                state = {"pid": os.getpid(), "worker_pids": worker_pids,
                         "world_size": actual_world, "status": "running", "started_at": time.time(),
                         "config": str(config_path.resolve())}
                _json_write(output / "run_state.json", state)
                (output / "train.pid").write_text(f"{os.getpid()}\n")
            except BaseException as error:
                if dist is None:
                    raise
                setup_error = f"{type(error).__name__}: {error}"
        if dist is not None:
            message = [setup_error]
            dist.broadcast_object_list(message, src=0)
            if message[0] is not None:
                raise RuntimeError(f"distributed run setup failed: {message[0]}")
        try:
            from .trainer import train_from_config

            result = train_from_config(cfg, seed=int(cfg["seed"]), device=device)
        except BaseException as error:
            if rank == 0:
                state.update(status="failed", completed_at=time.time(), error=f"{type(error).__name__}: {error}")
                _json_write(output / "run_state.json", state)
            else:
                _json_write(output / f"worker_{rank}_error.json",
                            {"rank": rank, "pid": os.getpid(), "error": f"{type(error).__name__}: {error}"})
            raise
        if rank == 0:
            state.update(status="completed", completed_at=time.time(), result=str(result))
            _json_write(output / "run_state.json", state)
    finally:
        if lock is not None:
            lock.close()
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def launch_command(config_path: Path, *, device: str, gpu: int | None,
                   gpus: tuple[int, ...] | None) -> tuple[list[str], dict[str, str]]:
    cfg = yaml.safe_load(config_path.read_text())
    world_size = int(cfg["training"].get("distributed_world_size", 1))
    if world_size < 1:
        raise ValueError("configured distributed_world_size must be positive")
    if gpu is not None and (gpus is not None or world_size != 1):
        raise ValueError("--gpu selects one worker; use --gpus for a multi-GPU run")
    if gpus is not None and len(gpus) != world_size:
        raise ValueError("the number of --gpus must match the prepared world size")
    if device == "cpu" and (gpu is not None or gpus is not None):
        raise ValueError("GPU selection cannot be used with --device cpu")
    environment = os.environ.copy()
    # A detached launcher is a new job, including if invoked from a torchrun shell.
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                 "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    if gpus is not None:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in gpus)
    elif gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONUNBUFFERED"] = "1"
    worker = [MODULE, "run", "--resolved-config", str(config_path), "--device", device]
    if world_size > 1:
        command = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
                   "--nnodes=1", f"--nproc_per_node={world_size}", "--max_restarts=0", "--module", *worker]
    else:
        command = [sys.executable, "-u", "-m", *worker]
    return command, environment


def _latest_json(path: Path) -> Any:
    if not path.is_file():
        return None
    with path.open("rb") as source:
        source.seek(0, os.SEEK_END)
        source.seek(max(0, source.tell() - 65536))
        lines = source.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (ValueError, UnicodeError):
            pass
    return None


def status(output: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"output_dir": str(output)}
    for filename in ["run_state.json", "launch_process.json", "progress.json"]:
        path = output / filename
        if path.is_file():
            result[filename.removesuffix(".json")] = json.loads(path.read_text())
    pid_path = output / "train.pid"
    result["process_matches_run"] = False
    pid = (int(pid_path.read_text().strip()) if pid_path.is_file() else
           result.get("launch_process", {}).get("pid"))
    if pid is not None:
        result["pid"] = pid
        result["process_role"] = "training_worker" if pid_path.is_file() else "launcher"
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode()
            result["process_matches_run"] = MODULE in command and str(output) in command
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    for name in ["train_metrics.jsonl", "training_history.jsonl", "metrics.jsonl", "validation_history.jsonl"]:
        latest = _latest_json(output / name)
        if latest is not None:
            result[name] = latest
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ["prepare", "run", "launch"]:
        sub = commands.add_parser(command)
        sub.add_argument("--scale", type=parse_scale, default="Cus500")
        sub.add_argument("--config")
        sub.add_argument("--machine-profile", choices=["4x2080ti"],
                         help="select a conservative four-GPU profile for Cus500 or Cus1000")
        sub.add_argument("--dataset-root")
        sub.add_argument("--output-dir")
        sub.add_argument("--seed", type=int, default=1234)
        gpu_selection = sub.add_mutually_exclusive_group()
        gpu_selection.add_argument("--gpu", type=int)
        gpu_selection.add_argument("--gpus", type=parse_gpus,
                                   help="visible local GPU IDs, e.g. 0,1,2,3")
        sub.add_argument("--world-size", type=int,
                         help="number of synchronous workers; defaults to the selected GPUs/profile")
        sub.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
        checkpoint = sub.add_mutually_exclusive_group()
        checkpoint.add_argument("--warm-start-checkpoint")
        checkpoint.add_argument("--resume", help="resume stable-cost state into a new run directory")
        sub.add_argument("--allow-batch-resize-resume", action="store_true",
                         help="allow physical/effective batch, worker count and replay chunk changes on resume")
        sub.add_argument("--allow-dataset-relocation-resume", action="store_true",
                         help="allow dataset mount paths to change when the saved index identity matches")
        for name in ["epochs", "physical-batch-size", "effective-batch-size", "n-traj", "rollout-steps", "ppo-step-chunk-size"]:
            sub.add_argument(f"--{name}", type=int)
        if command in {"run", "launch"}:
            sub.add_argument("--resolved-config", type=Path)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(status(args.output_dir.expanduser().resolve()), indent=2))
        return
    if args.gpu is not None and args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    config_path = getattr(args, "resolved_config", None)
    if config_path is None:
        if not args.output_dir:
            parser.error("--output-dir is required when preparing a fresh configuration")
        config_path = prepare(args)
    else:
        ignored = [name for name in ["config", "dataset_root", "output_dir", "epochs", "physical_batch_size",
                                     "effective_batch_size", "n_traj", "rollout_steps", "ppo_step_chunk_size",
                                     "warm_start_checkpoint", "resume", "world_size", "machine_profile"]
                   if getattr(args, name, None) is not None]
        if args.allow_batch_resize_resume:
            ignored.append("allow_batch_resize_resume")
        if args.allow_dataset_relocation_resume:
            ignored.append("allow_dataset_relocation_resume")
        if ignored:
            parser.error(f"--resolved-config cannot be combined with configuration overrides: {', '.join(ignored)}")
        config_path = config_path.expanduser().resolve(strict=True)
        resolved = yaml.safe_load(config_path.read_text())
        if Path(resolved["output_dir"]).resolve() != config_path.parent:
            parser.error("resolved configuration must reside in its recorded output directory")
    output = config_path.resolve().parent
    if args.command == "prepare":
        print(json.dumps({"resolved_config": str(config_path), "output_dir": str(output)}))
    elif args.command == "run":
        if args.gpus is not None:
            if int(os.environ.get("WORLD_SIZE", "1")) > 1:
                parser.error("select --gpus on the launcher, not in torchrun workers")
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in args.gpus)
        run(config_path, device=args.device, gpu=args.gpu)
    else:
        command, environment = launch_command(config_path, device=args.device, gpu=args.gpu, gpus=args.gpus)
        with (output / "stdout.log").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
            process = subprocess.Popen(command, cwd=REPO_ROOT, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr, start_new_session=True)
        launched = {"pid": process.pid, "command": command, "gpu": args.gpu,
                    "gpus": list(args.gpus) if args.gpus is not None else None,
                    "world_size": int(yaml.safe_load(config_path.read_text())["training"].get("distributed_world_size", 1)),
                    "output_dir": str(output)}
        _json_write(output / "launch_process.json", launched)
        print(json.dumps(launched))


if __name__ == "__main__":
    main()
