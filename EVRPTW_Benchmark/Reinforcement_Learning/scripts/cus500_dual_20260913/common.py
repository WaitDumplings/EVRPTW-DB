"""Portable paths and atomic records for the independent Cus500 dual-GPU experiments."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CONFIGS = HERE / "configs"
MODELS = ("rrnco", "drl_ts")
OUTPUT = REPO / "EVRPTW_Benchmark/results/cus500_dual_20260913"
RELEASE = "us_11city_full_clean_v7_bbde5db_20260823"
TRAIN_INDEX = "generation_plan/core/train/view_index.parquet"
VAL_INDEX = "generation_plan/core/val/view_index.parquet"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_gpus(value):
    fields = str(value).split(",")
    if not fields or any(not field.strip().isdigit() for field in fields):
        raise ValueError("CUS500_GPUS must be comma-separated physical GPU indices, e.g. 0,1")
    result = [int(field.strip()) for field in fields]
    if len(result) != len(set(result)) or len(result) != 2:
        raise ValueError("This deployment requires exactly two distinct GPU indices")
    return result


def resolve_road_root(value=None, repo=REPO):
    repo = Path(repo)
    requested = value or os.environ.get("CUS500_ROAD_ROOT") or os.environ.get("CUS100_ROAD_ROOT")
    if requested:
        root = Path(requested).expanduser()
        root = (repo / root).resolve() if not root.is_absolute() else root.resolve()
        if not (root / TRAIN_INDEX).is_file() or not (root / VAL_INDEX).is_file():
            raise FileNotFoundError(f"Road train/val indexes missing under {root}; set CUS500_ROAD_ROOT")
        return root
    candidates = [repo / "EVRPTW_Dataset/Instances_v2" / RELEASE,
                  repo / "EVRPTW_Dataset/Instances_v2/us_11city"]
    for root in candidates:
        if (root / TRAIN_INDEX).is_file() and (root / VAL_INDEX).is_file():
            return root.resolve()
    raise FileNotFoundError("Road release is unavailable; set CUS500_ROAD_ROOT to the extracted full release")


def load_config(path=None, *, model="rrnco", gpus="0,1", batch=None, accumulation=None, cache_size=None):
    if model not in MODELS:
        raise ValueError(f"Unsupported model: {model}")
    path = Path(path) if path is not None else CONFIGS / f"{model}.json"
    config = json.loads(path.read_text())
    if (config.get("schema") != "cus500_dual_config_v1" or config.get("scale") != "Cus500"
            or config.get("model") != model):
        raise ValueError(f"Not a Cus500 {model} dual-GPU deployment configuration")
    config["gpus"] = parse_gpus(gpus)
    if batch is not None:
        config["physical_batch_size"] = int(batch)
    if accumulation is not None:
        config["gradient_accumulation_steps"] = int(accumulation)
    if cache_size is not None:
        config["instance_cache_size"] = int(cache_size)
    if (not isinstance(config["instance_cache_size"], int) or isinstance(config["instance_cache_size"], bool)
            or config["instance_cache_size"] < 0):
        raise ValueError("instance_cache_size must be a nonnegative integer")
    for key in ("physical_batch_size", "gradient_accumulation_steps", "training_epochs",
                "minimum_training_epochs", "samples_per_instance", "validation_limit"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["minimum_training_epochs"] > config["training_epochs"]:
        raise ValueError("minimum_training_epochs exceeds the maximum")
    if config["validation_rollout_steps"] != (3 * config["training_rollout_steps"] + 1) // 2:
        raise ValueError("Validation rollout cap must be ceil(1.5 * training cap)")
    for key in ("validation_every_epochs", "early_stop_patience_validations", "training_rollout_steps"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["training_epochs"] % config["validation_every_epochs"]:
        raise ValueError("Training budget must end on a validation checkpoint")
    config["world_size"] = len(config["gpus"])
    config["effective_batch_size"] = (config["physical_batch_size"] * config["world_size"]
                                      * config["gradient_accumulation_steps"])
    config["sample_count"] = config["effective_batch_size"] * config["training_epochs"]
    config["customer_exposure_budget"] = config["sample_count"] * 500
    return config


def source_snapshot(repo=REPO):
    """Record actual source hashes, including dirty and untracked source files."""
    repo = Path(repo)
    names = set()
    for args in (["ls-files", "-z"], ["ls-files", "--others", "--exclude-standard", "-z"]):
        raw = subprocess.check_output(["git", *args], cwd=repo)
        names.update(raw.decode().strip("\0").split("\0"))
    files = []
    for name in sorted(names - {""}):
        path = repo / name
        if path.is_symlink():
            files.append({"path": name, "symlink": os.readlink(path)})
        elif path.is_file():
            files.append({"path": name, "sha256": digest(path),
                          "executable": bool(path.stat().st_mode & 0o111)})
        else:
            files.append({"path": name, "absent": True})
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema": "cus500_actual_source_v1", "source_sha256": identity,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip(),
            "files": files}
