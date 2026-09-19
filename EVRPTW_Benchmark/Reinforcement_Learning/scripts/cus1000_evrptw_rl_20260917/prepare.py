#!/usr/bin/env python3
"""Verify existing Road Cus1000 data and freeze one global training-ID stream."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

if __package__:
    from .common import CONFIG, MODELS, OUTPUT, REPO, TRAIN_INDEX, VAL_INDEX, digest, load_config, resolve_road_root, timestamp, write_json
else:
    from common import CONFIG, MODELS, OUTPUT, REPO, TRAIN_INDEX, VAL_INDEX, digest, load_config, resolve_road_root, timestamp, write_json
sys.path.insert(0, str(REPO))


def inspect_data(root, config, *, load_samples=True):
    import pandas as pd
    from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import load_objective
    from EVRPTW_Benchmark.Reinforcement_Learning.common.reward_contract import load_reward_contract

    root = Path(root).resolve()
    hashes = {"train_index_sha256": digest(root / TRAIN_INDEX),
              "validation_index_sha256": digest(root / VAL_INDEX)}
    for name, actual in hashes.items():
        if actual != config["expected_" + name]:
            raise ValueError(f"Frozen Road release hash mismatch for {name}: {actual}")
    index = pd.read_parquet(root / TRAIN_INDEX)
    train = index.loc[(index.customer_count == 1000) & (index.split_id == "train") & (index.track_id == "train")]
    val_all = pd.read_parquet(root / VAL_INDEX)
    val = val_all.loc[(val_all.customer_count == 1000) & (val_all.split_id == "val") & (val_all.track_id == "validation")]
    if len(train) != 5000 or len(val) != 500:
        raise ValueError(f"Expected 5000 Cus1000 train and 500 val, found {len(train)} / {len(val)}")
    if train.view_id.duplicated().any() or val.view_id.duplicated().any():
        raise ValueError("Repeated train/val view IDs")
    if set(train.view_id) & set(val.view_id) or set(train.family_id) & set(val.family_id):
        raise ValueError("Training and validation views/families overlap")
    if set(train.terminal_count) != {1051} or set(val.terminal_count) != {1051}:
        raise ValueError("Expected 1051 nodes in every Cus1000 view")
    family_root = root / "materialized/families"
    for family in set(train.family_id) | set(val.family_id):
        if not (family_root / family / "family_manifest.json").is_file():
            raise FileNotFoundError(f"Missing materialized family {family} under {family_root}")
    objective = load_objective(REPO / config["objective_config"])
    reward = load_reward_contract(REPO / config["reward_contract"])
    scale = reward.for_scale("Cus1000", objective=objective)
    if load_samples:
        from EVRPTW_Benchmark.Reinforcement_Learning.common.stage2_data import Stage2TaskPool
        for split, path in (("train", TRAIN_INDEX), ("val", VAL_INDEX)):
            pool = Stage2TaskPool(root / path, scale="Cus1000", split_ids=split, representation="G")
            instance = pool.instance(pool.tasks[0])
            if instance.distance_matrix_km.shape != (1051, 1051):
                raise ValueError(f"Unexpected materialized {split} matrix shape")
    auxiliary_path = REPO / config["method_auxiliary_profile"] if config.get("method_auxiliary_profile") else None
    return {"method_auxiliary_profile_file_sha256": digest(auxiliary_path) if auxiliary_path else None,
            "root": str(root), **hashes, "train_views": len(train), "validation_views": len(val),
            "train_families": int(train.family_id.nunique()), "validation_families": int(val.family_id.nunique()),
            "train_val_view_overlap": 0, "train_val_family_overlap": 0,
            "objective": objective.to_dict(), "objective_config_sha256": digest(REPO / config["objective_config"]),
            "reward_contract_sha256": reward.digest, "reward_contract_file_sha256": digest(REPO / config["reward_contract"]),
            "reward_objective_scale": scale.objective_scale, "reward_failure_base": scale.failure_base,
            "reward_unserved_coefficient": scale.unserved_coefficient, "test_data_read": False}


def prepare_stream(root, artifact_root, config, *, audit=None):
    # Two different models can share one deterministic stream when budgets match.
    # Serialize creation of its parquet + manifest pair across launcher processes.
    directory = Path(artifact_root).resolve() / "streams"
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / f".road_cus1000_seed{config['seed']}_n{config['sample_count']}.lock"
    with lock.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return _prepare_stream_unlocked(root, artifact_root, config, audit=audit)


def _prepare_stream_unlocked(root, artifact_root, config, *, audit=None):
    import pandas as pd
    from EVRPTW_Benchmark.Reinforcement_Learning.common.training_stream import (
        atomic_write_stream, build_training_stream, load_training_stream_contract,
    )
    audit = audit or inspect_data(root, config)
    root = Path(root).resolve()
    path = Path(artifact_root).resolve() / "streams" / f"road_cus1000_seed{config['seed']}_n{config['sample_count']}.parquet"
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    if path.exists() != sidecar.exists():
        raise RuntimeError(f"Incomplete stream artifact; inspect {path} and {sidecar}")
    if not path.exists():
        index = pd.read_parquet(root / TRAIN_INDEX)
        frame, metadata = build_training_stream(index, scale="Cus1000", seed=config["seed"], sample_count=config["sample_count"])
        metadata.update(source_index=str(root / TRAIN_INDEX), source_index_sha256=audit["train_index_sha256"],
                        source_kind="stage2_road", allowed_family_ids_source=None, allowed_family_ids_sha256=None)
        atomic_write_stream(path, frame, metadata)
    contract = load_training_stream_contract(path)
    expected = {"source_index_sha256": audit["train_index_sha256"], "sample_count": config["sample_count"],
                "seed": config["seed"], "scale": "Cus1000"}
    if any(contract.get(k) != value for k, value in expected.items()):
        raise ValueError(f"Existing stream does not match the current data/seed/global budget: {path}")
    result = {"schema": "cus1000_global_stream_preparation_v1", "time": timestamp(), "data": audit,
              "path": str(path), "file_sha256": digest(path), "contract": contract,
              "world_size": config["world_size"], "local_physical_batch": config["physical_batch_size"],
              "gradient_accumulation_steps": config["gradient_accumulation_steps"],
              "global_effective_batch": config["effective_batch_size"]}
    write_json(path.with_suffix(".preparation.json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="evrptw_rl")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--road-root")
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("CUS1000_OUTPUT_ROOT", OUTPUT)))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=os.environ.get("CUS1000_BATCH_SIZE"))
    parser.add_argument("--accumulation-steps", type=int, default=os.environ.get("CUS1000_ACCUMULATION_STEPS"))
    parser.add_argument("--instance-cache-size", type=int, default=os.environ.get("CUS1000_INSTANCE_CACHE_SIZE"))
    args = parser.parse_args()
    config = load_config(args.config, model=args.model, gpus=args.gpus, batch=args.batch_size, accumulation=args.accumulation_steps, cache_size=args.instance_cache_size)
    result = prepare_stream(resolve_road_root(args.road_root), args.output_root / "artifacts", config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
