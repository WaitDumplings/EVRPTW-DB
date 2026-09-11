#!/usr/bin/env python3
"""Freeze method-independent ID-stream prefixes for measured Cus100 batches."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from .build_manifest import build_jobs
from .launch import MANIFEST, REPO, RUN_ROOT, dataset_root, load_jobs, sha256, timestamp, write_json


def prepare(jobs, artifact_root):
    import pandas as pd
    from EVRPTW_Benchmark.Reinforcement_Learning.common.training_stream import (
        atomic_write_stream, build_training_stream, load_training_stream_contract,
        read_stream_view_ids, stream_content_sha256,
    )
    groups = {}
    for job in jobs:
        for field in ("physical_batch_size", "effective_batch_size"):
            if not isinstance(job.get(field), int) or job[field] <= 0:
                raise ValueError(f"{job['experiment_id']} needs a measured {field}")
        count = job["training_epochs"] * job["effective_batch_size"]
        job["customer_exposure_budget"] = count * 100
        job["minimum_customer_exposure_budget"] = job["minimum_training_epochs"] * job["effective_batch_size"] * 100
        root = dataset_root(job)
        key = (job["source_kind"], str(root), job["train_index"], job["seed"])
        groups.setdefault(key, []).append(job)
    reports = []
    for (source, raw_root, index_relative, seed), selected in groups.items():
        root = Path(raw_root)
        if source == "terran_synthetic":
            from .data_contract import inspect_synthetic
            synthetic_contract = inspect_synthetic(root, verify_payloads=False)
            for job in selected:
                job["data_source_manifest"] = "corpus_manifest.json"
                job["data_source_manifest_sha256"] = synthetic_contract["manifest_sha256"]
        index_path = root / index_relative
        index = pd.read_parquet(index_path)
        train = index.loc[(index["customer_count"] == 100) &
                          (index["split_id"] == "train") & (index["track_id"] == "train")]
        val_path = root / selected[0]["validation_index"]
        val = pd.read_parquet(val_path)
        val = val.loc[(val["customer_count"] == 100) & (val["split_id"] == "val")]
        if len(train) != 50000 or len(val) != 500:
            raise ValueError(f"{source}: expected 50000 train / 500 val, found {len(train)} / {len(val)}")
        if set(train.view_id) & set(val.view_id) or set(train.family_id) & set(val.family_id):
            raise ValueError(f"{source}: training and validation IDs overlap")
        source_sha = sha256(index_path)
        val_sha = sha256(val_path)
        counts = sorted({job["training_epochs"] * job["effective_batch_size"] for job in selected})
        previous_ids = None
        previous_path = None
        entries = []
        for count in counts:
            path = artifact_root / "streams" / source / f"seed{seed}_n{count}.parquet"
            manifest_path = path.with_suffix(path.suffix + ".manifest.json")
            if path.exists() != manifest_path.exists():
                raise RuntimeError(f"Incomplete pre-existing stream artifact: {path}")
            if not path.exists():
                stream, metadata = build_training_stream(index, scale="Cus100", seed=seed, sample_count=count)
                metadata.update(source_index=str(index_path), source_index_sha256=source_sha,
                                source_kind=source, allowed_family_ids_source=None,
                                allowed_family_ids_sha256=None)
                atomic_write_stream(path, stream, metadata)
                del stream
                gc.collect()
            contract = load_training_stream_contract(path)
            if (contract["source_index_sha256"] != source_sha or contract["sample_count"] != count
                    or contract["seed"] != seed or contract["scale"] != "Cus100"):
                raise RuntimeError(f"Existing stream does not match this source/seed/budget: {path}")
            ids = read_stream_view_ids(path)
            prefix = None
            if previous_ids is not None:
                if ids[:len(previous_ids)] != previous_ids:
                    raise RuntimeError(f"Method streams are not identical ordered prefixes: {previous_path}, {path}")
                prefix = {"shorter_path": str(previous_path.relative_to(REPO)),
                          "prefix_length": len(previous_ids), "all_ordered_ids_equal": True,
                          "shorter_content_sha256": stream_content_sha256(previous_ids),
                          "longer_prefix_content_sha256": stream_content_sha256(ids[:len(previous_ids)])}
            relative = str(path.relative_to(REPO))
            file_hash = sha256(path)
            for job in selected:
                if job["training_epochs"] * job["effective_batch_size"] == count:
                    job.update(training_stream_path=relative, training_stream_path_sha256=file_hash,
                               training_stream_contract_sha256=contract["sha256"],
                               training_stream_contract_snapshot=contract,
                               train_index_sha256=source_sha, validation_index_sha256=val_sha,
                               stream_integrity_mode="runtime_content_rehash", file_hash_validation_performed=True)
            entries.append({"path": relative, "count": count, "contract": contract,
                            "prefix_comparison": prefix, "file_sha256": file_hash})
            previous_ids, previous_path = ids, path
        reports.append({"source_kind": source, "seed": seed, "train_views": len(train),
                        "train_families": int(train.family_id.nunique()), "val_views": len(val),
                        "val_families": int(val.family_id.nunique()), "train_val_view_overlap": 0,
                        "train_val_family_overlap": 0, "train_index_sha256": source_sha,
                        "validation_index_sha256": val_sha, "streams": entries,
                        "stream_identity": "same ordered IDs up to each job's declared exposure cap"})
        del previous_ids, index, train, val
        gc.collect()
    return {"schema": "cus100_shared_stream_preparation_v1", "time": timestamp(),
            "sources": reports, "test_data_read": False, "jobs": [job["experiment_id"] for job in jobs]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--profiles", type=Path,
                        help="Rebuild manifest using JSON field overrides keyed by method and/or TR ID")
    parser.add_argument("--artifact-root", type=Path, default=REPO / RUN_ROOT / "artifacts")
    parser.add_argument("--enable", action="store_true", help="Enable only jobs whose calibration_status is passed")
    args = parser.parse_args()
    jobs = build_jobs(json.loads(args.profiles.read_text())) if args.profiles else load_jobs(args.manifest)
    if not args.artifact_root.resolve().is_relative_to(REPO):
        raise ValueError("Artifacts must remain under the repository for portable relative paths")
    report = prepare(jobs, args.artifact_root.resolve())
    if args.enable:
        if any(job.get("calibration_status") != "passed" for job in jobs):
            raise RuntimeError("All ten jobs must pass calibration before enabling the manifest")
        for job in jobs:
            job["enabled"] = True
    args.manifest.write_text("".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs))
    write_json(args.artifact_root / "shared_stream_preparation.json", report)
    print(json.dumps({"manifest": str(args.manifest), "report": str(args.artifact_root / "shared_stream_preparation.json"),
                      "streams": sum(len(source["streams"]) for source in report["sources"]),
                      "launchable_jobs": sum(bool(job["enabled"]) for job in jobs)}, indent=2))


if __name__ == "__main__":
    main()
