"""Audited, one-time EMA(1000) -> rollout(300) continuation of Road Cus500.

This module never changes or controls the source run. Its caller must observe
checkpoint_epoch_0300.pt committed, including validation, and stop the source
before launching the continuation. Later speculative log rows are archived and
excluded from active history. Publication uses one directory rename.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any

import numpy as np
import torch

from ...EVRPTW_RL.distributed_train import prepare_method
from ...common.data_pass import DataPassState
from ...common.distributed import DistributedContext
from ...common.distributed_protocol import configure_distributed_contract
from ...common.training_protocol import (
    assert_checkpoint_training_signature, freeze_resolved_training_signature,
    resolved_training_signature_from_args, validation_key,
)

# Audited checkpoint_epoch_0200.pt of the existing, unmodified source run.
# Every scientific/configuration field (including the stream path) is frozen by
# this digest. Deliberately no command-line override or generic migration mode.
EXPECTED_SOURCE_SIGNATURE_SHA256 = "825225b33273cb31f7f4e4f9541c57fe91bb48183d6ef962b594d74a52e9bbc8"
BOUNDARY_EPOCH = 300
SOURCE_WARMUP = 1000
REPORT_NAME = "stage2_transition.json"
SCHEMA = "cus500_evrptw_rl_stage2_transition_v1"
HISTORIES = (
    "logical_epoch_history.jsonl", "reward_diagnostics.jsonl",
    "sampled_view_ids.jsonl", "baseline_history.jsonl", "validation_history.jsonl",
)
ALIASES = {
    "best_validation_summary": (
        ("best.ckpt", "best_overall.ckpt", "checkpoint_selected.pt"),
        ("validation_summary.json", "validation_summary_overall.json")),
    "best_within_minimum_summary": (
        ("best_within_5000.ckpt",), ("validation_summary_within_5000.json",)),
}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def state_equal(left: Any, right: Any) -> bool:
    """Exact recursive equality, including optimizer tensors and NumPy RNG."""
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, np.ndarray):
        return (isinstance(right, np.ndarray) and left.dtype == right.dtype
                and left.shape == right.shape and np.array_equal(left, right))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(state_equal(value, right[key]) for key, value in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(state_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and math.isnan(left):
        return math.isnan(right)
    return left == right


def _validate_source_payload(payload: dict, epoch: int) -> argparse.Namespace:
    _require(payload.get("method") == "EVRPTW-RL", "source method is not EVRPTW-RL")
    _require(isinstance(payload.get("args"), dict), "source checkpoint has no argument mapping")
    args = argparse.Namespace(**deepcopy(payload["args"]))
    assert_checkpoint_training_signature(payload, args)
    signature = resolved_training_signature_from_args(args)
    _require(signature["sha256"] == EXPECTED_SOURCE_SIGNATURE_SHA256,
             "source resolved training signature differs from the audited run")
    _require(args.ema_warmup_steps == SOURCE_WARMUP, "source EMA cutoff must be 1000")
    _require(getattr(args, "reinforce_baseline", "paper") == "paper", "source baseline must be paper")
    _require(args.expected_world_size == 2 and args.distributed_backend == "nccl",
             "source topology must be two NCCL workers")
    _require(args.batch_size == args.physical_batch_size == 24 and args.effective_batch_size == 48,
             "source batch contract must be 24 per rank / 48 global")
    _require(args.training_rollout_steps == 1700 and args.validation_rollout_steps == 2550
             and args.samples_per_instance == 30, "source rollout/trajectory contract differs")
    _require(args.training_epochs == 10000 and args.minimum_training_epochs == 5000
             and args.customer_exposure_budget == 240000000, "source epoch budget differs")
    _require(args.scale == "Cus500" and args.training_representation == "G" and args.seed == 1234,
             "source scale/representation/seed differs")
    _require(not args.resume and not args.warm_start_checkpoint and not payload.get("warm_start_provenance"),
             "source must be the audited original uninterrupted run")
    _require(not args.pilot_mode and args.max_batches_per_pass is None and args.data_passes is None,
             "source partial/pilot configuration differs")
    _require(not payload.get("early_stopped") and payload.get("data_pass") == 0,
             "source checkpoint is terminal")
    # Recompute exactly as the real entrypoint does, detecting stale method
    # metadata as well as a validly self-hashed but incorrect contract.
    resolved = deepcopy(args)
    prepare_method(resolved)
    contract = configure_distributed_contract(resolved, DistributedContext(rank=0, world_size=2), method="EVRPTW-RL")
    _require(contract == payload.get("distributed_contract") == args.distributed_contract,
             "source distributed contract differs")
    _require(resolved_training_signature_from_args(resolved) == signature,
             "source method metadata differs from the actual trainer configuration")
    _require(payload.get("logical_epoch") == epoch and payload.get("stream_cursor") == epoch * 48,
             "source epoch/global stream cursor disagree")
    state = DataPassState(**payload["data_pass_state"])
    _require(state.protocol_id == args.protocol_id == payload.get("protocol_id")
             and state.optimizer_steps == epoch and state.instances_seen == epoch * 48
             and state.customer_exposures == epoch * 48 * 500
             and state.completed_data_passes == 0 and state.environment_transitions > 0,
             "source embedded progress disagrees with the committed epoch")
    _require(payload.get("completed_validation_checks") == epoch // 100
             and epoch > 0 and epoch % 100 == 0, "source completed validation count differs")
    _require(payload.get("baseline_eval_count") == 0 and payload.get("baseline_update_count") == 0,
             "source has unexpected pre-boundary baseline evaluations/updates")
    _require(isinstance(payload.get("ema_cost"), (int, float)) and math.isfinite(payload["ema_cost"]),
             "source EMA cost is missing or nonfinite")
    _require(len(payload.get("baseline_probe_view_ids", [])) == args.baseline_eval_size == 64,
             "source training-pool baseline probe is incomplete")
    for payload_key, args_key in (("training_stream_contract", "training_stream_contract_snapshot"),
                                  ("reward_contract", "reward_contract_snapshot"),
                                  ("method_auxiliary_profile", "method_auxiliary_snapshot")):
        _require(payload.get(payload_key) is not None and state_equal(payload[payload_key], vars(args).get(args_key)),
                 f"source {payload_key} snapshot differs from saved args")
    _require(payload["training_stream_contract"]["sha256"] == args.training_stream_contract_sha256,
             "source training stream hash differs")
    optimizer = payload.get("optimizer", {})
    _require(bool(optimizer.get("state")) and bool(optimizer.get("param_groups")), "source optimizer state missing")
    for group in optimizer["param_groups"]:
        _require(group["lr"] == args.learning_rate and group["weight_decay"] == args.weight_decay,
                 "source optimizer hyperparameters differ")
        _require(all(parameter in optimizer["state"] for parameter in group["params"]),
                 "source optimizer parameter state missing")
    for value in optimizer["state"].values():
        _require(float(value.get("step", -1)) == epoch and "exp_avg" in value and "exp_avg_sq" in value,
                 "source optimizer step/moments disagree with the committed epoch")
    rngs = payload.get("rank_rng_states", [])
    _require(len(rngs) == 2, "source per-rank RNG state missing")
    for rng in rngs:
        _require(set(rng) == {"python", "numpy", "torch_cpu", "torch_cuda", "pool"},
                 "source per-rank RNG fields differ")
        random.Random().setstate(rng["python"])
        np.random.RandomState().set_state(rng["numpy"])
        torch.Generator(device="cpu").set_state(rng["torch_cpu"])
        _require(isinstance(rng["torch_cuda"], torch.Tensor) and rng["torch_cuda"].dtype == torch.uint8
                 and rng["torch_cuda"].ndim == 1 and rng["torch_cuda"].numel() > 0,
                 "source CUDA RNG state missing")
        generator = np.random.default_rng()
        generator.bit_generator.state = rng["pool"]
    _require(bool(payload.get("model")) and payload.get("baseline", {}).keys() == payload["model"].keys(),
             "source model/baseline structure differs")
    for key, tensor in payload["model"].items():
        reference = payload["baseline"][key]
        _require(isinstance(tensor, torch.Tensor) and isinstance(reference, torch.Tensor)
                 and tensor.shape == reference.shape and tensor.dtype == reference.dtype,
                 f"source model/baseline tensor differs: {key}")
    for summary_name, key_name in (("best_validation_summary", "best_validation_key"),
                                   ("best_within_minimum_summary", "best_within_minimum_key")):
        summary = payload.get(summary_name)
        _require(isinstance(summary, dict) and 0 < int(summary["logical_epoch"]) <= epoch,
                 "source historical validation selection is missing or ahead")
        _require(tuple(payload.get(key_name, [])) == validation_key(summary),
                 "source historical validation selection key differs")
    return args


def stage2_args(source_args: argparse.Namespace, destination_run: Path | str) -> argparse.Namespace:
    """Resolve the exact destination signature using real method helpers."""
    args = deepcopy(source_args)
    args.output_dir = Path(destination_run).resolve()
    args.resume = True
    args.ema_warmup_steps = BOUNDARY_EPOCH
    prepare_method(args)
    configure_distributed_contract(args, DistributedContext(rank=0, world_size=2), method="EVRPTW-RL")
    signature = freeze_resolved_training_signature(args)
    expected = deepcopy(source_args.resolved_training_signature)
    expected["method_specific"]["ema_warmup_steps"] = BOUNDARY_EPOCH
    expected["method_specific"]["rollout_baseline_warmup_optimizer_updates"] = BOUNDARY_EPOCH
    expected.pop("sha256")
    _require({key: value for key, value in signature.items() if key != "sha256"} == expected,
             "stage2 changes a training-signature field other than EMA cutoff")
    return args


def assert_stage2_launch_args(payload: dict, launch_args: argparse.Namespace) -> None:
    """Call after the launch command's args and objective/stream are resolved."""
    _require(launch_args.resume and launch_args.warm_start_checkpoint is None,
             "stage2 launch must use --resume without --warm-start-checkpoint")
    _require(Path(launch_args.output_dir).resolve() == Path(payload["args"]["output_dir"]).resolve(),
             "stage2 launch output directory differs")
    assert_checkpoint_training_signature(payload, launch_args)


def _retained_history(path: Path, boundary: int) -> tuple[list[dict], str]:
    rows, kept = [], []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            _require(index == len(lines) - 1, f"corrupt committed history: {path.name}")
            continue
        epoch = row.get("logical_epoch", row.get("optimizer_step"))
        _require(isinstance(epoch, int), f"history row missing epoch: {path.name}")
        if epoch <= boundary:
            rows.append(row)
            kept.append(line + "\n")
    return rows, "".join(kept)


def _validate_histories(histories: dict[str, list[dict]], payloads: dict[int, dict], boundary: int) -> None:
    expected = list(range(1, boundary + 1))
    for name in HISTORIES[:3]:
        _require([row["logical_epoch"] for row in histories[name]] == expected,
                 f"committed history must contain each epoch exactly once: {name}")
    for row in histories[HISTORIES[0]]:
        epoch = row["logical_epoch"]
        _require(row.get("baseline_kind") == "paper_ema" and row.get("baseline_warmup_synchronized") is False
                 and row.get("optimizer_steps_total") == epoch and row.get("global_stream_cursor") == epoch * 48,
                 "source logical history has an unexpected baseline/progress state")
    for row in histories[HISTORIES[2]]:
        epoch = row["logical_epoch"]
        _require(row.get("start_cursor") == (epoch - 1) * 48 and row.get("end_cursor") == epoch * 48
                 and len(row.get("view_ids", [])) == 48, "source sampled stream cursor/history differs")
    _require(not histories.get("baseline_history.jsonl"), "source has pre-boundary baseline probe history")
    validation = histories["validation_history.jsonl"]
    _require([row["logical_epoch"] for row in validation] == list(range(100, boundary + 1, 100)),
             "boundary validation history is not complete")
    by_epoch = {row["logical_epoch"]: row for row in validation}
    for epoch, payload in payloads.items():
        _require(by_epoch[epoch].get("instances") == 500 and by_epoch[epoch].get("decode_type") == "sampling"
                 and by_epoch[epoch].get("candidate_count") == 30,
                 "source validation cohort/decoding differs")
        for summary_name in ALIASES:
            summary = payload[summary_name]
            _require(state_equal(summary, by_epoch[int(summary["logical_epoch"])]),
                     "historical selected validation summary differs from committed history")


def _migrate_payload(payload: dict, destination: Path, epoch: int, source_path: Path, source_sha: str) -> dict:
    migrated = deepcopy(payload)
    args = stage2_args(argparse.Namespace(**deepcopy(payload["args"])), destination)
    migrated["args"] = vars(args)
    migrated["resolved_training_signature"] = deepcopy(args.resolved_training_signature)
    migrated["distributed_contract"] = deepcopy(args.distributed_contract)
    migrated["data_pass_state"]["last_checkpoint"] = str(destination / "checkpoint_latest.pt")
    if epoch == BOUNDARY_EPOCH:
        migrated["baseline"] = deepcopy(migrated["model"])
    migrated["stage2_transition_provenance"] = {
        "schema": SCHEMA, "boundary_epoch": BOUNDARY_EPOCH,
        "source_checkpoint": str(source_path), "source_checkpoint_sha256": source_sha,
        "source_resolved_training_signature_sha256": payload["resolved_training_signature"]["sha256"],
        "destination_resolved_training_signature_sha256": args.resolved_training_signature_sha256,
        "source_ema_warmup_steps": SOURCE_WARMUP, "destination_ema_warmup_steps": BOUNDARY_EPOCH,
        "baseline_handoff_applied": epoch == BOUNDARY_EPOCH,
        "baseline_handoff": "copy_synchronized_actor_after_last_ema_optimizer_update",
        "historical_checkpoint_compatible_prefix": epoch < BOUNDARY_EPOCH,
        "historical_behavior": "epochs_1_through_300_use_paper_ema",
        "preserved": "actor_optimizer_rank_rng_stream_epoch_validation_ema_probe_and_early_stop_state",
    }
    # Whole-payload comparison is deliberately strict. Only these explicit
    # changes are authorized, so an omitted optimizer/RNG/selection field fails.
    compare = deepcopy(migrated)
    compare.pop("stage2_transition_provenance")
    for key in ("args", "resolved_training_signature", "distributed_contract", "baseline", "data_pass_state"):
        compare[key] = payload[key]
    _require(state_equal(compare, payload), "migration changed an unauthorized checkpoint field")
    assert_checkpoint_training_signature(migrated, args)
    return migrated


def prepare_stage2_run(source_run: Path | str, destination_run: Path | str,
                       boundary_epoch: int = BOUNDARY_EPOCH,
                       expected_source_checkpoint_sha: str | None = None,
                       source_checkpoint: Path | str | None = None) -> dict:
    """Publish a validated continuation; return its JSON-serializable manifest.

    The destination must be absent. Repeating an already completed identical
    transition returns its existing report without resetting any training files.
    Original files are retained under source_checkpoint_archive with SHA256s.
    Only the explicitly named epoch-300 checkpoint may supply continuation state.
    """
    _require(boundary_epoch == BOUNDARY_EPOCH, "only the audited epoch-300 transition is supported")
    source, destination = Path(source_run).resolve(), Path(destination_run).resolve()
    _require(source.is_dir(), "source run directory does not exist")
    _require(source != destination and source not in destination.parents and destination not in source.parents,
             "source and destination run directories must be separate")
    checkpoint = source / f"checkpoint_epoch_{boundary_epoch:04d}.pt"
    if source_checkpoint is not None:
        _require(Path(source_checkpoint).resolve() == checkpoint, "source checkpoint must be exact committed epoch 300")
    _require(checkpoint.is_file() and not checkpoint.is_symlink(), "committed epoch-300 checkpoint is not ready")
    source_sha = sha256_file(checkpoint)
    if expected_source_checkpoint_sha is not None:
        _require(source_sha == expected_source_checkpoint_sha, "source boundary checkpoint SHA256 changed")
    report_path = destination / REPORT_NAME
    if destination.exists():
        _require(report_path.is_file(), "destination exists without a completed transition report")
        report = json.loads(report_path.read_text())
        _require(report.get("schema") == SCHEMA and report.get("source_run") == str(source)
                 and report.get("destination_run") == str(destination)
                 and report.get("source_checkpoint_sha256") == source_sha
                 and report.get("boundary_epoch") == boundary_epoch,
                 "destination transition provenance differs")
        for record in report["migrated_checkpoints"]:
            _require(sha256_file(destination / record["name"]) == record["destination_sha256"],
                     "immutable migrated checkpoint SHA256 changed")
        for record in report["source_archive"]:
            _require(sha256_file(destination / "source_checkpoint_archive" / record["name"]) == record["sha256"],
                     "archived source artifact SHA256 changed")
        return report
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage2-", dir=destination.parent))
    try:
        archive = staging / "source_checkpoint_archive"
        archive.mkdir()
        archived = []
        for path in sorted(source.iterdir()):
            if path.name.startswith(".") or ".tmp" in path.name or path.is_dir():
                continue
            _require(path.is_file() and not path.is_symlink(), f"unexpected source artifact: {path.name}")
            immutable = path.name.startswith("checkpoint_epoch_") and path.suffix == ".pt"
            before = sha256_file(path) if immutable else None
            target = archive / path.name
            shutil.copy2(path, target)
            captured_sha = sha256_file(target)
            if immutable:
                _require(before == captured_sha == sha256_file(path),
                         f"immutable source checkpoint changed while copying: {path.name}")
            archived.append({"name": path.name, "sha256": captured_sha,
                             "size_bytes": target.stat().st_size,
                             "immutable_source_sha_verified": immutable})
        _require(sha256_file(archive / checkpoint.name) == source_sha, "source boundary checkpoint changed while copying")
        _require((archive / "checkpoint_latest.pt").is_file()
                 and sha256_file(archive / "checkpoint_latest.pt") == source_sha,
                 "source latest checkpoint is not the committed boundary checkpoint")
        payloads = {}
        for epoch in range(100, boundary_epoch + 1, 100):
            original = archive / f"checkpoint_epoch_{epoch:04d}.pt"
            _require(original.is_file(), f"historical committed checkpoint missing: {original.name}")
            payload = torch.load(original, map_location="cpu", weights_only=False)
            _validate_source_payload(payload, epoch)
            payloads[epoch] = payload
        boundary = payloads[boundary_epoch]
        histories = {}
        for name in HISTORIES:
            path = archive / name
            _require(path.is_file() or name == "baseline_history.jsonl", f"committed history missing: {name}")
            rows, retained = _retained_history(path, boundary_epoch) if path.exists() else ([], "")
            histories[name] = rows
            (staging / name).write_text(retained)
        _validate_histories(histories, payloads, boundary_epoch)
        state_path = archive / "data_pass_state.json"
        _require(state_path.is_file(), "source data-pass sidecar is missing")
        sidecar = DataPassState(**json.loads(state_path.read_text()))
        _require(sidecar.protocol_id == boundary["protocol_id"]
                 and sidecar.optimizer_steps <= boundary_epoch and sidecar.instances_seen <= boundary_epoch * 48,
                 "source sidecar is ahead of boundary or from another protocol")
        migrated_records = []
        for epoch, payload in payloads.items():
            name = f"checkpoint_epoch_{epoch:04d}.pt"
            original_sha = sha256_file(archive / name)
            migrated = _migrate_payload(payload, destination, epoch, source / name, original_sha)
            torch.save(migrated, staging / name)
            reloaded = torch.load(staging / name, map_location="cpu", weights_only=False)
            _require(state_equal(migrated, reloaded), "serialized transition checkpoint does not round-trip exactly")
            migrated_records.append({"name": name, "logical_epoch": epoch,
                                     "source_sha256": original_sha,
                                     "destination_sha256": sha256_file(staging / name),
                                     "baseline_handoff_applied": epoch == boundary_epoch})
        shutil.copy2(staging / checkpoint.name, staging / "checkpoint_latest.pt")
        for summary_name, (checkpoint_names, summary_names) in ALIASES.items():
            summary = boundary[summary_name]
            selected = staging / f"checkpoint_epoch_{int(summary['logical_epoch']):04d}.pt"
            for name in checkpoint_names:
                shutil.copy2(selected, staging / name)
            for name in summary_names:
                _json(staging / name, summary)
        final = torch.load(staging / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
        _json(staging / "data_pass_state.json", final["data_pass_state"])
        _json(staging / "progress.json", {"status": "stage2_prepared", "logical_epoch": boundary_epoch,
                                          "phase": "awaiting_resume", "world_size": 2})
        report = {
            "schema": SCHEMA, "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_run": str(source), "destination_run": str(destination),
            "ready_path": str(report_path), "boundary_epoch": boundary_epoch,
            "resume_start_epoch": boundary_epoch + 1, "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": source_sha,
            "destination_checkpoint": str(destination / "checkpoint_latest.pt"),
            "destination_checkpoint_sha256": sha256_file(staging / "checkpoint_latest.pt"),
            "source_resolved_training_signature_sha256": EXPECTED_SOURCE_SIGNATURE_SHA256,
            "destination_resolved_training_signature_sha256": final["resolved_training_signature"]["sha256"],
            "source_ema_warmup_steps": SOURCE_WARMUP, "destination_ema_warmup_steps": boundary_epoch,
            "baseline_handoff": "copy_synchronized_actor_after_last_ema_optimizer_update",
            "first_greedy_rollout_epoch": boundary_epoch + 1,
            "first_baseline_probe_epoch": ((boundary_epoch // 100) + 1) * 100,
            "optimizer_steps": boundary_epoch, "stream_cursor": boundary_epoch * 48,
            "completed_validation_checks": final["completed_validation_checks"],
            "source_archive": archived, "migrated_checkpoints": migrated_records,
            "history_policy": "original bytes archived; active JSONL retains completed epochs <= 300 unchanged",
            "source_was_modified": False,
        }
        _json(staging / REPORT_NAME, report)
        for path in staging.rglob("*"):
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        _fsync_directory(archive)
        _fsync_directory(staging)
        _require(not destination.exists(), "destination appeared during migration")
        os.rename(staging, destination)
        _fsync_directory(destination.parent)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)
