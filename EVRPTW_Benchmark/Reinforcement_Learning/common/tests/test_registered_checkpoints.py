from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from EVRPTW_Benchmark.Reinforcement_Learning.common import protocol_trainers
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_protocol import (
    add_data_pass_arguments,
    assert_checkpoint_training_signature,
    freeze_resolved_training_signature,
    resolved_training_signature_from_args,
    parse_float_checkpoints,
    parse_int_checkpoints,
    require_validation_rollout_steps,
    validation_rollout_steps,
)


def test_checkpoint_schedule_parsers_are_sorted_and_unique() -> None:
    assert parse_int_checkpoints("500,100,500") == (100, 500)
    assert parse_float_checkpoints("24,6,12,6") == (6.0, 12.0, 24.0)


def test_validation_rollout_steps_are_exact_ceiling_three_halves() -> None:
    assert validation_rollout_steps(580) == 870
    assert validation_rollout_steps(1200) == 1800
    assert validation_rollout_steps(65) == 98


def test_validation_rollout_steps_reject_a_mismatched_explicit_cap() -> None:
    args = SimpleNamespace(
        training_rollout_steps=65,
        validation_rollout_steps=97,
    )
    with pytest.raises(ValueError, match=r"ceil\(3/2"):
        require_validation_rollout_steps(args)



def test_validation_rollout_policy_cli_defaults_and_explicit_cap() -> None:
    parser = argparse.ArgumentParser()
    add_data_pass_arguments(parser)
    default = parser.parse_args(["--training-rollout-steps", "600"])
    assert default.validation_rollout_policy == "ceil_1_5"
    assert require_validation_rollout_steps(default) == 900
    explicit = parser.parse_args([
        "--training-rollout-steps", "600", "--validation-rollout-steps", "700",
        "--validation-rollout-policy", "explicit",
    ])
    # Adapter setup and protocol configuration both resolve the same args.
    assert require_validation_rollout_steps(explicit) == 700
    assert require_validation_rollout_steps(explicit) == 700
    assert explicit.validation_rollout_steps == 700
    with pytest.raises(SystemExit):
        parser.parse_args(["--validation-rollout-policy", "unknown"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--validation-rollout-policy", "explicit", "--validation-rollout-steps", "700.5"])


@pytest.mark.parametrize("policy", [None, "ceil_1_5"])
def test_default_validation_rollout_still_rejects_600_700(policy) -> None:
    args = SimpleNamespace(training_rollout_steps=600, validation_rollout_steps=700)
    if policy is not None:
        args.validation_rollout_policy = policy
    with pytest.raises(ValueError, match=r"ceil\(3/2"):
        require_validation_rollout_steps(args)
    assert args.validation_rollout_steps == 700


@pytest.mark.parametrize("cap", [None, 0, -1, 599, 700.5, 700.0, True, "700"])
def test_explicit_validation_rollout_rejects_missing_invalid_or_short_cap(cap) -> None:
    args = SimpleNamespace(
        training_rollout_steps=600, validation_rollout_steps=cap,
        validation_rollout_policy="explicit",
    )
    with pytest.raises(ValueError, match="explicit.*validation-rollout-steps"):
        require_validation_rollout_steps(args)


def test_explicit_validation_rollout_allows_equal_training_cap() -> None:
    args = SimpleNamespace(
        training_rollout_steps=600, validation_rollout_steps=600,
        validation_rollout_policy="explicit",
    )
    assert require_validation_rollout_steps(args) == 600


def test_validation_rollout_rejects_unknown_policy() -> None:
    args = SimpleNamespace(
        training_rollout_steps=600, validation_rollout_steps=700,
        validation_rollout_policy="unknown",
    )
    with pytest.raises(ValueError, match="unsupported validation rollout policy"):
        require_validation_rollout_steps(args)


def test_default_validation_rollout_signature_matches_historical_checkpoint() -> None:
    args = SimpleNamespace(training_rollout_steps=600, validation_rollout_steps=900)
    signature = freeze_resolved_training_signature(args)
    # Golden SHA256 verified against HEAD 93d9893's pre-policy signature function.
    assert signature["sha256"] == "544553367f4e717b26c9513d97652c7f1a83dde08665a39628e51d3769a7dbf3"
    assert "validation_rollout_policy" not in signature
    payload = {"resolved_training_signature": signature, "args": deepcopy(vars(args))}
    args.validation_rollout_policy = "ceil_1_5"
    assert resolved_training_signature_from_args(args) == signature
    assert_checkpoint_training_signature(payload, args)


@pytest.mark.parametrize("cap", [700, 900])
def test_explicit_validation_rollout_is_signed_and_resume_cannot_change_policy(cap) -> None:
    args = SimpleNamespace(
        training_rollout_steps=600, validation_rollout_steps=cap,
        validation_rollout_policy="explicit",
    )
    require_validation_rollout_steps(args)
    signature = freeze_resolved_training_signature(args)
    assert signature["validation_rollout_steps"] == cap
    assert signature["validation_rollout_policy"] == "explicit"
    payload = {"resolved_training_signature": signature, "args": deepcopy(vars(args))}
    assert_checkpoint_training_signature(payload, args)
    changed_cap = deepcopy(args)
    changed_cap.validation_rollout_steps = cap + 1
    with pytest.raises(ValueError, match="signature mismatch"):
        assert_checkpoint_training_signature(payload, changed_cap)
    legacy = SimpleNamespace(training_rollout_steps=600, validation_rollout_steps=900)
    old_signature = freeze_resolved_training_signature(legacy)
    assert old_signature["sha256"] != signature["sha256"]
    with pytest.raises(ValueError, match="signature mismatch"):
        assert_checkpoint_training_signature(payload, legacy)
    old_payload = {"resolved_training_signature": old_signature, "args": deepcopy(vars(legacy))}
    with pytest.raises(ValueError, match="signature mismatch"):
        assert_checkpoint_training_signature(old_payload, args)

def test_registered_snapshot_crossing_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    saved_paths: list[Path] = []

    def fake_save(path: Path, **_kwargs) -> None:
        path.write_bytes(b"checkpoint")
        saved_paths.append(path)

    monkeypatch.setattr(protocol_trainers, "_save_checkpoint", fake_save)
    exposure_saved: set[int] = set()
    gpu_saved: set[float] = set()
    kwargs = {
        "output": tmp_path,
        "method": "AM-EVRPTW",
        "args": SimpleNamespace(protocol_id="test"),
        "data_pass": 0,
        "policy": object(),
        "baseline": object(),
        "optimizer": object(),
        "observed_exposure": 600,
        "observed_gpu_hours": 7.0,
        "exposure_checkpoints": (100, 500, 1000),
        "gpu_hour_checkpoints": (6.0, 12.0),
        "saved_exposure": exposure_saved,
        "saved_gpu_hours": gpu_saved,
    }
    protocol_trainers._save_registered_snapshots(**kwargs)
    protocol_trainers._save_registered_snapshots(**kwargs)
    assert {path.name for path in saved_paths} == {
        "checkpoint_customer_exposure_100.pt",
        "checkpoint_customer_exposure_500.pt",
        "checkpoint_gpu_hours_6.pt",
    }
    assert len(saved_paths) == 3
    events = [json.loads(line) for line in (tmp_path / "checkpoint_events.jsonl").read_text().splitlines()]
    assert len(events) == 3
    assert all(row["schema"] == "drl_training_checkpoint_event_v1" for row in events)
