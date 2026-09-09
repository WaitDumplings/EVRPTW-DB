from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "drl_job_runtime.py"
SPEC = importlib.util.spec_from_file_location("drl_job_runtime", SCRIPT)
assert SPEC and SPEC.loader
RUNTIME = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNTIME
SPEC.loader.exec_module(RUNTIME)


def _context(tmp_path: Path):
    return {
        "repo": tmp_path,
        "dataset": tmp_path / "dataset",
        "output": tmp_path / "output",
        "branch": "drl-benchmark-adapters",
        "commit": "abc123",
        "conda_env": "maojie",
    }


def _job(job_id: str = "train__R__am_evrptw__Cus100__seed1234"):
    return {
        "job_id": job_id,
        "kind": "train",
        "representation": "R",
        "method": "am_evrptw",
        "scale": "Cus100",
        "seed": 1234,
        "global_slot": 0,
        "queue_position": 0,
        "primary_checkpoint": "best_overall.ckpt",
        "minimum_budget_checkpoint": "best_within_5000.ckpt",
        "extended_checkpoint": "best_overall.ckpt",
    }



def test_exact_job_warm_start_resolution_never_crosses_scope(tmp_path: Path) -> None:
    context = _context(tmp_path)
    job = _job("full__G__Full-support__am_evrptw__Cus50__seed1234")
    job.update(
        representation="G",
        condition="Full-support",
        warm_start_source_commit="source-commit",
        warm_start_scope="exact_job_only",
        warm_start_checkpoint_name="best.ckpt",
        warm_start_missing_policy="fresh",
    )
    source_context = {**context, "commit": "source-commit"}
    expected = RUNTIME.output_dir(job, source_context) / "best.ckpt"
    assert RUNTIME.resolve_warm_start_checkpoint(job, context) is None
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"weights")
    assert RUNTIME.resolve_warm_start_checkpoint(job, context) == expected.resolve()
    wrong_scope = {**job, "warm_start_scope": "method_scale"}
    with pytest.raises(RuntimeError, match="exact_job_only"):
        RUNTIME.resolve_warm_start_checkpoint(wrong_scope, context)

def _write_formal_gate(
    root: Path,
    *,
    allowed: bool,
    protocol_id: str = "drl_rq_protocol_frozen_v1",
    authorized_job_ids: list[str] | None = None,
) -> Path:
    statuses = {
        "G1": "PILOT_WAIVED_BY_USER",
        "G2": "IMPLEMENTED_UNIT_TESTED",
        "G3": "PILOT_WAIVED_BY_USER",
        "G4": "PILOT_WAIVED_BY_USER",
        "G5": "PILOT_WAIVED_BY_USER",
        "G6": "PILOT_WAIVED_BY_USER",
        "G7": "NOT_APPLICABLE_G_ONLY",
        "G8": "IMPLEMENTED_UNIT_TESTED",
    }
    policy = (
        RUNTIME.AUTHORIZED_LAUNCH_POLICY
        if allowed
        else "reward_contract_v2_short_validation_pending_user_authorization"
    )
    authorized = list(
        authorized_job_ids
        if authorized_job_ids is not None
        else [_job()["job_id"]]
    )
    registry = {
        "training_stream_registry_path": (
            "EVRPTW_Benchmark/Reinforcement_Learning/configs/"
            "drl_training_stream_registry_v1.json"
        ),
        "training_stream_registry_sha256": "1" * 64,
    }
    config_root = root / "EVRPTW_Benchmark/Reinforcement_Learning/configs"
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "drl_rq_runtime_candidates_v2.yaml").write_text(
        yaml.safe_dump(
            {
                "protocol_id": "drl_rq_protocol_frozen_v1",
                "formal_launch_allowed": allowed,
                "launch_policy": policy,
                "authorized_job_ids": authorized,
                "formal_launch_gates": statuses,
                **registry,
            }
        ),
        encoding="utf-8",
    )
    (config_root / "drl_rq_protocol_frozen_v1.yaml").write_text(
        yaml.safe_dump(
            {
                "protocol_id": "drl_rq_protocol_frozen_v1",
                "formal_launch_allowed": allowed,
                "launch_policy": policy,
                "authorized_job_ids": authorized,
                "formal_launch_gates": {
                    key: {"status": value} for key, value in statuses.items()
                },
                **registry,
            }
        ),
        encoding="utf-8",
    )
    path = root / "formal_gate.json"
    path.write_text(
        json.dumps(
            {
                "schema": "drl_rq_formal_launch_gate_v1",
                "protocol_id": protocol_id,
                "formal_launch_allowed": allowed,
                "launch_policy": policy,
                "authorized_job_ids": authorized,
                "formal_launch_gates": statuses,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_closed_formal_gate_blocks_execution_but_allows_dry_run_validation(
    tmp_path: Path,
) -> None:
    gate = _write_formal_gate(tmp_path, allowed=False)
    job = _job()
    job.update(
        {
            "formal_gate_file": gate.name,
            "protocol_id": "drl_rq_protocol_frozen_v1",
        }
    )
    RUNTIME.validate_formal_launch_gates(tmp_path, [job], require_open=False)
    with pytest.raises(RuntimeError, match="formal launch gate is closed"):
        RUNTIME.validate_formal_launch_gates(tmp_path, [job], require_open=True)


def test_formal_gate_fails_closed_on_missing_or_mismatched_contract(
    tmp_path: Path,
) -> None:
    job = _job()
    job["protocol_id"] = "drl_rq_protocol_frozen_v1"
    with pytest.raises(RuntimeError, match="missing formal_gate_file"):
        RUNTIME.validate_formal_launch_gates(tmp_path, [job], require_open=True)

    gate = _write_formal_gate(tmp_path, allowed=True, protocol_id="other")
    job["formal_gate_file"] = gate.name
    with pytest.raises(RuntimeError, match="protocol does not match"):
        RUNTIME.validate_formal_launch_gates(tmp_path, [job], require_open=True)


@pytest.mark.parametrize("document_name", ["gate", "runtime", "protocol"])
def test_flipping_only_one_formal_launch_boolean_fails_closed(
    tmp_path: Path, document_name: str,
) -> None:
    gate_path = _write_formal_gate(tmp_path, allowed=False)
    config_root = tmp_path / "EVRPTW_Benchmark/Reinforcement_Learning/configs"
    paths = {
        "gate": gate_path,
        "runtime": config_root / "drl_rq_runtime_candidates_v2.yaml",
        "protocol": config_root / "drl_rq_protocol_frozen_v1.yaml",
    }
    path = paths[document_name]
    if path.suffix == ".json":
        document = json.loads(path.read_text(encoding="utf-8"))
        document["formal_launch_allowed"] = True
        path.write_text(json.dumps(document), encoding="utf-8")
    else:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        document["formal_launch_allowed"] = True
        path.write_text(yaml.safe_dump(document), encoding="utf-8")

    job = {
        **_job(),
        "protocol_id": "drl_rq_protocol_frozen_v1",
        "formal_gate_file": gate_path.name,
    }
    # Cross-checking is unconditional: even a dry-run may not bless a
    # contradictory release decision.
    with pytest.raises(RuntimeError, match="decisions disagree"):
        RUNTIME.validate_formal_launch_gates(
            tmp_path, [job], require_open=False
        )
    with pytest.raises(RuntimeError, match="decisions disagree"):
        RUNTIME.validate_formal_launch_gates(
            tmp_path, [job], require_open=True
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("launch_policy", "decisions disagree"),
        ("formal_launch_gates", "G1-G8 evidence disagrees"),
    ],
)
def test_single_document_policy_or_gate_evidence_mutation_fails_closed(
    tmp_path: Path, field: str, message: str,
) -> None:
    gate_path = _write_formal_gate(tmp_path, allowed=False)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if field == "launch_policy":
        gate[field] = "locally-edited-policy"
    else:
        gate[field]["G4"] = "PARTIAL_LOCAL_EDIT"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    job = {
        **_job(),
        "protocol_id": "drl_rq_protocol_frozen_v1",
        "formal_gate_file": gate_path.name,
    }
    with pytest.raises(RuntimeError, match=message):
        RUNTIME.validate_formal_launch_gates(
            tmp_path, [job], require_open=False
        )


def test_consistent_explicitly_authorized_gate_can_open(tmp_path: Path) -> None:
    gate = _write_formal_gate(tmp_path, allowed=True)
    job = {
        **_job(),
        "protocol_id": "drl_rq_protocol_frozen_v1",
        "formal_gate_file": gate.name,
    }
    RUNTIME.validate_formal_launch_gates(tmp_path, [job], require_open=False)
    RUNTIME.validate_formal_launch_gates(tmp_path, [job], require_open=True)


def test_formal_gate_allows_only_nonempty_subsets_of_authorized_job_ids(
    tmp_path: Path,
) -> None:
    authorized_ids = ["formal-terran-cus500", "formal-terran-cus1000"]
    gate = _write_formal_gate(
        tmp_path,
        allowed=True,
        authorized_job_ids=authorized_ids,
    )

    def scoped_job(job_id: str) -> dict:
        return {
            **_job(job_id),
            "protocol_id": "drl_rq_protocol_frozen_v1",
            "formal_gate_file": gate.name,
        }

    first = scoped_job(authorized_ids[0])
    second = scoped_job(authorized_ids[1])
    RUNTIME.validate_formal_launch_gates(
        tmp_path, [first], require_open=True
    )
    RUNTIME.validate_formal_launch_gates(
        tmp_path, [first, second], require_open=True
    )

    with pytest.raises(RuntimeError, match="outside authorized_job_ids"):
        RUNTIME.validate_formal_launch_gates(
            tmp_path, [scoped_job("formal-am-cus500")], require_open=True
        )


def test_formal_gate_fails_closed_when_authorized_job_ids_disagree(
    tmp_path: Path,
) -> None:
    gate_path = _write_formal_gate(tmp_path, allowed=True)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["authorized_job_ids"].append("extra-job")
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    job = {
        **_job(),
        "protocol_id": "drl_rq_protocol_frozen_v1",
        "formal_gate_file": gate_path.name,
    }
    with pytest.raises(RuntimeError, match="authorized_job_ids disagree"):
        RUNTIME.validate_formal_launch_gates(
            tmp_path, [job], require_open=False
        )


def test_consistent_open_boolean_without_authorized_policy_stays_closed(
    tmp_path: Path,
) -> None:
    gate_path = _write_formal_gate(tmp_path, allowed=True)
    config_root = tmp_path / "EVRPTW_Benchmark/Reinforcement_Learning/configs"
    pending_policy = "reward_contract_v2_validation_complete_but_not_authorized"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["launch_policy"] = pending_policy
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    for name in (
        "drl_rq_runtime_candidates_v2.yaml",
        "drl_rq_protocol_frozen_v1.yaml",
    ):
        path = config_root / name
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        document["launch_policy"] = pending_policy
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
    job = {
        **_job(),
        "protocol_id": "drl_rq_protocol_frozen_v1",
        "formal_gate_file": gate_path.name,
    }
    RUNTIME.validate_formal_launch_gates(tmp_path, [job], require_open=False)
    with pytest.raises(RuntimeError, match="explicit user authorization"):
        RUNTIME.validate_formal_launch_gates(tmp_path, [job], require_open=True)


@pytest.mark.parametrize(
    "incomplete_status",
    [
        "PARTIAL_IMPLEMENTATION",
        "NOT_PASSED",
        "NOT_UNIT_TESTED",
        "NOT_APPLICABLE_PENDING",
    ],
)
def test_consistent_open_gate_still_rejects_incomplete_g1_g8_evidence(
    tmp_path: Path, incomplete_status: str,
) -> None:
    gate_path = _write_formal_gate(tmp_path, allowed=True)
    config_root = tmp_path / "EVRPTW_Benchmark/Reinforcement_Learning/configs"

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["formal_launch_gates"]["G7"] = incomplete_status
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    runtime_path = config_root / "drl_rq_runtime_candidates_v2.yaml"
    runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    runtime["formal_launch_gates"]["G7"] = incomplete_status
    runtime_path.write_text(yaml.safe_dump(runtime), encoding="utf-8")

    protocol_path = config_root / "drl_rq_protocol_frozen_v1.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol["formal_launch_gates"]["G7"]["status"] = incomplete_status
    protocol_path.write_text(yaml.safe_dump(protocol), encoding="utf-8")

    job = {
        **_job(),
        "protocol_id": "drl_rq_protocol_frozen_v1",
        "formal_gate_file": gate_path.name,
    }
    RUNTIME.validate_formal_launch_gates(tmp_path, [job], require_open=False)
    with pytest.raises(RuntimeError, match="formal launch gates are incomplete"):
        RUNTIME.validate_formal_launch_gates(tmp_path, [job], require_open=True)


def _artifact_command(
    output: Path, exit_code: int = 0, *, include_manifest_checkpoints: bool = True
):
    manifest_artifacts = ""
    if include_manifest_checkpoints:
        manifest_artifacts = (
            "(p/'best_overall.ckpt').write_bytes(b'overall'); "
            "(p/'best_within_5000.ckpt').write_bytes(b'within'); "
        )
    source = (
        "from pathlib import Path; "
        f"p=Path({str(output)!r}); p.mkdir(parents=True, exist_ok=True); "
        "(p/'checkpoint_selected.pt').write_bytes(b'x'); "
        "(p/'validation_summary.json').write_text('{}'); "
        "(p/'training_result.json').write_text('{}'); "
        f"{manifest_artifacts}"
        f"raise SystemExit({exit_code})"
    )
    return [sys.executable, "-c", source]


def _write_completed_test_job(job: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for artifact in RUNTIME.required_training_artifacts(job, output):
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if artifact.name == "training_result.json":
            artifact.write_text('{"generation": 1}', encoding="utf-8")
        else:
            artifact.write_bytes(b"selected")
    (output / "job_result.json").write_text(
        json.dumps({"status": "passed"}), encoding="utf-8"
    )


def test_job_complete_requires_training_result_and_rechecks_all_validators(
    tmp_path: Path, monkeypatch,
) -> None:
    job = _job()
    output = tmp_path / "completed"
    _write_completed_test_job(job, output)
    calls: list[str] = []
    validators = (
        ("validate_completed_training_reward_contract", "reward"),
        ("validate_completed_method_auxiliary_contract", "auxiliary"),
        ("validate_completed_training_stream_contract", "stream"),
        ("validate_completed_training_signature", "signature"),
    )
    for attribute, label in validators:
        monkeypatch.setattr(
            RUNTIME,
            attribute,
            lambda *_args, label=label, **_kwargs: calls.append(label),
        )

    expected = [label for _, label in validators]
    assert RUNTIME.job_complete(job, output, _context(tmp_path))
    assert calls == expected

    # A second status/skip decision must re-run trainer-evidence validation;
    # it may not trust the previous job_result.json decision.
    calls.clear()
    assert RUNTIME.job_complete(job, output, _context(tmp_path))
    assert calls == expected

    calls.clear()
    (output / "training_result.json").unlink()
    assert not RUNTIME.job_complete(job, output, _context(tmp_path))
    assert calls == []
    (output / "training_result.json").write_text("not-json", encoding="utf-8")
    assert not RUNTIME.job_complete(job, output, _context(tmp_path))
    assert calls == []


@pytest.mark.parametrize(
    "validator_name",
    [
        "validate_completed_training_reward_contract",
        "validate_completed_method_auxiliary_contract",
    ],
)
def test_job_complete_rejects_checkpoint_corruption_reported_by_contract_validator(
    tmp_path: Path, monkeypatch, validator_name: str,
) -> None:
    job = _job()
    output = tmp_path / validator_name
    _write_completed_test_job(job, output)
    for attribute in (
        "validate_completed_training_reward_contract",
        "validate_completed_method_auxiliary_contract",
        "validate_completed_training_stream_contract",
        "validate_completed_training_signature",
    ):
        monkeypatch.setattr(RUNTIME, attribute, lambda *_args, **_kwargs: None)

    def reject_corrupt_checkpoint(
        _job_payload, _context_payload, _training_result, checkpoint: Path,
    ) -> None:
        if checkpoint.read_bytes() != b"selected":
            raise RuntimeError("contract checkpoint corruption")

    monkeypatch.setattr(RUNTIME, validator_name, reject_corrupt_checkpoint)
    assert RUNTIME.job_complete(job, output, _context(tmp_path))
    (output / "checkpoint_selected.pt").write_bytes(b"corrupt")
    assert not RUNTIME.job_complete(job, output, _context(tmp_path))


@pytest.mark.parametrize("damage", ["delete-result", "corrupt-checkpoint"])
def test_run_job_never_skips_a_damaged_completed_training_directory(
    tmp_path: Path, monkeypatch, damage: str,
) -> None:
    context = _context(tmp_path)
    context["dataset"].mkdir()
    job = _job()
    output = RUNTIME.output_dir(job, context)
    _write_completed_test_job(job, output)
    for validator in (
        "validate_completed_training_reward_contract",
        "validate_completed_method_auxiliary_contract",
        "validate_completed_training_stream_contract",
        "validate_completed_training_signature",
    ):
        monkeypatch.setattr(RUNTIME, validator, lambda *_args, **_kwargs: None)
    if damage == "delete-result":
        (output / "training_result.json").unlink()
    else:
        def reject_corruption(_job, _context, _result, checkpoint: Path) -> None:
            if checkpoint.read_bytes() != b"selected":
                raise RuntimeError("corrupt checkpoint")

        monkeypatch.setattr(
            RUNTIME,
            "validate_completed_training_reward_contract",
            reject_corruption,
        )
        (output / "checkpoint_selected.pt").write_bytes(b"corrupt")
    marker = tmp_path / "unexpected-execution"
    job["test_command"] = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
    ]

    assert not RUNTIME.job_complete(job, output, context)
    with pytest.raises(RuntimeError, match="refusing fresh training"):
        RUNTIME.run_job(job, context, 0, False, False)
    assert not marker.exists()


def test_one_job_run_and_valid_resume_skip(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context["dataset"].mkdir()
    job = _job()
    output = RUNTIME.output_dir(job, context)
    job["test_command"] = _artifact_command(output)
    assert RUNTIME.run_job(job, context, 0, False, False)
    assert RUNTIME.job_complete(job, output)
    result = json.loads((output / "job_result.json").read_text())
    assert result["peak_cpu_memory_bytes"] >= 0
    assert result["peak_gpu_memory_bytes"] >= 0
    job["test_command"] = _artifact_command(output, exit_code=19)
    assert RUNTIME.run_job(job, context, 0, True, False)


def test_final_validation_audit_is_required_when_registered(tmp_path: Path) -> None:
    context = _context(tmp_path)
    job = _job()
    job["final_validation_views"] = 500
    output = RUNTIME.output_dir(job, context)
    output.mkdir(parents=True)
    (output / "checkpoint_selected.pt").write_bytes(b"checkpoint")
    (output / "validation_summary.json").write_text("{}")
    (output / "best_overall.ckpt").write_bytes(b"overall")
    (output / "best_within_5000.ckpt").write_bytes(b"within")
    (output / "training_result.json").write_text("{}")
    (output / "job_result.json").write_text(json.dumps({"status": "passed"}))
    assert not RUNTIME.job_complete(job, output)
    (output / "validation_final_audit.json").write_text("{}")
    assert RUNTIME.job_complete(job, output)


def test_training_completion_requires_each_distinct_manifest_checkpoint(
    tmp_path: Path,
) -> None:
    job = _job()
    output = tmp_path / "job"
    output.mkdir()
    (output / "checkpoint_selected.pt").write_bytes(b"selected")
    (output / "validation_summary.json").write_text("{}")
    (output / "best_overall.ckpt").write_bytes(b"overall")
    (output / "training_result.json").write_text("{}")
    (output / "job_result.json").write_text(json.dumps({"status": "passed"}))

    required = RUNTIME.required_training_artifacts(job, output)
    assert required.count(output / "best_overall.ckpt") == 1
    assert output / "best_within_5000.ckpt" in required
    assert not RUNTIME.job_complete(job, output)

    (output / "best_within_5000.ckpt").write_bytes(b"within")
    assert RUNTIME.job_complete(job, output)


def test_successful_training_process_fails_without_manifest_checkpoint(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    context["dataset"].mkdir()
    job = _job()
    output = RUNTIME.output_dir(job, context)
    job["test_command"] = _artifact_command(
        output, include_manifest_checkpoints=False
    )

    assert not RUNTIME.run_job(job, context, 0, False, False)
    result = json.loads((output / "job_result.json").read_text())
    assert result["returncode"] == 0
    assert result["status"] == "failed"


def test_failure_stops_only_its_serial_queue(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context["dataset"].mkdir()
    first = _job("train__R__am_evrptw__Cus100__seed1234")
    second = _job("train__R__am_evrptw__Cus100__seed2345")
    marker = tmp_path / "must_not_run"
    first["test_command"] = [sys.executable, "-c", "raise SystemExit(7)"]
    second["test_command"] = [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).write_text('bad')"]
    failures: list[str] = []
    RUNTIME.STOP.clear()
    RUNTIME.worker(0, [first, second], context, 0, False, False, failures)
    assert failures == [first["job_id"]]
    assert not marker.exists()


def test_dry_run_does_not_execute_command(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context["dataset"].mkdir()
    marker = tmp_path / "not_created"
    job = _job()
    job["test_command"] = [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).write_text('bad')"]
    assert RUNTIME.run_job(job, context, 0, False, True)
    assert not marker.exists()


def test_signal_is_propagated_to_child_process_groups(monkeypatch) -> None:
    class Child:
        pid = 4242

    observed = []
    RUNTIME.STOP.clear()
    RUNTIME.CHILDREN[0] = Child()
    monkeypatch.setattr(RUNTIME.os, "killpg", lambda pid, signal_number: observed.append((pid, signal_number)))
    RUNTIME.handle_signal(RUNTIME.signal.SIGTERM, None)
    RUNTIME.CHILDREN.clear()
    assert observed == [(4242, RUNTIME.signal.SIGTERM)]
    assert RUNTIME.STOP.is_set()


def test_training_command_passes_frozen_rollout_budget_to_all_trainers(tmp_path: Path) -> None:
    context = _context(tmp_path)
    common = {
        "train_index": "train.parquet",
        "validation_index": "val.parquet",
        "training_epochs": 25,
        "training_rollout_steps": 140,
        "optimizer_name": "adamw",
        "optimizer_weight_decay": 0.01,
        "physical_batch_size": 4,
        "effective_batch_size": 4,
        "validation_views": 100,
        "validation_decode_type": "sampling",
        "validation_candidate_count": 100,
        "training_trajectory_count": 100,
        "training_stream_path": "artifacts/stream.parquet",
        "customer_exposure_budget": 10_000,
        "exposure_checkpoints": [2_500, 5_000, 10_000],
        "gpu_hour_checkpoints": [6, 12, 24],
        "final_validation_views": 500,
        "validation_every_epochs": 50,
        "validation_checkpoints": 1,
        "protocol_id": "rollout-budget-test",
        "run_mode": "full",
    }
    for method, module in (
        ("am_evrptw", "EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.train"),
        ("evrptw_rl", "EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.train"),
        ("drl_ts", "EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.train"),
        ("terran", "EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.train"),
    ):
        job = _job(f"train__R__{method}__Cus100__seed1234")
        job.update(common)
        job.update({"method": method, "train_module": module})
        command = RUNTIME.training_command(
            job, context, tmp_path / method, resume=False
        )
        index = command.index("--training-rollout-steps")
        assert command[index + 1] == "140"
        validation_steps_index = command.index("--validation-rollout-steps")
        assert command[validation_steps_index + 1] == "210"
        epoch_index = command.index("--training-epochs")
        assert command[epoch_index + 1] == "25"
        validation_index = command.index("--validation-every-epochs")
        assert command[validation_index + 1] == "50"
        decode_index = command.index("--validation-decode-type")
        assert command[decode_index + 1] == "sampling"
        candidate_index = command.index("--validation-candidates")
        assert command[candidate_index + 1] == "100"
        optimizer_index = command.index("--optimizer")
        assert command[optimizer_index + 1] == "adamw"
        weight_decay_index = command.index("--weight-decay")
        assert command[weight_decay_index + 1] == "0.01"
        if method == "terran":
            assert command.count("--num-customers") == 1
            customer_index = command.index("--num-customers")
            assert command[customer_index + 1] == "100"
            trajectory_index = command.index("--n-traj")
            assert command[trajectory_index + 1] == "100"
        else:
            trajectory_index = command.index("--samples-per-instance")
            assert command[trajectory_index + 1] == "100"
        final_validation_index = command.index("--final-validation-limit")
        assert command[final_validation_index + 1] == "500"
        assert command.count("--exposure-checkpoints") == 1
        exposure_index = command.index("--exposure-checkpoints")
        assert command[exposure_index + 1] == "2500,5000,10000"
        assert command.count("--gpu-hour-checkpoints") == 1
        gpu_hours_index = command.index("--gpu-hour-checkpoints")
        assert command[gpu_hours_index + 1] == "6,12,24"
        assert "--data-passes" not in command
        assert "--num-minibatches" not in command
        assert "--ppo-step-chunk-size" not in command
        if method == "terran":
            job["num_minibatches"] = 1
            job["ppo_step_chunk_size"] = 720
            job["terran_terminal_success_bonus"] = 1.0
            overridden = RUNTIME.training_command(
                job, context, tmp_path / method, resume=False
            )
            minibatch_index = overridden.index("--num-minibatches")
            assert overridden[minibatch_index + 1] == "1"
            chunk_index = overridden.index("--ppo-step-chunk-size")
            assert overridden[chunk_index + 1] == "720"
            bonus_index = overridden.index("--terminal-success-bonus")
            assert overridden[bonus_index + 1] == "1.0"

    non_terran = _job("train__R__am_evrptw__Cus1000__seed1234")
    non_terran.update(common)
    non_terran.update(
        {
            "method": "am_evrptw",
            "scale": "Cus1000",
            "train_module": (
                "EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.train"
            ),
            "num_minibatches": 1,
            "ppo_step_chunk_size": 736,
        }
    )
    non_terran_command = RUNTIME.training_command(
        non_terran, context, tmp_path / "non-terran", resume=False
    )
    assert "--num-minibatches" not in non_terran_command
    assert "--ppo-step-chunk-size" not in non_terran_command


def test_resume_only_marks_jobs_with_complete_resume_evidence(tmp_path: Path) -> None:
    job = _job()
    output = tmp_path / "job"
    output.mkdir()
    assert not RUNTIME.should_resume_job(job, output, True)
    (output / "data_pass_state.json").write_text("{}")
    try:
        RUNTIME.should_resume_job(job, output, True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("partial resume evidence was accepted")
    (output / "checkpoint_latest.pt").write_bytes(b"checkpoint")
    assert RUNTIME.should_resume_job(job, output, True)


def test_gpu_name_pattern_supports_controlled_aliases() -> None:
    accepted = "RTX A6000|RTX 6000 Ada Generation"
    assert RUNTIME.gpu_name_matches("NVIDIA RTX A6000", accepted)
    assert RUNTIME.gpu_name_matches("NVIDIA RTX 6000 Ada Generation", accepted)
    assert not RUNTIME.gpu_name_matches("NVIDIA GeForce RTX 3090", accepted)


def test_explicit_slot_gpu_mapping_preserves_physical_gpu_identity() -> None:
    assert RUNTIME.parse_slot_gpu_map(
        None, slots={0, 1}, local_gpu_count=2
    ) == {0: 0, 1: 1}
    assert RUNTIME.parse_slot_gpu_map(
        "1:1", slots={1}, local_gpu_count=2
    ) == {1: 1}
    with pytest.raises(ValueError, match="map exactly"):
        RUNTIME.parse_slot_gpu_map("0:0", slots={1}, local_gpu_count=2)
    with pytest.raises(ValueError, match="outside"):
        RUNTIME.parse_slot_gpu_map("1:2", slots={1}, local_gpu_count=2)
    with pytest.raises(ValueError, match="concurrent slots"):
        RUNTIME.parse_slot_gpu_map("0:1,1:1", slots={0, 1}, local_gpu_count=2)


def test_dedicated_job_routing_fails_closed_on_namespace_or_gpu_drift() -> None:
    job = {
        **_job("replacement"),
        "global_slot": 1,
        "required_launcher_id": "replacement-v1",
        "required_local_gpu": 1,
    }
    RUNTIME.validate_job_routing(
        [job], launcher_id="replacement-v1", slot_gpu_map={1: 1}
    )
    with pytest.raises(ValueError, match="requires launcher"):
        RUNTIME.validate_job_routing(
            [job], launcher_id="default", slot_gpu_map={1: 1}
        )
    with pytest.raises(ValueError, match="requires local GPU"):
        RUNTIME.validate_job_routing(
            [job], launcher_id="replacement-v1", slot_gpu_map={1: 0}
        )


def test_main_routes_a_single_selected_slot_to_explicit_gpu1(
    tmp_path: Path, monkeypatch,
) -> None:
    job = {
        **_job("replacement"),
        "enabled": True,
        "run_mode": "full",
        "global_slot": 1,
        "scale": "Cus1000",
        "required_launcher_id": "replacement-v1",
        "required_local_gpu": 1,
    }
    manifest = tmp_path / "replacement.jsonl"
    manifest.write_text(json.dumps(job) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "drl_job_runtime.py",
            "full",
            "--manifest",
            str(manifest),
            "--slots",
            "1",
            "--slot-gpu-map",
            "1:1",
            "--launcher-id",
            "replacement-v1",
            "--local-gpu-count",
            "2",
            "--gpu-name-pattern",
            "unused",
        ],
    )
    monkeypatch.setattr(RUNTIME, "preflight", lambda *_args: _context(tmp_path))
    routed = []
    monkeypatch.setattr(
        RUNTIME,
        "run_job",
        lambda selected, context, local_gpu, resume, dry_run: (
            routed.append((selected["global_slot"], local_gpu, context["launcher_id"]))
            or True
        ),
    )
    RUNTIME.STOP.clear()
    RUNTIME.main()
    assert routed == [(1, 1, "replacement-v1")]


def test_job_loading_filters_formal_seed_and_scale(
    tmp_path: Path,
) -> None:
    rows = []
    for seed in (1234, 2345):
        for scale in ("Cus50", "Cus1000"):
            rows.append(
                {
                    "job_id": f"full-{seed}-{scale}",
                    "run_mode": "full",
                    "seed": seed,
                    "scale": scale,
                    "global_slot": 0,
                    "queue_position": len(rows),
                    "enabled": True,
                }
            )
    manifest = tmp_path / "jobs.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    formal = RUNTIME.load_jobs(
        manifest, {0}, "full", seeds={2345}, scales={"Cus50"}
    )
    assert [row["job_id"] for row in formal] == ["full-2345-Cus50"]


def _terran_job():
    job = _job("full__G__Full-support__terran__Cus100__seed1234")
    job.update(
        method="terran",
        training_gamma=1.0,
        reward_contract_id="terran_undiscounted_distance_pbrs_v1",
    )
    return job


def _cost_job(method="am_evrptw"):
    # Read the checked-in scheduling contract; command/validator tests should
    # not regenerate every server queue from unrelated local training artifacts.
    manifest = ROOT / "scripts/rq_v1/2080ti_4_1/jobs.jsonl"
    row = next(json.loads(line) for line in manifest.read_text().splitlines()
               if json.loads(line)["method"] == method)
    # These legacy runtime fixtures use a synthetic repository. Machine profile
    # routing and hash checks are exercised in test_terran_machine_profile_runtime.
    row.pop("terran_config_path", None)
    row.pop("terran_config_sha256", None)
    return row


def _formal_stream_completion_fixture(tmp_path: Path, monkeypatch):
    manifest = ROOT / "scripts/rq_v1/a6000_2_1/jobs.jsonl"
    job = next(
        item for item in map(json.loads, manifest.read_text().splitlines())
        if item["method"] == "am_evrptw" and item["scale"] == "Cus1000"
    )
    repository = ROOT.parents[1]
    relative_stream = Path(job["training_stream_path"])
    source_stream = repository / relative_stream
    copied_stream = tmp_path / relative_stream
    copied_stream.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_stream, copied_stream)
    shutil.copy2(
        source_stream.with_suffix(source_stream.suffix + ".manifest.json"),
        copied_stream.with_suffix(copied_stream.suffix + ".manifest.json"),
    )

    context = {**_context(tmp_path), "repo": tmp_path}
    output = tmp_path / "formal-completed"
    output.mkdir(parents=True)
    for artifact in RUNTIME.required_training_artifacts(job, output):
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"artifact")

    snapshot = job["training_stream_contract_snapshot"]
    training_result = {
        "status": "passed",
        "protocol_id": job["protocol_id"],
        "requested_training_epochs": job["training_epochs"],
        "completed_training_epochs": job["training_epochs"],
        "early_stopped": False,
        "training_stream_contract_snapshot": snapshot,
        "training_stream_contract_sha256": job[
            "training_stream_contract_sha256"
        ],
    }
    (output / "training_result.json").write_text(
        json.dumps(training_result), encoding="utf-8"
    )
    checkpoint_payload = {
        "training_stream_contract": snapshot,
        "args": {
            "training_stream_contract_snapshot": snapshot,
            "training_stream_contract_sha256": job[
                "training_stream_contract_sha256"
            ],
        },
    }
    torch.save(checkpoint_payload, output / "checkpoint_selected.pt")

    contract = RUNTIME.training_contract(job)
    (output / "job_result.json").write_text(
        json.dumps({"status": "passed", **contract}), encoding="utf-8"
    )
    (output / "provenance.json").write_text(
        json.dumps({"job": job, **contract}), encoding="utf-8"
    )
    for validator in (
        "validate_completed_training_reward_contract",
        "validate_completed_method_auxiliary_contract",
        "validate_completed_training_signature",
    ):
        monkeypatch.setattr(RUNTIME, validator, lambda *_args, **_kwargs: None)
    return job, output, context, copied_stream, training_result, checkpoint_payload


def test_formal_completion_revalidates_stream_result_checkpoint_and_provenance(
    tmp_path: Path, monkeypatch,
) -> None:
    (
        job,
        output,
        context,
        copied_stream,
        training_result,
        checkpoint_payload,
    ) = _formal_stream_completion_fixture(tmp_path, monkeypatch)
    assert RUNTIME.job_complete(job, output, context)

    job_result = json.loads((output / "job_result.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    for document in (job_result, provenance):
        assert (
            document["training_stream_contract_sha256"]
            == job["training_stream_contract_sha256"]
        )
        assert (
            document["training_stream_contract_snapshot"]
            == job["training_stream_contract_snapshot"]
        )

    stale_result = {
        **training_result,
        "training_stream_contract_sha256": "0" * 64,
    }
    (output / "training_result.json").write_text(
        json.dumps(stale_result), encoding="utf-8"
    )
    assert not RUNTIME.job_complete(job, output, context)
    (output / "training_result.json").write_text(
        json.dumps(training_result), encoding="utf-8"
    )

    stale_checkpoint = {
        **checkpoint_payload,
        "args": {
            **checkpoint_payload["args"],
            "training_stream_contract_sha256": "f" * 64,
        },
    }
    torch.save(stale_checkpoint, output / "checkpoint_selected.pt")
    assert not RUNTIME.job_complete(job, output, context)
    torch.save(checkpoint_payload, output / "checkpoint_selected.pt")

    # Logical sequence hashing must notice a changed view ID even though the
    # Parquet path, row count, and contiguous positions remain unchanged.
    frame = pd.read_parquet(copied_stream)
    frame.loc[0, "view_id"] = str(frame.loc[0, "view_id"]) + "__tampered"
    frame.to_parquet(copied_stream, index=False)
    assert not RUNTIME.job_complete(job, output, context)


def test_stream_preflight_validates_each_method_specific_exact_snapshot(
    tmp_path: Path, monkeypatch,
) -> None:
    manifest = ROOT / "scripts/rq_v1/2080ti_4_2/jobs.jsonl"
    jobs = [
        item for item in map(json.loads, manifest.read_text().splitlines())
        if item["scale"] == "Cus100" and item["representation"] == "G"
    ]
    # This test specifically exercises full content revalidation on copied
    # streams, independently of the launcher no-rehash mode.
    for job in jobs:
        job.pop("stream_integrity_mode", None)
        job["file_hash_validation_performed"] = True
    assert {job["method"] for job in jobs} == RUNTIME.METHODS
    assert len({job["training_stream_path"] for job in jobs}) == 4
    repository = tmp_path / "repo"
    source_repository = ROOT.parents[1]
    for relative in {job["training_stream_path"] for job in jobs}:
        source = source_repository / relative
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        shutil.copy2(
            source.with_suffix(source.suffix + ".manifest.json"),
            destination.with_suffix(destination.suffix + ".manifest.json"),
        )
    registry_relative = Path(jobs[0]["training_stream_registry_path"])
    copied_registry = repository / registry_relative
    copied_registry.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_repository / registry_relative, copied_registry)
    dataset = tmp_path / "dataset"
    index = dataset / jobs[0]["train_index"]
    index.parent.mkdir(parents=True)
    index.write_bytes(b"source-index-placeholder")
    marker_relative = Path(jobs[0]["artifact_preparation_marker_path"])
    marker = json.loads(
        (source_repository / marker_relative).read_text(encoding="utf-8")
    )
    marker["dataset_root"] = str(dataset)
    marker["training_stream_contracts"] = [
        {"relative_path": job["training_stream_path"],
         "sha256": job["training_stream_contract_sha256"],
         "snapshot": job["training_stream_contract_snapshot"]}
        for job in jobs
    ]
    # The host marker may describe no-rehash launch reuse. This isolated test
    # requests full file validation, so give its copied marker/registry a new,
    # self-consistent identity without editing host artifacts.
    marker["file_hash_validation_performed"] = True
    marker.pop("stream_integrity_mode", None)
    marker["marker_sha256"] = RUNTIME.hashlib.sha256(json.dumps(
        {key: value for key, value in marker.items()
         if key not in {"marker_sha256", "dataset_root"}},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    registry = json.loads(copied_registry.read_text())
    registry["artifact_preparation_marker_sha256"] = marker["marker_sha256"]
    registry["sha256"] = RUNTIME.hashlib.sha256(json.dumps(
        {key: value for key, value in registry.items() if key != "sha256"},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    copied_registry.write_text(json.dumps(registry))
    for job in jobs:
        job["artifact_preparation_marker_sha256"] = marker["marker_sha256"]
        job["training_stream_registry_sha256"] = registry["sha256"]
    copied_marker = repository / marker_relative
    copied_marker.parent.mkdir(parents=True, exist_ok=True)
    copied_marker.write_text(json.dumps(marker), encoding="utf-8")
    expected_source_sha = jobs[0]["training_stream_contract_snapshot"][
        "source_index_sha256"
    ]
    monkeypatch.setattr(RUNTIME, "file_sha256", lambda _path: expected_source_sha)

    RUNTIME.validate_training_stream_contracts(jobs, repository, dataset)

    tampered_marker = json.loads(json.dumps(marker))
    matching_entry = next(
        item
        for item in tampered_marker["training_stream_contracts"]
        if item["relative_path"] == jobs[0]["training_stream_path"]
    )
    matching_entry["sha256"] = "a" * 64
    canonical_marker = {
        key: value
        for key, value in tampered_marker.items()
        if key not in {"marker_sha256", "dataset_root"}
    }
    tampered_marker["marker_sha256"] = RUNTIME.hashlib.sha256(
        json.dumps(
            canonical_marker,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    copied_marker.write_text(json.dumps(tampered_marker), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest artifact marker SHA256 mismatch"):
        RUNTIME.validate_training_stream_contracts(jobs, repository, dataset)
    copied_marker.write_text(json.dumps(marker), encoding="utf-8")

    stale_sha = [dict(job) for job in jobs]
    stale_sha[0]["training_stream_contract_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="training-stream contract mismatch"):
        RUNTIME.validate_training_stream_contracts(
            stale_sha, repository, dataset
        )

    stale_snapshot = [dict(job) for job in jobs]
    stale_snapshot[0]["training_stream_contract_snapshot"] = {
        **stale_snapshot[0]["training_stream_contract_snapshot"],
        "sample_count": 1,
    }
    with pytest.raises(RuntimeError, match="training-stream contract mismatch"):
        RUNTIME.validate_training_stream_contracts(
            stale_snapshot, repository, dataset
        )


def test_preverified_terran_stream_path_skips_large_rehashes_but_checks_metadata(
    tmp_path: Path, monkeypatch,
) -> None:
    repository = tmp_path / "repo"
    dataset = tmp_path / "dataset"
    stream_relative = Path("artifacts/terran-cus100.parquet")
    stream_path = repository / stream_relative
    stream_path.parent.mkdir(parents=True)
    stream_path.write_bytes(b"preverified-parquet-placeholder")
    dataset.mkdir()
    (dataset / "train.parquet").write_bytes(b"source-index-placeholder")

    snapshot = {
        "schema": "drl_training_stream_contract_v1",
        "stream_schema": "drl_training_id_stream_v3",
        "content_digest_scheme": "sha256_length_prefixed_ordered_view_ids_v1",
        "stream_content_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "sample_count": 2,
        "scale": "Cus100",
        "seed": 1234,
        "source_index_sha256": "c" * 64,
        "allowed_family_ids_sha256": None,
        "sha256": "d" * 64,
    }
    stream_manifest = {
        "schema": snapshot["stream_schema"],
        "content_digest_scheme": snapshot["content_digest_scheme"],
        "stream_content_sha256": snapshot["stream_content_sha256"],
        "manifest_sha256": snapshot["manifest_sha256"],
        "sample_count": snapshot["sample_count"],
        "scale": snapshot["scale"],
        "seed": snapshot["seed"],
        "source_index_sha256": snapshot["source_index_sha256"],
        "allowed_family_ids_sha256": None,
        "file_hash_validation_performed": True,
    }
    stream_path.with_suffix(".parquet.manifest.json").write_text(
        json.dumps(stream_manifest), encoding="utf-8"
    )
    registry_relative = Path("artifacts/registry.json")
    marker_relative = Path("artifacts/marker.json")
    registry_sha = "e" * 64
    marker_sha = "f" * 64
    registry_key = "G/Full-support/terran/Cus100/seed_1234"
    (repository / registry_relative).write_text(
        json.dumps(
            {
                "schema": "drl_training_stream_registry_v1",
                "source_scope": "training_split_and_track_only",
                "sha256": registry_sha,
                "artifact_preparation_marker_sha256": marker_sha,
                "streams": {
                    registry_key: {
                        "path": str(stream_relative),
                        "snapshot": snapshot,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (repository / marker_relative).write_text(
        json.dumps(
            {
                "schema": "drl_rq_artifact_preparation_v2",
                "status": "passed",
                "file_hash_validation_performed": False,
                "stream_integrity_mode": "reuse_preverified_snapshot_no_rehash",
                "marker_sha256": marker_sha,
                "dataset_root": str(dataset),
                "training_stream_contracts": [
                    {
                        "relative_path": str(stream_relative),
                        "sha256": snapshot["sha256"],
                        "snapshot": snapshot,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    job = {
        **_terran_job(),
        "representation": "G",
        "condition": "Full-support",
        "train_index": "train.parquet",
        "training_stream_path": str(stream_relative),
        "training_stream_contract_sha256": snapshot["sha256"],
        "training_stream_contract_snapshot": snapshot,
        "training_stream_registry_path": str(registry_relative),
        "training_stream_registry_sha256": registry_sha,
        "artifact_preparation_marker_path": str(marker_relative),
        "artifact_preparation_marker_sha256": marker_sha,
        "target_environments": 2,
        "customer_exposure_budget": 200,
        "stream_integrity_mode": "reuse_preverified_snapshot_no_rehash",
        "file_hash_validation_performed": False,
    }
    monkeypatch.setattr(
        RUNTIME,
        "load_training_stream_contract",
        lambda *_args, **_kwargs: pytest.fail("stream content was rehashed"),
    )
    monkeypatch.setattr(
        RUNTIME,
        "file_sha256",
        lambda *_args, **_kwargs: pytest.fail("source index was rehashed"),
    )

    RUNTIME.validate_training_stream_contracts(
        [job], repository, dataset, reuse_preverified=True
    )
    with pytest.raises(RuntimeError, match="requires --reuse-preverified"):
        RUNTIME.validate_training_stream_contracts(
            [job], repository, dataset, reuse_preverified=False
        )


def test_preverified_terran_completion_uses_saved_snapshot_without_rehash(
    tmp_path: Path, monkeypatch,
) -> None:
    snapshot = {
        "schema": "drl_training_stream_contract_v1",
        "sha256": "a" * 64,
    }
    job = {
        **_terran_job(),
        "training_stream_path": "unused.parquet",
        "training_stream_contract_sha256": snapshot["sha256"],
        "training_stream_contract_snapshot": snapshot,
        "stream_integrity_mode": "reuse_preverified_snapshot_no_rehash",
        "file_hash_validation_performed": False,
    }
    training_result = {
        "training_stream_contract_snapshot": snapshot,
        "training_stream_contract_sha256": snapshot["sha256"],
        "stream_integrity_mode": "reuse_preverified_snapshot_no_rehash",
    }
    checkpoint = tmp_path / "selected.pt"
    torch.save(
        {
            "config": {
                "protocol": {
                    "training_stream_contract_snapshot": snapshot,
                    "training_stream_contract_sha256": snapshot["sha256"],
                    "stream_integrity_mode": (
                        "reuse_preverified_snapshot_no_rehash"
                    ),
                }
            }
        },
        checkpoint,
    )
    monkeypatch.setattr(
        RUNTIME,
        "load_training_stream_contract",
        lambda *_args, **_kwargs: pytest.fail("completion rehashed stream"),
    )

    RUNTIME.validate_completed_training_stream_contract(
        job,
        {"repo": tmp_path},
        training_result,
        checkpoint,
        reuse_preverified=True,
    )


def test_all_formal_cost_manifests_pass_and_commands_forward_profile(tmp_path):
    for method in sorted(RUNTIME.METHODS):
        job = _cost_job(method)
        RUNTIME.validate_objective_contracts([job])
        RUNTIME.validate_terran_training_contracts([job])
        RUNTIME.validate_method_auxiliary_contracts([job])
        RUNTIME.validate_optimizer_contracts([job])
        context = _context(tmp_path)
        command = RUNTIME.training_command(job, context, tmp_path / "run", False)
        assert command[command.index("--objective-config") + 1] == str(
            context["repo"] / job["objective_config_path"]
        )
        assert RUNTIME.training_contract(job)["objective_config"] == job["objective_config"]
        assert RUNTIME.training_contract(job)["optimizer_name"] == "adamw"
        assert RUNTIME.training_contract(job)["optimizer_weight_decay"] == 0.01
        stream_sha_index = command.index("--training-stream-contract-sha256")
        assert command[stream_sha_index + 1] == job[
            "training_stream_contract_sha256"
        ]
        contract = RUNTIME.training_contract(job)
        assert contract["training_stream_contract_snapshot"] == job[
            "training_stream_contract_snapshot"
        ]
        assert contract["artifact_preparation_marker_sha256"] == job[
            "artifact_preparation_marker_sha256"
        ]
        auxiliary_path = job.get("method_auxiliary_profile_path")
        if auxiliary_path is not None:
            assert command[command.index("--method-auxiliary-profile") + 1] == str(
                context["repo"] / auxiliary_path
            )
        if method == "terran":
            bonus_index = command.index("--terminal-success-bonus")
            assert float(command[bonus_index + 1]) == job[
                "terran_terminal_success_bonus"
            ]
            assert RUNTIME.training_contract(job)[
                "terran_terminal_success_bonus"
            ] == job["terran_terminal_success_bonus"]
            assert RUNTIME.expected_resolved_training_signature(
                job, context
            )["method_specific"]["task_reward"] == {
                "terminal_success_bonus": job[
                    "terran_terminal_success_bonus"
                ],
                "unit": "normalized_objective_cost",
            }


@pytest.mark.parametrize("field", ["optimizer_name", "optimizer_weight_decay"])
def test_optimizer_preflight_rejects_stale_scientific_metadata(field):
    job = _cost_job()
    job[field] = "adam" if field == "optimizer_name" else 0.0
    with pytest.raises(RuntimeError, match="optimizer contract mismatch"):
        RUNTIME.validate_optimizer_contracts([job])


@pytest.mark.parametrize("method", sorted(RUNTIME.METHODS))
def test_cost_eval_and_transfer_also_require_the_checkpoint_objective(tmp_path, method):
    job = _cost_job(method)
    job.update(
        kind="evaluate", eval_module="method.eval", dataset_index="test.parquet",
        track_id="test1_new_seed", candidate_count=100, candidate_chunk_size=1,
        expected_views=500, decode_type="sampling", source_scale="Cus100",
        scale="Cus2000",
    )
    context = _context(tmp_path)
    command = RUNTIME.evaluation_command(job, context, tmp_path / "eval")
    assert command[command.index("--objective-config") + 1] == str(
        context["repo"] / job["objective_config_path"]
    )
    RUNTIME.validate_objective_contracts([job])
    job["objective_config"] = {**job["objective_config"], "mode": "distance"}
    with pytest.raises(RuntimeError, match="objective contract mismatch"):
        RUNTIME.validate_objective_contracts([job])


@pytest.mark.parametrize("field", ["missing", "coefficient", "path", "selection"])
def test_objective_preflight_rejects_stale_scientific_metadata(field):
    job = _cost_job()
    if field == "missing":
        job.pop("objective_config")
    elif field == "coefficient":
        job["objective_config"] = {**job["objective_config"], "vehicle_fixed_cost_usd": 0.0}
    elif field == "path":
        job["objective_config_path"] = "old-profile.json"
    else:
        job["candidate_selection"] = "verifier_feasible_then_min_directed_distance"
    with pytest.raises(RuntimeError, match="objective contract mismatch"):
        RUNTIME.preflight(object(), [job])


@pytest.mark.parametrize("method", sorted(RUNTIME.METHODS))
def test_all_methods_completion_is_bound_to_exact_cost_profile(
    tmp_path, method, monkeypatch
):
    context = _context(tmp_path)
    context["dataset"].mkdir()
    job = _cost_job(method)
    output = RUNTIME.output_dir(job, context)
    job["test_command"] = _artifact_command(output)
    checked = []
    monkeypatch.setattr(
        RUNTIME,
        "validate_completed_training_reward_contract",
        lambda checked_job, *_args: checked.append(checked_job["method"]),
    )
    monkeypatch.setattr(
        RUNTIME,
        "validate_completed_method_auxiliary_contract",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        RUNTIME,
        "validate_completed_training_stream_contract",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        RUNTIME,
        "validate_completed_training_signature",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        RUNTIME,
        "validate_training_result_outcome",
        lambda *_args, **_kwargs: None,
    )
    assert RUNTIME.run_job(job, context, 0, False, False)
    assert checked == [method]
    assert RUNTIME.job_complete(job, output)
    result_path = output / "job_result.json"
    result = json.loads(result_path.read_text())
    result["objective_config"]["vehicle_fixed_cost_usd"] = 0.0
    result_path.write_text(json.dumps(result))
    assert not RUNTIME.job_complete(job, output)
    with pytest.raises(RuntimeError, match="refusing fresh training"):
        RUNTIME.run_job(job, context, 0, False, False)


@pytest.mark.parametrize("method", sorted(RUNTIME.METHODS))
def test_completed_training_reward_contract_is_cross_checked(
    tmp_path, method
):
    job = _cost_job(method)
    repo = RUNTIME.ROOT.parents[1]
    context = {**_context(tmp_path), "repo": repo}
    contract = RUNTIME.load_reward_contract(
        repo / job["reward_contract_config_path"]
    )
    terms = contract.for_scale(job["scale"], job["objective_config"])
    snapshot = contract.to_dict()
    derived = {
        "reward_contract_id": terms.contract_id,
        "reward_contract_sha256": terms.digest,
        "reward_contract_scale": terms.scale_label,
        "reward_objective_scale": terms.objective_scale,
        "reward_failure_base": terms.failure_base,
        "reward_unserved_coefficient": terms.unserved_coefficient,
    }
    result = {
        "objective_config": terms.objective_config.to_dict(),
        "reward_contract_snapshot": snapshot,
        **derived,
    }
    checkpoint = tmp_path / f"{method}.pt"
    if method == "terran":
        completion_bonus = float(job["terran_terminal_success_bonus"])
        terminal_fields = {
            "terran_terminal_success_bonus": completion_bonus,
            "terran_terminal_success_bonus_unit": "normalized_objective_cost",
            "terran_terminal_success_bonus_equivalent_usd": (
                completion_bonus * terms.objective_scale
            ),
        }
        result.update(terminal_fields)
        torch.save(
            {
                "config": {
                    "objective": terms.objective_config.to_dict(),
                    "reward_contract": snapshot,
                    "training": {"reward_contract_id": terms.contract_id},
                    "normalization": {
                        "reward_contract_sha256": terms.digest,
                        "reward_contract_scale": terms.scale_label,
                        "reward_objective_scale": terms.objective_scale,
                        "failure_base": terms.failure_base,
                        "unserved_coefficient": terms.unserved_coefficient,
                        **terminal_fields,
                    },
                    "pbrs": {"terminal_success_bonus": completion_bonus},
                }
            },
            checkpoint,
        )
    else:
        torch.save(
            {
                "objective_config": terms.objective_config.to_dict(),
                "reward_contract": snapshot,
                "args": {"reward_contract_snapshot": snapshot, **derived},
            },
            checkpoint,
        )

    RUNTIME.validate_completed_training_reward_contract(
        job, context, result, checkpoint
    )
    stale_result = {**result, "reward_failure_base": terms.failure_base + 1.0}
    with pytest.raises(RuntimeError, match="training_result reward contract mismatch"):
        RUNTIME.validate_completed_training_reward_contract(
            job, context, stale_result, checkpoint
        )
    if method == "terran":
        stale_bonus_result = {
            **result,
            "terran_terminal_success_bonus": (
                result["terran_terminal_success_bonus"] + 1.0
            ),
        }
        with pytest.raises(
            RuntimeError, match="training_result reward contract mismatch"
        ):
            RUNTIME.validate_completed_training_reward_contract(
                job, context, stale_bonus_result, checkpoint
            )
        checkpoint_payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        checkpoint_payload["config"]["pbrs"]["terminal_success_bonus"] += 1.0
        torch.save(checkpoint_payload, checkpoint)
        with pytest.raises(
            RuntimeError, match="derived reward-contract fields are inconsistent"
        ):
            RUNTIME.validate_completed_training_reward_contract(
                job, context, result, checkpoint
            )


def test_completed_drl_ts_auxiliary_contract_is_cross_checked(tmp_path):
    job = _cost_job("drl_ts")
    repo = RUNTIME.ROOT.parents[1]
    context = {**_context(tmp_path), "repo": repo}
    profile = RUNTIME.load_method_auxiliary_profile(
        repo / job["method_auxiliary_profile_path"]
    ).require_method("drl_ts")
    expected = {
        "method_auxiliary_profile_id": profile.profile_id,
        "method_auxiliary_sha256": profile.digest,
        "method_auxiliary_snapshot": profile.to_dict(),
        "method_auxiliary_method": profile.method,
        "method_auxiliary_applicability": profile.applicability,
        "method_auxiliary_aggregation": profile.aggregation,
        "method_auxiliary_denominator": profile.denominator,
        "method_auxiliary_step_clip": profile.step_clip,
        "method_auxiliary_component_clip": profile.component_clip,
        "method_auxiliary_weights": dict(profile.weights),
    }
    runtime_fields = {
        "soft_violation_contract_id": profile.profile_id,
        "soft_violation_step_clip": profile.step_clip,
        "soft_violation_component_clip": profile.component_clip,
        "soft_violation_denominator": profile.denominator,
        "capacity_penalty": profile.weights["capacity"],
        "time_penalty": profile.weights["time_window"],
        "energy_penalty": profile.weights["energy"],
        "soft_stage_end_epoch": job["soft_stage_end_epoch"],
    }
    training_result = {
        **expected,
        "soft_stage_end_epoch": job["soft_stage_end_epoch"],
    }
    checkpoint = tmp_path / "drl-ts.pt"
    torch.save(
        {
            "args": {**expected, **runtime_fields},
            "method_auxiliary_profile": profile.to_dict(),
        },
        checkpoint,
    )
    RUNTIME.validate_completed_method_auxiliary_contract(
        job, context, training_result, checkpoint
    )

    stale_result = {**training_result, "method_auxiliary_step_clip": 2.0}
    with pytest.raises(
        RuntimeError, match="training_result method auxiliary mismatch"
    ):
        RUNTIME.validate_completed_method_auxiliary_contract(
            job, context, stale_result, checkpoint
        )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["args"]["time_penalty"] = 0.0
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="did not consume its auxiliary profile"):
        RUNTIME.validate_completed_method_auxiliary_contract(
            job, context, training_result, checkpoint
        )


@pytest.mark.parametrize("provenance", [None, {}, {"objective_config": {"mode": "distance"}}])
def test_cost_resume_rejects_missing_or_legacy_provenance(tmp_path, provenance):
    context = _context(tmp_path)
    job = _cost_job()
    output = RUNTIME.output_dir(job, context)
    output.mkdir(parents=True)
    (output / "data_pass_state.json").write_text("{}")
    (output / "checkpoint_latest.pt").write_bytes(b"original")
    if provenance is not None:
        (output / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(RuntimeError, match="provenance"):
        RUNTIME.run_job(job, context, 0, True, False)
    assert (output / "checkpoint_latest.pt").read_bytes() == b"original"
    assert not (output / "job_result.json").exists()


def test_terran_manifest_contract_is_checked_before_preflight_side_effects(
    monkeypatch, tmp_path: Path,
) -> None:
    config = tmp_path / "terran.yaml"
    config.write_text(
        "training:\n  gamma: 1.0\n"
        "  reward_contract_id: terran_undiscounted_distance_pbrs_v1\n"
    )
    monkeypatch.setattr(RUNTIME, "TERRAN_CONFIG", config)
    RUNTIME.validate_terran_training_contracts([_terran_job(), _job()])
    for updates in (
        {"training_gamma": 0.999},
        {"reward_contract_id": "old-contract"},
        {"reward_contract_id": None, "training_gamma": None},
    ):
        stale = {**_terran_job(), **updates}
        # The minimal object intentionally has no runtime options: contract
        # rejection precedes output writes, GPU checks, or dataset discovery.
        with pytest.raises(RuntimeError, match="manifest/config reward contract mismatch"):
            RUNTIME.preflight(object(), [stale])
    legacy = {key: value for key, value in _terran_job().items()
              if key not in {"training_gamma", "reward_contract_id"}}
    with pytest.raises(RuntimeError, match="manifest/config reward contract mismatch"):
        RUNTIME.validate_terran_training_contracts([legacy])
    # Unaffected methods do not require or load the TERRAN YAML.
    monkeypatch.setattr(RUNTIME, "TERRAN_CONFIG", tmp_path / "missing.yaml")
    RUNTIME.validate_terran_training_contracts([_job()])


def test_formal_terran_terminal_bonus_must_match_runtime_and_protocol() -> None:
    job = _cost_job("terran")
    RUNTIME.validate_terran_training_contracts([job])

    stale = {
        **job,
        "terran_terminal_success_bonus": (
            float(job["terran_terminal_success_bonus"]) + 0.5
        ),
    }
    with pytest.raises(RuntimeError, match="success-bonus contract mismatch"):
        RUNTIME.validate_terran_training_contracts([stale])


def test_versioned_terran_completion_requires_matching_saved_contract(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context["dataset"].mkdir()
    job = _terran_job()
    output = RUNTIME.output_dir(job, context)
    job["test_command"] = _artifact_command(output)
    assert RUNTIME.run_job(job, context, 0, False, False)
    assert RUNTIME.job_complete(job, output)
    result_path = output / "job_result.json"
    provenance_path = output / "provenance.json"
    result = json.loads(result_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    for field in ("training_gamma", "reward_contract_id"):
        assert result[field] == provenance[field] == provenance["job"][field] == job[field]
        missing = dict(result)
        missing.pop(field)
        result_path.write_text(json.dumps(missing))
        assert not RUNTIME.job_complete(job, output)
        result_path.write_text(json.dumps(result))
        missing = dict(provenance)
        missing.pop(field)
        provenance_path.write_text(json.dumps(missing))
        assert not RUNTIME.job_complete(job, output)
        provenance_path.write_text(json.dumps(provenance))
    stale = {**result, "training_gamma": 0.999}
    result_path.write_text(json.dumps(stale))
    assert not RUNTIME.job_complete(job, output)
    # A mismatched completed directory must neither skip nor start over it.
    with pytest.raises(RuntimeError, match="refusing fresh training"):
        RUNTIME.run_job(job, context, 0, False, False)
    assert json.loads(result_path.read_text()) == stale
    assert json.loads(provenance_path.read_text()) == provenance
    result_path.write_text(json.dumps(result))
    provenance["job"]["training_gamma"] = 0.999
    provenance_path.write_text(json.dumps(provenance))
    assert not RUNTIME.job_complete(job, output)


@pytest.mark.parametrize("relative", [
    "checkpoint_latest.pt", "data_pass_state.json", "validation_history.jsonl",
    "validation_summary_overall.json", "best_overall.ckpt", "training_result.json",
    "logs/train_log.csv", "checkpoints/checkpoint_epoch_0001.pt",
    "logical_epoch_history.jsonl", "train_history.jsonl",
])
def test_full_refuses_existing_training_state_without_overwriting(
    tmp_path: Path, relative: str,
) -> None:
    context = _context(tmp_path)
    job = _terran_job()
    output = RUNTIME.output_dir(job, context)
    marker = output / relative
    marker.parent.mkdir(parents=True)
    marker.write_bytes(b"original-training-evidence")
    job["test_command"] = [sys.executable, "-c", "raise SystemExit(99)"]
    with pytest.raises(RuntimeError, match="refusing fresh training"):
        RUNTIME.run_job(job, context, 0, False, False)
    assert marker.read_bytes() == b"original-training-evidence"
    assert not (output / "provenance.json").exists()


def test_launcher_only_files_allow_fresh_start_and_commit_changes_root(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context["dataset"].mkdir()
    job = _terran_job()
    output = RUNTIME.output_dir(job, context)
    output.mkdir(parents=True)
    (output / "stdout.log").write_text("launcher only")
    (output / "provenance.json").write_text("{}")
    job["test_command"] = _artifact_command(output)
    assert RUNTIME.run_job(job, context, 0, False, False)
    new_context = {**context, "commit": "gamma1-commit"}
    assert RUNTIME.output_dir(job, new_context) != output
    assert not RUNTIME.existing_training_state(RUNTIME.output_dir(job, new_context))


def test_resume_with_matching_evidence_remains_allowed(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context["dataset"].mkdir()
    job = _terran_job()
    output = RUNTIME.output_dir(job, context)
    output.mkdir(parents=True)
    (output / "data_pass_state.json").write_text("{}")
    (output / "checkpoint_latest.pt").write_bytes(b"checkpoint")
    job["test_command"] = _artifact_command(output)
    assert RUNTIME.run_job(job, context, 0, True, False)
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["resumed_from_checkpoint"]


@pytest.mark.parametrize("stale_field", ["training_gamma", "reward_contract_id", "missing_contract"])
def test_resume_rejects_stale_provenance_before_overwriting(
    tmp_path: Path, stale_field: str,
) -> None:
    context = _context(tmp_path)
    job = _terran_job()
    output = RUNTIME.output_dir(job, context)
    output.mkdir(parents=True)
    (output / "data_pass_state.json").write_text("original-state")
    (output / "checkpoint_latest.pt").write_bytes(b"original-checkpoint")
    stale_job = dict(job)
    if stale_field == "missing_contract":
        stale_job.pop("training_gamma")
        stale_job.pop("reward_contract_id")
    else:
        stale_job[stale_field] = 0.999 if stale_field == "training_gamma" else "old-contract"
    provenance = {"job": stale_job, **RUNTIME.training_contract(stale_job)}
    provenance_path = output / "provenance.json"
    original = json.dumps(provenance)
    provenance_path.write_text(original)
    with pytest.raises(RuntimeError, match="resume provenance reward contract mismatch"):
        RUNTIME.run_job(job, context, 0, True, False)
    assert provenance_path.read_text() == original
    assert (output / "data_pass_state.json").read_text() == "original-state"
    assert (output / "checkpoint_latest.pt").read_bytes() == b"original-checkpoint"
    assert not (output / "job_result.json").exists()


def test_refused_training_is_reported_as_queue_failure(tmp_path: Path, capsys) -> None:
    context = _context(tmp_path)
    job = _terran_job()
    output = RUNTIME.output_dir(job, context)
    output.mkdir(parents=True)
    (output / "validation_history.jsonl").write_text("old-history")
    failures = []
    RUNTIME.STOP.clear()
    RUNTIME.worker(0, [job], context, 0, False, False, failures)
    assert failures == [job["job_id"]]
    assert "refusing fresh training" in capsys.readouterr().err


def test_method_filter_applies_to_full_resume_and_status(tmp_path: Path, monkeypatch, capsys) -> None:
    rows = []
    for method in ("am_evrptw", "terran"):
        for mode in ("full", "evaluate"):
            row = _job(f"{mode}-{method}")
            row.update(method=method, run_mode=mode, enabled=True)
            if mode == "evaluate":
                row.update(kind="eval", test_id="test1", decode_budget="sample100")
            rows.append(row)
    manifest = tmp_path / "jobs.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    for mode in ("full", "resume", "evaluate", "status"):
        selected = RUNTIME.load_jobs(manifest, {0}, mode, methods={"terran"})
        assert selected and all(row["method"] == "terran" for row in selected)
    assert len(RUNTIME.load_jobs(manifest, {0}, "full")) == 2
    monkeypatch.setattr(sys, "argv", [
        "drl_job_runtime.py", "status", "--manifest", str(manifest), "--slots", "0",
        "--local-gpu-count", "1", "--gpu-name-pattern", "unused", "--methods", "terran",
    ])
    monkeypatch.setattr(RUNTIME, "preflight", lambda args, jobs: _context(tmp_path))
    RUNTIME.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["jobs"] == 2
    assert {row["job_id"] for row in payload["rows"]} == {"full-terran", "evaluate-terran"}
