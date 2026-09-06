from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


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
        f"{manifest_artifacts}"
        f"raise SystemExit({exit_code})"
    )
    return [sys.executable, "-c", source]


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
        assert "--data-passes" not in command
        assert "--num-minibatches" not in command
        assert "--ppo-step-chunk-size" not in command
        if method == "terran":
            job["num_minibatches"] = 1
            job["ppo_step_chunk_size"] = 736
            overridden = RUNTIME.training_command(
                job, context, tmp_path / method, resume=False
            )
            minibatch_index = overridden.index("--num-minibatches")
            assert overridden[minibatch_index + 1] == "1"
            chunk_index = overridden.index("--ppo-step-chunk-size")
            assert overridden[chunk_index + 1] == "736"

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
    from EVRPTW_Benchmark.Reinforcement_Learning.scripts.build_rq_server_manifests import build
    return next(dict(job) for jobs in build().values() for job in jobs if job["method"] == method)


def test_all_formal_cost_manifests_pass_and_commands_forward_profile(tmp_path):
    for method in sorted(RUNTIME.METHODS):
        job = _cost_job(method)
        RUNTIME.validate_objective_contracts([job])
        RUNTIME.validate_terran_training_contracts([job])
        RUNTIME.validate_optimizer_contracts([job])
        context = _context(tmp_path)
        command = RUNTIME.training_command(job, context, tmp_path / "run", False)
        assert command[command.index("--objective-config") + 1] == str(
            context["repo"] / job["objective_config_path"]
        )
        assert RUNTIME.training_contract(job)["objective_config"] == job["objective_config"]
        assert RUNTIME.training_contract(job)["optimizer_name"] == "adamw"
        assert RUNTIME.training_contract(job)["optimizer_weight_decay"] == 0.01


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
def test_all_methods_completion_is_bound_to_exact_cost_profile(tmp_path, method):
    context = _context(tmp_path)
    context["dataset"].mkdir()
    job = _cost_job(method)
    output = RUNTIME.output_dir(job, context)
    job["test_command"] = _artifact_command(output)
    assert RUNTIME.run_job(job, context, 0, False, False)
    assert RUNTIME.job_complete(job, output)
    result_path = output / "job_result.json"
    result = json.loads(result_path.read_text())
    result["objective_config"]["vehicle_fixed_cost_usd"] = 0.0
    result_path.write_text(json.dumps(result))
    assert not RUNTIME.job_complete(job, output)
    with pytest.raises(RuntimeError, match="refusing fresh training"):
        RUNTIME.run_job(job, context, 0, False, False)


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
