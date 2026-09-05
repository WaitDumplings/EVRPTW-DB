from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


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
