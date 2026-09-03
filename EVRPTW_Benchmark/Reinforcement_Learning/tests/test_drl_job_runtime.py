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
    }


def _artifact_command(output: Path, exit_code: int = 0):
    source = (
        "from pathlib import Path; "
        f"p=Path({str(output)!r}); p.mkdir(parents=True, exist_ok=True); "
        "(p/'checkpoint_selected.pt').write_bytes(b'x'); "
        "(p/'validation_summary.json').write_text('{}'); "
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


def test_full_gate_requires_only_this_servers_commit_scoped_pilots(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    context["output"].mkdir(parents=True)
    pilot = _job("pilot__G__am_evrptw__Cus100__seed1234")
    pilot["kind"] = "pilot"
    pilot["stage"] = "runtime_memory"
    output = RUNTIME.output_dir(pilot, context)
    try:
        RUNTIME.require_server_local_pilots(context, [pilot])
    except RuntimeError as error:
        assert pilot["job_id"] in str(error)
    else:
        raise AssertionError("missing local pilot was accepted")

    output.mkdir(parents=True)
    (output / "checkpoint_selected.pt").write_bytes(b"checkpoint")
    (output / "validation_summary.json").write_text("{}")
    (output / "job_result.json").write_text(json.dumps({"status": "failed"}))
    try:
        RUNTIME.require_server_local_pilots(context, [pilot])
    except RuntimeError:
        pass
    else:
        raise AssertionError("failed local pilot was accepted")

    (output / "job_result.json").write_text(json.dumps({"status": "passed"}))
    RUNTIME.require_server_local_pilots(context, [pilot])


def test_full_gate_rejects_a_manifest_without_local_pilots(tmp_path: Path) -> None:
    context = _context(tmp_path)
    try:
        RUNTIME.require_server_local_pilots(context, [])
    except RuntimeError as error:
        assert "no pilot jobs" in str(error)
    else:
        raise AssertionError("manifest without local pilots was accepted")


def test_training_command_passes_frozen_rollout_budget_to_all_trainers(tmp_path: Path) -> None:
    context = _context(tmp_path)
    common = {
        "train_index": "train.parquet",
        "validation_index": "val.parquet",
        "training_epochs": 25,
        "training_rollout_steps": 140,
        "physical_batch_size": 4,
        "effective_batch_size": 4,
        "validation_views": 500,
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
        assert "--data-passes" not in command


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
