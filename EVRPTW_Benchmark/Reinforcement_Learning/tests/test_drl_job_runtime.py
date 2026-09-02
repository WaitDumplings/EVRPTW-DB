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


def test_full_gate_rejects_missing_or_failed_report(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context["output"].mkdir(parents=True)
    try:
        RUNTIME.require_pilot_gate(context)
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing pilot gate was accepted")
    (context["output"] / "pilot_gate_report.json").write_text(json.dumps({"passed": False}))
    try:
        RUNTIME.require_pilot_gate(context)
    except RuntimeError:
        pass
    else:
        raise AssertionError("failed pilot gate was accepted")
