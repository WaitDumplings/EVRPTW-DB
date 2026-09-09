"""Exercise the actual Bash supervisor against launchers in a temporary git repo."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/2080ti_4_1/rrnco_then_full.sh"


def _git(repo, *arguments):
    return subprocess.run(["git", "-C", str(repo), *arguments], check=True, capture_output=True, text=True)


def _setup(tmp_path, *, busy=False, full_exit=0):
    repo = tmp_path / "repo with spaces"
    script_dir = repo / "EVRPTW_Benchmark/Reinforcement_Learning/scripts/2080ti_4_1"
    rrnco_dir = repo / "EVRPTW_Benchmark/Reinforcement_Learning/RRNCO_EVRPTW"
    script_dir.mkdir(parents=True)
    rrnco_dir.mkdir(parents=True)
    shutil.copyfile(SOURCE, script_dir / "rrnco_then_full.sh")
    state = tmp_path / "state"
    state.mkdir()
    # One append-only write per event permits concurrent workers without a lock.
    event_code = '''
def event(kind, **values):
    line = (json.dumps({"kind": kind, **values}) + "\\n").encode()
    fd = os.open(Path(os.environ["PIPELINE_TEST_STATE"]) / "events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)
'''
    (script_dir / "full.sh").write_text('''#!/usr/bin/env bash
set -euo pipefail
"$PYTHON_BIN" - "$@" <<'PYCODE'
import json, os, sys
from pathlib import Path
''' + event_code + '''
args = sys.argv[1:]
event("benchmark_start", slots=args[args.index("--slots") + 1],
      mapping=args[args.index("--slot-gpu-map") + 1],
      launcher=args[args.index("--launcher-id") + 1])
PYCODE
''')
    (rrnco_dir / "run_optimized_long_training.sh").write_text('''#!/usr/bin/env bash
set -euo pipefail
"$PYTHON_BIN" - <<'PYCODE'
import json, os, sys, time
from pathlib import Path
''' + event_code + '''
gpu, mode = os.environ["GPU"], os.environ["GRAPH_MODE"]
event("rrnco_start", gpu=gpu, mode=mode)
release = Path(os.environ["PIPELINE_TEST_STATE"]) / ("release_" + mode)
deadline = time.monotonic() + 20
while not release.exists():
    if time.monotonic() > deadline:
        sys.exit(99)
    time.sleep(0.01)
event("rrnco_end", gpu=gpu, mode=mode)
sys.exit(int(os.environ.get("PIPELINE_TEST_FULL_EXIT", "0")) if mode == "full" else 0)
PYCODE
''')
    (repo / "tracked_source.txt").write_text("original source\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "pipeline-test@example.invalid")
    _git(repo, "config", "user.name", "Pipeline Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    bin_dir = tmp_path / "stub-bin"
    bin_dir.mkdir()
    nvidia = bin_dir / "nvidia-smi"
    nvidia.write_text('''#!/usr/bin/env bash
set -euo pipefail
case "$1" in
  --query-gpu=name) printf '%s\\n' 'NVIDIA GeForce RTX 2080 Ti' 'NVIDIA GeForce RTX 2080 Ti' 'NVIDIA GeForce RTX 2080 Ti' 'NVIDIA GeForce RTX 2080 Ti' ;;
  --query-compute-apps=pid) [[ "${PIPELINE_TEST_BUSY:-0}" == 0 ]] || printf '%s\\n' 12345 ;;
  *) exit 91 ;;
esac
''')
    nvidia.chmod(0o755)
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
           "PYTHON_BIN": sys.executable, "RUN_TAG": "mock_pipeline",
           "PIPELINE_LOG_ROOT": str(state / "logs"), "RRNCO_OUTPUT_ROOT": str(state / "rrnco"),
           "PIPELINE_TEST_STATE": str(state), "PIPELINE_TEST_BUSY": "1" if busy else "0",
           "PIPELINE_TEST_FULL_EXIT": str(full_exit)}
    return repo, script_dir / "rrnco_then_full.sh", state, env


def _events(state):
    path = state / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _wait(state, predicate, process):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        events = _events(state)
        if predicate(events):
            return events
        if process.poll() is not None:
            pytest.fail(f"supervisor exited early: {process.returncode}; events={events}; output={process.stdout.read()}")
        time.sleep(0.01)
    pytest.fail(f"timed out; events={_events(state)}")


def _release(state, mode):
    (state / ("release_" + mode)).touch()


def _cleanup(state, process):
    for mode in ("full", "node_only"):
        _release(state, mode)
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.communicate(timeout=5)


@pytest.mark.parametrize("full_exit", [0, 7])
def test_each_rrnco_gpu_hands_off_independently_after_completion(tmp_path, full_exit):
    _, script, state, env = _setup(tmp_path, full_exit=full_exit)
    process = subprocess.Popen(["bash", str(script)], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        events = _wait(state, lambda rows: sum(row["kind"] == "rrnco_start" for row in rows) == 2, process)
        assert events[0] == {"kind": "benchmark_start", "slots": "0,3", "mapping": "0:0,3:3", "launcher": "mock_pipeline_main"}
        assert {row["gpu"] for row in events if row["kind"] == "rrnco_start"} == {"1", "2"}
        assert [row["slots"] for row in events if row["kind"] == "benchmark_start"] == ["0,3"]
        _release(state, "node_only")
        events = _wait(state, lambda rows: any(row.get("slots") == "2" for row in rows), process)
        assert not any(row.get("slots") == "1" for row in events)
        assert not any(row["kind"] == "rrnco_end" and row["gpu"] == "1" for row in events)
        _release(state, "full")
        output, _ = process.communicate(timeout=10)
        assert process.returncode == (0 if full_exit == 0 else 1), output
        events = _events(state)
        for gpu in ("1", "2"):
            ended = next(i for i, row in enumerate(events) if row["kind"] == "rrnco_end" and row["gpu"] == gpu)
            started = next(i for i, row in enumerate(events) if row.get("slots") == gpu)
            assert ended < started
            assert events[started]["mapping"] == f"{gpu}:{gpu}"
        assert (state / "logs/rrnco_full.exit_code").read_text().strip() == str(full_exit)
        assert (state / "logs/scheduling_result.txt").read_text().strip() == f"full={full_exit} node_only=0"
    finally:
        _cleanup(state, process)


def test_busy_gpu_refuses_before_any_launcher(tmp_path):
    _, script, state, env = _setup(tmp_path, busy=True)
    result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, timeout=10)
    assert result.returncode == 3
    assert "12345" in result.stderr
    assert not _events(state)
    assert not (state / "logs").exists()


@pytest.mark.parametrize("change", ["dirty", "new_commit"])
def test_deferred_benchmarks_reject_changed_source(tmp_path, change):
    repo, script, state, env = _setup(tmp_path)
    process = subprocess.Popen(["bash", str(script)], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        _wait(state, lambda rows: sum(row["kind"] == "rrnco_start" for row in rows) == 2, process)
        (repo / "tracked_source.txt").write_text("changed source\n")
        if change == "new_commit":
            _git(repo, "add", "tracked_source.txt")
            _git(repo, "commit", "-qm", "changed")
            assert not _git(repo, "status", "--porcelain").stdout
        _release(state, "full")
        _release(state, "node_only")
        output, _ = process.communicate(timeout=10)
        assert process.returncode == 1, output
        assert [row["slots"] for row in _events(state) if row["kind"] == "benchmark_start"] == ["0,3"]
        for gpu in ("1", "2"):
            assert "original clean source" in (state / f"logs/benchmark_{gpu}_blocked.txt").read_text()
        assert (state / "logs/scheduling_result.txt").read_text().strip() == "full=4 node_only=4"
    finally:
        _cleanup(state, process)
