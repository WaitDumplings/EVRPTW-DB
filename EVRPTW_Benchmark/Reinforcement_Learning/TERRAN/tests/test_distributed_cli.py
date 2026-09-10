"""Exercise actual torchrun worker startup with inexpensive CPU collectives."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import textwrap

import pytest
import torch.distributed as distributed
import yaml

from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import stable_cli
from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.tests.test_stable_cli import _arguments


pytestmark = pytest.mark.skipif(
    not distributed.is_available() or not distributed.is_gloo_available(),
    reason="CPU Gloo support is required for torchrun integration",
)


_WORKER = r'''
import fcntl
import json
import os
from pathlib import Path
import sys
import types

import torch
import torch.distributed as dist
import yaml

from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN import stable_cli

config_path, mode = Path(sys.argv[1]), sys.argv[2]
config = yaml.safe_load(config_path.read_text())
output = Path(config["output_dir"])
rank = int(os.environ["RANK"])
events = []
write_json = stable_cli._json_write

def observed_write(path, value):
    if path.name == "run_state.json":
        events.append({"pid": os.getpid(), "rank": rank, "status": value["status"]})
    return write_json(path, value)

stable_cli._json_write = observed_write

def tiny_training(cfg, seed, device):
    assert mode == "success", "duplicate or mismatched run reached training"
    assert seed == 1234 and device == "cpu"
    assert dist.is_initialized() and dist.get_world_size() == 4
    state = json.loads((output / "run_state.json").read_text())
    assert state["status"] == "running"
    assert state["pid"] == state["worker_pids"][0]
    assert state["worker_pids"][rank] == os.getpid()
    assert int((output / "train.pid").read_text()) == state["pid"]
    with (output / "run.lock").open("a") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_held = True
        else:
            raise AssertionError("run lock was not held throughout training")
    total = torch.tensor(rank + 1, dtype=torch.int64)
    dist.all_reduce(total)
    assert total.item() == 10
    result = {"rank": rank, "pid": os.getpid(), "all_reduce_sum": total.item(),
              "lock_held": lock_held, "state_owner": state["pid"]}
    (output / f"probe_{rank}.json").write_text(json.dumps(result))
    dist.barrier()
    return output / "tiny-checkpoint.pt"

# Importing the full trainer is unnecessary for testing the CLI's own process
# group lifecycle, lock ownership and rank-zero file writes.
trainer_module = types.ModuleType("EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.trainer")
trainer_module.train_from_config = tiny_training
sys.modules[trainer_module.__name__] = trainer_module

error = None
try:
    stable_cli.run(config_path, device="cpu", gpu=None)
except Exception as exc:
    error = f"{type(exc).__name__}: {exc}"
    if mode == "success":
        raise
    if mode == "existing":
        assert "distributed run setup failed" in error and "already started" in error, error
    elif mode == "mismatch":
        assert "requires 3 workers, got 4" in error, error
    else:
        raise
else:
    assert mode == "success", "invalid run was accepted"

assert not dist.is_initialized(), "CLI leaked its process group"
(output / f"worker_{mode}_{rank}.json").write_text(json.dumps(
    {"rank": rank, "pid": os.getpid(), "events": events, "error": error}))
'''


def _prepared_config(tmp_path):
    args = _arguments(tmp_path)
    args.world_size = 4
    args.physical_batch_size = 32
    return stable_cli.prepare(args)


def _run_workers(config_path: Path, mode: str, tmp_path: Path):
    worker = tmp_path / "cli_worker.py"
    worker.write_text(textwrap.dedent(_WORKER))
    command, environment = stable_cli.launch_command(
        config_path, device="cpu", gpu=None, gpus=None,
    )
    # The real launcher starts four workers; replace only its worker entry point
    # so the CLI setup is exercised without a dataset/model training workload.
    command = command[:command.index("--module")] + [str(worker), str(config_path), mode]
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["TERRAN_DISTRIBUTED_TIMEOUT_S"] = "20"
    environment["GLOO_SOCKET_IFNAME"] = "lo"
    environment["PYTHONPATH"] = str(stable_cli.REPO_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    process = subprocess.Popen(command, cwd=stable_cli.REPO_ROOT, env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, start_new_session=True)
    try:
        stdout, _ = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        stdout, _ = process.communicate(timeout=10)
        pytest.fail(f"torchrun did not finish within 60 seconds:\n{stdout}")
    assert process.returncode == 0, stdout


def test_four_cpu_workers_share_one_run_owner_and_reject_duplicate_run(tmp_path):
    path = _prepared_config(tmp_path)
    _run_workers(path, "success", tmp_path)
    state_path = path.parent / "run_state.json"
    original_state = state_path.read_bytes()
    state = json.loads(original_state)
    assert state["status"] == "completed"
    assert state["world_size"] == 4
    assert len(set(state["worker_pids"])) == 4
    assert state["pid"] == state["worker_pids"][0]
    assert int((path.parent / "train.pid").read_text()) == state["pid"]
    assert state["result"] == str(path.parent / "tiny-checkpoint.pt")
    events = []
    for rank in range(4):
        probe = json.loads((path.parent / f"probe_{rank}.json").read_text())
        worker = json.loads((path.parent / f"worker_success_{rank}.json").read_text())
        assert probe["all_reduce_sum"] == 10 and probe["lock_held"]
        assert probe["pid"] == worker["pid"] == state["worker_pids"][rank]
        assert probe["state_owner"] == state["pid"]
        assert worker["error"] is None
        events.extend(worker["events"])
    assert events == [{"pid": state["pid"], "rank": 0, "status": status}
                      for status in ("running", "completed")]

    _run_workers(path, "existing", tmp_path)
    assert state_path.read_bytes() == original_state
    assert int((path.parent / "train.pid").read_text()) == state["pid"]
    for rank in range(4):
        worker = json.loads((path.parent / f"worker_existing_{rank}.json").read_text())
        assert "already started" in worker["error"]
        assert worker["events"] == []


def test_four_cpu_workers_reject_declared_world_mismatch_without_hanging(tmp_path):
    path = _prepared_config(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["training"]["distributed_world_size"] = 3
    path.write_text(yaml.safe_dump(cfg))
    # The declared world differs from the actual torchrun worker count.
    original = stable_cli.launch_command

    def four_workers(*args, **kwargs):
        command, environment = original(*args, **kwargs)
        command[command.index("--nproc_per_node=3")] = "--nproc_per_node=4"
        return command, environment

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(stable_cli, "launch_command", four_workers)
        _run_workers(path, "mismatch", tmp_path)
    assert not (path.parent / "run_state.json").exists()
    assert not (path.parent / "train.pid").exists()
    assert not (path.parent / "run.lock").exists()
    for rank in range(4):
        worker = json.loads((path.parent / f"worker_mismatch_{rank}.json").read_text())
        assert "requires 3 workers, got 4" in worker["error"]
        assert worker["events"] == []
