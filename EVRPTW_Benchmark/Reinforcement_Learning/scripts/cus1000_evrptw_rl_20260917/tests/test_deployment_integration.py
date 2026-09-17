"""CPU-only integration checks for the Cus1000 four-GPU deployment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus1000_evrptw_rl_20260917 import common, launch, prepare
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913 import launch as old_launch


def gpu(index, used=10):
    return {"index": index, "uuid": f"GPU-test-{index}", "name": "NVIDIA GeForce RTX 2080 Ti",
            "memory.total": 11264, "memory.used": used}


def process(index, executable="/env/bin/python", memory=156):
    return {"gpu_uuid": f"GPU-test-{index}", "pid": 900001 + index,
            "used_memory_mib": memory, "executable": executable}


@pytest.mark.parametrize("batch,accumulation", [(None, None), (2, 3)])
def test_generated_command_is_accepted_by_actual_trainer_parser(tmp_path, batch, accumulation):
    from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.distributed_train import parse_args
    from EVRPTW_Benchmark.Reinforcement_Learning.common.training_protocol import require_validation_rollout_steps

    config = common.load_config(batch=batch, accumulation=accumulation)
    command = launch.build_command(config, tmp_path / "road", tmp_path / "run",
                                   {"path": str(tmp_path / "stream.parquet"), "contract": {"sha256": "abc"}})
    assert command.count("--nproc_per_node=4") == 1
    assert command.count(config["train_module"]) == 1
    args = parse_args(command[command.index(config["train_module"]) + 1:])
    assert args.expected_world_size == config["world_size"] == 4
    assert config["gpus"] == [0, 1, 2, 3]
    assert args.scale == "Cus1000"
    assert args.graph_aggregation == "mean"
    assert args.samples_per_instance == args.validation_candidates == 30
    assert args.training_representation == "G"
    assert args.physical_batch_size == config["physical_batch_size"]
    assert args.effective_batch_size == (args.physical_batch_size * 4 * config["gradient_accumulation_steps"])
    assert args.customer_exposure_budget == args.effective_batch_size * args.training_epochs * 1000
    assert require_validation_rollout_steps(args) == (3 * args.training_rollout_steps + 1) // 2
    assert args.validation_limit == 500
    assert args.instance_cache_size == config["instance_cache_size"]
    assert not args.resume


@pytest.mark.parametrize("occupied_index", range(4))
def test_any_selected_compute_process_blocks_without_touching_processes(occupied_index, monkeypatch):
    def cannot_signal(*args, **kwargs):
        raise AssertionError("GPU validation must never signal an existing process")
    monkeypatch.setattr(os, "kill", cannot_signal)
    inventory = [gpu(i) for i in range(4)]
    with pytest.raises(RuntimeError, match=f"GPU {occupied_index} has compute PID"):
        launch.validate_gpus(common.load_config(), inventory, [process(occupied_index)])


def test_only_exact_gpu0_desktop_process_is_preserved():
    config = common.load_config()
    desktop = process(0, executable=launch.DESKTOP_EXECUTABLE)
    inventory = [gpu(0, 397), gpu(1), gpu(2), gpu(3)]
    selected = launch.validate_gpus(config, inventory, [desktop])
    assert [row["index"] for row in selected] == [0, 1, 2, 3]
    assert selected[0]["preserved_desktop_processes"] == [desktop]
    assert selected[0]["free_memory_mib"] == 10867
    for unsafe in [process(1, executable=launch.DESKTOP_EXECUTABLE),
                   process(0, executable="/tmp/gnome-remote-desktop-daemon"),
                   process(0, executable=None),
                   process(0, executable=launch.DESKTOP_EXECUTABLE, memory=None)]:
        with pytest.raises(RuntimeError, match="compute PID"):
            launch.validate_gpus(config, inventory, [unsafe])


def test_missing_duplicate_and_reordered_physical_cards_are_rejected():
    for selection in ["0,1,2", "0,1,2,2", "1,2,3,4", "3,2,1,0", "0,1,2,-1"]:
        with pytest.raises(ValueError):
            common.load_config(gpus=selection)
    config = common.load_config()
    for missing in range(4):
        with pytest.raises(RuntimeError, match=f"GPU {missing} is not"):
            launch.validate_gpus(config, [gpu(i) for i in range(4) if i != missing], [])
    with pytest.raises(RuntimeError, match="GPU 2 is not"):
        launch.validate_gpus(config, [gpu(0), gpu(1), gpu(2), gpu(2), gpu(3)], [])


def test_old_and_new_default_gpu_locks_conflict_and_release_partial_acquisitions(tmp_path, monkeypatch):
    # Both launchers must choose the same DEFAULT namespace; only its temp parent is isolated.
    monkeypatch.setattr(launch.tempfile, "gettempdir", lambda: str(tmp_path))
    old_fds = old_launch.lock_gpus([gpu(2), gpu(3)])
    try:
        with pytest.raises(RuntimeError, match="reserved"):
            launch.lock_gpus([gpu(i) for i in range(4)])
        free_fds = launch.lock_gpus([gpu(0), gpu(1)])
        for fd in free_fds:
            os.close(fd)
    finally:
        for fd in old_fds:
            os.close(fd)
    new_fds = launch.lock_gpus([gpu(i) for i in range(4)])
    try:
        with pytest.raises(RuntimeError, match="reserved"):
            old_launch.lock_gpus([gpu(0), gpu(1)])
    finally:
        for fd in new_fds:
            os.close(fd)


def test_cus1000_stream_exact_budget_reuse_and_integrity(tmp_path):
    from EVRPTW_Benchmark.Reinforcement_Learning.common.training_stream import read_stream_view_ids

    index_path = tmp_path / "road" / common.TRAIN_INDEX
    index_path.parent.mkdir(parents=True)
    rows = [{"view_id": f"cus1000-v{i}", "family_id": f"f{i}", "split_id": "train", "track_id": "train",
             "city_slug": "fixture", "scale_id": "Cus1000", "customer_count": 1000, "day_type": "weekday"}
            for i in range(7)]
    # Ensure another scale present in the source cannot leak into this stream.
    rows.append({**rows[0], "view_id": "cus500-only", "scale_id": "Cus500", "customer_count": 500})
    pd.DataFrame(rows).to_parquet(index_path)
    config = common.load_config(batch=2, accumulation=1)
    config.update(training_epochs=3, minimum_training_epochs=3, sample_count=24, customer_exposure_budget=24000)
    assert config["sample_count"] == config["effective_batch_size"] * config["training_epochs"]
    audit = {"train_index_sha256": common.digest(index_path)}
    first = prepare.prepare_stream(tmp_path / "road", tmp_path / "artifacts", config, audit=audit)
    path = Path(first["path"])
    original_bytes = path.read_bytes()
    original_mtime = path.stat().st_mtime_ns
    second = prepare.prepare_stream(tmp_path / "road", tmp_path / "artifacts", config, audit=audit)
    assert first["contract"] == second["contract"]
    assert path.read_bytes() == original_bytes and path.stat().st_mtime_ns == original_mtime
    ids = read_stream_view_ids(path)
    assert len(ids) == 24 and len(set(ids[:7])) == len(set(ids[7:14])) == 7
    assert "cus500-only" not in ids
    assert first["contract"]["scale"] == "Cus1000"
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    assert json.loads(sidecar.read_text())["customer_exposures"] == 24000
    with pytest.raises(ValueError, match="does not match"):
        prepare.prepare_stream(tmp_path / "road", tmp_path / "artifacts", config,
                               audit={"train_index_sha256": "changed-source"})
    frame = pd.read_parquet(path)
    frame.loc[0, "view_id"] = "corrupted"
    frame.to_parquet(path)
    with pytest.raises(ValueError):
        prepare.prepare_stream(tmp_path / "road", tmp_path / "artifacts", config, audit=audit)
    path.write_bytes(original_bytes)
    sidecar.unlink()
    with pytest.raises(RuntimeError, match="Incomplete stream artifact"):
        prepare.prepare_stream(tmp_path / "road", tmp_path / "artifacts", config, audit=audit)


def test_server_shell_status_works_outside_repo_without_cuda(tmp_path):
    output = tmp_path / "unused-output"
    env = {**os.environ, "CUS1000_PYTHON": sys.executable, "CUS1000_OUTPUT_ROOT": str(output),
           "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.check_output(["bash", str(common.HERE / "2080ti_4_2/full.sh"), "--mode", "status"],
                                     cwd=tmp_path, env=env, text=True)
    state = json.loads(result)
    assert state["status"] == "not_started"
    assert Path(state["status_file"]) == output / "launchers/2080ti_4_2/evrptw_rl/status.json"
    assert not output.exists()
