from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_am_multigpu_20260912.common import (
    CONFIG, TRAIN_INDEX, VAL_INDEX, load_config, parse_gpus, resolve_road_root,
)
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_am_multigpu_20260912.launch import (
    DESKTOP_EXECUTABLE, build_command, lock_output, validate_gpus, validate_resume,
)


def gpu(index, used=10):
    return {"index": index, "uuid": f"GPU-{index}", "name": "NVIDIA GeForce RTX 2080 Ti",
            "memory.total": 11264, "memory.used": used}


def process(index=0, executable=DESKTOP_EXECUTABLE, memory=156):
    return {"gpu_uuid": f"GPU-{index}", "pid": 123, "used_memory_mib": memory, "executable": executable}


@pytest.mark.parametrize("value", ["", "0", "0,0", "-1,2", "0,,1", "0,banana"])
def test_gpu_list_rejects_invalid(value):
    with pytest.raises(ValueError):
        parse_gpus(value)


def test_two_and_three_card_global_budgets():
    three = load_config(gpus="0,1,2", batch=8, accumulation=2)
    two = load_config(gpus="1,2", batch=8, accumulation=2)
    assert three["effective_batch_size"] == 48
    assert two["effective_batch_size"] == 32
    assert three["sample_count"] == 480000
    assert three["customer_exposure_budget"] == 240000000


def test_desktop_exact_executable_on_gpu_zero_is_preserved():
    cfg = load_config()
    result = validate_gpus(cfg, [gpu(0, 397), gpu(1), gpu(2)], [process()])
    assert result[0]["preserved_desktop_processes"] == [process()]
    assert result[0]["free_memory_mib"] == 10867


@pytest.mark.parametrize("bad", [process(executable="/tmp/gnome-remote-desktop-daemon"),
                                  process(executable=None), process(executable=DESKTOP_EXECUTABLE + " (deleted)"),
                                  process(memory=1024), process(memory=None), process(index=1)])
def test_compute_occupancy_fails_closed(bad):
    with pytest.raises(RuntimeError, match="compute PID"):
        validate_gpus(load_config(), [gpu(0), gpu(1), gpu(2)], [bad])


def test_total_desktop_memory_limit_is_enforced():
    with pytest.raises(RuntimeError, match="already uses"):
        validate_gpus(load_config(), [gpu(0, 1024), gpu(1), gpu(2)], [])


def test_desktop_needs_model_headroom():
    with pytest.raises(RuntimeError, match="insufficient free memory"):
        validate_gpus(load_config(), [gpu(0, 900), gpu(1), gpu(2)], [process()])


def test_unused_gpu_does_not_block_training():
    cfg = load_config(gpus="1,2")
    result = validate_gpus(cfg, [gpu(0, 11000), gpu(1), gpu(2)], [process(executable="/bin/python")])
    assert [row["index"] for row in result] == [1, 2]


def test_command_uses_one_torchrun_model_and_global_batch(tmp_path):
    cfg = load_config(batch=8, accumulation=2)
    command = build_command(cfg, tmp_path / "road", tmp_path / "run", {"path": "/stream.parquet", "contract": {"sha256": "abc"}}, python="/env/python")
    assert command[:4] == ["/env/python", "-m", "torch.distributed.run", "--standalone"]
    assert "--nproc_per_node=3" in command
    assert command[command.index("--physical-batch-size") + 1] == "8"
    assert command[command.index("--effective-batch-size") + 1] == "48"
    assert command[command.index("--training-rollout-steps") + 1] == "1700"
    assert command[command.index("--validation-rollout-steps") + 1] == "2550"
    assert command[command.index("--samples-per-instance") + 1] == "30"
    assert command[command.index("--validation-candidates") + 1] == "30"
    assert command[command.index("--expected-world-size") + 1] == "3"
    assert "--resume" not in command
    assert command[command.index("--training-representation") + 1] == "G"


def test_resume_requires_explicit_flag(tmp_path):
    command = build_command(load_config(), tmp_path, tmp_path, {"path": "/stream", "contract": {"sha256": "abc"}}, resume=True)
    assert command[-1] == "--resume"


def test_output_lock_prevents_duplicate_launcher(tmp_path):
    fd, directory = lock_output(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            lock_output(tmp_path)
    finally:
        os.close(fd)
    second, _ = lock_output(tmp_path)
    os.close(second)


def test_explicit_bad_road_root_is_not_silently_ignored(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_road_root(tmp_path / "missing", repo=tmp_path)


def test_road_root_override(tmp_path):
    for relative in (TRAIN_INDEX, VAL_INDEX):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"index")
    assert resolve_road_root(tmp_path) == tmp_path


@pytest.mark.parametrize("changed", ["config", "data", "source"])
def test_resume_rejects_changed_experiment(changed):
    previous = {"preflight": {"config": {"batch": 4}, "data": {"sha": "old"}}, "source": {"source_sha256": "old"}}
    current = copy.deepcopy(previous)
    if changed == "source":
        current["source"]["source_sha256"] = "new"
    else:
        current["preflight"][changed] = {"changed": True}
    with pytest.raises(ValueError):
        validate_resume(previous, current)


def test_config_rejects_wrong_validation_cap(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["validation_rollout_steps"] = 1200
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="cap"):
        load_config(path)


def test_stream_preparation_reuses_exact_global_budget_and_rejects_corrupt_content(tmp_path):
    import pandas as pd
    from EVRPTW_Benchmark.Reinforcement_Learning.common.training_stream import read_stream_view_ids
    from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_am_multigpu_20260912.common import digest
    from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_am_multigpu_20260912.prepare import prepare_stream

    index_path = tmp_path / "road" / TRAIN_INDEX
    index_path.parent.mkdir(parents=True)
    pd.DataFrame([{"view_id": f"v{i}", "family_id": f"f{i}", "split_id": "train", "track_id": "train",
                   "city_slug": "a", "scale_id": "cus500", "customer_count": 500, "day_type": "weekday"}
                  for i in range(10)]).to_parquet(index_path)
    cfg = load_config(batch=4)
    cfg["sample_count"] = 24
    audit = {"train_index_sha256": digest(index_path)}
    first = prepare_stream(tmp_path / "road", tmp_path / "artifacts", cfg, audit=audit)
    second = prepare_stream(tmp_path / "road", tmp_path / "artifacts", cfg, audit=audit)
    assert first["contract"] == second["contract"]
    ids = read_stream_view_ids(first["path"])
    assert len(ids) == 24
    assert len(set(ids[:10])) == 10
    assert first["contract"]["scale"] == "Cus500"
    path = Path(first["path"])
    frame = pd.read_parquet(path)
    frame.loc[0, "view_id"] = "corrupted"
    frame.to_parquet(path)
    with pytest.raises(ValueError):
        prepare_stream(tmp_path / "road", tmp_path / "artifacts", cfg, audit=audit)
