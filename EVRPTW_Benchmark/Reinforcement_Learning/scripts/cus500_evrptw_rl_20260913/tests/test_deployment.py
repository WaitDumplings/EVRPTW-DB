from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913.common import (
    CONFIG, TRAIN_INDEX, VAL_INDEX, load_config, parse_gpus, resolve_road_root,
)
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913.launch import (
    DESKTOP_EXECUTABLE, build_command, lock_output, lock_gpus, read_status, last_jsonl, validate_gpus, validate_resume,
)


def gpu(index, used=10):
    return {"index": index, "uuid": f"GPU-{index}", "name": "NVIDIA GeForce RTX 2080 Ti",
            "memory.total": 11264, "memory.used": used}


def process(index=0, executable=DESKTOP_EXECUTABLE, memory=156):
    return {"gpu_uuid": f"GPU-{index}", "pid": 123, "used_memory_mib": memory, "executable": executable}


@pytest.mark.parametrize("value", ["", "0", "0,0", "-1,2", "0,,1", "0,banana", "0,1,2", "1,2", "1,0"])
def test_gpu_list_rejects_invalid(value):
    with pytest.raises(ValueError):
        parse_gpus(value)


def test_two_card_global_budgets():
    config = load_config(gpus="0,1", batch=8, accumulation=2)
    assert config["effective_batch_size"] == 32
    assert config["sample_count"] == 320000
    assert config["customer_exposure_budget"] == 160000000


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
    cfg = load_config()
    result = validate_gpus(cfg, [gpu(0), gpu(1), gpu(2, 11000)], [process(index=2, executable="/bin/python")])
    assert [row["index"] for row in result] == [0, 1]


def test_command_uses_one_torchrun_model_and_global_batch(tmp_path):
    cfg = load_config(batch=8, accumulation=2)
    command = build_command(cfg, tmp_path / "road", tmp_path / "run", {"path": "/stream.parquet", "contract": {"sha256": "abc"}}, python="/env/python")
    assert command[:4] == ["/env/python", "-m", "torch.distributed.run", "--standalone"]
    assert "--nproc_per_node=2" in command
    assert command[command.index("--physical-batch-size") + 1] == "8"
    assert command[command.index("--effective-batch-size") + 1] == "32"
    assert command[command.index("--training-rollout-steps") + 1] == "1700"
    assert command[command.index("--validation-rollout-steps") + 1] == "2550"
    assert command[command.index("--samples-per-instance") + 1] == "30"
    assert command[command.index("--validation-candidates") + 1] == "30"
    assert command[command.index("--expected-world-size") + 1] == "2"
    assert "--resume" not in command
    assert command[command.index("--training-representation") + 1] == "G"


def test_resume_requires_explicit_flag(tmp_path):
    command = build_command(load_config(), tmp_path, tmp_path, {"path": "/stream", "contract": {"sha256": "abc"}}, resume=True)
    assert command[-1] == "--resume"


def test_output_lock_prevents_duplicate_launcher(tmp_path):
    fd, directory = lock_output(tmp_path, load_config())
    try:
        with pytest.raises(RuntimeError, match="already running"):
            lock_output(tmp_path, load_config())
    finally:
        os.close(fd)
    second, _ = lock_output(tmp_path, load_config())
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
    from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913.common import digest
    from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913.prepare import prepare_stream

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


def test_native_model_recipe_and_cache(tmp_path):
    cfg = load_config()
    command = build_command(cfg, tmp_path, tmp_path, {"path": "/stream", "contract": {"sha256": "abc"}})
    assert "EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.distributed_train" in command
    for key, expected in (("--instance-cache-size", "256"), ("--learning-rate", "0.001"),
                          ("--activation-checkpoint-stride", "1"), ("--ema-warmup-steps", "1000"),
                          ("--ema-decay", "0.9"), ("--baseline-eval-interval", "100"),
                          ("--baseline-eval-size", "64"), ("--max-grad-norm", "2.0")):
        assert command[command.index(key) + 1] == expected
    assert command[command.index("--method-auxiliary-profile") + 1].endswith("evrptw_rl_station_auxiliary_v1.json")
    assert "--graph-mode" not in command
    assert "--soft-stage-end-epoch" not in command


def test_other_model_config_is_rejected(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["model"] = "drl_ts"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="evrptw_rl"):
        load_config(path)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_cache_rejects_invalid_config_value(tmp_path, value):
    config = json.loads(CONFIG.read_text())
    config["instance_cache_size"] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="nonnegative"):
        load_config(path)


def test_cache_can_be_disabled_or_reduced():
    assert load_config(cache_size=0)["instance_cache_size"] == 0
    assert load_config(cache_size=64)["instance_cache_size"] == 64


def test_per_gpu_lock_blocks_other_outputs_and_releases_partial_acquisitions(tmp_path):
    fds = lock_gpus([gpu(1), gpu(2)], lock_root=tmp_path)
    try:
        with pytest.raises(RuntimeError, match="reserved"):
            lock_gpus([gpu(0), gpu(1)], lock_root=tmp_path)
        free = lock_gpus([gpu(0)], lock_root=tmp_path)
        for fd in free:
            os.close(fd)
    finally:
        for fd in fds:
            os.close(fd)
    again = lock_gpus([gpu(0), gpu(1)], lock_root=tmp_path)
    for fd in again:
        os.close(fd)


def test_model_launcher_output_path_is_independent(tmp_path):
    fd, path = lock_output(tmp_path, load_config())
    try:
        assert path.parts[-2:] == ("local_after_am_gpu01", "evrptw_rl")
    finally:
        os.close(fd)


def test_status_without_launch_does_not_create_output(tmp_path):
    out = tmp_path / "unused"
    result = read_status(out, load_config())
    assert result["status"] == "not_started"
    assert not out.exists()


def test_status_reports_train_validation_best_and_ignores_unfinished_row(tmp_path):
    cfg = load_config()
    run = tmp_path / "runs" / cfg["run_id"]
    run.mkdir(parents=True)
    launcher = tmp_path / "launchers" / cfg["server"] / cfg["model"]
    launcher.mkdir(parents=True)
    (launcher / "status.json").write_text(json.dumps({"status": "completed", "output_dir": str(run)}))
    (run / "logical_epoch_history.jsonl").write_text('{"logical_epoch": 100}\n{"logical_epoch":')
    (run / "validation_history.jsonl").write_text('{"logical_epoch": 100, "mean_total_cost": 1000.0}\n')
    (run / "validation_summary.json").write_text('{"logical_epoch": 100, "successful_instances": 500}')
    result = read_status(tmp_path, cfg)
    assert result["latest_train"]["logical_epoch"] == 100
    assert result["latest_validation"]["mean_total_cost"] == 1000.0
    assert result["best_validation"]["successful_instances"] == 500


def test_cuda_unavailable_does_not_prevent_shell_status(tmp_path):
    import subprocess
    import sys
    script = CONFIG.parent / "full.sh"
    env = {**os.environ, "CUS500_PYTHON": sys.executable, "CUDA_VISIBLE_DEVICES": ""}
    out = subprocess.check_output(["bash", str(script), "--mode", "status", "--output-root", str(tmp_path)], env=env, text=True)
    assert json.loads(out)["status"] == "not_started"


@pytest.mark.parametrize("result_status,has_best,returncode,expected", [
    ("passed", True, 0, "completed"),
    ("early_stopped", True, 0, "completed"),
    ("passed", False, 0, "failed"),
    ("passed", True, 2, "failed"),
    ("pilot_partial", True, 0, "failed"),
])
def test_worker_requires_successful_formal_result_and_best(tmp_path, monkeypatch, result_status, has_best, returncode, expected):
    import sys
    import time
    from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913 import launch

    cfg = load_config()
    fd, launcher = lock_output(tmp_path, cfg)
    run = tmp_path / "run"
    run.mkdir()
    code = (
        "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); "
        f"(p/'training_result.json').write_text(json.dumps({{'status': {result_status!r}}})); "
        + ("(p/'best.ckpt').write_bytes(b'best'); " if has_best else "")
        + f"sys.exit({returncode})"
    )
    request = {"command": [sys.executable, "-c", code, str(run)], "environment": {},
               "source": {"source_sha256": "source"},
               "preflight": {"output_dir": str(run), "gpus": [gpu(0), gpu(1)], "config": cfg}}
    path = launcher / "launch_request.json"
    path.write_text(json.dumps(request))
    monkeypatch.setattr(launch, "source_snapshot", lambda: {"source_sha256": "source"})
    monkeypatch.setattr(launch, "gpu_inventory", lambda: [gpu(0), gpu(1)])
    monkeypatch.setattr(launch, "gpu_processes", lambda: [])
    sleep = time.sleep
    monkeypatch.setattr(launch.time, "sleep", lambda duration: sleep(min(duration, .01)))
    code = launch.worker(path, [fd])
    state = json.loads((launcher / "status.json").read_text())
    assert state["status"] == expected
    assert (code == 0) == (expected == "completed")
    # Worker closes its inherited reservation, allowing the next explicit launch.
    second, _ = lock_output(tmp_path, cfg)
    os.close(second)


def test_full_shell_fixes_model_and_gpus_after_user_args(tmp_path):
    import subprocess
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    fake_python.chmod(0o755)
    script = CONFIG.parent / "full.sh"
    env = {**os.environ, "CUS500_PYTHON": str(fake_python), "CUS500_GPUS": "2,3"}
    output = subprocess.check_output(["bash", str(script), "--config", "/final/config.json",
                                     "--gpus", "2,3", "--model", "rrnco", "--resume"], env=env, text=True)
    argv = json.loads(output)
    assert argv[-4:] == ["--model", "evrptw_rl", "--gpus", "0,1"]
    assert "/final/config.json" in argv
    assert "--resume" in argv


def test_cli_defaults_fixed_gpu_pair_despite_environment(tmp_path):
    import subprocess
    import sys
    env = {**os.environ, "CUS500_GPUS": "broken", "CUDA_VISIBLE_DEVICES": ""}
    output = subprocess.check_output([sys.executable, str(CONFIG.parent / "launch.py"),
                                     "--mode", "status", "--output-root", str(tmp_path)], env=env, text=True)
    assert json.loads(output)["model"] == "evrptw_rl"
