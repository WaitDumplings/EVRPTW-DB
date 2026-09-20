"""Portable command and corpus checks; these tests never launch a trainer."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

from script_curriculum import launch


def archived_records():
    payload = json.loads((launch.HERE / "checkpoints.json").read_text())
    return {(row["method"], row["source_domain"]): row for row in payload["checkpoints"]}


def test_archive_has_all_ten_portable_and_distinct_sources():
    records = archived_records()
    assert set(records) == {(method, domain) for method in launch.METHODS for domain in ("G", "E")}
    assert len({row["sha256"] for row in records.values()}) == 10
    for (method, domain), row in records.items():
        path = Path(row["relative_path"])
        assert not path.is_absolute() and ".." not in path.parts
        assert "Cus100" in path.parts and path.name.endswith(f"_{domain}_100.ckpt")
        assert len(row["sha256"]) == 64
        assert row["logical_epoch"] > 0
        if method == "drl_ts":
            assert row["logical_epoch"] > row["training_semantics"]["soft_stage_end_epoch"]


@pytest.mark.parametrize("method", launch.METHODS)
@pytest.mark.parametrize("domain", ("G", "E"))
def test_default_commands_parse_with_native_architecture_and_additional_budget(method, domain, monkeypatch, tmp_path):
    # A stream path is attached during prepare_data in real runs. Supplying one
    # here exercises the formal CLI path without reading or generating data.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    args = launch.parse_args([method, domain, "2"])
    job = launch.curriculum_job(args)
    job.update(training_stream_path=str(tmp_path / "stream.parquet"), training_stream_contract_sha256="f" * 64)
    checkpoint = tmp_path / "archived.ckpt"
    command = launch.build_command(job, tmp_path / "data", tmp_path / "new_run", checkpoint)
    monkeypatch.setattr(sys, "argv", command[2:])
    native = importlib.import_module(job["train_module"]).parse_args()
    source = archived_records()[(method, domain)]

    assert native.warm_start_checkpoint == checkpoint
    assert native.warm_start_objective_transition is True
    assert native.resume is False
    assert native.training_epochs == native.minimum_training_epochs == 2000
    assert native.early_stop_patience_validations == 0
    assert native.validation_every_epochs == 100
    assert native.validation_checkpoints == 20
    assert native.validation_limit == 500
    assert native.validation_candidates == 30
    assert native.validation_decode_type == "sampling"
    assert native.validation_seed == 910001234
    assert native.training_rollout_steps == 240
    assert native.validation_rollout_steps == 360
    assert native.training_representation == domain
    assert native.euclidean_manifest is None
    assert native.objective_distance_source == "running_time_path_distance_km"
    assert native.physical_batch_size == native.effective_batch_size == launch.DEFAULT_BATCHES[domain][method]
    assert native.customer_exposure_budget == 2000 * launch.DEFAULT_BATCHES[domain][method] * 100
    train_path = Path(native.stage2_dataset_path if method == "terran" else native.dataset_path)
    expected = "generation_plan/core/train/view_index.parquet" if domain == "G" else "train/view_index.parquet"
    assert train_path == tmp_path / "data" / expected
    assert job["source_kind"] == ("stage2_road" if domain == "G" else "terran_synthetic")

    if method == "terran":
        assert native.warm_start_epoch_mode == "reset"
        assert native.n_traj == 30
        from EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.trainer import load_config
        config = load_config(native.config)
        assert config["model"] == source["architecture"]
        assert config["training"]["learning_rate"] == source["training_semantics"]["learning_rate"]
    else:
        assert native.samples_per_instance == 30
        for field, value in source["architecture"].items():
            assert getattr(native, field) == value, field
        assert native.learning_rate == source["training_semantics"]["learning_rate"]
        if method == "rrnco":
            assert native.reinforce_baseline == "leave_one_out"
        if method == "drl_ts":
            assert native.soft_stage_end_epoch == 0


@pytest.mark.parametrize("argv", [
    ["am_evrptw", "X", "0"], ["am_evrptw", "G", "-1"],
    ["am_evrptw", "G", "0", "--validation-limit", "0"],
    ["am_evrptw", "G", "0", "--validation-limit", "501"],
])
def test_invalid_cli_rejected(argv):
    with pytest.raises(SystemExit):
        launch.parse_args(argv)


@pytest.mark.parametrize("extra", [
    ["--epochs", "0"], ["--epochs", "-1"], ["--epochs", "201"],
    ["--validation-every", "0"], ["--batch-size", "0"], ["--batch-size", "-1"],
])
def test_invalid_training_budget_rejected(extra):
    with pytest.raises((ValueError, SystemExit)):
        launch.curriculum_job(launch.parse_args(["am_evrptw", "G", "0", *extra]))


def test_explicit_missing_source_does_not_silently_fall_back(tmp_path):
    with pytest.raises(FileNotFoundError):
        launch.resolve_data("E", synthetic_root=tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        launch.resolve_data("G", road_root=tmp_path / "missing")


def test_checkpoint_custom_root_and_hash_rejection(tmp_path, monkeypatch):
    record = dict(method="am_evrptw", source_domain="G", relative_path="custom/source.ckpt")
    source = tmp_path / record["relative_path"]
    source.parent.mkdir()
    source.write_bytes(b"trusted local archive bytes")
    record["sha256"] = launch.digest(source)
    metadata_dir = tmp_path / "launcher"
    metadata_dir.mkdir()
    (metadata_dir / "checkpoints.json").write_text(json.dumps({"checkpoints": [record]}))
    monkeypatch.setattr(launch, "HERE", metadata_dir)
    assert launch.checkpoint_record("am_evrptw", "G", tmp_path)[0] == source
    source.write_bytes(b"replaced archive")
    with pytest.raises(ValueError, match="differs"):
        launch.checkpoint_record("am_evrptw", "G", tmp_path)


def make_index(count, split):
    return pd.DataFrame({
        "view_id": [f"{split}-{i:05}" for i in range(count)],
        "family_id": [f"{split}-family-{i:05}" for i in range(count)],
        "split_id": split,
        "track_id": "train" if split == "train" else "validation",
        "customer_count": 100, "scale_id": "Cus100", "city_slug": "test_city", "day_type": "weekday",
    })


def test_stream_freezes_only_train_ids_with_exact_additional_exposures(tmp_path):
    from EVRPTW_Benchmark.Reinforcement_Learning.common.training_stream import load_training_stream_contract, read_stream_view_ids
    job = launch.curriculum_job(launch.parse_args([
        "am_evrptw", "G", "0", "--epochs", "3", "--validation-every", "1", "--batch-size", "7",
    ]))
    train, val = make_index(50000, "train"), make_index(500, "val")
    for frame, name in ((train, job["train_index"]), (val, job["validation_index"])):
        target = tmp_path / "data" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
    audit = launch.prepare_data(tmp_path / "data", tmp_path / "new_run", job)
    contract = load_training_stream_contract(job["training_stream_path"])
    ids = read_stream_view_ids(job["training_stream_path"])
    assert audit["test_data_read"] is False
    assert audit["additional_instance_exposures"] == contract["sample_count"] == len(ids) == 21
    assert len(set(ids)) == 21
    assert set(ids).issubset(set(train.view_id))
    assert set(ids).isdisjoint(set(val.view_id))
    assert contract["sha256"] == job["training_stream_contract_sha256"]
    assert job["customer_exposure_budget"] == 2100

    # A validation family leaking into training must abort before any stream is
    # created in another output directory, even when view IDs remain disjoint.
    val.loc[0, "family_id"] = train.loc[0, "family_id"]
    val.to_parquet(tmp_path / "data" / job["validation_index"], index=False)
    with pytest.raises(ValueError, match="overlap"):
        launch.prepare_data(tmp_path / "data", tmp_path / "bad_run", job)
    assert not (tmp_path / "bad_run/artifacts/training_stream.parquet").exists()


@pytest.fixture
def supervisor_case(tmp_path, monkeypatch):
    import time
    a = launch.parse_args(["am_evrptw", "G", "0", "--epochs", "2", "--validation-every", "1",
                           "--run-dir", str(tmp_path / "run")])
    job = launch.curriculum_job(a)
    gpu = {"index": 0, "uuid": "pytest-" + tmp_path.name}
    monkeypatch.setattr(launch, "compute_busy_uuids", lambda: set())
    monkeypatch.setattr(launch, "prepare_data", lambda *unused: {"test_data_read": False})
    real_sleep = time.sleep
    monkeypatch.setattr(launch.time, "sleep", lambda unused: real_sleep(0.01))
    return a, job, gpu


@pytest.mark.parametrize("completion", ["valid", "malformed"])
def test_supervisor_reaps_child_and_writes_terminal_status(supervisor_case, tmp_path, monkeypatch, completion):
    a, job, gpu = supervisor_case
    code = """
import fcntl, json, os, pathlib, sys, time
run, completion, lock_path = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
with lock_path.open('a+') as lock:
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        (run / 'lock_observed').write_text('held')
    else:
        raise RuntimeError('trainer did not inherit a live GPU lock')
time.sleep(0.08)
(run / 'reward_diagnostics.jsonl').write_text(json.dumps({'logical_epoch': 2}) + chr(10))
(run / 'validation_history.jsonl').write_text(json.dumps({'logical_epoch': 2, 'instances': 500}) + chr(10))
(run / 'training_result.json').write_text(json.dumps({'completed_training_epochs': 2}) if completion == 'valid' else '{invalid json')
print('fake CPU trainer completed')
"""
    import os
    lock_path = Path(f"/tmp/evrptw-ablation-gpu-locks-{os.getuid()}/{gpu['uuid']}.lock")
    monkeypatch.setattr(launch, "build_command", lambda *unused: [sys.executable, "-c", code, str(a.run_dir), completion, str(lock_path)])
    result = launch.run_training(a, job, tmp_path / "data", tmp_path / "source.ckpt", {}, gpu)
    status = json.loads((a.run_dir / "status.json").read_text())
    request = json.loads((a.run_dir / "request.json").read_text())
    assert (a.run_dir / "lock_observed").read_text() == "held"
    assert (a.run_dir / "validation_summary.csv").is_file()
    assert "script_curriculum/checkpoints.json" in request["source_hashes"]
    assert request["source_hashes"]["script_curriculum/launch.py"] == launch.digest(launch.HERE / "launch.py")
    assert status["status"] == ("completed" if completion == "valid" else "failed")
    assert result == (0 if completion == "valid" else 1)
    assert status["logical_epoch"] == 2
    if completion == "malformed":
        assert "error" in status
    with pytest.raises(ProcessLookupError):
        os.kill(status["pid"], 0)


def test_monitor_failure_terminates_and_reaps_owned_trainer(supervisor_case, tmp_path, monkeypatch):
    import os
    a, job, gpu = supervisor_case
    monkeypatch.setattr(launch, "build_command", lambda *unused: [sys.executable, "-c", "import time; time.sleep(30)"])
    def fail_export(unused):
        raise OSError("simulated summary failure")
    monkeypatch.setattr(launch, "export_validation", fail_export)
    assert launch.run_training(a, job, tmp_path / "data", tmp_path / "source.ckpt", {}, gpu) == 1
    status = json.loads((a.run_dir / "status.json").read_text())
    assert status["status"] == "failed"
    assert "simulated summary failure" in status["error"]
    with pytest.raises(ProcessLookupError):
        os.kill(status["pid"], 0)


def test_reserved_gpu_aborts_before_output_or_subprocess(supervisor_case, tmp_path):
    import fcntl
    import os
    a, job, gpu = supervisor_case
    lock_path = Path(f"/tmp/evrptw-ablation-gpu-locks-{os.getuid()}/{gpu['uuid']}.lock")
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a+") as reserved:
        fcntl.flock(reserved.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already reserved"):
            launch.run_training(a, job, tmp_path / "data", tmp_path / "source.ckpt", {}, gpu)
    assert not a.run_dir.exists()


@pytest.mark.parametrize("offset, expected", [(0, "curriculum_full_batch_cuda_smoke_passed"), (1, "explicit_batch_override_not_profiled")])
def test_batch_override_provenance(offset, expected):
    batch = launch.DEFAULT_BATCHES["G"]["am_evrptw"] + offset
    job = launch.curriculum_job(launch.parse_args(["am_evrptw", "G", "0", "--batch-size", str(batch)]))
    assert job["physical_batch_size"] == job["effective_batch_size"] == batch
    assert job["calibration_status"] == expected


def test_two_gpu_supervisor_exposes_both_devices_and_inherits_locks(supervisor_case, tmp_path, monkeypatch):
    import os
    a, job, first = supervisor_case
    second = {"index": 1, "uuid": first["uuid"] + "-second"}
    job.update(world_size=2, effective_batch_size=2 * job['physical_batch_size'])
    code = """
import fcntl, json, os, pathlib, sys
run = pathlib.Path(sys.argv[1])
for uuid in sys.argv[2:]:
    path = pathlib.Path(f'/tmp/evrptw-ablation-gpu-locks-{os.getuid()}/{uuid}.lock')
    with path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError('a GPU lock was not inherited')
(run / 'visible_devices').write_text(os.environ['CUDA_VISIBLE_DEVICES'])
(run / 'training_result.json').write_text(json.dumps({'completed_training_epochs': 2}))
"""
    monkeypatch.setattr(launch, 'build_command', lambda *unused: [sys.executable, '-c', code,
        str(a.run_dir), first['uuid'], second['uuid']])
    assert launch.run_training(a, job, tmp_path / 'data', tmp_path / 'source.ckpt', {}, [first, second]) == 0
    assert (a.run_dir / 'visible_devices').read_text() == first['uuid'] + ',' + second['uuid']
    state = json.loads((a.run_dir / 'status.json').read_text())
    assert state['gpu'] == [0, 1]
    with pytest.raises(ProcessLookupError):
        os.kill(state['pid'], 0)


def test_second_gpu_lock_failure_releases_first_gpu(supervisor_case, tmp_path):
    import fcntl
    import os
    a, job, first = supervisor_case
    second = {'index': 1, 'uuid': first['uuid'] + '-second'}
    job['world_size'] = 2
    root = Path(f'/tmp/evrptw-ablation-gpu-locks-{os.getuid()}')
    root.mkdir(exist_ok=True)
    with (root / (second['uuid'] + '.lock')).open('a+') as occupied:
        fcntl.flock(occupied, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='GPU 1 is already reserved'):
            launch.run_training(a, job, tmp_path / 'data', tmp_path / 'source.ckpt', {}, [first, second])
        with (root / (first['uuid'] + '.lock')).open('a+') as released:
            fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not a.run_dir.exists()
