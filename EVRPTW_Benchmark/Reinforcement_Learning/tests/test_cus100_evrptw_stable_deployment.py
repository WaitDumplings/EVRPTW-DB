from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from EVRPTW_Benchmark.Reinforcement_Learning.common.training_diagnostics import summarize_values

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/cus100_20260911/2080ti_4_2/stable_launch.py"
spec = importlib.util.spec_from_file_location("cus100_stable_launch", SCRIPT)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def flag(command, key):
    assert command.count(key) == 1
    return command[command.index(key) + 1]


def test_selects_only_fresh_tr17_tr18_and_preserves_frozen_contracts():
    source = {job["experiment_id"]: job for job in launch.legacy.load_jobs()}
    jobs = launch.stable_jobs()
    assert [(job["experiment_id"], job["gpu"]) for job in jobs] == [("TR18_stable_mean", 2), ("TR17_stable_mean", 3)]
    for job in jobs:
        old = source[job["original_experiment_id"]]
        for key in ("seed", "training_stream_path_sha256", "training_stream_contract_sha256", "training_stream_contract_snapshot", "reward_contract_config_path_sha256", "reward_objective_scale", "reward_failure_base", "reward_unserved_coefficient", "objective_config_path_sha256", "method_auxiliary_profile_path_sha256", "effective_batch_size", "customer_exposure_budget"):
            assert job[key] == old[key]
        assert job["calibration_status"] == "pending_local_gpu_probe"
        assert job["enabled"] is False
        assert Path(job["training_stream_path"]).is_absolute()


def test_formal_command_remains_same_protocol_and_fresh(tmp_path):
    for job in launch.stable_jobs():
        command = launch.command_for(job, tmp_path / job["experiment_id"], physical=190)
        values = {"--graph-aggregation": "mean", "--physical-batch-size": "190", "--effective-batch-size": "200", "--training-epochs": "10000", "--minimum-training-epochs": "5000", "--training-rollout-steps": "240", "--validation-rollout-steps": "360", "--samples-per-instance": "30", "--validation-candidates": "30", "--validation-limit": "500", "--validation-every-epochs": "100", "--ema-warmup-steps": "1000", "--learning-rate": "0.001", "--customer-exposure-budget": "200000000"}
        for key, value in values.items():
            assert flag(command, key) == value
        assert "--resume" not in command and "--warm-start-checkpoint" not in command
        assert flag(command, "--training-stream-path") == job["training_stream_path"]


def test_probe_keeps_long_stream_and_rollouts_but_uses_exact_short_budget(tmp_path):
    job = launch.stable_jobs()[0]
    command = launch.command_for(job, tmp_path / "disposable", physical=190, probe=True)
    for key, value in {"--training-epochs": "6", "--minimum-training-epochs": "3", "--physical-batch-size": "190", "--effective-batch-size": "200", "--customer-exposure-budget": "120000", "--training-rollout-steps": "240", "--validation-rollout-steps": "360", "--ema-warmup-steps": "2", "--baseline-eval-interval": "2", "--baseline-eval-size": "2", "--validation-checkpoints": "2", "--validation-limit": "10"}.items():
        assert flag(command, key) == value
    assert flag(command, "--training-stream-path") == job["training_stream_path"]
    assert job["training_epochs"] == 10000
    assert job["physical_batch_size"] == 200


def test_explicit_roots_and_artifacts_are_preserved(tmp_path):
    jobs = launch.stable_jobs(artifact_root=tmp_path / "artifacts", road_root=tmp_path / "road", synthetic_root=tmp_path / "synthetic")
    assert jobs[0]["dataset_root"] == str(tmp_path / "road")
    assert jobs[1]["dataset_root"] == str(tmp_path / "synthetic")
    assert str(tmp_path / "artifacts") in jobs[0]["training_stream_path"]
    assert str(tmp_path / "artifacts") in jobs[1]["reward_contract_config_path"]


def test_sibling_fallback_and_explicit_absolute_path(tmp_path):
    repo = tmp_path / "new-worktree"
    fallback = tmp_path / "EVRPTW-DB" / "relative"
    fallback.mkdir(parents=True)
    assert launch.resolve_existing("relative", repo=repo) == fallback
    assert launch.resolve_existing(repo / "relative", repo=repo) == repo / "relative"


def test_busy_assigned_gpu_rejected_other_gpus_ignored(monkeypatch):
    jobs = launch.stable_jobs()
    monkeypatch.setattr(launch.legacy, "gpu_inventory", lambda: [{"index": i, "uuid": f"GPU-{i}", "name": "RTX 2080 Ti"} for i in range(4)])
    monkeypatch.setattr(launch.legacy, "gpu_processes", lambda: [{"gpu_uuid": "GPU-0", "pid": "100", "used_memory_mib": "10000"}])
    assert [row["index"] for row in launch.available_gpus(jobs)] == [2, 3]
    monkeypatch.setattr(launch.legacy, "gpu_processes", lambda: [{"gpu_uuid": "GPU-3", "pid": "101", "used_memory_mib": "10000"}])
    with pytest.raises(RuntimeError, match="No existing process was stopped"):
        launch.available_gpus(jobs)


def test_existing_formal_output_blocks_fresh_launch(tmp_path):
    jobs = launch.stable_jobs()
    run = tmp_path / "runs" / jobs[0]["experiment_id"]
    run.mkdir(parents=True)
    (run / "best.ckpt").write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="never resumes"):
        launch.preflight(jobs, tmp_path)


def test_repeated_launcher_record_blocks_second_launch(tmp_path):
    jobs = launch.stable_jobs()
    status = tmp_path / "launchers/2080ti_4_2/status.json"
    status.parent.mkdir(parents=True)
    status.write_text('{}')
    with pytest.raises(FileExistsError, match="launcher record"):
        launch.preflight(jobs, tmp_path)


def test_input_hash_failure_blocks_changed_artifact(tmp_path, monkeypatch):
    job = launch.stable_jobs()[0]
    job["dataset_root"] = str(tmp_path)
    for key in ("train_index", "validation_index", "training_stream_path", "objective_config_path", "reward_contract_config_path", "method_auxiliary_profile_path"):
        job[key] = str(tmp_path / key)
        Path(job[key]).write_text(key)
        job[key + "_sha256"] = launch.sha256(job[key])
    job["training_stream_path_sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="Frozen training_stream_path hash differs"):
        launch.audit_inputs([job])


def write_smoke(run):
    def rows(name, values):
        (run / name).write_text("".join(json.dumps(value) + "\n" for value in values))
    run.mkdir(exist_ok=True)
    (run / "training_result.json").write_text(json.dumps({"status": "passed", "completed_training_epochs": 6}))
    torch.save({"model": {"encoder": torch.tensor([1.0])}}, run / "checkpoint_epoch_0003.pt")
    torch.save({"model": {"encoder": torch.tensor([2.0])}}, run / "checkpoint_latest.pt")
    (run / "best.ckpt").write_bytes(b"model")
    rows("logical_epoch_history.jsonl", [{"mean_loss": 1.0, "logical_epoch": i} for i in range(1, 7)])
    rows("reward_diagnostics.jsonl", [{"baseline_kind": kind, "gradients": {"pre_clip_norm": summarize_values(0.01)}} for kind in ["paper_ema"] * 2 + ["greedy_rollout"] * 4])
    rows("baseline_history.jsonl", [{"optimizer_step": i} for i in (4, 6)])
    rows("validation_history.jsonl", [{"logical_epoch": i, "instances": 10, "complete_and_feasible": 10, "mean_verified_cost_usd": cost} for i, cost in ((3, 100.0), (6, 90.0))])


def test_audit_accepts_effective_smoke(tmp_path):
    write_smoke(tmp_path)
    assert launch.audit_probe(tmp_path)["completed_updates"] == 6


@pytest.mark.parametrize("filename, replacement, message", [
    ("training_result.json", '{"status":"passed","completed_training_epochs":5}', "six updates"),
    ("reward_diagnostics.jsonl", '\n'.join('{"baseline_kind":"paper_ema"}' for _ in range(6)), "EMA-to-greedy"),
    ("baseline_history.jsonl", '{"optimizer_step":4}\n', "baseline probes"),
    ("logical_epoch_history.jsonl", '\n'.join('{"mean_loss":NaN}' for _ in range(6)), "nonfinite"),
    ("validation_history.jsonl", '\n'.join(json.dumps({"logical_epoch": i, "instances": 10, "mean_verified_cost_usd": 100.0}) for i in (3, 6)), "exactly unchanged"),
])
def test_bad_smoke_is_rejected(tmp_path, filename, replacement, message):
    write_smoke(tmp_path)
    (tmp_path / filename).write_text(replacement)
    with pytest.raises(RuntimeError, match=message):
        launch.audit_probe(tmp_path)


@pytest.mark.parametrize("which", ["checkpoint_epoch_0003.pt", "checkpoint_latest.pt"])
def test_nonfinite_checkpoint_rejected(tmp_path, which):
    write_smoke(tmp_path)
    torch.save({"model": {"encoder": torch.tensor([float("nan")])}}, tmp_path / which)
    with pytest.raises(RuntimeError, match="Nonfinite"):
        launch.audit_probe(tmp_path)


def test_unchanged_checkpoint_rejected(tmp_path):
    write_smoke(tmp_path)
    torch.save({"model": {"encoder": torch.tensor([1.0])}}, tmp_path / "checkpoint_latest.pt")
    with pytest.raises(RuntimeError, match="did not change"):
        launch.audit_probe(tmp_path)


def test_calibration_retries_only_memory_failures_and_preserves_effective_batch(tmp_path, monkeypatch):
    job = launch.stable_jobs()[0]
    calls = []
    def probe(job, root, gpu, physical, locks, source_sha):
        calls.append(physical)
        return {"status": "passed", "peak_process_gib": 10.4 if physical == 200 else 10.0, "run": "probe"}
    monkeypatch.setattr(launch, "run_probe", probe)
    monkeypatch.setattr(launch, "diagnose_probe", lambda *args: {"policy_gate": {"passed": True}})
    selected = launch.calibrate_job(job, tmp_path, {}, [], "sha")
    assert calls == [200, 190]
    assert selected["physical_batch_size"] == 190
    assert selected["effective_batch_size"] == 200
    assert selected["customer_exposure_budget"] == job["customer_exposure_budget"]


def test_non_oom_probe_failure_never_starts_formal(tmp_path, monkeypatch):
    job = launch.stable_jobs()[0]
    monkeypatch.setattr(launch, "run_probe", lambda *args: {"status": "failed", "error": "bad gradients"})
    with pytest.raises(RuntimeError, match="bad gradients"):
        launch.calibrate_job(job, tmp_path, {}, [], "sha")
    assert not (tmp_path / "runs").exists()


def test_source_change_blocks_execution(monkeypatch):
    monkeypatch.setattr(launch, "source_files", lambda _: [])
    with pytest.raises(RuntimeError, match="Source changed"):
        launch.require_source("other-source")


def test_failed_policy_diagnostic_blocks_formal_after_finite_smoke(tmp_path, monkeypatch):
    job = launch.stable_jobs()[0]
    monkeypatch.setattr(launch, "run_probe", lambda *args: {"status": "passed", "peak_process_gib": 10.0, "run": "probe"})
    def fail(*args):
        raise RuntimeError("collapsed encoder")
    monkeypatch.setattr(launch, "diagnose_probe", fail)
    with pytest.raises(RuntimeError, match="collapsed encoder"):
        launch.calibrate_job(job, tmp_path, {}, [], "sha")
    assert not (tmp_path / "runs").exists()


def test_explicit_dataset_root_cannot_be_overridden_by_later_environment(tmp_path, monkeypatch):
    jobs = launch.stable_jobs(road_root=tmp_path / "selected-road", synthetic_root=tmp_path / "selected-synthetic")
    monkeypatch.setenv("CUS100_ROAD_ROOT", str(tmp_path / "wrong-road"))
    monkeypatch.setenv("CUS100_SYNTHETIC_ROOT", str(tmp_path / "wrong-synthetic"))
    for job in jobs:
        command = launch.command_for(job, tmp_path / job["experiment_id"])
        assert flag(command, "--dataset-path") == str(Path(job["dataset_root"]) / job["train_index"])


def test_preflight_foreground_combination_never_starts_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(launch.sys, "argv", ["launch", "--mode", "preflight", "--foreground", "--output-root", str(tmp_path)])
    monkeypatch.setattr(launch, "preflight", lambda *args: {"status": "preflight_only"})
    monkeypatch.setattr(launch, "worker", lambda *_: pytest.fail("preflight must not start worker"))
    assert launch.main() == 0
    assert not (tmp_path / "launchers").exists()


def test_start_holds_lock_and_freezes_jobs_before_spawning(tmp_path, monkeypatch):
    import fcntl
    import os
    from types import SimpleNamespace
    monkeypatch.setattr(launch.sys, "argv", ["launch", "--output-root", str(tmp_path)])
    monkeypatch.setattr(launch, "preflight", lambda *args: {"input_sha256": {"file": "hash"}})
    monkeypatch.setattr(launch, "capture_source", lambda *args: {"source_sha256": "frozen-source"})
    def fake_popen(command, **kwargs):
        request = json.loads((tmp_path / "launchers/2080ti_4_2/launch_request.json").read_text())
        assert request["source_sha256"] == "frozen-source"
        assert len(request["jobs"]) == 2
        fd = kwargs["pass_fds"][0]
        assert flag(command, "--launch-lock-fd") == str(fd)
        os.fstat(fd)
        with (tmp_path / "launchers/2080ti_4_2/launcher.lock").open("a") as competing:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return SimpleNamespace(pid=12345)
    monkeypatch.setattr(launch.subprocess, "Popen", fake_popen)
    assert launch.main() == 0
    request = json.loads((tmp_path / "launchers/2080ti_4_2/launch_request.json").read_text())
    assert request["pid"] == 12345
    assert request["input_audit"]["input_sha256"] == {"file": "hash"}


def test_compact_status_preserves_full_precision_cost_and_log_age(tmp_path):
    run = tmp_path / "runs/TR18_stable_mean"
    run.mkdir(parents=True)
    (run / "logical_epoch_history.jsonl").write_text('{"logical_epoch":20,"mean_environment_feasible_rate":0.9}\n')
    (run / "validation_history.jsonl").write_text('{"logical_epoch":10,"mean_verified_cost_usd":123.456789,"complete_and_feasible":499,"instances":500}\n')
    (run / "validation_summary.json").write_text(json.dumps({"logical_epoch":10,"mean_verified_cost_usd":123.456789,"complete_and_feasible":499,"instances":500}, indent=2))
    launch.write_json(tmp_path / "launchers/2080ti_4_2/status.json", {"status": "running", "jobs": [{"experiment_id": "TR18_stable_mean", "status": "running", "pid": 123, "gpu": {"index": 2}, "job": {"huge": "details"}, "command": ["long"], "output_dir": str(run)}]})
    state = launch.compact_status(tmp_path)
    row = state["jobs"][0]
    assert row["epoch"] == 20
    assert row["train_write_age_s"] >= 0
    assert row["latest_validation"]["mean_verified_cost_usd"] == 123.456789
    assert row["best_validation"]["mean_verified_cost_usd"] == 123.456789
    assert "command" not in row and "job" not in row


@pytest.mark.parametrize("field", ["baseline", "optimizer"])
def test_nonfinite_training_state_rejected(tmp_path, field):
    write_smoke(tmp_path)
    torch.save({"model": {"encoder": torch.tensor([2.0])}, field: {"nested": [torch.tensor(float("inf"))]}}, tmp_path / "checkpoint_latest.pt")
    with pytest.raises(RuntimeError, match="Nonfinite"):
        launch.audit_probe(tmp_path)


@pytest.mark.parametrize("mutation", [{}, {"counterfactual": True}, {"effective_graph_aggregation": "sum"}, {"checkpoint_sha256": "different"}, {"training_index_sha256": "different"}, {"gate": {"passed": False}}])
def test_diagnostic_is_bound_to_actual_checkpoint_and_training_data(tmp_path, monkeypatch, mutation):
    from types import SimpleNamespace
    job = launch.stable_jobs()[0]
    run = tmp_path / "calibration/TR18_stable_mean/batch_200"
    run.mkdir(parents=True)
    (run / "checkpoint_latest.pt").write_bytes(b"actual checkpoint")
    monkeypatch.setattr(launch, "require_source", lambda _: None)
    def fake_popen(command, **kwargs):
        assert "--graph-aggregation" not in command
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert flag(command, "--checkpoint") == str(run / "checkpoint_latest.pt")
        report = {"gate": {"passed": True}, "counterfactual": False,
                  "effective_graph_aggregation": "mean", "checkpoint_sha256": launch.sha256(run / "checkpoint_latest.pt"),
                  "training_index_sha256": job["train_index_sha256"]}
        report.update(mutation)
        Path(flag(command, "--output")).write_text(json.dumps(report))
        return SimpleNamespace(returncode=0, poll=lambda: 0)
    monkeypatch.setattr(launch.subprocess, "Popen", fake_popen)
    if mutation:
        with pytest.raises(RuntimeError, match="actual mean-policy"):
            launch.diagnose_probe(job, run, "sha")
    else:
        assert launch.diagnose_probe(job, run, "sha")["policy_gate"]["passed"]


@pytest.mark.parametrize("gradient", [0.0, 1e-12, float("nan"), float("inf")])
def test_actual_training_gradient_required_beyond_parameter_decay(tmp_path, gradient):
    write_smoke(tmp_path)
    rows = launch.read_rows(tmp_path / "reward_diagnostics.jsonl")
    for row in rows:
        row["gradients"]["pre_clip_norm"] = summarize_values(gradient)
    (tmp_path / "reward_diagnostics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(RuntimeError, match="training gradients"):
        launch.audit_probe(tmp_path)
