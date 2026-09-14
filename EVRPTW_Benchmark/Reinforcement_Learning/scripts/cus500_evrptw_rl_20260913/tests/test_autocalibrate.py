"""CPU-only coverage for batch selection, probe safety and real EVR artifacts."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import signal
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist

from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913 import autocalibrate as calibration
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus500_evrptw_rl_20260913.common import load_config, write_json


def measured(batch, *, threshold=7, failure="oom"):
    if batch > threshold and failure == "oom":
        return {"status": "oom", "peak_process_gib": {}}
    return {"status": "passed", "peak_process_gib": {"0": batch, "1": batch + 0.1}}


@pytest.mark.parametrize("initial,maximum,threshold", [(4, 16, 7), (8, 16, 1), (1, 1, 1), (4, 7, 7), (1, 16, 9)])
def test_choose_batch_brackets_oom_and_returns_largest_measured_safe(initial, maximum, threshold):
    selected, tested = calibration.choose_batch(lambda batch: measured(batch, threshold=threshold),
                                               initial_batch=initial, max_batch=maximum)
    assert selected == threshold
    assert tested[selected]["status"] == "passed"
    assert all(1 <= batch <= maximum for batch in tested)


def test_choose_batch_uses_hotter_rank_and_inclusive_ceiling():
    def measure(batch):
        return {"status": "passed", "peak_process_gib": {"0": 1.0, "1": batch + 0.3}}
    selected, tested = calibration.choose_batch(measure, initial_batch=4, max_batch=16, ceiling_gib=10.3)
    assert selected == 10 and tested[10]["peak_process_gib"]["1"] == 10.3


@pytest.mark.parametrize("peaks", [{}, {"0": 9.5}, {"0": 9.5, "1": 0}, {"0": 9.5, "1": -1},
                                   {"0": 9.5, "1": float("nan")}, {"0": float("inf"), "1": 9.5},
                                   {"0": 9.5, "1": 9.5, "2": 9.5}])
def test_choose_batch_requires_two_positive_finite_rank_measurements(peaks):
    with pytest.raises(RuntimeError, match="both ranks"):
        calibration.choose_batch(lambda _batch: {"status": "passed", "peak_process_gib": peaks})


def test_no_fitting_batch_and_non_oom_failure_abort():
    with pytest.raises(RuntimeError, match="No tested batch"):
        calibration.choose_batch(lambda _batch: {"status": "oom"})
    calls = []
    def fail(batch):
        calls.append(batch)
        return {"status": "failed", "error": "shape mismatch"}
    with pytest.raises(RuntimeError, match="shape mismatch"):
        calibration.choose_batch(fail)
    assert calls == [4]


@pytest.mark.parametrize("kwargs", [{"initial_batch": 0}, {"initial_batch": 5, "max_batch": 4},
                                    {"ceiling_gib": 0}, {"ceiling_gib": float("nan")}])
def test_choose_batch_rejects_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        calibration.choose_batch(lambda batch: measured(batch), **kwargs)


def flag(arguments, name):
    return arguments[arguments.index(name) + 1]


@pytest.mark.parametrize("confirmation", [False, True])
def test_probe_preserves_horizon_trajectories_architecture_objective_and_auxiliary(confirmation):
    formal = load_config(batch=8, accumulation=2)
    original = deepcopy(formal)
    probe = calibration.probe_config(formal, 3, confirmation=confirmation)
    assert formal == original
    for name in ("training_rollout_steps", "validation_rollout_steps", "samples_per_instance", "validation_candidates",
                 "objective_config", "reward_contract", "method_auxiliary_profile", "learning_rate", "weight_decay",
                 "train_module", "instance_cache_size", "seed", "validation_seed"):
        assert probe[name] == formal[name], name
    assert (probe["training_rollout_steps"], probe["validation_rollout_steps"], probe["samples_per_instance"]) == (1700, 2550, 30)
    assert flag(probe["extra_args"], "--activation-checkpoint-stride") == "1"
    assert flag(probe["extra_args"], "--ema-decay") == "0.9"
    assert flag(probe["extra_args"], "--max-grad-norm") == "2.0"
    assert flag(probe["extra_args"], "--ema-warmup-steps") == ("2" if confirmation else "0")
    assert flag(probe["extra_args"], "--baseline-eval-interval") == ("2" if confirmation else "100")
    assert probe["training_epochs"] == (6 if confirmation else 2)
    assert probe["validation_limit"] == (10 if confirmation else 2)
    assert probe["effective_batch_size"] == 6 and probe["gradient_accumulation_steps"] == 1
    assert probe["sample_count"] == 6 * probe["training_epochs"]
    assert probe["customer_exposure_budget"] == probe["sample_count"] * 500
    from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.distributed_train import parse_args
    parsed = parse_args(["--dataset-path", "unused", "--output-dir", "unused", *probe["extra_args"]])
    assert parsed.embedding_dim == 128 and parsed.structure2vec_rounds == 3


def _synthetic_checkpoint_artifacts(run, *, confirmation=True):
    """Real torch serialization and expected logger schema, no CUDA allocation."""
    run.mkdir(parents=True)
    epochs, first, cohort = (6, 3, 10) if confirmation else (2, 1, 2)
    write_json(run / "training_result.json", {"status": "passed", "completed_training_epochs": epochs})
    before = {"model": {"weight": torch.tensor([1.0, 2.0])}}
    after = {"model": {"weight": torch.tensor([1.1, 2.2])}}
    torch.save(before, run / f"checkpoint_epoch_{first:04d}.pt")
    torch.save(after, run / "checkpoint_latest.pt")
    torch.save(after, run / "best.ckpt")
    history = [{"logical_epoch": epoch, "mean_loss": 0.2,
                "baseline_kind": "paper_ema" if confirmation and epoch <= 2 else "greedy_rollout",
                "baseline_warmup_synchronized": confirmation and epoch == 2}
               for epoch in range(1, epochs + 1)]
    (run / "logical_epoch_history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
    (run / "validation_history.jsonl").write_text("".join(json.dumps({"logical_epoch": epoch, "instances": cohort,
        "complete_and_feasible": cohort}) + "\n" for epoch in (first, epochs)))
    if confirmation:
        (run / "baseline_history.jsonl").write_text(''.join(json.dumps({"optimizer_step": epoch}) + '\n' for epoch in (4, 6)))
    return run


@pytest.mark.parametrize("confirmation", [False, True])
def test_checkpoint_audit_accepts_real_cpu_serialized_probe_schema(tmp_path, confirmation):
    run = _synthetic_checkpoint_artifacts(tmp_path / "run", confirmation=confirmation)
    result = calibration.checkpoint_audit(run, expected_updates=6 if confirmation else 2, confirmation=confirmation)
    assert result["finite_parameters"] and result["changed_float_tensors"] == 1
    assert result["baseline_transition_checked"] is confirmation


@pytest.mark.parametrize("mutation,match", [("nan_final", "Non-finite"), ("nan_first", "Non-finite"),
    ("unchanged", "no parameter changes"), ("missing_copy", "baseline copy"),
    ("wrong_greedy", "greedy baseline"), ("wrong_probe", "baseline probes"),
    ("wrong_cohort", "validation cohort"), ("nonfinite_loss", "non-finite")])
def test_checkpoint_audit_rejects_failed_updates_and_incomplete_transition(tmp_path, mutation, match):
    run = _synthetic_checkpoint_artifacts(tmp_path / "run")
    if mutation in {"nan_final", "nan_first", "unchanged"}:
        path = run / ("checkpoint_epoch_0003.pt" if mutation == "nan_first" else "checkpoint_latest.pt")
        tensor = torch.tensor([1.0, 2.0]) if mutation == "unchanged" else torch.tensor([float("nan"), 2.0])
        torch.save({"model": {"weight": tensor}}, path)
    elif mutation == "wrong_probe":
        (run / "baseline_history.jsonl").write_text('{"optimizer_step": 2}\n{"optimizer_step": 6}\n')
    elif mutation == "wrong_cohort":
        (run / "validation_history.jsonl").write_text('{"instances": 9}\n{"instances": 10}\n')
    else:
        rows = calibration.read_jsonl(run / "logical_epoch_history.jsonl")
        if mutation == "missing_copy": rows[1]["baseline_warmup_synchronized"] = False
        if mutation == "wrong_greedy": rows[2]["baseline_kind"] = "paper_ema"
        if mutation == "nonfinite_loss": rows[3]["mean_loss"] = float("nan")
        (run / "logical_epoch_history.jsonl").write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(RuntimeError, match=match):
        calibration.checkpoint_audit(run, expected_updates=6, confirmation=True)



@pytest.mark.parametrize("component", ["baseline", "optimizer"])
def test_checkpoint_audit_rejects_nonfinite_baseline_and_optimizer(tmp_path, component):
    run = _synthetic_checkpoint_artifacts(tmp_path / "run")
    final = torch.load(run / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    final[component] = {"nested": [torch.tensor([complex(1, float("inf"))])]} if component == "baseline" else {
        "state": {0: {"exp_avg_sq": torch.tensor([float("nan")])}}}
    torch.save(final, run / "checkpoint_latest.pt")
    with pytest.raises(RuntimeError, match=f"Non-finite {component}"):
        calibration.checkpoint_audit(run, expected_updates=6, confirmation=True)


def test_checkpoint_audit_rejects_changed_model_schema(tmp_path):
    run = _synthetic_checkpoint_artifacts(tmp_path / "run")
    torch.save({"model": {"other_weight": torch.tensor([1.0])}}, run / "checkpoint_epoch_0003.pt")
    with pytest.raises(RuntimeError, match="schema differs"):
        calibration.checkpoint_audit(run, expected_updates=6, confirmation=True)

def _gpu(index):
    return {"index": index, "uuid": f"GPU-{index}", "name": "NVIDIA GeForce RTX 2080 Ti",
            "memory.total": 11264, "memory.used": 10}


def test_non_oom_probe_failure_never_publishes_formal_config(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "require_source", lambda _expected: None)
    monkeypatch.setattr(calibration, "preflight", lambda *_args: {"gpus": [_gpu(0), _gpu(1)], "data": {}})
    monkeypatch.setattr(calibration, "lock_gpus", lambda _selected: [])
    monkeypatch.setattr(calibration, "validate_gpus", lambda *_args: None)
    monkeypatch.setattr(calibration, "gpu_inventory", lambda: [])
    monkeypatch.setattr(calibration, "gpu_processes", lambda: [])
    calls = []
    def probe(*_args, **kwargs):
        calls.append(kwargs["batch"])
        return {"status": "failed", "error": "NCCL fatal error", "peak_process_gib": {}}
    monkeypatch.setattr(calibration, "run_probe", probe)
    monkeypatch.setattr(calibration, "prepare_stream", lambda *_a, **_k: pytest.fail("formal stream must not be built after failure"))
    with pytest.raises(RuntimeError, match="NCCL fatal"):
        calibration.calibrate(load_config(), tmp_path, tmp_path / "output")
    assert calls == [4]
    assert not (tmp_path / "output/calibrated_config.json").exists()
    assert json.loads((tmp_path / "output/calibration/status.json").read_text())["status"] == "failed"


class _Child:
    pid = 424242
    def __init__(self, code=0):
        self.code, self.returncode, self.polls = code, None, 0
    def poll(self):
        self.polls += 1
        if self.polls > 1: self.returncode = self.code
        return self.returncode
    def wait(self, timeout=None):
        self.returncode = self.code
        return self.returncode


def _mock_probe_io(monkeypatch, *, exit_code=0, stderr_text=""):
    child, launches, signals = _Child(exit_code), [], []
    monkeypatch.setattr(calibration, "prepare_stream", lambda *_a, **_k: {"path": "unused", "contract": {"sha256": "abc"}})
    monkeypatch.setattr(calibration, "build_command", lambda *_a, **_k: ["mock-probe"])
    monkeypatch.setattr(calibration, "checkpoint_audit", lambda *_a, **_k: {"finite_parameters": True})
    monkeypatch.setattr(calibration.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(calibration, "gpu_inventory", lambda: [_gpu(0), _gpu(1)])
    def start(command, **kwargs):
        launches.append((command, kwargs))
        run = Path(kwargs["stdout"].name).parent
        write_json(run / "distributed_workers.json", {"workers": [{"rank": 0, "pid": 424243}, {"rank": 1, "pid": 424244}]})
        kwargs["stderr"].write(stderr_text); kwargs["stderr"].flush()
        return child
    monkeypatch.setattr(calibration.subprocess, "Popen", start)
    def processes():
        if not launches or child.returncode is not None: return []
        return [{"pid": 424243, "gpu_uuid": "GPU-0", "used_memory_mib": 10000},
                {"pid": 424244, "gpu_uuid": "GPU-1", "used_memory_mib": 10200},
                {"pid": 999999, "gpu_uuid": "GPU-2", "used_memory_mib": 11000}]
    monkeypatch.setattr(calibration, "gpu_processes", processes)
    monkeypatch.setattr(calibration.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    return child, launches, signals


@pytest.mark.parametrize("code,stderr,expected", [(0, "", "passed"),
    (1, "torch.OutOfMemoryError: CUDA out of memory.", "oom"),
    (1, "RuntimeError: CUDA error: out of memory", "oom"),
    (1, "RuntimeError: expected matrix shape mismatch", "failed")])
def test_probe_classifies_cuda_oom_and_samples_only_its_own_workers(tmp_path, monkeypatch, code, stderr, expected):
    child, launches, signals = _mock_probe_io(monkeypatch, exit_code=code, stderr_text=stderr)
    result = calibration.run_probe(load_config(), tmp_path, tmp_path / "probe", [_gpu(0), _gpu(1)], [81, 82], audit={}, batch=2)
    assert result["status"] == expected
    assert result["peak_process_mib"] == {"0": 10000, "1": 10200}
    assert launches[0][1]["start_new_session"] is True
    assert launches[0][1]["pass_fds"] == (81, 82)
    assert launches[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-0,GPU-1"
    assert not signals


def test_probe_timeout_stops_only_new_probe_session(tmp_path, monkeypatch):
    child, launches, signals = _mock_probe_io(monkeypatch)
    result = calibration.run_probe(load_config(), tmp_path, tmp_path / "probe", [_gpu(0), _gpu(1)], [], audit={}, batch=2, timeout_seconds=-1)
    assert result["status"] == "failed" and "exceeded" in result["error"]
    assert signals == [(child.pid, signal.SIGTERM)]


def test_monitor_failure_stops_only_new_probe_session(tmp_path, monkeypatch):
    child, launches, signals = _mock_probe_io(monkeypatch)
    def processes():
        if not launches: return []
        raise RuntimeError("nvidia-smi unavailable")
    monkeypatch.setattr(calibration, "gpu_processes", processes)
    with pytest.raises(RuntimeError, match="nvidia-smi unavailable"):
        calibration.run_probe(load_config(), tmp_path, tmp_path / "probe", [_gpu(0), _gpu(1)], [], audit={}, batch=2)
    assert signals == [(child.pid, signal.SIGTERM)]



@pytest.mark.parametrize("problem", ["foreign_compute", "gpu_uuid_changed"])
def test_new_probe_refuses_changed_gpu_occupancy_without_launch_or_kill(tmp_path, monkeypatch, problem):
    child, launches, signals = _mock_probe_io(monkeypatch)
    if problem == "foreign_compute":
        monkeypatch.setattr(calibration, "gpu_processes", lambda: [{"pid": 919191, "gpu_uuid": "GPU-0",
            "used_memory_mib": 64, "executable": "/external/training/python"}])
    else:
        changed = [_gpu(0), _gpu(1)]
        changed[0]["uuid"] = "GPU-replaced"
        monkeypatch.setattr(calibration, "gpu_inventory", lambda: changed)
    with pytest.raises(RuntimeError, match="compute PID|identity changed"):
        calibration.run_probe(load_config(), tmp_path, tmp_path / "probe", [_gpu(0), _gpu(1)], [], audit={}, batch=2)
    assert not launches and not signals

def _actual_confirmation_worker(rank, world_size, rendezvous, output):
    from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.tests.test_am_model import _instance
    from EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.distributed_train import build_policy, prepare_method
    from EVRPTW_Benchmark.Reinforcement_Learning.common import distributed_protocol as protocol
    from EVRPTW_Benchmark.Reinforcement_Learning.common import protocol_entrypoints as entrypoints
    from EVRPTW_Benchmark.Reinforcement_Learning.common.objective import load_objective
    from EVRPTW_Benchmark.Reinforcement_Learning.common.tests.test_distributed_evrptw_protocol import _evr_args, CONFIGS
    from EVRPTW_Benchmark.Reinforcement_Learning.common.tests.test_distributed_helpers import init_gloo

    class Pool:
        def __init__(self, validation=False):
            prefix, count = ("val", 10) if validation else ("train", 24)
            self.tasks = [SimpleNamespace(view_id=f"{prefix}-{index}") for index in range(count)]
            self._task_by_view_id = {task.view_id: task for task in self.tasks}
            self.rng = np.random.default_rng(1234)
            self.reward_scale_metadata = {}
        def __len__(self): return len(self.tasks)
        def instance(self, task): return replace(_instance(), instance_id=task.view_id)
        def reward_distance_scale_km(self, _mode): return 1.0

    ctx = init_gloo(rank, world_size, rendezvous)
    try:
        protocol.read_stream_view_ids = lambda _path, *, stop: [f"train-{index}" for index in range(stop)]
        protocol.make_validation_pool = lambda *_a, **_k: Pool(validation=True)
        original_make_envs = entrypoints.make_envs
        entrypoints.make_envs = lambda *a, **k: original_make_envs(*a, **k, use_jit_mask=False)
        args = _evr_args(Path(output) / "run", output)
        args.objective = load_objective(CONFIGS / "rivian_energy_vehicle_cost_v2.json").to_dict()
        args.reward_contract = CONFIGS / "drl_reward_contract_energy_vehicle_v3.json"
        args.embedding_dim = 16
        args.physical_batch_size = args.batch_size = 2
        args.instance_cache_size = 256
        args.training_rollout_steps, args.validation_rollout_steps = 32, 48
        args.validation_limit = 10
        args.validation_every_epochs = args.post_minimum_validation_every_epochs = 3
        args.minimum_training_epochs = args.early_stop_start_epoch = 3
        args.validation_checkpoints = 2
        args.baseline_eval_size = 2
        args.incomplete_penalty = 100.0
        prepare_method(args)
        protocol.configure_distributed_contract(args, ctx, method="EVRPTW-RL")
        torch.manual_seed(1234)
        policy = build_policy(args)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        entrypoints.run_evrptw_rl(args, Pool(), policy, optimizer)
    finally:
        dist.destroy_process_group()


def test_real_two_rank_cpu_evr_confirmation_artifacts_pass_production_audit(tmp_path):
    from EVRPTW_Benchmark.Reinforcement_Learning.common.tests.test_distributed_helpers import run_gloo_workers
    run_gloo_workers(_actual_confirmation_worker, tmp_path)
    run = tmp_path / "run"
    audit = calibration.checkpoint_audit(run, expected_updates=6, confirmation=True)
    assert audit["finite_parameters"] and audit["changed_float_tensors"] > 0
    assert audit["baseline_transition_checked"] and audit["validation_instances_each"] == 10
    final = torch.load(run / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    assert final["logical_epoch"] == 6 and final["stream_cursor"] == 24
    assert len(final["rank_rng_states"]) == 2
    assert final["method_auxiliary_profile"]["weights"] == {"station_visit": 0.3}
