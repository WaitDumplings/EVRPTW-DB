#!/usr/bin/env python3
"""Measure EVRPTW-RL on free GPU 0/1, freeze a batch, then return its config.

Called after AM has completed. Probes are independent scratch runs; the final
training always starts from the original seed, never from probe checkpoints.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

if __package__:
    from .common import CONFIG, OUTPUT, REPO, load_config, resolve_road_root, source_snapshot, timestamp, write_json
    from .launch import build_command, gpu_inventory, gpu_processes, lock_gpus, preflight, validate_gpus
    from .prepare import prepare_stream
else:
    from common import CONFIG, OUTPUT, REPO, load_config, resolve_road_root, source_snapshot, timestamp, write_json
    from launch import build_command, gpu_inventory, gpu_processes, lock_gpus, preflight, validate_gpus
    from prepare import prepare_stream


def require_source(expected):
    if expected and source_snapshot()["source_sha256"] != expected:
        raise RuntimeError("Source changed after the watcher was armed")


def override_flag(arguments, flag, value):
    result = list(arguments)
    while flag in result:
        index = result.index(flag)
        del result[index:index + 2]
    result.extend((flag, str(value)))
    return result


def probe_config(config, batch, *, confirmation=False):
    """Leave every model/rollout/reward parameter intact; shorten only the test."""
    result = copy.deepcopy(config)
    epochs, minimum, interval = (6, 3, 3) if confirmation else (2, 1, 1)
    result.update(physical_batch_size=int(batch), gradient_accumulation_steps=1,
                  training_epochs=epochs, minimum_training_epochs=minimum,
                  validation_every_epochs=interval, early_stop_start_epoch=minimum,
                  early_stop_patience_validations=5, validation_limit=10 if confirmation else 2,
                  protocol_id="cus500_evrptw_rl_disposable_memory_probe_v1",
                  calibration_status="engineering_probe")
    for flag, value in (("--ema-warmup-steps", 2 if confirmation else 0),
                        ("--baseline-eval-interval", 2 if confirmation else 100),
                        ("--baseline-eval-size", 2)):
        result["extra_args"] = override_flag(result["extra_args"], flag, value)
    result["world_size"] = 2
    result["effective_batch_size"] = int(batch) * 2
    result["sample_count"] = result["effective_batch_size"] * epochs
    result["customer_exposure_budget"] = result["sample_count"] * 500
    return result


def choose_batch(measure, *, ceiling_gib=10.3, initial_batch=4, max_batch=128):
    """Bracket and binary-search the largest observed-safe integer batch."""
    if not 1 <= initial_batch <= max_batch or not math.isfinite(ceiling_gib) or ceiling_gib <= 0:
        raise ValueError("invalid batch-search bounds")
    tested = {}

    def safe(batch):
        result = measure(batch)
        tested[batch] = result
        if result["status"] == "oom":
            return False
        if result["status"] != "passed":
            raise RuntimeError(f"GPU probe failed at batch {batch}: {result.get('error', result['status'])}")
        peaks = result.get("peak_process_gib", {})
        if set(peaks) != {"0", "1"} or any(not math.isfinite(v) or v <= 0 for v in peaks.values()):
            raise RuntimeError("Probe did not collect positive memory measurements on both ranks")
        return max(peaks.values()) <= ceiling_gib

    low, high = 0, max_batch + 1
    batch = initial_batch
    if safe(batch):
        low = batch
        while low < max_batch:
            batch = min(max_batch, low * 2)
            if safe(batch):
                low = batch
            else:
                high = batch
                break
    else:
        high = batch
        while high > 1:
            batch = max(1, high // 2)
            if safe(batch):
                low = batch
                break
            high = batch
    if low == 0:
        raise RuntimeError("No tested batch fits GPU 0/1; formal training was not started")
    while high - low > 1:
        batch = (low + high) // 2
        if safe(batch):
            low = batch
        else:
            high = batch
    return low, tested


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def checkpoint_audit(run, *, expected_updates, confirmation):
    import torch

    result = json.loads((run / "training_result.json").read_text())
    if result.get("status") != "passed" or result.get("completed_training_epochs") != expected_updates:
        raise RuntimeError("Probe did not complete the requested optimizer updates")
    for name in ("best.ckpt", "checkpoint_latest.pt"):
        if not (run / name).is_file() or not (run / name).stat().st_size:
            raise RuntimeError(f"Probe checkpoint missing: {name}")
    final = torch.load(run / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    model = final.get("model", final.get("model_state_dict"))
    if model is None:
        raise RuntimeError("Unknown checkpoint model schema")
    def require_finite_tensors(value, label):
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Non-finite {label} tensors in probe checkpoint")
        elif isinstance(value, dict):
            for nested in value.values():
                require_finite_tensors(nested, label)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                require_finite_tensors(nested, label)

    for label, value in (("model", model), ("baseline", final.get("baseline")),
                         ("optimizer", final.get("optimizer"))):
        require_finite_tensors(value, label)
    history = read_jsonl(run / "logical_epoch_history.jsonl")
    if len(history) != expected_updates or any(not math.isfinite(row["mean_loss"]) for row in history):
        raise RuntimeError("Missing or non-finite probe updates")
    first_epoch = 3 if confirmation else 1
    first = torch.load(run / f"checkpoint_epoch_{first_epoch:04d}.pt", map_location="cpu", weights_only=False)
    before = first.get("model", first.get("model_state_dict"))
    if before is None or set(before) != set(model):
        raise RuntimeError("Reference checkpoint model schema differs")
    require_finite_tensors(before, "reference model")
    changed = sum(not torch.equal(before[name], value) for name, value in model.items()
                  if torch.is_tensor(value) and value.is_floating_point())
    if not changed:
        raise RuntimeError("Probe produced no parameter changes")
    if confirmation:
        if history[0]["baseline_kind"] != "paper_ema" or history[1]["baseline_kind"] != "paper_ema":
            raise RuntimeError("Confirmation did not exercise EMA")
        if not history[1].get("baseline_warmup_synchronized"):
            raise RuntimeError("Confirmation missed the EMA-to-greedy baseline copy")
        if any(row["baseline_kind"] != "greedy_rollout" for row in history[2:]):
            raise RuntimeError("Confirmation did not exercise greedy baseline")
        probes = read_jsonl(run / "baseline_history.jsonl")
        if [row["optimizer_step"] for row in probes] != [4, 6]:
            raise RuntimeError("Confirmation missed post-warmup baseline probes")
    validations = read_jsonl(run / "validation_history.jsonl")
    if len(validations) != 2 or any(row["instances"] != (10 if confirmation else 2) for row in validations):
        raise RuntimeError("Probe validation cohort was not completed")
    return {"finite_parameters": True, "changed_float_tensors": changed,
            "optimizer_updates": expected_updates, "baseline_transition_checked": confirmation,
            "validation_checkpoints": len(validations), "validation_instances_each": validations[0]["instances"],
            "final_validation_feasible": validations[-1]["complete_and_feasible"]}


def stop_probe_group(child):
    """Terminate only the new session created for this disposable torchrun."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=10)


def run_probe(config, road_root, directory, selected, lock_fds, *, audit, batch,
              confirmation=False, timeout_seconds=3600):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    scratch = probe_config(config, batch, confirmation=confirmation)
    scratch["run_id"] = directory.name
    write_json(directory / "config.json", scratch)
    scratch = load_config(directory / "config.json")
    stream = prepare_stream(road_root, directory / "artifacts", scratch, audit=audit)
    run = directory / "run"
    run.mkdir()
    command = build_command(scratch, road_root, run, stream)
    write_json(directory / "command.json", command)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpu["uuid"] for gpu in selected),
               CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[key] = os.environ.get(key, "2")
    started = time.monotonic()
    peaks, host_peaks, workers = {}, {}, {}
    error, timed_out = None, False
    child = None
    with (run / "stdout.log").open("w") as stdout, (run / "stderr.log").open("w") as stderr:
        try:
            # Our shared locks cannot exclude unrelated jobs started externally.
            current = validate_gpus(config, gpu_inventory(), gpu_processes())
            if [gpu["uuid"] for gpu in current] != [gpu["uuid"] for gpu in selected]:
                raise RuntimeError("Physical GPU identity changed between calibration probes")
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=stdout, stderr=stderr,
                                     start_new_session=True, pass_fds=tuple(lock_fds))
            write_json(directory / "process.json", {"pid": child.pid, "time": timestamp()})
            while child.poll() is None:
                if time.monotonic() - started > timeout_seconds:
                    timed_out = True
                    stop_probe_group(child)
                    break
                identity = run / "distributed_workers.json"
                if identity.exists() and not workers:
                    workers = {int(row["pid"]): str(row["rank"]) for row in json.loads(identity.read_text())["workers"]}
                for row in gpu_processes():
                    if row["pid"] not in workers or row.get("used_memory_mib") is None:
                        continue
                    rank = workers[row["pid"]]
                    peaks[rank] = max(peaks.get(rank, 0), row["used_memory_mib"])
                    try:
                        for line in Path(f"/proc/{row['pid']}/status").read_text().splitlines():
                            if line.startswith("VmHWM:"):
                                host_peaks[rank] = max(host_peaks.get(rank, 0), int(line.split()[1]))
                    except FileNotFoundError:
                        pass
                time.sleep(0.5)
        except BaseException:
            if child is not None:
                stop_probe_group(child)
            raise
    stderr_text = (run / "stderr.log").read_text(errors="replace")
    status = "passed"
    if timed_out:
        status, error = "failed", f"Probe exceeded {timeout_seconds}s"
    elif child.returncode:
        cuda_oom = "CUDA out of memory" in stderr_text or "CUDA error: out of memory" in stderr_text
        status = "oom" if cuda_oom else "failed"
        error = stderr_text[-6000:]
    result = {"status": status, "batch_per_gpu": batch, "global_batch": batch * 2,
              "confirmation": confirmation, "returncode": child.returncode,
              "wall_seconds": time.monotonic() - started,
              "peak_process_mib": peaks, "peak_process_gib": {k: v / 1024 for k, v in peaks.items()},
              "peak_host_rss_gib": {k: v / 1024 / 1024 for k, v in host_peaks.items()},
              "output_dir": str(directory.resolve()), "error": error}
    if status == "passed":
        try:
            result["checkpoint_audit"] = checkpoint_audit(run, expected_updates=scratch["training_epochs"],
                                                        confirmation=confirmation)
        except Exception as caught:
            result.update(status="failed", error=f"{type(caught).__name__}: {caught}")
    # torchrun normally joins all workers; allow contexts to disappear before another probe.
    deadline = time.monotonic() + 30
    while any(row["pid"] in workers for row in gpu_processes()):
        if time.monotonic() > deadline:
            stop_probe_group(child)
            raise RuntimeError("Disposable GPU probe workers did not release their devices")
        time.sleep(1)
    write_json(directory / "profile_summary.json", result)
    return result


def calibrate(config, road_root, output_root, *, expected_source_sha256=None):
    require_source(expected_source_sha256)
    output_root = Path(output_root).resolve()
    root = output_root / "calibration"
    root.mkdir(parents=True, exist_ok=True)
    audit = preflight(config, road_root, output_root)
    selected = audit["gpus"]
    locks = lock_gpus(selected)
    try:
        validate_gpus(config, gpu_inventory(), gpu_processes())
        session = root / f"attempt_{time.time_ns()}"
        session.mkdir()
        write_json(session / "preflight.json", audit)
        count = 0

        def measure(batch, confirmation=False):
            nonlocal count
            require_source(expected_source_sha256)
            count += 1
            directory = session / f"{count:02d}_batch{batch}_{'confirm' if confirmation else 'search'}"
            write_json(root / "status.json", {"status": "calibrating", "time": timestamp(),
                       "batch_per_gpu": batch, "phase": "confirmation" if confirmation else "search",
                       "probe_output": str(directory)})
            result = run_probe(config, road_root, directory, selected, locks, audit=audit["data"],
                               batch=batch, confirmation=confirmation)
            print(json.dumps({"calibration": result["status"], "batch": batch,
                              "memory_gib": result["peak_process_gib"]}), flush=True)
            return result

        ceiling = float(config["target_process_memory_gib"][1])
        selected_batch, tested = choose_batch(measure, ceiling_gib=ceiling)
        confirmation = None
        # The longer mixed-baseline test may need one or two fewer instances.
        for batch in range(selected_batch, max(0, selected_batch - 5), -1):
            result = measure(batch, confirmation=True)
            if result["status"] not in {"passed", "oom"}:
                raise RuntimeError(f"Confirmation failed: {result.get('error')}")
            peaks = result["peak_process_gib"]
            if (result["status"] == "passed" and set(peaks) == {"0", "1"}
                    and all(0 < value <= ceiling for value in peaks.values())):
                selected_batch, confirmation = batch, result
                break
        if confirmation is None:
            raise RuntimeError("No batch passed the longer baseline-transition confirmation")
        require_source(expected_source_sha256)
        final = copy.deepcopy(config)
        final.update(physical_batch_size=selected_batch, world_size=2,
                     effective_batch_size=selected_batch * int(final["gradient_accumulation_steps"]) * 2,
                     calibration_status="passed_local_two_gpu_profile")
        final["sample_count"] = final["effective_batch_size"] * final["training_epochs"]
        final["customer_exposure_budget"] = final["sample_count"] * 500
        final_path = output_root / "calibrated_config.json"
        write_json(final_path, final)
        # Build the full-length stream now; formal launch revalidates and reuses it.
        stream = prepare_stream(road_root, output_root / "artifacts", final, audit=audit["data"])
        report = {"status": "passed", "time": timestamp(), "selected_batch_per_gpu": selected_batch,
                  "global_batch": final["effective_batch_size"], "target_process_memory_gib": final["target_process_memory_gib"],
                  "within_target": all(float(final["target_process_memory_gib"][0]) <= value <= ceiling
                                       for value in confirmation["peak_process_gib"].values()),
                  "confirmation": confirmation, "search": tested,
                  "formal_config": str(final_path), "stream_contract": stream["contract"],
                  "source_sha256": expected_source_sha256,
                  "scope": "GPU engineering calibration, not convergence or benchmark performance"}
        write_json(root / "report.json", report)
        write_json(root / "status.json", report)
        return final_path
    except Exception as error:
        write_json(root / "status.json", {"status": "failed", "time": timestamp(),
                                         "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        for fd in locks:
            os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--road-root")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = calibrate(load_config(args.config), resolve_road_root(args.road_root), args.output_root,
                       expected_source_sha256=source_snapshot()["source_sha256"])
    print(result)


if __name__ == "__main__":
    main()
