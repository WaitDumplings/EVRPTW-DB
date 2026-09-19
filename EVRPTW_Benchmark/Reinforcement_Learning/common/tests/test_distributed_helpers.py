"""Real CPU process groups exercise synchronization without requiring idle GPUs."""
from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from EVRPTW_Benchmark.Reinforcement_Learning.common.distributed import (
    DistributedContext,
    initialize_distributed,
)


def run_gloo_workers(worker, tmp_path: Path, *, world_size: int = 2, extra=()) -> None:
    """Bound test duration so an accidental mismatched collective cannot hang CI."""
    rendezvous = tmp_path / "gloo_rendezvous"
    context = mp.spawn(
        worker,
        args=(world_size, str(rendezvous), str(tmp_path), *extra),
        nprocs=world_size,
        join=False,
    )
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise AssertionError("Gloo workers did not finish within 90 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)


def init_gloo(rank: int, world_size: int, rendezvous: str) -> DistributedContext:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size,
        init_method=f"file://{rendezvous}", timeout=timedelta(seconds=25),
    )
    return DistributedContext.current()


class _ConditionalPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1, dtype=torch.float64)
        self.branch = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
        self.globally_unused = torch.nn.Parameter(torch.tensor(4.0, dtype=torch.float64))

    def forward(self, sample, branch):
        value = self.linear(sample).squeeze(-1)
        return value + self.branch if branch else value


def _gradient_worker(rank, world_size, rendezvous, output):
    ctx = init_gloo(rank, world_size, rendezvous)
    try:
        torch.manual_seed(123 + rank)
        model = _ConditionalPolicy()
        ctx.broadcast_model(model)
        reference = _ConditionalPolicy()
        reference.load_state_dict(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.2)
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=0.03, weight_decay=0.2,
        )
        # Seven samples deliberately give workers unequal trajectory counts.
        features = torch.arange(21, dtype=torch.float64).reshape(7, 3) / 5
        target = torch.arange(7, dtype=torch.float64).square() / 3
        local_indices = list(range(rank, 7, world_size))
        observed_sum, observed_count = ctx.sum_values(
            [float(target[local_indices].sum()), len(local_indices)]
        )
        global_mean = observed_sum / observed_count
        ema = 0.8 * 2.5 + 0.2 * global_mean
        assert ema == pytest.approx(0.8 * 2.5 + 0.2 * float(target.mean()))
        for _step in range(2):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            # Worker-specific decode lengths must not trigger any collectives.
            with ctx.local_phase("variable-length rollout and backward"):
                losses = []
                for index in local_indices:
                    value = model(features[index], index % world_size == 0)
                    for _ in range(index + 1):
                        value = value + 0.0
                    losses.append((value - target[index]).square())
                (torch.stack(losses).mean() * len(local_indices) / len(target)).backward()
            assert (model.branch.grad is None) == (rank != 0)
            ctx.sum_gradients(model.parameters())
            assert model.globally_unused.grad is None
            assert model.branch.grad is not None
            reference_loss = torch.stack([
                (reference(features[i], i % world_size == 0) - target[i]).square()
                for i in range(len(target))
            ]).mean()
            reference_loss.backward()
            for parameter, expected in zip(model.parameters(), reference.parameters()):
                if expected.grad is None:
                    assert parameter.grad is None
                else:
                    torch.testing.assert_close(parameter.grad, expected.grad)
            # Clip only after SUM: clipping local gradients changes the result.
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.15)
            expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.15)
            torch.testing.assert_close(norm, expected_norm)
            optimizer.step()
            reference_optimizer.step()
            for parameter, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter, expected)
            assert model.globally_unused.item() == 4.0
        gathered = ctx.gather_objects(local_indices)
        assert sorted(index for shard in gathered for index in shard) == list(range(7))
        assert ctx.main_call(lambda: {"writer_rank": rank}) == {"writer_rank": 0}
        Path(output, f"grad_rank_{rank}.json").write_text(json.dumps({"ema": ema}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 3])
def test_global_gradient_update_matches_serial_with_unused_parameters(tmp_path, world_size):
    run_gloo_workers(_gradient_worker, tmp_path, world_size=world_size)
    assert len(list(tmp_path.glob("grad_rank_*.json"))) == world_size


def _buffer_and_error_worker(rank, world_size, rendezvous, output):
    from EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.model import AMEVRPTWPolicy

    ctx = init_gloo(rank, world_size, rendezvous)
    try:
        model = AMEVRPTWPolicy(embedding_dim=16, hidden_dim=16, n_encode_layers=1, n_heads=2)
        ctx.broadcast_model(model)
        batch_norms = [layer for layer in model.modules() if isinstance(layer, torch.nn.BatchNorm1d)]
        assert batch_norms, "The actual AM batch-normalized encoder must be covered"
        with torch.no_grad():
            for layer in batch_norms:
                layer.running_mean.fill_(10.0 + rank)
                layer.running_var.fill_(20.0 + rank)
                layer.num_batches_tracked.fill_(30 + rank)
            # The buffer-only operation must not accidentally broadcast parameters.
            first_parameter = next(model.parameters())
            first_parameter.fill_(rank + 1.0)
        ctx.broadcast_buffers(model)
        for layer in batch_norms:
            torch.testing.assert_close(layer.running_mean, torch.full_like(layer.running_mean, 10.0))
            torch.testing.assert_close(layer.running_var, torch.full_like(layer.running_var, 20.0))
            assert layer.num_batches_tracked.item() == 30
        torch.testing.assert_close(first_parameter, torch.full_like(first_parameter, rank + 1.0))

        def write_once():
            with Path(output, "rank_zero_log.jsonl").open("a") as stream:
                stream.write(json.dumps({"rank": rank}) + "\n")
            return {"early_stop": True, "epoch": 4}

        assert ctx.main_call(write_once) == {"early_stop": True, "epoch": 4}

        def broken_write():
            raise OSError("simulated checkpoint write failure")

        with pytest.raises(RuntimeError, match="rank-zero operation failed.*checkpoint write failure"):
            ctx.main_call(broken_write)
        with pytest.raises(RuntimeError, match="rank 1.*simulated rollout failure"):
            with ctx.local_phase("rollout"):
                if rank == 1:
                    raise ValueError("simulated rollout failure")
        # Subsequent synchronized work still succeeds: neither failure hangs a peer.
        assert ctx.sum_values([1]) == [world_size]
        Path(output, f"error_rank_{rank}.json").write_text("{}")
    finally:
        dist.destroy_process_group()


def test_actual_am_buffers_rank_zero_io_and_exception_propagation(tmp_path):
    run_gloo_workers(_buffer_and_error_worker, tmp_path)
    assert [json.loads(line) for line in (tmp_path / "rank_zero_log.jsonl").read_text().splitlines()] == [{"rank": 0}]
    assert len(list(tmp_path.glob("error_rank_*.json"))) == 2


def test_single_process_helpers_and_device_validation(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    ctx, device = initialize_distributed(device="cpu", expected_world_size=1)
    assert ctx.is_main and ctx.world_size == 1 and device == "cpu"
    assert ctx.gather_objects({"rank": 0}) == [{"rank": 0}]
    assert ctx.sum_values([2, 3]) == [2, 3]
    with pytest.raises(ValueError, match="expected 3 workers"):
        initialize_distributed(device="cpu", expected_world_size=3)
    with pytest.raises(ValueError, match="NCCL requires CUDA"):
        initialize_distributed(device="cpu", backend="nccl")
    with pytest.raises(OSError, match="local failure"):
        with ctx.local_phase("single process"):
            raise OSError("local failure")
