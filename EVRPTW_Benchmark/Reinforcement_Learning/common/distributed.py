"""Synchronous helpers for policies whose rollout calls encode/logits directly.

Adapted from TERRAN/distributed.py at repository commit 699ebb37e5da6277553b45d452f3660fa18ea265.
Each worker accumulates losses with the GLOBAL denominator. Gradients are
SUM-reduced once per logical optimizer update, before clipping. No collective
is hidden inside the variable-length decoder loop.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int = 0
    world_size: int = 1

    @classmethod
    def current(cls) -> "DistributedContext":
        if dist.is_available() and dist.is_initialized():
            return cls(dist.get_rank(), dist.get_world_size())
        return cls()

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> torch.device:
        if self.world_size > 1 and dist.get_backend() == "nccl":
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def _reduce(self, values, operation):
        values = list(values)
        if self.world_size == 1:
            return values
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor, op=operation)
        return tensor.cpu().tolist()

    def sum_values(self, values):
        return self._reduce(values, dist.ReduceOp.SUM)

    def max_values(self, values):
        return self._reduce(values, dist.ReduceOp.MAX)

    def gather_objects(self, value):
        if self.world_size == 1:
            return [value]
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return gathered

    def broadcast_object(self, value):
        if self.world_size == 1:
            return value
        values = [value if self.is_main else None]
        dist.broadcast_object_list(values, src=0)
        return values[0]

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def main_call(self, callback: Callable):
        """Execute rank-zero I/O and propagate its result or Python exception."""
        if self.world_size == 1:
            return callback()
        result = None
        if self.is_main:
            try:
                result = (True, callback())
            except BaseException as error:
                result = (False, f"{type(error).__name__}: {error}")
        success, value = self.broadcast_object(result)
        if not success:
            raise RuntimeError(f"rank-zero operation failed: {value}")
        return value

    @contextmanager
    def local_phase(self, name: str):
        """Collect exceptions after local work; the body must not use collectives."""
        error = None
        try:
            yield
        except BaseException as caught:
            if self.world_size == 1:
                raise
            error = f"rank {self.rank}: {type(caught).__name__}: {caught}"
        if self.world_size > 1:
            errors = self.gather_objects(error)
            if any(errors):
                raise RuntimeError(f"distributed {name} failed: " + "; ".join(e for e in errors if e))

    @torch.no_grad()
    def broadcast_model(self, model):
        if self.world_size > 1:
            for tensor in model.state_dict().values():
                dist.broadcast(tensor, src=0)

    @torch.no_grad()
    def broadcast_buffers(self, model):
        """Share rank-zero BN running statistics; forward normalization stays local."""
        if self.world_size > 1:
            for tensor in model.buffers():
                dist.broadcast(tensor, src=0)

    @torch.no_grad()
    def sum_gradients(self, parameters):
        """SUM dtype/device buckets, retaining grad=None for globally unused weights."""
        if self.world_size == 1:
            return
        parameters = list(parameters)
        presence = torch.tensor([p.grad is not None for p in parameters],
                                dtype=torch.int32, device=self.device)
        dist.all_reduce(presence, op=dist.ReduceOp.MAX)
        buckets = {}
        for present, parameter in zip(presence.cpu().tolist(), parameters):
            if present:
                buckets.setdefault((parameter.device, parameter.dtype), []).append(parameter)
        for bucket in buckets.values():
            flattened = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                                   for p in bucket])
            dist.all_reduce(flattened, op=dist.ReduceOp.SUM)
            offset = 0
            for parameter in bucket:
                gradient = flattened[offset:offset + parameter.numel()].view_as(parameter)
                if parameter.grad is None:
                    parameter.grad = gradient.clone()
                else:
                    parameter.grad.copy_(gradient)
                offset += parameter.numel()


def initialize_distributed(*, device: str, backend: str | None = None,
                           timeout_seconds: int = 7200,
                           expected_world_size: int | None = None) -> tuple[DistributedContext, str]:
    """Bind the torchrun worker before constructing any CUDA model or environment."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world < 1 or timeout_seconds <= 0:
        raise ValueError("world size and distributed timeout must be positive")
    if expected_world_size is not None and world != int(expected_world_size):
        raise ValueError(f"expected {expected_world_size} workers, got WORLD_SIZE={world}")
    if str(device).startswith("cuda"):
        selected = local_rank if world > 1 else (torch.device(device).index or 0)
        if selected >= torch.cuda.device_count():
            raise ValueError("not enough visible CUDA devices for torchrun workers")
        torch.cuda.set_device(selected)
        device = f"cuda:{selected}"
        selected_backend = backend or "nccl"
    else:
        selected_backend = backend or "gloo"
        if selected_backend == "nccl":
            raise ValueError("NCCL requires CUDA devices")
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend=selected_backend,
                                timeout=timedelta(seconds=int(timeout_seconds)))
    return DistributedContext.current(), str(device)


def close_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
