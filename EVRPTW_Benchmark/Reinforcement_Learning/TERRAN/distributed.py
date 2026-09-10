"""Small synchronous-process helpers for TERRAN's chunked logical updates.

Loss chunks already use the global trajectory denominator. Their gradients
must therefore be SUM-reduced, never averaged a second time by world size.
No collective is issued inside a rollout or time chunk: ranks may have
different numbers of valid actions and finish their local work independently.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
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
        """Only rank zero executes I/O/evaluation; all ranks receive its result.

        This is also a synchronization boundary and propagates rank-zero
        exceptions instead of leaving peers waiting at a following barrier.
        """
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
        """Propagate Python failures at a boundary after independent local work.

        The body must contain no collectives. Process death remains subject to
        the process-group timeout and torchrun's worker supervision.
        """
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
    def sum_gradients(self, parameters):
        """One flattened SUM per dtype/device bucket, including unused tensors.

        The globally unused parameters retain grad=None, preserving AdamW's
        existing behavior. Locally unused parameters receive remote gradients.
        """
        if self.world_size == 1:
            return
        parameters = list(parameters)
        presence = torch.tensor([p.grad is not None for p in parameters],
                                dtype=torch.int32, device=self.device)
        dist.all_reduce(presence, op=dist.ReduceOp.MAX)
        active = presence.cpu().tolist()
        buckets = {}
        for present, parameter in zip(active, parameters):
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
