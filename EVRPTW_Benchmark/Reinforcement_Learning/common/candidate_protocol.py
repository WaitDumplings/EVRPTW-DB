from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import torch


MASK64 = (1 << 64) - 1


def splitmix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def candidate_seed(base_seed: int, instance_index: int, candidate_index: int) -> int:
    key = (
        (int(base_seed) & 0xFFFFFFFF) << 32
        ^ (int(instance_index) & 0xFFFFFF) << 8
        ^ int(candidate_index)
    )
    return int(splitmix64(key) & 0x7FFFFFFF)


def candidate_chunks(count: int, chunk_size: int) -> Iterable[range]:
    if count <= 0 or chunk_size <= 0:
        raise ValueError("candidate count and chunk size must be positive")
    for start in range(0, int(count), int(chunk_size)):
        yield range(start, min(start + int(chunk_size), int(count)))


def _seed_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def merge_single_candidate_infos(infos: list[dict[str, Any]]) -> dict[str, Any]:
    if not infos:
        raise ValueError("at least one candidate result is required")
    merged: dict[str, Any] = {}
    for key in infos[0]:
        values = [info[key] for info in infos]
        if key == "routes":
            merged[key] = [value[0] for value in values]
            continue
        arrays = [np.asarray(value) for value in values]
        if (
            all(array.ndim >= 1 and array.shape[0] == 1 for array in arrays)
            and len({array.shape[1:] for array in arrays}) == 1
        ):
            merged[key] = np.concatenate(arrays, axis=0)
        else:
            merged[key] = values
    return merged


@dataclass
class CandidateBatch:
    infos: list[dict[str, Any]]
    runtime_s: float


def independent_candidate_batch(
    instances: list[Any],
    *,
    candidate_count: int,
    candidate_chunk_size: int,
    base_seed: int,
    instance_offset: int,
    solve_one: Callable[[Any, int], tuple[dict[str, Any], float]],
) -> CandidateBatch:
    """Evaluate candidate IDs independently so chunking cannot alter results."""

    started = time.perf_counter()
    merged_rows: list[dict[str, Any]] = []
    for local_index, instance in enumerate(instances):
        absolute_index = int(instance_offset) + local_index
        candidate_infos: list[dict[str, Any]] = []
        for chunk in candidate_chunks(candidate_count, candidate_chunk_size):
            for candidate_index in chunk:
                seed = candidate_seed(base_seed, absolute_index, candidate_index)
                _seed_runtime(seed)
                info, _runtime = solve_one(instance, seed)
                candidate_infos.append(info)
        merged_rows.append(merge_single_candidate_infos(candidate_infos))
    return CandidateBatch(infos=merged_rows, runtime_s=time.perf_counter() - started)


__all__ = [
    "CandidateBatch",
    "candidate_chunks",
    "candidate_seed",
    "independent_candidate_batch",
    "merge_single_candidate_infos",
    "splitmix64",
]
